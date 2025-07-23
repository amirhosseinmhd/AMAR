import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pandas import DataFrame
from copy import deepcopy
from sklearn.decomposition import PCA
import os
import math
import json
import sys

import shutil
from torch.utils.data import TensorDataset, DataLoader
from ptflops import get_model_complexity_info
import torch.optim as optim
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_data_x, load_data_y
from preset import preset
from utils import *
from utils import NumpyEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.losses.SSL_loss import CombinedJEPALoss
from model.losses.hybrid_loss import CombinedHybridLoss
from model.losses.supervised_loss import HungarianMatchingLoss
from model.data.datasets import JEPADataset
from model.data.sampling import SegmentBlockSampler
from model.modules.molecules import PCAFeatureExtractor, Transformer_Encoder, TransformerDecoder, Predictor
from model.modules.helper import save_checkpoint, load_checkpoint, get_cosine_schedule_with_warmup, generate_tsne_visualizations, compute_representation_svd_stats


class LinearProbe(nn.Module):
    """A simple linear layer for probing the quality of representations."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def run_linear_probe(jepa_model, dataloader_train, dataloader_test, device, probe_epochs=10):
    """
    Trains a linear probe on representations from the training set and evaluates it on both train and test sets.
    """
    jepa_model.eval()  # Freeze JEPA model

    # --- Extract representations and labels for training set ---
    train_reps, train_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader_train:
            data_batch_x = batch_x.to(device)
            representations = jepa_model.extract_representations(data_batch_x)
            train_reps.append(representations)
            num_people = batch_y[:, :, :-1].sum(axis=(1, 2))
            train_labels.append(num_people)
    
    train_reps_tensor = torch.cat(train_reps, dim=0)
    train_labels_tensor = torch.cat(train_labels, dim=0).long().to(device)

    # --- Extract representations and labels for test set ---
    test_reps, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader_test:
            data_batch_x = batch_x.to(device)
            representations = jepa_model.extract_representations(data_batch_x)
            test_reps.append(representations)
            num_people = batch_y[:, :, :-1].sum(axis=(1, 2))
            test_labels.append(num_people)

    test_reps_tensor = torch.cat(test_reps, dim=0)
    test_labels_tensor = torch.cat(test_labels, dim=0).long().to(device)

    # --- Initialize and Train the probe ONLY on the training set representations ---
    num_classes = int(max(train_labels_tensor.max(), test_labels_tensor.max())) + 1
    probe_model = LinearProbe(input_dim=train_reps_tensor.shape[1], num_classes=num_classes).to(device)
    probe_optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-3)
    probe_criterion = nn.CrossEntropyLoss()
    
    probe_dataset = TensorDataset(train_reps_tensor, train_labels_tensor)
    probe_dataloader = DataLoader(probe_dataset, batch_size=preset["nn"]["batch_size"], shuffle=True)

    for _ in range(probe_epochs):
        for reps, labels in probe_dataloader:
            probe_optimizer.zero_grad()
            outputs = probe_model(reps)
            loss = probe_criterion(outputs, labels)
            loss.backward()
            probe_optimizer.step()

    # --- Evaluate the probe on both training and test sets ---
    train_loss, train_mae = 0, 0
    test_loss, test_mae = 0, 0

    with torch.no_grad():
        # Evaluate on training data
        train_outputs = probe_model(train_reps_tensor)
        train_loss = probe_criterion(train_outputs, train_labels_tensor).item()
        train_predictions = torch.argmax(train_outputs, dim=1)
        train_mae = torch.abs(train_predictions - train_labels_tensor).float().mean().item()

        # Evaluate on test data
        test_outputs = probe_model(test_reps_tensor)
        test_loss = probe_criterion(test_outputs, test_labels_tensor).item()
        test_predictions = torch.argmax(test_outputs, dim=1)
        test_mae = torch.abs(test_predictions - test_labels_tensor).float().mean().item()

    jepa_model.train()  # Set JEPA model back to train mode
    return train_loss, train_mae, test_loss, test_mae


class JEPA(nn.Module):
    """
    Implements the Joint Embedding Predictive Architecture (JEPA).
    Consists of an online encoder, a target encoder (EMA of the online encoder),
    a segment/block sampler, and a predictor (to be fully implemented).
    The goal is to predict representations of target blocks using representations of context blocks.
    """
    def __init__(self, pca_components=None):
        super().__init__()
        # Number of tokens generated by the CNN for each segment.
        self.tokens_per_segment = 1
        # Maximum number of tokens possible in a full view (all segments processed).
        self.max_total_tokens_in_view = preset["jepa"]["num_segments_total_view"] * self.tokens_per_segment
        self.total_num_epoch = preset["nn"]["epoch"]

        # --- Online Encoder (learnable) ---
        # self.online_cnn_feature_extractor = CNNFeatureExtractor(
        #     input_channels=270,
        #     output_channels=preset["nn"]["d_embedding"],
        #     embedding_time_dim=preset["cnn_embedding_time_dim"] # This is tokens_per_segment for the CNN
        # )
        self.online_cnn_feature_extractor = PCAFeatureExtractor(input_channels=270, output_channels=preset["nn"]["d_embedding"])
                                                               # pca_components=pca_components)

        self.online_transformer_encoder = Transformer_Encoder(
            d_model=preset["nn"]["d_embedding"],             # Feature dimension of tokens.
            nhead=preset["nn"]["n_attention_heads"],
            num_layers=preset["jepa"]["encoder_layers"],
            max_total_tokens=self.max_total_tokens_in_view          # Capable of handling all tokens if needed.
        )
        self.target_cnn_feature_extractor = deepcopy(self.online_cnn_feature_extractor)
        self.target_transformer_encoder = deepcopy(self.online_transformer_encoder)

        for param in self.target_cnn_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.target_transformer_encoder.parameters():
            param.requires_grad = False
        self.ema_decay_base = preset["jepa"]["ema_decay"]
        self.sampler = SegmentBlockSampler(
            weight_decay_factor=preset["jepa"].get("sampling_weight_decay_factor", 0.9)  # Make configurable
        )

        self.sampling_call_count = 0
        self.last_weight_reset = 0
        self.predictor = Predictor(preset)
        self.predictor.set_shared_pos_encoder(self.online_transformer_encoder.pos_encoder)
        
        # Add projector for VICReg
        self.vicreg_projector = nn.Sequential(
            nn.Linear(preset["nn"]["d_embedding"], preset["nn"]["d_embedding"] * 2),
            nn.ReLU(),
            nn.Linear(preset["nn"]["d_embedding"] * 2, preset["nn"]["d_embedding"])
        )

    @torch.no_grad()
    def extract_representations(self, x_raw_input):
        """
        Extracts fixed-size representations for a batch of input data using the target encoder.
        The representation is the mean of all tokens from the full view.
        """
        self.eval() # Ensure the model is in evaluation mode
        
        # 1. Prepare the input view, same as in the forward pass
        required_len = preset["jepa"]["num_segments_total_view"] * preset["jepa"]["segment_length"]
        if x_raw_input.shape[1] < required_len:
            # Pad the input if it's too short
            padding_needed = required_len - x_raw_input.shape[1]
            x_raw_input = F.pad(x_raw_input, (0, 0, 0, padding_needed), 'replicate')

        # Use a fixed center crop for consistent representation extraction
        max_t_init = x_raw_input.shape[1] - required_len
        t_init = max_t_init // 2 if max_t_init >= 0 else 0
        x_full_view = x_raw_input[:, t_init : t_init + required_len, :]

        # 2. Process the full view with the target encoder
        all_tokens_encoded = self._process_full_view_for_target_encoder(
            x_full_view,
            self.target_cnn_feature_extractor,
            self.target_transformer_encoder
        )
        # all_tokens_encoded shape: (batch_size, num_tokens, feature_dim)

        # 3. Average the tokens to get a single representation vector per sample
        representations = torch.mean(all_tokens_encoded, dim=1)
        # representations shape: (batch_size, feature_dim)
        
        return representations
    
    @torch.no_grad()
    def _update_ema(self, current_iter, total_iters):
        """Updates the target encoder's weights as an EMA of the online encoder's weights."""

        # 1. Calculate the scheduled decay rate for the current iteration
        # ema_decay = 1 - (1 - self.ema_decay_base) * (np.cos(np.pi * current_iter / total_iters) + 1) / 2
        # ema_decay = 1 - (1 - self.ema_decay_base) * (np.cos(np.pi * iter / self.max_iters) + 1) / 2
        ema_decay = self.ema_decay_base + (1 - self.ema_decay_base) * (current_iter / total_iters)

        # 2. Update the CNN parameters using the scheduled decay rate
        for online_param, target_param in zip(self.online_cnn_feature_extractor.parameters(),
                                              self.target_cnn_feature_extractor.parameters()):
            target_param.data = target_param.data * ema_decay + online_param.data * (1. - ema_decay)
        for online_param, target_param in zip(self.online_transformer_encoder.parameters(),
                                              self.target_transformer_encoder.parameters()):
            target_param.data = target_param.data * ema_decay + online_param.data * (1. - ema_decay)

    def update_ema(self, current_iter, total_iters):
        """Wrapper to call the EMA update mechanism."""
        self._update_ema(current_iter, total_iters)
        
    """
           ************* # REWRITE THIS FUNCTION. WRITE IT USING TORCH INDEX ACESSING, FIRST RESHAPE IT and THEN SELECT SEGMENTS FROM IT USING TOECH   *************
    """
    def _extract_segments_from_x(self, x_item_full_view, segment_indices_list_for_item):
        """
        Goal of this method is that given full CSI and segment list, it outputs a tesnsor containing all of segments
         inside the list, so if we have 10 segment "id"s in segment_indices_list_for_item, we would return a tensor
         with size 10, Segment_Length
        """

        # x_item_full_view: Tensor for a single batch item, representing the full processing window.
        # Shape: (self.config["num_segments_total_view"] * self.config["segment_length"], 270).
        # segment_indices_list_for_item: List of segment indices to extract for this item.
        # Output: Stacked tensor of extracted segments.
        # Shape: (num_chosen_segments_for_item, self.config["segment_length"], 270).

        segments_to_extract = []  # here we make a list of segments, each item in list correspond to one segment.
        # Ensure segment_indices_list_for_item is not empty and contains valid indices
        if not segment_indices_list_for_item:
            return torch.empty(0, preset["jepa"]["segment_length"], 270,
                               device=x_item_full_view.device)

        for seg_idx in segment_indices_list_for_item:
            start_ts = seg_idx * preset["jepa"]["segment_length"]
            end_ts = start_ts + preset["jepa"]["segment_length"]
            segments_to_extract.append(x_item_full_view[start_ts:end_ts, :])

        if not segments_to_extract:  # Handle cases like an empty context (already handled above, but good for safety).
            return torch.empty(0, preset["jepa"]["segment_length"], 270,
                               device=x_item_full_view.device)

        return torch.stack(segments_to_extract, dim=0)

    def _get_tokens_for_context(self,
                                x_full_view_batch,
                                context_mask,
                                cnn_extractor,
                                transformer_encoder):
        # x_full_view_batch: (batch_size, total_view_len, input_channels)
        # list_of_context_segment_indices: List (batch_size) of lists of segment indices.
        # sampled_context_token_indices_tensor: Tensor (batch_size, max_context_tokens_in_batch) - original token indices, padded.
        # sampled_context_padding_mask_tensor: Tensor (batch_size, max_context_tokens_in_batch) - True for padded.
        # cnn_extractor: Online CNN.
        # transformer_encoder: Online Transformer.
        # Output:
        #   context_tokens_encoded: (batch_size, max_context_tokens_in_batch, cnn_output_channels)
        #   context_padding_mask: (batch_size, max_context_tokens_in_batch) - this is sampled_context_padding_mask_tensor
        #   padded_context_original_indices: (batch_size, max_context_tokens_in_batch) - this is sampled_context_token_indices_tensor

        batch_size = x_full_view_batch.shape[0]

        max_total_tokens = preset["jepa"]["num_segments_total_view"]
        window_length = preset["jepa"]["segment_length"]
        x_full_view_reshaped = x_full_view_batch.reshape(batch_size*max_total_tokens, window_length ,270)
        token_cnn_output = cnn_extractor(x_full_view_reshaped) # batch_size*max_total_tokens, 1, 270
        token_encoder_unmasked = token_cnn_output.reshape(batch_size, max_total_tokens, -1)
        original_indices = torch.arange(
            max_total_tokens,
            device=x_full_view_batch.device
        )
        token_original_indices = original_indices.unsqueeze(0).expand(batch_size, -1)

        context_embeddings = transformer_encoder(
            src=token_encoder_unmasked,
            token_original_indices=token_original_indices,
            src_key_padding_mask=context_mask.bool()
        )

        return context_embeddings

    def _process_full_view_for_target_encoder(self, x_full_view_batch, cnn_extractor, transformer_encoder):
        # x_full_view_batch: Batch of full processing windows.
        # Shape: (batch_size, self.config["num_segments_total_view"] * self.config["segment_length"], 270).
        # cnn_extractor: The target CNN feature extractor.
        # transformer_encoder: The target Transformer encoder.
        # Output:
        #   all_tokens_encoded: All tokens from the full view, processed by target encoder.
        #   Shape: (batch_size, self.max_total_tokens_in_view, cnn_output_channels).

        batch_size = x_full_view_batch.shape[0]
        num_total_segments = preset["jepa"]["num_segments_total_view"]
        segment_len = preset["jepa"]["segment_length"]
        input_channels = 270
        # tokens_per_segment is self.tokens_per_segment
        cnn_out_channels = preset["nn"]["d_embedding"]

        # Reshape for CNN:
        # (batch_size, num_total_segments * segment_len, input_channels) ->
        # (batch_size * num_total_segments, segment_len, input_channels)
        cnn_input = x_full_view_batch.reshape(
            batch_size* num_total_segments,segment_len,
            input_channels
        )

        # Pass through CNN. Output shape:
        # (batch_size * num_total_segments, self.tokens_per_segment, cnn_out_channels)
        cnn_output_tokens = cnn_extractor(cnn_input)

        # Reshape for Transformer:
        # (batch_size, num_total_segments * self.tokens_per_segment, cnn_out_channels)
        # which is (batch_size, self.max_total_tokens_in_view, cnn_out_channels)
        transformer_input_tokens = cnn_output_tokens.reshape(
            batch_size,
            self.max_total_tokens_in_view,
            cnn_out_channels
         )

        # Create original token indices for the full view
        original_indices_full_view = torch.arange(self.max_total_tokens_in_view, device=transformer_input_tokens.device)
        original_indices_full_view = original_indices_full_view.unsqueeze(0).expand(batch_size, -1) # (B, max_total_tokens)

        all_tokens_encoded = transformer_encoder(transformer_input_tokens,
                                                 token_original_indices=original_indices_full_view,
                                                 src_key_padding_mask=None)
        return all_tokens_encoded

    def _prepare_target_tokens_for_loss(self, batch_size, device,
                                         target_block_token_indices_tensor,
                                         z_target_ema_all_tokens):
        """
     Prepares the actual target token representations for the loss calculation.
    These are selected from z_target_ema_all_tokens using the sampled target_block_segment_indices.
    Optimized to return torch tensors instead of nested lists.
    
    Args:
        batch_size (int): Number of samples in the current batch
        device (torch.device): Device to place tensors on (CPU/GPU)
        target_block_token_indices_tensor (torch.Tensor): Tensor of absolute token indices for each target block.
                                                              Shape: (batch_size, num_target_blocks, tokens_per_block).
            z_target_ema_all_tokens (torch.Tensor): Complete set of target representations
                                                   from the EMA target encoder, shape:
                                                   (batch_size, total_tokens, embed_dim)
    
    Returns:
        torch.Tensor: Selected target token representations corresponding to the sampled
                     target blocks, shape: (batch_size, num_target_blocks, tokens_per_block, embed_dim)
    
    Process:
        1. Pre-allocates output tensor for memory efficiency
        2. Iterates through each batch sample
        3. For each target block in the sampling info:
           - Extracts the corresponding token indices
           - Selects the matching representations from z_target_ema_all_tokens
           - Places them in the pre-allocated tensor
        4. Returns the structured target representations ready for loss calculation
    
    Note: This method is optimized to avoid nested list operations and directly work
          with PyTorch tensors for better performance during training.
        """
        # Pre-allocate tensor for all target blocks across the batch
        # Shape: (batch_size, num_target_blocks, tokens_per_block, cnn_output_channels)
        tokens_per_block = preset["jepa"]["target_block_size_segments"] * self.tokens_per_segment
        num_target_blocks = preset["jepa"]["num_target_blocks"]
        cnn_output_channels = preset["nn"]["d_embedding"]
        
        actual_targets_tensor = torch.empty(
            batch_size, num_target_blocks, tokens_per_block, cnn_output_channels,
            device=device, dtype=z_target_ema_all_tokens.dtype
        )
        
        for i in range(batch_size):
            for block_idx in range(preset["jepa"]["num_target_blocks"]):
                # Get the token indices for the current block for item i
                # These are absolute token indices, e.g., [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                # for a block of 2 segments with 5 tokens each.
                current_block_token_indices = target_block_token_indices_tensor[i, block_idx, :] # Shape: (tokens_per_block)
                
                if not current_block_token_indices.numel() > 0 : # Should not happen if tokens_per_block > 0
                     actual_targets_tensor[i, block_idx] = torch.zeros(
                        tokens_per_block, cnn_output_channels,
                        device=device, dtype=z_target_ema_all_tokens.dtype
                    )
                     continue

                # Since tokens within a block are contiguous as per sampler logic:
                # First token index in the block
                final_slice_start = current_block_token_indices[0].item()
                # Last token index in the block + 1 for exclusive slicing
                final_slice_end = current_block_token_indices[-1].item() + 1
                
                # Verify that the number of tokens derived from slice matches tokens_per_block
                if (final_slice_end - final_slice_start) != tokens_per_block:
                    raise ValueError(
                        f"Slice length {final_slice_end - final_slice_start} does not match tokens_per_block {tokens_per_block} "
                        f"for item {i}, block {block_idx}. Indices: {current_block_token_indices.tolist()}"
                    )

                max_token_idx_available = z_target_ema_all_tokens.shape[1]
                if final_slice_start >= max_token_idx_available or final_slice_end > max_token_idx_available or final_slice_start < 0:
                     raise ValueError(
                        f"Invalid token slice for target block: start {final_slice_start}, end {final_slice_end} "
                        f"for item {i}, block {block_idx} with max token index {max_token_idx_available-1}. "
                        f"Token indices in block: {current_block_token_indices.tolist()}"
                    )

                selected_block_tokens = z_target_ema_all_tokens[i, final_slice_start:final_slice_end, :]
                actual_targets_tensor[i, block_idx] = selected_block_tokens
        
        return actual_targets_tensor

    @torch.no_grad()
    def _prepare_full_view_CSI(self, x_raw_input):
        # This window's length is preset["jepa"]["num_segments_total_view"] * preset["jepa"]["segment_length"].
        required_len = preset["jepa"]["num_segments_total_view"] * preset["jepa"]["segment_length"]
        if x_raw_input.shape[1] < required_len:
            raise ValueError(f"Input time series length {x_raw_input.shape[1]} is less than required {required_len}")

        max_t_init = x_raw_input.shape[1] - required_len
        batch_size = x_raw_input.shape[0]

        if self.training:  # Random start during training.

            t_init = np.random.randint(0, max_t_init + 1, size=(batch_size,))

            # Use numpy advanced indexing to extract different segments for each batch item
            batch_indices = np.arange(batch_size)[:, None]  # Shape: (batch_size, 1)
            time_indices = np.arange(required_len)[None, :] + t_init[:, None]  # Shape: (batch_size, required_len)

            x_full_view_of_segments = x_raw_input[batch_indices, time_indices, :]


        else:  # Fixed start during evaluation.
            t_init = max_t_init // 2 if max_t_init >= 0 else 0
            x_full_view_of_segments = x_raw_input[:, t_init: t_init + required_len, :]

        return x_full_view_of_segments
    def update_ema(self, current_iter, total_iters):
        """Wrapper to call the EMA update mechanism."""
        self._update_ema(current_iter, total_iters)

    def forward(self, x_raw_input):
        # x_raw_input: Raw input time series.
        # Shape: (batch_size, total_timestamps, 270).
        batch_size = x_raw_input.shape[0]
        device = x_raw_input.device

        # --- Step 0: Create the processing window from the raw input ---
        # here we start off with a random point from 0 to segment length, then continue this for num_total_segments
        # this is done indepnedently for each CSI to ensure the CNN learn a robust represntation
        x_full_view_of_segments = self._prepare_full_view_CSI(x_raw_input)

        # --- Step 1: Sample segment indices for context and target blocks ---
        # Sampler now also returns token indices as tensors and context padding mask
        sampling_info = self.sampler(batch_size, device=device)
        # Track sampling calls for monitoring
        self.sampling_call_count += batch_size * preset["jepa"]["num_target_blocks"]
        # Periodic weight reset to prevent over-bias
        reset_interval = preset["jepa"].get("sampling_weight_reset_interval", 1000)
        if self.sampling_call_count - self.last_weight_reset > reset_interval:
            if self.training:  # Only reset during training
                self.sampler.reset_weights()
                self.last_weight_reset = self.sampling_call_count
                # print(f"Reset sampling weights at call {self.sampling_call_count}")



        # --- Step 2: Process Full View with Target Encoder (EMA Path) ---
        # This generates representations for all possible tokens in the view using the target encoder.
        with torch.no_grad():  # Target encoder is not trained via backpropagation.
            z_target_ema_all_tokens = self._process_full_view_for_target_encoder(
                x_full_view_of_segments,
                self.target_cnn_feature_extractor,
                self.target_transformer_encoder
            )
            # z_target_ema_all_tokens shape: (batch_size, self.max_total_tokens_in_view, self.config["d_embedding"])
        # --- Prepare actual target token representations for the loss ---
        actual_targets_for_loss = self._prepare_target_tokens_for_loss(
            batch_size,
            device,
            sampling_info["target_block_token_indices_tensor"],  # Use tensor of token indices
            z_target_ema_all_tokens
        )
        # context_original_indices, context_padding_mask = sampling_info["context_token_indices_tensor"], sampling_info[
        #     "context_padding_mask_tensor"]

        # --- Step 3: Process Context Segments with Online Encoder (Online Path) ---
        z_context_online = self._get_tokens_for_context(
            x_full_view_of_segments,
            sampling_info["context_mask"],  # Original segment indices for data extraction
            self.online_cnn_feature_extractor,
            self.online_transformer_encoder
        )
        # z_context_online shape: (batch_size, max_context_tokens_in_batch, preset["nn"]["d_embedding"])

        # --- Step 4: Pass Context Tokens through Decoder ---
        if self.training:
            src_key_padding_mask = sampling_info["context_mask"]
        else:
            src_key_padding_mask = None


        return {
            "z_context_online": z_context_online,
            "actual_targets_for_loss": actual_targets_for_loss,  # Renamed from actual_targets_for_loss_list
            "z_target_ema_all_tokens": z_target_ema_all_tokens,
            "sampling_info": sampling_info  # Renamed from sampling_info_segments
        }


def train_jepa(jepa_model, dataloader_train, dataloader_test, optimizer, device, num_epochs=10,
                          resume_from=None, checkpoint_dir=None, checkpoint_interval=10,
                          # VICReg hyperparameters
                          prediction_coeff=1.0,
                          vicreg_coeff=0.001,
                          vicreg_std_coeff=25.0,
                          vicreg_cov_coeff=1.0):
    """
    Train JEPA model with VICReg regularization where:
    - Invariance loss = MSE between predictions and targets (main JEPA loss)
    - Variance/Covariance losses are applied to context embeddings for regularization
    
    total_loss = prediction_loss + beta_vicreg * (beta_std * std_loss + beta_cov * cov_loss)
    
    Args:
        jepa_model: JEPA model to train
        dataloader_train: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        resume_from: Path to checkpoint to resume from
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: How often to save checkpoints
        prediction_coeff: Coefficient for JEPA prediction loss (includes invariance)
        vicreg_coeff: β_vicreg - coefficient for VICReg regularization terms
        vicreg_std_coeff: β_std - variance coefficient
        vicreg_cov_coeff: β_cov - covariance coefficient
    """
    # Create checkpoint directory if provided
    if checkpoint_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(
            preset["path"].get("models_dir", "./saved_models"), 
            f"jepa_vicreg_checkpoints_{timestamp}"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize combined loss function
    combined_loss_fn = CombinedJEPALoss(
        prediction_coeff=prediction_coeff,
        vicreg_coeff=vicreg_coeff,
        vicreg_std_coeff=vicreg_std_coeff,
        vicreg_cov_coeff=vicreg_cov_coeff
    ).to(device)
    
    # Set up wandb logging
    run_name = f"JEPA_VICReg_training_{'+'.join(preset['data']['environment'])}"
    run = wandb.init(
        project="jepa_vicreg",
        name=run_name,
        config={
            **preset,
            'vicreg_coeff': vicreg_coeff,
            'vicreg_std_coeff': vicreg_std_coeff,
            'vicreg_cov_coeff': vicreg_cov_coeff,
            'prediction_coeff': prediction_coeff
        },
        reinit=True,
        mode="online"
    )
    
    start_epoch = 0
    best_loss = float('inf')
    best_state_dict = None
    
    # Resume from checkpoint if provided
    if resume_from:
        checkpoint = load_checkpoint(resume_from, jepa_model, optimizer)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            if 'best_state_dict' in checkpoint:
                best_state_dict = checkpoint['best_state_dict']
            print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")
    
    jepa_model.train()
    
    # Calculate total iterations for EMA updates
    total_iters = num_epochs * len(dataloader_train)
    current_iter = start_epoch * len(dataloader_train)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        total_prediction_loss = 0
        total_vicreg_loss = 0
        total_vicreg_std_loss = 0
        total_vicreg_cov_loss = 0
        num_batches = 0
        
        for batch_idx, (data_batch_x, data_batch_y) in enumerate(dataloader_train):
            # For JEPADataset, data_batch_x is just the tensor
            # For regular DataLoader with TensorDataset, data_batch_x is a tuple (data, _)
            if isinstance(data_batch_x, tuple):
                data_batch_x = data_batch_x[0]
            data_batch_x = data_batch_x.to(device)
            optimizer.zero_grad()
            
            # Forward pass through JEPA model
            jepa_outputs = jepa_model(data_batch_x)
            
            # Extract outputs for loss calculation
            z_context_online = jepa_outputs["z_context_online"]
            actual_targets_for_loss = jepa_outputs["actual_targets_for_loss"]
            sampling_info = jepa_outputs["sampling_info"]


            # Perform prediction using the model's predictor
            predictions = jepa_model.predictor(
                z_context_online,
                sampling_info["context_mask"],
                sampling_info["target_block_token_indices_tensor"]
            )
            
            # Reshape for loss
            b, num_blocks, tokens_per_block, d = actual_targets_for_loss.shape
            actual_targets = actual_targets_for_loss.reshape(b * num_blocks, tokens_per_block, d)
            
            # Calculate combined loss (Invariance as prediction loss + VICReg regularization)
            total_loss_val, prediction_loss_val, total_loss_vicreg_val, std_loss_val, cov_loss_val = combined_loss_fn(
                predictions=predictions,
                actual_targets=actual_targets,
                z_context_online=z_context_online,
                context_padding_mask=sampling_info["context_mask"],
                projector=jepa_model.vicreg_projector
            )
            
            loss = total_loss_val
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update EMA model after each iteration
            current_iter += 1
            jepa_model.update_ema(current_iter=current_iter, total_iters=total_iters)

            # Accumulate losses for logging
            total_loss += loss.item()
            total_prediction_loss += prediction_loss_val.item()
            total_vicreg_std_loss += std_loss_val.item()
            total_vicreg_cov_loss += cov_loss_val.item()
            total_vicreg_loss += total_loss_vicreg_val.item()
            num_batches += 1
        
        # Removed the epoch-level EMA update since we now update per iteration

        # Epoch-level logging
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_prediction_loss = total_prediction_loss / num_batches
            avg_vicreg_loss = total_vicreg_loss / num_batches
            avg_vicreg_std_loss = total_vicreg_std_loss / num_batches
            avg_vicreg_cov_loss = total_vicreg_cov_loss / num_batches
        else:
            avg_loss = avg_prediction_loss = avg_vicreg_loss = 0
            avg_vicreg_std_loss = avg_vicreg_cov_loss = 0

        print(f"Epoch {epoch}, Total Loss: {avg_loss:.4f}, "
              f"Prediction Loss: {avg_prediction_loss:.4f}, "
              f"VICReg Loss: {avg_vicreg_loss:.4f}")

        wandb.log({
            "epoch": epoch,
            "total_loss": avg_loss,
            "prediction_loss": avg_prediction_loss,
            "vicreg_total_loss": avg_vicreg_loss,
            "vicreg_std_loss": avg_vicreg_std_loss,
            "vicreg_cov_loss": avg_vicreg_cov_loss,
        }, step=epoch)

        # --- Linear Probing ---
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Running linear probe at epoch {epoch}...")
            probe_loss_train, probe_mae_train, probe_loss_test, probe_mae_test = run_linear_probe(
                jepa_model, dataloader_train, dataloader_test, device
            )
            print(f"  Probe Train - Loss: {probe_loss_train:.4f}, MAE: {probe_mae_train:.4f}")
            print(f"  Probe Test - Loss: {probe_loss_test:.4f}, MAE: {probe_mae_test:.4f}")

            wandb.log({
                "probe/train_loss": probe_loss_train,
                "probe/train_mae": probe_mae_train,
                "probe/test_loss": probe_loss_test,
                "probe/test_mae": probe_mae_test,
            }, step=epoch)


        # Generate t-SNE visualizations every 25 epochs or on the last epoch
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            print(f"Generating t-SNE visualizations at epoch {epoch}...")
            tsne_figure_dict = generate_tsne_visualizations(
                jepa_model, 
                dataloader_train,
                device, 
                epoch
            )
            
            if tsne_figure_dict:
                fig = tsne_figure_dict["tsne_visualizations"]
                wandb.log({"t-SNE Visualizations": wandb.Image(fig)}, step=epoch)
                plt.close(fig)  # Close the figure to free memory

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Computing SVD statistics at epoch {epoch}...")
            svd_stats = compute_representation_svd_stats(jepa_model, dataloader_train, device, max_samples=500)
            if svd_stats:
                wandb.log({**svd_stats, "epoch": epoch}, step=epoch)
                print(f"SVD Stats - Effective Rank: {svd_stats.get('svd/effective_rank', 'N/A')}, "
                    f"Condition Number: {svd_stats.get('svd/condition_number', 'N/A'):.2f}")

        # Save best model (only updating the state dict, not saving the file each time)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_state_dict = jepa_model.state_dict().copy()
            print(f"New best model at epoch {epoch} with loss {best_loss:.4f}")
        
        # Save checkpoint only at intervals or last epoch
        save_this_epoch = (epoch % checkpoint_interval == 0) or (epoch == num_epochs - 1)
        
        if save_this_epoch:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': jepa_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_state_dict': best_state_dict,  # Always include the best model state
                'config': preset,
                'vicreg_params': {
                    'vicreg_coeff': vicreg_coeff,
                    'vicreg_std_coeff': vicreg_std_coeff,
                    'vicreg_cov_coeff': vicreg_cov_coeff,
                    'prediction_coeff': prediction_coeff
                }
            }
            
            # Save the current checkpoint
            save_checkpoint(
                checkpoint, 
                False,  # Not marking regular checkpoints as best
                checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch}.pth"
            )
            
            # If we have a best model state, save it separately
            if best_state_dict is not None:
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_state_dict,  # Use the best state dict
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'config': preset,
                    'vicreg_params': {
                    'vicreg_coeff': vicreg_coeff,
                    'vicreg_std_coeff': vicreg_std_coeff,
                    'vicreg_cov_coeff': vicreg_cov_coeff,
                    'prediction_coeff': prediction_coeff
                    }
                }
                # Only save the best model file at intervals to avoid frequent disk writes
                torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"Updated best model file at epoch {epoch} (best loss: {best_loss:.4f})")
    
    wandb.finish()
    return best_state_dict




def run_JEPA(data_train_x,
                     data_train_y,
                     data_test_x,
                     data_test_y,
                     var_repeat=10,
                     resume_from=None):
    """
    Run JEPA SSL training on specified environments.
    Args:
        environments: List of environment names to train on
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        resume_from: Path to a checkpoint to resume training from
    Returns:
        saved_model_path: Path to the saved model
    """
    ## ================= Device Setup ========================
    # Update device selection to check for CUDA first, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    environments = preset["data"]["environment"]
    num_epochs = preset["nn"]["epoch"]
    

    var_x_shape = data_train_x[0].shape

    # data_x_mean = np.mean(data_x, axis=1)
    # pca = PCA(n_components=50)
    # pca.fit(data_x_mean)
    # pca_components = torch.from_numpy(pca.components_.T).float().to(device)
    # Create dataset and dataloader
    jepa_dataset_train = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    jepa_dataset_test = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))

    dataloader_train = DataLoader(jepa_dataset_train, batch_size=preset["nn"]["batch_size"], shuffle=True, pin_memory=True)
    dataloader_test = DataLoader(jepa_dataset_test, batch_size=preset["nn"]["batch_size"], shuffle=True, pin_memory=True)

    jepa_model = JEPA().to(device)
    jepa_model.online_cnn_feature_extractor = torch.compile(jepa_model.online_cnn_feature_extractor,
                                                                mode="default")
    jepa_model.target_cnn_feature_extractor = torch.compile(jepa_model.target_cnn_feature_extractor,
                                                                mode="default")
        # pca_components=pca_components).to(device)
    # jepa_model.max_iters = num_epochs * len(dataloader)  # Set max iterations for EMA updates

    macs, params = get_model_complexity_info(JEPA(), var_x_shape, as_strings=False)
    print(f"Model Parameters: {params:,}, FLOPs: {macs * 2:,}")
    
    optimizer = optim.AdamW(
        jepa_model.parameters(),
        lr=preset["nn"]["lr"],
        weight_decay=preset["nn"]["weight_decay"]
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(
        preset["path"].get("models_dir", "./saved_models"),
        f"jepa_ssl_{'+'.join(environments)}_{timestamp}"
    )
    
    print("Starting JEPA SSL training...")
    best_state_dict = train_jepa(jepa_model=jepa_model, dataloader_train=dataloader_train,dataloader_test=dataloader_test,
                                 optimizer=optimizer, device=device,
                                 num_epochs=num_epochs, resume_from=resume_from,
                                 checkpoint_dir=checkpoint_dir,
                                 checkpoint_interval=50,
                                 prediction_coeff=preset["jepa"]["loss"]["prediction_coef"],
                                 vicreg_coeff=preset["jepa"]["loss"]["vicreg_coef"],
                                 vicreg_std_coeff=preset["jepa"]["loss"]["vicreg_std_coef"],
                                 vicreg_cov_coeff=preset["jepa"]["loss"]["vicreg_cov_coef"]
                                 )

    model_filename = f"jepa_ssl_final_{'+'.join(environments)}_{timestamp}.pth"
    model_path = os.path.join(checkpoint_dir, model_filename)

    torch.save({
        'model_state_dict': best_state_dict,
        'config': preset,
        "environment": environments
    }, model_path)
    
    print(f"Final model saved to {model_path}")
    
    results = {
        "model": "JEPA_SSL",
        "environment": environments,
        "config": preset,
        "model_path": model_path,
        "checkpoint_dir": checkpoint_dir
    }
    
    results_path = os.path.join(checkpoint_dir, f"jepa_ssl_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    return best_state_dict

# def resume_jepa_training(checkpoint_path, environments=None, num_epochs=None, batch_size=None):
#     """
#     Resume JEPA training from a checkpoint
    
#     Args:
#         checkpoint_path: Path to checkpoint file
#         environments: Optional override for environments to train on
#         num_epochs: Optional override for number of epochs
#         batch_size: Optional override for batch size
    
#     Returns:
#         saved_model_path: Path to the saved model
#     """
#     if not os.path.exists(checkpoint_path):
#         raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     config = checkpoint.get('config', {})
    
#     if environments is None and "environment" in config:
#         environments = config["environment"]
#     elif environments is None:
#         environments = ["meeting_room", "empty_room", "classroom"]
        
#     if num_epochs is None and 'epoch' in config:
#         num_epochs = config['epoch']
#     elif num_epochs is None:
#         num_epochs = 50
        
#     if batch_size is None and 'batch_size' in config:
#         batch_size = config['batch_size']
#     elif batch_size is None:
#         batch_size = 16
    
#     return run_jepa(
#         environments=environments,
#         num_epochs=num_epochs,
#         batch_size=batch_size,
#         resume_from=checkpoint_path
#     )



