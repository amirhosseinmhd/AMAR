

#
##
import os
import math
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
#
import torch.nn as nn
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from itertools import permutations
from sklearn.metrics import classification_report, accuracy_score
from scipy.optimize import linear_sum_assignment
import sys
modules_path = "/home/amirmhd/Documents/multi_modal_CSI/benchmark/wifi_csi"

if modules_path not in sys.path:
    sys.path.insert(0, modules_path) # Insert at the beginning to prioritize it

from train import train
from preset import preset
import torch.nn.functional as F
from utils import *
import wandb
from collections import Counter
import itertools # Ensure itertools is imported
from JEPA import JEPA_Model, JEPA_CONFIG, CNNFeatureExtractor as JEPA_CNNFeatureExtractor, Transformer_Encoder as JEPA_Transformer_Encoder # Import JEPA components

def strat_train_test_split(data_x, data_y, test_size, shuffle, random_state):
    """
    Splits data into training and testing sets using stratification based on the sum of data_y along a specified axis.
    Ensures that data_y is returned in its original form. Handles cases where some patterns occur only once.

    Args:
        data_x: Features (numpy array).
        data_y: Labels (numpy array), expected to be multi-dimensional (e.g., m, num_activities, num_people).
        test_size: Proportion of the dataset to include in the test split.
        shuffle: Whether or not to shuffle the data before splitting.
        random_state: Controls the shuffling applied to the data before applying the split.
        stratify_axis: The axis along which to sum data_y for creating stratification labels.
                       For data_y with shape (m, num_activities, num_people), summing axis=1
                       (num_activities) gives patterns of shape (m, num_people).

    Returns:
        A tuple containing (X_train, X_test, y_train, y_test).
    """
    stratify_axis = 1
    # Sum along the specified axis to get a pattern for stratification.

    y_summed_patterns = data_y.sum(axis=stratify_axis)[:, :9]

    # Convert each row (pattern) into a hashable type (tuple) for stratification.
    pattern_tuples = [tuple(pattern) for pattern in y_summed_patterns]

    # Count occurrences of each pattern
    pattern_counts = Counter(pattern_tuples)
    
    # Separate single-occurrence patterns from the rest
    single_occurrence_indices = [i for i, pattern in enumerate(pattern_tuples) if pattern_counts[pattern] == 1] # i is index in original data?
    multi_occurrence_indices = [i for i, pattern in enumerate(pattern_tuples) if pattern_counts[pattern] > 1]
    
    if len(single_occurrence_indices) > 0:
        print(f"Found {len(single_occurrence_indices)} samples with unique patterns. Adding to test set.")
        

        # Extract samples with multi-occurrence patterns for stratified split
        multi_x = data_x[multi_occurrence_indices]
        multi_y = data_y[multi_occurrence_indices]
        multi_patterns = [pattern_tuples[i] for i in multi_occurrence_indices]
        
        # Split the multi-occurrence patterns using stratification
        train_size = 1 - (test_size * len(pattern_tuples) - len(single_occurrence_indices)) / len(multi_patterns)
        train_size = max(0.1, min(0.9, train_size))  # Keep train_size between 0.1 and 0.9
        
        X_test, X_test_strat, y_test, y_test_strat = train_test_split(
            multi_x, multi_y, 
            test_size=1-train_size,
            shuffle=shuffle, 
            random_state=random_state,
            stratify=multi_patterns
        )
        
        # Add single-occurrence samples to test set
        X_valid = np.vstack([X_test_strat, data_x[single_occurrence_indices]])
        y_valid = np.vstack([y_test_strat, data_y[single_occurrence_indices]])
        
        return X_test, X_valid, y_test, y_valid
    else:
        # If no single-occurrence patterns, perform normal stratified split
        return train_test_split(
            data_x, data_y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=pattern_tuples
        )


#
class MemoryPositionalEncoding(nn.Module):
    """
    Simple 1D positional encoding for the memory (encoder output)
    to be used in the decoder's cross-attention mechanism.
    """

    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings once and for all
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Make pe not a model parameter (we don't need to train it)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Positional encoding to be added to x, same shape as x
        """
        # Get positional encoding for the length of the input sequence
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len, :]

        # Broadcast to batch dimension
        return pos_encoding
#
##
## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- Transformer_Encoder -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=270, nhead=5, num_decoder_layers=9, num_queries=5, dim_feedforward=512, dropout=0.1,
                 temp_cross_attention=1, seq_length=200, query_dropout_rate=0.0): 
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query_dropout_rate = query_dropout_rate 

        # Create activity queries - learnable parameters
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))  # 10 object queries
        # Create fixed random query embeddings (non-learnable)
        # query_embed = torch.randn(num_queries, d_model)
        # self.register_buffer('query_embed', query_embed)

        self.memory_pos_encoding = MemoryPositionalEncoding(d_model, seq_length, dropout)

        # Create decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            temp_cross_attention=temp_cross_attention,
            dropout=dropout
        )
        self.decoder_layers = nn.ModuleList([
            decoder_layer for _ in range(num_decoder_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection for classification
        # Assuming 10 is the number of activity classes
        self.class_embed = nn.Linear(d_model, 10)

        # self.tgt_embed = nn.Parameter(torch.zeros(num_queries, d_model))
        # Create fixed random target embeddings (non-learnable)
        tgt_embed = torch.zeros(num_queries, d_model)  # Using randn instead of zeros for random initialization
        self.register_buffer('tgt_embed', tgt_embed)

    def forward(self, memory):
        """
        Args:
            memory: Output from encoder (B, T, 270)
        Returns:
            outputs: List of output predictions from each decoder layer
        """
        B = memory.shape[0]

        # Initialize decoder input with zero queries
        tgt = self.tgt_embed.unsqueeze(0).expand(B, -1, -1)

        # Get positions
        query_pos = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        
        if self.training and self.query_dropout_rate > 0:
            num_queries = query_pos.shape[1]
            num_to_drop = int(num_queries * self.query_dropout_rate)
            
            if num_to_drop > 0:
                # Generate random indices of queries to drop. These indices are the same across the batch.
                drop_indices = torch.randperm(num_queries, device=query_pos.device)[:num_to_drop]
                
                # Create a multiplicative mask
                # mask will be (num_queries)
                query_mask = torch.ones(num_queries, device=query_pos.device)
                query_mask[drop_indices] = 0.0 # Set to 0.0 for dropout
                
                # Apply mask by broadcasting: (B, num_queries, d_model) * (1, num_queries, 1)
                query_pos = query_pos * query_mask.unsqueeze(0).unsqueeze(-1)

        memory_pos = self.memory_pos_encoding(memory)
        # Store intermediate outputs
        intermediate = []

        # Run through decoder layers
        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            output = layer(
                tgt=output,
                memory=memory,
                query_pos=query_pos,
                memory_pos=memory_pos  # New parameter passed to layer

            )

            pred = self.class_embed(output)
            intermediate.append(pred)

        return torch.stack(intermediate)  # Shape: [num_layers, B, num_queries, num_classes]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=270, nhead=8, dim_feedforward=2048, dropout=0.1, temp_cross_attention=1):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = TemperatureMultiheadAttention(d_model, nhead, dropout=dropout,
                                                        temperature=temp_cross_attention, batch_first=True)
        # self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn_weights = None
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos=None, memory_pos=None):
        # Cross attention
        tgt2, self.cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Self attention
        q = k = self.with_pos_embed(tgt, None)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Feed forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
class TemperatureMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, temperature=2.0, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.temperature = temperature
        self.attention_weights = None  # Store attention weights

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True):
        # Regular attention computation
        attn_output, attn_weights = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,  # Always compute weights
            attn_mask=attn_mask,
            average_attn_weights=False # gets the average across all heads
        )

        # Apply temperature scaling to attention output
        attn_output = attn_output / self.temperature

        return attn_output, attn_weights

class DETR_MultiUser(nn.Module):
    def __init__(self,
                 # Decoder specific parameters
                 num_decoder_layers: int,
                 temp_cross: float,
                 n_attention_heads_decoder: int,
                 num_queries: int,
                 dim_feedforward_decoder: int,
                 query_dropout_rate: float,
                 # Pre-trained components and their config
                 pretrained_cnn_feature_extractor: JEPA_CNNFeatureExtractor,
                 pretrained_transformer_encoder: JEPA_Transformer_Encoder,
                 jepa_config: dict
                ):
        super().__init__()

        if not pretrained_cnn_feature_extractor or not pretrained_transformer_encoder or not jepa_config:
            raise ValueError("Pretrained CNN, Transformer Encoder, and JEPA config must be provided.")

        self.feature_extractor = pretrained_cnn_feature_extractor
        self.encoder = pretrained_transformer_encoder # This is JEPA's Transformer_Encoder
        self.jepa_config = jepa_config # Store for forward pass logic

        # Derive feature_dim and embedding_time_dim from JEPA config for the decoder
        decoder_d_model = self.jepa_config["d_embedding"]
        decoder_seq_length = self.jepa_config["num_segments_total_view"] * self.jepa_config["cnn_embedding_time_dim"]

        self.decoder = TransformerDecoder( # This is the TransformerDecoder from detr.py/detr copy.py
            d_model=decoder_d_model,
            nhead=n_attention_heads_decoder,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_decoder,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross,
            seq_length=decoder_seq_length,
            query_dropout_rate=query_dropout_rate
        )

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        # --- Prepare full view for JEPA Encoder (mirroring JEPA_Model._prepare_full_view_CSI) ---
        required_len = self.jepa_config["num_segments_total_view"] * self.jepa_config["segment_length"]
        
        if x.shape[1] < required_len:
            padding_len = required_len - x.shape[1]
            # Pad the time dimension (dim=1). Assumes x is (B, T, C)
            x_padded = F.pad(x, (0, 0, 0, padding_len, 0, 0), mode='constant', value=0)
        else:
            x_padded = x

        max_t_init = x_padded.shape[1] - required_len
        
        if self.training:
            # Ensure max_t_init is not negative if x_padded.shape[1] == required_len
            # torch.randint(low, high, size) requires high > low.
            if max_t_init < 0: # Should not happen if padding is correct
                 t_init_val = 0
            elif max_t_init == 0:
                 t_init_val = 0
            else:
                 t_init_val = max_t_init + 1 # Upper bound for randint is exclusive

            t_init_batch = torch.randint(0, t_init_val if t_init_val > 0 else 1, (batch_size,), device=device)
            if max_t_init < 0 : t_init_batch.fill_(0) # if max_t_init was negative, force to 0
            elif max_t_init == 0: t_init_batch.fill_(0)


            x_full_view_list = [x_padded[i, t_init_batch[i] : t_init_batch[i] + required_len, :] for i in range(batch_size)]
            x_full_view = torch.stack(x_full_view_list)
        else: # Evaluation
            t_init = max_t_init // 2 if max_t_init >= 0 else 0
            x_full_view = x_padded[:, t_init : t_init + required_len, :]


        # --- Pass through pre-trained JEPA CNN Feature Extractor and Transformer Encoder ---
        # (mirroring JEPA_Model._process_full_view_for_target_encoder)
        num_total_segments = self.jepa_config["num_segments_total_view"]
        segment_len = self.jepa_config["segment_length"]
        input_channels_cnn = x_full_view.shape[-1]
        tokens_per_segment_cnn = self.jepa_config["cnn_embedding_time_dim"]
        cnn_out_channels = self.jepa_config["d_embedding"]
        max_total_tokens_in_view_jepa = num_total_segments * tokens_per_segment_cnn

        # Reshape for CNN
        cnn_input = x_full_view.reshape(
            batch_size * num_total_segments,
            segment_len,
            input_channels_cnn
        )
        # Pass through CNN
        cnn_output_tokens = self.feature_extractor(cnn_input) 

        # Reshape for Transformer Encoder
        transformer_input_tokens = cnn_output_tokens.reshape(
            batch_size,
            max_total_tokens_in_view_jepa,
            cnn_out_channels
        )

        # Create original token indices for JEPA's Transformer_Encoder
        original_indices_full_view = torch.arange(max_total_tokens_in_view_jepa, device=transformer_input_tokens.device)
        original_indices_full_view = original_indices_full_view.unsqueeze(0).expand(batch_size, -1)

        # Pass through JEPA's Transformer Encoder
        memory = self.encoder(
            src=transformer_input_tokens,
            token_original_indices=original_indices_full_view,
            src_key_padding_mask=None 
        )
        
        # --- Pass `memory` to the DETR Decoder ---
        outputs_class = self.decoder(memory)

        return outputs_class

class HungarianMatchingLoss(nn.Module):
    def __init__(self, cost_class_weight, aux_loss_weight, label_smoothing, class_imbalance_weight):
        super().__init__()
        self.cost_class = cost_class_weight
        self.aux_loss_weight = aux_loss_weight

        weights = torch.ones(10)
        weights[-1] = class_imbalance_weight
        weights = weights * (len(weights) / weights.sum())

        self.ce_loss = nn.CrossEntropyLoss(
            weight=weights.to(torch.device('cuda')),
            label_smoothing=label_smoothing
        )

    @torch.no_grad()
    def Hungarian_matching(self, outputs, targets):
        """
        Performs the matching between predictions and ground truth
        Args:
            outputs: Tensor of shape (batch_size, num_queries, num_classes)
            targets: Tensor of shape (batch_size, num_queries, num_classes)
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs.shape[:2]

        # Compute classification cost matrix
        out_prob = outputs.softmax(-1)  # [batch_size, num_queries, num_classes]
        tgt_ids = targets.argmax(-1)  # [batch_size, num_queries]

        indices = []
        # Process each batch independently
        for b in range(bs):
            # Compute cost matrix for current
            # Create cost matrix showing how well each prediction matches each target
            cost_matrix = -out_prob[b][:, tgt_ids[b]]  # [num_queries, num_queries]
            cost_matrix = self.cost_class * cost_matrix.cpu().numpy()

            # Run Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Convert to tensors and move to correct device
            row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=outputs.device) # Which queries to use
            col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=outputs.device) # Which targets they match to
            indices.append((row_ind, col_ind))
            """
            tgt_ids[b]
            Out[1]: tensor([1, 1, 1, 1, 9], device='cuda:0')
            Example how it works:
            [[-0.1645718 , -0.1645718 , -0.1645718 , -0.1645718 , -0.07412154],  # Query 0
             [-0.14723456, -0.14723456, -0.14723456, -0.14723456, -0.08399412],  # Query 1
             [-0.16388056, -0.16388056, -0.16388056, -0.16388056, -0.05493092],  # Query 2
             [-0.1868437 , -0.1868437 , -0.1868437 , -0.1868437 , -0.07625321],  # Query 3
             [-0.20076165, -0.20076165, -0.20076165, -0.20076165, -0.0671524 ]]  # Query 4
            row_ind = [0, 1, 2, 3, 4]  # Which queries to use
            col_ind = [0, 4, 2, 3, 1]  # Which targets they match to
            This means:
                
                Query 0 matches with target 0 (class 1)
                Query 1 matches with target 4 (class 9)
                Query 2 matches with target 2 (class 1)
                Query 3 matches with target 3 (class 1)
                Query 4 matches with target 1 (class 1)
            Another Example which is easier:
            Ground truth labels tgt_ids[b] = [8, 0, 9, 1, 8] (5 targets)
            # Cost Matrix:
            [[-0.0751, -0.1023, -0.0876, -0.2142, -0.0751],  # Query 0
             [-0.0911, -0.0929, -0.0980, -0.2144, -0.0911],  # Query 1
             [-0.0606, -0.0813, -0.1075, -0.2602, -0.0606],  # Query 2
             [-0.0867, -0.1077, -0.1456, -0.1916, -0.0867],  # Query 3
             [-0.0803, -0.1265, -0.1192, -0.1970, -0.0803]]  # Query 4
            
                IMPORTANT:
                This cost matrix was created by selecting probabilities from out_prob based on tgt_ids:

                Column 0: probabilities for class 8 (first target)
                Column 1: probabilities for class 0 (second target)
                Column 2: probabilities for class 9 (third target)
                Column 3: probabilities for class 1 (fourth target)
                Column 4: probabilities for class 8 (fifth target)
                
                row_ind = [0, 1, 2, 3, 4]    # Predictions to use
                col_ind = [4, 0, 3, 2, 1]    # Which targets they match to
            """

        return indices

    def _get_layer_loss(self, pred, target, indices):
        """Helper to compute loss for a single layer's predictions"""
        losses = []
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            pred_i = pred[batch_idx][pred_idx]
            tgt_i = target[batch_idx][tgt_idx]
            loss = self.ce_loss(pred_i, tgt_i.argmax(-1))
            losses.append(loss.mean()) ### REMOVE MEAN HERE! UNNECESSARY
        return torch.stack(losses).mean()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: If auxiliary losses enabled: Tensor of shape [num_layers + 1, B, num_queries, num_classes]
                    Otherwise: Tensor of shape [B, num_queries, num_classes]
            targets: Tensor of shape [B, num_queries, num_classes]
        """
        # Check if we have auxiliary outputs
        if outputs.dim() == 4:  # Has auxiliary outputs [num_layers + 1, B, num_queries, num_classes]
            # Split predictions from different decoder layers
            aux_outputs = outputs[:-1]  # Predictions from intermediate layers
            outputs_final = outputs[-1]  # Predictions from final layer

            # Calculate matching using only the final layer predictions
            indices = self.Hungarian_matching(outputs_final, targets)

            # Calculate loss for final predictions using final layer matching
            final_loss = self._get_layer_loss(outputs_final, targets, indices)

            # Calculate auxiliary losses using the SAME indices from final layer
            aux_losses = []
            for aux_output in aux_outputs:
                # Use the same matching indices for all auxiliary layers
                layer_loss = self._get_layer_loss(aux_output, targets, indices)
                aux_losses.append(layer_loss)

            # Combine losses
            aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0)
            total_loss = final_loss + self.aux_loss_weight * aux_loss

            return total_loss

        else:  # No auxiliary outputs, just compute regular loss
            indices = self.Hungarian_matching(outputs, targets)
            return self._get_layer_loss(outputs, targets, indices)



def run_that_detr(data_train_x,
                     data_train_y,
                     data_test_x,
                     data_test_y,
                     var_repeat=10):
    """
    [description]
    : run WiFi-based model Transformer_Encoder_DECODER
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ## Remove the internal validation split since validation data is now provided directly


    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(data_test_x, data_test_y,
                                                                            test_size=0.5,
                                                                            shuffle=True,
                                                                            random_state=39)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape = data_train_x[0].shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    # data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))


    # ===================================== Load Pre-trained JEPA Model =========================================
    jepa_pretrained_path = preset.get("jepa_pretrained_path")
    if not jepa_pretrained_path:
        raise ValueError("Path to pre-trained JEPA model ('jepa_pretrained_path') not found in preset.")

    jepa_checkpoint = torch.load(jepa_pretrained_path, map_location=device)
    loaded_jepa_config = jepa_checkpoint.get('config', JEPA_CONFIG) 
    
    temp_jepa_model = JEPA_Model(loaded_jepa_config).to(device)
    
    if 'model_state_dict' in jepa_checkpoint:
        temp_jepa_model.load_state_dict(jepa_checkpoint['model_state_dict'])
    elif 'online_cnn_feature_extractor' in jepa_checkpoint and 'online_transformer_encoder' in jepa_checkpoint:
         temp_jepa_model.online_cnn_feature_extractor.load_state_dict(jepa_checkpoint['online_cnn_feature_extractor'])
         temp_jepa_model.online_transformer_encoder.load_state_dict(jepa_checkpoint['online_transformer_encoder'])
    else: 
        temp_jepa_model.load_state_dict(jepa_checkpoint)

    pretrained_cnn = temp_jepa_model.online_cnn_feature_extractor
    pretrained_encoder = temp_jepa_model.online_transformer_encoder
    ## ========================================= Train & Evaluate =========================================
    result_accuracy = []
    result_ppp = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []
    result_avg_count_error = []

    #
    complexity_model_instance = DETR_MultiUser(
        num_decoder_layers=preset["nn"]["num_decoder_layers"],
        temp_cross=preset["nn"]["cross_attention_temp"],
        n_attention_heads_decoder=preset["nn"]["n_attention_heads"],
        num_queries=preset["nn"]["num_obj_queries"],
        dim_feedforward_decoder=preset["nn"]["dim_FFN"],
        query_dropout_rate=preset["nn"]["query_dropout_rate"],
        pretrained_cnn_feature_extractor=pretrained_cnn,
        pretrained_transformer_encoder=pretrained_encoder,
        jepa_config=loaded_jepa_config
    ).to(device)

    # get_model_complexity_info expects input_res as a tuple (e.g., (timesteps, features))
    var_macs, var_params = get_model_complexity_info(
        complexity_model_instance,
        var_x_shape, # (timesteps, features)
        as_strings=False,
        print_per_layer_stat=False, # Reduce verbosity
        verbose=False
    )
    del complexity_model_instance # Free memory

    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #

    for var_r in range(var_repeat):
        #
        ##
        var_mode = "multi_head"
        name_run = "Empty"
        # Modify run name to reflect JEPA pretraining
        jepa_model_name_part = os.path.splitext(os.path.basename(jepa_pretrained_path))[0]
        if preset["pretrained_path"]: # This was for DETR pretraining, adapt or remove
            name_run = f"DETR_SSL_{jepa_model_name_part}_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + preset["transfer_scenario"]
        else:
            name_run = f"DETR_SSL_{jepa_model_name_part}_{var_r}_" + "_".join(preset["data"]["environment"]) + "_NPT" # NPT for No Pretrained DETR (but SSL from JEPA)
        
        print("Repeat", var_r)
        run = wandb.init(
            project="test_detr_with_jepa_ssl", # Consider a new project name
            name= name_run + preset["wandb_name"] ,
            config=preset,
            reinit=True 
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_detr = DETR_MultiUser(
            num_decoder_layers=preset["nn"]["num_decoder_layers"],
            temp_cross=preset["nn"]["cross_attention_temp"],
            n_attention_heads_decoder=preset["nn"]["n_attention_heads"],
            num_queries=preset["nn"]["num_obj_queries"],
            dim_feedforward_decoder=preset["nn"]["dim_FFN"],
            query_dropout_rate=preset["nn"]["query_dropout_rate"],
            pretrained_cnn_feature_extractor=pretrained_cnn,
            pretrained_transformer_encoder=pretrained_encoder,
            jepa_config=loaded_jepa_config
        ).to(device)
        
        # --- Optimizer Setup: Freezing or Fine-tuning ---
        finetune_strategy = preset.get("finetune_strategy", "freeze_all") 
        params_to_optimize = []
        
        if finetune_strategy == "freeze_all":
            print("Freezing pre-trained CNN and Encoder. Training only Decoder.")
            for param in model_detr.feature_extractor.parameters():
                param.requires_grad = False
            for param in model_detr.encoder.parameters():
                param.requires_grad = False
            params_to_optimize.append({'params': model_detr.decoder.parameters(), 'lr': preset["nn"]["lr"]})

        elif finetune_strategy == "finetune_encoder_small_lr":
            print("Fine-tuning pre-trained CNN and Encoder with smaller LR. Training Decoder with main LR.")
            for param in model_detr.feature_extractor.parameters():
                param.requires_grad = True
            params_to_optimize.append({
                'params': model_detr.feature_extractor.parameters(),
                'lr': preset["nn"].get("lr_finetune_cnn", preset["nn"]["lr"] * 0.1)
            })
            for param in model_detr.encoder.parameters():
                param.requires_grad = True
            params_to_optimize.append({
                'params': model_detr.encoder.parameters(),
                'lr': preset["nn"].get("lr_finetune_encoder", preset["nn"]["lr"] * 0.1)
            })
            params_to_optimize.append({'params': model_detr.decoder.parameters(), 'lr': preset["nn"]["lr"]})
        
        elif finetune_strategy == "finetune_all_same_lr":
            print("Fine-tuning all components (CNN, Encoder, Decoder) with the same LR.")
            for param in model_detr.parameters():
                param.requires_grad = True
            params_to_optimize.append({'params': model_detr.parameters(), 'lr': preset["nn"]["lr"]})
        
        else: 
            print(f"Unknown finetune_strategy '{finetune_strategy}'. Defaulting to training only Decoder.")
            for param in model_detr.feature_extractor.parameters():
                param.requires_grad = False
            for param in model_detr.encoder.parameters():
                param.requires_grad = False
            params_to_optimize.append({'params': model_detr.decoder.parameters(), 'lr': preset["nn"]["lr"]})

        optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=preset["nn"]["weight_decay"])
        # Original detr.py used Adam. If you prefer Adam:
        # optimizer = torch.optim.Adam(params_to_optimize, weight_decay=preset["nn"]["weight_decay"])

        loss_fn = HungarianMatchingLoss(
            cost_class_weight=preset["nn"]["loss"]["cost_class_weight"],
            aux_loss_weight=preset["nn"]["loss"]["aux_loss_weight"],
            label_smoothing=preset["nn"]["loss"]["label_smoothing"],
            class_imbalance_weight=preset["nn"]["loss"]["class_imbalance_weight"]
        )
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model=model_detr,
                                optimizer=optimizer,
                                loss=loss_fn, # Use the renamed loss function
                                data_train_set=data_train_set,
                                data_test_set=data_valid_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device,
                                var_mode=var_mode)
        # Save model components based on scenario
        if preset.get("save_model"):
            save_model_components(preset, model_detr)
        #
        var_time_1 = time.time()
        #


        ## ---------------------------------------- Test ------------------------------------------
        #
        model_detr.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_detr(torch.from_numpy(data_test_x).to(device))
        #
        # predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        layers_idxs = ["layer_"+str(i) for i in range(preset["nn"]["num_decoder_layers"])]
        last_layer_only = True
        # Store results for each layer
        all_layers_results = {}
        dict_layer_acc = performance_metrics(data_test_y, predict_test_y, var_mode=var_mode)

        # Process each layer separately
        for idx, layer_idx in enumerate(layers_idxs):
            layer_metrics = dict_layer_acc[layer_idx]
            if var_r == 0:  # Initialize lists on first repeat
                result_ppp.append([])
                result_time_train.append([])
                result_time_test.append([])
                result_total_error.append([])
                result_precision.append([])
                result_recall.append([])
                result_f1_score.append([])
                result_avg_count_error.append([])
                result_accuracy.append([])
            result_accuracy[idx].append(layer_metrics['accuracy'])
            result_ppp[idx].append(layer_metrics['perfect_prediction_percentage'])
            result_time_train[idx].append(var_time_1 - var_time_0)
            result_time_test[idx].append(var_time_2 - var_time_1)
            result_total_error[idx].append(layer_metrics['total_error'])
            result_precision[idx].append(layer_metrics['precision'])
            result_recall[idx].append(layer_metrics['recall'])
            result_f1_score[idx].append(layer_metrics['f1_score'])
            result_avg_count_error[idx].append(layer_metrics['mean_count_error'])

        if last_layer_only:
            layer_metrics = dict_layer_acc["layer_" +str(preset["nn"]["num_decoder_layers"] - 1)]
            wandb.log({
                f"test_results/repeat": var_r,
                f"test_results/train_time": var_time_1 - var_time_0,
                f"test_results/test_time": var_time_2 - var_time_1,
                f"test_results/TOTAL_TESTSET_ERROR": layer_metrics['total_error'],
                f"test_results/TOTAL_TESTSET_perfect_prediction_percentage": layer_metrics[
                    'perfect_prediction_percentage'],
                f"test_results/TOTAL_ACCURACY": layer_metrics['accuracy'],
                f"test_results/mean_count_error": layer_metrics['mean_count_error'],
                f"test_results/error_per_person_1": layer_metrics['error_per_person'][0],
                f"test_results/error_per_person_2": layer_metrics['error_per_person'][1],
                f"test_results/error_per_person_3": layer_metrics['error_per_person'][2],
                f"test_results/error_per_person_4": layer_metrics['error_per_person'][3],
                f"test_results/error_per_person_5": layer_metrics['error_per_person'][4],
                f"test_results/precision": layer_metrics['precision'],
                f"test_results/recall": layer_metrics['recall'],
                f"test_results/f1_score": layer_metrics['f1_score']
            }, step=var_r + 100000)

            print(
                  "- Total Error %.6f" % layer_metrics['total_error'],
                  "- Perfect Prediction Percentage %.6f" % layer_metrics['perfect_prediction_percentage'])


    for layer_idx_num, layer_idx in enumerate(layers_idxs):
        wandb.log({
            f"test_results/{layer_idx}/avg_PPP": sum(result_ppp[layer_idx_num]) / len(result_ppp[layer_idx_num]),
            f"test_results/{layer_idx}/avg_train_time": sum(result_time_train[layer_idx_num]) / len(
                result_time_train[layer_idx_num]),
            f"test_results/{layer_idx}/avg_test_time": sum(result_time_test[layer_idx_num]) / len(
                result_time_test[layer_idx_num]),
            f"test_results/{layer_idx}/avg_total_error": sum(result_total_error[layer_idx_num]) / len(
                result_total_error[layer_idx_num]),
            f"test_results/{layer_idx}/avg_precision": sum(result_precision[layer_idx_num]) / len(
                result_precision[layer_idx_num]),
            f"test_results/{layer_idx}/avg_recall": sum(result_recall[layer_idx_num]) / len(result_recall[layer_idx_num]),
            f"test_results/{layer_idx}/avg_f1_score": sum(result_f1_score[layer_idx_num]) / len(result_f1_score[layer_idx_num]),
            f"test_results/{layer_idx}/avg_count_error": sum(result_avg_count_error[layer_idx_num]) / len(
                result_avg_count_error[layer_idx_num]),
            f"test_results/{layer_idx}/avg_accuracy": sum(result_accuracy[layer_idx_num]) / len(
                result_accuracy[layer_idx_num])
        })  # Use an even larger offset for averages

    # Use the last layer for visualization and final results
    last_layer = layers_idxs[-1]
    last_layer_predictions = predict_test_y[last_layer] if isinstance(predict_test_y, dict) else predict_test_y
    # dict_true_acc = all_layers_results[last_layer]

    # Run visualization with the last layer's predictions
    
    log_random_attention_weights_final(model_detr, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), 1000000000)
    
    viz_stats = visualize_model_performance(
        y_pred=last_layer_predictions,
        y_true=data_test_y,
        var_mode=var_mode,
        save_dir=f'./visualizations/experiment_{var_r}_{var_mode}'
    )
    print("\nDetailed Performance Analysis:")
    print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    print("\nClass-wise Mean Absolute Error:")
    for i, error in enumerate(viz_stats['class_wise_mae']):
        print(f"Class {i}: {error:.4f}")
    print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return all_layers_results

