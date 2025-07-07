from torch import nn
import torch
from model.modules.elements import DepthwiseSeparableConv, DilatedConvBlock, Dilated_Blocks
from preset import preset

# class PCAFeatureExtractor(nn.Module):
#     def __init__(self, input_channels=270, output_channels=16, embedding_time_dim=1, pca_components=None):
#         super(PCAFeatureExtractor, self).__init__()
#         self.embedding_time_dim = embedding_time_dim
#
#         if pca_components is None:
#             raise ValueError("PCA components must be provided.")
#         self.register_buffer('pca_components', pca_components)
#
#         pca_output_dim = self.pca_components.shape[1]
#         c_intermediate = 64  # Intermediate channel size
#
#         # 1. Simplified feature extraction and temporal reduction
#         # This single layer replaces the low-pass filter and the first ds_conv
#         # Assuming input time is 40, stride=4 reduces it to 10
#         self.feature_extractor = DepthwiseSeparableConv(pca_output_dim, c_intermediate, kernel_size=5, padding=2,
#                                                         stride=4)
#         self.bn1 = nn.BatchNorm1d(c_intermediate)
#         self.relu1 = nn.ReLU()
#
#         # 2. Channel Adjustment
#         self.channel_adjust = nn.Conv1d(c_intermediate, output_channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm1d(output_channels)
#         self.relu2 = nn.ReLU()
#
#         # 3. Final Temporal Reduction to a single token
#         # From T=10 to T=1. Kernel size 10 and stride 10 will map 10 -> 1.
#         self.final_reduction_conv = nn.Conv1d(output_channels, output_channels, kernel_size=10, stride=10)
#         self.bn_final = nn.BatchNorm1d(output_channels)
#         self.relu_final = nn.ReLU()
#
#     def forward(self, x):
#         # Input x shape: (batch, time, channels_in) e.g. (B, 40, 270)
#         # PCA projection
#         x = torch.matmul(x, self.pca_components)  # (B, 40, pca_dim)
#
#         x = x.permute(0, 2, 1)  # (batch, pca_dim, time) e.g. (B, pca_dim, 40)
#
#         # 1. Feature extraction and temporal reduction
#         x = self.feature_extractor(x)  # T: 40 -> 10
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         # 2. Channel Adjustment
#         x = self.channel_adjust(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         # 3. Final Temporal Reduction
#         x = self.final_reduction_conv(x)  # T: 10 -> 1
#         x = self.bn_final(x)
#         x = self.relu_final(x)
#
#         # Output shape: (batch, 1, output_channels)
#         return x.permute(0, 2, 1)

class PCAFeatureExtractor(nn.Module):
    def __init__(self, input_channels=270, output_channels=16, embedding_time_dim=10, pca_components=None):
        super(PCAFeatureExtractor, self).__init__()

        if pca_components is None:
            raise ValueError("PCA components must be provided.")
        self.register_buffer('pca_components', pca_components)

        pca_output_dim = self.pca_components.shape[1]
        c_initial = 64  # Channels after initial convolution
        c_hierarchical_out = 100  # Channels after hierarchical dilated blocks

        # 1. Low-pass filter after PCA
        self.low_pass_conv = nn.Conv1d(pca_output_dim, 48, kernel_size=5, padding=2)
        self.bn_initial = nn.BatchNorm1d(48)
        self.relu_initial = nn.ReLU()

        self.ds_conv1 = DepthwiseSeparableConv(48, 64, kernel_size=5, padding=2, stride=4)  # T: 200 -> 50
        self.bn_ds1 = nn.BatchNorm1d(64)
        self.relu_ds1 = nn.ReLU()

        # 2. Hierarchical Dilated Convolutions (on T=50)
        self.hierarchical_dilated_1 = DilatedConvBlock(c_initial, c_initial, dilation_rate=1)
        self.hierarchical_dilated_2 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=2)
        # self.hierarchical_dilated_3 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=4)

        # 3. Second Temporal Reduction (T: 50 -> 25)
        self.ds_conv2 = DepthwiseSeparableConv(c_hierarchical_out, c_hierarchical_out, kernel_size=5, padding=2, stride=2)  # T: 100 -> 50
        self.bn_ds2 = nn.BatchNorm1d(c_hierarchical_out)
        self.relu_ds2 = nn.ReLU()

        # 4. Channel Adjustment to output_channels (T: 25)
        self.channel_adjust_before_parallel = nn.Conv1d(c_hierarchical_out, output_channels, kernel_size=1)
        self.bn_adjust = nn.BatchNorm1d(output_channels)
        self.relu_adjust = nn.ReLU()

        # 5. Parallel Dilated Convolutions (on T=25)
        self.parallel_dilated_blocks = Dilated_Blocks(output_channels)
        self.bn_parallel = nn.BatchNorm1d(output_channels)
        self.relu_parallel = nn.ReLU()

        # 6. Final Temporal Reduction (from T=25 to 5)
        self.final_reduction_conv = nn.Conv1d(output_channels, output_channels,
                                              kernel_size=5, stride=5, padding=2)
        self.bn_final = nn.BatchNorm1d(output_channels)
        self.relu_final = nn.ReLU()

    def forward(self, x):
        # Input x shape: (batch, time, channels_in) e.g. (B, 200, 270)
        # PCA projection
        x = torch.matmul(x, self.pca_components)  # (B, 200, 32)

        x = x.permute(0, 2, 1)  # (batch, channels_out_pca, time) e.g. (B, 32, 200)

        # 1. Low-pass filter
        x = self.low_pass_conv(x)  # (B, 128, 200)
        x = self.bn_initial(x)
        x = self.relu_initial(x)

        x = self.ds_conv1(x)  # T: 200 -> 100
        x = self.bn_ds1(x)
        x = self.relu_ds1(x)

        # 2. Hierarchical Dilated Convolutions
        x = self.hierarchical_dilated_1(x)
        x = self.hierarchical_dilated_2(x)
        # x = self.hierarchical_dilated_3(x)  # Channels: c_initial -> c_hierarchical_out =64

        # 3. Second Temporal Reduction
        x = self.ds_conv2(x)  # T: 100 -> 50
        x = self.bn_ds2(x)
        x = self.relu_ds2(x)

        # 4. Channel Adjustment
        x = self.channel_adjust_before_parallel(x)  # Channels: c_hierarchical_out -> output_channels
        x = self.bn_adjust(x)
        x = self.relu_adjust(x)

        # 5. Parallel Dilated Convolutions
        x = self.parallel_dilated_blocks(x)
        x = self.bn_parallel(x)
        x = self.relu_parallel(x)

        # 6. Final Temporal Reduction
        x = self.final_reduction_conv(x)  # T: 50 -> embedding_time_dim (approx)
        x = self.bn_final(x)
        x = self.relu_final(x)

        return x.permute(0, 2, 1)  # (batch, embedding_time_dim, output_channels)

"""


# class CNNFeatureExtractor(nn.Module):
#     def __init__(self, input_channels=270, output_channels=16, embedding_time_dim=10): # Default embedding_time_dim set to 10
#         super(CNNFeatureExtractor, self).__init__()
#         self.embedding_time_dim = embedding_time_dim
#
#         c_initial = 128  # Channels after initial convolution
#         c_hierarchical_out = 64  # Channels after hierarchical dilated blocks
#
#         # 1. Initial Processing & First Temporal Reduction (T: 200 -> 100)
#         self.initial_conv = nn.Conv1d(input_channels, c_initial, kernel_size=1, padding=0)
#         self.bn_initial = nn.BatchNorm1d(c_initial)
#         self.relu_initial = nn.ReLU()
#
#         self.ds_conv1 = DepthwiseSeparableConv(c_initial, c_initial, kernel_size=5, padding=2, stride=2) # T: 200 -> 100
#         self.bn_ds1 = nn.BatchNorm1d(c_initial)
#         self.relu_ds1 = nn.ReLU()
#
#         # 2. Hierarchical Dilated Convolutions (on T=100)
#         # Input: (B, c_initial, 100)
#         self.hierarchical_dilated_1 = DilatedConvBlock(c_initial, c_initial, dilation_rate=1)
#         self.hierarchical_dilated_2 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=2)
#         # Reduce channels in the last hierarchical block
#         # self.hierarchical_dilated_3 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=4)
#         # Output: (B, c_hierarchical_out, 100)
#
#         # 3. Second Temporal Reduction (T: 100 -> 50)
#         # Input: (B, c_hierarchical_out, 100)
#         self.ds_conv2 = DepthwiseSeparableConv(c_hierarchical_out, c_hierarchical_out, kernel_size=5, padding=2, stride=2) # T: 100 -> 50
#         self.bn_ds2 = nn.BatchNorm1d(c_hierarchical_out)
#         self.relu_ds2 = nn.ReLU()
#         # Output: (B, c_hierarchical_out, 50)
#
#         # 4. Channel Adjustment to output_channels (T: 50)
#         # Input: (B, c_hierarchical_out, 50)
#         self.channel_adjust_before_parallel = nn.Conv1d(c_hierarchical_out, output_channels, kernel_size=1)
#         self.bn_adjust = nn.BatchNorm1d(output_channels)
#         self.relu_adjust = nn.ReLU()
#         # Output: (B, output_channels, 50)
#
#         # 5. Parallel Dilated Convolutions (on T=50)
#         # Input: (B, output_channels, 50)
#         self.parallel_dilated_blocks = Dilated_Blocks(output_channels) # Assumes Dilated_Blocks handles its internal ReLUs
#         self.bn_parallel = nn.BatchNorm1d(output_channels)
#         self.relu_parallel = nn.ReLU()
#         # Output: (B, output_channels, 50)
#
#         # 6. Final Temporal Reduction (from T=50 to embedding_time_dim)
#         # Input: (B, output_channels, 50)
#
#         self.final_reduction_conv = nn.Conv1d(output_channels, output_channels,
#                                               kernel_size=5, stride=5, padding=2)
#         self.bn_final = nn.BatchNorm1d(output_channels)
#         self.relu_final = nn.ReLU()
#         # Output: (B, output_channels, embedding_time_dim)
#
#     def forward(self, x):
#         # Input x shape: (batch, time, channels_in) e.g. (B, 200, 270)
#         x = x.permute(0, 2, 1)  # (batch, channels_in, time) e.g. (B, 270, 200)
#
#         # 1. Initial Processing & First Temporal Reduction
#         x = self.initial_conv(x) #d from 270 ->128
#         x = self.bn_initial(x)
#         x = self.relu_initial(x)
#
#         x = self.ds_conv1(x) # T: 200 -> 100
#         x = self.bn_ds1(x)
#         x = self.relu_ds1(x)
#
#         # 2. Hierarchical Dilated Convolutions
#         x = self.hierarchical_dilated_1(x)
#         x = self.hierarchical_dilated_2(x)
#         # x = self.hierarchical_dilated_3(x) # Channels: c_initial -> c_hierarchical_out =64
#
#         # 3. Second Temporal Reduction
#         x = self.ds_conv2(x) # T: 100 -> 50
#         x = self.bn_ds2(x)
#         x = self.relu_ds2(x)
#
#         # 4. Channel Adjustment
#         x = self.channel_adjust_before_parallel(x) # Channels: c_hierarchical_out -> output_channels
#         x = self.bn_adjust(x)
#         x = self.relu_adjust(x)
#
#         # 5. Parallel Dilated Convolutions
#         x = self.parallel_dilated_blocks(x)
#         x = self.bn_parallel(x)
#         x = self.relu_parallel(x)
#
#         # 6. Final Temporal Reduction
#         x = self.final_reduction_conv(x) # T: 50 -> embedding_time_dim (approx)
#         x = self.bn_final(x)
#         x = self.relu_final(x)
#
#         # print(f"Final shape after CNNFeatureExtractor: {x.shape}")
#         return x.permute(0, 2, 1)  # (batch, embedding_time_dim, output_channels)

"""


class Predictor(nn.Module):
    def __init__(self, preset):
        super().__init__()
        self.encoder_d_model = preset["nn"]["d_embedding"]
        self.predictor_d_model = preset["jepa"]["predictor_d_model"]
        self.max_total_tokens_in_view = preset["jepa"]["num_segments_total_view"] * preset["cnn_embedding_time_dim"]

        # Learnable shared mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.predictor_d_model))

        # Projection layers
        self.input_proj = nn.Linear(self.encoder_d_model, self.predictor_d_model)
        self.output_proj = nn.Linear(self.predictor_d_model, self.encoder_d_model)

        # Positional embedding
        # HERE WE Want TO BORROW FROM CONTEXT ENCODER!!!!!#########################
        # self.pos_encoder = nn.Embedding(self.max_total_tokens_in_view, self.predictor_d_model)
        self.shared_pos_encoder = None

        # Transformer encoder
        predictor_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.predictor_d_model,
            nhead=preset["jepa"]["predictor_attention_heads"],
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            predictor_encoder_layer,
            num_layers=preset["jepa"]["predictor_layers"],
        )

        # Layer norms
        self.layer_norm_input = nn.LayerNorm(self.predictor_d_model)
        self.layer_norm_output = nn.LayerNorm(self.encoder_d_model)
        # Project positional embeddings from encoder dimension to predictor dimension
        self.pos_proj = nn.Linear(self.encoder_d_model, self.predictor_d_model)

    def set_shared_pos_encoder(self, encoder_pos_layer):
        """
        Set reference to the encoder's positional embedding layer.

        Args:
            encoder_pos_layer: The nn.Embedding layer from the context encoder
        """
        self.shared_pos_encoder = encoder_pos_layer



    def get_target_pos_embedding(self, target_token_indices):
        """
        Get positional embeddings from shared encoder and project to predictor dimension.
        Args:
            target_token_indices: Shape (batch_size, num_blocks, num_target_tokens)
        Returns:
            target_pos_embeddings: Shape (batch_size, num_blocks, num_target_tokens, predictor_d_model)
        """
        if self.shared_pos_encoder is None:
            raise ValueError("Shared positional encoder not set. Call set_shared_pos_encoder() first.")

        batch_size, num_blocks, num_target_tokens = target_token_indices.shape

        # Reshape to flat tensor for embedding lookup
        flat_indices = target_token_indices.reshape(-1)

        # Get positional embeddings from shared encoder (in encoder dimension)
        flat_pos_embeddings = self.shared_pos_encoder(flat_indices)
        # Shape: (B*num_blocks*num_target_tokens, encoder_d_model)

        # Project to predictor dimension
        flat_pos_embeddings_proj = self.pos_proj(flat_pos_embeddings)
        # Shape: (B*num_blocks*num_target_tokens, predictor_d_model)

        # Reshape back to preserve block structure
        target_pos_embeddings = flat_pos_embeddings_proj.reshape(
            batch_size, num_blocks, num_target_tokens, self.predictor_d_model
        )
        return target_pos_embeddings


    def forward(self, z_context_online, context_mask, target_token_indices):
        """
        Args:
            z_context_online (torch.Tensor): Encoded context tokens from the online encoder.
                                             Shape: (batch_size, num_context_tokens, encoder_d_model).
            context_mask (torch.Tensor): Boolean mask for z_context_online true for places that we need to ignore in context
                                        Shape: (batch_size, num_context_tokens).
            target_token_indices (torch.Tensor): The original absolute indices of the tokens to be predicted.
                                                 Shape: (batch_size, num_blocks, num_target_tokens).

        Returns:
            torch.Tensor: The predicted token representations.
                          Shape: (batch_size, num_blocks, num_target_tokens, encoder_d_model).
        """
        batch_size, num_blocks, num_target_tokens = target_token_indices.shape

        # Step 1: Get positional embeddings for all target blocks
        target_pos_embeddings = self.get_target_pos_embedding(target_token_indices)
        # Shape: (B, num_blocks, num_target_tokens, predictor_dim)

        # Step 2: Project context to predictor dimension and apply layer norm
        projected_context = self.input_proj(z_context_online)  # Shape: (B, num_context, predictor_dim)
        projected_context = self.layer_norm_input(projected_context)

        # Step 3: Prepare mask token (shared across all blocks)
        # Note: We'll expand this per block to avoid memory issues with large num_blocks

        # Initialize output tensor
        predicted_targets = torch.zeros(
            batch_size, num_blocks, num_target_tokens, self.encoder_d_model,
            device=z_context_online.device, dtype=z_context_online.dtype
        )

        # Process each target block separately
        for block_idx in range(num_blocks):
            # Step 4: Get positional embeddings for current block
            target_pos_embeddings_block = target_pos_embeddings[:, block_idx, :, :]
            # Shape: (B, num_target_tokens, predictor_dim)

            # Step 5: Create query tokens (mask token + positional embeddings)
            mask_tokens = self.mask_token.expand(batch_size, num_target_tokens, self.predictor_d_model)
            query_tokens = mask_tokens + target_pos_embeddings_block
            # Shape: (B, num_target_tokens, predictor_dim)

            # Step 6: Concatenate context and query tokens
            combined_sequence = torch.cat([projected_context, query_tokens], dim=1)
            # Shape: (B, num_context + num_target_tokens, predictor_dim)

            # Step 7: Create combined attention mask
            query_padding_mask = torch.full(
                (batch_size, num_target_tokens),
                fill_value=False,
                device=context_mask.device,
                dtype=context_mask.dtype
            )
            combined_mask = torch.cat([context_mask, query_padding_mask], dim=1)
            # Shape: (B, num_context + num_target_tokens)

            # Step 8: Apply transformer encoder
            predictor_output = self.transformer_encoder(
                src=combined_sequence,
                src_key_padding_mask=combined_mask
            )
            # Shape: (B, num_context + num_target_tokens, predictor_dim)

            # Step 9: Extract predictions for query tokens only
            pred_narrow = predictor_output[:, -num_target_tokens:, :]
            # Shape: (B, num_target_tokens, predictor_dim)

            # Step 10: Project back to encoder dimension and apply layer norm
            predicted_block = self.output_proj(pred_narrow)
            predicted_block = self.layer_norm_output(predicted_block)
            # Shape: (B, num_target_tokens, encoder_d_model)

            # Store predictions for this block
            predicted_targets[:, block_idx, :, :] = predicted_block

        return predicted_targets


class Transformer_Encoder(nn.Module):
    """
    Transformer Encoder module.
    Processes a sequence of tokens and applies self-attention.
    Uses learnable positional embeddings based on the token's original absolute index.
    Handles variable sequence lengths using padding masks.
    """

    def __init__(self, d_model, nhead, num_layers,
                 max_total_tokens):  # d_model is feature_dim, max_total_tokens is the max possible original index + 1
        super().__init__()
        self.d_model = d_model
        # Standard Transformer Encoder Layer.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        # Stack of Transformer Encoder Layers.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Learnable positional embeddings for each possible original token position.
        self.pos_encoder = nn.Embedding(max_total_tokens, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, token_original_indices, src_key_padding_mask=None):
        # Get positional embeddings based on original indices.
        pos_embeddings = self.pos_encoder(token_original_indices)

        # Zero out positional embeddings for padded positions (if mask provided)
        if src_key_padding_mask is not None:
            pos_embeddings = pos_embeddings.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)

        src = src + pos_embeddings
        src = self.layer_norm(src)

        if src_key_padding_mask is not None:
            output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer_encoder(src)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=20, nhead=2, num_decoder_layers=9, num_queries=5, dim_feedforward=512, dropout=0.1,
                 temp_cross_attention=1, query_dropout_rate=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query_dropout_rate = query_dropout_rate

        # Create activity queries - learnable parameters
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))

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

        tgt_embed = torch.zeros(num_queries, d_model)  # Using randn instead of zeros for random initialization
        self.register_buffer('tgt_embed', tgt_embed)

    def forward(self, context_tokens, src_key_padding_mask=None):
        """
        Args:
            context_tokens: Context token representations from JEPA encoder 
                           Shape: (batch_size, max_context_tokens, d_model)
            token_original_indices: Original indices of context tokens (optional)
                                   Shape: (batch_size, max_context_tokens)
            src_key_padding_mask: Padding mask for context tokens
                                 Shape: (batch_size, max_context_tokens)
                                 True for padded positions
        Returns:
            outputs: List of output predictions from each decoder layer
                    Shape: [num_layers, batch_size, num_queries, num_classes]
        """
        B = context_tokens.shape[0]

        # Initialize decoder input with zero queries
        tgt = self.tgt_embed.unsqueeze(0).expand(B, -1, -1)

        # Get query positions
        query_pos = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        # Store intermediate outputs
        intermediate = []

        # Run through decoder layers
        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            output = layer(
                tgt=output,
                memory=context_tokens,
                query_pos=query_pos,
                memory_pos=None,  # Context tokens already have positional information
                key_padding_mask=src_key_padding_mask  # Pass padding mask
            )

            pred = self.class_embed(output)
            intermediate.append(pred)

        return torch.stack(intermediate)  # Shape: [num_layers, B, num_queries, num_classes]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=20, nhead=2, dim_feedforward=512, dropout=0.1, temp_cross_attention=1):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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

    def forward(self, tgt, memory, query_pos=None, memory_pos=None, key_padding_mask=None):
        # Cross attention with context tokens
        tgt2, self.cross_attn_weights = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),  # query
            self.with_pos_embed(memory, memory_pos),  # key
            memory,  # value
            key_padding_mask=key_padding_mask  # keep this as keyword
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

