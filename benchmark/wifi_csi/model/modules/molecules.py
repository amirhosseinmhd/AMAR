import math
from torch import nn
import torch
from model.modules.elements import DepthwiseSeparableConv, DilatedConvBlock, Dilated_Blocks
from preset import preset
import torch.nn.functional as F

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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, initial_embeddings=None):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if initial_embeddings is not None:
            self.embedding.weight.data.copy_(initial_embeddings)
        else:
            self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # convert inputs from B, T, C -> B, T, C
        inputs_contiguous = inputs.contiguous()
        input_shape = inputs_contiguous.shape
        
        # Flatten input
        flat_input = inputs_contiguous.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Straight-through estimator
        quantized_st = inputs + (quantized - inputs).detach()
        
        return quantized_st, quantized, encoding_indices.view(input_shape[0], input_shape[1])

    def compute_loss(self, continuous_embeddings, quantized_embeddings):
        # Commitment loss
        commitment_loss = F.mse_loss(continuous_embeddings.detach(), quantized_embeddings)
        
        # Codebook loss
        codebook_loss = F.mse_loss(continuous_embeddings, quantized_embeddings.detach())
        
        return codebook_loss + self.commitment_cost * commitment_loss
        
class PCAFeatureExtractor(nn.Module):
    def __init__(self, input_channels=270, output_channels=16, pca_components=None):
        super(PCAFeatureExtractor, self).__init__()

        self.register_buffer('pca_components', pca_components)

        pca_output_dim = 64
        c_initial = 64  # Channels after initial convolution
        c_hierarchical_out = 64  # Channels after hierarchical dilated blocks

        # Alternative to PCA: learnable convolution for dimensionality reduction
        self.low_pass_conv_pca_replace = nn.Conv1d(input_channels, pca_output_dim, kernel_size=3, padding="same")
        self.bn_initial_pca_replace = nn.BatchNorm1d(pca_output_dim)
        self.relu_initial_pca = nn.ReLU()

        # 1. Low-pass filter with major temporal downsampling
        # Temporal reduction: T -> T/4 (reduces input length by factor of 4)
        self.low_pass_conv = nn.Conv1d(pca_output_dim, 48, kernel_size=5, padding=2, stride=4)
        self.bn_initial = nn.BatchNorm1d(48)
        self.relu_initial = nn.ReLU()

        # 2. First depthwise separable convolution with temporal downsampling
        # Temporal reduction: T -> T/2 (halves the previous length)
        self.ds_conv1 = DepthwiseSeparableConv(48, 64, kernel_size=5, padding=2, stride=2)
        self.bn_ds1 = nn.BatchNorm1d(64)
        self.relu_ds1 = nn.ReLU()

        # 3. Hierarchical Dilated Convolutions (no temporal reduction)
        # Temporal: unchanged, Channels: 64 -> 64 -> 100
        self.hierarchical_dilated_1 = DilatedConvBlock(c_initial, c_initial, dilation_rate=1)
        self.hierarchical_dilated_2 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=2)
        # self.hierarchical_dilated_3 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=4)

        # 4. Second temporal reduction with depthwise separable convolution
        # Temporal reduction: T -> T/2 (halves the previous length again)
        self.ds_conv2 = DepthwiseSeparableConv(c_hierarchical_out, c_hierarchical_out, kernel_size=5, padding=2,
                                               stride=2)
        self.bn_ds2 = nn.BatchNorm1d(c_hierarchical_out)
        self.relu_ds2 = nn.ReLU()

        # 5. Channel adjustment to target output channels (no temporal reduction)
        # Temporal: unchanged, Channels: c_hierarchical_out (100) -> output_channels (16)
        self.channel_adjust_before_parallel = nn.Conv1d(c_hierarchical_out, output_channels, kernel_size=1)
        self.bn_adjust = nn.BatchNorm1d(output_channels)
        self.relu_adjust = nn.ReLU()

        # 6. Parallel dilated convolutions (no reduction in any dimension)
        # Temporal: unchanged, Channels: unchanged
        self.parallel_dilated_blocks = Dilated_Blocks(output_channels)
        self.bn_parallel = nn.BatchNorm1d(output_channels)
        self.relu_parallel = nn.ReLU()

        # 7. Final temporal reduction to create compact embeddings
        # Temporal reduction: T -> T/5 (reduces previous length by factor of 5)
        # output_size = floor((input_size + 2×padding - kernel_size) / stride) + 1

        self.final_reduction_conv = nn.Conv1d(output_channels, output_channels,
                                              kernel_size=5, stride=3, padding=1)
        self.bn_final = nn.BatchNorm1d(output_channels)
        self.relu_final = nn.ReLU()

    def forward(self, x):
        # Input x shape: (batch, time, channels) - any temporal length T, channels=270

        if self.pca_components is not None:
            # PCA projection: reduce channel dimension 270 -> PCA_components
            # Temporal dimension unchanged: (B, T, 270) -> (B, T, pca_dim)
            x = torch.matmul(x, self.pca_components)
            x = x.permute(0, 2, 1)  # Conv1d format: (B, pca_dim, T)
        else:
            # Alternative: learnable channel reduction 270 -> 64
            # Temporal dimension unchanged: (B, T, 270) -> (B, T, 64)
            x = x.permute(0, 2, 1)  # Conv1d format: (B, 270, T)
            x = self.low_pass_conv_pca_replace(x)  # (B, 64, T)
            x = self.bn_initial_pca_replace(x)
            x = self.relu_initial_pca(x)

        # 1. Low-pass filtering with major temporal downsampling
        # Reduces temporal dimension by 4x: (B, channels, T) -> (B, 48, T/4)
        x = self.low_pass_conv(x)
        x = self.bn_initial(x)
        x = self.relu_initial(x)

        # 2. First depthwise separable convolution
        # Halves temporal dimension: (B, 48, T) -> (B, 64, T/2)
        x = self.ds_conv1(x)
        x = self.bn_ds1(x)
        x = self.relu_ds1(x)

        # 3. Hierarchical dilated convolutions
        # Capture multi-scale temporal patterns, no size reduction
        # (B, 64, T) -> (B, 64, T) -> (B, 100, T)
        x = self.hierarchical_dilated_1(x)
        x = self.hierarchical_dilated_2(x)
        # x = self.hierarchical_dilated_3(x)

        # 4. Second temporal reduction
        # Halves temporal dimension again: (B, 100, T) -> (B, 100, T/2)
        x = self.ds_conv2(x)
        x = self.bn_ds2(x)
        x = self.relu_ds2(x)

        # 5. Channel adjustment to output dimensions
        # No temporal change, only channel reduction: (B, 100, T) -> (B, output_channels, T)
        x = self.channel_adjust_before_parallel(x)
        x = self.bn_adjust(x)
        x = self.relu_adjust(x)

        # 6. Parallel dilated convolutions for feature refinement
        # No dimension changes: (B, output_channels, T) -> (B, output_channels, T)
        x = self.parallel_dilated_blocks(x)
        x = self.bn_parallel(x)
        x = self.relu_parallel(x)

        # 7. Final temporal compression
        # Reduces temporal by 5x: (B, output_channels, T) -> (B, output_channels, T/5)
        x = self.final_reduction_conv(x)
        x = self.bn_final(x)
        x = self.relu_final(x)

        # Return in (batch, time, channels) format
        #
        # OVERALL TEMPORAL COMPRESSION SUMMARY:
        # Total reduction factor = 4 × 2 × 2 × 5 = 80x compression
        # Input_T -> Output_T = Input_T/80
        #
        # Examples:
        # • Input T=3000 -> Output T ≈ 38
        # • Input T=30 -> Output T ≈ 0.375 → 1 (conv layers enforce minimum of 1)
        return x.permute(0, 2, 1)

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, indices=None):
        """
        Args:
            indices: Tensor of shape (batch_size, seq_len) containing position indices
        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        if indices is None:
            raise ValueError('indices must not be None')
        return self.pe[0, indices, :]




class Predictor(nn.Module):
    def __init__(self, preset):
        super().__init__()
        self.encoder_d_model = preset["nn"]["d_embedding"]
        self.predictor_d_model = preset["jepa"]["predictor_d_model"]
        self.max_total_tokens_in_view = preset["jepa"]["num_segments_total_view"]

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



class Transformer_Encoder(torch.nn.Module):
    """
    Enhanced Transformer Encoder module with masking support and standard positional encoding.
    Processes a sequence of tokens and applies self-attention.
    Uses learnable positional embeddings based on the token's original absolute index.
    Handles variable sequence lengths using padding masks.
    """

    def __init__(self, d_model, nhead, num_layers, max_total_tokens):
        super(Transformer_Encoder, self).__init__()

        self.d_model = d_model

        self.pos_encoder = FixedPositionalEncoding(d_model, max_total_tokens)

        # Custom transformer encoder layers
        self.layer_embedding_encoder = torch.nn.ModuleList([
            Encoder(var_dim_feature=d_model, var_num_head=nhead)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.layer_embedding_norm = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, token_original_indices=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            token_original_indices: Original indices for positional encoding
            src_key_padding_mask: Mask for padded positions (True for padded positions)
        """
        # Get positional embeddings based on original indices
        if token_original_indices is None:
            token_original_indices = torch.arange(
                src.shape[1], device=src.device, dtype=torch.long
            ).unsqueeze(0).expand(src.shape[0], -1)

        pos_embeddings = self.pos_encoder(token_original_indices)

        # Zero out positional embeddings for padded positions (if mask provided)
        if src_key_padding_mask is not None:
            pos_embeddings = pos_embeddings.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)

        # Add positional encodings
        var_embedding = src + pos_embeddings

        # Process through custom transformer encoder layers
        for layer in self.layer_embedding_encoder:
            var_embedding = var_embedding + layer(var_embedding, src_key_padding_mask)

        # Apply final layer norm
        var_embedding = self.layer_embedding_norm(var_embedding)

        return var_embedding


class Encoder(torch.nn.Module):
    """
    Enhanced custom encoder layer with masking support and improved FFN.
    """

    def __init__(self, var_dim_feature, var_num_head=10):
        super(Encoder, self).__init__()

        # First layer norm (pre-norm for attention)
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)

        # Multi-head attention
        self.layer_attention = torch.nn.MultiheadAttention(
            var_dim_feature, var_num_head, batch_first=True, dropout=0.1
        )

        # Dropout after attention
        self.layer_dropout_0 = torch.nn.Dropout(0.1)

        # Second layer norm (pre-norm for FFN)
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)

        # Enhanced FFN with 4x expansion (standard transformer practice)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(var_dim_feature, var_dim_feature * 4),  # Expand
            torch.nn.ReLU(),  # Standard activation
            torch.nn.Dropout(0.1),
            torch.nn.Linear(var_dim_feature * 4, var_dim_feature)  # Contract
        )

        # Dropout after FFN
        self.layer_dropout_1 = torch.nn.Dropout(0.1)

    def forward(self, var_input, src_key_padding_mask=None):
        """
        Args:
            var_input: Input tensor of shape (batch_size, seq_len, d_model)
            src_key_padding_mask: Mask for padded positions (True for padded positions)
        """
        # Self-attention block with pre-norm
        var_t = self.layer_norm_0(var_input)

        # Apply attention with masking support
        if src_key_padding_mask is not None:
            var_t, _ = self.layer_attention(
                var_t, var_t, var_t,
                key_padding_mask=src_key_padding_mask
            )
        else:
            var_t, _ = self.layer_attention(var_t, var_t, var_t)

        var_t = self.layer_dropout_0(var_t)

        # First residual connection
        var_t = var_t + var_input

        # FFN block with pre-norm
        var_s = self.layer_norm_1(var_t)
        var_s = self.ffn(var_s)
        var_s = self.layer_dropout_1(var_s)

        # Second residual connection
        var_output = var_s + var_t

        return var_output

# class Transformer_Encoder(nn.Module):
#     """
#     Transformer Encoder module.
#     Processes a sequence of tokens and applies self-attention.
#     Uses learnable positional embeddings based on the token's original absolute index.
#     Handles variable sequence lengths using padding masks.
#     """
#
#     def __init__(self, d_model, nhead, num_layers,
#                  max_total_tokens):  # d_model is feature_dim, max_total_tokens is the max possible original index + 1
#         super().__init__()
#         self.d_model = d_model
#         # Standard Transformer Encoder Layer.
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
#         # Stack of Transformer Encoder Layers.
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # Learnable positional embeddings for each possible original token position.
#         self.pos_encoder = nn.Embedding(max_total_tokens, d_model)
#
#     def forward(self, src, token_original_indices=None, src_key_padding_mask=None):
#         # Get positional embeddings based on original indices.
#         if token_original_indices is None:
#             token_original_indices = torch.arange(
#                 src.shape[1], device=src.device, dtype=torch.long
#             ).unsqueeze(0).expand(src.shape[0], -1)
#         pos_embeddings = self.pos_encoder(token_original_indices)
#
#         # Zero out positional embeddings for padded positions (if mask provided)
#         if src_key_padding_mask is not None:
#             pos_embeddings = pos_embeddings.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
#
#         src = src + pos_embeddings
#
#         if src_key_padding_mask is not None:
#             output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
#         else:
#             output = self.transformer_encoder(src)
#         return output


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
        self.memory_pos_encoding = None

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
        B, seq_len, _ = context_tokens.shape

        # Initialize decoder input with zero queries
        tgt = self.tgt_embed.unsqueeze(0).expand(B, -1, -1)

        # Get query positions
        query_pos = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        position_ids = torch.arange(seq_len, device=context_tokens.device).unsqueeze(0).expand(B, -1)
        memory_pos = self.memory_pos_encoding(position_ids)

        # Store intermediate outputs
        intermediate = []

        # Run through decoder layers
        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            output = layer(
                tgt=output,
                memory=context_tokens,
                query_pos=query_pos,
                memory_pos=memory_pos,
                key_padding_mask=src_key_padding_mask 
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
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False)

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

