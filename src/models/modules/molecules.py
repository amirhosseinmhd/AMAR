import math
from torch import nn
import torch
from src.models.modules.elements import DepthwiseSeparableConv, DilatedConvBlock
from configs.preset import preset
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):

        inputs_contiguous = inputs.contiguous()
        input_shape = inputs_contiguous.shape
        
        # Flatten input
        flat_input = inputs_contiguous.view(-1, self.embedding_dim)
        
        # Here we want to calculate the distance between flat_input and embeddings, we use
        # : ||a-b||² = ||a||² + ||b||² - 2a·b
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Now we find out which of the quantized inputs are closes to what we have there.
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # creating a zeros matrix, then put 1 where at indexes in in encoding_indicies
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


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ) implementation.
    Uses multiple quantization layers to quantize residuals progressively.
    """
    def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost, 
                 quantization_dropout=0.2):
        super(ResidualVectorQuantizer, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.quantization_dropout = quantization_dropout
        
        # Create multiple VQ layers
        self.vq_layers = nn.ModuleList()
        
        # All layers with random initialization
        for _ in range(num_layers):
            vq_layer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
            self.vq_layers.append(vq_layer)
    
    def _get_num_active_layers(self):
        """
        Determines the number of active quantization layers during training.
        Uses quantization dropout to randomly disable the last 0 to V layers.
        
        Strategy: With probability q, randomly choose to use 1 to num_layers layers.
        This forces early layers to be more robust and prevents over-dependence on later layers.
        
        Returns:
            int: Number of active layers (always at least 1)
        """
        if not self.training or self.quantization_dropout == 0.0:
            return self.num_layers
        
        # With probability q, apply quantization dropout
        if torch.rand(1).item() < self.quantization_dropout:
            # Randomly select number of layers to keep (1 to num_layers)
            # This implements the "disable last 0 to V layers" strategy
            num_layers_to_drop = torch.randint(0, self.num_layers, (1,)).item()
            num_active = self.num_layers - num_layers_to_drop
            # Ensure we always keep at least 1 layer
            num_active = max(1, num_active)
        else:
            # Use all layers
            num_active = self.num_layers
            
        return num_active
    
    def forward(self, inputs):
        """
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, embedding_dim)
        
        Returns:
            quantized_st: Straight-through quantized features (sum of active layers)
            quantized: Detached quantized features (sum of active layers)
            all_indices: List of indices from each active quantization layer
            all_quantized_layers: List of quantized outputs from each active layer
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Determine number of active layers (with quantization dropout during training)
        num_active_layers = self._get_num_active_layers()
        
        # Initialize residual with input features
        residual = inputs
        
        # Store outputs from each active layer
        all_quantized_st = []
        all_quantized = []
        all_indices = []
        
        # Progressive residual quantization (only through active layers)
        for layer_idx in range(num_active_layers):
            vq_layer = self.vq_layers[layer_idx]
            
            # Quantize current residual
            quantized_st, quantized, indices = vq_layer(residual)
            
            # Store layer outputs
            all_quantized_st.append(quantized_st)
            all_quantized.append(quantized)
            all_indices.append(indices)
            
            # Update residual for next layer (subtract quantized representation)
            if layer_idx < num_active_layers - 1:  # Don't compute residual for last active layer
                residual = residual - quantized.detach()
        
        # Sum all quantized representations from active layers
        final_quantized_st = torch.stack(all_quantized_st, dim=0).sum(dim=0)
        final_quantized = torch.stack(all_quantized, dim=0).sum(dim=0)
        
        return final_quantized_st, final_quantized, all_indices, all_quantized
    
    def compute_loss(self, inputs, all_quantized_layers):
        """
        Compute total VQ loss across all active layers.
        
        Args:
            inputs: Original input features
            all_quantized_layers: List of quantized outputs from each active layer
        
        Returns:
            total_loss: Sum of losses from all active quantization layers
        """
        total_loss = 0.0
        residual = inputs
        num_active_layers = len(all_quantized_layers)
        
        for layer_idx in range(num_active_layers):
            vq_layer = self.vq_layers[layer_idx]
            quantized = all_quantized_layers[layer_idx]
            
            # Compute loss for current layer
            layer_loss = vq_layer.compute_loss(residual, quantized)
            total_loss += layer_loss
            
            # Update residual for next layer
            if layer_idx < num_active_layers - 1:
                residual = residual - quantized.detach()
        
        return total_loss

    def set_quantization_dropout(self, dropout_rate):
        """
        Set the quantization dropout rate.
        
        Args:
            dropout_rate (float): Dropout probability in [0, 1]
        """
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1], got {dropout_rate}")
        self.quantization_dropout = dropout_rate
    
    def get_quantization_dropout(self):
        """Get the current quantization dropout rate."""
        return self.quantization_dropout

class Backbone(nn.Module):
    def __init__(self, input_channels=270, output_channels=16):
        super(Backbone, self).__init__()
        pca_output_dim = 180
        c_initial = 64  # Channels after initial convolution
        c_hierarchical_out = 64  # Channels after hierarchical dilated blocks

        # Alternative to PCA: depthwise separable convolution for dimensionality reduction
        # First: reduce channels 270 -> 180 using depthwise separable conv
        self.ds_conv_pca_replace = DepthwiseSeparableConv(
            input_channels,
            pca_output_dim,
            kernel_size=3,
            padding=1,
            stride=1
        )

        # 1. Depthwise separable conv with major temporal downsampling
        # Temporal reduction: T -> T/4, Channels: 180 -> 64
        self.ds_conv_temporal_reduce = DepthwiseSeparableConv(
            pca_output_dim,
            c_initial,
            kernel_size=5,
            padding=2,
            stride=4
        )

        # 2. First depthwise separable convolution with temporal downsampling
        # Temporal reduction: T -> T/2, Channels: 64 -> 64
        self.ds_conv1 = DepthwiseSeparableConv(c_initial, c_initial, kernel_size=5, padding=2, stride=2)

        # 3. Hierarchical Dilated Convolutions (no temporal reduction)
        self.hierarchical_dilated_1 = DilatedConvBlock(c_initial, c_initial, dilation_rate=1)
        self.hierarchical_dilated_2 = DilatedConvBlock(c_initial, c_initial, dilation_rate=2)
        self.hierarchical_dilated_3 = DilatedConvBlock(c_initial, c_hierarchical_out, dilation_rate=4)

        # 4. Second temporal reduction with depthwise separable convolution
        # Temporal reduction: T -> T/6 (stride=2*3=6 total), Channels: 128 -> 16
        self.ds_conv2 = DepthwiseSeparableConv(
            c_hierarchical_out,
            output_channels,
            kernel_size=5,
            padding=2,
            stride=2  # Combined stride for more compression
        )


    def forward(self, x):
        # Apply PCA replacement
        x = x.permute(0, 2, 1)  # Conv1d format: (B, 270, T)
        x = self.ds_conv_pca_replace(x)

        # Temporal reduction
        x = self.ds_conv_temporal_reduce(x)

        # First conv with temporal downsampling
        x = self.ds_conv1(x)

        # Hierarchical dilated convolutions
        x = self.hierarchical_dilated_1(x)
        x = self.hierarchical_dilated_2(x)
        x = self.hierarchical_dilated_3(x)

        # Second temporal reduction (now includes final compression)
        x = self.ds_conv2(x)

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




class Encoder(torch.nn.Module):
    """
    Enhanced custom encoder layer with masking support and improved FFN.
    Used for THAT model
    """

    def __init__(self, var_dim_feature, var_num_head=10):
        super(Encoder, self).__init__()

        # First layer norm (pre-norm for attention)
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)

        # Multi-head attention
        self.layer_attention = torch.nn.MultiheadAttention(
            var_dim_feature, var_num_head, batch_first=True, dropout=0.2
        )

        # Dropout after attention
        self.layer_dropout_0 = torch.nn.Dropout(0.2)

        # Second layer norm (pre-norm for FFN)
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)

        # Enhanced FFN with 4x expansion (standard transformer practice)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(var_dim_feature, var_dim_feature),  # Expand
            torch.nn.ReLU(),  # Standard activation
            torch.nn.Dropout(0.2),
            torch.nn.Linear(var_dim_feature, var_dim_feature)  # Contract
        )

        # Dropout after FFN
        self.layer_dropout_1 = torch.nn.Dropout(0.2)

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

