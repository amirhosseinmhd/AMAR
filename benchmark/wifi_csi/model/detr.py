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
from train import train
from preset import preset
import torch.nn.functional as F
from utils import *
import wandb
from collections import Counter


def strat_train_test_split(data_x, data_y, test_size, shuffle, random_state, stratify_axis=1):
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
    # Sum along the specified axis to get a pattern for stratification.
    if data_y.ndim <= stratify_axis:
        raise ValueError(f"stratify_axis {stratify_axis} is out of bounds for data_y with {data_y.ndim} dimensions.")
    
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
        
        X_train, X_test_strat, y_train, y_test_strat = train_test_split(
            multi_x, multi_y, 
            test_size=1-train_size,
            shuffle=shuffle, 
            random_state=random_state,
            stratify=multi_patterns
        )
        
        # Add single-occurrence samples to test set
        X_test = np.vstack([X_test_strat, data_x[single_occurrence_indices]])
        y_test = np.vstack([y_test_strat, data_y[single_occurrence_indices]])
        
        return X_train, X_test, y_train, y_test
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
##
## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Gaussian Encoding ------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Gaussian_Position(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_dim_time,
                 var_num_gaussian=10):
        #
        ##
        super(Gaussian_Position, self).__init__()
        #
        ## var_embedding: shape (var_dim_k, var_dim_feature)
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype=torch.float)
        self.var_embedding = torch.nn.Parameter(var_embedding, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.var_embedding)
        #
        ## var_position: shape (var_dim_time, var_dim_k)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = torch.nn.Parameter(var_position, requires_grad=False)
        #
        ## var_mu: shape (1, var_dim_k)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time / var_num_gaussian).unsqueeze(0)
        self.var_mu = torch.nn.Parameter(var_mu, requires_grad=True)
        #
        ## var_sigma: shape (1, var_dim_k)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = torch.nn.Parameter(var_sigma, requires_grad=True)

    #
    ##
    def calculate_pdf(self,
                      var_position,
                      var_mu,
                      var_sigma):
        #
        ##
        var_pdf = var_position - var_mu  # (position-mu)
        #
        var_pdf = - var_pdf * var_pdf  # -(position-mu)^2
        #
        var_pdf = var_pdf / var_sigma / var_sigma / 2  # -(position-mu)^2 / (2*sigma^2)
        #
        var_pdf = var_pdf - torch.log(var_sigma)  # -(position-mu)^2 / (2*sigma^2) - log(sigma)
        #
        return var_pdf

    #
    ##
    def forward(self,
                var_input):
        var_pdf = self.calculate_pdf(self.var_position, self.var_mu, self.var_sigma)

        var_pdf = torch.softmax(var_pdf, dim=-1)
        #
        var_position_encoding = torch.matmul(var_pdf, self.var_embedding)
        #
        # print(var_input.shape, var_position_encoding.shape)
        var_output = var_input + var_position_encoding.unsqueeze(0)
        #
        return var_output


#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- Encoder ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##

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


class Encoder(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_num_head=10,
                 var_size_cnn=[1, 3, 5]):
        #
        ##
        super(Encoder, self).__init__()
        #
        ##
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
        self.layer_attention = torch.nn.MultiheadAttention(var_dim_feature,
                                                           var_num_head,
                                                           batch_first=True)
        #
        self.layer_dropout_0 = torch.nn.Dropout(0.1)
        #
        ##
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
        #
        # Replace CNN layers with a standard FFN√√Dropout
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(var_dim_feature, var_dim_feature),  # Expand dimension (typically 4x)
            torch.nn.LeakyReLU(),  # Standard activation in modern transformers
            torch.nn.Dropout(0.1),
            torch.nn.Linear(var_dim_feature, var_dim_feature)  # Project back to original dimension
        )
        #
        self.layer_dropout_1 = torch.nn.Dropout(0.1)
        # Add layer norm after FFN
        self.layer_norm_2 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = self.layer_norm_0(var_t)
        #
        var_t, _ = self.layer_attention(var_t, var_t, var_t)
        var_t = self.layer_dropout_0(var_t)
        #
        var_t = var_t + var_input
        #
        ##
        var_s = self.layer_norm_1(var_t)
        var_s = self.ffn(var_s)
        var_s = self.layer_dropout_1(var_s)
        var_output = var_s + var_t
        # Apply final layer norm
        # var_output = self.layer_norm_2(var_output)
        #
        return var_output

#
##
## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- Transformer_Encoder -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##

# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, stride=stride
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Dilated Convolution Block
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Dilated_Blocks(nn.Module):
    def __init__(self, output_channels):
        super().__init__()

        # Parallel dilated convolutions instead of sequential blocks
        self.dilated_conv1 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=1, padding='same')
        self.dilated_conv2 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=2, padding='same')
        self.dilated_conv4 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=4, padding='same')
        self.dilated_conv8 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=8, padding='same')

        # 1x1 convolution to combine features (if needed)
        self.combine_conv = nn.Conv1d(output_channels, output_channels, kernel_size=1)

        self.relu = nn.ReLU()
    def forward(self, x):
        # Parallel dilated convolutions
        out1 = self.relu(self.dilated_conv1(x))  # Shape: [batch, output_channels//4, 25]
        out2 = self.relu(self.dilated_conv2(x))  # Shape: [batch, output_channels//4, 25]
        out4 = self.relu(self.dilated_conv4(x))  # Shape: [batch, output_channels//4, 25]
        out8 = self.relu(self.dilated_conv8(x))  # Shape: [batch, output_channels//4, 25]
        # Concatenate along channel dimension
        out_concat = torch.cat([out1, out2, out4, out8], dim=1)  # Shape: [batch, output_channels, 25]

        return out_concat
        #
# Channel Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


# Backbone Network
class CNNFeatureExtractor(nn.Module):
## The goal is to make this feature extractor from this spatial mapping : 3000->T_1=100 and 270->C_1 to 500-> T_2=10, and 270->C_2
## The change here is that the model will be instantiated 6 times to match the original output.
## The model will start at randomly intiallied point in the from time stamp 0 to 250, and then applies -> This is next level of the model
    def __init__(self, input_channels=270, output_channels=16, embedding_time_dim=50):
        super(CNNFeatureExtractor, self).__init__()
        self.embedding_time_dim = embedding_time_dim
        #T = 500
        # Gradual channel reduction with efficient operations
        self.channel_reduction = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=1),
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128, kernel_size=7, padding=3, stride=2), #500 ->250
            # nn.MaxPool1d(kernel_size=3, stride=3),  # Temp: 3000 -> 1000
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            DepthwiseSeparableConv(64, 64, kernel_size=5, padding=2, stride=2),#250 ->125
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            DepthwiseSeparableConv(32, output_channels, kernel_size=5, padding=1, stride=5), # 25 tokens
            # nn.Conv1d(32, output_channels, kernel_size=1),
            # nn.ReLU(),
        )
        self.dilated_blocks = Dilated_Blocks(output_channels)
        # Dilated convolution blocks maintain temporal resolution
        # self.dilated_blocks = nn.Sequential(
        #     DilatedConvBlock(output_channels, output_channels, dilation_rate=1),
        #     DilatedConvBlock(output_channels, output_channels, dilation_rate=2),
        #     DilatedConvBlock(output_channels, output_channels, dilation_rate=4),
        #     DilatedConvBlock(output_channels, output_channels, dilation_rate=8),
        # )

        # # Final temporal reduction
        # kernel_final = 1000 // embedding_time_dim  # 1000/100 = 10
        # self.final_conv = nn.Conv1d(output_channels, output_channels,
        #                           kernel_size=kernel_final, stride=kernel_final)

    def forward(self, x):

        x = x.permute(0, 2, 1)  # (batch, channels, time)
        x = self.channel_reduction(x)
        x = self.dilated_blocks(x)
        # x = self.final_conv(x)
        return x.permute(0, 2, 1)  # (batch, time, channels)


class Transformer_Encoder(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_embedding_shape,
                 num_attention_heads=2,
                 num_transformer_encoder_layers=4):
        #
        ##
        super(Transformer_Encoder, self).__init__()
        #
        var_dim_feature = var_embedding_shape[-1]
        var_dim_time = var_embedding_shape[-2]

        self.layer_embedding_gaussian = Gaussian_Position(var_dim_feature, var_dim_time)  # 100 tokens for left stream



        self.layer_embedding_encoder = torch.nn.ModuleList([Encoder(var_dim_feature=var_dim_feature,
                                                               var_num_head=num_attention_heads,
                                                               var_size_cnn=[1])
                                                       for _ in range(num_transformer_encoder_layers)])
        #
        self.layer_embedding_norm = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)


    #
    ##
    def forward(self, var_embedding):
        # Apply Gaussian position encoding
        var_embedding = self.layer_embedding_gaussian(var_embedding)  # Output: (batch_size, 100, features)

        # Process left stream through transformer encoders
        for layer in self.layer_embedding_encoder:
            var_embedding = var_embedding + layer(var_embedding)
        var_embedding = self.layer_embedding_norm(var_embedding)


        return var_embedding

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
    def __init__(self, var_x_shape, features_dim = 20, embedding_time_dim=100, num_decoder_layers=12,
                 temp_cross=1, n_attention_heads=2, num_queries=5, dim_feedforward=1024, query_dropout_rate=0.0):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(input_channels=var_x_shape[-1], output_channels=features_dim,embedding_time_dim=embedding_time_dim)
        var_embedding_shape = (embedding_time_dim, features_dim)
        self.encoder = Transformer_Encoder(var_embedding_shape, num_attention_heads=n_attention_heads,
                                           num_transformer_encoder_layers=4)
        self.decoder = TransformerDecoder(
            d_model=features_dim,
            nhead=n_attention_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross, 
            seq_length=embedding_time_dim,
            query_dropout_rate=query_dropout_rate 
        )

    def forward(self, x):
        # Extracting Features
        var_embedding = torch.concat([ self.feature_extractor(x[:, i*500:(i+1)*500, :]) for i in range(0, 6)]    , dim=1)
        # var_embedding = self.feature_extractor(x)


        memory = self.encoder(var_embedding)

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
    data_valid_x, new_data_test_x, data_valid_y, new_data_test_y = strat_train_test_split(
        data_x=data_test_x,
        data_y=data_test_y,
        test_size=0.5,
        shuffle=True,
        random_state=39,
        stratify_axis=1  # Sum along the 'num_activities' axis
    )
    data_test_x = new_data_test_x
    data_test_y = new_data_test_y

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

    #
    ##
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
    var_macs, var_params = get_model_complexity_info(DETR_MultiUser(var_x_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"],
                                    query_dropout_rate=preset["nn"]["query_dropout_rate"]),
                                                     var_x_shape, as_strings=False)

    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #

    for var_r in range(var_repeat):
        #
        ##
        var_mode = "multi_head"
        name_run = "Empty"
        if preset["pretrained_path"]:
            name_run = f"DETR_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + preset["transfer_scenario"]
        else:
            pretrained_state = "NPT"
            name_run = f"DETR_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + pretrained_state 
        print("Repeat", var_r)
        run = wandb.init(
            project="test",
            name= name_run +preset["wandb_name"] ,
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_detr = DETR_MultiUser(var_x_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"],
                                    query_dropout_rate=preset["nn"]["query_dropout_rate"]).to(device) 
        # wandb.watch(
        #     model_detr.feature_extractor,  # Directly target the CNN backbone
        #     log="all",  # Log gradients and parameters
        #     log_freq=50,  # Frequency of logging
        #     log_graph=True  # Optional: visualize computation graph
        # )

        if preset.get("pretrained_path"):
            model_detr, param_groups = load_model_components(
                model=model_detr,
                load_path=preset["pretrained_path"],
                lr = preset["nn"]["lr"],
                scenario=preset.get("transfer_scenario"),
                device=device
            )
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(model_detr.parameters(),
                                         lr=preset["nn"]["lr"],
                                         weight_decay=preset["nn"]["weight_decay"])

        loss = HungarianMatchingLoss(
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
                                loss=loss,
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

        layers_idxs = preset["layers_idxs"]

        # Store results for each layer
        all_layers_results = {}
        dict_layer_acc = performance_metrics(data_test_y, predict_test_y, var_mode=var_mode)

        # Process each layer separately
        for idx, layer_idx in enumerate(layers_idxs):

            layer_metrics = dict_layer_acc[layer_idx]

            # Log metrics for this layer
# In the test results section of run_that_detr
            wandb.log({
                f"test_results/{layer_idx}/repeat": var_r,
                f"test_results/{layer_idx}/train_time": var_time_1 - var_time_0,
                f"test_results/{layer_idx}/test_time": var_time_2 - var_time_1,
                f"test_results/{layer_idx}/TOTAL_TESTSET_ERROR": layer_metrics['total_error'],
                f"test_results/{layer_idx}/TOTAL_TESTSET_perfect_prediction_percentage": layer_metrics[
                    'perfect_prediction_percentage'],
                f"test_results/{layer_idx}/TOTAL_ACCURACY": layer_metrics['accuracy'],
                f"test_results/{layer_idx}/mean_count_error": layer_metrics['mean_count_error'],
                f"test_results/{layer_idx}/error_per_person_1": layer_metrics['error_per_person'][0],
                f"test_results/{layer_idx}/error_per_person_2": layer_metrics['error_per_person'][1],
                f"test_results/{layer_idx}/error_per_person_3": layer_metrics['error_per_person'][2],
                f"test_results/{layer_idx}/error_per_person_4": layer_metrics['error_per_person'][3],
                f"test_results/{layer_idx}/error_per_person_5": layer_metrics['error_per_person'][4],
                f"test_results/{layer_idx}/precision": layer_metrics['precision'],
                f"test_results/{layer_idx}/recall": layer_metrics['recall'],
                f"test_results/{layer_idx}/f1_score": layer_metrics['f1_score']
            }, step=var_r+100000)  # Use a large offset + repeat index as step

            print(f"Layer {layer_idx} - %.6fs" % (var_time_2 - var_time_1),
                  "- Total Error %.6f" % layer_metrics['total_error'],
                  "- Perfect Prediction Percentage %.6f" % layer_metrics['perfect_prediction_percentage'])

            # Store the results for each layer
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

            # Append results for this layer
            result_accuracy[idx].append(layer_metrics['accuracy'])
            result_ppp[idx].append(layer_metrics['perfect_prediction_percentage'])
            result_time_train[idx].append(var_time_1 - var_time_0)
            result_time_test[idx].append(var_time_2 - var_time_1)
            result_total_error[idx].append(layer_metrics['total_error'])
            result_precision[idx].append(layer_metrics['precision'])
            result_recall[idx].append(layer_metrics['recall'])
            result_f1_score[idx].append(layer_metrics['f1_score'])
            result_avg_count_error[idx].append(layer_metrics['mean_count_error'])

    # Log average metrics across all layers and repeats
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

