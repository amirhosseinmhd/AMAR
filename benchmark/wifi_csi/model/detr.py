"""
[file]          detr.py
[description]   implement and evaluate WiFi-based model Transformer_Encoder
                https://github.com/windofshadow/THAT
"""
#
##
import os
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
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, 1e-6)
        #
        layer_cnn = []
        #
        for var_size in var_size_cnn:
            #
            layer = torch.nn.Sequential(torch.nn.Conv1d(var_dim_feature,
                                                        var_dim_feature,
                                                        var_size,
                                                        padding="same"),
                                        torch.nn.BatchNorm1d(var_dim_feature),
                                        torch.nn.Dropout(0.1),
                                        torch.nn.LeakyReLU())
            layer_cnn.append(layer)
        #
        self.layer_cnn = torch.nn.ModuleList(layer_cnn)
        #
        self.layer_dropout_1 = torch.nn.Dropout(0.1)

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

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim=0)
        #
        var_s = torch.sum(var_c, dim=0) / len(self.layer_cnn)
        #
        var_s = self.layer_dropout_1(var_s)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_output = var_s + var_t
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
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
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

# Non-Local Block for Global Context Modeling
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv1d(channels, channels // 2, kernel_size=1)
        self.phi = nn.Conv1d(channels, channels // 2, kernel_size=1)
        self.g = nn.Conv1d(channels, channels // 2, kernel_size=1)
        self.out_conv = nn.Conv1d(channels // 2, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, t = x.size()
        theta = self.theta(x).view(b, c // 2, -1)
        phi = self.phi(x).view(b, c // 2, -1)
        g = self.g(x).view(b, c // 2, -1)
        attn = self.softmax(torch.matmul(theta.transpose(1, 2), phi))
        out = torch.matmul(g, attn.transpose(1, 2))
        out = self.out_conv(out.view(b, c // 2, t))
        return x + out

# Backbone Network
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=270, output_channels=16, embedding_time_dim=100):
        super(CNNFeatureExtractor, self).__init__()
        self.embedding_time_dim = embedding_time_dim

        # Gradual channel reduction with efficient operations
        self.channel_reduction = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=1),
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128, kernel_size=7, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=3),  # Temp: 3000 -> 1000
            nn.Conv1d(128, output_channels, kernel_size=1),
            nn.ReLU(),
            # DepthwiseSeparableConv(output_channels, output_channels, kernel_size=5, padding=2),
            # nn.Conv1d(64, output_channels, kernel_size=1),
            # nn.ReLU()
            # DepthwiseSeparableConv(32, 32, kernel_size=3, padding=1),
            # nn.Conv1d(32, output_channels, kernel_size=1),
            # nn.ReLU(),
        )

        # Dilated convolution blocks maintain temporal resolution
        self.dilated_blocks = nn.Sequential(
            DilatedConvBlock(output_channels, output_channels, dilation_rate=1),
            DilatedConvBlock(output_channels, output_channels, dilation_rate=2),
            DilatedConvBlock(output_channels, output_channels, dilation_rate=4),
            DilatedConvBlock(output_channels, output_channels, dilation_rate=8),
        )

        # Final temporal reduction
        kernel_final = 1000 // embedding_time_dim  # 1000/100 = 10
        self.final_conv = nn.Conv1d(output_channels, output_channels,
                                  kernel_size=kernel_final, stride=kernel_final)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        x = self.channel_reduction(x)
        x = self.dilated_blocks(x)
        x = self.final_conv(x)
        return x.permute(0, 2, 1)  # (batch, time, channels)


class Transformer_Encoder(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_embedding_shape,
                 var_y_shape,
                 num_attention_heads=2,
                 num_transformer_encoder_layers=4):
        #
        ##
        super(Transformer_Encoder, self).__init__()
        #
        var_dim_feature = var_embedding_shape[-1]
        var_dim_time = var_embedding_shape[-2]
        var_dim_output = var_y_shape[-1]

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
                 temp_cross_attention=1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Create activity queries - learnable parameters
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))  # 10 object queries

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

        self.tgt_embed = nn.Parameter(torch.zeros(num_queries, d_model))

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

        # Get positional queries
        query_pos = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        # Store intermediate outputs
        intermediate = []

        # Run through decoder layers
        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            output = layer(
                tgt=output,
                memory=memory,
                query_pos=query_pos
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

    def forward(self, tgt, memory, query_pos=None):
        # Cross attention
        tgt2, self.cross_attn_weights = self.cross_attn(
            tgt + query_pos,
            memory,
            memory
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
            average_attn_weights=True # gets the average across all heads
        )

        # Apply temperature scaling to attention output
        attn_output = attn_output / self.temperature

        return attn_output, attn_weights


class DETR_MultiUser(nn.Module):
    def __init__(self, var_x_shape, var_y_shape, features_dim = 20, embedding_time_dim=100, num_decoder_layers=12,
                 temp_cross=1, n_attention_heads=2, num_queries=5, dim_feedforward=1024):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(input_channels=var_x_shape[-1], output_channels=features_dim,embedding_time_dim=embedding_time_dim)
        var_embedding_shape = (embedding_time_dim, features_dim)
        self.encoder = Transformer_Encoder(var_embedding_shape, var_y_shape, num_attention_heads=n_attention_heads,
                                           num_transformer_encoder_layers=4)
        self.decoder = TransformerDecoder(
            d_model=features_dim,  # Matches encoder output feature dimension
            nhead=n_attention_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross
        )

    def forward(self, x):
        # Extracting Features
        var_embedding = self.feature_extractor(x)


        memory = self.encoder(var_embedding)  # Shape: (B, 420, 270)

        # Pass through decoder to get predictions from all layers
        outputs_class = self.decoder(memory)  # Shape: [num_layers + 1, B, num_queries, num_classes]

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
            losses.append(loss.mean())
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
    ##
    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(data_test_x, data_test_y,
                                                                            test_size=0.5,
                                                                            shuffle=True,
                                                                            random_state=39)

    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape = data_train_x[0].shape, [data_train_y[0].shape[1]]
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    # data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    result_ppp = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []
    result_avg_count_error = []

    #
    var_macs, var_params = get_model_complexity_info(DETR_MultiUser(var_x_shape, var_y_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"]),
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
            project="experiment_classrom",
            name= name_run +preset["wandb_name"] ,
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_detr = DETR_MultiUser(var_x_shape, var_y_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"]).to(device)
        # wandb.watch(
        #     model_detr.feature_extractor,  # Directly target the CNN backbone
        #     log="all",  # Log gradients and parameters
        #     log_freq=50,  # Frequency of logging
        #     log_graph=True  # Optional: visualize computation graph
        # )

        if preset.get("pretrained_path"):
            model_detr, param_groups = load_model_components(
                model_detr,
                preset["pretrained_path"],
                preset["nn"]["lr"],
                preset.get("transfer_scenario"),
                device
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

        # data_test_y_c = data_test_y.sum(axis=1)
        dict_true_acc = performance_metrics(data_test_y, predict_test_y, var_mode=var_mode)
        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,
            "TOTAL_TESTSET_ERROR": dict_true_acc['total_error'],
            "TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc['perfect_prediction_percentage'],
            "TOTAL_ACCURACY": dict_true_acc['accuracy'],
            "mean_count_error": dict_true_acc['mean_count_error'],
            "error_per_person_1": dict_true_acc['error_per_person'][0],
            "error_per_person_2": dict_true_acc['error_per_person'][1],
            "error_per_person_3": dict_true_acc['error_per_person'][2],
            "error_per_person_4": dict_true_acc['error_per_person'][3],
            "error_per_person_5": dict_true_acc['error_per_person'][4],
            "precision": dict_true_acc['precision'],
            "recall": dict_true_acc['recall'],
            "f1_score": dict_true_acc['f1_score']
        })
        print(" %.6fs" % (time.time() - var_time_1),
              "- Total Error %.6f" % dict_true_acc['total_error'],
              "-  perfect_prediction_percentage %.6f" % dict_true_acc['perfect_prediction_percentage'],
              )
        #
        #

        #
        result_ppp.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_total_error.append(dict_true_acc['total_error'])
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])
        result_avg_count_error.append(dict_true_acc['mean_count_error'])

    wandb.log({
        "avg_accuracy": sum(result_ppp) / len(result_ppp),
        "avg_train_time": sum(result_time_train) / len(result_time_train),
        "avg_test_time": sum(result_time_test) / len(result_time_test),
        "avg_total_error": sum(result_total_error) / len(result_total_error),
        "avg_precision": sum(result_precision) / len(result_precision),
        "avg_recall": sum(result_recall) / len(result_recall),
        "avg_f1_score": sum(result_f1_score) / len(result_f1_score),
        "avg_count_error": sum(result_avg_count_error) / len(result_avg_count_error),
    })
    viz_stats = visualize_model_performance(
        y_pred=predict_test_y,
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
    return dict_true_acc

