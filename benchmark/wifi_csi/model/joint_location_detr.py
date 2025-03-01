"""
[file]          detr.py
[description]   implement and evaluate WiFi-based model THAT_ENCODER
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
from torch.utils.data import Dataset
import torch.nn as nn

from ptflops import get_model_complexity_info
from itertools import permutations
from sklearn.metrics import classification_report, accuracy_score
from scipy.optimize import linear_sum_assignment
from train_joint import train
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
                 temp_cross_attention=1, num_locations=2):
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

        """
        Num locations can be changed to 2 or 5. If the num locaitons be 2, then we are solving the problem as regression probelm
        Regression makes more sense but harder problem to solve. The classification task is easier yet less informative. Locations might be close together and we would like the model understand the
        structure of the environment.
        
        """
        self.location_embed = nn.Linear(d_model, num_locations)
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
        intermediate_act = []
        intermediate_loc = []
        # Run through decoder layers
        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            output = layer(
                tgt=output,
                memory=memory,
                query_pos=query_pos
            )

            pred_activity = self.class_embed(output)
            pred_location = self.location_embed(output)
            intermediate_loc.append(pred_location)
            intermediate_act.append(pred_activity)

        out_ = (torch.stack(intermediate_act), torch.stack(intermediate_loc))
        # Shape activity: [num_layers, B, num_queries, num_classes]
        # Shape location: [num_layers, B, num_queries, locations]

        return  out_


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


class DETR_MultiUser_JointActivity(nn.Module):
    def __init__(self, var_x_shape, features_dim = 20, embedding_time_dim=100, num_decoder_layers=12,
                 temp_cross=1, n_attention_heads=2, num_queries=5, dim_feedforward=1024, output_dim = 5):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(input_channels=var_x_shape[-1], output_channels=features_dim,embedding_time_dim=embedding_time_dim)
        var_embedding_shape = (embedding_time_dim, features_dim)
        self.encoder = Transformer_Encoder(var_embedding_shape, num_attention_heads=n_attention_heads,
                                           num_transformer_encoder_layers=4)
        self.decoder = TransformerDecoder(
            d_model=features_dim,  # Matches encoder output feature dimension
            nhead=n_attention_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross,
            num_locations=output_dim
        )

    def forward(self, x):
        # Extracting Features
        var_embedding = self.feature_extractor(x)
        memory = self.encoder(var_embedding)  # Shape: (B, 420, 270)

        # Pass through decoder to get predictions from all layers
        outputs_class_all = self.decoder(memory)  # We expect a tuple here,  Shape: ( [num_layers, B, num_queries, num_classes], shape_locations)

        return outputs_class_all


class HungarianMatchingLoss(nn.Module):
    def __init__(self, cost_class_weight, alfa, aux_loss_weight, label_smoothing, class_imbalance_weight):
        super().__init__()
        # self.cost_class = cost_class_weight
        self.aux_loss_weight = aux_loss_weight

        weights = torch.ones(10)
        weights[-1] = class_imbalance_weight
        weights = weights * (len(weights) / weights.sum())

        self.ce_loss = nn.CrossEntropyLoss(
            weight=weights.to(torch.device('cuda')),
            label_smoothing=label_smoothing
        )

        self.ce_loss_loc = nn.CrossEntropyLoss()
        self.MSE_Loss = nn.MSELoss()
        self.alfa = alfa

    @torch.no_grad()
    def Hungarian_matching(self, act_pred, act_target, loc_pred, loc_target):
        """
        Performs the matching between predictions and ground truth
        Args:
            act_pred: Tensor of shape (batch_size, num_queries, num_classes)
            act_target: Tensor of shape (batch_size, num_queries, num_classes)
            loc_pred: Tensor of shape (batch_size, num_queries, 2) - x,y coordinates
            loc_target: Tensor of shape (batch_size, num_queries, 2) - x,y coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = act_pred.shape[:2]

        # Compute classification cost matrix
        out_prob = act_pred.softmax(-1)  # [batch_size, num_queries, num_classes]
        tgt_ids = act_target.argmax(-1)  # [batch_size, num_queries]
        flag_classification = 1 if loc_pred[0, 0].shape[0] > 2 else 0
        loc_prob = loc_pred.softmax(-1)  # [batch_size, num_queries, num_classes]
        loc_tgt_ids = loc_target.argmax(-1)  # [batch_size, num_queries]
        indices = []
        # Process each batch independently
        for b in range(bs):
            # Compute activity cost matrix for current batch
            # Create cost matrix showing how well each prediction matches each target
            activity_cost = -out_prob[b][:, tgt_ids[b]]

            if flag_classification:
                location_cost = torch.zeros((num_queries, num_queries), device=act_pred.device)
                for i in range(num_queries):
                    for j in range(num_queries):
                        flag_valid_location = loc_target[b, j].sum()
                        if not flag_valid_location:
                            location_cost[i, j] = 0
                        else:
                            # Use negative probability for the correct class as the cost
                            location_cost[i, j] = -loc_prob[b, i, loc_tgt_ids[b, j]]

            else:
                # Compute location cost matrix using MSE
                location_cost = torch.zeros((num_queries, num_queries), device=act_pred.device)
                for i in range(num_queries):
                    for j in range(num_queries):
                        flag_valid_location = loc_target[b, j].sum()
                        if not flag_valid_location:
                            location_cost[i, j] = 0
                        else:

                        # Calculate MSE between predicted and target locations

                            location_cost[i, j] = torch.sum((loc_pred[b, i] - loc_target[b, j]) ** 2)

            # Normalize location costs to be in similar scale as activity costs
            if location_cost.max() > 0:
                location_cost = location_cost / location_cost.max()

            # Combine costs using alpha weighting
            combined_cost = (1 - self.alfa) * activity_cost + self.alfa * location_cost

            combined_cost = combined_cost.cpu().numpy()


            # Run Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(combined_cost)

            # Convert to tensors and move to correct device
            row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=act_pred.device)  # Which queries to use
            col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=act_pred.device)  # Which targets they match to
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


    def _get_layer_loss(self, pred_activity, target_activity, pred_location, target_location, indices):
        """Helper to compute loss for a single layer's predictions"""
        losses = []
        flag_classification = 1 if pred_location[0, 0].shape[0] > 2 else 0

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            pred_i_act = pred_activity[batch_idx][pred_idx]
            tgt_i_act = target_activity[batch_idx][tgt_idx]

            pred_i_loc = pred_location[batch_idx][pred_idx]
            tgt_i_loc = target_location[batch_idx][tgt_idx]

            loss_activity = self.ce_loss(pred_i_act, tgt_i_act.argmax(-1))
            ######
            # Check if location is valid (non-zero)
            valid_locations = []
            for i, target_loc in enumerate(tgt_i_loc):
                # If location sum is zero, mask it out
                if target_loc.sum() == 0:
                    valid_locations.append(False)
                else:
                    valid_locations.append(True)

            valid_locations = torch.tensor(valid_locations, device=pred_i_loc.device)

            # If there are valid locations, calculate location loss
            if valid_locations.any():
                # Extract only valid locations for loss calculation
                valid_pred_loc = pred_i_loc[valid_locations]
                valid_tgt_loc = tgt_i_loc[valid_locations]

                # Calculate location loss only for valid locations

                if flag_classification:
                    loss_location = self.ce_loss_loc(valid_pred_loc, valid_tgt_loc)
                else:
                    loss_location = self.MSE_Loss(valid_pred_loc, valid_tgt_loc)
            else:
                # No valid locations, set location loss to zero
                loss_location = torch.tensor(0.0, device=loss_activity.device)
            ########
            joint_loss = ( 1 - self.alfa) * loss_activity + self.alfa * loss_location

            losses.append(joint_loss)
        return torch.stack(losses).mean()

    def forward(self, act_pred, act_target, loc_pred, loc_target):
        """
        Args:
            outputs: If auxiliary losses enabled: Tensor of shape [num_layers + 1, B, num_queries, num_classes]
                    Otherwise: Tensor of shape [B, num_queries, num_classes]
            targets: Tensor of shape [B, num_queries, num_classes]
        """
        # Check if we have auxiliary outputs
        if act_pred.dim() == 4:  # Has auxiliary outputs [num_layers + 1, B, num_queries, num_classes]
            # Split predictions from different decoder layers
            aux_act_pred = act_pred[:-1]  # Predictions from intermediate layers
            act_pred_final = act_pred[-1]  # Predictions from final layer
            aux_loc_pred = loc_pred[:-1]  # Predictions from intermediate layers
            loc_pred_final = loc_pred[-1]  # Predictions from final layer
            indices = self.Hungarian_matching(act_pred_final, act_target, loc_pred_final, loc_target)

            # Calculate loss for final predictions using final layer matching
            final_loss = self._get_layer_loss( act_pred_final, act_target, loc_pred_final, loc_target, indices)

            aux_losses = []
            for layer_idx, (layer_act_pred, layer_loc_pred) in enumerate(zip(aux_act_pred, aux_loc_pred)):
                layer_loss = self._get_layer_loss(layer_act_pred, act_target, layer_loc_pred, loc_target, indices)
                aux_losses.append(layer_loss)

                # Combine all losses with weights
                aux_loss = torch.stack(aux_losses).mean()
                total_loss = final_loss + self.aux_loss_weight * aux_loss
            return total_loss
        else:
            raise ValueError("Dim Loss not expected")



class JointActLocDataset(Dataset):
    def __init__(self, CSI, y_act, y_loc):
        self.X = CSI
        self.y_loc = y_loc
        self.y_act = y_act

    def __len__(self)  -> int :
        return self.X.shape[0]
    def __getitem__(self, idx) -> tuple:
        return self.X[idx], self.y_act[idx], self.y_loc[idx]


def run_joint_detr(data_train_x,
                    data_train_y_loc,
                    data_train_y_act,
                    data_test_x,
                    data_test_y_loc,
                    data_test_y_act,
                     var_repeat=10):
    """
    [description]
    : run WiFi-based model THAT_ENCODER_DECODER
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
    (data_valid_x, data_test_x,
     data_valid_y_loc, data_test_y_loc,
     data_valid_y_act, data_test_y_act) = my_train_test_split(data_test_x, data_test_y_loc, data_test_y_act, test_size=0.5, random_state=103)


    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape_loc, var_y_shape_act = data_train_x[0].shape, data_train_y_loc.shape[1:], data_train_y_act.shape[1:]
    #
    data_train_set = JointActLocDataset(data_train_x, data_train_y_act, data_train_y_loc)

    data_valid_set = JointActLocDataset(data_valid_x, data_valid_y_act, data_valid_y_loc)

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    result_ppp_act = []
    result_total_error_act = []
    result_precision_act = []
    result_recall_act = []
    result_f1_score_act = []
    result_avg_count_error_act = []

    # Store location results
    result_ppp_loc = []
    result_total_error_loc = []
    result_precision_loc = []
    result_recall_loc = []
    result_f1_score_loc = []
    result_avg_count_error_loc = []

    # Store timing results
    result_time_train = []
    result_time_test = []

    #
    var_macs, var_params = get_model_complexity_info(DETR_MultiUser_JointActivity(var_x_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"],
                                    output_dim=var_y_shape_loc[-1]),
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
            project="JointDETR",
            name= name_run + "very_small",
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        ################ CHANGE HERE,
        model_detr = DETR_MultiUser_JointActivity(var_x_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"],
                                                  output_dim= var_y_shape_loc[-1]).to(device)
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
            alfa=0.1,
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
            predict_test_y_act, predict_test_y_loc = model_detr(torch.from_numpy(data_test_x).to(device))
        #
        # predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()
        predict_test_act = predict_test_y_act.detach().cpu().numpy()
        predict_test_loc = predict_test_y_loc.detach().cpu().numpy()

        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        dict_true_acc_act, dict_true_acc_loc = performance_metrics_joint( data_test_y_act, predict_test_act, data_test_y_loc, predict_test_loc)

        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,

            # Activity metrics
            "ACT_TOTAL_TESTSET_ERROR": dict_true_acc_act['total_error'],
            "ACT_TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc_act['perfect_prediction_percentage'],
            "ACT_TOTAL_ACCURACY": dict_true_acc_act['accuracy'],
            "ACT_mean_count_error": dict_true_acc_act['mean_count_error'],
            "ACT_error_per_person_1": dict_true_acc_act['error_per_person'][0],
            "ACT_error_per_person_2": dict_true_acc_act['error_per_person'][1],
            "ACT_error_per_person_3": dict_true_acc_act['error_per_person'][2],
            "ACT_error_per_person_4": dict_true_acc_act['error_per_person'][3],
            "ACT_error_per_person_5": dict_true_acc_act['error_per_person'][4],
            "ACT_precision": dict_true_acc_act['precision'],
            "ACT_recall": dict_true_acc_act['recall'],
            "ACT_f1_score": dict_true_acc_act['f1_score'],

            # Location metrics
            "LOC_TOTAL_TESTSET_ERROR": dict_true_acc_loc['total_error'],
            "LOC_TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc_loc['perfect_prediction_percentage'],
            "LOC_TOTAL_ACCURACY": dict_true_acc_loc['accuracy'],
            "LOC_mean_count_error": dict_true_acc_loc['mean_count_error'],
            "LOC_error_per_person_1": dict_true_acc_loc['error_per_person'][0],
            "LOC_error_per_person_2": dict_true_acc_loc['error_per_person'][1],
            "LOC_error_per_person_3": dict_true_acc_loc['error_per_person'][2],
            "LOC_error_per_person_4": dict_true_acc_loc['error_per_person'][3],
            "LOC_error_per_person_5": dict_true_acc_loc['error_per_person'][4],
            "LOC_precision": dict_true_acc_loc['precision'],
            "LOC_recall": dict_true_acc_loc['recall'],
            "LOC_f1_score": dict_true_acc_loc['f1_score']
        })
        #
        #

        #
        result_ppp_act.append(dict_true_acc_act['perfect_prediction_percentage'])
        result_total_error_act.append(dict_true_acc_act['total_error'])
        result_precision_act.append(dict_true_acc_act['precision'])
        result_recall_act.append(dict_true_acc_act['recall'])
        result_f1_score_act.append(dict_true_acc_act['f1_score'])
        result_avg_count_error_act.append(dict_true_acc_act['mean_count_error'])

        result_ppp_loc.append(dict_true_acc_loc['perfect_prediction_percentage'])
        result_total_error_loc.append(dict_true_acc_loc['total_error'])
        result_precision_loc.append(dict_true_acc_loc['precision'])
        result_recall_loc.append(dict_true_acc_loc['recall'])
        result_f1_score_loc.append(dict_true_acc_loc['f1_score'])
        result_avg_count_error_loc.append(dict_true_acc_loc['mean_count_error'])

    wandb.log({
        # Activity averages
        "ACT_avg_accuracy": sum(result_ppp_act) / len(result_ppp_act),
        "ACT_avg_total_error": sum(result_total_error_act) / len(result_total_error_act),
        "ACT_avg_precision": sum(result_precision_act) / len(result_precision_act),
        "ACT_avg_recall": sum(result_recall_act) / len(result_recall_act),
        "ACT_avg_f1_score": sum(result_f1_score_act) / len(result_f1_score_act),
        "ACT_avg_count_error": sum(result_avg_count_error_act) / len(result_avg_count_error_act),

        # Location averages
        "LOC_avg_accuracy": sum(result_ppp_loc) / len(result_ppp_loc),
        "LOC_avg_total_error": sum(result_total_error_loc) / len(result_total_error_loc),
        "LOC_avg_precision": sum(result_precision_loc) / len(result_precision_loc),
        "LOC_avg_recall": sum(result_recall_loc) / len(result_recall_loc),
        "LOC_avg_f1_score": sum(result_f1_score_loc) / len(result_f1_score_loc),
        "LOC_avg_count_error": sum(result_avg_count_error_loc) / len(result_avg_count_error_loc),
    })

    # viz_stats = visualize_model_performance(
    #     y_pred=predict_test_y,
    #     y_true=data_test_y_act,
    #     var_mode=var_mode,
    #     save_dir=f'./visualizations/experiment_{var_r}_{var_mode}'
    # )
    # print("\nDetailed Performance Analysis:")
    # print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    # print("\nClass-wise Mean Absolute Error:")
    # for i, error in enumerate(viz_stats['class_wise_mae']):
    #     print(f"Class {i}: {error:.4f}")
    # print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return dict_true_acc_act, dict_true_acc_loc

