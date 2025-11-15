"""
[file]          that.py
[description]   implement and evaluate WiFi-based model THAT_ENCODER
                https://github.com/windofshadow/THAT
"""
#
##
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from itertools import permutations
from sklearn.metrics import classification_report, accuracy_score
#
from scipy.optimize import linear_sum_assignment
from src.train import train
from configs.preset import preset
from src.utils import *
import wandb
import torch.nn.functional as F
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
                 var_num_gaussian = 10):
        #
        ##
        super(Gaussian_Position, self).__init__()
        #
        ## var_embedding: shape (var_dim_k, var_dim_feature)
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype = torch.float)
        self.var_embedding = torch.nn.Parameter(var_embedding, requires_grad = True)
        torch.nn.init.xavier_uniform_(self.var_embedding)
        #
        ## var_position: shape (var_dim_time, var_dim_k)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = torch.nn.Parameter(var_position, requires_grad = False)
        #
        ## var_mu: shape (1, var_dim_k)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time/var_num_gaussian).unsqueeze(0)
        self.var_mu = torch.nn.Parameter(var_mu, requires_grad = True)
        #
        ## var_sigma: shape (1, var_dim_k)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = torch.nn.Parameter(var_sigma, requires_grad = True)

    #
    ##
    def calculate_pdf(self,
                      var_position, 
                      var_mu, 
                      var_sigma):
        #
        ##
        var_pdf = var_position - var_mu                 # (position-mu)
        #
        var_pdf = - var_pdf * var_pdf                   # -(position-mu)^2
        #
        var_pdf = var_pdf / var_sigma / var_sigma / 2   # -(position-mu)^2 / (2*sigma^2)
        #
        var_pdf = var_pdf - torch.log(var_sigma)        # -(position-mu)^2 / (2*sigma^2) - log(sigma)
        #
        return var_pdf

    #
    ##
    def forward(self, 
                var_input):
        

        var_pdf = self.calculate_pdf(self.var_position, self.var_mu, self.var_sigma)
        
        var_pdf = torch.softmax(var_pdf, dim = -1)
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
                 var_num_head = 10,
                 var_size_cnn = [1, 3, 5]):
        #
        ##
        super(Encoder, self).__init__()
        #
        ##
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps = 1e-6)
        self.layer_attention = torch.nn.MultiheadAttention(var_dim_feature, 
                                                           var_num_head,
                                                           batch_first = True)
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
                                                        padding = "same"),
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
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim = 0)
        #
        var_s = torch.sum(var_c, dim = 0) / len(self.layer_cnn)
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
## ---------------------------------------- THAT_ENCODER -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class THAT_COUNT_PRED_with_Constraint(torch.nn.Module):
    #
    ##
    def __init__(self, 
                 var_x_shape, 
                 var_y_shape):
        #
        ##
        super(THAT_COUNT_PRED_with_Constraint, self).__init__()
        #
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        # var_dim_output = var_y_shape[-1]
        self.num_classes = 10
        var_dim_output = 5 * self.num_classes
        #
        ## ---------------------------------------- left ------------------------------------------
        #
        self.layer_left_pooling = torch.nn.AvgPool1d(kernel_size = 20, stride = 20)
        self.layer_left_gaussian = Gaussian_Position(var_dim_feature, var_dim_time // 20)
        #
        var_num_left = 4
        var_dim_left = var_dim_feature
        self.layer_left_encoder = torch.nn.ModuleList([Encoder(var_dim_feature = var_dim_left,
                                                               var_num_head = 10,
                                                               var_size_cnn = [1, 3, 5])
                                                               for _ in range(var_num_left)])
        #
        self.layer_left_norm = torch.nn.LayerNorm(var_dim_left, eps = 1e-6)
        #
        self.layer_left_cnn_0 =  torch.nn.Conv1d(in_channels = var_dim_left,
                                                 out_channels = 128,
                                                 kernel_size = 8)
        
        self.layer_left_cnn_1 =  torch.nn.Conv1d(in_channels = var_dim_left,
                                                 out_channels = 128,
                                                 kernel_size = 16)
        #
        self.layer_left_dropout = torch.nn.Dropout(0.5)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        self.layer_right_pooling = torch.nn.AvgPool1d(kernel_size = 20, stride = 20)
        #
        var_num_right = 1 
        var_dim_right = var_dim_time // 20
        self.layer_right_encoder = torch.nn.ModuleList([Encoder(var_dim_feature = var_dim_right,
                                                                var_num_head = 10,
                                                                var_size_cnn = [1, 2, 3])
                                                                for _ in range(var_num_right)])
        #
        self.layer_right_norm = torch.nn.LayerNorm(var_dim_right, eps = 1e-6)
        #
        self.layer_right_cnn_0 =  torch.nn.Conv1d(in_channels = var_dim_right,
                                                  out_channels = 16,
                                                  kernel_size = 2)
        
        self.layer_right_cnn_1 =  torch.nn.Conv1d(in_channels = var_dim_right,
                                                  out_channels = 16,
                                                  kernel_size = 4)
        #
        self.layer_right_dropout = torch.nn.Dropout(0.5)
        #
        ##
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        ##
        self.soft_max = torch.nn.Softmax()
        self.layer_output = torch.nn.Linear(256 + 32, var_dim_output)
    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input   # shape (batch_size, time_steps, features)
        #
        ## ---------------------------------------- left ------------------------------------------
        #
        var_left = torch.permute(var_t, (0, 2, 1))
        var_left = self.layer_left_pooling(var_left)
        var_left = torch.permute(var_left, (0, 2, 1))
        #
        var_left = self.layer_left_gaussian(var_left)
        #
        for layer in self.layer_left_encoder: var_left = layer(var_left)
        #
        var_left = self.layer_left_norm(var_left)
        #
        var_left = torch.permute(var_left, (0, 2, 1))
        var_left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(var_left))
        var_left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(var_left))
        #
        var_left_0 = torch.sum(var_left_0, dim = -1)
        var_left_1 = torch.sum(var_left_1, dim = -1)
        #
        var_left = torch.concat([var_left_0, var_left_1], dim = -1)
        var_left = self.layer_left_dropout(var_left)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        var_right = torch.permute(var_t, (0, 2, 1)) # shape (batch_size, features, time_steps)
        var_right = self.layer_right_pooling(var_right)
        #
        for layer in self.layer_right_encoder: var_right = layer(var_right)
        #
        var_right = self.layer_right_norm(var_right)
        #
        var_right = torch.permute(var_right, (0, 2, 1))
        var_right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(var_right))
        var_right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(var_right))
        #
        var_right_0 = torch.sum(var_right_0, dim = -1)
        var_right_1 = torch.sum(var_right_1, dim = -1)
        #
        var_right = torch.concat([var_right_0, var_right_1], dim = -1)
        var_right = self.layer_right_dropout(var_right)
        #
        ## concatenate
        var_t = torch.concat([var_left, var_right], dim = -1)
        #
        concatinated_logits_of_activities = self.layer_output(var_t)
        extended_logits = concatinated_logits_of_activities.view(-1, 5, self.num_classes)
        prob_activity_perperson = torch.nn.functional.softmax(extended_logits, dim=2)
        pred_probs = prob_activity_perperson.sum(axis=1) # [batch_size, 10]        return pred_probs
        return pred_probs
#
##
class CountBasedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        predictions: [batch_size, 10] (already summed in model's forward pass)
        targets: [batch_size, 5, 10] (one-hot encoded)
        """
        # Sum the one-hot targets along person dimension to get true counts
        # target_counts = targets.sum(dim=1)  # [batch_size, 10]

        # Since the sum of predictions should equal 5 (total number of people),
        # we can either normalize the predictions or use them directly
        pred_counts = predictions  # Already normalized in forward pass

        # You can use various loss functions here:
        # Option 1: MSE Loss
        # loss = F.mse_loss(pred_counts, targets)

        # Option 2: L1 Loss
        # loss = F.l1_loss(pred_counts, target_counts)

        # Option 3: Smooth L1 Loss (what you're currently using)
        loss = F.smooth_l1_loss(pred_counts, targets)

        return loss
# class CountBasedLoss(nn.Module):
#     def __init__(self, count_weight=0.1):
#         super().__init__()
#         self.count_weight = count_weight
#
#     def compute_matching_loss(self, predictions, targets):
#         """
#         predictions: [batch_size, 5, 10] (after softmax)
#         targets: [batch_size, 5] (class indices 0-9)
#         """
#         batch_size = predictions.size(0)
#         total_loss = 0
#
#         for b in range(batch_size):
#             cost_matrix = torch.zeros(5, 5, device=predictions.device)
#             for i in range(5):
#                 for j in range(5):
#                     cost_matrix[i, j] = -predictions[b, i, targets[b, j]]
#
#             cost_matrix_np = cost_matrix.detach().cpu().numpy()
#             pred_indices, target_indices = linear_sum_assignment(cost_matrix_np)
#
#             for pred_idx, target_idx in zip(pred_indices, target_indices):
#                 loss = -torch.log(predictions[b, pred_idx, targets[b, target_idx]])
#                 total_loss += loss
#
#         return total_loss / batch_size
#
#     def compute_count_loss(self, predictions, targets):
#         """
#         predictions: [batch_size, 5, 10] (after softmax)
#         targets: [batch_size, 5] (class indices 0-9)
#         """
#         batch_size = predictions.size(0)
#
#         # Get predicted class distribution by averaging across the 5 predictions
#         pred_probs = predictions.mean(dim=1)  # [batch_size, 10]
#
#         # Get target counts
#         target_counts = torch.zeros((batch_size, 10), device=predictions.device)
#         for b in range(batch_size):
#             unique, counts = torch.unique(targets[b], return_counts=True)
#             # Convert unique indices to long/int64 type
#             unique = unique.long()
#             target_counts[b, unique] = counts.float()  # Convert counts to float
#         target_probs = target_counts / 5  # Normalize to get proportions
#
#         # KL divergence for the count distributions
#         count_loss = F.kl_div(pred_probs.log(), target_probs, reduction='batchmean')
#
#         return count_loss
#
#     def forward(self, predictions, targets):
#         # matching_loss = self.compute_matching_loss(predictions, targets)
#         count_loss = self.compute_count_loss(predictions, targets)
#         return count_loss
#         # total_loss = matching_loss + self.count_weight * count_loss
#         #
#         # return total_loss, {
#         #     'matching_loss': matching_loss.item(),
#         #     'count_loss': count_loss.item()
#         # }
#         #

def run_that_count_pred_contrained(data_train_x,
             data_train_y,
             data_test_x,
             data_test_y,
             var_repeat = 10):
    """
    [description]
    : run WiFi-based model THAT_ENCODER
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
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape = data_train_x[0].shape,[data_train_y.shape[1]]
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    wandb.init(project="wifi-based-model-THAT_ENCODER", config={
        "model": "THAT_ENCODER",
        "repeat_experiments": var_repeat,
    })
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    #
    ##
    var_macs, var_params = get_model_complexity_info(THAT_COUNT_PRED_with_Constraint(var_x_shape, var_y_shape),
                                                     var_x_shape, as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)

        #
        torch.random.manual_seed(var_r + 39)
        #
        model_that = THAT_COUNT_PRED_with_Constraint(var_x_shape, var_y_shape).to(device)
        #
        optimizer = torch.optim.Adam(model_that.parameters(),
                                     lr = 0.00003,
                                     weight_decay = 0.0001 )

        # loss_mode = "count_classification"
        loss = CountBasedLoss().to(device)

        loss_mode = "count_classification_withConstrain"
        run = wandb.init(
            project="wifi-based-model-THAT_ENCODER",
            name=f"Repeat_{var_r} " + loss_mode,
            config={
                "model": "THAT_COUNT_PRED",
                "repeat": var_r,
            },
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        # loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([4] * var_y_shape[-1]).to(device))
        # loss = PermutationMatchingLoss()

        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_that,
                                optimizer = optimizer,
                                loss = loss,
                                data_train_set = data_train_set,
                                data_test_set = data_test_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device,
                                var_mode = loss_mode)
        #
        var_time_1 = time.time()
        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_that.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_that(torch.from_numpy(data_test_x).to(device))
        #
        # predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        data_test_y_c = data_test_y
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        dict_true_acc = calculate_matrix_absolute_error(data_test_y_c, predict_test_y, var_mode=loss_mode)


        # Create visualizations
        viz_stats = visualize_model_performance(
            predictions=predict_test_y,
            targets=data_test_y_c,
            save_dir=f'./visualizations/experiment_{var_r}_constrained_count',
            var_mode=loss_mode
        )

        # Print additional statistics
        print("\nDetailed Performance Analysis:")
        print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
        print("\nClass-wise Mean Absolute Error:")
        for i, error in enumerate(viz_stats['class_wise_mae']):
            print(f"Class {i}: {error:.4f}")
        print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")

        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,
            "TOTAL_TESTSET_ERROR": dict_true_acc['total_error'],
            "TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc['perfect_prediction_percentage'],
            "TOTAL_ACCURACY": dict_true_acc['accuracy'],
        })
        print(" %.6fs" % (time.time() - var_time_1),
              "- Total Error %.6f" % dict_true_acc['total_error'],
              "-  perfect_prediction_percentage %.6f" % dict_true_acc['perfect_prediction_percentage'],
              )
        #
        #

        #
        result_accuracy.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
    wandb.log({
        "avg_accuracy": sum(result_accuracy) / len(result_accuracy),
        "avg_train_time": sum(result_time_train) / len(result_time_train),
        "avg_test_time": sum(result_time_test) / len(result_time_test),
    })
    wandb.finish()
    return dict_true_acc