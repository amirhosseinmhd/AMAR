"""
[file]          that.py
[description]   implement and evaluate WiFi-based model THAT_ENCODER
                https://github.com/windofshadow/THAT
"""
#
##
import time
import torch
import numpy as np
#
import torch.nn as nn
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from itertools import permutations
from sklearn.metrics import classification_report, accuracy_score
#
from src.train import train
from configs.preset import preset
from src.utils import *
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
class THAT(torch.nn.Module):
    #
    ##
    def __init__(self, 
                 var_x_shape, 
                 var_y_shape):
        #
        ##
        super(THAT, self).__init__()
        #
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]
        self.num_heads = 5

        # Replace the single output layer with multiple prediction heads
        self.layer_output = torch.nn.ModuleList([
            torch.nn.Linear(256 + 32, var_dim_output) for _ in range(self.num_heads)
        ])

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
        # self.layer_output = torch.nn.Linear(256 + 32, var_dim_output)
    
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
        var_outputs = [head(var_t) for head in self.layer_output]
        var_output = torch.stack(var_outputs, dim=1)  # Shape: (batch_size, num_heads, var_dim_output)
#
        ##
        return var_output
class PermutationMatchingLoss(nn.Module):
    def __init__(self):
        super(PermutationMatchingLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        batch_size, num_heads, num_classes = predictions.shape
        assert num_heads == 5, "Number of prediction heads must be 5"
        if targets.shape != (batch_size, num_heads, num_classes):
            print("target miss match")
            print(targets.shape)
        all_permutations = list(permutations(range(num_heads)))
        best_loss = torch.full((batch_size,), float('inf'), device=predictions.device)
        best_perm_indices = torch.zeros(batch_size, dtype=torch.long, device=predictions.device)

        for batch_idx in range(batch_size):
            for perm_idx, perm in enumerate(all_permutations):
                perm_predictions = predictions[batch_idx, perm, :]
                perm_targets = targets[batch_idx]
                loss = self.ce_loss(perm_predictions, perm_targets.argmax(dim=1))
                loss = loss.mean()  # Average loss across heads for this permutation
                if loss < best_loss[batch_idx]:
                    best_loss[batch_idx] = loss
                    best_perm_indices[batch_idx] = perm_idx

        # Create a new tensor with the best permutations
        best_predictions = torch.zeros_like(predictions)
        for batch_idx in range(batch_size):
            best_perm = all_permutations[best_perm_indices[batch_idx]]
            best_predictions[batch_idx] = predictions[batch_idx, best_perm, :]

        # Compute the final loss using the best permutations
        final_loss = self.ce_loss(best_predictions.view(-1, num_classes), targets.view(-1, num_classes).argmax(dim=1))
        return final_loss.mean()
#
##
def run_that_multihead(data_train_x,
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
    var_x_shape, var_y_shape = data_train_x[0].shape,[data_train_y[0].shape[1]]
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    # wandb.init(project="wifi-based-model-THAT_ENCODER", config={
    #     "model": "THAT_multi_head",
    #     "repeat_experiments": var_repeat,
    # })
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    result_precision = []
    result_recall = []
    result_f1_score = []
    result_total_error = []
    result_ppp = []
    #
    ##
    var_macs, var_params = get_model_complexity_info(THAT(var_x_shape, var_y_shape),
                                                     var_x_shape, as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        run = wandb.init(
            project="wifi-based-model-THAT_ENCODER",
            name=f"Repeat_{var_r}",
            config={
                "model": "THAT_MultiHead",
                "repeat": var_r,
            },
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_that = THAT(var_x_shape, var_y_shape).to(device)
        #
        optimizer = torch.optim.Adam(model_that.parameters(),
                                     lr = preset["nn"]["lr"],
                                     weight_decay = 0)
        #
        # loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([4] * var_y_shape[-1]).to(device))
        # loss = torch.nn.MSELoss()
        # loss = torch.nn.SmoothL1Loss()
        loss = PermutationMatchingLoss()
        var_mode = "multi_head"

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
                                var_mode=var_mode)
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
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        # data_test_y_c = data_test_y.sum(axis=1)
        dict_true_acc = calculate_matrix_absolute_error(data_test_y, predict_test_y, var_mode=var_mode)
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
        result_accuracy.append(dict_true_acc['accuracy'])
        result_ppp.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])
        result_total_error.append(dict_true_acc['total_error'])
    # Calculate aggregated metrics with standard errors
    ppp_array = np.array(result_ppp)
    precision_array = np.array(result_precision)
    recall_array = np.array(result_recall)
    f1_array = np.array(result_f1_score)
    accuracy_array = np.array(result_accuracy)
    total_error_array = np.array(result_total_error)
    
    # Create result dictionary with averaged metrics and standard errors
    result = {
        'avg_PPP': float(np.mean(ppp_array)),
        'avg_precision': float(np.mean(precision_array)),
        'avg_recall': float(np.mean(recall_array)),
        'avg_f1_score': float(np.mean(f1_array)),
        'avg_accuracy': float(np.mean(accuracy_array)),
        'avg_total_error': float(np.mean(total_error_array)),
        'std_PPP': float(np.std(ppp_array, ddof=1)) if len(ppp_array) > 1 else 0.0,
        'std_precision': float(np.std(precision_array, ddof=1)) if len(precision_array) > 1 else 0.0,
        'std_recall': float(np.std(recall_array, ddof=1)) if len(recall_array) > 1 else 0.0,
        'std_f1_score': float(np.std(f1_array, ddof=1)) if len(f1_array) > 1 else 0.0,
        'std_accuracy': float(np.std(accuracy_array, ddof=1)) if len(accuracy_array) > 1 else 0.0,
        'std_total_error': float(np.std(total_error_array, ddof=1)) if len(total_error_array) > 1 else 0.0,
        'se_PPP': float(np.std(ppp_array, ddof=1) / np.sqrt(len(ppp_array))) if len(ppp_array) > 1 else 0.0,
        'se_precision': float(np.std(precision_array, ddof=1) / np.sqrt(len(precision_array))) if len(precision_array) > 1 else 0.0,
        'se_recall': float(np.std(recall_array, ddof=1) / np.sqrt(len(recall_array))) if len(recall_array) > 1 else 0.0,
        'se_f1_score': float(np.std(f1_array, ddof=1) / np.sqrt(len(f1_array))) if len(f1_array) > 1 else 0.0,
        'se_accuracy': float(np.std(accuracy_array, ddof=1) / np.sqrt(len(accuracy_array))) if len(accuracy_array) > 1 else 0.0,
        'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0,
        'avg_train_time': sum(result_time_train) / len(result_time_train),
        'avg_test_time': sum(result_time_test) / len(result_time_test),
    }
    
    wandb.log({
        "avg_accuracy": result['avg_accuracy'],
        "avg_train_time": result['avg_train_time'],
        "avg_test_time": result['avg_test_time'],
        "avg_precision": result['avg_precision'],
        "avg_recall": result['avg_recall'],
        "avg_f1_score": result['avg_f1_score'],
        "avg_PPP": result['avg_PPP'],
        "avg_total_error": result['avg_total_error']
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
    return result
    #
    #     predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
    #     predict_test_y = predict_test_y.detach().cpu().numpy()
    #     #
    #     var_time_2 = time.time()
    #     #
    #     ## -------------------------------------- Evaluate ----------------------------------------
    #     #
    #     ##
    #     data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
    #     predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])
    #     #
    #     ## Accuracy
    #     result_acc = accuracy_score(data_test_y_c.astype(int),
    #                                 predict_test_y_c.astype(int))
    #     #
    #     ## Report
    #     result_dict = classification_report(data_test_y_c,
    #                                         predict_test_y_c,
    #                                         digits = 6,
    #                                         zero_division = 0,
    #                                         output_dict = True)
    #     #
    #     result["repeat_" + str(var_r)] = result_dict
    #     #
    #     result_accuracy.append(result_acc)
    #     result_time_train.append(var_time_1 - var_time_0)
    #     result_time_test.append(var_time_2 - var_time_1)
    #     #
    #     print("repeat_" + str(var_r), result_accuracy)
    #     print(result)
    # #
    # ##
    # result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    # result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    # result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    # result["complexity"] = {"parameter": var_params, "flops": var_macs * 2}
    # #
    # return result