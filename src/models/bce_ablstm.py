"""
[file]          bce_ablstm.py
[description]   implement the encoder based on ABLSTM and having BCE cross enthropy based on WiMANS paper
                https://github.com/huangshk/WiMANS
"""
#
##
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
#
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
from src.train import train
from configs.preset import preset
from src.utils import *
import wandb

#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- ABLSTM ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class ABLSTM(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(ABLSTM, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        #
        self.layer_bilstm = torch.nn.LSTM(input_size = var_dim_input,
                                          hidden_size = 512,
                                          batch_first = True,
                                          bidirectional = True)
        #
        ##
        self.layer_linear = torch.nn.Linear(2*512, 2*512)
        self.layer_activation = torch.nn.LeakyReLU()
        #
        ##
        self.layer_output = torch.nn.Linear(2*512, var_dim_output)
        #
        ##
        self.layer_softmax = torch.nn.Softmax(dim = -2)
        #
        ##
        self.layer_pooling = torch.nn.AvgPool1d(8, 8)
        #
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_dropout = torch.nn.Dropout(0.6)  

        torch.nn.init.xavier_uniform_(self.layer_linear.weight)
        torch.nn.init.xavier_uniform_(self.layer_output.weight)
    
    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = torch.permute(var_t, (0, 2, 1))
        var_t = self.layer_norm(var_t)
        var_t = self.layer_pooling(var_t)
        var_t = torch.permute(var_t, (0, 2, 1))
        #
        var_h, _ = self.layer_bilstm(var_t)

        var_s = self.layer_linear(var_h)
        var_s = self.layer_activation(var_s)

        var_a = self.layer_softmax(var_s)

        var_t = var_h * var_a

        #
        var_t = torch.sum(var_t, dim = -2)

        var_t = self.layer_dropout(var_t)
        
        var_t = self.layer_output(var_t)

        var_output = var_t
        #
        return var_output

#
##
def run_bce_ablstm(data_train_x,
               data_train_y,
               data_test_x,
               data_test_y,
               var_repeat = 10):
    """
    [description]
    : run WiFi-based model ABLSTM
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
                                                                            test_size = 0.5,
                                                                            shuffle = True,
                                                                            random_state = 39)

    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape = data_train_x[0].shape, data_train_y[0].reshape(-1).shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []
    #
    ##
    var_macs, var_params = get_model_complexity_info(ABLSTM(var_x_shape, var_y_shape), 
                                                     var_x_shape, as_strings = False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"BCE_ABLSTM_{var_r}_" + "_".join(preset["data"]["environment"])

        run = wandb.init(
            project="FINAL_FINAL_EEEFINAL",
            name= name_run,
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_ablstm = ABLSTM(var_x_shape, var_y_shape).to(device)
        #
        if preset.get("pretrained_path"):
            model_ablstm, param_groups = load_model_components(
                model_ablstm,
                preset["pretrained_path"],
                preset["nn"]["lr"],
                preset.get("transfer_scenario"),
                device
            )
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(model_ablstm.parameters(), 
                                         lr = preset["nn"]["lr"],
                                         weight_decay = preset["nn"]["weight_decay"])

        #
        loss_mode = "baseline"
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([4] * var_y_shape[-1]).to(device))
        # loss = torch.nn.MSELoss()
        # loss = torch.nn.SmoothL1Loss()
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_ablstm, 
                                optimizer = optimizer, 
                                loss = loss, 
                                data_train_set = data_train_set,
                                data_valid_set = data_valid_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device,
                                var_mode = loss_mode)
        #
        var_time_1 = time.time()

        if preset.get("save_model"):
            save_model_components(preset, model_ablstm)

        #
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_ablstm.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_ablstm(torch.from_numpy(data_test_x).to(device))
        #
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        dict_true_acc = performance_metrics(data_test_y, predict_test_y, var_mode=loss_mode, var_threshold= preset["nn"]["threshold"])
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
        result_accuracy.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_total_error.append(dict_true_acc['total_error'])
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])

    # Calculate aggregated metrics with standard errors
    ppp_array = np.array(result_accuracy)  # result_accuracy contains PPP values
    precision_array = np.array(result_precision)
    recall_array = np.array(result_recall)
    f1_array = np.array(result_f1_score)
    total_error_array = np.array(result_total_error)
    
    # Create result dictionary with averaged metrics and standard errors
    aggregated_result = {
        'avg_PPP': float(np.mean(ppp_array)),
        'avg_precision': float(np.mean(precision_array)),
        'avg_recall': float(np.mean(recall_array)),
        'avg_f1_score': float(np.mean(f1_array)),
        'avg_total_error': float(np.mean(total_error_array)),
        'std_PPP': float(np.std(ppp_array, ddof=1)) if len(ppp_array) > 1 else 0.0,
        'std_precision': float(np.std(precision_array, ddof=1)) if len(precision_array) > 1 else 0.0,
        'std_recall': float(np.std(recall_array, ddof=1)) if len(recall_array) > 1 else 0.0,
        'std_f1_score': float(np.std(f1_array, ddof=1)) if len(f1_array) > 1 else 0.0,
        'std_total_error': float(np.std(total_error_array, ddof=1)) if len(total_error_array) > 1 else 0.0,
        'se_PPP': float(np.std(ppp_array, ddof=1) / np.sqrt(len(ppp_array))) if len(ppp_array) > 1 else 0.0,
        'se_precision': float(np.std(precision_array, ddof=1) / np.sqrt(len(precision_array))) if len(precision_array) > 1 else 0.0,
        'se_recall': float(np.std(recall_array, ddof=1) / np.sqrt(len(recall_array))) if len(recall_array) > 1 else 0.0,
        'se_f1_score': float(np.std(f1_array, ddof=1) / np.sqrt(len(f1_array))) if len(f1_array) > 1 else 0.0,
        'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0,
        'avg_train_time': sum(result_time_train) / len(result_time_train),
        'avg_test_time': sum(result_time_test) / len(result_time_test),
    }
    
    wandb.log({
        "avg_accuracy": aggregated_result['avg_PPP'],
        "avg_train_time": aggregated_result['avg_train_time'],
        "avg_test_time": aggregated_result['avg_test_time'],
        "avg_total_error": aggregated_result['avg_total_error'],
        "avg_precision": aggregated_result['avg_precision'],
        "avg_recall": aggregated_result['avg_recall'],
        "avg_f1_score": aggregated_result['avg_f1_score'],
    })
    viz_stats = visualize_model_performance(
        y_pred=predict_test_y,
        y_true=data_test_y,
        var_mode=loss_mode,
        save_dir=f'./visualizations/experiment__{loss_mode}'
    )

    # Print additional statistics
    print("\nDetailed Performance Analysis:")
    print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    print("\nClass-wise Mean Absolute Error:")
    for i, error in enumerate(viz_stats['class_wise_mae']):
        print(f"Class {i}: {error:.4f}")
    print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")

    wandb.finish()
    return aggregated_result