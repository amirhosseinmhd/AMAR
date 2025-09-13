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
from sklearn.decomposition import PCA
from model.modules.molecules import PCAFeatureExtractor, Transformer_Encoder, TransformerDecoder
from model.losses.supervised_loss import HungarianMatchingLoss
from train import train
from preset import preset
from utils import *
import wandb





class DETR_MultiUser(nn.Module):
    def __init__(self, var_x_shape, features_dim = 20, embedding_time_dim=100, num_decoder_layers=12,
                 temp_cross=1, n_attention_heads=2, num_queries=5, dim_feedforward=1024, query_dropout_rate=0.0
                 , pca_embeddings=None,):
        super().__init__()
        # self.feature_extractor = CNNFeatureExtractor(input_channels=var_x_shape[-1], output_channels=features_dim,embedding_time_dim=embedding_time_dim)
        self.feature_extractor = PCAFeatureExtractor(input_channels=270, output_channels=preset["nn"]["d_embedding"])
                                                     # embedding_time_dim=preset["cnn_embedding_time_dim"])
                                                                # pca_components=pca_embeddings)

        # self.encoder = Transformer_Encoder(var_embedding_shape, num_attention_heads=n_attention_heads,
        #                                    num_transformer_encoder_layers=8)
        self.encoder = Transformer_Encoder( d_model=preset["nn"]["d_embedding"], nhead=n_attention_heads, num_layers=preset["nn"]["n_layers_encoder"],
                 max_total_tokens=preset["nn"]["token_length"])
        self.decoder = TransformerDecoder(
            d_model=features_dim,
            nhead=n_attention_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross, 
            query_dropout_rate=query_dropout_rate
        )
        self.decoder.memory_pos_encoding = self.encoder.pos_encoder
    def forward(self, x):

        extracted_features = self.feature_extractor(x)

        memory = self.encoder(extracted_features)

        outputs_class = self.decoder(memory)

        return outputs_class


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
    ## ============================================ Preprocess ============================================
    #
    # Update device selection to check for CUDA first, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    #
    ## Remove the internal validation split since validation data is now provided directly
    # data_test_x, data_valid_x, data_test_y, data_valid_y = strat_train_test_split(
    #     data_x=data_test_x,
    #     data_y=data_test_y,
    #     test_size=0.5,
    #     shuffle=True,
    #     random_state=39)

    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(data_test_x, data_test_y,
                                                                            test_size=0.5,
                                                                            shuffle=True,
                                                                            random_state=39)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    data_x_mean = np.mean(data_train_x, axis=1)
    pca = PCA(n_components=50)
    pca.fit(data_x_mean)
    pca_components = torch.from_numpy(pca.components_.T).float().to(device)

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
                                    query_dropout_rate=preset["nn"]["query_dropout_rate"],
                                    pca_embeddings=pca_components.to(torch.device("cpu"))),var_x_shape, as_strings=False)

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
                                    query_dropout_rate=preset["nn"]["query_dropout_rate"]#,
                                    # pca_embeddings=pca_components
                                    ).to(device)

        # model_detr.feature_extractor = torch.compile(model_detr.feature_extractor)
        # model_detr.decoder = torch.compile(model_detr.decoder)        # wandb.watch(
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
                                data_valid_set=data_valid_set,
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

