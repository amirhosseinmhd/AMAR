#
##
import os
import math
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
#
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ptflops import get_model_complexity_info
from sklearn.decomposition import PCA
from model.modules.molecules import PCAFeatureExtractor, Transformer_Encoder, TransformerDecoder, VectorQuantizer, ResidualVectorQuantizer
from model.losses.supervised_loss import HungarianMatchingLoss
from train_vq import train as train_vq_func
from train_rvq import train as train_rvq_func
from preset import preset
from utils import *
import wandb




class AMAR_MultiUser_RVQ(nn.Module):
    """
    AMAR model with Residual Vector Quantization (RVQ).
    Architecture: CNN → RVQ → TransformerEncoder → Decoder
    """
    def __init__(self, var_x_shape, features_dim=20, embedding_time_dim=100, num_decoder_layers=12,
                 temp_cross=1, n_attention_heads=2, num_queries=5, dim_feedforward=1024, query_dropout_rate=0.0,
                 pca_embeddings=None, num_embeddings=1024, commitment_cost=0.25, 
                 codebook_initial_embeddings=None, num_rvq_layers=3):
        super().__init__()
        
        # Feature extractor (CNN part)
        self.feature_extractor = PCAFeatureExtractor(
            input_channels=270, 
            output_channels=preset["nn"]["d_embedding"],
            pca_components=None
        )

        # Residual Vector Quantizer (RVQ part)
        self.rvq_layer = ResidualVectorQuantizer(
            num_layers=num_rvq_layers,
            num_embeddings=num_embeddings,
            embedding_dim=preset["nn"]["d_embedding"],
            commitment_cost=commitment_cost,
            initial_embeddings_first_layer=None,
            quantization_dropout=preset["nn"].get("quantization_dropout", 0.2),
        )
        
        # Transformer Encoder
        self.encoder = Transformer_Encoder(
            d_model=preset["nn"]["d_embedding"], 
            nhead=n_attention_heads, 
            num_layers=preset["nn"]["n_layers_encoder"],
            max_total_tokens=preset["nn"]["token_length"]
        )
        
        # Transformer Decoder
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
        
        # Share positional encoding between encoder and decoder
        self.decoder.memory_pos_encoding = self.encoder.pos_encoder
    
    def forward(self, x):
        # Step 1: CNN Feature Extraction
        continuous_emb = self.feature_extractor(x)

        # Step 2: Residual Vector Quantization
        quantized_emb_st, quantized_emb, all_indices, all_quantized_layers = self.rvq_layer(continuous_emb)

        # Step 3: Transformer Encoder
        memory = self.encoder(quantized_emb_st)

        # Step 4: Transformer Decoder
        outputs_class = self.decoder(memory)

        return outputs_class, continuous_emb, quantized_emb, all_indices, all_quantized_layers

    def compute_rvq_loss(self, continuous_emb, all_quantized_layers):
        """
        Compute RVQ loss across all quantization layers.
        """
        return self.rvq_layer.compute_loss(continuous_emb, all_quantized_layers)



def run_that_AMARRVQ(data_train_x,
                     data_train_y,
                     data_test_x,
                     data_test_y,
                     var_repeat=10):
    """
    [description]
    : run WiFi-based model Transformer_Encoder_DECODER with Residual Vector Quantization (RVQ)
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
    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(data_test_x, data_test_y,
                                                                            test_size=0.5,
                                                                            shuffle=True,
                                                                            random_state=39)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    data_x_mean = np.mean(data_train_x, axis=1)
    if preset["nn"]["PCA"]:
        pca = PCA(n_components=50)
        pca.fit(data_x_mean)
        pca_components = torch.from_numpy(pca.components_.T).float()
        print(" Using PCA embeddings: mapping 270 to PCA components")
    else:
        pca_components = None
    ## shape for model
    var_x_shape = data_train_x[0].shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))

    
    if preset["nn"]["KMEANS_Initialization"]:
        # K-means initialization for VQ-VAE codebook
        print("Initializing codebook with K-means...")
        feature_extractor_for_init = PCAFeatureExtractor(input_channels=270, output_channels=preset["nn"]["d_embedding"]).to(device)

        # Get a subset of data for initialization
        init_loader = DataLoader(data_train_set, batch_size=1024, shuffle=True)
        init_data, _ = next(iter(init_loader))
        init_data = init_data.to(device)

        with torch.no_grad():
            initial_embeddings = feature_extractor_for_init(init_data)

        # Flatten embeddings for KMeans
        initial_embeddings_flat = initial_embeddings.reshape(-1, preset["nn"]["d_embedding"]).cpu().numpy()

        num_embeddings = preset["nn"]["num_codes"]
        kmeans = KMeans(n_clusters=num_embeddings, random_state=0, n_init=8).fit(initial_embeddings_flat)
        codebook_initial_embeddings = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
    else:
        codebook_initial_embeddings = None # Initialize codebook with random embeddings
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

    # Calculate model complexity for RVQ model
    var_macs, var_params = get_model_complexity_info(AMAR_MultiUser_RVQ(var_x_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"],
                                    query_dropout_rate=preset["nn"]["query_dropout_rate"],
                                    num_embeddings=preset["nn"]["num_codes"],
                                    pca_embeddings=pca_components,
                                    codebook_initial_embeddings=codebook_initial_embeddings,
                                    num_rvq_layers=preset["nn"]["num_rvq_layers"]),var_x_shape, as_strings=False)

    print("RVQ Model Parameters:", var_params, "- FLOPs:", var_macs * 2)

    for var_r in range(var_repeat):
        #
        ##
        var_mode = "multi_head"
        if preset["pretrained_path"]:
            name_run = f"AMAR_RVQ_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + preset["transfer_scenario"]
        else:
            pretrained_state = "NPT"
            name_run = f"AMAR_RVQ_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + pretrained_state 
        print("Repeat", var_r)
        run = wandb.init(
            project="REALREAL_FINAL_RVQ",
            name= name_run + preset["wandb_name"],
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_AMARRVQ = AMAR_MultiUser_RVQ(var_x_shape,
                                    n_attention_heads=preset["nn"]["n_attention_heads"],
                                    features_dim=preset["nn"]["d_embedding"],
                                    embedding_time_dim=preset["nn"]["token_length"],
                                    num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                    temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"],
                                    dim_feedforward=preset["nn"]["dim_FFN"],
                                    num_embeddings=preset["nn"]["num_codes"],
                                    query_dropout_rate=preset["nn"]["query_dropout_rate"],
                                    commitment_cost=preset["nn"]["commitment_cost"],
                                    pca_embeddings=pca_components if not pca_components else pca_components.to(device),
                                    codebook_initial_embeddings=codebook_initial_embeddings if not codebook_initial_embeddings else codebook_initial_embeddings.to(device),
                                    num_rvq_layers=preset["nn"]["num_rvq_layers"]).to(device)

        # model_AMARRVQ.feature_extractor = torch.compile(model_AMARRVQ.feature_extractor)
        #
        ##
        # Separate learning rates: 0.1x for codebook, 1x for other parameters
        codebook_params = []
        other_params = []
        
        for name, param in model_AMARRVQ.named_parameters():
            if 'rvq_layer' in name and 'embedding.weight' in name:
                codebook_params.append(param)
            else:
                other_params.append(param)
        
        if preset.get("pretrained_path"):
            model_AMARRVQ, param_groups = load_model_components(
                model=model_AMARRVQ,
                load_path=preset["pretrained_path"],
                lr = preset["nn"]["lr"],
                scenario=preset.get("transfer_scenario"),
                device=device
            )
            # Update param_groups to include separate learning rate for codebook
            # Note: This assumes load_model_components returns standard param_groups
            # We'll need to add codebook-specific groups
            param_groups_with_codebook = param_groups + [
                {'params': codebook_params, 'lr': preset["nn"]["lr"] * 0.1, 'weight_decay': preset["nn"]["weight_decay"]}
            ]
            optimizer = torch.optim.Adam(param_groups_with_codebook)
        else:
            param_groups = [
                {'params': other_params, 'lr': preset["nn"]["lr"], 'weight_decay': preset["nn"]["weight_decay"]},
                {'params': codebook_params, 'lr': preset["nn"]["lr"] * 0.1, 'weight_decay': preset["nn"]["weight_decay"]}
            ]
            optimizer = torch.optim.Adam(param_groups)
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
        var_best_weight = train_rvq_func(model=model_AMARRVQ,
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
            save_model_components(preset, model_AMARRVQ)
        #
        var_time_1 = time.time()
        #

        ## ---------------------------------------- Test ------------------------------------------
        #
        model_AMARRVQ.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            data_test_x_tensor = torch.from_numpy(data_test_x).to(device)
            outputs_class, continuous_emb, quantized_emb, all_indices, all_quantized_layers = model_AMARRVQ(data_test_x_tensor)
        #
        predict_test_y = outputs_class.detach().cpu().numpy()
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

    # Log aggregated results only if not in last_layer_only mode
        if last_layer_only:
            layer_metrics = dict_layer_acc["layer_" + str(preset["nn"]["num_decoder_layers"] - 1)]
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

    # Calculate averages and standard errors for each layer
    for layer_idx_num, layer_idx in enumerate(layers_idxs):
        # Calculate metrics with standard errors
        ppp_array = np.array(result_ppp[layer_idx_num])
        precision_array = np.array(result_precision[layer_idx_num])
        recall_array = np.array(result_recall[layer_idx_num])
        f1_array = np.array(result_f1_score[layer_idx_num])
        accuracy_array = np.array(result_accuracy[layer_idx_num])
        total_error_array = np.array(result_total_error[layer_idx_num])
        
        # Store results for this layer
        all_layers_results[layer_idx] = {
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
            'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0
        }
        
        wandb.log({
            f"test_results/{layer_idx}/avg_PPP": all_layers_results[layer_idx]['avg_PPP'],
            f"test_results/{layer_idx}/avg_train_time": sum(result_time_train[layer_idx_num]) / len(result_time_train[layer_idx_num]),
            f"test_results/{layer_idx}/avg_test_time": sum(result_time_test[layer_idx_num]) / len(result_time_test[layer_idx_num]),
            f"test_results/{layer_idx}/avg_total_error": all_layers_results[layer_idx]['avg_total_error'],
            f"test_results/{layer_idx}/avg_precision": all_layers_results[layer_idx]['avg_precision'],
            f"test_results/{layer_idx}/avg_recall": all_layers_results[layer_idx]['avg_recall'],
            f"test_results/{layer_idx}/avg_f1_score": all_layers_results[layer_idx]['avg_f1_score'],
            f"test_results/{layer_idx}/avg_count_error": sum(result_avg_count_error[layer_idx_num]) / len(result_avg_count_error[layer_idx_num]),
            f"test_results/{layer_idx}/avg_accuracy": all_layers_results[layer_idx]['avg_accuracy']
        })  # Use an even larger offset for averages

    # Use the last layer for visualization and final results
    last_layer = layers_idxs[-1]
    last_layer_predictions = predict_test_y[last_layer] if isinstance(predict_test_y, dict) else predict_test_y
    # dict_true_acc = all_layers_results[last_layer]

    # Run visualization with the last layer's predictions
    
    log_random_attention_weights_final(model_AMARRVQ, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), 1000000000)
    
    viz_stats = visualize_model_performance(
        y_pred=last_layer_predictions,
        y_true=data_test_y,
        var_mode=var_mode,
        save_dir=f'./visualizations/experiment_{var_r}_{var_mode}_rvq{preset["nn"]["num_rvq_layers"]}'
    )
    print("\nDetailed Performance Analysis:")
    print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    print("\nClass-wise Mean Absolute Error:")
    for i, error in enumerate(viz_stats['class_wise_mae']):
        print(f"Class {i}: {error:.4f}")
    print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return all_layers_results

