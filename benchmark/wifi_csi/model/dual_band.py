import time
import torch
import numpy as np
#
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
import time
import torch
import torch._dynamo
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
# from train import train
from preset import preset
from model.detr import *
import wandb


class DualBandDETR(torch.nn.Module):
    def __init__(self, var_x_shape_band1, var_x_shape_band2, var_y_shape):
        super(DualBandDETR, self).__init__()
        
        # Feature extractors for each band
        self.feature_extractor_band1 = CNNFeatureExtractor(
            input_channels=var_x_shape_band1[-1],
            output_channels=preset["nn"]["d_embedding"],
            embedding_time_dim=preset["nn"]["token_length"]
        )
        
        self.feature_extractor_band2 = CNNFeatureExtractor(
            input_channels=var_x_shape_band2[-1],
            output_channels=preset["nn"]["d_embedding"],
            embedding_time_dim=preset["nn"]["token_length"]
        )
        
        # Encoders for each band
        var_embedding_shape = (preset["nn"]["token_length"], preset["nn"]["d_embedding"])
        self.encoder_band1 = Transformer_Encoder(
            var_embedding_shape=var_embedding_shape,
            num_attention_heads=preset["nn"]["n_attention_heads"],
            num_transformer_encoder_layers=4
        )
        
        self.encoder_band2 = Transformer_Encoder(
            var_embedding_shape=var_embedding_shape,
            num_attention_heads=preset["nn"]["n_attention_heads"],
            num_transformer_encoder_layers=4
        )
        
        # Single decoder for concatenated features
        self.decoder = TransformerDecoder(
            d_model=preset["nn"]["d_embedding"],  # Doubled because of concatenation
            nhead=preset["nn"]["n_attention_heads"],
            num_decoder_layers=preset["nn"]["num_decoder_layers"],
            dim_feedforward=preset["nn"]["dim_FFN"],
            dropout=0.1,
            num_queries=preset["nn"]["num_obj_queries"],
            temp_cross_attention=preset["nn"]["cross_attention_temp"]
        )

    def forward(self, x_band1, x_band2):
        # Extract features for each band
        features_band1 = self.feature_extractor_band1(x_band1)  # Shape: [B, token_length, d_embedding]
        features_band2 = self.feature_extractor_band2(x_band2)  # Shape: [B, token_length, d_embedding]
        
        # Encode features for each band
        encoded_band1 = self.encoder_band1(features_band1)  # Shape: [B, token_length, d_embedding]
        encoded_band2 = self.encoder_band2(features_band2)  # Shape: [B, token_length, d_embedding]
        
        # Concatenate encoded features along the feature dimension
        combined_features = torch.cat((encoded_band1, encoded_band2), dim=-1)  # Shape: [B, token_length, d_embedding*2]
        
        # Decode the combined features
        outputs = self.decoder(combined_features)  # Shape: [num_layers, B, num_queries, num_classes]
        
        
        return outputs
def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device,
          var_mode: str,
          patience: int = 150):  # Added patience parameter

    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))

    # Initialize early stopping variables
    var_best_f1_score = 0
    var_best_PPP = 0
    var_best_weight = None
    counter = 0  # Counter for patience
    var_epoch_saved = 0

    if var_mode == "multi_head":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(data_train_loader),
            num_training_steps=preset["nn"]["epoch"] * len(data_train_loader),
            min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"]
        )

    def apply_augmentation(x_band1, x_band2):
        # Apply noise
        noise1 = torch.randn_like(x_band1) * 0.1
        noise2 = torch.randn_like(x_band2) * 0.1
        x_band1 = x_band1 + noise1
        x_band2 = x_band2 + noise2
        
        # Apply scaling
        scale1 = torch.rand(x_band1.size(0), 1, device=x_band1.device) * 0.2 + 0.9
        scale2 = torch.rand(x_band2.size(0), 1, device=x_band2.device) * 0.2 + 0.9
        x_band1 = x_band1 * scale1.unsqueeze(-1)
        x_band2 = x_band2 * scale2.unsqueeze(-1)
        
        # Apply masking
        mask1 = torch.bernoulli(torch.ones_like(x_band1) * 0.96)
        mask2 = torch.bernoulli(torch.ones_like(x_band2) * 0.96)
        x_band1 = x_band1 * mask1
        x_band2 = x_band2 * mask2

        return x_band1, x_band2

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        total_batches = len(data_train_loader)

        for batch_idx, data_batch in enumerate(data_train_loader):
            data_batch_x_band1, data_batch_x_band2, data_batch_y = data_batch
            data_batch_x_band1 = data_batch_x_band1.to(device)
            data_batch_x_band2 = data_batch_x_band2.to(device)
            data_batch_y = data_batch_y.to(device)

            if model.training:
                data_batch_x_band1, data_batch_x_band2 = apply_augmentation(data_batch_x_band1, data_batch_x_band2)

            predict_train_y = model(data_batch_x_band1, data_batch_x_band2)
            var_loss_train = loss(predict_train_y, data_batch_y.float())
            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()
            if var_mode == "multi_head":
                scheduler.step()

        data_batch_y = data_batch_y.detach().cpu().numpy()
        predict_train_y = predict_train_y.detach().cpu().numpy()

        dict_error_train = performance_metrics(data_batch_y.astype(int), predict_train_y,
                                               var_mode=var_mode, var_threshold=var_threshold)

        model.eval()
        with torch.no_grad():
            data_test_x_band1, data_test_x_band2, data_test_y = next(iter(data_test_loader))
            data_test_x_band1 = data_test_x_band1.to(device)
            data_test_x_band2 = data_test_x_band2.to(device)
            data_test_y = data_test_y.to(device)
            predict_test_y = model(data_test_x_band1, data_test_x_band2)
            var_loss_test = loss(predict_test_y, data_test_y.float())
            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()

            dict_error_test = performance_metrics(data_test_y, predict_test_y, var_mode, var_threshold)
            
            # Log attention weights every N batches
            if preset["model"] == "DETR" and var_epoch % 10 == 0:
                log_attention_weights(model, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), var_epoch)

        # Log metrics
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss_train.item(),
            "test_loss": var_loss_test.item(),
            "total_error_train": dict_error_train['total_error'],
            "total_error_test": dict_error_test['total_error'],
            "perfect_prediction_percentage_test": dict_error_test['perfect_prediction_percentage'],
            "perfect_prediction_percentage_train": dict_error_train['perfect_prediction_percentage'],
            "accuracy_test": dict_error_test['accuracy'],
            "accuracy_train": dict_error_train['accuracy'],
            "learning_rate": optimizer.param_groups[0]['lr'],
            "precision": dict_error_test['precision'],
            "recall": dict_error_test['recall'],
            "f1_score": dict_error_test['f1_score']
        })

        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss_train.cpu(),
              "- Test Loss %.6f" % var_loss_test.cpu(),
              "- Total Error %.6f" % dict_error_test['total_error'],
              "- Perfect Prediction Percentage Train %.6f" % dict_error_train['perfect_prediction_percentage'],
              "- Perfect Prediction Percentage Test %.6f" % dict_error_test['perfect_prediction_percentage'],
              "- Accuracy Test %.6f" % dict_error_test['accuracy'],
              "- Accuracy Train %.6f" % dict_error_train['accuracy'],
              "- Precision %.6f" % dict_error_test['precision'],
              "- Recall %.6f" % dict_error_test['recall'],
              "- F1 Score %.6f" % dict_error_test['f1_score'])

        if (dict_error_test['f1_score'] > var_best_f1_score and
                dict_error_test['perfect_prediction_percentage'] > var_best_PPP):
            var_best_PPP = dict_error_test['perfect_prediction_percentage']
            var_best_f1_score = dict_error_test['f1_score']
            var_best_weight = deepcopy(model.state_dict())
            var_epoch_saved = var_epoch
            counter = 0  # Reset counter
        else:
            counter += 1  # Increment counter

        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break
            
    print(f"Epoch that the model was saved {var_epoch_saved}")
    return var_best_weight


def run_dual_band(data_train_x_band1, data_train_y_band1,
                  data_test_x_band1, data_test_y_band1,
                  data_train_x_band2, data_train_y_band2,
                  data_test_x_band2, data_test_y_band2,
                  var_repeat=10):
    """
    [description]
    : run WiFi-based dual-band model using DETR architecture
    [parameter]
    : data_train_x_band1, data_train_x_band2: numpy array, CSI amplitude to train model for each band
    : data_train_y_band1, data_train_y_band2: numpy array, labels to train model for each band
    : data_test_x_band1, data_test_x_band2: numpy array, CSI amplitude to test model for each band
    : data_test_y_band1, data_test_y_band2: numpy array, labels to test model for each band
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    data_train_x_band1 = data_train_x_band1.reshape(data_train_x_band1.shape[0], data_train_x_band1.shape[1], -1)
    data_test_x_band1 = data_test_x_band1.reshape(data_test_x_band1.shape[0], data_test_x_band1.shape[1], -1)
    data_train_x_band2 = data_train_x_band2.reshape(data_train_x_band2.shape[0], data_train_x_band2.shape[1], -1)
    data_test_x_band2 = data_test_x_band2.reshape(data_test_x_band2.shape[0], data_test_x_band2.shape[1], -1)

    var_x_shape_band1, var_x_shape_band2 = data_train_x_band1[0].shape, data_train_x_band2[0].shape
    var_y_shape = data_train_y_band1[0].reshape(-1).shape

    data_train_set = TensorDataset(torch.from_numpy(data_train_x_band1),
                                   torch.from_numpy(data_train_x_band2),
                                   torch.from_numpy(data_train_y_band1))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x_band1),
                                  torch.from_numpy(data_test_x_band2),
                                  torch.from_numpy(data_test_y_band1))

    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []
    result_avg_count_error = []

    # Calculate model complexity
    # var_macs, var_params = get_model_complexity_info(DualBandDETR(var_x_shape_band1, var_x_shape_band2, var_y_shape),
    #                                                  (var_x_shape_band1, var_x_shape_band2), as_strings=False)
    # print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    for var_r in range(var_repeat):
        print("Repeat", var_r)
        var_mode = "multi_head"
        name_run = "Empty"
        if preset["pretrained_path"]:
            name_run = f"DualBandDETR_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + preset["transfer_scenario"]
        else:
            pretrained_state = "NPT"
            name_run = f"DualBandDETR_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + pretrained_state

        run = wandb.init(
            project="experiment_dual_band",
            name=name_run + preset["wandb_name"],
            config=preset,
            reinit=True
        )

        torch.random.manual_seed(var_r + 39)
        model_dual_band = torch.compile(DualBandDETR(var_x_shape_band1, var_x_shape_band2, var_y_shape).to(device))

        optimizer = torch.optim.Adam(model_dual_band.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])

        loss = HungarianMatchingLoss(
            cost_class_weight=preset["nn"]["loss"]["cost_class_weight"],
            aux_loss_weight=preset["nn"]["loss"]["aux_loss_weight"],
            label_smoothing=preset["nn"]["loss"]["label_smoothing"],
            class_imbalance_weight=preset["nn"]["loss"]["class_imbalance_weight"]
        )

        var_time_0 = time.time()

        var_best_weight = train(model=model_dual_band,
                                optimizer=optimizer,
                                loss=loss,
                                data_train_set=data_train_set,
                                data_test_set=data_test_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device,
                                var_mode=var_mode,)

        var_time_1 = time.time()

        # Test
        model_dual_band.load_state_dict(var_best_weight)
        model_dual_band.eval()

        with torch.no_grad():
            predict_test_y = model_dual_band(torch.from_numpy(data_test_x_band1).to(device),
                                             torch.from_numpy(data_test_x_band2).to(device))

        predict_test_y = predict_test_y.detach().cpu().numpy()

        var_time_2 = time.time()

        # Evaluate
        data_test_y_c = data_test_y_band1.reshape(-1, data_test_y_band1.shape[-1])
        predict_test_y_c = predict_test_y.reshape(-1, data_test_y_band1.shape[-1])

        # Calculate performance metrics
        dict_true_acc = performance_metrics(data_test_y_c, predict_test_y_c, var_mode=var_mode)

        # Log metrics to wandb
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
              "- perfect_prediction_percentage %.6f" % dict_true_acc['perfect_prediction_percentage'])

        # Store results
        result_accuracy.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_total_error.append(dict_true_acc['total_error'])
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])
        result_avg_count_error.append(dict_true_acc['mean_count_error'])

    # Log average metrics
    wandb.log({
        "avg_accuracy": sum(result_accuracy) / len(result_accuracy),
        "avg_train_time": sum(result_time_train) / len(result_time_train),
        "avg_test_time": sum(result_time_test) / len(result_time_test),
        "avg_total_error": sum(result_total_error) / len(result_total_error),
        "avg_precision": sum(result_precision) / len(result_precision),
        "avg_recall": sum(result_recall) / len(result_recall),
        "avg_f1_score": sum(result_f1_score) / len(result_f1_score),
        "avg_count_error": sum(result_avg_count_error) / len(result_avg_count_error),
    })

    # Visualize performance
    viz_stats = visualize_model_performance(
        y_pred=predict_test_y,
        y_true=data_test_y_band1,
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

    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    result["total_error"] = {"avg": np.mean(result_total_error), "std": np.std(result_total_error)}
    result["precision"] = {"avg": np.mean(result_precision), "std": np.std(result_precision)}
    result["recall"] = {"avg": np.mean(result_recall), "std": np.std(result_recall)}
    result["f1_score"] = {"avg": np.mean(result_f1_score), "std": np.std(result_f1_score)}
    result["count_error"] = {"avg": np.mean(result_avg_count_error), "std": np.std(result_avg_count_error)}

    return result

