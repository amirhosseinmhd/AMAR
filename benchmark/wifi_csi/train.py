"""
[file]          train.py
[description]   function to train WiFi-based models
"""
#
##
import time
import torch
import torch._dynamo
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score
import wandb
from utils import *
from torch.optim.lr_scheduler import LambdaLR
import math
from preset import preset

#
##
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
#
##
def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: TensorDataset,
          data_valid_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device,
          var_mode: str,
          patience: int = 150):  # Added patience parameter
    var_epoch_saved = 0
    g = torch.Generator()
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True, generator=g)
    data_valid_loader = DataLoader(data_valid_set, len(data_valid_set))

    # Initialize early stopping variables 
    var_best_f1_score = 0
    var_best_PPP = 0
    var_best_weight = deepcopy(model.state_dict())
    counter = 0  # Counter for patience

    if var_mode == "multi_head":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(data_train_loader),
            num_training_steps=preset["nn"]["epoch"] * len(data_train_loader),
            min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"]
        )

    def apply_augmentation(x_batch):
        noise = torch.randn_like(x_batch) * 0.1
        x_batch = x_batch + noise
        scale = torch.rand(x_batch.size(0), 1, device=x_batch.device) * 0.2 + 1
        x_batch = x_batch * scale.unsqueeze(-1)
        mask = torch.bernoulli(torch.ones_like(x_batch) * 0.96)
        x_batch = x_batch * mask

        return x_batch

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        total_batches = len(data_train_loader)

        # Adjust learning rate for feature_extractor after 50 epochs
        if var_epoch == 50 and not preset.get("pretrained_path"):
            for param_group in optimizer.param_groups:
                if param_group.get('name') == 'feature_extractor':
                    param_group['lr'] /= 5
                    print(f"Reduced LR for feature_extractor to {param_group['lr']} at epoch {var_epoch}")

        for batch_idx, data_batch in enumerate(data_train_loader):
            data_batch_x, data_batch_y = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device)

            if model.training:
                data_batch_x = apply_augmentation(data_batch_x)

            if var_mode == "count_classification":
                data_batch_y = data_batch_y.sum(axis=1)
            elif var_mode == "baseline":
                data_batch_y = data_batch_y.reshape(data_batch_y.shape[0], -1)

            predict_train_y = model(data_batch_x)
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
            data_valid_x, data_valid_y = next(iter(data_valid_loader))
            data_valid_x = data_valid_x.to(device)
            data_valid_y = data_valid_y.to(device)
            if var_mode == "count_classification":
                data_valid_y = data_valid_y.sum(axis=1)
            if var_mode == "baseline":
                data_valid_y = data_valid_y.reshape(data_valid_y.shape[0], -1)

            predict_valid_y = model(data_valid_x)
            var_loss_valid = loss(predict_valid_y, data_valid_y.float())

            data_valid_y = data_valid_y.detach().cpu().numpy()
            predict_valid_y = predict_valid_y.detach().cpu().numpy()

            dict_error_valid = performance_metrics(data_valid_y, predict_valid_y, var_mode, var_threshold)
        #     # Log attention weights every N batches
        # if preset["model"] == "AMAR"  and var_epoch % 40 == 0:
        #     log_attention_weights(model, np.argmax(predict_valid_y[-1], axis=-1), np.argmax(data_valid_y, axis=-1), var_epoch)
        if preset["model"] == "AMAR":
            layers_idxs = ["layer_" +str(preset["nn"]["num_decoder_layers"] - 1)]
            for layer_idx in layers_idxs:
                layer_metrics = dict_error_valid[layer_idx]
                layer_train_metrics = dict_error_train[layer_idx]
                wandb.log({
                    f"{layer_idx}/epoch": var_epoch,
                    f"{layer_idx}/train_loss": var_loss_train.item(),
                    f"{layer_idx}/valid_loss": var_loss_valid.item(),
                    f"{layer_idx}/total_error_train": layer_train_metrics['total_error'],
                    f"{layer_idx}/total_error_valid": layer_metrics['total_error'],
                    f"{layer_idx}/perfect_prediction_percentage_valid": layer_metrics['perfect_prediction_percentage'],
                    f"{layer_idx}/perfect_prediction_percentage_train": layer_train_metrics[
                        'perfect_prediction_percentage'],
                    f"{layer_idx}/accuracy_valid": layer_metrics['accuracy'],
                    f"{layer_idx}/accuracy_train": layer_train_metrics['accuracy'],
                    f"{layer_idx}/precision": layer_metrics['precision'],
                    f"{layer_idx}/recall": layer_metrics['recall'],
                    f"{layer_idx}/f1_score": layer_metrics['f1_score'],
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, 
                step=var_epoch)

                if var_epoch % 10 == 0:
                    print(f"--- Layer {layer_idx} - Epoch {var_epoch}/{var_epochs} ---")
                    print(f"  Time: {time.time() - var_time_e0:.6f}s")
                    print(f"  Loss Train: {var_loss_train.cpu():.6f} | Loss valid: {var_loss_valid.cpu():.6f}")
                    print(f"  Total Error Train: {layer_train_metrics['total_error']:.6f} | Total Error valid: {layer_metrics['total_error']:.6f}")
                    print(f"  Perfect Prediction % Train: {layer_train_metrics['perfect_prediction_percentage']:.6f} | Perfect Prediction % valid: {layer_metrics['perfect_prediction_percentage']:.6f}")
                    print(f"  Accuracy Train: {layer_train_metrics['accuracy']:.6f} | Accuracy valid: {layer_metrics['accuracy']:.6f}")
                    print(f"  Precision: {layer_metrics['precision']:.6f} | Recall: {layer_metrics['recall']:.6f} | F1 Score: {layer_metrics['f1_score']:.6f}")
                    print("-" * 30)
        else:
            # Original logging for non-multi_head modes
            wandb.log({
                "epoch": var_epoch,
                "train_loss": var_loss_train.item(),
                "valid_loss": var_loss_valid.item(),
                "total_error_train": dict_error_train['total_error'],
                "total_error_valid": dict_error_valid['total_error'],
                "perfect_prediction_percentage_valid": dict_error_valid['perfect_prediction_percentage'],
                "perfect_prediction_percentage_train": dict_error_train['perfect_prediction_percentage'],
                "accuracy_valid": dict_error_valid['accuracy'],
                "accuracy_train": dict_error_train['accuracy'],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "precision": dict_error_valid['precision'],
                "recall": dict_error_valid['recall'],
                "f1_score": dict_error_valid['f1_score']
            }, 
            step=var_epoch)
            if var_epoch % 10 == 0:
                print(f"--- Epoch {var_epoch}/{var_epochs} ---")
                print(f"  Time: {time.time() - var_time_e0:.6f}s")
                print(f"  Loss Train: {var_loss_train.cpu():.6f} | Loss valid: {var_loss_valid.cpu():.6f}")
                print(f"  Total Error Train: {dict_error_train['total_error']:.6f} | Total Error valid: {dict_error_valid['total_error']:.6f}")
                print(f"  Perfect Prediction % Train: {dict_error_train['perfect_prediction_percentage']:.6f} | Perfect Prediction % valid: {dict_error_valid['perfect_prediction_percentage']:.6f}")
                print(f"  Accuracy Train: {dict_error_train['accuracy']:.6f} | Accuracy valid: {dict_error_valid['accuracy']:.6f}")
                print(f"  Precision: {dict_error_valid['precision']:.6f} | Recall: {dict_error_valid['recall']:.6f} | F1 Score: {dict_error_valid['f1_score']:.6f}")
                print("-" * 30)

        if (var_epoch > 0 and dict_error_valid['perfect_prediction_percentage'] > var_best_PPP):
            var_best_PPP = dict_error_valid['perfect_prediction_percentage']

            var_best_f1_score = dict_error_valid['f1_score']
            var_best_weight = deepcopy(model.state_dict())
            var_epoch_saved = var_epoch

    print(f"Epoch that the model was saved {var_epoch_saved}")
    return var_best_weight

