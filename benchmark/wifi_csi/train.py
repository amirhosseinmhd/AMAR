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
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device,
          var_mode: str,
          patience: int = 150):  # Added patience parameter
    var_epoch_saved = 0
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))

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
        scale = torch.rand(x_batch.size(0), 1, device=x_batch.device) * 0.2 + 0.9
        x_batch = x_batch * scale.unsqueeze(-1)
        mask = torch.bernoulli(torch.ones_like(x_batch) * 0.96)
        x_batch = x_batch * mask

        return x_batch

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()
        model.train()
        total_batches = len(data_train_loader)

        for batch_idx, data_batch in enumerate(data_train_loader):
            data_batch_x, data_batch_y = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device)

            if model.training:
                data_batch_x = apply_augmentation(data_batch_x)

            if var_mode == "count_classification":
                data_batch_y = data_batch_y.sum(axis=1)
            if var_mode == "baseline":
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
            data_test_x, data_test_y = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device)
            if var_mode == "count_classification":
                data_test_y = data_test_y.sum(axis=1)
            if var_mode == "baseline":
                data_test_y = data_test_y.reshape(data_test_y.shape[0], -1)

            predict_test_y = model(data_test_x)
            var_loss_test = loss(predict_test_y, data_test_y.float())

            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()

            dict_error_test = performance_metrics(data_test_y, predict_test_y, var_mode, var_threshold)
        #     # Log attention weights every N batches
        # if preset["model"] == "DETR"  and var_epoch % 40 == 0:
        #     log_attention_weights(model, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), var_epoch)
        if preset["model"] == "DETR":
            print("*****************")
            print("*****************")
            layers_idxs =preset["layers_idxs"]
            for layer_idx in layers_idxs:
                layer_metrics = dict_error_test[layer_idx]
                layer_train_metrics = dict_error_train[layer_idx]
                wandb.log({
                    f"{layer_idx}/epoch": var_epoch,
                    f"{layer_idx}/train_loss": var_loss_train.item(),
                    f"{layer_idx}/test_loss": var_loss_test.item(),
                    f"{layer_idx}/total_error_train": layer_train_metrics['total_error'],
                    f"{layer_idx}/total_error_test": layer_metrics['total_error'],
                    f"{layer_idx}/perfect_prediction_percentage_test": layer_metrics['perfect_prediction_percentage'],
                    f"{layer_idx}/perfect_prediction_percentage_train": layer_train_metrics[
                        'perfect_prediction_percentage'],
                    f"{layer_idx}/accuracy_test": layer_metrics['accuracy'],
                    f"{layer_idx}/accuracy_train": layer_train_metrics['accuracy'],
                    f"{layer_idx}/precision": layer_metrics['precision'],
                    f"{layer_idx}/recall": layer_metrics['recall'],
                    f"{layer_idx}/f1_score": layer_metrics['f1_score'],
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, 
                step=var_epoch)

                print(f"Layer {layer_idx} - Epoch {var_epoch}/{var_epochs}",
                      "- %.6fs" % (time.time() - var_time_e0),
                      f"- Loss Train %.6f" % var_loss_train.cpu(),
                      f"- Loss Test %.6f" % var_loss_test.cpu(),
                      f"- Total Error Train %.6f" % layer_train_metrics['total_error'],
                      f"- Total Error Test %.6f" % layer_metrics['total_error'],
                      f"- Perfect Prediction Percentage Train %.6f" % layer_train_metrics[
                          'perfect_prediction_percentage'],
                      f"- Perfect Prediction Percentage Test %.6f" % layer_metrics['perfect_prediction_percentage'],
                      f"- Accuracy Train %.6f" % layer_train_metrics['accuracy'],
                      f"- Accuracy Test %.6f" % layer_metrics['accuracy'],
                      f"- Precision %.6f" % layer_metrics['precision'],
                      f"- Recall %.6f" % layer_metrics['recall'],
                      f"- F1 Score %.6f" % layer_metrics['f1_score'])
        else:
            # Original logging for non-multi_head modes
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
            }, 
            step=var_epoch)
            print(f"Epoch {var_epoch}/{var_epochs}",
                  "- %.6fs" % (time.time() - var_time_e0),
                  "- Loss Train %.6f" % var_loss_train.cpu(),
                  "- Loss Test %.6f" % var_loss_test.cpu(),
                  "- Total Error Train %.6f" % dict_error_train['total_error'],
                  "- Total Error Test %.6f" % dict_error_test['total_error'],
                  "- Perfect Prediction Percentage Train %.6f" % dict_error_train['perfect_prediction_percentage'],
                  "- Perfect Prediction Percentage Test %.6f" % dict_error_test['perfect_prediction_percentage'],
                  "- Accuracy Train %.6f" % dict_error_train['accuracy'],
                  "- Accuracy Test %.6f" % dict_error_test['accuracy'],
                  "- Precision %.6f" % dict_error_test['precision'],
                  "- Recall %.6f" % dict_error_test['recall'],
                  "- F1 Score %.6f" % dict_error_test['f1_score'])

        if (var_epoch > 0 and dict_error_test['perfect_prediction_percentage'] > var_best_PPP):
            var_best_PPP = dict_error_test['perfect_prediction_percentage']

            var_best_f1_score = dict_error_test['f1_score']
            var_best_weight = deepcopy(model.state_dict())
            var_epoch_saved = var_epoch

    print(f"Epoch that the model was saved {var_epoch_saved}")
    return var_best_weight

