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
from src.utils import *
from torch.optim.lr_scheduler import LambdaLR
import math
from configs.preset import preset

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

    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))

    # Initialize early stopping variables
    var_best_f1_score_act = 0
    var_best_PPP_act = 0
    var_best_f1_score_loc = 0
    var_best_PPP_loc = 0
    var_best_weight = None
    counter = 0  # Counter for patience

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

            data_batch_x, data_batch_y_act, data_batch_y_loc = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y_act = data_batch_y_act.to(device)
            data_batch_y_loc = data_batch_y_loc.to(device)

            if model.training:
                data_batch_x = apply_augmentation(data_batch_x)

            predict_train_y_act, predict_train_y_loc = model(data_batch_x)

            var_loss_train = loss(predict_train_y_act, data_batch_y_act.float(),
                                  predict_train_y_loc,  data_batch_y_loc.float())

            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()
            scheduler.step()

        data_batch_y_act = data_batch_y_act.detach().cpu().numpy()
        data_batch_y_loc = data_batch_y_loc.detach().cpu().numpy()

        predict_train_y_act = predict_train_y_act.detach().cpu().numpy()
        predict_train_y_loc = predict_train_y_loc.detach().cpu().numpy()

        # Calculate performance metrics for training
        dict_error_train_act, dict_error_train_loc = performance_metrics_joint(
            y_true_act=data_batch_y_act,
            y_pred_act=predict_train_y_act,
            y_true_loc=data_batch_y_loc,
            y_pred_loc=predict_train_y_loc,
        )
        # dict_error_train_act = performance_metrics(data_batch_y_act, predict_train_y_act,
        #                                            var_mode=var_mode, var_threshold=var_threshold)
        # dict_error_train_loc = performance_metrics(data_batch_y_loc, predict_train_y_loc,
        #                                            var_mode=var_mode, var_threshold=var_threshold)

        model.eval()
        with torch.no_grad():
            data_test_x, data_test_y_act, data_test_y_loc = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y_act = data_test_y_act.to(device)
            data_test_y_loc = data_test_y_loc.to(device)
            predict_test_y_act, predict_test_y_loc = model(data_test_x)
            var_loss_test = loss(predict_test_y_act, data_test_y_act.float(),
                                 predict_test_y_loc, data_test_y_loc.float())

            # Convert to numpy for metrics calculation
            data_test_y_act = data_test_y_act.detach().cpu().numpy()
            data_test_y_loc = data_test_y_loc.detach().cpu().numpy()
            predict_test_y_act = predict_test_y_act.detach().cpu().numpy()
            predict_test_y_loc = predict_test_y_loc.detach().cpu().numpy()

            # # Calculate performance metrics for both activity and location
            # dict_error_test_act = performance_metrics(data_test_y_act, predict_test_y_act,
            #                                           var_mode, var_threshold)
            # dict_error_test_loc = performance_metrics(data_test_y_loc, predict_test_y_loc,
            #                                           var_mode, var_threshold)

            dict_error_test_act, dict_error_test_loc = performance_metrics_joint(
                y_true_act=data_test_y_act,
                y_pred_act=predict_test_y_act,
                y_true_loc=data_test_y_loc,
                y_pred_loc=predict_test_y_loc
            )
        # Log metrics for both activity and location
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss_train.item(),
            "test_loss": var_loss_test.item(),

            # Activity metrics
            "ACT_total_error_train": dict_error_train_act['total_error'],
            "ACT_total_error_test": dict_error_test_act['total_error'],
            "ACT_perfect_prediction_percentage_test": dict_error_test_act['perfect_prediction_percentage'],
            "ACT_perfect_prediction_percentage_train": dict_error_train_act['perfect_prediction_percentage'],
            "ACT_accuracy_test": dict_error_test_act['accuracy'],
            "ACT_accuracy_train": dict_error_train_act['accuracy'],
            "ACT_precision": dict_error_test_act['precision'],
            "ACT_recall": dict_error_test_act['recall'],
            "ACT_f1_score": dict_error_test_act['f1_score'],

            # Location metrics
            "LOC_total_error_train": dict_error_train_loc['total_error'],
            "LOC_total_error_test": dict_error_test_loc['total_error'],
            "LOC_perfect_prediction_percentage_test": dict_error_test_loc['perfect_prediction_percentage'],
            "LOC_perfect_prediction_percentage_train": dict_error_train_loc['perfect_prediction_percentage'],
            "LOC_accuracy_test": dict_error_test_loc['accuracy'],
            "LOC_accuracy_train": dict_error_train_loc['accuracy'],
            "LOC_precision": dict_error_test_loc['precision'],
            "LOC_recall": dict_error_test_loc['recall'],
            "LOC_f1_score": dict_error_test_loc['f1_score'],

            # Other metrics
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Print training progress
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss_train.cpu(),
              "- Test Loss %.6f" % var_loss_test.cpu())

        print("ACTIVITY:",
              "- Total Error Train %.6f" % dict_error_train_act['total_error'],
              "- Total Error Test %.6f" % dict_error_test_act['total_error'],
              "- PPP Train %.6f" % dict_error_train_act['perfect_prediction_percentage'],
              "- PPP Test %.6f" % dict_error_test_act['perfect_prediction_percentage'],
              "- Acc Train %.6f" % dict_error_train_act['accuracy'],
              "- Acc Test %.6f" % dict_error_test_act['accuracy'],
              "- Precision %.6f" % dict_error_test_act['precision'],
              "- Recall %.6f" % dict_error_test_act['recall'],
              "- F1 Score %.6f" % dict_error_test_act['f1_score'])

        print("LOCATION:",
              "- Total Error Train %.6f" % dict_error_train_loc['total_error'],
              "- Total Error Test %.6f" % dict_error_test_loc['total_error'],
              "- PPP Train %.6f" % dict_error_train_loc['perfect_prediction_percentage'],
              "- PPP Test %.6f" % dict_error_test_loc['perfect_prediction_percentage'],
              "- Acc Train %.6f" % dict_error_train_loc['accuracy'],
              "- Acc Test %.6f" % dict_error_test_loc['accuracy'],
              "- Precision %.6f" % dict_error_test_loc['precision'],
              "- Recall %.6f" % dict_error_test_loc['recall'],
              "- F1 Score %.6f" % dict_error_test_loc['f1_score'])

        # # Early stopping check - consider both activity and location performance
        # if ((dict_error_test_act['f1_score'] > var_best_f1_score_act and
        #      dict_error_test_act['perfect_prediction_percentage'] > var_best_PPP_act) or
        #         (dict_error_test_loc['f1_score'] > var_best_f1_score_loc and
        #          dict_error_test_loc['perfect_prediction_percentage'] > var_best_PPP_loc)):
        #
        #     # Update best scores
        #     var_best_PPP_act = max(var_best_PPP_act, dict_error_test_act['perfect_prediction_percentage'])
        #     var_best_f1_score_act = max(var_best_f1_score_act, dict_error_test_act['f1_score'])
        #     var_best_PPP_loc = max(var_best_PPP_loc, dict_error_test_loc['perfect_prediction_percentage'])
        #     var_best_f1_score_loc = max(var_best_f1_score_loc, dict_error_test_loc['f1_score'])
        #
        #     var_best_weight = deepcopy(model.state_dict())
        #     var_epoch_saved = var_epoch
        #     counter = 0  # Reset counter
        # else:
        #     counter += 1  # Increment counter
        if (dict_error_test_act['f1_score'] > var_best_f1_score_act and
                dict_error_test_act['perfect_prediction_percentage'] > var_best_PPP_act):

            # Update best scores
            var_best_PPP_act = dict_error_test_act['perfect_prediction_percentage']
            var_best_f1_score_act = dict_error_test_act['f1_score']

            # Still track location metrics, but don't use them for model selection
            var_best_PPP_loc = dict_error_test_loc['perfect_prediction_percentage']
            var_best_f1_score_loc = dict_error_test_loc['f1_score']

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
    print(f"Best activity metrics - F1: {var_best_f1_score_act:.6f}, PPP: {var_best_PPP_act:.6f}")
    print(f"Best location metrics - F1: {var_best_f1_score_loc:.6f}, PPP: {var_best_PPP_loc:.6f}")

    return var_best_weight
