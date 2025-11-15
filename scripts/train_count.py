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

    var_best_weight = None
    var_best_MCE = 100
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

            predict_train_y = model(data_batch_x)
            var_loss_train = loss(predict_train_y, data_batch_y.long())
            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()

        data_batch_y = data_batch_y.detach().cpu().numpy()
        predict_train_y = predict_train_y.detach().cpu().numpy()


        model.eval()
        with torch.no_grad():
            data_test_x, data_test_y = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device)
            predict_test_y = model(data_test_x)
            var_loss_test = loss(predict_test_y, data_test_y.long())

            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()

            MCE = sum( (data_test_y  != predict_test_y.argmax(axis=1)) )/ data_test_y.shape[0]

            MCE = sum( (np.abs(data_test_y  - predict_test_y.argmax(axis=1))) )/ data_test_y.shape[0]

        # Log metrics
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss_train.item(),
            "test_loss": var_loss_test.item(),
            "MSE": MCE
        })

        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss_train.cpu(),
              "- Test Loss %.6f" % var_loss_test.cpu(),
              "MSE %.9f"%MCE,
              )

        if MCE < var_best_MCE:
            var_best_MCE = MCE
            var_best_weight = deepcopy(model.state_dict())
            var_epoch_saved = var_epoch

    print(f"Epoch that the model was saved {var_epoch_saved}")
    return var_best_weight

