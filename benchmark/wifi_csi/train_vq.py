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
# torch.set_float32_matmul_precision("high")
# torch._dynamo.config.cache_size_limit = 65536
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
    g = torch.Generator()
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True, generator=g)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))

    # Initialize early stopping variables 
    var_best_PPP = 0
    var_best_weight = deepcopy(model.state_dict())

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

            outputs_class, continuous_emb, quantized_emb, indices = model(data_batch_x)
            classification_loss = loss(outputs_class, data_batch_y.float())
            vq_loss = model.compute_vq_loss(continuous_emb, quantized_emb)
            total_loss = classification_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        data_batch_y = data_batch_y.detach().cpu().numpy()
        predict_train_y = outputs_class.detach().cpu().numpy()

        dict_error_train = performance_metrics(data_batch_y.astype(int), predict_train_y,
                                               var_mode=var_mode, var_threshold=var_threshold)

        model.eval()
        with torch.no_grad():
            data_test_x, data_test_y = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device)
            predict_test_y, _, _, indices = model(data_test_x)
            var_loss_test = loss(predict_test_y, data_test_y.float())
            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()
            dict_error_test = performance_metrics(data_test_y, predict_test_y, var_mode, var_threshold)
        #     # Log attention weights every N batches
        if var_epoch % 40 == 0:
            log_attention_weights(model, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), var_epoch)

        layers_idxs = ["layer_" +str(preset["nn"]["num_decoder_layers"] - 1)]
        for layer_idx in layers_idxs:
            layer_metrics = dict_error_test[layer_idx]
            layer_train_metrics = dict_error_train[layer_idx]
            wandb.log({
                f"{layer_idx}/epoch": var_epoch,
                f"{layer_idx}/train_loss": total_loss.item(),
                f"{layer_idx}/classification_loss": classification_loss.item(),
                f"{layer_idx}/vq_loss": vq_loss.item(),
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

            if var_epoch % 10 == 0:
                print(f"--- Layer {layer_idx} - Epoch {var_epoch}/{var_epochs} ---")
                print(f"  Time: {time.time() - var_time_e0:.6f}s")
                print(f"  Total Loss: {total_loss.cpu():.6f} | Class Loss: {classification_loss.cpu():.6f} | VQ Loss: {vq_loss.cpu():.6f} | Test Loss: {var_loss_test.cpu():.6f}")
                print(f"  Total Error Train: {layer_train_metrics['total_error']:.6f} | Total Error Test: {layer_metrics['total_error']:.6f}")
                print(f"  Perfect Prediction % Train: {layer_train_metrics['perfect_prediction_percentage']:.6f} | Perfect Prediction % Test: {layer_metrics['perfect_prediction_percentage']:.6f}")
                print(f"  Accuracy Train: {layer_train_metrics['accuracy']:.6f} | Accuracy Test: {layer_metrics['accuracy']:.6f}")
                print(f"  Precision: {layer_metrics['precision']:.6f} | Recall: {layer_metrics['recall']:.6f} | F1 Score: {layer_metrics['f1_score']:.6f}")
                print("-" * 30)

        if (var_epoch > 0 and dict_error_test['perfect_prediction_percentage'] > var_best_PPP):
            var_best_PPP = dict_error_test['perfect_prediction_percentage']

            var_best_f1_score = dict_error_test['f1_score']
            var_best_weight = deepcopy(model.state_dict())
            var_epoch_saved = var_epoch

        if var_epoch % 5 == 0:
            with torch.no_grad():
                # VQ statistics
                codebook_indices = indices.cpu().numpy().flatten()

                # 1. Codebook Usage
                unique_indices, counts = np.unique(codebook_indices, return_counts=True)
                codebook_usage = {f"code_{i}": count for i, count in zip(unique_indices, counts)}

                # 2. Percentage of Codebook Used
                percent_codebook_used = (len(unique_indices) / model.vq_layer.num_embeddings) * 100

                # 3. Representation Diversity (Per Sample)
                unique_symbols_per_sample = [len(np.unique(s)) for s in indices.cpu().numpy()]
                avg_unique_symbols = np.mean(unique_symbols_per_sample)

                # 4. Codebook Usage Rank (Top 5)
                sorted_usage = sorted(zip(unique_indices, counts), key=lambda x: x[1], reverse=True)
                top_5_symbols = {f"rank_{i+1}": {"index": int(idx), "count": int(cnt)} for i, (idx, cnt) in enumerate(sorted_usage[:5])}

                # Create a wandb.Table for the top 5 symbols
                top_5_table = wandb.Table(columns=["Rank", "Symbol Index", "Count"])
                for i, (idx, cnt) in enumerate(sorted_usage[:5]):
                    top_5_table.add_data(f"Rank {i+1}", int(idx), int(cnt))

                # Logging to wandb
                wandb.log({
                    "vq_stats/percent_codebook_used": percent_codebook_used,
                    "vq_stats/avg_unique_symbols_per_sample": avg_unique_symbols,
                    "vq_stats/codebook_usage": wandb.plot.bar(
                        wandb.Table(columns=["symbol_index", "count"], rows=list(zip(unique_indices, counts))),
                        "symbol_index", "count", title="Codebook Usage"
                    ),
                    "vq_stats/top_5_used_symbols": top_5_table
                }, step=var_epoch)

                # Printing to console
                print(f"--- VQ Stats at Epoch {var_epoch} ---")
                print(f"  Percentage of Codebook Used: {percent_codebook_used:.2f}%")
                print(f"  Average Unique Symbols per Sample: {avg_unique_symbols:.2f}")
                print(f"  Top 5 Most Used Symbols: {top_5_symbols}")
                print("-" * 30)

    print(f"Epoch that the model was saved {var_epoch_saved}")
    return var_best_weight

