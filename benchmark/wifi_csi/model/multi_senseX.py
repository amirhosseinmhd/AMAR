
import time
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
import copy
from model.that import THAT
from utils import *
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from preset import preset
import torch.nn.functional as F


torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

class MultiSenseX(torch.nn.Module):
    def __init__(self,
                 var_x_shape,
                 embedding_dim=100,
                 threshold=0.5):

        super().__init__()
        self.backbone = THAT(var_x_shape, [embedding_dim])


        # Location head (multilabel classification)
        self.location_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 5)
        )


        # Activity heads (one per location)
        self.act_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 9)
            ) for _ in range(5)
        ])

        self.sigmoid = torch.nn.Sigmoid()
        self.threshold = threshold  # For activating locations

    def forward(self, x):
        """
        Args:
            z: Backbone features of shape (batch_size, var_dim_in)
        Returns:
            loc_pred: Location probabilities (batch_size, 5)
            act_preds: Dictionary of activity logits for active locations
        """
        z = self.backbone(x)

        # --- Location Prediction ---
        logit_loc = self.location_head(z)
        loc_pred = self.sigmoid(logit_loc)  # Shape: (batch_size, 5)

        # --- Mask for Active Locations ---
        mask = loc_pred > self.threshold  # (batch_size, 5)

        # --- Activity Prediction for Active Locations ---

        act_logits = torch.stack([head(z) for head in self.act_heads], dim=1) #(batch_size, 5, 9)
        # act_logits = {}
        #
        # for loc_idx in range(5):
        #     # Check if ANY sample in the batch has this location active
        #     if mask[:, loc_idx].any():
        #         # Compute activity logits for this location
        #         logits = self.act_heads[loc_idx](z)  # (batch_size, 9)
        #         act_logits[loc_idx] = logits * mask[:, loc_idx].unsqueeze(-1).float()

        return act_logits, loc_pred, mask

class JointActLocDataset(torch.utils.data.Dataset):
    def __init__(self, CSI, y_act, y_loc):
        self.X = CSI
        self.y_loc = y_loc
        self.y_act = y_act

    def __len__(self)  -> int :
        return self.X.shape[0]
    def __getitem__(self, idx) -> tuple:
        return self.X[idx], self.y_act[idx], self.y_loc[idx]

#
##
def run_multi_senseX(data_train_x,
                    data_train_y_loc,
                    data_train_y_act,
                    data_test_x,
                    data_test_y_loc,
                    data_test_y_act,
                     var_repeat=10):
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
    (data_valid_x, data_test_x,
     data_valid_y_loc, data_test_y_loc,
     data_valid_y_act, data_test_y_act) = my_train_test_split(data_test_x, data_test_y_loc, data_test_y_act,
                                                              test_size=0.5, random_state=103)

    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape_loc, var_y_shape_act = data_train_x[0].shape, data_train_y_loc.shape[
                                                                           1:], data_train_y_act.shape[1:]
    #
    data_train_set = JointActLocDataset(data_train_x, data_train_y_act, data_train_y_loc)

    data_valid_set = JointActLocDataset(data_valid_x, data_valid_y_act, data_valid_y_loc)

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    #
    ##
    result_ppp_act = []
    result_total_error_act = []
    result_precision_act = []
    result_recall_act = []
    result_f1_score_act = []
    result_avg_count_error_act = []

    # Store location results
    result_ppp_loc = []
    result_total_error_loc = []
    result_precision_loc = []
    result_recall_loc = []
    result_f1_score_loc = []
    result_avg_count_error_loc = []

    # Store timing results
    result_time_train = []
    result_time_test = []

    #
    var_macs, var_params = get_model_complexity_info(MultiSenseX(var_x_shape),
                                                     var_x_shape, as_strings=False)
    #
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    ##
    for var_r in range(var_repeat):
        #
        ##
        print("Repeat", var_r)
        name_run = f"MultiSenseX{var_r}_" + "_".join(preset["data"]["environment"])

        run = wandb.init(
            project="multiSenseX",
            name= name_run,
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_multiSenseX = MultiSenseX(var_x_shape,
                 embedding_dim=100,
                 threshold=0.5).to(device)
        #

        optimizer = torch.optim.Adam(model_multiSenseX.parameters(),
                                         lr=preset["nn"]["lr"],
                                         weight_decay=preset["nn"]["weight_decay"])

        #
        loss_mode = "multi_senseX"
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model = model_multiSenseX,
                                optimizer = optimizer,
                                data_train_set = data_train_set,
                                data_test_set = data_valid_set,
                                var_threshold = preset["nn"]["threshold"],
                                var_batch_size = preset["nn"]["batch_size"],
                                var_epochs = preset["nn"]["epoch"],
                                device = device,
                                var_mode = loss_mode)
        #
        var_time_1 = time.time()

        ##
        ## ---------------------------------------- Test ------------------------------------------
        #
        model_multiSenseX.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y_act, predict_test_y_loc, mask = model_multiSenseX(torch.from_numpy(data_test_x).to(device))
        #
        # predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()
        predict_test_act = predict_test_y_act.detach().cpu().numpy()
        predict_test_loc = predict_test_y_loc.detach().cpu().numpy()

        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        dict_true_acc_act, dict_true_acc_loc = performance_metrics_joint_multiSensX(data_test_y_act, predict_test_act,
                                                                         data_test_y_loc, predict_test_loc)

        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,

            # Activity metrics
            "ACT_TOTAL_TESTSET_ERROR": dict_true_acc_act['total_error'],
            "ACT_TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc_act['perfect_prediction_percentage'],
            "ACT_TOTAL_ACCURACY": dict_true_acc_act['accuracy'],
            "ACT_mean_count_error": dict_true_acc_act['mean_count_error'],
            "ACT_error_per_person_1": dict_true_acc_act['error_per_person'][0],
            "ACT_error_per_person_2": dict_true_acc_act['error_per_person'][1],
            "ACT_error_per_person_3": dict_true_acc_act['error_per_person'][2],
            "ACT_error_per_person_4": dict_true_acc_act['error_per_person'][3],
            "ACT_error_per_person_5": dict_true_acc_act['error_per_person'][4],
            "ACT_precision": dict_true_acc_act['precision'],
            "ACT_recall": dict_true_acc_act['recall'],
            "ACT_f1_score": dict_true_acc_act['f1_score'],

            # Location metrics
            "LOC_TOTAL_TESTSET_ERROR": dict_true_acc_loc['total_error'],
            "LOC_TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc_loc['perfect_prediction_percentage'],
            "LOC_TOTAL_ACCURACY": dict_true_acc_loc['accuracy'],
            "LOC_mean_count_error": dict_true_acc_loc['mean_count_error'],
            "LOC_error_per_person_1": dict_true_acc_loc['error_per_person'][0],
            "LOC_error_per_person_2": dict_true_acc_loc['error_per_person'][1],
            "LOC_error_per_person_3": dict_true_acc_loc['error_per_person'][2],
            "LOC_error_per_person_4": dict_true_acc_loc['error_per_person'][3],
            "LOC_error_per_person_5": dict_true_acc_loc['error_per_person'][4],
            "LOC_precision": dict_true_acc_loc['precision'],
            "LOC_recall": dict_true_acc_loc['recall'],
            "LOC_f1_score": dict_true_acc_loc['f1_score']
        })
        #
        #

        #
        result_ppp_act.append(dict_true_acc_act['perfect_prediction_percentage'])
        result_total_error_act.append(dict_true_acc_act['total_error'])
        result_precision_act.append(dict_true_acc_act['precision'])
        result_recall_act.append(dict_true_acc_act['recall'])
        result_f1_score_act.append(dict_true_acc_act['f1_score'])
        result_avg_count_error_act.append(dict_true_acc_act['mean_count_error'])

        result_ppp_loc.append(dict_true_acc_loc['perfect_prediction_percentage'])
        result_total_error_loc.append(dict_true_acc_loc['total_error'])
        result_precision_loc.append(dict_true_acc_loc['precision'])
        result_recall_loc.append(dict_true_acc_loc['recall'])
        result_f1_score_loc.append(dict_true_acc_loc['f1_score'])
        result_avg_count_error_loc.append(dict_true_acc_loc['mean_count_error'])

    wandb.log({
        # Activity averages
        "ACT_avg_accuracy": sum(result_ppp_act) / len(result_ppp_act),
        "ACT_avg_total_error": sum(result_total_error_act) / len(result_total_error_act),
        "ACT_avg_precision": sum(result_precision_act) / len(result_precision_act),
        "ACT_avg_recall": sum(result_recall_act) / len(result_recall_act),
        "ACT_avg_f1_score": sum(result_f1_score_act) / len(result_f1_score_act),
        "ACT_avg_count_error": sum(result_avg_count_error_act) / len(result_avg_count_error_act),

        # Location averages
        "LOC_avg_accuracy": sum(result_ppp_loc) / len(result_ppp_loc),
        "LOC_avg_total_error": sum(result_total_error_loc) / len(result_total_error_loc),
        "LOC_avg_precision": sum(result_precision_loc) / len(result_precision_loc),
        "LOC_avg_recall": sum(result_recall_loc) / len(result_recall_loc),
        "LOC_avg_f1_score": sum(result_f1_score_loc) / len(result_f1_score_loc),
        "LOC_avg_count_error": sum(result_avg_count_error_loc) / len(result_avg_count_error_loc),
    })

    # viz_stats = visualize_model_performance(
    #     y_pred=predict_test_y,
    #     y_true=data_test_y_act,
    #     var_mode=var_mode,
    #     save_dir=f'./visualizations/experiment_{var_r}_{var_mode}'
    # )
    # print("\nDetailed Performance Analysis:")
    # print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    # print("\nClass-wise Mean Absolute Error:")
    # for i, error in enumerate(viz_stats['class_wise_mae']):
    #     print(f"Class {i}: {error:.4f}")
    # print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return dict_true_acc_act, dict_true_acc_loc





def  multisense_loss(act_logits, activity_targets, loc_pred, location_targets,   mask):
    """
    Calculate the combined loss for location and activity prediction.

    Args:
        act_logits: Activity logits of shape (batch_size, 5, 9)
        loc_pred: Location predictions (sigmoid probabilities) of shape (batch_size, 5)
        location_targets: Ground truth location labels of shape (batch_size, 5)
        activity_targets: Ground truth activity targets of shape (batch_size, 5, 9)
        mask: Boolean mask of shape (batch_size, 5) indicating locations with predicted people
    """

    # Location loss - Binary Cross Entropy for multi-label classification
    location_loss = F.binary_cross_entropy(loc_pred, location_targets)

    # Convert activity_targets to class indices if they're one-hot encoded
    if activity_targets.dim() == 3 and activity_targets.size(2) == 9:
        activity_indices = torch.argmax(activity_targets, dim=2)  # (batch_size, 5)
    else:
        activity_indices = activity_targets  # Already indices

    # Get ground truth mask (where there's actually a person)
    gt_mask = (location_targets > 0.5)

    # Combine prediction mask with ground truth mask
    valid_mask = mask & gt_mask

    # If there are any valid locations, compute activity loss
    if valid_mask.any():
        # Reshape logits and indices for loss calculation
        b, l, c = act_logits.shape  # batch, locations, classes

        # Creating a mask that keeps the same shape for proper element selection
        expanded_mask = valid_mask.unsqueeze(-1).expand_as(act_logits)

        # Compute loss only for valid locations
        act_flat = act_logits.reshape(-1, c)
        indices_flat = activity_indices.reshape(-1)
        mask_flat = valid_mask.reshape(-1)

        # Select only entries where mask is True
        act_masked = act_flat[mask_flat]
        indices_masked = indices_flat[mask_flat]

        # Compute cross-entropy only on valid entries
        activity_loss = F.cross_entropy(act_masked, indices_masked)
    else:
        # No valid locations
        activity_loss = torch.tensor(0.0, device=loc_pred.device)

    total_loss = location_loss + activity_loss

    return total_loss, activity_loss, location_loss

def train(model,
          optimizer,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device,
          var_mode: str,
          patience: int = 150):  # Added patience parameter

    data_train_loader = torch.utils.data.DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader =  torch.utils.data.DataLoader(data_test_set, len(data_test_set))

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

            predict_train_y_act, predict_train_y_loc, mask = model(data_batch_x)

            var_loss_train, _, _ = multisense_loss(predict_train_y_act, data_batch_y_act,
                                  predict_train_y_loc,  data_batch_y_loc.float(), mask)

            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()
            scheduler.step()

        data_batch_y_act = data_batch_y_act.detach().cpu().numpy()
        data_batch_y_loc = data_batch_y_loc.detach().cpu().numpy()

        predict_train_y_act = predict_train_y_act.detach().cpu().numpy()
        predict_train_y_loc = predict_train_y_loc.detach().cpu().numpy()

        # Calculate performance metrics for training
        dict_error_train_act, dict_error_train_loc = performance_metrics_joint_multiSensX(
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
            predict_test_y_act, predict_test_y_loc, mask= model(data_test_x)
            var_loss_test, _, _ = multisense_loss(predict_test_y_act, data_test_y_act.float(),
                                 predict_test_y_loc, data_test_y_loc.float(), mask)

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

            dict_error_test_act, dict_error_test_loc = performance_metrics_joint_multiSensX(
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

        if (dict_error_test_act['f1_score'] > var_best_f1_score_act and
                dict_error_test_act['perfect_prediction_percentage'] > var_best_PPP_act):

            # Update best scores
            var_best_PPP_act = dict_error_test_act['perfect_prediction_percentage']
            var_best_f1_score_act = dict_error_test_act['f1_score']

            # Still track location metrics, but don't use them for model selection
            var_best_PPP_loc = dict_error_test_loc['perfect_prediction_percentage']
            var_best_f1_score_loc = dict_error_test_loc['f1_score']

            var_best_weight = copy.deepcopy(model.state_dict())
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
