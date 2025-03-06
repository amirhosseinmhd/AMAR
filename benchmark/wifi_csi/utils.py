import numpy as np
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import torch
import json
import numpy as np
import wandb
import os


def load_model_components(model, load_path, lr,  scenario="full", device=None):
    """
    Selectively load model components based on scenario from full model state dict
    Args:
        model: DETR_MultiUser model
        load_path: Path to load full model state dict
        scenario: One of ["full", "feature_extractor", "feature_encoder"]
        device: torch device
    Returns:
        model: Updated model
        param_groups: List of parameter groups with their learning rates
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full state dict
    state_dict = torch.load(load_path, map_location=device)

    param_groups = []

    if scenario == "full":
        # Use full model as initialization
        model.load_state_dict(state_dict)
        param_groups.append({'params': model.parameters(), 'lr': lr})

    elif scenario == "feature_extractor":
        # Only load feature extractor, keep other components random
        feature_extractor_dict = {k: v for k, v in state_dict.items()
                                  if k.startswith('feature_extractor.')}
        model.feature_extractor.load_state_dict(
            {k.replace('feature_extractor.', ''): v
             for k, v in feature_extractor_dict.items()}
        )

        # Different learning rates for different components
        param_groups.extend([
            {'params': model.feature_extractor.parameters(), 'lr': lr * 0.01},  # Very small lr
            {'params': model.encoder.parameters(), 'lr': lr},
            {'params': model.decoder.parameters(), 'lr': lr}
        ])

    elif scenario == "feature_encoder":
        # Load feature extractor and encoder, keep decoder random
        fe_encoder_dict = {k: v for k, v in state_dict.items()
                           if k.startswith(('feature_extractor.', 'encoder.'))}

        # Load feature extractor
        feature_extractor_dict = {k.replace('feature_extractor.', ''): v
                                  for k, v in fe_encoder_dict.items()
                                  if k.startswith('feature_extractor.')}
        model.feature_extractor.load_state_dict(feature_extractor_dict)

        # Load encoder
        encoder_dict = {k.replace('encoder.', ''): v
                        for k, v in fe_encoder_dict.items()
                        if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_dict)

        # Freeze feature extractor, small lr for encoder, normal lr for decoder
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        param_groups.extend([
            {'params': model.encoder.parameters(), 'lr': lr * 0.1},
            {'params': model.decoder.parameters(), 'lr': lr}
        ])

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return model, param_groups


def save_model_components(preset, model):
    """
    Save model components based on scenario
    Args:
        model: DETR_MultiUserJoint model
        save_dir: Directory to save model
        scenario: One of ["full", "feature_extractor", "feature_encoder"]
    """
    save_dir = preset.get("saving_path") + f"model_0"
    os.makedirs(save_dir, exist_ok=True)
    env = "_".join(preset["data"]["environment"])
    model_ = preset["model"]
    torch.save(model.state_dict(), f"{save_dir}/PT_{env}_{model_}.pth")
    
def error_per_number_person(y_pred, y_true):
    """
    Args:
        y_pred: numpy array of shape (num_samples, 9) containing prediction for each activity
        y_true: numpy array of shape (num_samples, 9) containing true values

    Returns:
        error count if we have one activity, error count if we have two persons, and so on
    """
    count_num_people = y_true.sum(axis=1) # finding number of people in each sample
    error_count = np.abs(y_pred - y_true).sum(axis=1) # finding error count in each sample

    error_per_person = []
    for count_index in range(1, 6):
        index = np.where(count_num_people == count_index) # gives us index of samples with count_index people
        error_per_person.append(error_count[index].mean()) # finding mean error count for samples with count_index people

    return error_per_person

def count_error(y_pred, y_true):
    """
    Args:
        y_pred: numpy array of shape (num_samples, 9) containing prediction for each activity
        y_true: numpy array of shape (num_samples, 9) containing true values

    Returns:
        error for count number people in each sample, (num_samples, 1)
    """
    count_num_people_y = y_true.sum(axis=1) # finding number of people in each sample
    count_num_people_y_pred = y_pred.sum(axis=1) # finding number of people in each sample
    error_count = np.abs(count_num_people_y_pred - count_num_people_y) # finding error count in each sample
    return error_count


def threshold_round(x, threshold=0.3):
    """
    Custom rounding function that uses a threshold.
    If decimal part > threshold, rounds up; otherwise rounds down.
    """
    # Get the decimal part
    decimal_part = x - np.floor(x)
    # Round up if decimal part > threshold, down otherwise
    return np.ceil(x) if decimal_part > threshold else np.floor(x)

def process_predictions(y_pred, y_true, var_threshold=0.5):
    """
    Process activity predictions to count activities above threshold.

    Args:
        y_pred: numpy array of shape (num_samples, 6, 9) containing probabilities
        y_true: numpy array of shape (num_samples, 6, 9) containing true values
        var_threshold: minimum probability threshold for counting an activity

    Returns:
        y_pred_processed: summed activity counts (num_samples, 9)
        y_true_processed: summed true activity counts (num_samples, 9)
        batch_size: number of samples in batch
    """
    # Get the indices of max probabilities for each user
    max_indices = np.argmax(y_pred, axis=2)  # Shape: (num_samples, 6)

    # Get the corresponding maximum probabilities
    max_probs = np.take_along_axis(y_pred, np.expand_dims(max_indices, axis=2), axis=2)
    max_probs = max_probs.squeeze(axis=2)  # Shape: (num_samples, 6)

    # Create a mask for probabilities above threshold
    above_threshold = max_probs > var_threshold  # Shape: (num_samples, 6)

    # Create one-hot encoded matrix for the predicted activities
    y_pred_one_hot = np.zeros_like(y_pred)  # Shape: (num_samples, 6, 9)
    batch_indices = np.arange(y_pred.shape[0])[:, None]
    user_indices = np.arange(y_pred.shape[1])[None, :]
    y_pred_one_hot[batch_indices, user_indices, max_indices] = above_threshold

    # Sum up the activities across users
    y_pred_processed = y_pred_one_hot.sum(axis=1)  # Shape: (num_samples, 9)
    y_true_processed = y_true.sum(axis=1)  # Shape: (num_samples, 9)

    batch_size = y_true.shape[0]

    return y_pred_processed, y_true_processed, batch_size

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def calculate_scores(y_true, y_pred):

    tp = np.minimum(y_true, y_pred)
    tn = np.where(np.maximum(y_true, y_pred) == 0, 1, 0)
    fp = np.maximum(0, y_pred - y_true) # Extra predictions
    fn = np.maximum(0, y_true - y_pred)  # Missed objects
    tp_per_activity = tp.sum(axis=0)
    tn_per_activity = tn.sum(axis=0)
    fp_per_activity = fp.sum(axis=0)
    fn_per_activity = fn.sum(axis=0)
    precision_ = np.where((tp_per_activity + fp_per_activity) > 0,  tp_per_activity / (tp_per_activity + fp_per_activity + 1e-6) , 0)
    recall_ = np.where((tp_per_activity + fn_per_activity) > 0,  tp_per_activity / (tp_per_activity + fn_per_activity + 1e-6) , 0)
    f1_score_ = np.where((precision_ + recall_) > 0,  2 * (precision_ * recall_) / (precision_ + recall_ + 1e-6)  , 0)
    accuracy_ = (tp_per_activity + tn_per_activity )/ (tp_per_activity + fn_per_activity + tn_per_activity + fp_per_activity)

    return precision_.mean(), recall_.mean(), f1_score_.mean(), accuracy_.mean()

def my_train_test_split(X, y_location_n, y_activity_n, test_size=0.2, random_state=103):
    m = X.shape[0]
    rng = np.random.RandomState(random_state)
    random_indices = np.arange(0, m)
    rng.shuffle(random_indices)
    m_train = int(m * (1 - test_size))
    indices_train = random_indices[0:m_train]
    indices_test = random_indices[m_train:]
    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train_loc = y_location_n[indices_train]
    y_test_loc = y_location_n [indices_test]
    y_train_act = y_activity_n[indices_train]
    y_test_act = y_activity_n [indices_test]

    return X_train, X_test, y_train_loc, y_test_loc, y_train_act, y_test_act
def performance_metrics_joint(y_true_act, y_pred_act, y_true_loc, y_pred_loc):
    # Ensure inputs are numpy arrays
    y_true_act = np.array(y_true_act)
    y_pred_act = np.array(y_pred_act)
    y_true_loc = np.array(y_true_loc)
    y_pred_loc = np.array(y_pred_loc)
    last_act_pred = y_pred_act[-1]
    last_loc_pred = y_pred_loc[-1]
    batch_size, num_heads, num_classes_act = last_act_pred.shape
    num_classes_loc = last_loc_pred.shape[-1]


    ## Creating predicted indicies in order to convert it to one hot encoded
    y_pred_indices_act = np.argmax(last_act_pred, axis=-1)
    y_pred_indices_loc = np.argmax(last_loc_pred, axis=-1)
    last_act_pred = np.eye(num_classes_act)[y_pred_indices_act]
    last_loc_pred = np.eye(num_classes_loc)[y_pred_indices_loc] #
    ##

    last_loc_pred[y_pred_indices_act == 9] = np.zeros(num_classes_loc)
    last_loc_pred = last_loc_pred.sum(axis=1)
    last_act_pred = last_act_pred.sum(axis=1)

    y_true_loc = y_true_loc.sum(axis=1)
    y_true_act = y_true_act.sum(axis=1)

    # Metrics for activity prediction
    act_metrics = calculate_performance_metrics(y_true_act, last_act_pred, batch_size)

    # Metrics for location prediction
    loc_metrics = calculate_performance_metrics(y_true_loc, last_loc_pred, batch_size)

    return act_metrics, loc_metrics

def calculate_performance_metrics(y_true, y_pred, batch_size):
        # Calculate the absolute difference
        absolute_diff = np.abs(y_true - y_pred)

        # Find perfect predictions
        perfect_prediction_mask = np.all(absolute_diff == 0, axis=1)
        perfect_predictions = np.sum(perfect_prediction_mask)
        perfect_prediction_percentage = (perfect_predictions / batch_size) * 100

        # Calculate total error
        total_error = np.sum(absolute_diff) / batch_size

        # Calculate error per person
        error_per_person = error_per_number_person(y_pred, y_true)

        # Calculate counting error per person
        counting_error_perPerson = count_error(y_pred, y_true)
        mean_count_error = counting_error_perPerson.mean()

        # Calculate precision, recall, f1-score and accuracy
        precision_, recall_, f1_score_, acc = calculate_scores(y_true, y_pred)

        return {
            'total_error': total_error,
            'perfect_prediction_percentage': perfect_prediction_percentage,
            'accuracy': acc,
            'error_per_person': error_per_person,
            'mean_count_error': mean_count_error,
            'counting_error_perPerson': counting_error_perPerson,
            'precision': precision_,
            'recall': recall_,
            'f1_score': f1_score_
        }



def performance_metrics(y_true, y_pred, var_mode="joint_multihead", var_threshold=0.5):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if var_mode == "count_classification_withConstrain":
        batch_size, num_classes = y_pred.shape
    elif var_mode == "multi_head":
        y_pred = y_pred[-1]
        batch_size, num_heads, num_classes = y_pred.shape
        y_pred_indices = np.argmax(y_pred, axis=-1)
        y_pred = np.eye(num_classes)[y_pred_indices]
        y_pred = y_pred.sum(axis=1)
        y_true = y_true.sum(axis=1)
        y_pred = y_pred[:, :-1]
        y_true = y_true[:, :-1]
    elif var_mode == "count_classification":
        batch_size, num_classes = y_pred.shape
        # Apply custom threshold rounding and clipping
        threshold_round_vec = np.vectorize(threshold_round)
        y_pred = np.clip(threshold_round_vec(y_pred, threshold=0.5), a_min=0, a_max=5)
    elif var_mode == "baseline":
        y_pred = (1 / (1 + np.exp(-y_pred))).astype(float)
        y_true = y_true.reshape(y_true.shape[0], -1, 9)
        y_pred = y_pred.reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])
        y_pred, y_true, batch_size = process_predictions(y_pred, y_true, var_threshold=0.5)
        batch_size = y_true.shape[0]
    else:
        raise ValueError(f"Unsupported var_mode: {var_mode}")

    # Calculate the absolute difference
    absolute_diff = np.abs(y_true - y_pred)
    # acc_bysample = (1 * absolute_diff == 0).sum(axis=1) / absolute_diff.shape[1]
    # acc = acc_bysample.mean()
    # std = acc_bysample.std()

    # Find perfect predictions
    perfect_prediction_mask = np.all(absolute_diff == 0, axis=1)
    perfect_predictions = np.sum(perfect_prediction_mask)
    perfect_prediction_percentage = (perfect_predictions / batch_size) * 100
    total_error = np.sum(absolute_diff) / batch_size
    error_per_person = error_per_number_person(y_pred, y_true)
    counting_error_perPerson = count_error(y_pred,  y_true)
    mean_count_error = counting_error_perPerson.mean()

    precision_, recall_, f1_score_, acc = calculate_scores(y_true, y_pred)

    return {
        'total_error': total_error,
        'perfect_prediction_percentage': perfect_prediction_percentage,
        'accuracy': acc,
        'error_per_person': error_per_person,
        'mean_count_error': mean_count_error,
        'counting_error_perPerson': counting_error_perPerson,
        'precision': precision_,
        'recall': recall_,
        'f1_score': f1_score_
    }


def reduce_dataset_joint(data_activity, data_location, num_object_queries=None):
    new_data_activity = []
    new_data_location = []

    zero = np.zeros((5, 1))

    for i in range(0, data_location.shape[0]):
        sample_location = data_location[i]
        sample_activity = data_activity[i]
        if i ==1000:
            print("ff")
        # Count non-zero rows-pp
        legend_non_zero = sample_activity.sum(axis=1) # This gives us which queries are redundant or possible to eliminate

        new_sample_location= np.delete(sample_location, (legend_non_zero == 0).argmax(), axis=0) # deleting the first empy query

        new_sample_activity = np.delete(sample_activity, (legend_non_zero == 0).argmax(), axis=0)


        new_sample = np.hstack((new_sample_activity, zero))
        legend_non_zero = new_sample.sum(axis=1)
        new_sample[legend_non_zero == 0, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


        if num_object_queries:
            new_matrix = np.repeat([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], num_object_queries-5, axis=0)
            new_sample = np.concatenate((new_sample, new_matrix))
        new_data_activity.append(new_sample)
        new_data_location.append(new_sample_location)
    # we expect to return two arrays...
    return (np.array(new_data_activity), np.array(new_data_location))


def reduce_dataset(data, num_object_queries=None):
    new_data = []
    zero = np.zeros((5, 1))

    for sample in data:
        # Count non-zero rows-pp
        legend_non_zero = sample.sum(axis=1)
        new_sample = np.delete(sample, (legend_non_zero == 0).argmax(), axis=0)
        new_sample = np.hstack((new_sample, zero))
        legend_non_zero = new_sample.sum(axis=1)
        new_sample[legend_non_zero == 0, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        if num_object_queries:
            new_matrix = np.repeat([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], num_object_queries-5, axis=0)
            new_sample = np.concatenate((new_sample, new_matrix))
        indices = np.random.permutation(5)
        new_data.append(new_sample[indices])
    return np.array(new_data)

def visualize_model_performance(y_pred, y_true, save_dir="./visualizations",var_mode="multi_head"):
    """
    Creates and saves various visualizations of model performance
    y_pred: numpy array [batch_size, 10] (predicted counts)
    y_true: numpy array [batch_size, 10] (true counts)
    """
    print(var_mode)
    if var_mode == "count_classification_withConstrain":
        pass
    elif var_mode == "multi_head":
        y_pred = y_pred[-1]
        batch_size, num_heads, num_classes = y_pred.shape

        y_pred_indices = np.argmax(y_pred, axis=-1)
        y_pred = np.eye(num_classes)[y_pred_indices] # this gives us one hot encoded version of it.
        y_pred = y_pred.sum(axis=1)  # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)
        y_pred = y_pred[:, :-1]
        y_true = y_true[:, :-1]

    elif var_mode == "count_classification":
        batch_size, num_classes = y_pred.shape
        threshold_round_vec = np.vectorize(threshold_round)
        y_pred = np.clip(threshold_round_vec(y_pred, threshold=0.3), a_min=0, a_max=5)
    elif var_mode == "baseline":
        y_pred = (1 / (1 + np.exp(-y_pred)) > 0.5).astype(float)
        y_true = y_true.reshape(y_true.shape[0], -1, 9)
        y_pred = y_pred.reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])
        y_pred = y_pred.sum(axis=1) # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)
    else:
        raise ValueError(f"Unsupported var_mode: {var_mode}")
    os.makedirs(f"{save_dir}", exist_ok=True)

    # 1. Distribution of Predictions vs Ground Truth
    plt.figure(figsize=(15, 5))

    # Plot for each class
    for i in range(int(y_pred.shape[1])):
        plt.subplot(2, 5, i + 1)
        plt.hist(y_true[:, i], alpha=0.5, label='Ground Truth', bins=range(7))
        plt.hist(y_pred[:, i], alpha=0.5, label='Predicted', bins=range(7))
        plt.title(f'Class {i}')
        plt.xlabel('Count')
        plt.ylabel('Frequency')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/count_distributions_{var_mode}.png')
    plt.close()

    # 2. Confusion Matrix for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(int(y_pred.shape[1])):
        ax = axes[i // 5, i % 5]
        cm = confusion_matrix(y_true[:, i], np.round(y_pred[:, i]))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(f'Class {i}')
        ax.set_xlabel('Predicted Count')
        ax.set_ylabel('True Count')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices_{var_mode}.png')
    plt.close()

    # 3. Error Distribution
    plt.figure(figsize=(10, 6))
    errors = np.abs(y_pred - y_true).mean(axis=1)
    plt.hist(errors, bins=30)
    plt.title('Distribution of Mean Absolute Error per Sample')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_dir}/error_distribution_{var_mode}.png')
    plt.close()

    # 4. Class-wise Error Analysis
    plt.figure(figsize=(10, 6))
    class_errors = np.abs(y_pred - y_true).mean(axis=0)
    plt.bar(range(int(y_pred.shape[1])), class_errors)
    plt.title('Mean Absolute Error by Class')
    plt.xlabel('Class')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(f'{save_dir}/class_errors_{var_mode}.png')
    plt.close()

    # 5. Prediction vs Ground Truth Scatter
    plt.figure(figsize=(10, 10))
    for i in range(int(y_pred.shape[1])):
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.1, label=f'Class {i}')
    plt.plot([0, 5], [0, 5], 'r--')  # Perfect prediction line
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('Predicted vs True Counts')
    plt.legend()
    plt.savefig(f'{save_dir}/prediction_scatter_{var_mode}.png')
    plt.close()

    # Return summary statistics
    return {
        'class_wise_mae': class_errors.tolist(),
        'mean_error': errors.mean(),
        'error_std': errors.std(),
        'perfect_predictions': (np.abs(y_pred - y_true) < 0.5).all(axis=1).mean()
    }


def log_attention_weights(model, y_pred, y_actual, epoch):
    """
    Log attention weights from all decoder layers to wandb based on count of people.
    For each count (0-4), select one random sample and visualize its attention weights across all decoder layers.
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(y_actual, np.ndarray):
        y_actual = torch.from_numpy(y_actual)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Get count of people (number of non-9 values) in each sequence
    where_9_happens = y_actual == 9
    count_no_person = torch.sum(~where_9_happens, dim=1)

    decoder_layers = model.decoder.decoder_layers

    # For each possible count (0, 1, 2, 3, 4, 5)
    for i in range(6):
        # Find examples with this count
        indices = torch.where(count_no_person == i)[0]
        if len(indices) == 0:
            continue

        # Randomly select one example with this count - will be used for all layers
        random_idx = indices[torch.randint(0, len(indices), (1,)).item()]

        # Get actual labels and predictions for this example
        actual_sequence = y_actual[random_idx].cpu().numpy()
        pred_sequence = y_pred[random_idx].cpu().numpy()

        # Now iterate through layers using the same sample
        for layer_idx, layer in enumerate(decoder_layers):
            attn_weights = layer.cross_attn_weights.detach().clone()
            if attn_weights is None:
                continue

            # Average over batch dimension if necessary
            if attn_weights.dim() > 3:
                attn_weights = attn_weights.mean(0)

            # Get attention weights for this example
            example_attn_weights = attn_weights[random_idx]

            # Create heatmap figure with subplots
            fig, ax = plt.subplots(figsize=(12, 10))

            # Add heatmap
            sns.heatmap(
                example_attn_weights.cpu().numpy(),
                cmap='viridis',
                xticklabels=range(example_attn_weights.shape[1]),
                yticklabels=[f'Q {j}' for j in range(example_attn_weights.shape[0])],
                ax=ax
            )

            # Add title with sample info
            plt.title(f'Cross-Attention Weights - Layer {layer_idx} - {i} People\n' +
                      f'Sample Index: {random_idx}\n' +
                      f'Actual: {actual_sequence}\n' +
                      f'Prediction: {pred_sequence}',
                      fontsize=12)

            plt.xlabel('Encoder Sequence Position')
            plt.ylabel('Query')
            plt.tight_layout()

            # Log to wandb
            wandb.log({
                f'attention_weights/layer_{layer_idx}_people_{i}': wandb.Image(fig),
                'epoch': epoch
            })

            plt.close('all')