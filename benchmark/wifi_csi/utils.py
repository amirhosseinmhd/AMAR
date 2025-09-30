import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import torch
import json
import wandb
import os
import re
import time

# def load_model_components(model, load_path, lr, scenario="full", device=None):
#     """
#     Selectively load model components based on scenario from full model state dict
#     Args:
#         model: DETR_MultiUser model
#         load_path: Path to load full model state dict
#         lr: Base learning rate
#         scenario: One of ["full", "feature_extractor", "feature_encoder", "decoder_only"]
#         device: torch device
#     Returns:
#         model: Updated model
#         param_groups: List of parameter groups with their learning rates
#     """
#     if device is None:
#         # Update device selection to check for CUDA first, then MPS (Apple Silicon), then CPU
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             device = torch.device("mps")
#         else:
#             device = torch.device("cpu")
#
#         print(f"Using device: {device}")
#     # Load full state dict
#     state_dict = torch.load(load_path, map_location=device)
#     param_groups = []
#
#     if scenario == "full":
#         # Use full model as initialization
#         model.load_state_dict(state_dict)
#         param_groups.append({'params': model.parameters(), 'lr': lr})
#
#     elif scenario == "feature_extractor":
#         # Only load feature extractor, keep other components random
#         feature_extractor_dict = {k: v for k, v in state_dict.items()
#                                   if k.startswith('feature_extractor.')}
#         model.feature_extractor.load_state_dict(
#             {re.sub('^feature_extractor\.', '', k): v
#              for k, v in feature_extractor_dict.items()}
#         )
#         # Different learning rates for different components
#         param_groups.extend([
#             {'params': model.feature_extractor.parameters(), 'lr': lr * 0.0001},  # Very small lr for pretrained component
#             {'params': model.encoder.parameters(), 'lr': lr},  # Regular lr for new components
#             {'params': model.decoder.parameters(), 'lr': lr}
#         ])
#
#     elif scenario == "feature_encoder":
#         # Load feature extractor and encoder, keep decoder random
#         feature_encoder_dict = {k: v for k, v in state_dict.items()
#                                 if k.startswith('feature_extractor.') or k.startswith('encoder.')}
#         # Load feature extractor
#         feature_dict = {re.sub('^feature_extractor\.', '', k): v
#                         for k, v in feature_encoder_dict.items()
#                         if k.startswith('feature_extractor.')}
#         model.feature_extractor.load_state_dict(feature_dict)
#         # # Freeze feature extractor parameters
#         # for param in model.feature_extractor.parameters():
#         #     param.requires_grad = False
#         # Load encoder
#         encoder_dict = {re.sub('^encoder\.', '', k): v
#                         for k, v in feature_encoder_dict.items()
#                         if k.startswith('encoder.')}
#         model.encoder.load_state_dict(encoder_dict)
#         # Different learning rates for different components
#         param_groups.extend([
#             # feature_extractor parameters are frozen, so not included here
#             {'params': model.feature_extractor.parameters(), 'lr': lr * 0.01},
#             # Very small lr for pretrained component
#             {'params': model.encoder.parameters(), 'lr': lr * 0.1},  # Small lr
#             {'params': model.decoder.parameters(), 'lr': lr}  # Regular lr
#         ])
#
#     elif scenario == "decoder_only":
#         # Keep feature extractor and encoder random, only load decoder
#         decoder_dict = {k: v for k, v in state_dict.items()
#                         if k.startswith('decoder.')}
#         model.decoder.load_state_dict(
#             {re.sub('^decoder\.', '', k): v
#              for k, v in decoder_dict.items()}
#         )
#         # Different learning rates for different components
#         param_groups.extend([
#             {'params': model.feature_extractor.parameters(), 'lr': lr},  # Regular lr
#             {'params': model.encoder.parameters(), 'lr': lr},  # Regular lr
#             {'params': model.decoder.parameters(), 'lr': lr * 0.1}  # Small lr for pretrained component
#         ])
#
#     else:
#         raise ValueError(
#             f"Unknown scenario: {scenario}. Choose from: 'full', 'feature_extractor', 'feature_encoder', 'decoder_only'")
#
#     print(f"Loaded model components for scenario: {scenario}")
#     return model, param_groups
#
# def save_model_components(preset, model):
#     """
#     Save model components based on scenario
#     Args:
#         model: DETR_MultiUserJoint model
#         save_dir: Directory to save model
#         scenario: One of ["full", "feature_extractor", "feature_encoder"]
#     """
#     save_dir = preset.get("saving_path") + f"model_0"
#     saving_path = preset.get("saving_path", "0")
#     if not os.path.exists(saving_path):
#         raise ValueError(f"Saving path {saving_path} does not exist.")
#
#     save_dir = saving_path + f"model_0"
#     env = "_".join(preset["data"]["environment"])
#     model_ = preset["model"]
#     torch.save(model.state_dict(), f"{save_dir}/PT_{env}_{model_}.pth")
#     print(f"Model saved at {save_dir}/PT_{env}_{model_}.pth")



def load_model_components(model, load_path, lr, scenario="full", device=None):
    """
    Selectively load JEPA model components based on scenario from full model state dict

    Args:
        model: JEPA_Sup model
        load_path: Path to load full model state dict
        lr: Base learning rate
        scenario: One of ["full", "freeze_encoder", "low_lr_encoder", "decoder_only", "new_init"]
        device: torch device

    Returns:
        model: Updated model with loaded weights
        param_groups: List of parameter groups with their learning rates
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

    # Handle the case where no pretrained model should be loaded
    if scenario == "new_init":
        print("Using random initialization for all components")
        param_groups = [{'params': model.parameters(), 'lr': lr}]
        return model, param_groups

    # Load pretrained state dict
    try:
        checkpoint = torch.load(load_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using random initialization for all components")
        param_groups = [{'params': model.parameters(), 'lr': lr}]
        return model, param_groups

    param_groups = []
    loaded_components = []

    def clean_key_name(key):
        """Remove _orig_mod. prefix from torch.compile() keys"""
        return key.replace('_orig_mod.', '')

    def load_component(component_name, target_module, lr_multiplier=1.0, freeze=False):
        """Helper function to load a specific component"""
        # Find keys that match the component (handle both compiled and non-compiled)
        component_dict = {}
        prefixes = [f'{component_name}.', f'{component_name}._orig_mod.']

        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    # Clean the key name for loading
                    clean_key = clean_key_name(key)
                    target_key = clean_key.replace(f'{component_name}.', '')
                    component_dict[target_key] = value
                    break

        if component_dict:
            try:
                target_module.load_state_dict(component_dict, strict=False)
                loaded_components.append(component_name)
                print(f"✅ Loaded {component_name} ({len(component_dict)} parameters)")

                # Set learning rate and freezing
                if freeze:
                    for param in target_module.parameters():
                        param.requires_grad = False
                    print(f"🔒 Frozen {component_name}")
                else:
                    param_groups.append({
                        'params': target_module.parameters(),
                        'lr': lr * lr_multiplier,
                        'name': component_name
                    })
                    print(f"📚 {component_name} LR: {lr * lr_multiplier}")
            except Exception as e:
                print(f"❌ Failed to load {component_name}: {e}")
        else:
            print(f"⚠️ No weights found for {component_name}")

    # Execute loading based on scenario
    if scenario == "full":
        # Load everything and use normal learning rates
        try:
            # Handle compiled model state dict
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                cleaned_key = clean_key_name(key)
                cleaned_state_dict[cleaned_key] = value

            model.load_state_dict(cleaned_state_dict, strict=False)
            param_groups.append({'params': model.parameters(), 'lr': lr})
            print("✅ Loaded full model with all pretrained weights")
        except Exception as e:
            print(f"❌ Failed to load full model: {e}")
            param_groups.append({'params': model.parameters(), 'lr': lr})

    elif scenario == "freeze_encoder":
        # A. Freeze feature extractor and transformer encoder, normal learning for decoder
        load_component('online_cnn_feature_extractor', model.online_cnn_feature_extractor, freeze=True)
        load_component('online_transformer_encoder', model.online_transformer_encoder, freeze=True)

        # Decoder gets normal learning rate (not loaded from pretrained)
        param_groups.append({
            'params': model.decoder.parameters(),
            'lr': lr,
            'name': 'decoder'
        })
        print(f"📚 decoder LR: {lr}")

    elif scenario == "low_lr_encoder":
        # B. Low learning rate for encoders, normal lr for decoder
        load_component('online_cnn_feature_extractor', model.online_cnn_feature_extractor, lr_multiplier=0.01)
        load_component('online_transformer_encoder', model.online_transformer_encoder, lr_multiplier=0.1)

        # Decoder gets normal learning rate (not loaded from pretrained)
        param_groups.append({
            'params': model.decoder.parameters(),
            'lr': lr,
            'name': 'decoder'
        })
        print(f"📚 decoder LR: {lr}")

    elif scenario == "decoder_only":
        # Load only decoder, keep encoders random with normal learning rates
        load_component('decoder', model.decoder, lr_multiplier=0.1)

        # Encoders get normal learning rates (random initialization)
        param_groups.extend([
            {
                'params': model.online_cnn_feature_extractor.parameters(),
                'lr': lr,
                'name': 'online_cnn_feature_extractor'
            },
            {
                'params': model.online_transformer_encoder.parameters(),
                'lr': lr,
                'name': 'online_transformer_encoder'
            }
        ])
        print(f"📚 Encoders LR: {lr}")

    else:
        raise ValueError(
            f"Unknown scenario: {scenario}. Choose from: "
            f"'full', 'freeze_encoder', 'low_lr_encoder', 'decoder_only', 'new_init'"
        )

    # Always update target encoder to match online encoder (for EMA)
    if hasattr(model, 'target_cnn_feature_extractor') and hasattr(model, 'target_transformer_encoder'):
        try:
            model.target_cnn_feature_extractor.load_state_dict(
                model.online_cnn_feature_extractor.state_dict()
            )
            model.target_transformer_encoder.load_state_dict(
                model.online_transformer_encoder.state_dict()
            )
            print("✅ Updated target encoders to match online encoders")
        except Exception as e:
            print(f"⚠️ Could not update target encoders: {e}")

    print(f"🎯 Loaded model components for scenario: {scenario}")
    print(f"📊 Components loaded: {loaded_components}")

    return model, param_groups


# Helper function to save model components
def save_model_components(preset, model):
    """Save model components based on preset configuration"""
    if not preset.get("save_model"):
        return

    import os
    import time

    # Create save directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        preset.get("saving_path", "./saved_models"),
        f"jepa_sup_{'+'.join(preset['data']['environment'])}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save full model
    model_path = os.path.join(save_dir, "full_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': preset,
        'environment': preset['data']['environment']
    }, model_path)

    print(f"💾 Model saved to: {model_path}")
    return model_path

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
    """
    Calculate all performance metrics for the predictions
    Args:
        y_true: numpy array of ground truth values
        y_pred: numpy array of predicted values
    Returns:
        Dictionary containing all performance metrics
    """
    # Calculate basic metrics
    absolute_diff = np.abs(y_true - y_pred)
    
    # Calculate TP, TN, FP, FN
    tp = np.minimum(y_true, y_pred)
    tn = np.where(np.maximum(y_true, y_pred) == 0, 1, 0)
    fp = np.maximum(0, y_pred - y_true) # Extra predictions
    fn = np.maximum(0, y_true - y_pred)  # Missed objects
    
    # Calculate per-activity metrics
    tp_per_activity = tp.sum(axis=0)
    tn_per_activity = tn.sum(axis=0)
    fp_per_activity = fp.sum(axis=0)
    fn_per_activity = fn.sum(axis=0)
    
    # Calculate precision, recall, f1-score
    precision_ = np.where((tp_per_activity + fp_per_activity) > 0, 
                         tp_per_activity / (tp_per_activity + fp_per_activity + 1e-6), 0)
    recall_ = np.where((tp_per_activity + fn_per_activity) > 0,
                      tp_per_activity / (tp_per_activity + fn_per_activity + 1e-6), 0)
    f1_score_ = np.where((precision_ + recall_) > 0,
                        2 * (precision_ * recall_) / (precision_ + recall_ + 1e-6), 0)
    accuracy_ = (tp_per_activity + tn_per_activity) / (tp_per_activity + fn_per_activity + tn_per_activity + fp_per_activity)
    
    # Calculate perfect predictions
    perfect_prediction_mask = np.all(absolute_diff == 0, axis=1)
    perfect_predictions = np.sum(perfect_prediction_mask)
    batch_size = len(y_true)
    perfect_prediction_percentage = (perfect_predictions / batch_size) * 100
    
    # Calculate total error
    total_error = np.sum(absolute_diff) / batch_size
    
    # Calculate error per person and counting error
    error_per_person = error_per_number_person(y_pred, y_true)
    counting_error_perPerson = count_error(y_pred, y_true)
    mean_count_error = counting_error_perPerson.mean()
    
    return {
        'total_error': total_error,
        'perfect_prediction_percentage': perfect_prediction_percentage,
        'accuracy': accuracy_.mean(),
        'error_per_person': error_per_person,
        'mean_count_error': mean_count_error,
        'counting_error_perPerson': counting_error_perPerson,
        'precision': precision_.mean(),
        'recall': recall_.mean(),
        'f1_score': f1_score_.mean()
    }

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
def performance_metrics_joint_multiSensX(y_true_act, y_pred_act, y_true_loc, y_pred_loc):
    y_true_act = np.array(y_true_act)
    y_pred_act = np.array(y_pred_act)
    y_true_loc = np.array(y_true_loc)
    y_pred_loc = np.array((y_pred_loc>0.5).astype(int) ) # batch size by 5

    y_pred_indices = np.argmax(y_pred_act, axis=-1)
    y_pred_act = np.eye(y_pred_act.shape[-1])[y_pred_indices]

    # batchsize by 5 by 9
    mask_ = y_pred_loc==0
    y_pred_act[mask_] = [0, 0 , 0 , 0 , 0 , 0, 0,0 , 0]
    y_pred_act = y_pred_act.sum(axis=1)



    y_true_act = y_true_act.sum(axis=1)

    # Metrics for activity prediction
    act_metrics = calculate_scores(y_true_act, y_pred_act)
    # Metrics for location prediction
    loc_metrics = calculate_scores(y_true_loc, y_pred_loc)
    return act_metrics, loc_metrics

def performance_metrics_joint(y_true_act, y_pred_act, y_true_loc, y_pred_loc):

    ### This function should be integrated to main performance metrics function. The method involve
    ### in changing the main function so that it returns the two metrics.
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
    act_metrics = calculate_scores(y_true_act, last_act_pred)

    # Metrics for location prediction
    loc_metrics = calculate_scores(y_true_loc, last_loc_pred)

    return act_metrics, loc_metrics

##### THIS CODE IS DEPRICET AND OUT OF DATE
# def calculate_performance_metrics(y_true, y_pred, batch_size):
#     # seems like this function is out of date..
#         # Calculate the absolute difference
#         absolute_diff = np.abs(y_true - y_pred)

#         # Find perfect predictions
#         perfect_prediction_mask = np.all(absolute_diff == 0, axis=1)
#         perfect_predictions = np.sum(perfect_prediction_mask)
#         perfect_prediction_percentage = (perfect_predictions / batch_size) * 100

#         # Calculate total error
#         total_error = np.sum(absolute_diff) / batch_size

#         # Calculate error per person
#         error_per_person = error_per_number_person(y_pred, y_true)

#         # Calculate counting error per person
#         counting_error_perPerson = count_error(y_pred, y_true)
#         mean_count_error = counting_error_perPerson.mean()

#         # Calculate precision, recall, f1-score and accuracy
#         precision_, recall_, f1_score_, acc = calculate_scores(y_true, y_pred)

#         return {
#             'total_error': total_error,
#             'perfect_prediction_percentage': perfect_prediction_percentage,
#             'accuracy': acc,
#             'error_per_person': error_per_person,
#             'mean_count_error': mean_count_error,
#             'counting_error_perPerson': counting_error_perPerson,
#             'precision': precision_,
#             'recall': recall_,
#             'f1_score': f1_score_
#         }



def performance_metrics(y_true, y_pred, var_mode="joint_multihead", var_threshold=0.5):
    """
    Process predictions based on mode and calculate performance metrics
    Args:
        y_true: numpy array of ground truth values
        y_pred: numpy array of predicted values
        var_mode: prediction mode
        var_threshold: threshold for baseline mode
    Returns:
        Dictionary containing all performance metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Process predictions based on mode
    if var_mode == "count_classification_withConstrain":
        pass
    elif var_mode == "multi_head":
        # Initialize dictionary to store metrics for each layer
        all_layer_metrics = {}
        
        # Process each layer's predictions
        for layer_idx in range(len(y_pred)):
            layer_pred = y_pred[layer_idx]
            # Convert predictions to one-hot encoded format
            y_pred_indices = np.argmax(layer_pred, axis=-1)
            layer_pred_one_hot = np.eye(layer_pred.shape[-1])[y_pred_indices]
            
            # Sum across heads
            layer_pred_sum = layer_pred_one_hot.sum(axis=1)
            y_true_sum = y_true.sum(axis=1)
            
            # Remove the last class (no-object class)
            layer_pred_sum = layer_pred_sum[:, :-1]
            y_true_sum = y_true_sum[:, :-1]
            
            # Calculate metrics for this layer
            layer_metrics = calculate_scores(y_true_sum, layer_pred_sum)
            all_layer_metrics[f'layer_{layer_idx}'] = layer_metrics
            
            # Store the last layer metrics at the top level for backward compatibility
            if layer_idx == len(y_pred) - 1:
                all_layer_metrics.update(layer_metrics)
        
        return all_layer_metrics
    elif var_mode == "count_classification":
        batch_size, num_classes = y_pred.shape
        # Apply custom threshold rounding and clipping
        threshold_round_vec = np.vectorize(threshold_round)
        y_pred = np.clip(threshold_round_vec(y_pred, threshold=0.5), a_min=0, a_max=5)
    elif var_mode == "baseline":
        y_pred = (1 / (1 + np.exp(-y_pred))).astype(float)
        y_true = y_true.reshape(y_true.shape[0], -1, 9)
        y_pred = y_pred.reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])
        y_pred, y_true, _ = process_predictions(y_pred, y_true, var_threshold=0.5)
    else:
        raise ValueError(f"Unsupported var_mode: {var_mode}")

    # Calculate all metrics
    return calculate_scores(y_true, y_pred)

def reduce_dataset_joint_multiSenseX(data_activity, data_location):
    new_data_activity = []
    y_location_n = data_location.sum(axis=1)
    for i in range(0, data_location.shape[0]):
        sample_activity = data_activity[i]
        # Count non-zero rows-pp
        legend_non_zero = sample_activity.sum(axis=1) # This gives us which queries are redundant or possible to eliminate


        new_sample_activity = np.delete(sample_activity, (legend_non_zero == 0).argmax(), axis=0)


        new_data_activity.append(new_sample_activity)
    new_data_activity = np.array(new_data_activity)

    return new_data_activity, y_location_n


def reduce_dataset_joint(data_activity, data_location, num_object_queries=None):
    new_data_activity = []
    new_data_location = []

    zero = np.zeros((5, 1))

    for i in range(0, data_location.shape[0]):
        sample_location = data_location[i]
        sample_activity = data_activity[i]
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

def reduce_dataset_dualband(data, indicies_, num_object_queries=None):
    new_data = []
    zero = np.zeros((5, 1))

    for (i, sample) in enumerate(data):
        # Count non-zero rows-pp
        legend_non_zero = sample.sum(axis=1)
        new_sample = np.delete(sample, (legend_non_zero == 0).argmax(), axis=0)
        new_sample = np.hstack((new_sample, zero))
        legend_non_zero = new_sample.sum(axis=1)
        new_sample[legend_non_zero == 0, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        if num_object_queries:
            new_matrix = np.repeat([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], num_object_queries-5, axis=0)
            new_sample = np.concatenate((new_sample, new_matrix))
        index_ = indicies_[i]
        new_data.append(new_sample[index_])
    return np.array(new_data)


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
        indices = np.random.permutation(num_object_queries)
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
    Log averaged attention weights from all decoder layers to wandb based on count of people.
    For each count (0-4), select one random sample and visualize its averaged attention weights across all decoder layers.
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
        num_layers = len(decoder_layers)
        # Now iterate through layers using the same sample
        for layer_idx, layer in enumerate(decoder_layers):
            attn_weights = layer.cross_attn_weights
            if attn_weights is None:
                continue
            attn_weights = attn_weights.detach().clone()
            if layer_idx != num_layers - 1:
                continue
            # Assuming attn_weights shape is now [batch_size, num_heads, target_seq_len, source_seq_len]
            example_attn_weights = attn_weights[random_idx]  # Shape: [num_heads, target_seq_len, source_seq_len]

            # Average attention weights across all heads
            avg_attn_weights = example_attn_weights.mean(dim=0)  # Shape: [target_seq_len, source_seq_len]

            # Create a heatmap for the averaged attention weights
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                avg_attn_weights.cpu().numpy(),
                cmap='viridis',
                xticklabels=range(avg_attn_weights.shape[1]),
                yticklabels=[f'Q {j}' for j in range(avg_attn_weights.shape[0])],
                cbar=True
            )

            # Add main title with sample info
            plt.title(f'Average Cross-Attention Weights - Layer {layer_idx} - {i} People\n' +
                        f'Sample Index: {random_idx}\n' +
                        f'Actual: {actual_sequence}\n' +
                        f'Prediction: {pred_sequence}',
                        fontsize=14)

            # Log to wandb
            wandb.log({
                f'attention_weights/layer_{layer_idx}_people_{i}': wandb.Image(plt.gcf()),
            }, step=epoch)

            plt.close('all')
    return None
def log_random_attention_weights_final(model, y_pred, y_actual, epoch, num_samples=50):
    """
    Log averaged attention weights from all decoder layers to wandb:
    1. For 50 randomly selected samples across all counts
    2. For averaged attention weights per people count (0-5)
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(y_actual, np.ndarray):
        y_actual = torch.from_numpy(y_actual)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    
    # Get total batch size
    batch_size = y_actual.shape[0]
    
    # Count people for each sample
    where_9_happens = y_actual == 9
    count_no_person = torch.sum(~where_9_happens, dim=1)
    
    decoder_layers = model.decoder.decoder_layers
    num_layers = len(decoder_layers)
    # PART 1: Plot random samples
    # Randomly select indices (up to the requested number or batch size, whichever is smaller)
    num_to_select = min(num_samples, batch_size)
    random_indices = torch.randperm(batch_size)[:num_to_select]
    
    # For each randomly selected example
    for sample_idx, random_idx in enumerate(random_indices):
        # Get actual labels and predictions for this example
        actual_sequence = y_actual[random_idx].cpu().numpy()
        pred_sequence = y_pred[random_idx].cpu().numpy()
        people_count = count_no_person[random_idx].item()
        
        # Now iterate through layers using the same sample
        for layer_idx, layer in enumerate(decoder_layers):
            attn_weights = layer.cross_attn_weights.detach().clone()
            if attn_weights is None:
                continue
            # Only plotting the last layer
            if layer_idx != num_layers - 1:
                continue    
            # Get attention weights for this example - Shape: [num_heads, target_seq_len, source_seq_len]
            example_attn_weights = attn_weights[random_idx]
            
            # Average attention weights across all heads
            avg_attn_weights = example_attn_weights.mean(dim=0)  # Shape: [target_seq_len, source_seq_len]
            
            # Create a heatmap for the averaged attention weights
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                avg_attn_weights.cpu().numpy(),
                cmap='viridis',
                xticklabels=range(avg_attn_weights.shape[1]),
                yticklabels=[f'Q {j}' for j in range(avg_attn_weights.shape[0])],
                cbar=True
            )
            
            # Add main title with sample info
            plt.title(f'Average Cross-Attention Weights - Layer {layer_idx}\n' +
                    f'Sample {sample_idx+1}/50 (Index: {random_idx}) - People Count: {people_count}\n' +
                    f'Actual: {actual_sequence}\n' +
                    f'Prediction: {pred_sequence}',
                    fontsize=14)
            
            # Log to wandb
            wandb.log({
                f'random_attention_weights/sample_{sample_idx+1}_layer_{layer_idx}': wandb.Image(plt.gcf()),
            }, step=epoch)
            
            plt.close('all')
    
    # PART 2: Plot averaged attention weights per people count
    # For each possible count (0, 1, 2, 3, 4, 5)
    for i in range(6):
        # Find all examples with this count
        indices = torch.where(count_no_person == i)[0]
        
        if len(indices) == 0:
            continue
        
        # Now iterate through layers
        for layer_idx, layer in enumerate(decoder_layers):
            attn_weights = layer.cross_attn_weights.detach().clone()
            if attn_weights is None:
                continue
            if layer_idx != num_layers - 1:
                continue    
            # Initialize tensor to accumulate attention weights for this count
            first_sample = attn_weights[indices[0]]
            accumulated_attn = torch.zeros_like(first_sample.mean(dim=0))
            
            # Calculate average attention pattern for all samples with this count
            for idx in indices:
                example_attn_weights = attn_weights[idx]
                # Average attention weights across all heads for this example
                avg_head_attn = example_attn_weights.mean(dim=0)
                accumulated_attn += avg_head_attn
            
            # Calculate the average across all samples with this count
            avg_count_attn = accumulated_attn / len(indices)
            
            # Create a heatmap for the averaged attention weights
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                avg_count_attn.cpu().numpy(),
                cmap='viridis',
                xticklabels=range(avg_count_attn.shape[1]),
                yticklabels=[f'Q {j}' for j in range(avg_count_attn.shape[0])],
                cbar=True
            )
            
            # Add main title with count info
            plt.title(f'Average Cross-Attention Weights - Layer {layer_idx} - {i} People\n' +
                    f'Averaged across {len(indices)} samples',
                    fontsize=14)
            
            # Log to wandb
            wandb.log({
                f'average_attention_weights/count_{i}_layer_{layer_idx}': wandb.Image(plt.gcf()),
            }, step=epoch)
            
            plt.close('all')