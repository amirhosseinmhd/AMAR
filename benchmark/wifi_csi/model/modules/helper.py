import os
import shutil
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from load_data import load_data_x, load_data_y, encode_activity
from preset import preset



def generate_tsne_visualizations(model, dataloader_train, device, epoch):
    """
    Generate t-SNE visualizations.
    - A plot of all representations colored by the number of people.
    - A plot of representations for single-person scenarios, colored by activity.

    Args:
        model: The JEPA model for generating representations.
        dataloader_train: DataLoader containing training data (X and y).
        device: Device to run the model on.
        epoch: Current training epoch (for logging).

    Returns:
        dict: Dictionary containing a matplotlib figure for wandb logging.
    """
    model.eval()  # Set model to evaluation mode

    # --- Extract all data from dataloader ---
    all_representations = []
    all_data_y = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader_train:
            data_batch_x = batch_x.to(device)
            representations = model.extract_representations(data_batch_x)
            all_representations.append(representations.cpu().numpy())
            all_data_y.append(batch_y)

    
    data_y_tensor = torch.cat(all_data_y, dim=0)
    data_y_np = data_y_tensor.cpu().numpy()
    del data_y_tensor
    del all_data_y
    data_y_np = data_y_np[:, :, :-1]
    num_people = data_y_np.sum(axis=(1, 2))


    representations_np = np.concatenate(all_representations, axis=0)

    # --- t-SNE model ---
    tsne_model = TSNE(
        n_components=2,
        perplexity=50,
        random_state=42,
        n_jobs=-1,
        init='pca',
        learning_rate='auto'
    )
    tsne_results = tsne_model.fit_transform(representations_np)

    # --- Create Subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), squeeze=False)
    axes = axes.flatten()
    # --- Plot 1: All data, colored by number of people ---
    ax1 = axes[0]
    unique_labels = np.unique(num_people)
    print(unique_labels)

    cmap1 = plt.get_cmap('viridis', len(unique_labels))
    scatter1 = ax1.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=num_people,
        cmap=cmap1,
        alpha=0.8,
        s=40,
        vmin=unique_labels.min() - 0.5,
        vmax=unique_labels.max() + 0.5
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, ticks=unique_labels, label="Number of People")
    cbar1.ax.tick_params(labelsize=10)
    ax1.set_title(f"t-SNE by Number of People (Epoch {epoch})")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")

    # --- Plot 2: Single-person data, colored by activity ---
    ax2 = axes[1]
    single_person_mask = (num_people == 1)
    
    if np.any(single_person_mask):
        single_person_reps = tsne_results[single_person_mask]
        single_person_y = data_y_np[single_person_mask]
        
        # Find activity index for each single-person sample
        activity_indices = np.argmax(single_person_y.sum(axis=1), axis=1)
        
        unique_activities = np.unique(activity_indices)
        activity_map = preset["encoding"]["activity"]
        
        # Create a mapping from activity index to activity name
        activity_labels = {}
        for name, one_hot in activity_map.items():
            if sum(one_hot) == 1: # Ensure it's a valid activity encoding
                activity_labels[np.argmax(one_hot)] = name
        
        cmap2 = plt.get_cmap('tab10', len(unique_activities))
        scatter2 = ax2.scatter(
            single_person_reps[:, 0],
            single_person_reps[:, 1],
            c=activity_indices,
            cmap=cmap2,
            alpha=0.8,
            s=40
        )
        
        cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=unique_activities, label="Activity")
        cbar2.ax.set_yticklabels([activity_labels.get(i, "Unknown") for i in unique_activities])
        cbar2.ax.tick_params(labelsize=10)
        ax2.set_title(f"t-SNE for Single Person by Activity (Epoch {epoch})")
    else:
        ax2.text(0.5, 0.5, "No single-person data available", ha='center', va='center')
        ax2.set_title("t-SNE for Single Person by Activity")

    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")

    fig.suptitle(f'JEPA t-SNE Visualizations (Epoch {epoch})', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1])

    model.train()  # Return to training mode
    return {"tsne_visualizations": fig}


def compute_representation_svd_stats(model, dataloader, device, max_samples=1000):
    """
    Compute simplified SVD statistics on model representations for monitoring representation collapse

    Args:
        model: JEPA model
        dataloader: DataLoader for sampling data
        device: torch device
        max_samples: Maximum number of samples to use for SVD analysis

    Returns:
        dict: Simplified SVD statistics for logging (only 3 key metrics)
    """
    model.eval()
    representations = []
    samples_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list):
                data_batch_x = batch[0]
            elif isinstance(batch, tuple):
                data_batch_x = batch[0]
            else:
                data_batch_x = batch
            data_batch_x = data_batch_x.to(device)

            # Extract representations using the target encoder
            batch_representations = model.extract_representations(data_batch_x)
            representations.append(batch_representations.cpu().numpy())

            samples_processed += data_batch_x.shape[0]
            if samples_processed >= max_samples:
                break

    if not representations:
        return {}

    # Concatenate all representations
    representations = np.concatenate(representations, axis=0)

    # Compute SVD
    U, s, Vt = np.linalg.svd(representations, full_matrices=False)

    # Compute simplified statistics focused on collapse detection
    total_variance = np.sum(s ** 2)
    explained_variance_ratio = (s ** 2) / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Effective rank (number of singular values needed to explain 90% of variance)
    effective_rank = np.argmax(cumulative_variance >= 0.9) + 1

    stats = {
        'svd/effective_rank': int(effective_rank),
        'svd/condition_number': float(s[0] / s[-1]) if s[-1] > 1e-10 else float('inf'),
        'svd/top_singular_value_ratio': float(s[0] / np.sum(s)),  # Dominance of first component
    }

    model.train()  # Return to training mode
    return stats

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """Save checkpoint to disk

    Args:
        state (dict): Contains model state_dict, optimizer state_dict, epoch, best_loss, etc.
        is_best (bool): Whether this checkpoint is the best so far
        checkpoint_dir (str): Directory to save the checkpoint
        filename (str): Filename for the checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copy(os.path.join(checkpoint_dir, filename),
                    os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f"Saved best model to {os.path.join(checkpoint_dir, 'best_model.pth')}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint from disk

    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optimizer, optional): Optimizer to load state into

    Returns:
        dict: Checkpoint contents with epoch, best_loss, etc.
    """
    if not os.path.exists(checkpoint_path):
        return None

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

