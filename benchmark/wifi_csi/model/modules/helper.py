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



def generate_tsne_visualizations(model, environments, device, epoch):
    """
    Generate t-SNE visualizations for each environment and a combined visualization
    in a single figure with subplots.

    Args:
        model: The JEPA model for generating representations
        environments: List of environments to visualize
        device: Device to run the model on
        epoch: Current training epoch (for logging)

    Returns:
        dict: Dictionary containing a single matplotlib figure for wandb logging
    """
    model.eval()  # Set model to evaluation mode

    num_envs = len(environments)
    has_combined_plot = num_envs > 1
    num_plots = num_envs + 1 if has_combined_plot else 1

    # Determine grid size for subplots
    if num_plots <= 1:
        nrows, ncols = 1, 1
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = math.ceil(num_plots / 2)
        ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows), squeeze=False)
    axes = axes.flatten()

    all_data_x = []
    all_environment_labels = []

    # Process each environment separately
    for i, env in enumerate(environments):
        ax = axes[i]
        print(f"Generating t-SNE visualization for {env}...")

        # Load data for this environment
        data_pd_y = load_data_y(
            preset["path"]["data_y"],
            var_environment=[env],
            var_wifi_band=preset["data"]["wifi_band"],
            var_num_users=preset["data"]["num_users"]
        )
        var_label_list = data_pd_y["label"].to_list()
        env_data_x = load_data_x(preset["path"]["data_x"], var_label_list)

        env_data_x = env_data_x.reshape(env_data_x.shape[0], env_data_x.shape[1], -1)

        if has_combined_plot:
            all_data_x.append(env_data_x)
            all_environment_labels.extend([env] * len(env_data_x))

        y = encode_activity(data_pd_y)
        data_y = np.array(y)
        num_people = data_y.sum(axis=1).sum(axis=1)

        dataset = TensorDataset(torch.from_numpy(env_data_x).float())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_representations = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch_x = batch[0].to(device)
                representations = model.extract_representations(data_batch_x)
                all_representations.append(representations.cpu().numpy())

        representations_np = np.concatenate(all_representations, axis=0)

        tsne_model = TSNE(
            n_components=2,
            perplexity=min(50, max(5, len(representations_np) // 5)),
            random_state=42,
            n_jobs=-1,
            init='pca',
            learning_rate='auto'
        )
        tsne_results = tsne_model.fit_transform(representations_np)

        unique_labels = np.unique(num_people)
        cmap = plt.get_cmap('tab10', max(8, len(unique_labels)))

        scatter = ax.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=num_people,
            cmap=cmap,
            alpha=0.8,
            s=40,
            vmin=unique_labels.min() - 0.5,
            vmax=unique_labels.max() + 0.5
        )

        cbar = fig.colorbar(scatter, ax=ax, ticks=unique_labels, label="Number of People")
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(f"t-SNE of JEPA Reps - {env} (Epoch {epoch})")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

    # Generate combined visualization if there are multiple environments
    if has_combined_plot:
        ax = axes[num_envs]
        combined_data_x = np.concatenate(all_data_x, axis=0)

        dataset = TensorDataset(torch.from_numpy(combined_data_x).float())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_representations = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch_x = batch[0].to(device)
                representations = model.extract_representations(data_batch_x)
                all_representations.append(representations.cpu().numpy())

        representations_np = np.concatenate(all_representations, axis=0)

        tsne_model = TSNE(
            n_components=2,
            perplexity=min(50, max(5, len(representations_np) // 5)),
            # max_iter=1000,
            random_state=42,
            n_jobs=-1,
            init='pca',
            learning_rate='auto'
        )
        tsne_results = tsne_model.fit_transform(representations_np)

        unique_envs = list(set(all_environment_labels))
        env_to_id = {env: i for i, env in enumerate(unique_envs)}
        env_ids = [env_to_id[env] for env in all_environment_labels]

        cmap = plt.get_cmap('tab10', len(unique_envs))

        scatter = ax.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=env_ids,
            cmap=cmap,
            alpha=0.8,
            s=40,
            vmin=-0.5,
            vmax=len(unique_envs) - 0.5
        )

        cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(unique_envs)), label="Environment")
        cbar.ax.set_yticklabels(unique_envs)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(f"t-SNE of JEPA Reps - All Environments (Epoch {epoch})")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

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
            # Handle both tuple and tensor batch formats
            if isinstance(batch, tuple):
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
#

