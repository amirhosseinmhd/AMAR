import torch.nn as nn
import torch
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def forward(self, block_embeddings):
        """
        Apply VICReg to block embeddings treating each block as different view
        Args:
            block_embeddings: (batch_size, num_blocks, feature_dim)
        """
        batch_size, num_blocks, feature_dim = block_embeddings.shape
        
        total_loss = 0.0
        num_pairs = 0
        std_loss_val = 0.0
        cov_loss_val = 0.0
        # Iterate over all pairs of blocks (i, j) where i != j
        for i in range(num_blocks):
            for j in range(num_blocks):
                if i != j:
                    block_i = block_embeddings[:, i, :]  # (batch_size, feature_dim)
                    block_j = block_embeddings[:, j, :]  # (batch_size, feature_dim)
                    
                    # Invariance (similarity) loss between blocks i and j
                    sim_loss = F.mse_loss(block_i, block_j)
                    
                    # Variance loss for both blocks
                    std_i = torch.sqrt(block_i.var(dim=0) + self.eps)
                    std_j = torch.sqrt(block_j.var(dim=0) + self.eps)
                    std_loss = (torch.mean(F.relu(self.gamma - std_i)) + 
                               torch.mean(F.relu(self.gamma - std_j))) / 2
                    
                    # Covariance loss for both blocks
                    cov_loss = (self._covariance_loss(block_i) + 
                               self._covariance_loss(block_j)) / 2
                    
                    # Combine losses for this pair
                    pair_loss = (self.sim_coeff * sim_loss + 
                                self.std_coeff * std_loss + 
                                self.cov_coeff * cov_loss)
                    std_loss_val += std_loss
                    cov_loss_val += cov_loss
                    total_loss += pair_loss
                    num_pairs += 1
        
        # Average over all pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
            std_loss = std_loss_val / num_pairs
            cov_loss = cov_loss_val / num_pairs
            
        return total_loss, std_loss, cov_loss
    
    def _covariance_loss(self, z):
        """Compute covariance loss for decorrelation"""
        batch_size, feature_dim = z.shape
        z_centered = z - z.mean(dim=0)
        cov_matrix = torch.mm(z_centered.T, z_centered) / (batch_size - 1)
        off_diag_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=z.device)
        return (cov_matrix[off_diag_mask] ** 2).sum() / feature_dim

class CombinedJEPALoss(nn.Module):
    """
    Combined loss for JEPA with VICReg regularization following C-JEPA pseudocode:
    
    The invariance loss is the same as the original JEPA prediction loss (MSE between predictions and targets).
    VICReg adds variance and covariance regularization to context embeddings.
    
    total_loss = prediction_loss + beta_vicreg * (beta_std * std_loss + beta_cov * cov_loss)
    """
    
    def __init__(self, 
                 prediction_coeff=1.0,
                 vicreg_coeff=0.001,
                 vicreg_std_coeff=25.0,
                 vicreg_cov_coeff=1.0):
        """
        Args:
            prediction_coeff: Coefficient for original JEPA prediction loss (includes invariance)
            vicreg_coeff: Coefficient for VICReg loss (β_vicreg in paper)
            vicreg_std_coeff: VICReg variance coefficient
            vicreg_cov_coeff: VICReg covariance coefficient
        """
        super().__init__()
        self.prediction_coeff = prediction_coeff
        self.vicreg_coeff = vicreg_coeff
        
        # Only need variance and covariance components since invariance = prediction loss
        self.vicreg_loss = VICRegLoss(
            std_coeff=vicreg_std_coeff,
            cov_coeff=vicreg_cov_coeff
        )
    
    def pool_representations(self, representations, mask=None):
        """
        Pool representations by averaging, considering padding mask if provided
        
        Args:
            representations: Tensor of shape (batch_size, seq_len, feature_dim)
            mask: Optional padding mask (True for padded positions)
        
        Returns:
            Pooled representations of shape (batch_size, feature_dim)
        """
        if mask is not None:
            # Mask out padded positions
            mask_expanded = mask.unsqueeze(-1).expand_as(representations)
            representations_masked = representations.masked_fill(mask_expanded, 0)
            
            # Count valid positions
            valid_counts = (~mask).sum(dim=1, keepdim=True).float()
            valid_counts = torch.clamp(valid_counts, min=1)  # Avoid division by zero
            
            # Average only over valid positions
            pooled = representations_masked.sum(dim=1) / valid_counts
        else:
            # Simple average if no mask
            pooled = representations.mean(dim=1)
            
        return pooled
    
    def forward(self, predictions, actual_targets, z_context_online, context_padding_mask, projector):
        """
        Compute combined JEPA + VICReg loss where:
        - Invariance loss is applied to predicted tokens vs actual target tokens (replaces JEPA prediction loss)
        - Variance/Covariance losses are applied to context embeddings
        
        Args:
            predictions: Predicted target representations from predictor b , num_blocks, tokens_per_block, d
            actual_targets: Actual target representations (ground truth)  shape: b * num_blocks, tokens_per_block, d
            z_context_online: Context representations, shape (batch_size, context_len, feature_dim)
            context_padding_mask: Padding mask for context, shape (batch_size, context_len)
        
        Returns:
            dict: Dictionary with total loss and individual components
        """
        batch_size, num_blocks, num_target_tokens, dim_context = predictions.shape
        
        # 1. JEPA prediction loss (MSE between predictions and targets)
        predictions_flat = predictions.view(-1, num_target_tokens, dim_context)
        jepa_loss = F.mse_loss(predictions_flat, actual_targets)
        
        # 2. Prepare block embeddings for VICReg
        # Average within each block: (batch_size, num_blocks, dim)
        block_embeddings = predictions.mean(dim=2)  # Average over tokens_per_block
        
        # Apply projector to each block embedding
        block_embeddings_flat = block_embeddings.view(-1, dim_context)  # (batch_size * num_blocks, dim)
        projected_flat = projector(block_embeddings_flat)  # (batch_size * num_blocks, dim)
        projected_blocks = projected_flat.view(batch_size, num_blocks, dim_context)  # (batch_size, num_blocks, dim)
        
        # 3. Apply VICReg to projected block embeddings
        vicreg_loss, std_loss, cov_loss = self.vicreg_loss(projected_blocks)
        
        # 4. Combine losses
        total_loss = (1 - self.vicreg_coeff) * jepa_loss + self.vicreg_coeff * vicreg_loss
        
        return total_loss, jepa_loss, vicreg_loss, std_loss, cov_loss  # Return vicreg_loss twice for compatibility
