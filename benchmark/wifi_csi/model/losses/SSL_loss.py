import torch.nn as nn
import torch
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """
    VICReg Loss implementation for C-JEPA following the exact pseudocode:
    - sim_loss: between context and target embeddings
    - std_loss: applied only to context embeddings
    - cov_loss: applied only to context embeddings
    """
    
    def __init__(self, 
                 std_coeff=25.0,
                 cov_coeff=1.0,
                 gamma=1.0,
                 eps=1e-4):
        """
        Args:
            sim_coeff: Coefficient for similarity (invariance) loss
            std_coeff: Coefficient for variance loss  
            cov_coeff: Coefficient for covariance loss
            gamma: Target standard deviation for variance regularization
            eps: Small value for numerical stability
        """
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    
    def variance_loss(self, z):
        """
        Variance loss: Ensures representation variance doesn't collapse
        Args:
            z: Tensor of shape (batch_size, feature_dim)
        """
        batch_size, feature_dim = z.shape
        std_z = torch.sqrt(z.var(dim=0) + self.eps)  # Standard deviation along batch dimension
        std_loss = torch.mean(F.relu(self.gamma - std_z))  # Hinge loss
        return std_loss
    
    def covariance_loss(self, z):
        """
        Covariance loss: Decorrelates features by minimizing off-diagonal covariance
        Args:
            z: Tensor of shape (batch_size, feature_dim)
        """
        batch_size, feature_dim = z.shape
        z_centered = z - z.mean(dim=0)  # Center the features
        
        # Compute covariance matrix
        cov_matrix = torch.mm(z_centered.T, z_centered) / (batch_size - 1)
        
        # Zero out diagonal and sum squared off-diagonal elements
        off_diag_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=z.device)
        cov_loss = (cov_matrix[off_diag_mask] ** 2).sum() / feature_dim
        
        return cov_loss
    
    def forward(self, z_context_pooled):
        """
        Compute VICReg regularization terms (variance and covariance only):
        - std_loss: applied to context embeddings  
        - cov_loss: applied to context embeddings
        
        Note: Invariance/similarity loss is handled separately as the main prediction loss
        
        Args:
            z_context_pooled: Pooled context representations, shape (batch_size, feature_dim)
        
        Returns:
            dict: Dictionary with total loss and individual components
        """
        # VICReg regularization terms applied only to context embeddings
        std_loss = self.variance_loss(z_context_pooled)
        cov_loss = self.covariance_loss(z_context_pooled)
        
        # Total VICReg regularization (variance + covariance only)
        total_loss = (self.std_coeff * std_loss + self.cov_coeff * cov_loss)
        
        return {
            'std_loss': std_loss,
            'cov_loss': cov_loss,
            'total_loss': total_loss,
        }

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
    
    def forward(self, predictions, actual_targets, z_context_online, context_padding_mask):
        """
        Compute combined JEPA + VICReg loss where:
        - Invariance loss is applied to predicted tokens vs actual target tokens (replaces JEPA prediction loss)
        - Variance/Covariance losses are applied to context embeddings
        
        Args:
            predictions: Predicted target representations from predictor
            actual_targets: Actual target representations (ground truth)
            z_context_online: Context representations, shape (batch_size, context_len, feature_dim)
            context_padding_mask: Padding mask for context, shape (batch_size, context_len)
        
        Returns:
            dict: Dictionary with total loss and individual components
        """
        # 1. Invariance loss: between predicted tokens and actual target tokens (this IS the prediction loss)
        batch_size, num_blocks, num_target_tokens, dim_context = predictions.shape
        invariance_loss = F.mse_loss(predictions.view(-1, num_target_tokens, dim_context), actual_targets)
        
        # 2. Pool context representations for variance/covariance losses
        z_context_pooled = self.pool_representations(z_context_online, context_padding_mask)
        
        # 3. Apply VICReg variance and covariance losses to context embeddings
        vicreg_losses = self.vicreg_loss(z_context_pooled)
        
        # 4. Combined loss: invariance (prediction) + VICReg regularization terms
        total_loss = self.prediction_coeff * invariance_loss + self.vicreg_coeff * vicreg_losses["total_loss"]
        
        return total_loss, invariance_loss,  vicreg_losses['std_loss'], vicreg_losses['cov_loss']
