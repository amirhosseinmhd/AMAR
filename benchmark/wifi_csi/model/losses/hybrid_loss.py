import torch
import torch.nn as nn


class CombinedHybridLoss(nn.Module):
    """
    Combined loss function for hybrid JEPA_Sup model that handles both:
    1. Supervised loss (HungarianMatchingLoss)
    2. SSL loss (CombinedJEPALoss for JEPA + VICReg)
    """

    def __init__(self,
                 supervised_loss_fn,
                 jepa_loss_fn,
                 SSL_coeff=1.0):
        """
        Args:
            supervised_loss_fn: Instance of HungarianMatchingLoss
            jepa_loss_fn: Instance of CombinedJEPALoss
            SSL_coeff: Coefficient for SSL loss component
        """
        super().__init__()
        self.supervised_loss_fn = supervised_loss_fn
        self.jepa_loss_fn = jepa_loss_fn
        self.SSL_coeff = SSL_coeff

    def forward(self, model_outputs, targets_supervised):
        """
        Calculate combined supervised + SSL loss for hybrid JEPA models

        Args:
            model_outputs: Dictionary containing hybrid model outputs
            targets_supervised: Ground truth labels

        Returns:
            Dictionary containing total loss and all components
        """
        # Extract outputs from hybrid model
        outputs_class = model_outputs["outputs_class"]
        z_context_online = model_outputs["z_context_online"]
        actual_targets_for_loss = model_outputs["actual_targets_for_loss"]
        sampling_info = model_outputs["sampling_info"]
        context_mask = sampling_info["context_mask"]
        ssl_predictions = model_outputs["ssl_predictions"]
        batch_size, num_target_blocks, num_target_tokens, dim_context = ssl_predictions.shape

        # Calculate supervised loss
        supervised_loss_val = self.supervised_loss_fn(outputs_class, targets_supervised.float())

        # target_indices = sampling_info["target_block_token_indices_tensor"].reshape(b * num_blocks,
        #                                                                             tokens_per_block)

        actual_targets = actual_targets_for_loss.reshape(-1, num_target_tokens, dim_context)

        total_loss_jepa,  predictive_loss, jepa_std_loss, jepa_cov_loss = self.jepa_loss_fn(
            predictions=ssl_predictions,
            actual_targets=actual_targets,
            z_context_online=z_context_online,
            context_padding_mask=context_mask
        )


        total_loss = (1-self.SSL_coeff) * supervised_loss_val + self.SSL_coeff * total_loss_jepa

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss_val,
            'ssl_loss': total_loss_jepa,
            'jepa_predictive_loss': predictive_loss,
            'jepa_std_loss': jepa_std_loss,
            'jepa_cov_loss': jepa_cov_loss
        }