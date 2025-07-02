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

    def forward(self, model_outputs, targets, model):
        """
        Calculate combined supervised + SSL loss for hybrid JEPA models

        Args:
            model_outputs: Dictionary containing hybrid model outputs
            targets: Ground truth labels
            model: The JEPA_Sup model (needed for predictor)

        Returns:
            Dictionary containing total loss and all components
        """
        # Extract outputs from hybrid model
        outputs_class = model_outputs["outputs_class"]
        z_context_online = model_outputs["z_context_online"]
        actual_targets_for_loss = model_outputs["actual_targets_for_loss"]
        sampling_info = model_outputs["sampling_info"]

        # Calculate supervised loss
        supervised_loss_val = self.supervised_loss_fn(outputs_class, targets.float())

        # Initialize SSL loss components
        jepa_loss_val = torch.tensor(0.0, device=outputs_class.device)
        jepa_sim_loss = torch.tensor(0.0, device=outputs_class.device)
        jepa_std_loss = torch.tensor(0.0, device=outputs_class.device)
        jepa_cov_loss = torch.tensor(0.0, device=outputs_class.device)

        # Calculate JEPA loss (only if we have context tokens)
        if z_context_online.shape[1] > 0:
            # JEPA prediction setup
            context_padding_mask = sampling_info["context_padding_mask_tensor"]
            context_original_indices = sampling_info["context_token_indices_tensor"]

            b, num_blocks, tokens_per_block, d = actual_targets_for_loss.shape

            # Reshape for prediction
            target_indices = sampling_info["target_block_token_indices_tensor"].reshape(b * num_blocks,
                                                                                        tokens_per_block)
            actual_targets = actual_targets_for_loss.reshape(b * num_blocks, tokens_per_block, d)

            # Expand context to match target blocks
            expanded_context = z_context_online.repeat_interleave(repeats=num_blocks, dim=0)
            expanded_context_mask = context_padding_mask.repeat_interleave(repeats=num_blocks, dim=0)
            expanded_context_indices = context_original_indices.repeat_interleave(repeats=num_blocks, dim=0)

            # JEPA predictions
            predictions = model.predictor(
                expanded_context,
                expanded_context_mask,
                target_indices,
                expanded_context_indices
            )

            # Calculate JEPA loss with VICReg regularization
            jepa_loss_dict = self.jepa_loss_fn(
                predictions=predictions,
                actual_targets=actual_targets,
                z_context_online=z_context_online,
                context_padding_mask=context_padding_mask
            )
            jepa_loss_val = jepa_loss_dict['total_loss']
            jepa_sim_loss = jepa_loss_dict['vicreg_sim_loss']
            jepa_std_loss = jepa_loss_dict['vicreg_std_loss']
            jepa_cov_loss = jepa_loss_dict['vicreg_cov_loss']

        # Combined loss
        total_loss = supervised_loss_val + self.SSL_coeff * jepa_loss_val

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss_val,
            'ssl_loss': jepa_loss_val,
            'jepa_sim_loss': jepa_sim_loss,
            'jepa_std_loss': jepa_std_loss,
            'jepa_cov_loss': jepa_cov_loss
        }