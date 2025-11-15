import numpy as np
import torch
import torch.nn as nn
from configs.preset import preset
class SegmentBlockSampler(nn.Module):
    """
    Samples target blocks and determines context segments for JEPA.
    Ensures that target blocks are contiguous sets of segments and context segments are disjoint from target segments.
    Outputs segment indices (for data extraction) and token indices (as tensors for positional encoding and loss),
    including a padding mask for context tokens.

    Uses dynamic weighting to ensure fair token sampling over multiple calls.
    """

    def __init__(self, weight_decay_factor=0.9):
        super().__init__()
        self.total_segments_in_view = preset["jepa"]["num_segments_total_view"]
        self.num_target_blocks = preset["jepa"]["num_target_blocks"]
        self.target_block_size_segments = preset["jepa"]["target_block_size_segments"]
        self.tokens_per_segment = 1#preset["cnn_embedding_time_dim"]
        self.tokens_per_target_block = self.target_block_size_segments * self.tokens_per_segment
        self.weight_decay_factor = weight_decay_factor  # Factor to reduce weights of sampled indices

        # Initialize dynamic weights for fair sampling
        S = self.target_block_size_segments
        T = self.total_segments_in_view
        self.num_possible_starts = T - S + 1

        # Start with uniform weights
        self.sampling_weights = np.ones(self.num_possible_starts, dtype=float) / self.num_possible_starts

    def _update_weights_after_sampling(self, start_segment):
        """
        Updates the sampling weights after a segment has been sampled.
        Reduces weights for the sampled segment and its block to achieve fairer sampling.
        """
        # Reduce weights for all segments in the sampled block
        block_end = min(start_segment + self.target_block_size_segments, self.num_possible_starts)

        # Reduce weights for overlapping starting positions
        self.sampling_weights[start_segment:block_end] *= self.weight_decay_factor

        # Ensure no weight becomes zero (add small epsilon)
        min_weight = 1e-4
        self.sampling_weights = np.maximum(self.sampling_weights, min_weight)

        # Normalize weights to maintain probability distribution
        self.sampling_weights = self.sampling_weights / np.sum(self.sampling_weights)

    def _get_fair_start_segment(self):
        """
        Selects a starting segment using the dynamic weighted probability distribution
        to give all tokens a fairer chance of being selected over multiple samples.
        """
        possible_start_indices = np.arange(self.num_possible_starts)

        # Sample using current weights
        start_segment = np.random.choice(possible_start_indices, p=self.sampling_weights)

        # Update weights for future sampling
        self._update_weights_after_sampling(start_segment)

        return start_segment

    def reset_weights(self):
        """
        Resets the sampling weights to uniform distribution.
        Useful for starting fresh or after a certain number of samples.
        """
        self.sampling_weights = np.ones(self.num_possible_starts, dtype=float)
        self.sampling_weights = self.sampling_weights / np.sum(self.sampling_weights)

    def get_weight_statistics(self):
        """
        Returns statistics about current sampling weights for monitoring fairness.
        """
        return {
            'min_weight': np.min(self.sampling_weights),
            'max_weight': np.max(self.sampling_weights),
            'weight_std': np.std(self.sampling_weights),
            'weight_mean': np.mean(self.sampling_weights),
            'weights': self.sampling_weights.copy()
        }

    def forward(self, batch_size, device=torch.device("cpu")):
        # Stores lists of segment indices for target blocks for each item in the batch.
        batch_target_segment_indices_for_loss_list = []
        # Stores lists of segment indices for context for each item in the batch.
        # Stores lists of token indices for target blocks (before converting to tensor).
        batch_target_token_indices_list_of_lists = []
        # Stores lists of token indices for context (before converting to tensor).
        batch_context_token_indices_list_of_lists = []

        max_context_tokens_for_batch = 0
        context_mask = torch.ones(batch_size, self.total_segments_in_view, device=device)
        # for _ in range(self.num_target_blocks):
        # start_segment = self._get_fair_start_segment() # This should return a M by num_blocks as starting_point of
                                                        # where we mask the context



        for sample_iter in range(batch_size):
            all_possible_segment_indices = list(range(self.total_segments_in_view))
            current_target_blocks_segments_for_item = []
            current_target_blocks_tokens_for_item_list = []  # List of lists (tokens per block)
            union_of_target_segments_for_item = set()

            # Sample `num_target_blocks` target blocks.
            for target_iter in range(self.num_target_blocks):
                start_segment = self._get_fair_start_segment()
                # Define the segments belonging to this block.
                block_segments = list(range(start_segment, start_segment + self.target_block_size_segments))
                current_target_blocks_segments_for_item.append(block_segments)
                union_of_target_segments_for_item.update(block_segments)

                # Convert segment indices to token indices for this block
                block_token_indices = list(range(start_segment * self.tokens_per_segment,
                                                 (
                                                             start_segment + self.target_block_size_segments) * self.tokens_per_segment))
                current_target_blocks_tokens_for_item_list.append(block_token_indices)

            # Context segments are those not included in any target block.
            context_segments_for_item = [s for s in all_possible_segment_indices if
                                         s not in union_of_target_segments_for_item]
            context_mask[sample_iter,  context_segments_for_item] = 0



            batch_target_segment_indices_for_loss_list.append(current_target_blocks_segments_for_item)
            batch_target_token_indices_list_of_lists.append(current_target_blocks_tokens_for_item_list)



        # Convert target token indices to tensor
        # Shape: (batch_size, num_target_blocks, tokens_per_target_block)
        target_block_token_indices_tensor = torch.zeros(
            (batch_size, self.num_target_blocks, self.tokens_per_target_block), dtype=torch.long, device=device
        )
        for i, item_blocks_list in enumerate(batch_target_token_indices_list_of_lists):
            for block_idx, block_tokens_list in enumerate(item_blocks_list):
                if len(block_tokens_list) != self.tokens_per_target_block:
                    raise ValueError(f"Sampler generated a target block with {len(block_tokens_list)} tokens, "
                                     f"but expected {self.tokens_per_target_block} tokens.")
                target_block_token_indices_tensor[i, block_idx, :] = torch.tensor(block_tokens_list, dtype=torch.long,
                                                                                  device=device)

        return {
            "target_block_segment_indices_for_loss": batch_target_segment_indices_for_loss_list,
            "context_mask": context_mask.bool(), # VERY IMPORTANT:  THIS MASK is global igg
            "target_block_token_indices_tensor": target_block_token_indices_tensor,
        }
        """
        LET ME DIG INTO IT.
        LETS SAY WE HAVE 74 POSSIBLE TOKENS. FROM THESE TOKENS WE MIGHT SELECT 10, 12, 20, 42 TARGETS TOKENS. 
        THE TARGET TOKENS (HERE SEGMENT) WILL BE SAVED WITHING TARGET target_block_segment_indices_for_loss,
        
        IMPORTANTLY SINCE WE NEED CONTEXT AS A TENSORE AND NOW OUR CONTEXT TENSOR HAS A VARIABLE LENGTH, WE WOULD 
        
        """
