# src/cardioscar/engines/loss.py

"""
Group-Based Reconstruction Loss

Custom loss function that enforces the physical constraint:
    mean(predictions[group]) = target_pixel_value

This ensures that all 3D mesh nodes within a 2D pixel's spatial extent
(a "group") have predictions that average to the observed pixel value.

Reference:
    Sen et al. (2025) "Weakly supervised learning for scar reconstruction"
    https://doi.org/10.1016/j.compbiomed.2025.111219
"""

import torch
from typing import Optional


def compute_group_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    return_per_group: bool = False
) -> torch.Tensor:
    """
    Compute group-constrained reconstruction loss.
    
    For each unique group (pixels in 2D slice), computes:
        loss_group = (target - mean(predictions[group]))²
    
    Final loss is the mean across all groups in the batch.
    
    Args:
        predictions: (batch_size, 1) network predictions
        targets: (batch_size, 1) target scar values
        group_ids: (batch_size,) group assignment for each node
        return_per_group: If True, return per-group losses instead of mean
    
    Returns:
        Scalar loss tensor (or per-group losses if return_per_group=True)
    
    Example:
        >>> predictions = torch.tensor([[0.7], [0.65], [0.75], [0.3]])
        >>> targets = torch.tensor([[0.7], [0.7], [0.7], [0.3]])
        >>> group_ids = torch.tensor([0, 0, 0, 1])
        >>> loss = compute_group_reconstruction_loss(predictions, targets, group_ids)
        >>> loss.item()  # Close to 0 if group means match targets
        0.0
    
    Notes:
        - All nodes in a group should have the same target value
        - Groups can have variable sizes (handled automatically)
        - This function is differentiable and suitable for backpropagation
    """
    # Find unique groups in this batch
    unique_groups = torch.unique(group_ids)
    
    group_losses = []
    
    for group_id in unique_groups:
        # Get all nodes belonging to this group
        mask = (group_ids == group_id)
        group_preds = predictions[mask]
        group_target = targets[mask][0]  # All nodes in group have same target
        
        # Compute mean prediction for this group
        group_mean = group_preds.mean()
        
        # Squared error between target and group mean
        group_loss = (group_target - group_mean) ** 2
        group_losses.append(group_loss)
    
    # Stack and aggregate
    stacked_losses = torch.stack(group_losses)
    
    if return_per_group:
        return stacked_losses
    
    return stacked_losses.mean()


def compute_group_reconstruction_loss_vectorized(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    group_sizes: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Vectorized version of group reconstruction loss (faster for large batches).
    
    Uses scatter operations to compute group means in parallel.
    Requires torch-scatter: pip install torch-scatter
    
    Args:
        predictions: (batch_size, 1) network predictions
        targets: (batch_size, 1) target scar values
        group_ids: (batch_size,) group assignment for each node
        group_sizes: Optional (n_groups,) pre-computed group sizes
    
    Returns:
        Scalar loss tensor
    
    Note:
        This is a placeholder for future optimization. Currently not used.
        Uncomment when torch-scatter is added as a dependency.
    """
    raise NotImplementedError(
        "Vectorized loss requires torch-scatter. "
        "Use compute_group_reconstruction_loss() instead, or install: "
        "pip install torch-scatter"
    )
    
    # Future implementation with torch-scatter:
    # from torch_scatter import scatter_mean
    # 
    # # Compute mean prediction per group
    # group_means = scatter_mean(predictions.squeeze(), group_ids, dim=0)
    # 
    # # Get one target per group (all nodes in group have same target)
    # unique_groups = torch.unique(group_ids)
    # group_targets = torch.zeros_like(group_means)
    # for i, gid in enumerate(unique_groups):
    #     mask = (group_ids == gid)
    #     group_targets[i] = targets[mask][0]
    # 
    # # MSE between group means and targets
    # loss = ((group_targets - group_means) ** 2).mean()
    # return loss