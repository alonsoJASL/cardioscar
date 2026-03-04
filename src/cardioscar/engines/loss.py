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

    Fully vectorized via ``torch.scatter_add`` — no Python loops over groups.
    Runs on whichever device the input tensors reside on (CPU or CUDA).

    Args:
        predictions: (batch_size, 1) network predictions
        targets: (batch_size, 1) target scar values
        group_ids: (batch_size,) group assignment for each node
        return_per_group: If True, return per-group losses instead of mean

    Returns:
        Scalar loss tensor (or (n_groups,) per-group losses if
        return_per_group=True)

    Example:
        >>> predictions = torch.tensor([[0.7], [0.65], [0.75], [0.3]])
        >>> targets = torch.tensor([[0.7], [0.7], [0.7], [0.3]])
        >>> group_ids = torch.tensor([0, 0, 0, 1])
        >>> loss = compute_group_reconstruction_loss(predictions, targets, group_ids)
        >>> loss.item()  # Close to 0 if group means match targets
        0.0

    Notes:
        - All nodes in a group must have the same target value.
        - Groups can have variable sizes (handled automatically).
        - group_ids need not be contiguous or start at 0; they are remapped
          internally to a contiguous range for the scatter operations.
        - This function is differentiable and suitable for backpropagation.
    """
    preds = predictions.squeeze()   # (N,)
    tgts = targets.squeeze()        # (N,)

    # Remap group_ids to a contiguous 0..n_groups-1 range.
    # The batch may contain only a subset of all groups in the dataset,
    # so raw group_ids can have large gaps that would waste memory.
    _, remapped = torch.unique(group_ids, return_inverse=True)  # (N,)
    n_groups: int = int(remapped.max().item()) + 1

    device = preds.device

    # --- Group means via scatter_add ---
    group_sums = torch.zeros(n_groups, device=device).scatter_add(
        0, remapped, preds
    )
    group_counts = torch.zeros(n_groups, device=device).scatter_add(
        0, remapped, torch.ones_like(preds)
    )
    group_means = group_sums / group_counts  # (n_groups,)

    # --- One target per group ---
    # For each group, pick the target of whichever node is encountered last
    # during the scatter (all nodes in a group share the same target, so
    # the choice of node is arbitrary). scatter_ with reduce='replace' is
    # not universally available, so we use a plain scatter on indices instead.
    last_occurrence = torch.zeros(n_groups, dtype=torch.long, device=device)
    last_occurrence.scatter_(
        0,
        remapped,
        torch.arange(len(remapped), device=device)
    )
    group_targets = tgts[last_occurrence]  # (n_groups,)

    # --- Per-group squared error ---
    group_losses = (group_targets - group_means) ** 2  # (n_groups,)

    if return_per_group:
        return group_losses

    return group_losses.mean()


def compute_group_reconstruction_loss_vectorized(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    group_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Alias for ``compute_group_reconstruction_loss``.

    The primary implementation is now fully vectorized via ``torch.scatter_add``
    and requires no external dependencies. This function is retained for
    API compatibility and delegates directly.

    Args:
        predictions: (batch_size, 1) network predictions
        targets: (batch_size, 1) target scar values
        group_ids: (batch_size,) group assignment for each node
        group_sizes: Unused. Retained for API compatibility.

    Returns:
        Scalar loss tensor
    """
    return compute_group_reconstruction_loss(predictions, targets, group_ids)