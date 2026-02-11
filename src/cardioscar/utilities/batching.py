# src/cardioscar/utilities/batching.py

"""
Complete-Group Batching for Scar Reconstruction

Custom PyTorch Dataset and batching utilities that ensure training batches
contain only complete groups. This preserves the physical constraint that
all nodes within a 2D pixel must be processed together.

Key innovation: Pre-sorts data by group_id and creates batch boundaries
that respect group integrity.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


class ScarReconstructionDataset(Dataset):
    """
    Dataset for scar reconstruction with complete-group batching support.
    
    This dataset pre-sorts nodes by group ID to enable batching complete
    groups together, ensuring the physical constraint is preserved during
    training.
    
    Args:
        coordinates: (N, 3) normalized mesh coordinates
        intensities: (N, 1) target scar values
        group_ids: (N,) group assignment per node
        group_sizes: (M,) number of nodes per unique group
    
    Example:
        >>> coords = np.random.rand(1000, 3)
        >>> intensities = np.random.rand(1000, 1)
        >>> group_ids = np.repeat(np.arange(50), 20)  # 50 groups, 20 nodes each
        >>> group_sizes = np.full(50, 20)
        >>> dataset = ScarReconstructionDataset(coords, intensities, group_ids, group_sizes)
        >>> len(dataset)
        1000
    """
    
    def __init__(
        self,
        coordinates: np.ndarray,
        intensities: np.ndarray,
        group_ids: np.ndarray,
        group_sizes: np.ndarray
    ):
        # Sort by group ID for complete-group batching
        sort_indices = np.argsort(group_ids)
        
        self.coordinates = torch.from_numpy(coordinates[sort_indices]).float()
        self.intensities = torch.from_numpy(intensities[sort_indices]).float()
        self.group_ids = torch.from_numpy(group_ids[sort_indices]).long()
        self.group_sizes = torch.from_numpy(group_sizes).long()
        
        # Compute cumulative sum for fast group indexing
        self.group_cumsum = torch.cat([
            torch.tensor([0]),
            torch.cumsum(self.group_sizes, dim=0)
        ])
    
    def __len__(self) -> int:
        return len(self.coordinates)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single data point.
        
        Returns:
            Tuple of (coordinate, intensity, group_id)
        """
        return (
            self.coordinates[idx],
            self.intensities[idx],
            self.group_ids[idx]
        )


def create_complete_group_batches(
    dataset: ScarReconstructionDataset,
    target_batch_size: int = 10000
) -> List[Tuple[int, int]]:
    """
    Create batch boundaries that contain only complete groups.
    
    Iterates through groups and creates batches that:
    1. Contain complete groups only (no group split across batches)
    2. Target approximately target_batch_size nodes per batch
    3. May vary in actual size to respect group boundaries
    
    Args:
        dataset: ScarReconstructionDataset instance
        target_batch_size: Approximate target batch size (nodes)
    
    Returns:
        List of (start_idx, end_idx) tuples defining each batch
    
    Example:
        >>> dataset = ScarReconstructionDataset(coords, intensities, group_ids, group_sizes)
        >>> batches = create_complete_group_batches(dataset, target_batch_size=5000)
        >>> len(batches)
        6
        >>> batches[0]
        (0, 4993)
    
    Notes:
        - Batch sizes will vary to preserve group integrity
        - If a single group exceeds target_batch_size, it becomes its own batch
        - Last batch may be smaller than target
    """
    batches = []
    current_start = 0
    current_size = 0
    
    for group_idx in range(len(dataset.group_sizes)):
        group_size = dataset.group_sizes[group_idx].item()
        
        # Check if adding this group would exceed target
        if current_size + group_size > target_batch_size and current_size > 0:
            # Finalize current batch
            current_end = dataset.group_cumsum[group_idx].item()
            batches.append((current_start, current_end))
            
            # Start new batch
            current_start = current_end
            current_size = 0
        
        current_size += group_size
    
    # Add final batch if not empty
    if current_size > 0:
        batches.append((current_start, len(dataset)))
    
    return batches


def get_batch_statistics(batches: List[Tuple[int, int]]) -> dict:
    """
    Compute statistics about batch sizes.
    
    Args:
        batches: List of (start, end) batch boundaries
    
    Returns:
        Dictionary with min, max, mean, std of batch sizes
    
    Example:
        >>> batches = [(0, 5000), (5000, 10000), (10000, 12500)]
        >>> stats = get_batch_statistics(batches)
        >>> stats['mean']
        4166.666...
    """
    sizes = [end - start for start, end in batches]
    
    return {
        'n_batches': len(batches),
        'min': min(sizes),
        'max': max(sizes),
        'mean': np.mean(sizes),
        'std': np.std(sizes)
    }