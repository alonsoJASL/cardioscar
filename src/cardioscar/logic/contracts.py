# src/cardioscar/logic/contracts.py

"""
Data Contracts for Scar Reconstruction

These dataclasses define the explicit contracts for passing data between
orchestrators, logic layers, and utilities. They make dependencies clear
and enable type checking.

Following pycemrg principles:
- No hidden I/O
- Explicit path/data contracts
- Stateless - just data containers
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch


# =============================================================================
# PREPROCESSING CONTRACTS
# =============================================================================

@dataclass
class PreprocessingRequest:
    """
    Request to create training data from mesh and slices.
    
    Attributes:
        mesh_path: Path to 3D target mesh (VTK)
        grid_layer_paths: List of paths to 2D grid layers (VTK)
        slice_thickness_padding: Z-direction padding (mm) for slice thickness
    """
    mesh_path: Path
    grid_layer_paths: List[Path]
    slice_thickness_padding: float = 5.0


@dataclass
class PreprocessingResult:
    """
    Result of spatial mapping preprocessing.
    
    Attributes:
        coordinates: (N, 3) normalized mesh coordinates
        intensities: (N, 1) target scar values
        group_ids: (N,) group assignment per node
        group_sizes: (M,) number of nodes per unique group
        scaler_min: (3,) minimum values for denormalization
        scaler_max: (3,) maximum values for denormalization
        n_nodes: Total number of nodes
        n_groups: Total number of unique groups
    """
    coordinates: np.ndarray
    intensities: np.ndarray
    group_ids: np.ndarray
    group_sizes: np.ndarray
    scaler_min: np.ndarray
    scaler_max: np.ndarray
    n_nodes: int
    n_groups: int

# Training request contracts are in 
# src/cardioscar/training/config.py


# =============================================================================
# INFERENCE CONTRACTS
# =============================================================================

@dataclass
class InferenceRequest:
    """
    Request to apply trained model to mesh.
    
    Attributes:
        model_checkpoint_path: Path to trained .pth checkpoint
        mesh_path: Path to input mesh (VTK)
        mc_samples: Number of MC Dropout samples for uncertainty
        batch_size: Batch size for inference
        threshold: Optional threshold for binary scar classification
    """
    model_checkpoint_path: Path
    mesh_path: Path
    mc_samples: int = 10
    batch_size: int = 50000
    threshold: Optional[float] = None


@dataclass
class InferenceResult:
    """
    Result of model inference on mesh.
    
    Attributes:
        mean_predictions: (N,) mean scar probability per node
        std_predictions: (N,) uncertainty (std) per node
        binary_predictions: (N,) optional binary scar classification
        n_nodes: Total number of nodes
        mean_scar_probability: Global mean scar probability
        mean_uncertainty: Global mean uncertainty
    """
    mean_predictions: np.ndarray
    std_predictions: np.ndarray
    binary_predictions: Optional[np.ndarray]
    n_nodes: int
    mean_scar_probability: float
    mean_uncertainty: float