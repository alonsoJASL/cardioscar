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

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union 

import numpy as np

# =============================================================================
# PREPROCESSING CONTRACTS
# =============================================================================

class SliceInputType(Enum): 
    """Type of Slice Input"""
    VTK_GRIDS = "vtk_grids"
    IMAGE = "image" # nifti or nrrd 
    DICOM_SERIES = "dicom_series" # TODO: implement DICOM stack

@dataclass
class PreprocessingRequest:
    """Request to create training data from mesh and slices."""
    mesh_path: Path
    slice_thickness_padding: float = 5.0
    
    # SliceInputType.VTK_GRIDS
    vtk_grid_paths: Optional[List[Path]] = None
    vtk_scalar_field: str = "scalars" 

    # SliceInputType.IMAGE 
    image_path: Optional[Path] = None
    slice_axis: Optional[str] = 'z' 
    slice_indices: Optional[List[int]] = None

    def __post_init__(self):
        """Validate that exactly one input method is provided."""
        vtk_provided = self.vtk_grid_paths is not None
        image_provided = self.image_path is not None
        
        if not (vtk_provided ^ image_provided):
            raise ValueError(
                "Must provide exactly one of: vtk_grid_paths OR image_path"
            )
        
        if image_provided and self.slice_axis is None:
            raise ValueError("slice_axis required when using image_path")

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