# src/cardioscar/logic/orchestrators.py 

"""
High-Level Orchestrators

Pure orchestration functions that coordinate library components.
No argument parsing (argparse/Click), no logging setup - just business logic.

These functions are called by both:
- scripts/ (argparse-based standalone scripts)
- cli.py (Click-based CLI commands)
- Direct library users (import and call)

Design principles:
- Accept explicit parameters (Paths, primitives, dataclasses)
- Return dataclasses (contracts)
- Handle coordination logic only
- Delegate I/O to utilities
- Delegate computation to engines/training
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

from cardioscar.logic.contracts import (
    PreprocessingRequest,
    PreprocessingResult,
    SliceInputType,
    InferenceRequest,
    InferenceResult
)
from cardioscar.utilities.io import (
    load_mesh_points,
    load_grid_layer_data,
    save_mesh_with_scalars
)
from cardioscar.utilities.preprocessing import (
    compute_group_sizes,
    normalize_coordinates,
    process_vtk_grid_data,
    process_image_slice_data,
)
from cardioscar.utilities.batching import (
    ScarReconstructionDataset,
    create_complete_group_batches
)
from cardioscar.training.config import TrainingConfig
from cardioscar.training.trainer import train_model
from cardioscar.engines import BayesianNN
from cardioscar.logic.reconstruction import ReconstructionLogic

from pycemrg_image_analysis.utilities.geometry import *

logger = logging.getLogger(__name__)

# =============================================================================
# PREPROCESSING ORCHESTRATOR
# =============================================================================

def prepare_training_data(
    request: PreprocessingRequest
) -> PreprocessingResult:
    """Orchestrate training data preparation."""
    
    # 1. Load mesh (I/O)
    logger.info(f"Loading target mesh: {request.mesh_path}")
    mesh_coords = load_mesh_points(request.mesh_path)
    logger.info(f"  Loaded {len(mesh_coords)} mesh nodes")
    
    # 2. Load slice data based on input type (I/O)
    if request.vtk_grid_paths is not None:
        # VTK workflow (existing)
        logger.info(f"Loading {len(request.vtk_grid_paths)} VTK grid layers")
        grid_layers_data = []
        
        for grid_path in sorted(request.vtk_grid_paths):
            cell_bounds, scalar_values = load_grid_layer_data(
                grid_path,
                scalar_field_name=request.vtk_scalar_field
            )
            grid_layers_data.append((cell_bounds, scalar_values))
        
        cell_data = process_vtk_grid_data(
            mesh_coords=mesh_coords,
            grid_layers_data=grid_layers_data,
            z_padding=request.slice_thickness_padding
        )
    
    elif request.image_path is not None:
        # Image workflow (NEW - now implemented!)
        from cardioscar.utilities.io import extract_image_slice_data

        print("\nDEBUG: Extracting image slice data...")
        slice_data = extract_image_slice_data(
            image_path=request.image_path,
            slice_axis='z',
            slice_indices=[0]
        )

        print(f"Number of slices extracted: {len(slice_data)}")
        if len(slice_data) > 0:
            bounds, values = slice_data[0]
            print(f"Voxel bounds shape: {bounds.shape}")
            print(f"Voxel values shape: {values.shape}")
            print(f"First 3 bounds:\n{bounds[:3]}")
            print(f"First 3 values: {values[:3]}")
        else:
            print("ERROR: No slices extracted!")

        
        logger.info(f"Loading image: {request.image_path}")
        slice_layers_data = extract_image_slice_data(
            image_path=request.image_path,
            slice_axis=request.slice_axis,
            slice_indices=request.slice_indices
        )
        
        logger.info(f"  Extracted {len(slice_layers_data)} slices")
        
        cell_data = process_image_slice_data(
            mesh_coords=mesh_coords,
            slice_layers_data=slice_layers_data,
            z_padding=request.slice_thickness_padding
        )
    
    # 3. Post-process (same for both workflows)
    coordinates = cell_data[:, 0:3]
    intensities = cell_data[:, 3:4]
    group_ids = cell_data[:, 4].astype(np.int32)
    
    normalized_coords, scaler = normalize_coordinates(coordinates)
    group_sizes = compute_group_sizes(group_ids)
    
    n_nodes = len(coordinates)
    n_groups = len(group_sizes)
    
    logger.info("Dataset created:")
    logger.info(f"  Total nodes: {n_nodes}")
    logger.info(f"  Unique groups: {n_groups}")
    logger.info(f"  Avg nodes/group: {group_sizes.mean():.1f}")
    
    return PreprocessingResult(
        coordinates=normalized_coords.astype(np.float32),
        intensities=intensities.astype(np.float32),
        group_ids=group_ids,
        group_sizes=group_sizes.astype(np.int32),
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_,
        n_nodes=n_nodes,
        n_groups=n_groups
    )


def save_preprocessing_result(result: PreprocessingResult, output_path: Path) -> None:
    """
    Save preprocessing result to .npz file.
    
    Args:
        result: PreprocessingResult from prepare_training_data()
        output_path: Output path for .npz file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        coordinates=result.coordinates,
        intensities=result.intensities,
        group_ids=result.group_ids,
        group_sizes=result.group_sizes,
        scaler_min=result.scaler_min,
        scaler_max=result.scaler_max
    )
    
    logger.info(f"Training data saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================

def train_scar_model(
    training_data_path: Path,
    config: TrainingConfig,
    device: Optional[torch.device] = None
) -> dict:
    """
    Orchestrate model training workflow.
    
    Workflow:
    1. Load training data from .npz
    2. Create dataset and batches
    3. Initialize model
    4. Train with early stopping
    5. Return training history
    
    Args:
        training_data_path: Path to .npz training data
        config: Training configuration
        device: torch.device (auto-detected if None)
    
    Returns:
        Dictionary with training history and metadata
    
    Example:
        >>> config = TrainingConfig(max_epochs=5000)
        >>> history = train_scar_model(
        ...     training_data_path=Path("data.npz"),
        ...     config=config
        ... )
        >>> history['best_loss']
        0.040348
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # 1. Load training data
    logger.info(f"Loading training data: {training_data_path}")
    data = np.load(training_data_path)
    
    dataset = ScarReconstructionDataset(
        coordinates=data['coordinates'],
        intensities=data['intensities'],
        group_ids=data['group_ids'],
        group_sizes=data['group_sizes']
    )
    
    # 2. Create complete-group batches
    batches = create_complete_group_batches(dataset, config.batch_size)
    
    # 3. Initialize model
    logger.info("Initializing BayesianNN model")
    model = BayesianNN(dropout_rate=config.dropout_rate).to(device)
    
    total_params = model.count_parameters()
    logger.info(f"  Total parameters: {total_params:,}")
    
    # 4. Train
    history = train_model(
        model=model,
        dataset=dataset,
        batches=batches,
        config=config,
        device=device
    )
    
    # 5. Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'hyperparameters': {
            'dropout_rate': config.dropout_rate,
            'mc_samples': config.mc_samples,
            'batch_size': config.batch_size,
            'base_lr': config.base_lr,
            'max_lr': config.max_lr
        },
        'dataset_info': {
            'n_nodes': len(dataset),
            'n_groups': len(dataset.group_sizes),
            'scaler_min': data['scaler_min'],
            'scaler_max': data['scaler_max']
        }
    }
    
    return checkpoint


def save_trained_model(checkpoint: dict, output_path: Path) -> None:
    """
    Save trained model checkpoint.
    
    Args:
        checkpoint: Dictionary from train_scar_model()
        output_path: Output path for .pth file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    
    logger.info(f"Model saved to: {output_path}")
    logger.info("Training summary:")
    logger.info(f"  Final loss: {checkpoint['history']['losses'][-1]:.6f}")
    logger.info(f"  Best loss: {checkpoint['history']['best_loss']:.6f}")
    logger.info(f"  Converged in {checkpoint['history']['converged_epoch']} epochs")


# =============================================================================
# INFERENCE ORCHESTRATOR
# =============================================================================

def apply_scar_model(
    model_checkpoint_path: Path,
    mesh_path: Path,
    mc_samples: int = 10,
    batch_size: int = 50000,
    threshold: Optional[float] = None,
    device: Optional[torch.device] = None
) -> InferenceResult:
    """
    Orchestrate model inference on mesh.
    
    Workflow:
    1. Load mesh coordinates
    2. Create inference request
    3. Run reconstruction logic
    4. Return predictions
    
    Args:
        model_checkpoint_path: Path to trained .pth checkpoint
        mesh_path: Path to input mesh (VTK)
        mc_samples: Number of MC Dropout samples
        batch_size: Batch size for inference
        threshold: Optional threshold for binary classification
        device: torch.device (auto-detected if None)
    
    Returns:
        InferenceResult with predictions and metadata
    
    Example:
        >>> result = apply_scar_model(
        ...     model_checkpoint_path=Path("model.pth"),
        ...     mesh_path=Path("mesh.vtk"),
        ...     mc_samples=10,
        ...     threshold=0.5
        ... )
        >>> result.mean_scar_probability
        0.091
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # 1. Load mesh
    logger.info(f"Loading mesh: {mesh_path}")
    mesh_coords = load_mesh_points(mesh_path)
    logger.info(f"  Mesh nodes: {len(mesh_coords)}")
    
    # 2. Create request
    request = InferenceRequest(
        model_checkpoint_path=model_checkpoint_path,
        mesh_path=mesh_path,
        mc_samples=mc_samples,
        batch_size=batch_size,
        threshold=threshold
    )
    
    # 3. Run inference
    logic = ReconstructionLogic()
    result = logic.run_inference(request, mesh_coords, device)
    
    return result


def save_inference_result(
    result: InferenceResult,
    mesh_path: Path,
    output_path: Path
) -> None:
    """
    Save inference result as augmented mesh.
    
    Args:
        result: InferenceResult from apply_scar_model()
        mesh_path: Original mesh path (to load structure)
        output_path: Output path for augmented mesh
    """
    # Prepare scalar fields
    scalar_fields = {
        'scar_probability': result.mean_predictions,
        'scar_uncertainty': result.std_predictions
    }
    
    if result.binary_predictions is not None:
        scalar_fields['scar_binary'] = result.binary_predictions
    
    # Save
    save_mesh_with_scalars(mesh_path, output_path, scalar_fields)
    
    logger.info(f"Mesh saved to: {output_path}")
    logger.info("Scalar fields added:")
    for field_name in scalar_fields.keys():
        logger.info(f"  - {field_name}")
    logger.info(f"Mean scar probability: {result.mean_scar_probability:.3f}")
    logger.info(f"Mean uncertainty: {result.mean_uncertainty:.3f}")