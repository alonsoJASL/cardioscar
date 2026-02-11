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
    InferenceRequest,
    InferenceResult
)
from cardioscar.utilities.io import (
    load_mesh_points,
    load_grid_layer_data,
    save_mesh_with_scalars
)
from cardioscar.utilities.preprocessing import (
    create_group_mapping,
    remove_duplicate_nodes,
    compute_group_sizes,
    normalize_coordinates
)
from cardioscar.utilities.batching import (
    ScarReconstructionDataset,
    create_complete_group_batches
)
from cardioscar.training.config import TrainingConfig
from cardioscar.training.trainer import train_model
from cardioscar.engines import BayesianNN
from cardioscar.logic.reconstruction import ReconstructionLogic

logger = logging.getLogger(__name__)


# =============================================================================
# PREPROCESSING ORCHESTRATOR
# =============================================================================

def prepare_training_data(
    mesh_path: Path,
    grid_layer_paths: List[Path],
    slice_thickness_padding: float = 5.0
) -> PreprocessingResult:
    """
    Orchestrate training data preparation from mesh and grid layers.
    
    Workflow:
    1. Load mesh coordinates
    2. Process each grid layer (spatial mapping)
    3. Combine and deduplicate
    4. Normalize coordinates
    5. Compute group statistics
    
    Args:
        mesh_path: Path to 3D target mesh (VTK)
        grid_layer_paths: List of paths to 2D grid layers (VTK)
        slice_thickness_padding: Z-direction padding (mm) for slice thickness
    
    Returns:
        PreprocessingResult with training data arrays and metadata
    
    Example:
        >>> result = prepare_training_data(
        ...     mesh_path=Path("mesh.vtk"),
        ...     grid_layer_paths=[Path("slice1.vtk"), Path("slice2.vtk")],
        ...     slice_thickness_padding=5.0
        ... )
        >>> result.n_nodes
        25739
        >>> result.n_groups
        751
    """
    logger.info(f"Loading target mesh: {mesh_path}")
    mesh_coords = load_mesh_points(mesh_path)
    logger.info(f"  Loaded {len(mesh_coords)} mesh nodes")
    
    logger.info(f"Found {len(grid_layer_paths)} grid layers")
    
    # Process each grid layer
    all_mappings = []
    group_counter = 0
    
    for layer_idx, grid_path in enumerate(sorted(grid_layer_paths)):
        logger.info(f"Processing layer {layer_idx + 1}/{len(grid_layer_paths)}: {grid_path.name}")
        
        cell_bounds, scalar_values = load_grid_layer_data(grid_path)
        
        mapping = create_group_mapping(
            mesh_coords=mesh_coords,
            grid_cells_bounds=cell_bounds,
            grid_cells_scalars=scalar_values,
            z_padding=slice_thickness_padding,
            layer_id=layer_idx
        )
        
        if len(mapping) > 0:
            # Offset group IDs to make them globally unique
            mapping[:, 4] += group_counter
            group_counter = int(mapping[:, 4].max()) + 1
            all_mappings.append(mapping)
    
    # Combine all layers
    if not all_mappings:
        raise ValueError("No valid mappings found. Check grid layers and mesh overlap.")
    
    combined_data = np.vstack(all_mappings)
    
    # Remove duplicates (nodes appearing in multiple cells)
    unique_data = remove_duplicate_nodes(combined_data)
    
    # Extract components
    coordinates = unique_data[:, 0:3]
    intensities = unique_data[:, 3:4]
    group_ids = unique_data[:, 4].astype(np.int32)
    
    # Normalize coordinates
    normalized_coords, scaler = normalize_coordinates(coordinates)
    
    # Compute group sizes
    group_sizes = compute_group_sizes(group_ids)
    
    # Summary statistics
    n_nodes = len(coordinates)
    n_groups = len(group_sizes)
    
    logger.info("Dataset created:")
    logger.info(f"  Total nodes: {n_nodes}")
    logger.info(f"  Unique groups: {n_groups}")
    logger.info(f"  Avg nodes/group: {group_sizes.mean():.1f}")
    logger.info(f"  Min nodes/group: {group_sizes.min()}")
    logger.info(f"  Max nodes/group: {group_sizes.max()}")
    
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