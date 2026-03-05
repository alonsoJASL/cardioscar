"""
CardioScar CLI

Command-line interface for scar reconstruction workflows.
Uses Click for argument parsing and orchestrates library components.
"""

import logging
from pathlib import Path

import click
import torch

# Setup logging
from pycemrg.core.logs import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from cardioscar.training.config import DEFAULT_HIDDEN_SIZE, DEFAULT_N_HIDDEN_LAYERS

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Determine the device to use for computation.
    
    Args:
        force_cpu: If True, force CPU usage even if CUDA available
    
    Returns:
        torch.device (cuda or cpu)
    """
    if force_cpu:
        logger.info("Forcing CPU usage.")
        return torch.device('cpu')
    elif torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        return torch.device('cuda')
    else:
        logger.info("CUDA not available. Using CPU.")
        return torch.device('cpu')


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    CardioScar: Deep learning-based 3D myocardial scar reconstruction.
    
    Reconstructs continuous 3D scar probability fields from sparse 2D LGE-CMR slices.
    
    Authors: Ahmet Sen, Martin J. Bishop, Jose Alonso Solis-Lemus
    """
    pass


@cli.command()
@click.option('--mesh-vtk', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to 3D target mesh (VTK)')
# VTK grid input
@click.option('--grid-layers', type=click.Path(exists=True, path_type=Path), multiple=True,
              help='Paths to 2D grid layers (VTK). Can specify multiple times.')
@click.option('--vtk-scalar-field', type=str, default='ScalarValue',
              help='Scalar field name in VTK grids')
# Image input (NEW)
@click.option('--image', type=click.Path(exists=True, path_type=Path),
              help='Path to medical image (NIfTI, NRRD)')
@click.option('--slice-axis', type=click.Choice(['x', 'y', 'z']), default='z',
              help='Axis to slice along (default: z)')
@click.option('--slice-indices', type=str,
              help='Comma-separated slice indices (e.g., "2,5,8,11"). If not provided, uses ALL slices.')
# Common options
@click.option('--output', type=click.Path(path_type=Path), required=True,
              help='Output path for training data (.npz)')
@click.option('--slice-thickness-padding', type=float, default=5.0,
              help='Z-direction padding (mm) for slice thickness')
@click.option('--exclude-zero-intensity', is_flag=True, default=False,
              help=(
                  'Exclude nodes with intensity 0.0 from training data. '
                  'Use for synthetic rasterized images where 0.0 means '
                  'outside-mesh padding. Do not use for real LGE images.'
              ))
def prepare(mesh_vtk, grid_layers, vtk_scalar_field, image, slice_axis, slice_indices, output, slice_thickness_padding, exclude_zero_intensity):
    """Prepare training data from mesh and slices."""
    from cardioscar.logic.orchestrators import prepare_training_data, save_preprocessing_result
    from cardioscar.logic.contracts import PreprocessingRequest
    
    # Parse slice indices if provided
    parsed_indices = None
    if slice_indices:
        parsed_indices = [int(i.strip()) for i in slice_indices.split(',')]
    
    # Build request
    request = PreprocessingRequest(
        mesh_path=mesh_vtk,
        # VTK input
        vtk_grid_paths=list(grid_layers) if grid_layers else None,
        vtk_scalar_field=vtk_scalar_field,
        # Image input
        image_path=image,
        slice_axis=slice_axis,
        slice_indices=parsed_indices,
        # Common
        slice_thickness_padding=slice_thickness_padding,
        exclude_zero_intensity=exclude_zero_intensity,
    )
    
    result = prepare_training_data(request)
    save_preprocessing_result(result, output)


@cli.command()
@click.option('--training-data', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to training data (.npz)')
@click.option('--output', type=click.Path(path_type=Path), required=True,
              help='Output path for trained model (.pth)')
@click.option('--batch-size', type=int, default=10000,
              help='Target batch size')
@click.option('--max-epochs', type=int, default=10000,
              help='Maximum training epochs')
@click.option('--early-stopping-patience', type=int, default=500,
              help='Epochs without improvement before stopping')
@click.option('--mc-samples', type=int, default=3,
              help='MC Dropout samples during training')
@click.option('--hidden-size', type=int, default=DEFAULT_HIDDEN_SIZE,
              help=f'Neurons per hidden layer (default: {DEFAULT_HIDDEN_SIZE})')
@click.option('--hidden-layers', type=int, default=DEFAULT_N_HIDDEN_LAYERS,
              help=f'Number of hidden layers (default: {DEFAULT_N_HIDDEN_LAYERS})')
@click.option('--cpu', is_flag=True, default=False,
              help='Force CPU usage (default: auto-detect GPU)')
def train(training_data, output, batch_size, max_epochs, early_stopping_patience,
          mc_samples, hidden_size, hidden_layers, cpu):

    """Train scar reconstruction model."""
    from cardioscar.training.config import TrainingConfig
    from cardioscar.logic.orchestrators import train_scar_model, save_trained_model
    
    config = TrainingConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        mc_samples=mc_samples,
        hidden_size=hidden_size,
        n_hidden_layers=hidden_layers,
    )

    device = get_device(force_cpu=cpu)
    
    checkpoint = train_scar_model(
        training_data_path=training_data, 
        config=config, 
        device=device
    )

    save_trained_model(checkpoint, output)

@cli.command()
@click.option('--checkpoint', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to pretrained foundation model (.pth)')
@click.option('--training-data', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to fine-tuning training data (.npz)')
@click.option('--output', type=click.Path(path_type=Path), required=True,
              help='Output path for fine-tuned model (.pth)')
@click.option('--freeze-stages', type=int, default=0,
              help=(
                  'Number of network stages to freeze (0-4). '
                  'Stage 1: Linear(3→128)+ReLU. '
                  'Stage 2: adds Linear(128→128)+Dropout+ReLU. '
                  'Stage 3: adds Linear(128→128)+ReLU. '
                  'Stage 4: adds Linear(128→128)+Dropout+ReLU. '
                  'Output stage is always trainable. Default: 0 (full fine-tune).'
              ))
@click.option('--batch-size', type=int, default=10000,
              help='Target batch size')
@click.option('--max-epochs', type=int, default=1000,
              help='Maximum fine-tuning epochs')
@click.option('--early-stopping-patience', type=int, default=200,
              help='Epochs without improvement before stopping')
@click.option('--mc-samples', type=int, default=3,
              help='MC Dropout samples during training')
@click.option('--base-lr', type=float, default=1e-4,
              help='Base learning rate (default: 1e-4, lower than training)')
@click.option('--max-lr', type=float, default=1e-3,
              help='Maximum learning rate (default: 1e-3, lower than training)')
@click.option('--cpu', is_flag=True, default=False,
              help='Force CPU usage (default: auto-detect GPU)')
def fine_tune(checkpoint, training_data, output, freeze_stages, batch_size,
              max_epochs, early_stopping_patience, mc_samples, base_lr, max_lr,
              cpu):
    """Fine-tune a pretrained scar reconstruction model on new data."""
    from cardioscar.training.config import TrainingConfig, FineTuneConfig
    from cardioscar.logic.orchestrators import fine_tune_scar_model, save_trained_model

    config = FineTuneConfig(
        training=TrainingConfig(
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            mc_samples=mc_samples,
            base_lr=base_lr,
            max_lr=max_lr,
            # default hidden_size and n_hidden_layers will be 
            # overridden by checkpoint hyperparameters
        ),
        freeze_stages=freeze_stages,
    )


    device = get_device(force_cpu=cpu)

    checkpoint_data = fine_tune_scar_model(
        training_data_path=training_data,
        checkpoint_path=checkpoint,
        config=config,
        device=device,
    )

    save_trained_model(checkpoint_data, output)


@cli.command()
@click.option('--model', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to trained model (.pth)')
@click.option('--mesh', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to input mesh (VTK)')
@click.option('--output', type=click.Path(path_type=Path), required=True,
              help='Output path for augmented mesh (.vtk)')
@click.option('--mc-samples', type=int, default=10,
              help='MC Dropout samples for uncertainty estimation')
@click.option('--batch-size', type=int, default=50000,
              help='Batch size for inference')
@click.option('--threshold', type=float, default=None,
              help='Optional threshold for binary scar classification')
@click.option('--cpu', is_flag=True, default=False,
              help='Force CPU usage (default: auto-detect GPU)')
def apply(model, mesh, output, mc_samples, batch_size, threshold, cpu):
    """Apply trained model to mesh."""
    from cardioscar.logic.orchestrators import apply_scar_model, save_inference_result
    
    device = get_device(force_cpu=cpu)
    
    result = apply_scar_model(
        model_checkpoint_path=model,
        mesh_path=mesh,
        mc_samples=mc_samples,
        batch_size=batch_size,
        threshold=threshold,
        device=device
    )
    
    save_inference_result(result, mesh, output)


if __name__ == '__main__':
    cli()