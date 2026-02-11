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
@click.option('--grid-layers', type=click.Path(exists=True, path_type=Path), multiple=True, required=True,
              help='Paths to 2D grid layers (VTK). Can specify multiple times.')
@click.option('--output', type=click.Path(path_type=Path), required=True,
              help='Output path for training data (.npz)')
@click.option('--slice-thickness-padding', type=float, default=5.0,
              help='Z-direction padding (mm) for slice thickness')
def prepare(mesh_vtk, grid_layers, output, slice_thickness_padding):
    """Prepare training data from mesh and slice layers."""
    from cardioscar.logic.orchestrators import prepare_training_data, save_preprocessing_result

    result = prepare_training_data(
        mesh_path=mesh_vtk, 
        grid_layer_paths=list(grid_layers),  # Convert tuple to list
        slice_thickness_padding=slice_thickness_padding
    )
    
    save_preprocessing_result(result=result, output_path=output)


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
@click.option('--cpu', is_flag=True, default=False,
              help='Force CPU usage (default: auto-detect GPU)')
def train(training_data, output, batch_size, max_epochs, early_stopping_patience, mc_samples, cpu):
    """Train scar reconstruction model."""
    from cardioscar.training.config import TrainingConfig
    from cardioscar.logic.orchestrators import train_scar_model, save_trained_model
    
    config = TrainingConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        mc_samples=mc_samples,
    )

    device = get_device(force_cpu=cpu)
    
    checkpoint = train_scar_model(
        training_data_path=training_data, 
        config=config, 
        device=device
    )

    save_trained_model(checkpoint, output)


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