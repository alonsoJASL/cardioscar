#!/usr/bin/env python3
# TODO: Refactor to use functions in logic/orchestrators
"""
Apply Trained Scar Model to Mesh

This script loads a trained PyTorch model and applies it to predict scar
probabilities at every node in a mesh.

Orchestrator Responsibilities:
- Load trained model checkpoint
- Load target mesh (VTK)
- Coordinate normalization (using saved scaler)
- Model inference with MC Dropout
- Save augmented mesh with scar probability field

Usage:
    python apply_scar_model.py \
        --model models/patient_toy.pth \
        --mesh data/model.vtk \
        --output data/model_with_scar.vtk \
        --mc-samples 10

Output:
    VTK mesh with new scalar field 'scar_probability' at each node.
    Can be visualized in ParaView or used for cardiac simulations.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler

from pycemrg.core import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger("ApplyScarModel")


# =============================================================================
# MODEL ARCHITECTURE (must match training)
# =============================================================================

class BayesianNN(nn.Module):
    """
    Coordinate-based neural network with MC Dropout.
    
    NOTE: This must match the architecture in train_scar_model.py exactly.
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def enable_dropout(self) -> None:
        """Enable dropout during inference for MC sampling."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# =============================================================================
# INFERENCE LOGIC
# =============================================================================

def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple:
    """
    Load trained model and normalization parameters.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        device: torch device
        
    Returns:
        Tuple of (model, scaler_min, scaler_max)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    dropout_rate = checkpoint['hyperparameters']['dropout_rate']
    model = BayesianNN(dropout_rate=dropout_rate).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get normalization parameters
    scaler_min = checkpoint['dataset_info']['scaler_min']
    scaler_max = checkpoint['dataset_info']['scaler_max']
    
    logger.info("Model loaded successfully")
    logger.info(f"  Dropout rate: {dropout_rate}")
    logger.info(f"  Training nodes: {checkpoint['dataset_info']['n_nodes']}")
    logger.info(f"  Training groups: {checkpoint['dataset_info']['n_groups']}")
    
    return model, scaler_min, scaler_max


def normalize_coordinates(
    coords: np.ndarray,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray
) -> np.ndarray:
    """
    Normalize coordinates using saved scaler parameters.
    
    Args:
        coords: (N, 3) raw coordinates
        scaler_min: (3,) minimum values from training
        scaler_max: (3,) maximum values from training
        
    Returns:
        (N, 3) normalized coordinates in [0, 1]
    """
    # Reconstruct MinMaxScaler
    scaler = MinMaxScaler()
    scaler.data_min_ = scaler_min
    scaler.data_max_ = scaler_max
    scaler.data_range_ = scaler_max - scaler_min
    scaler.scale_ = 1.0 / scaler.data_range_  
    scaler.min_ = -scaler_min * scaler.scale_  
    scaler.n_samples_seen_ = 1  
    scaler.feature_names_in_ = None
    scaler.n_features_in_ = 3
    
    normalized = scaler.transform(coords)
    return normalized


def predict_with_uncertainty(
    model: BayesianNN,
    coords: torch.Tensor,
    mc_samples: int = 10,
    batch_size: int = 50000
) -> tuple:
    """
    Predict scar probabilities with uncertainty estimation.
    
    Args:
        model: Trained BayesianNN
        coords: (N, 3) normalized coordinates (torch tensor)
        mc_samples: Number of MC Dropout samples
        batch_size: Process in batches to avoid memory issues
        
    Returns:
        Tuple of (mean_predictions, std_predictions)
        - mean_predictions: (N,) mean scar probability
        - std_predictions: (N,) uncertainty (standard deviation)
    """
    logger.info(f"Running inference with {mc_samples} MC samples...")
    
    model.enable_dropout()  # Enable dropout for uncertainty
    
    n_nodes = len(coords)
    all_predictions = []
    
    with torch.no_grad():
        # MC Dropout sampling
        for mc_iter in range(mc_samples):
            batch_predictions = []
            
            # Process in batches
            for batch_start in range(0, n_nodes, batch_size):
                batch_end = min(batch_start + batch_size, n_nodes)
                batch_coords = coords[batch_start:batch_end]
                
                batch_pred = model(batch_coords).cpu().numpy().squeeze()
                batch_predictions.append(batch_pred)
            
            # Concatenate batch results
            mc_pred = np.concatenate(batch_predictions)
            all_predictions.append(mc_pred)
            
            if (mc_iter + 1) % 5 == 0 or (mc_iter + 1) == mc_samples:
                logger.info(f"  Completed {mc_iter + 1}/{mc_samples} samples")
    
    # Stack predictions: (mc_samples, n_nodes)
    all_predictions = np.stack(all_predictions, axis=0)
    
    # Compute statistics
    mean_pred = all_predictions.mean(axis=0)
    std_pred = all_predictions.std(axis=0)
    
    logger.info(f"Inference complete:")
    logger.info(f"  Mean scar probability: {mean_pred.mean():.3f} ± {mean_pred.std():.3f}")
    logger.info(f"  Mean uncertainty (std): {std_pred.mean():.3f}")
    logger.info(f"  Max uncertainty: {std_pred.max():.3f}")
    
    return mean_pred, std_pred


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Main inference orchestrator."""
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load model
    model, scaler_min, scaler_max = load_model_checkpoint(args.model, device)
    
    # 2. Load mesh
    logger.info(f"Loading mesh: {args.mesh}")
    mesh = pv.read(args.mesh)
    coords = np.array(mesh.points)  # (N, 3)
    logger.info(f"  Mesh nodes: {len(coords)}")
    
    # 3. Normalize coordinates
    logger.info("Normalizing coordinates...")
    normalized_coords = normalize_coordinates(coords, scaler_min, scaler_max)
    coords_tensor = torch.from_numpy(normalized_coords).float().to(device)
    
    # 4. Predict scar probabilities
    mean_scar, std_scar = predict_with_uncertainty(
        model=model,
        coords=coords_tensor,
        mc_samples=args.mc_samples,
        batch_size=args.batch_size
    )
    
    # 5. Add scalar fields to mesh
    logger.info("Adding scalar fields to mesh...")
    mesh['scar_probability'] = mean_scar
    mesh['scar_uncertainty'] = std_scar
    
    # Optional: Add binary threshold field
    if args.threshold is not None:
        binary_scar = (mean_scar >= args.threshold).astype(np.float32)
        mesh['scar_binary'] = binary_scar
        n_scar_nodes = binary_scar.sum()
        pct_scar = 100 * n_scar_nodes / len(coords)
        logger.info(f"Threshold {args.threshold}: {n_scar_nodes} nodes ({pct_scar:.1f}%) marked as scar")
    
    # 6. Save augmented mesh
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving mesh: {args.output}")
    mesh.save(args.output)
    
    # 7. Summary
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output mesh: {args.output}")
    logger.info(f"Scalar fields added:")
    logger.info(f"  - scar_probability (mean prediction)")
    logger.info(f"  - scar_uncertainty (standard deviation)")
    if args.threshold is not None:
        logger.info(f"  - scar_binary (threshold={args.threshold})")
    logger.info("")
    logger.info("Visualize in ParaView:")
    logger.info(f"  paraview {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply trained scar model to mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pth)"
    )
    
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to input mesh (VTK format)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_with_scar.vtk"),
        help="Output path for augmented mesh"
    )
    
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="Number of MC Dropout samples for uncertainty estimation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for inference (adjust based on available memory)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold for binary scar classification (e.g., 0.5)"
    )
    
    args = parser.parse_args()
    main(args)