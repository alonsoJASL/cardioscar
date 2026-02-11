#!/usr/bin/env python3
"""
Train Patient-Specific Scar Reconstruction Model

This script trains a coordinate-based neural network to reconstruct 3D scar
probability fields from sparse 2D MRI constraints.

Key Optimizations:
1. Mini-batch training with complete group preservation
2. Early stopping to prevent overfitting
3. Cyclical learning rate for faster convergence
4. Reduced network size (4 layers × 128 neurons)

Orchestrator Responsibilities:
- Load training data
- Configure training hyperparameters
- Execute training loop
- Save model checkpoint

Mathematical Logic (delegated to modules):
- Network architecture (bayesian_nn.py)
- Group-based loss computation (loss.py)
- MC Dropout uncertainty estimation

Usage:
    python scripts/train_scar_model.py \\
        --training-data data/training_data.npz \\
        --output models/patient_001.pth \\
        --batch-size 10000 \\
        --max-epochs 10000 \\
        --early-stopping-patience 500

Expected Training Time: ~45 minutes on NVIDIA T400 4GB
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pycemrg.core import setup_logging

# Setup logging
# Setup logging
setup_logging()
logger = logging.getLogger("TrainScarModel")


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class BayesianNN(nn.Module):
    """
    Coordinate-based neural network with MC Dropout for uncertainty estimation.
    
    Architecture: 4 layers × 128 neurons (reduced from original 6 × 256)
    Activation: ReLU
    Dropout: 0.1 after layers 2 and 4
    Output: Sigmoid for [0,1] probability
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
        """
        Forward pass.
        
        Args:
            x: (batch_size, 3) coordinates
            
        Returns:
            (batch_size, 1) scar probabilities
        """
        return self.network(x)
    
    def enable_dropout(self) -> None:
        """Enable dropout during inference for MC sampling."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# =============================================================================
# DATASET AND BATCHING
# =============================================================================

class ScarReconstructionDataset(Dataset):
    """
    Dataset with complete-group batching support.
    
    This dataset pre-sorts nodes by group ID to enable batching complete
    groups together, ensuring the constraint is preserved.
    """
    
    def __init__(
        self,
        coordinates: np.ndarray,
        intensities: np.ndarray,
        group_ids: np.ndarray,
        group_sizes: np.ndarray
    ):
        """
        Args:
            coordinates: (N, 3) normalized mesh coordinates
            intensities: (N, 1) target scar values
            group_ids: (N,) group assignment per node
            group_sizes: (M,) number of nodes per unique group
        """
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
        
        logger.info(f"Dataset initialized:")
        logger.info(f"  Total nodes: {len(self.coordinates)}")
        logger.info(f"  Unique groups: {len(self.group_sizes)}")
    
    def __len__(self) -> int:
        return len(self.coordinates)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.coordinates[idx],
            self.intensities[idx],
            self.group_ids[idx]
        )


def create_complete_group_batches(
    dataset: ScarReconstructionDataset,
    target_batch_size: int = 10000
) -> list:
    """
    Create batches containing only complete groups.
    
    Args:
        dataset: ScarReconstructionDataset instance
        target_batch_size: Approximate target batch size
        
    Returns:
        List of (start_idx, end_idx) tuples defining each batch
    """
    batches = []
    current_start = 0
    current_size = 0
    
    for group_idx in range(len(dataset.group_sizes)):
        group_size = dataset.group_sizes[group_idx].item()
        
        # If adding this group exceeds target, finalize current batch
        if current_size + group_size > target_batch_size and current_size > 0:
            current_end = dataset.group_cumsum[group_idx].item()
            batches.append((current_start, current_end))
            current_start = current_end
            current_size = 0
        
        current_size += group_size
    
    # Add final batch
    if current_size > 0:
        batches.append((current_start, len(dataset)))
    
    logger.info(f"Created {len(batches)} complete-group batches")
    logger.info(f"  Batch sizes: {min([b[1]-b[0] for b in batches])} to {max([b[1]-b[0] for b in batches])} nodes")
    
    return batches


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def compute_group_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    group_sizes: torch.Tensor,
    group_cumsum: torch.Tensor
) -> torch.Tensor:
    """
    Compute group-based reconstruction loss.
    
    Loss = Σ (target_pixel_value - mean(group_predictions))²
    
    Args:
        predictions: (batch_size, 1) network predictions
        targets: (batch_size, 1) target values
        group_ids: (batch_size,) group assignment
        group_sizes: (n_groups,) nodes per group
        group_cumsum: (n_groups+1,) cumulative sum for indexing
        
    Returns:
        Scalar loss tensor
    """
    # Find unique groups in this batch
    unique_groups = torch.unique(group_ids)
    
    losses = []
    for group_id in unique_groups:
        # Get indices for this group within the batch
        mask = (group_ids == group_id)
        group_preds = predictions[mask]
        group_target = targets[mask][0]  # All nodes in group have same target
        
        # Compute group mean and loss
        group_mean = group_preds.mean()
        group_loss = (group_target - group_mean) ** 2
        losses.append(group_loss)
    
    return torch.stack(losses).mean()


# =============================================================================
# TRAINING LOOP
# =============================================================================

class CyclicalLR:
    """Cyclical learning rate scheduler."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float = 1e-3,
        max_lr: float = 1e-2,
        step_size: int = 2000
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.step_count = 0
    
    def step(self) -> None:
        """Update learning rate."""
        cycle = np.floor(1 + self.step_count / (2 * self.step_size))
        x = np.abs(self.step_count / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1
    
    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def train_model(
    model: BayesianNN,
    dataset: ScarReconstructionDataset,
    batches: list,
    device: torch.device,
    max_epochs: int = 10000,
    mc_samples: int = 3,
    early_stopping_patience: int = 500,
    base_lr: float = 1e-3,
    max_lr: float = 1e-2
) -> Dict:
    """
    Training loop with early stopping.
    
    Args:
        model: BayesianNN instance
        dataset: Training dataset
        batches: List of (start, end) batch indices
        device: torch.device
        max_epochs: Maximum training epochs
        mc_samples: Number of MC dropout samples
        early_stopping_patience: Epochs without improvement before stopping
        base_lr: Base learning rate for cyclical schedule
        max_lr: Maximum learning rate for cyclical schedule
        
    Returns:
        Dictionary with training history
    """
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = CyclicalLR(optimizer, base_lr, max_lr, step_size=2000)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'losses': [], 'lrs': []}
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        
        # Iterate over complete-group batches
        for batch_start, batch_end in batches:
            coords = dataset.coordinates[batch_start:batch_end].to(device)
            targets = dataset.intensities[batch_start:batch_end].to(device)
            groups = dataset.group_ids[batch_start:batch_end].to(device)
            
            optimizer.zero_grad()
            
            # MC Dropout: average predictions over multiple forward passes
            mc_predictions = []
            for _ in range(mc_samples):
                preds = model(coords)
                mc_predictions.append(preds)
            
            # Use mean prediction for loss
            predictions = torch.stack(mc_predictions).mean(dim=0)
            
            # Compute group-based loss
            loss = compute_group_reconstruction_loss(
                predictions,
                targets,
                groups,
                dataset.group_sizes,
                dataset.group_cumsum
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        current_lr = scheduler.get_last_lr()
        
        history['losses'].append(avg_loss)
        history['lrs'].append(current_lr)
        
        # Logging
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1:5d}/{max_epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed/60:.1f}m"
            )
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            logger.info(f"Best loss: {best_loss:.6f}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    
    return history


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Main training orchestrator."""
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load training data
    logger.info(f"Loading training data: {args.training_data}")
    data = np.load(args.training_data)
    
    dataset = ScarReconstructionDataset(
        coordinates=data['coordinates'],
        intensities=data['intensities'],
        group_ids=data['group_ids'],
        group_sizes=data['group_sizes']
    )
    
    # 2. Create complete-group batches
    batches = create_complete_group_batches(dataset, args.batch_size)
    
    # 3. Initialize model
    logger.info("Initializing BayesianNN model")
    model = BayesianNN(dropout_rate=0.1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    
    # 4. Train
    logger.info("Starting training...")
    history = train_model(
        model=model,
        dataset=dataset,
        batches=batches,
        device=device,
        max_epochs=args.max_epochs,
        mc_samples=args.mc_samples,
        early_stopping_patience=args.early_stopping_patience,
        base_lr=args.base_lr,
        max_lr=args.max_lr
    )
    
    # 5. Save checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'hyperparameters': {
            'dropout_rate': 0.1,
            'mc_samples': args.mc_samples,
            'batch_size': args.batch_size,
            'base_lr': args.base_lr,
            'max_lr': args.max_lr
        },
        'dataset_info': {
            'n_nodes': len(dataset),
            'n_groups': len(dataset.group_sizes),
            'scaler_min': data['scaler_min'],
            'scaler_max': data['scaler_max']
        }
    }
    
    torch.save(checkpoint, args.output)
    logger.info(f"Model saved to: {args.output}")
    
    # 6. Final summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final loss: {history['losses'][-1]:.6f}")
    logger.info(f"Best loss: {min(history['losses']):.6f}")
    logger.info(f"Converged in {len(history['losses'])} epochs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train patient-specific scar reconstruction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--training-data",
        type=Path,
        required=True,
        help="Path to training data (.npz from prepare_training_data.py)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.pth"),
        help="Output path for trained model"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Target batch size (actual size varies to preserve complete groups)"
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10000,
        help="Maximum training epochs"
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=500,
        help="Epochs without improvement before early stopping"
    )
    
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=3,
        help="Number of MC Dropout samples per training step"
    )
    
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-3,
        help="Base learning rate for cyclical schedule"
    )
    
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-2,
        help="Maximum learning rate for cyclical schedule"
    )
    
    args = parser.parse_args()
    main(args)
