# src/cardioscar/training/trainer.py

"""
Training Logic for Scar Reconstruction

Stateless training loop with early stopping and cyclical learning rate.
Orchestrators provide data and configuration, this module handles the training.
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cardioscar.engines import BayesianNN, compute_group_reconstruction_loss
from cardioscar.utilities.batching import ScarReconstructionDataset
from cardioscar.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class CyclicalLR:
    """
    Cyclical learning rate scheduler.
    
    Implements triangular cyclical learning rate as described in:
    Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
    """
    
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
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def train_model(
    model: BayesianNN,
    dataset: ScarReconstructionDataset,
    batches: List[Tuple[int, int]],
    config: TrainingConfig,
    device: torch.device
) -> Dict:
    """
    Train scar reconstruction model with early stopping.
    
    Args:
        model: BayesianNN instance
        dataset: Training dataset
        batches: List of (start, end) batch boundaries
        config: Training configuration
        device: torch.device (cpu or cuda)
    
    Returns:
        Dictionary with training history:
        - 'losses': List of epoch losses
        - 'lrs': List of learning rates
        - 'best_loss': Best loss achieved
        - 'best_epoch': Epoch where best loss occurred
        - 'converged_epoch': Epoch where training stopped
        - 'training_time': Total training time (seconds)
    
    Example:
        >>> model = BayesianNN()
        >>> dataset = ScarReconstructionDataset(coords, intensities, groups, sizes)
        >>> batches = create_complete_group_batches(dataset)
        >>> config = TrainingConfig()
        >>> history = train_model(model, dataset, batches, config, torch.device('cpu'))
        >>> history['best_loss']
        0.040348
    """
    optimizer = optim.Adam(model.parameters(), lr=config.base_lr)
    scheduler = CyclicalLR(
        optimizer,
        base_lr=config.base_lr,
        max_lr=config.max_lr,
        step_size=config.lr_step_size
    )
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'losses': [], 'lrs': []}
    
    start_time = time.time()
    
    logger.info("Starting training...")
    logger.info(f"  Max epochs: {config.max_epochs}")
    logger.info(f"  Early stopping patience: {config.early_stopping_patience}")
    logger.info(f"  Batches per epoch: {len(batches)}")
    
    for epoch in range(config.max_epochs):
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
            for _ in range(config.mc_samples):
                preds = model(coords)
                mc_predictions.append(preds)
            
            # Use mean prediction for loss
            predictions = torch.stack(mc_predictions).mean(dim=0)
            
            # Compute group-based loss
            loss = compute_group_reconstruction_loss(
                predictions,
                targets,
                groups
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
                f"Epoch {epoch+1:5d}/{config.max_epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed/60:.1f}m"
            )
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            logger.info(f"Best loss: {best_loss:.6f} at epoch {best_epoch+1}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    
    return {
        'losses': history['losses'],
        'lrs': history['lrs'],
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'converged_epoch': len(history['losses']),
        'training_time': total_time
    }