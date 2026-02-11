# src/cardioscar/training/config.py

"""
Training Configuration

Hyperparameter configuration for scar reconstruction training.
Uses dataclass for type safety and clear documentation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration for training BayesianNN model.
    
    Attributes:
        batch_size: Target batch size (actual varies for complete groups)
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs without improvement before stopping
        mc_samples: Number of MC Dropout samples during training
        base_lr: Base learning rate for cyclical schedule
        max_lr: Maximum learning rate for cyclical schedule
        lr_step_size: Step size for cyclical LR (iterations per half-cycle)
        dropout_rate: Dropout probability
        seed: Random seed for reproducibility
    
    Example:
        >>> config = TrainingConfig(
        ...     batch_size=10000,
        ...     max_epochs=10000,
        ...     early_stopping_patience=500
        ... )
    """
    
    # Batching
    batch_size: int = 10000
    
    # Training duration
    max_epochs: int = 10000
    early_stopping_patience: int = 500
    
    # Monte Carlo Dropout
    mc_samples: int = 3
    
    # Learning rate schedule
    base_lr: float = 1e-3
    max_lr: float = 1e-2
    lr_step_size: int = 2000
    
    # Model architecture
    dropout_rate: float = 0.1
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive")
        
        if not 0 < self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in (0, 1), got {self.dropout_rate}")
        
        if self.base_lr >= self.max_lr:
            raise ValueError(f"base_lr must be < max_lr")