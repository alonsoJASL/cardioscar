# src/cardioscar/training/__init__.py
"""
Training Public API

Exposes training configuration and trainer logic.
"""

from cardioscar.training.config import TrainingConfig
from cardioscar.training.trainer import train_model, CyclicalLR

__all__ = [
    # Configuration/Data contract
    "TrainingConfig",
    # Training logic
    "train_model",
    "CyclicalLR",
]