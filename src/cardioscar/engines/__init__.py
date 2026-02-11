# src/cardioscar/engines/__init__.py

"""
Neural Network Engines

This module contains the computational engines (neural networks, loss functions)
for scar reconstruction. These are PyTorch nn.Module subclasses and related
functions that perform the core mathematical operations.

Engines are distinct from:
- Data contracts (logic/contracts.py) - dataclass structures
- Utilities (utilities/) - pure functions for data transformation
- Logic (logic/) - stateless orchestration

Available engines:
- BayesianNN: Coordinate-based MLP with MC Dropout
- compute_group_reconstruction_loss: Group-constrained loss function
"""

from cardioscar.engines.bayesian_nn import BayesianNN
from cardioscar.engines.loss import compute_group_reconstruction_loss

__all__ = [
    "BayesianNN",
    "compute_group_reconstruction_loss",
]