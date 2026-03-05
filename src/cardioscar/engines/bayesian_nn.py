# src/cardioscar/engines/bayesian_nn.py

"""
Bayesian Neural Network for Scar Reconstruction

Coordinate-based MLP that maps (X, Y, Z) → scar probability [0, 1].
Uses MC Dropout for uncertainty quantification.

Architecture:
    Input: (batch, 3) - normalized coordinates
    Hidden: 4 layers × 128 neurons with ReLU
    Dropout: 0.1 after layers 2 and 4
    Output: (batch, 1) - scar probability via Sigmoid

Reference:
    Sen et al. (2025) "Weakly supervised learning for scar reconstruction"
    https://doi.org/10.1016/j.compbiomed.2025.111219
"""

import torch
import torch.nn as nn
from typing import Optional


class BayesianNN(nn.Module):
    """
    Coordinate-based neural network with Monte Carlo Dropout.

    This network learns a continuous mapping from 3D spatial coordinates
    to scar probability values. Dropout layers enable uncertainty estimation
    via multiple stochastic forward passes.

    Args:
        dropout_rate: Probability of dropping neurons (default: 0.1)
        hidden_size: Number of neurons per hidden layer (default: 128)
        n_hidden_layers: Number of hidden layers (default: 4)

    Example:
        >>> model = BayesianNN(dropout_rate=0.1, hidden_size=256, n_hidden_layers=4)
        >>> coords = torch.randn(1000, 3)
        >>> probs = model(coords)
        >>> probs.shape
        torch.Size([1000, 1])
    """

    def __init__(
        self,
        dropout_rate: float = 0.1,
        hidden_size: int = 128,
        n_hidden_layers: int = 4,
    ):
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if not 0 < dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in (0, 1), got {dropout_rate}")

        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        layers = []

        # Input layer
        layers.append(nn.Linear(3, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers: dropout after every second layer
        for i in range(1, n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if i % 2 == 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: (batch_size, 3) normalized coordinates in [0, 1]
        
        Returns:
            (batch_size, 1) scar probabilities in [0, 1]
        """
        return self.network(x)
    
    def enable_dropout(self) -> None:
        """
        Enable dropout during inference for Monte Carlo sampling.
        
        Call this before running multiple forward passes to estimate
        prediction uncertainty via MC Dropout.
        
        Example:
            >>> model.eval()
            >>> model.enable_dropout()
            >>> samples = [model(coords) for _ in range(10)]
            >>> mean = torch.stack(samples).mean(dim=0)
            >>> std = torch.stack(samples).std(dim=0)
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)