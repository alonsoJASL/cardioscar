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
    
    Example:
        >>> model = BayesianNN(dropout_rate=0.1)
        >>> coords = torch.randn(1000, 3)  # 1000 points
        >>> probs = model(coords)  # (1000, 1) probabilities
        >>> probs.shape
        torch.Size([1000, 1])
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
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