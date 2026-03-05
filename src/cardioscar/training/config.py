# src/cardioscar/training/config.py

"""
Training Configuration

Hyperparameter configuration for scar reconstruction training.
Uses dataclass for type safety and clear documentation.
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Architecture defaults — single source of truth.
# Referenced by TrainingConfig, FineTuneConfig, and any checkpoint loading
# site that needs a fallback for legacy checkpoints lacking these keys.
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN_SIZE: int = 128
DEFAULT_N_HIDDEN_LAYERS: int = 4


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
        hidden_size: Neurons per hidden layer
        n_hidden_layers: Number of hidden layers
        seed: Random seed for reproducibility

    Example:
        >>> config = TrainingConfig(
        ...     batch_size=10000,
        ...     max_epochs=10000,
        ...     early_stopping_patience=500,
        ...     hidden_size=256,
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
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")

        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

        if not 0 < self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in (0, 1), got {self.dropout_rate}")

        if self.base_lr >= self.max_lr:
            raise ValueError("base_lr must be < max_lr")

        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")

        if self.n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {self.n_hidden_layers}")


# ---------------------------------------------------------------------------
# Stage-to-children index mapping for BayesianNN.network (nn.Sequential).
# Valid for DEFAULT_N_HIDDEN_LAYERS=4 and DEFAULT_HIDDEN_SIZE=128.
#
# Stage 1: Linear(3→H)   + ReLU            children[0:2]
# Stage 2: Linear(H→H)   + Dropout + ReLU  children[2:5]
# Stage 3: Linear(H→H)   + ReLU            children[5:7]
# Stage 4: Linear(H→H)   + Dropout + ReLU  children[7:10]
# Stage 5: Linear(H→1)   + Sigmoid         children[10:12] — output, never frozen
#
# NOTE: these ranges are only valid for n_hidden_layers=4. If n_hidden_layers
# differs, frozen_child_range() will raise rather than silently freeze
# the wrong layers.
# ---------------------------------------------------------------------------

_STAGE_TO_CHILD_RANGE = {
    1: (0, 2),
    2: (0, 5),
    3: (0, 7),
    4: (0, 10),
}
MAX_FREEZE_STAGES = 4


@dataclass
class FineTuneConfig:
    """
    Configuration for fine-tuning a pretrained BayesianNN checkpoint.

    Composes a TrainingConfig with fine-tune-appropriate defaults and
    adds layer-freezing control via ``freeze_stages``.

    The BayesianNN backbone is divided into five stages (valid for
    n_hidden_layers=4):
        Stage 1: Linear(3→H)   + ReLU
        Stage 2: Linear(H→H)   + Dropout + ReLU
        Stage 3: Linear(H→H)   + ReLU
        Stage 4: Linear(H→H)   + Dropout + ReLU
        Stage 5: Linear(H→1)   + Sigmoid  (output — never frozen)

    Setting ``freeze_stages=2`` freezes Stages 1 and 2, leaving Stages 3-5
    trainable. The output stage (5) is always trainable regardless of this
    value.

    Note:
        freeze_stages is only supported for n_hidden_layers=4. Attempting
        to freeze stages on a model with a different depth will raise
        ValueError.

    Attributes:
        training: Inner TrainingConfig with fine-tune-appropriate defaults.
        freeze_stages: Number of stages (1–4) to freeze. 0 = full fine-tune.

    Example:
        >>> config = FineTuneConfig(freeze_stages=2)
        >>> config.training.max_epochs
        1000
    """

    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            max_epochs=1000,
            early_stopping_patience=200,
            base_lr=1e-4,
            max_lr=1e-3,
            lr_step_size=500,
            batch_size=10000,
            mc_samples=3,
            dropout_rate=0.1,
        )
    )
    freeze_stages: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.freeze_stages <= MAX_FREEZE_STAGES:
            raise ValueError(
                f"freeze_stages must be between 0 and {MAX_FREEZE_STAGES}, "
                f"got {self.freeze_stages}"
            )

    def frozen_child_range(self) -> Optional[tuple]:
        """
        Return the (start, end) child index range of model.network to freeze.

        Returns None if freeze_stages is 0 (nothing to freeze).

        Raises:
            ValueError: If freeze_stages > 0 and n_hidden_layers != 4,
                since the stage-to-child mapping is only valid for depth 4.
        """
        if self.freeze_stages == 0:
            return None
        if self.training.n_hidden_layers != DEFAULT_N_HIDDEN_LAYERS:
            raise ValueError(
                f"freeze_stages is only supported for n_hidden_layers="
                f"{DEFAULT_N_HIDDEN_LAYERS}, got {self.training.n_hidden_layers}. "
                f"Set freeze_stages=0 for non-default depth."
            )
        return _STAGE_TO_CHILD_RANGE[self.freeze_stages]