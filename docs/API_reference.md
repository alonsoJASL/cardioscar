# CardioScar API Reference

**Version:** 0.2.0  
**Authors:** Ahmet Sen, Martin J. Bishop, Jose Alonso Solis-Lemus

Deep learning-based 3D myocardial scar reconstruction from sparse 2D LGE-CMR slices.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command Line Interface](#command-line-interface)
4. [Python API](#python-api)
   - [Orchestrators](#orchestrators)
   - [Data Contracts](#data-contracts)
   - [Neural Network Models](#neural-network-models)
   - [Training Configuration](#training-configuration)
   - [Utilities](#utilities)
5. [Examples](#examples)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Common Pitfalls](#common-pitfalls)

---

## Installation

### From PyPI (when published)
```bash
pip install cardioscar
```

### From Source (Development)
```bash
git clone https://github.com/alonsoJASL/cardioscar.git
cd cardioscar
pip install -e .
```

### Dependencies
- Python >= 3.10
- PyTorch >= 2.0
- PyVista
- SimpleITK
- scikit-learn
- pycemrg-suite

### GPU Support (Recommended)
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### Complete Workflow (CLI)

```bash
# 1. Prepare training data from VTK slices
cardioscar prepare \
    --mesh-vtk data/mesh.vtk \
    --grid-layers data/slice_*.vtk \
    --output data/training_data.npz

# 2. Train model
cardioscar train \
    --training-data data/training_data.npz \
    --output models/scar_model.pth

# 3. Apply to mesh
cardioscar apply \
    --model models/scar_model.pth \
    --mesh data/mesh.vtk \
    --output results/scar_predictions.vtk
```

### Python API

```python
from pathlib import Path
from cardioscar.logic import (
    PreprocessingRequest,
    prepare_training_data,
    save_preprocessing_result,
    TrainingConfig,
    train_scar_model,
    save_trained_model,
    apply_scar_model,
    save_inference_result
)

# 1. Prepare data
request = PreprocessingRequest(
    mesh_path=Path("mesh.vtk"),
    vtk_grid_paths=[Path("slice1.vtk"), Path("slice2.vtk")],
    slice_thickness_padding=5.0
)
result = prepare_training_data(request)
save_preprocessing_result(result, Path("training_data.npz"))

# 2. Train model
config = TrainingConfig(max_epochs=10000, early_stopping_patience=500)
checkpoint = train_scar_model(Path("training_data.npz"), config)
save_trained_model(checkpoint, Path("model.pth"))

# 3. Apply model
inference_result = apply_scar_model(
    model_checkpoint_path=Path("model.pth"),
    mesh_path=Path("mesh.vtk"),
    mc_samples=10
)
save_inference_result(inference_result, Path("mesh.vtk"), Path("output.vtk"))
```

---

## Command Line Interface

### Global Options

```bash
cardioscar --version  # Show version
cardioscar --help     # Show help
```

---

### `cardioscar prepare`

Prepare training data from mesh and image slices.

#### VTK Grid Input

```bash
cardioscar prepare \
    --mesh-vtk MESH_PATH \
    --grid-layers SLICE1.vtk [SLICE2.vtk ...] \
    [--vtk-scalar-field FIELD_NAME] \
    --output OUTPUT.npz \
    [--slice-thickness-padding MM]
```

**Options:**

| Option                      | Type    | Required | Default       | Description                                                |
| --------------------------- | ------- | -------- | ------------- | ---------------------------------------------------------- |
| `--mesh-vtk`                | Path    | Yes      | -             | Path to 3D target mesh (VTK)                               |
| `--grid-layers`             | Path(s) | Yes*     | -             | Paths to 2D grid layers (VTK). Can specify multiple times. |
| `--vtk-scalar-field`        | str     | No       | `ScalarValue` | Name of scalar field in VTK grids                          |
| `--image`                   | Path    | Yes*     | -             | Path to medical image (NIfTI, NRRD)                        |
| `--slice-axis`              | str     | No       | `z`           | Axis to slice along: `x`, `y`, or `z`                      |
| `--slice-indices`           | str     | No       | All slices    | Comma-separated slice indices (e.g., `"2,5,8,11"`)         |
| `--output`                  | Path    | Yes      | -             | Output path for training data (.npz)                       |
| `--slice-thickness-padding` | float   | No       | 5.0           | Z-direction padding (mm) for slice thickness               |

*Must provide either `--grid-layers` OR `--image`

#### Medical Image Input

```bash
cardioscar prepare \
    --mesh-vtk MESH_PATH \
    --image IMAGE.nii.gz \
    [--slice-axis {x,y,z}] \
    [--slice-indices "2,5,8,11"] \
    --output OUTPUT.npz \
    [--slice-thickness-padding MM]
```

**Example:**
```bash
# From VTK grids
cardioscar prepare \
    --mesh-vtk heart.vtk \
    --grid-layers slice_*.vtk \
    --output data.npz

# From NIfTI image (specific slices)
cardioscar prepare \
    --mesh-vtk heart.vtk \
    --image lge.nii.gz \
    --slice-axis z \
    --slice-indices "5,10,15,20" \
    --output data.npz

# From NIfTI image (all slices)
cardioscar prepare \
    --mesh-vtk heart.vtk \
    --image lge.nii.gz \
    --output data.npz
```

---

### `cardioscar train`

Train scar reconstruction model.

```bash
cardioscar train \
    --training-data DATA.npz \
    --output MODEL.pth \
    [OPTIONS]
```

**Options:**

| Option                      | Type | Required | Default | Description                                           |
| --------------------------- | ---- | -------- | ------- | ----------------------------------------------------- |
| `--training-data`           | Path | Yes      | -       | Path to training data (.npz)                          |
| `--output`                  | Path | Yes      | -       | Output path for trained model (.pth)                  |
| `--batch-size`              | int  | No       | 10000   | Target batch size (actual varies for complete groups) |
| `--max-epochs`              | int  | No       | 10000   | Maximum training epochs                               |
| `--early-stopping-patience` | int  | No       | 500     | Epochs without improvement before stopping            |
| `--mc-samples`              | int  | No       | 3       | MC Dropout samples during training                    |
| `--hidden-size`             | int  | No       | 128     | Neurons per hidden layer                              |
| `--hidden-layers`           | int  | No       | 4       | Number of hidden layers                               |
| `--cpu`                     | flag | No       | False   | Force CPU usage (default: auto-detect GPU)            |

**Example:**
```bash
# Default settings (recommended)
cardioscar train \
    --training-data data.npz \
    --output model.pth

# Larger architecture
cardioscar train \
    --training-data data.npz \
    --output model.pth \
    --hidden-size 256 \
    --hidden-layers 4

# Fast prototyping
cardioscar train \
    --training-data data.npz \
    --output model.pth \
    --max-epochs 1000 \
    --early-stopping-patience 100
```

**Training Time:** ~2 minutes on GPU for typical single-patient datasets (~290k nodes)

---

### `cardioscar fine-tune`

Fine-tune a pretrained foundation model on new patient data.

Architecture (hidden size, number of layers) is always restored from the
checkpoint and cannot be overridden at fine-tune time.

```bash
cardioscar fine-tune \
    --checkpoint FOUNDATION.pth \
    --training-data PATIENT.npz \
    --output FINETUNED.pth \
    [OPTIONS]
```

**Options:**

| Option                      | Type  | Required | Default | Description                                              |
| --------------------------- | ----- | -------- | ------- | -------------------------------------------------------- |
| `--checkpoint`              | Path  | Yes      | -       | Path to pretrained foundation model (.pth)               |
| `--training-data`           | Path  | Yes      | -       | Path to fine-tuning training data (.npz)                 |
| `--output`                  | Path  | Yes      | -       | Output path for fine-tuned model (.pth)                  |
| `--freeze-stages`           | int   | No       | 0       | Stages to freeze (0-4). 0 = full fine-tune (recommended) |
| `--batch-size`              | int   | No       | 10000   | Target batch size                                        |
| `--max-epochs`              | int   | No       | 1000    | Maximum fine-tuning epochs                               |
| `--early-stopping-patience` | int   | No       | 200     | Epochs without improvement before stopping               |
| `--mc-samples`              | int   | No       | 3       | MC Dropout samples during training                       |
| `--base-lr`                 | float | No       | 1e-4    | Base learning rate (lower than scratch training)         |
| `--max-lr`                  | float | No       | 1e-3    | Maximum learning rate (lower than scratch training)      |
| `--cpu`                     | flag  | No       | False   | Force CPU usage (default: auto-detect GPU)               |

**Example:**
```bash
# Full fine-tune (recommended)
cardioscar fine-tune \
    --checkpoint foundation_model.pth \
    --training-data patient_001.npz \
    --output patient_001_finetuned.pth

# Frozen backbone (experimental - only valid for 4-layer models)
cardioscar fine-tune \
    --checkpoint foundation_model.pth \
    --training-data patient_001.npz \
    --output patient_001_finetuned.pth \
    --freeze-stages 2
```

**Freeze stages** (valid for default 4-layer architecture only):

| Value | Frozen layers                      |
| ----- | ---------------------------------- |
| 0     | None (full fine-tune, recommended) |
| 1     | Linear(3→H) + ReLU                 |
| 2     | + Linear(H→H) + Dropout + ReLU     |
| 3     | + Linear(H→H) + ReLU               |
| 4     | + Linear(H→H) + Dropout + ReLU     |

---

### `cardioscar apply`

Apply trained model to mesh.

```bash
cardioscar apply \
    --model MODEL.pth \
    --mesh MESH.vtk \
    --output OUTPUT.vtk \
    [OPTIONS]
```

**Options:**

| Option         | Type  | Required | Default | Description                                       |
| -------------- | ----- | -------- | ------- | ------------------------------------------------- |
| `--model`      | Path  | Yes      | -       | Path to trained model (.pth)                      |
| `--mesh`       | Path  | Yes      | -       | Path to input mesh (VTK)                          |
| `--output`     | Path  | Yes      | -       | Output path for augmented mesh (.vtk)             |
| `--mc-samples` | int   | No       | 10      | MC Dropout samples for uncertainty estimation     |
| `--batch-size` | int   | No       | 50000   | Batch size for inference                          |
| `--threshold`  | float | No       | None    | Optional threshold for binary scar classification |
| `--cpu`        | flag  | No       | False   | Force CPU usage                                   |

**Output Fields:**

The output mesh contains:
- `scar_probability`: (N,) Mean scar probability per node [0, 1]
- `scar_uncertainty`: (N,) Uncertainty (std) per node
- `scar_binary`: (N,) Binary classification (if `--threshold` provided)

**Example:**
```bash
# Basic inference
cardioscar apply \
    --model model.pth \
    --mesh heart.vtk \
    --output scar_predictions.vtk

# High-quality uncertainty
cardioscar apply \
    --model model.pth \
    --mesh heart.vtk \
    --output scar_predictions.vtk \
    --mc-samples 20

# With binary threshold
cardioscar apply \
    --model model.pth \
    --mesh heart.vtk \
    --output scar_predictions.vtk \
    --threshold 0.5
```

---

## Python API

### Orchestrators

High-level workflow functions that coordinate library components.

#### `prepare_training_data()`

```python
from cardioscar.logic import prepare_training_data, PreprocessingRequest, PreprocessingResult
from pathlib import Path

def prepare_training_data(
    request: PreprocessingRequest
) -> PreprocessingResult:
    """
    Orchestrate training data preparation from mesh and slices.

    Args:
        request: PreprocessingRequest specifying inputs and parameters

    Returns:
        PreprocessingResult with training arrays and metadata

    Raises:
        ValueError: If no valid mappings found (slices don't overlap mesh)
    """
```

**Example:**
```python
# VTK grids
request = PreprocessingRequest(
    mesh_path=Path("mesh.vtk"),
    vtk_grid_paths=[Path("slice1.vtk"), Path("slice2.vtk")],
    slice_thickness_padding=5.0
)
result = prepare_training_data(request)
print(f"Prepared {result.n_nodes} nodes in {result.n_groups} groups")
```

---

#### `train_scar_model()`

```python
from cardioscar.logic import train_scar_model
from cardioscar.training.config import TrainingConfig
import torch

def train_scar_model(
    training_data_path: Path,
    config: TrainingConfig,
    device: Optional[torch.device] = None
) -> dict:
    """
    Orchestrate model training workflow.

    Args:
        training_data_path: Path to .npz training data
        config: Training configuration (includes hidden_size, n_hidden_layers)
        device: torch.device (auto-detected if None)

    Returns:
        Dictionary containing:
        - 'model_state_dict': Trained model weights
        - 'history': Training history (losses, learning rates)
        - 'hyperparameters': Model config including hidden_size, n_hidden_layers
        - 'dataset_info': Scaler params and metadata

    Example:
        >>> config = TrainingConfig(max_epochs=5000, hidden_size=256)
        >>> checkpoint = train_scar_model(Path("data.npz"), config)
        >>> print(f"Best loss: {checkpoint['history']['best_loss']:.4f}")
    """
```

---

#### `fine_tune_scar_model()`

```python
from cardioscar.logic import fine_tune_scar_model
from cardioscar.training.config import FineTuneConfig

def fine_tune_scar_model(
    training_data_path: Path,
    checkpoint_path: Path,
    config: FineTuneConfig,
    device: Optional[torch.device] = None
) -> dict:
    """
    Orchestrate fine-tuning of a pretrained scar reconstruction model.

    Loads a pretrained checkpoint, restores architecture from its hyperparameters,
    optionally freezes early network stages, and continues training on new data
    using a lower learning rate schedule.

    Args:
        training_data_path: Path to .npz fine-tuning training data
        checkpoint_path: Path to pretrained .pth checkpoint
        config: FineTuneConfig controlling LR, epochs, and layer freezing
        device: torch.device (auto-detected if None)

    Returns:
        Checkpoint dictionary in the same structure as train_scar_model(),
        with additional keys:
        - 'fine_tuned_from': Source checkpoint path (str)
        - 'freeze_stages': Number of stages frozen (int)

    Example:
        >>> config = FineTuneConfig(freeze_stages=0)
        >>> checkpoint = fine_tune_scar_model(
        ...     training_data_path=Path("patient_001.npz"),
        ...     checkpoint_path=Path("foundation_model.pth"),
        ...     config=config
        ... )
        >>> checkpoint['fine_tuned_from']
        'foundation_model.pth'
    """
```

---

#### `apply_scar_model()`

```python
from cardioscar.logic import apply_scar_model, InferenceResult

def apply_scar_model(
    model_checkpoint_path: Path,
    mesh_path: Path,
    mc_samples: int = 10,
    batch_size: int = 50000,
    threshold: Optional[float] = None,
    device: Optional[torch.device] = None
) -> InferenceResult:
    """
    Orchestrate model inference on mesh.

    Architecture is restored automatically from the checkpoint hyperparameters.

    Args:
        model_checkpoint_path: Path to trained .pth checkpoint
        mesh_path: Path to input mesh (VTK)
        mc_samples: Number of MC Dropout samples
        batch_size: Batch size for inference
        threshold: Optional threshold for binary classification
        device: torch.device (auto-detected if None)

    Returns:
        InferenceResult with predictions and metadata

    Example:
        >>> result = apply_scar_model(
        ...     Path("model.pth"),
        ...     Path("mesh.vtk"),
        ...     mc_samples=20
        ... )
        >>> print(f"Mean scar probability: {result.mean_scar_probability:.3f}")
    """
```

---

### Data Contracts

#### `PreprocessingRequest`

```python
@dataclass
class PreprocessingRequest:
    """
    Request to create training data from mesh and slices.

    Attributes:
        mesh_path: Path to 3D target mesh (VTK)
        slice_thickness_padding: Z-direction padding (mm)

        # VTK Grid Input
        vtk_grid_paths: List of paths to VTK grid files
        vtk_scalar_field: Name of scalar field in VTK grids

        # Image Input
        image_path: Path to medical image (NIfTI, NRRD)
        slice_axis: Axis to slice along ('x', 'y', 'z')
        slice_indices: List of slice indices (None = all slices)
    """
```

---

#### `PreprocessingResult`

```python
@dataclass
class PreprocessingResult:
    """
    Result of spatial mapping preprocessing.

    Attributes:
        coordinates: (N, 3) normalized mesh coordinates [0, 1]
        intensities: (N, 1) target scar values
        group_ids: (N,) group assignment per node
        group_sizes: (M,) number of nodes per unique group
        scaler_min: (3,) minimum values for denormalization
        scaler_max: (3,) maximum values for denormalization
        n_nodes: Total number of nodes
        n_groups: Total number of unique groups
    """
```

---

#### `InferenceResult`

```python
@dataclass
class InferenceResult:
    """
    Result of model inference on mesh.

    Attributes:
        mean_predictions: (N,) mean scar probability per node [0, 1]
        std_predictions: (N,) uncertainty (std) per node
        binary_predictions: (N,) optional binary classification
        n_nodes: Total number of nodes
        mean_scar_probability: Global mean scar probability
        mean_uncertainty: Global mean uncertainty
    """
```

---

### Neural Network Models

#### `BayesianNN`

```python
from cardioscar.engines import BayesianNN

class BayesianNN(nn.Module):
    """
    Coordinate-based Bayesian neural network with MC Dropout.

    Maps (X, Y, Z) -> scar probability [0, 1].

    Args:
        dropout_rate: Dropout probability (default: 0.1)
        hidden_size: Neurons per hidden layer (default: 128)
        n_hidden_layers: Number of hidden layers (default: 4)

    Architecture (defaults):
        Input(3) -> Dense(128) -> Dense(128) -> Dropout(0.1) ->
        Dense(128) -> Dense(128) -> Dropout(0.1) ->
        Dense(1, sigmoid) -> Output

        Dropout is inserted after every second hidden layer.
        Output layer is never frozen during fine-tuning.

    Methods:
        forward(x): Forward pass
        enable_dropout(): Enable dropout for MC sampling
        count_parameters(): Count trainable parameters
    """
```

**Example:**
```python
# Default architecture (50,177 parameters)
model = BayesianNN()
print(f"Parameters: {model.count_parameters():,}")

# Wider architecture (198,657 parameters)
model = BayesianNN(hidden_size=256, n_hidden_layers=4)
print(f"Parameters: {model.count_parameters():,}")

# MC Dropout inference
model.eval()
model.enable_dropout()
samples = [model(input_coords) for _ in range(10)]
mean_pred = torch.stack(samples).mean(dim=0)
std_pred = torch.stack(samples).std(dim=0)
```

---

#### `compute_group_reconstruction_loss()`

```python
from cardioscar.engines import compute_group_reconstruction_loss

def compute_group_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    return_per_group: bool = False
) -> torch.Tensor:
    """
    Compute group-based reconstruction loss via scatter_add (fully vectorized).

    Enforces constraint: mean(predictions_in_group) = target_value

    Loss Formula:
        For each group g:
            loss_g = (mean(predictions[group_g]) - target_g)^2
        Total loss = mean(loss_g for all groups)

    Args:
        predictions: (N,) predicted scar probabilities
        targets: (N,) target scar values (same per group)
        group_ids: (N,) group assignment per node
        return_per_group: If True, return per-group losses tensor

    Returns:
        Scalar mean loss, or (N_groups,) per-group losses if return_per_group
    """
```

---

### Training Configuration

#### `TrainingConfig`

```python
from cardioscar.training.config import TrainingConfig, DEFAULT_HIDDEN_SIZE, DEFAULT_N_HIDDEN_LAYERS

@dataclass
class TrainingConfig:
    """
    Configuration for training BayesianNN model.

    Attributes:
        batch_size: Target batch size (default: 10000)
        max_epochs: Maximum training epochs (default: 10000)
        early_stopping_patience: Epochs without improvement (default: 500)
        mc_samples: MC Dropout samples during training (default: 3)
        dropout_rate: Dropout probability (default: 0.1)
        hidden_size: Neurons per hidden layer (default: DEFAULT_HIDDEN_SIZE=128)
        n_hidden_layers: Number of hidden layers (default: DEFAULT_N_HIDDEN_LAYERS=4)
        base_lr: Minimum learning rate for cyclical schedule (default: 1e-3)
        max_lr: Maximum learning rate for cyclical schedule (default: 1e-2)
        lr_step_size: Iterations per half-cycle (default: 2000)
        seed: Random seed (default: 42)
    """
```

**Example:**
```python
# Default (recommended for single-patient training)
config = TrainingConfig()

# Wider architecture
config = TrainingConfig(hidden_size=256, n_hidden_layers=4)

# Foundation model training (large batch, longer patience)
config = TrainingConfig(
    batch_size=500000,
    max_epochs=3000,
    early_stopping_patience=500,
    mc_samples=3,
)
```

---

#### `FineTuneConfig`

```python
from cardioscar.training.config import FineTuneConfig

@dataclass
class FineTuneConfig:
    """
    Configuration for fine-tuning a pretrained BayesianNN checkpoint.

    Composes a TrainingConfig with fine-tune-appropriate defaults and
    adds layer-freezing control via freeze_stages.

    Note:
        Architecture (hidden_size, n_hidden_layers) is always restored
        from the checkpoint. Values in the inner TrainingConfig are ignored
        for architecture - they control only training hyperparameters.

        freeze_stages is only supported for n_hidden_layers=4 (default).
        Attempting to freeze stages on a different depth raises ValueError.

    Attributes:
        training: Inner TrainingConfig with fine-tune defaults:
            max_epochs=1000, early_stopping_patience=200,
            base_lr=1e-4, max_lr=1e-3 (10x lower than scratch)
        freeze_stages: Stages to freeze (0-4). 0 = full fine-tune.

    Methods:
        frozen_child_range(): Returns (start, end) child indices to freeze,
            or None if freeze_stages=0.
    """
```

**Example:**
```python
# Full fine-tune (recommended)
config = FineTuneConfig()

# Frozen backbone
config = FineTuneConfig(freeze_stages=2)

# Custom training hyperparameters
config = FineTuneConfig(
    training=TrainingConfig(
        max_epochs=2000,
        early_stopping_patience=500,
        base_lr=1e-4,
        max_lr=1e-3,
    ),
    freeze_stages=0,
)
```

---

#### Architecture Constants

```python
from cardioscar.training.config import DEFAULT_HIDDEN_SIZE, DEFAULT_N_HIDDEN_LAYERS

DEFAULT_HIDDEN_SIZE: int = 128      # Neurons per hidden layer
DEFAULT_N_HIDDEN_LAYERS: int = 4    # Number of hidden layers
```

These are the single source of truth for default architecture. Used as
fallback defaults when loading legacy checkpoints that predate architecture
parameterisation.

---

### Utilities

#### `ScarReconstructionDataset`

```python
from cardioscar.utilities import ScarReconstructionDataset

class ScarReconstructionDataset(torch.utils.data.Dataset):
    """
    Dataset for scar reconstruction training.

    Pre-sorts data by group_id to enable complete-group batching.

    Attributes:
        coordinates: (N, 3) normalized coordinates
        intensities: (N, 1) target values
        group_ids: (N,) group assignments
        group_sizes: (M,) nodes per group
    """
```

---

#### `create_complete_group_batches()`

```python
def create_complete_group_batches(
    dataset: ScarReconstructionDataset,
    target_batch_size: int
) -> List[Tuple[int, int]]:
    """
    Create batches that respect group boundaries.

    No group is ever split across two batches. Actual batch sizes
    vary to maintain this constraint.

    Args:
        dataset: ScarReconstructionDataset
        target_batch_size: Approximate batch size

    Returns:
        List of (start_idx, end_idx) tuples defining batches
    """
```

---

## Examples

### Example 1: Basic Workflow

```python
from pathlib import Path
from cardioscar.logic import *
from cardioscar.training.config import TrainingConfig

request = PreprocessingRequest(
    mesh_path=Path("data/heart.vtk"),
    vtk_grid_paths=[Path(f"data/slice_{i}.vtk") for i in range(10)],
    slice_thickness_padding=5.0
)

result = prepare_training_data(request)
save_preprocessing_result(result, Path("data/training.npz"))

config = TrainingConfig()
checkpoint = train_scar_model(Path("data/training.npz"), config)
save_trained_model(checkpoint, Path("models/scar.pth"))

result = apply_scar_model(
    model_checkpoint_path=Path("models/scar.pth"),
    mesh_path=Path("data/heart.vtk"),
    mc_samples=20
)
save_inference_result(result, Path("data/heart.vtk"), Path("results/scar.vtk"))
```

---

### Example 2: Foundation Model + Fine-Tune

```python
from pathlib import Path
from cardioscar.logic import train_scar_model, fine_tune_scar_model, save_trained_model
from cardioscar.training.config import TrainingConfig, FineTuneConfig

# 1. Train foundation model on cohort data
foundation_config = TrainingConfig(
    batch_size=500000,
    max_epochs=3000,
    early_stopping_patience=500,
)
foundation_checkpoint = train_scar_model(
    Path("foundation_training.npz"),
    foundation_config
)
save_trained_model(foundation_checkpoint, Path("foundation_model.pth"))

# 2. Fine-tune per patient (full fine-tune recommended)
finetune_config = FineTuneConfig(freeze_stages=0)
patient_checkpoint = fine_tune_scar_model(
    training_data_path=Path("patient_001.npz"),
    checkpoint_path=Path("foundation_model.pth"),
    config=finetune_config,
)
save_trained_model(patient_checkpoint, Path("patient_001_finetuned.pth"))

print(f"Fine-tuned from: {patient_checkpoint['fine_tuned_from']}")
print(f"Best loss: {patient_checkpoint['history']['best_loss']:.6f}")
```

---

### Example 3: Uncertainty-Based Quality Control

```python
import numpy as np

def quality_control_check(result: InferenceResult, threshold_std: float = 0.3):
    """Flag high-uncertainty regions for manual review."""
    high_uncertainty_mask = result.std_predictions > threshold_std
    n_flagged = high_uncertainty_mask.sum()
    pct_flagged = 100 * n_flagged / result.n_nodes

    qc = {
        'total_nodes': result.n_nodes,
        'flagged_nodes': n_flagged,
        'flagged_percentage': pct_flagged,
        'mean_uncertainty': result.mean_uncertainty,
        'max_uncertainty': result.std_predictions.max(),
    }

    if pct_flagged > 20:
        qc['recommendation'] = "Consider acquiring additional slices"
    elif pct_flagged > 10:
        qc['recommendation'] = "Review high-uncertainty regions manually"
    else:
        qc['recommendation'] = "Quality acceptable"

    return qc

result = apply_scar_model(Path("model.pth"), Path("mesh.vtk"))
qc = quality_control_check(result)
print(f"Flagged {qc['flagged_percentage']:.1f}% of nodes")
print(f"Recommendation: {qc['recommendation']}")
```

---

## Performance Benchmarks

### Training Performance

| Configuration    | Parameters | Dataset        | Training Time | Best Loss |
| ---------------- | ---------- | -------------- | ------------- | --------- |
| 4×128 (default)  | 50,177     | Single patient | ~2 min        | 0.000884  |
| 4×256            | 198,657    | Single patient | ~5.5 min      | 0.000872  |
| 6×256 (legacy)   | 330,241    | Single patient | ~2 min*       | 0.013940* |
| Foundation 4×128 | 50,177     | 11 patients    | ~88 min       | 0.007490  |

*Variance collapse - did not converge. Legacy architecture requires careful
initialisation and is not recommended.

### Fine-Tuning vs Scratch (single patient, subset_013_n10_f1_s15)

| Condition          | Best loss | Epochs | Wall time |
| ------------------ | --------- | ------ | --------- |
| Scratch (4×128)    | 0.000884  | 2164   | 1.7 min   |
| Fine-tune frozen=0 | 0.000627  | 4629   | 3.6 min   |
| Fine-tune frozen=2 | 0.004748  | 300+   | 0.2 min   |

Fine-tuning achieves 29% lower loss than scratch. Frozen backbone (frozen=2)
is not recommended - insufficient capacity to overcome pretrained representation
in limited epochs.

### Inference Performance

| Mesh Size  | MC Samples | GPU      | Time    |
| ---------- | ---------- | -------- | ------- |
| 516k nodes | 10         | RTX 5090 | <30 sec |

### Model Size

| Implementation     | Parameters | Notes                       |
| ------------------ | ---------- | --------------------------- |
| Legacy (6×256)     | 330,241    | TensorFlow, unstable init   |
| CardioScar (4×128) | 50,177     | Default, recommended        |
| CardioScar (4×256) | 198,657    | Marginal improvement on 50k |

---

## Common Pitfalls

### 1. Mesh and Slices Don't Overlap

**Symptom:** `ValueError: No valid mappings found`

**Solutions:** Check coordinate ranges, verify slice axis, confirm image orientation matches mesh coordinate system.

---

### 2. CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

```bash
cardioscar train --training-data data.npz --batch-size 5000 --mc-samples 2
```

---

### 3. Training Doesn't Converge

Loss plateaus above 0.01: check data quality, verify group constraints, try lower learning rate.

---

### 4. Predictions All Zero (Variance Collapse)

Seen with 6-layer architectures. Switch to default 4-layer architecture. If using deeper networks, reduce learning rate significantly.

---

### 5. Checkpoint Architecture Mismatch

**Symptom:** `RuntimeError: size mismatch` when loading checkpoint.

Architecture is always read from the checkpoint `hyperparameters` dict. Do not manually pass `hidden_size` or `n_hidden_layers` to `apply_scar_model` or `fine_tune_scar_model` - they are restored automatically.

---

### 6. Legacy Checkpoint Compatibility

Checkpoints saved before architecture parameterisation (without `hidden_size`/`n_hidden_layers` in `hyperparameters`) load correctly via `.get()` fallbacks to `DEFAULT_HIDDEN_SIZE` and `DEFAULT_N_HIDDEN_LAYERS`.

---

### 7. Freeze Stages on Non-Default Depth

**Symptom:** `ValueError: freeze_stages is only supported for n_hidden_layers=4`

Layer freezing is only implemented for the default 4-layer architecture. Use `freeze_stages=0` for any model trained with `--hidden-layers != 4`.

---

## Citation

```bibtex
@article{SEN2025111219,
  title = {Weakly supervised learning for scar reconstruction in personalized cardiac models},
  journal = {Computers in Biology and Medicine},
  volume = {198},
  pages = {111219},
  year = {2025},
  doi = {https://doi.org/10.1016/j.compbiomed.2025.111219},
  author = {Ahmet SEN and Ursula Rohrer and Pranav Bhagirath and Reza Razavi and Mark O'Neill and John Whitaker and Martin Bishop},
}

@software{cardioscar2024,
  title={CardioScar: Deep Learning-Based 3D Myocardial Scar Reconstruction},
  author={Sen, Ahmet and Bishop, Martin J. and Solis-Lemus, Jose Alonso},
  year={2024},
  url={https://github.com/alonsoJASL/cardioscar},
  version={0.2.0}
}
```

---

**Version:** 0.2.0  
**Last Updated:** March 2026