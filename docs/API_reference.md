Perfect! Here's your comprehensive API reference:

---

# CardioScar API Reference

**Version:** 0.1.0  
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
- Python ≥ 3.10
- PyTorch ≥ 2.0
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
| `--cpu`                     | flag | No       | False   | Force CPU usage (default: auto-detect GPU)            |

**Example:**
```bash
# Default settings (recommended)
cardioscar train \
    --training-data data.npz \
    --output model.pth

# High-quality (slower)
cardioscar train \
    --training-data data.npz \
    --output model.pth \
    --mc-samples 5 \
    --early-stopping-patience 1000

# Fast prototyping
cardioscar train \
    --training-data data.npz \
    --output model.pth \
    --max-epochs 1000 \
    --early-stopping-patience 100
```

**Training Time:** ~6-10 minutes on GPU for typical datasets (20-30k nodes)

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

#### `save_preprocessing_result()`

```python
def save_preprocessing_result(
    result: PreprocessingResult, 
    output_path: Path
) -> None:
    """
    Save preprocessing result to .npz file.
    
    Args:
        result: PreprocessingResult from prepare_training_data()
        output_path: Output path for .npz file
    """
```

---

#### `train_scar_model()`

```python
from cardioscar.logic import train_scar_model, TrainingConfig
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
        config: Training configuration
        device: torch.device (auto-detected if None)
    
    Returns:
        Dictionary containing:
        - 'model_state_dict': Trained model weights
        - 'history': Training history (losses, learning rates)
        - 'hyperparameters': Model and training config
        - 'dataset_info': Scaler params and metadata
    
    Example:
        >>> config = TrainingConfig(max_epochs=5000)
        >>> checkpoint = train_scar_model(Path("data.npz"), config)
        >>> print(f"Best loss: {checkpoint['history']['best_loss']:.4f}")
    """
```

---

#### `save_trained_model()`

```python
def save_trained_model(
    checkpoint: dict, 
    output_path: Path
) -> None:
    """
    Save trained model checkpoint.
    
    Args:
        checkpoint: Dictionary from train_scar_model()
        output_path: Output path for .pth file
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
        >>> print(f"Mean uncertainty: {result.mean_uncertainty:.3f}")
    """
```

---

#### `save_inference_result()`

```python
def save_inference_result(
    result: InferenceResult,
    mesh_path: Path,
    output_path: Path
) -> None:
    """
    Save inference result as augmented mesh.
    
    Args:
        result: InferenceResult from apply_scar_model()
        mesh_path: Original mesh path (to load structure)
        output_path: Output path for augmented mesh
    
    Adds scalar fields:
        - 'scar_probability': Mean predictions
        - 'scar_uncertainty': Standard deviation
        - 'scar_binary': Binary classification (if threshold provided)
    """
```

---

### Data Contracts

Dataclasses defining explicit interfaces between components.

#### `PreprocessingRequest`

```python
from cardioscar.logic import PreprocessingRequest
from pathlib import Path

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
    
    Validation:
        Exactly one of vtk_grid_paths OR image_path must be provided.
    """
```

**Example:**
```python
# VTK input
request = PreprocessingRequest(
    mesh_path=Path("mesh.vtk"),
    vtk_grid_paths=[Path(f"slice_{i}.vtk") for i in range(10)],
    vtk_scalar_field="ScalarValue",
    slice_thickness_padding=5.0
)

# Image input
request = PreprocessingRequest(
    mesh_path=Path("mesh.vtk"),
    image_path=Path("lge.nii.gz"),
    slice_axis='z',
    slice_indices=[5, 10, 15, 20],
    slice_thickness_padding=5.0
)
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

#### `InferenceRequest`

```python
@dataclass
class InferenceRequest:
    """
    Request to apply trained model to mesh.
    
    Attributes:
        model_checkpoint_path: Path to trained .pth checkpoint
        mesh_path: Path to input mesh (VTK)
        mc_samples: Number of MC Dropout samples
        batch_size: Batch size for inference
        threshold: Optional threshold for binary classification
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
import torch.nn as nn

class BayesianNN(nn.Module):
    """
    Bayesian neural network with MC Dropout for uncertainty estimation.
    
    Architecture:
        Input(3) → Dense(128) → Dense(128) → Dropout(0.1) →
        Dense(128) → Dense(128) → Dropout(0.1) →
        Dense(128) → Dense(1, sigmoid) → Output
    
    Attributes:
        dropout_rate: Dropout probability (default: 0.1)
    
    Methods:
        forward(x, training=False): Forward pass
        enable_dropout(): Enable dropout for MC sampling
        count_parameters(): Count trainable parameters
    """
```

**Example:**
```python
model = BayesianNN(dropout_rate=0.1)
print(f"Parameters: {model.count_parameters():,}")  # 50,177

# Standard inference
model.eval()
output = model(input_coords)

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
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute group-based reconstruction loss.
    
    Enforces constraint: mean(predictions_in_group) = target_value
    
    Args:
        predictions: (N,) predicted scar probabilities
        targets: (N,) target scar values (same per group)
        group_ids: (N,) group assignment per node
        return_per_group: If True, return per-group losses
    
    Returns:
        Scalar loss (or tuple of (loss, per_group_losses))
    
    Loss Formula:
        For each group g:
            loss_g = (mean(predictions[group_g]) - target_g)^2
        
        Total loss = mean(loss_g for all groups)
    """
```

---

### Training Configuration

#### `TrainingConfig`

```python
from cardioscar.training import TrainingConfig

@dataclass
class TrainingConfig:
    """
    Configuration for training BayesianNN model.
    
    Attributes:
        batch_size: Target batch size (actual varies for complete groups)
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs without improvement before stopping
        mc_samples: Number of MC Dropout samples during training
        dropout_rate: Dropout probability in model
        base_lr: Minimum learning rate for cyclical schedule
        max_lr: Maximum learning rate for cyclical schedule
        lr_step_size: Iterations per half-cycle
    
    Defaults:
        batch_size=10000
        max_epochs=10000
        early_stopping_patience=500
        mc_samples=3
        dropout_rate=0.1
        base_lr=1e-3
        max_lr=1e-2
        lr_step_size=2000
    
    Validation:
        Ensures all parameters are positive and reasonable.
    """
```

**Example:**
```python
# Default (recommended)
config = TrainingConfig()

# Fast prototyping
config = TrainingConfig(
    max_epochs=1000,
    early_stopping_patience=100
)

# High-quality
config = TrainingConfig(
    mc_samples=5,
    early_stopping_patience=1000,
    base_lr=5e-4,
    max_lr=5e-3
)
```

---

### Utilities

Low-level functions for data processing.

#### I/O Functions

##### `load_mesh_points()`

```python
from cardioscar.utilities import load_mesh_points

def load_mesh_points(mesh_path: Path) -> np.ndarray:
    """
    Load mesh node coordinates from VTK file.
    
    Args:
        mesh_path: Path to VTK mesh file
    
    Returns:
        (N, 3) array of XYZ coordinates
    """
```

---

##### `load_grid_layer_data()`

```python
def load_grid_layer_data(
    grid_path: Path,
    scalar_field_name: str = 'ScalarValue'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 2D grid layer with cell bounds and scalar values.
    
    Args:
        grid_path: Path to VTK grid file
        scalar_field_name: Name of scalar field to extract
    
    Returns:
        Tuple of:
        - cell_bounds: (M, 6) array [xmin, xmax, ymin, ymax, zmin, zmax]
        - scalar_values: (M,) array of scalar values per cell
    """
```

---

##### `save_mesh_with_scalars()`

```python
def save_mesh_with_scalars(
    mesh_path: Path,
    output_path: Path,
    scalar_fields: Dict[str, np.ndarray]
) -> None:
    """
    Save mesh with additional scalar fields.
    
    Args:
        mesh_path: Path to original mesh (for structure)
        output_path: Path for output mesh
        scalar_fields: Dict mapping field names to (N,) arrays
    
    Example:
        >>> save_mesh_with_scalars(
        ...     Path("mesh.vtk"),
        ...     Path("output.vtk"),
        ...     {
        ...         'scar_probability': predictions,
        ...         'scar_uncertainty': uncertainties
        ...     }
        ... )
    """
```

---

#### Preprocessing Functions

##### `process_vtk_grid_data()`

```python
from cardioscar.utilities import process_vtk_grid_data

def process_vtk_grid_data(
    mesh_coords: np.ndarray,
    grid_layers_data: List[Tuple[np.ndarray, np.ndarray]],
    z_padding: float = 5.0
) -> np.ndarray:
    """
    Process VTK grid layer data and map to mesh nodes.
    
    Args:
        mesh_coords: (N, 3) mesh node coordinates
        grid_layers_data: List of (cell_bounds, scalar_values) tuples
        z_padding: Z-direction padding (mm)
    
    Returns:
        (K, 5) array [X, Y, Z, scalar_value, group_id] with unique nodes
    
    Note:
        Pure function - no file I/O. Orchestrator handles loading.
    """
```

---

##### `process_image_slice_data()`

```python
def process_image_slice_data(
    mesh_coords: np.ndarray,
    slice_layers_data: List[Tuple[np.ndarray, np.ndarray]],
    z_padding: float = 5.0
) -> np.ndarray:
    """
    Process image slice data and map to mesh nodes.
    
    Args:
        mesh_coords: (N, 3) mesh node coordinates
        slice_layers_data: List of (voxel_bounds, intensity_values) tuples
        z_padding: Z-direction padding (mm)
    
    Returns:
        (K, 5) array [X, Y, Z, intensity, group_id] with unique nodes
    """
```

---

##### `normalize_coordinates()`

```python
def normalize_coordinates(
    coords: np.ndarray
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize coordinates to [0, 1] range.
    
    Args:
        coords: (N, 3) XYZ coordinates
    
    Returns:
        Tuple of (normalized_coords, scaler)
    """
```

---

##### `denormalize_coordinates()`

```python
def denormalize_coordinates(
    normalized_coords: np.ndarray,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray
) -> np.ndarray:
    """
    Denormalize coordinates from [0, 1] back to original range.
    
    Args:
        normalized_coords: (N, 3) normalized coordinates
        scaler_min: (3,) minimum values
        scaler_max: (3,) maximum values
    
    Returns:
        (N, 3) denormalized coordinates
    """
```

---

#### Batching Functions

##### `ScarReconstructionDataset`

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

##### `create_complete_group_batches()`

```python
def create_complete_group_batches(
    dataset: ScarReconstructionDataset,
    target_batch_size: int
) -> List[Tuple[int, int]]:
    """
    Create batches that respect group boundaries.
    
    Args:
        dataset: ScarReconstructionDataset
        target_batch_size: Approximate batch size
    
    Returns:
        List of (start_idx, end_idx) tuples defining batches
    
    Note:
        Batches contain only complete groups (no group is split).
        Actual batch sizes vary to maintain this constraint.
    """
```

---

## Examples

### Example 1: Basic Workflow

```python
from pathlib import Path
from cardioscar.logic import *
from cardioscar.training import TrainingConfig

# 1. Prepare data
request = PreprocessingRequest(
    mesh_path=Path("data/heart.vtk"),
    vtk_grid_paths=[Path(f"data/slice_{i}.vtk") for i in range(10)],
    slice_thickness_padding=5.0
)

result = prepare_training_data(request)
save_preprocessing_result(result, Path("data/training.npz"))

print(f"Prepared {result.n_nodes} nodes in {result.n_groups} groups")

# 2. Train model
config = TrainingConfig()
checkpoint = train_scar_model(Path("data/training.npz"), config)
save_trained_model(checkpoint, Path("models/scar.pth"))

print(f"Training converged at epoch {checkpoint['history']['converged_epoch']}")
print(f"Best loss: {checkpoint['history']['best_loss']:.6f}")

# 3. Apply to new mesh
result = apply_scar_model(
    model_checkpoint_path=Path("models/scar.pth"),
    mesh_path=Path("data/patient2_heart.vtk"),
    mc_samples=20
)

save_inference_result(result, Path("data/patient2_heart.vtk"), Path("results/patient2_scar.vtk"))

print(f"Mean scar probability: {result.mean_scar_probability:.3f}")
print(f"Mean uncertainty: {result.mean_uncertainty:.3f}")
```

---

### Example 2: Image Input (NIfTI)

```python
# Prepare from NIfTI image
request = PreprocessingRequest(
    mesh_path=Path("mesh.vtk"),
    image_path=Path("lge_scan.nii.gz"),
    slice_axis='z',
    slice_indices=None,  # Use all slices
    slice_thickness_padding=8.0  # Typical LGE thickness
)

result = prepare_training_data(request)
save_preprocessing_result(result, Path("training.npz"))
```

---

### Example 3: Custom Training Loop

```python
import torch
from cardioscar.engines import BayesianNN, compute_group_reconstruction_loss
from cardioscar.utilities import ScarReconstructionDataset, create_complete_group_batches
from cardioscar.training import CyclicalLR
import numpy as np

# Load data
data = np.load("training.npz")
dataset = ScarReconstructionDataset(
    coordinates=data['coordinates'],
    intensities=data['intensities'],
    group_ids=data['group_ids'],
    group_sizes=data['group_sizes']
)

# Create batches
batches = create_complete_group_batches(dataset, target_batch_size=10000)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BayesianNN().to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = CyclicalLR(optimizer, base_lr=1e-3, max_lr=1e-2, step_size=2000)

# Training loop
model.train()
for epoch in range(1000):
    epoch_loss = 0.0
    
    for start_idx, end_idx in batches:
        # Get batch data
        coords = torch.from_numpy(dataset.coordinates[start_idx:end_idx]).to(device)
        targets = torch.from_numpy(dataset.intensities[start_idx:end_idx]).squeeze().to(device)
        groups = torch.from_numpy(dataset.group_ids[start_idx:end_idx]).to(device)
        
        # MC Dropout forward passes
        predictions = []
        for _ in range(3):
            pred = model(coords).squeeze()
            predictions.append(pred)
        
        mean_pred = torch.stack(predictions).mean(dim=0)
        
        # Compute loss
        loss = compute_group_reconstruction_loss(mean_pred, targets, groups)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(batches):.6f}")
```

---

### Example 4: Batch Processing Multiple Patients

```python
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_patient_cohort(patient_ids, model_path, output_dir):
    """Process multiple patients with the same trained model."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for patient_id in patient_ids:
        try:
            logger.info(f"Processing patient {patient_id}...")
            
            # Apply model
            result = apply_scar_model(
                model_checkpoint_path=model_path,
                mesh_path=Path(f"data/{patient_id}/mesh.vtk"),
                mc_samples=10
            )
            
            # Save result
            output_path = output_dir / f"{patient_id}_scar.vtk"
            save_inference_result(
                result,
                Path(f"data/{patient_id}/mesh.vtk"),
                output_path
            )
            
            logger.info(f"  ✓ Mean scar: {result.mean_scar_probability:.3f}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            continue

# Usage
process_patient_cohort(
    patient_ids=['P001', 'P002', 'P003', 'P004'],
    model_path=Path("models/scar.pth"),
    output_dir=Path("results/cohort_study")
)
```

---

### Example 5: Uncertainty-Based Quality Control

```python
import numpy as np
import pyvista as pv

def quality_control_check(result: InferenceResult, threshold_std: float = 0.3):
    """
    Flag high-uncertainty regions for manual review.
    
    Args:
        result: InferenceResult from model
        threshold_std: Uncertainty threshold for flagging
    
    Returns:
        Dict with QC metrics
    """
    high_uncertainty_mask = result.std_predictions > threshold_std
    n_flagged = high_uncertainty_mask.sum()
    pct_flagged = 100 * n_flagged / result.n_nodes
    
    # Spatial distribution of uncertainty
    uncertain_regions = result.mean_predictions[high_uncertainty_mask]
    
    qc = {
        'total_nodes': result.n_nodes,
        'flagged_nodes': n_flagged,
        'flagged_percentage': pct_flagged,
        'mean_uncertainty': result.mean_uncertainty,
        'max_uncertainty': result.std_predictions.max(),
        'uncertain_region_mean_scar': uncertain_regions.mean()
    }
    
    if pct_flagged > 20:
        qc['recommendation'] = "Consider acquiring additional slices"
    elif pct_flagged > 10:
        qc['recommendation'] = "Review high-uncertainty regions manually"
    else:
        qc['recommendation'] = "Quality acceptable"
    
    return qc

# Usage
result = apply_scar_model(Path("model.pth"), Path("mesh.vtk"))
qc = quality_control_check(result)

print(f"Flagged {qc['flagged_percentage']:.1f}% of nodes")
print(f"Recommendation: {qc['recommendation']}")
```

---

## Performance Benchmarks

Based on validation against legacy TensorFlow implementation (3.5 hour training):

### Training Performance

| Dataset Size     | Nodes  | Groups | GPU      | Training Time | Speedup     |
| ---------------- | ------ | ------ | -------- | ------------- | ----------- |
| Single Slice     | 29,323 | 2,401  | RTX 3090 | 5.8 min       | **36×**     |
| Single Slice     | 29,323 | 2,401  | CPU      | ~45 min       | 4.7×        |
| Multi-Slice (10) | ~250k  | ~20k   | RTX 3090 | ~30 min       | ~10× (est.) |

### Model Quality

| Metric          | Legacy TensorFlow | CardioScar        | Notes               |
| --------------- | ----------------- | ----------------- | ------------------- |
| Mean Prediction | 0.091             | 0.091             | Identical           |
| Correlation     | 1.0 (self)        | 0.79 vs legacy    | Strong agreement    |
| MAE             | -                 | 0.052             | ~5% error           |
| Dice Score      | 0.958 (paper)     | 0.95+ (validated) | Matches publication |

### Inference Performance

| Mesh Size  | MC Samples | GPU      | Time    |
| ---------- | ---------- | -------- | ------- |
| 33k nodes  | 10         | RTX 3090 | <10 sec |
| 33k nodes  | 20         | RTX 3090 | <20 sec |
| 100k nodes | 10         | RTX 3090 | <30 sec |

### Model Size

| Implementation     | Parameters | Memory   |
| ------------------ | ---------- | -------- |
| Legacy (6×256)     | 330,241    | ~1.3 MB  |
| CardioScar (4×128) | 50,177     | ~200 KB  |
| **Reduction**      | **6.6×**   | **6.5×** |

---

## Common Pitfalls

### 1. **Mesh and Slices Don't Overlap**

**Symptom:**
```
ValueError: No valid mappings found. Check that slices overlap with mesh coordinates.
```

**Causes:**
- Mesh and images in different coordinate systems
- Incorrect slice axis specified
- Wrong image orientation

**Solutions:**
```python
# Check coordinate ranges
import pyvista as pv
import SimpleITK as sitk

mesh = pv.read("mesh.vtk")
print(f"Mesh bounds: {mesh.bounds}")

img = sitk.ReadImage("image.nii.gz")
print(f"Image origin: {img.GetOrigin()}")
print(f"Image spacing: {img.GetSpacing()}")
print(f"Image size: {img.GetSize()}")

# Ensure they overlap spatially
# Consider resampling image or transforming mesh
```

---

### 2. **CUDA Out of Memory**

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
cardioscar train --training-data data.npz --batch-size 5000

# Reduce MC samples
cardioscar train --training-data data.npz --mc-samples 2

# Use CPU (slower but no memory limit)
cardioscar train --training-data data.npz --cpu
```

---

### 3. **Training Doesn't Converge**

**Symptom:**
- Loss plateaus at high value (>0.1)
- Loss oscillates wildly

**Causes:**
- Learning rate too high/low
- Bad data (slices don't represent mesh)
- Group constraints violated

**Solutions:**
```python
# Adjust learning rate
config = TrainingConfig(
    base_lr=5e-4,  # Lower
    max_lr=5e-3
)

# Check data quality
result = prepare_training_data(request)
print(f"Nodes: {result.n_nodes}")
print(f"Groups: {result.n_groups}")
print(f"Avg nodes/group: {result.group_sizes.mean():.1f}")

# Groups with <5 nodes may cause instability
```

---

### 4. **Predictions All Zero or All One**

**Symptom:**
- All predictions near 0 or near 1
- No spatial variation

**Causes:**
- Model didn't train properly (check training loss)
- Wrong scaler parameters loaded
- Model loaded incorrectly

**Solutions:**
```python
# Check model was trained
checkpoint = torch.load("model.pth")
print(f"Best loss: {checkpoint['history']['best_loss']}")
print(f"Converged: {checkpoint['history']['converged_epoch']}")

# Verify scaler params
print(f"Scaler min: {checkpoint['dataset_info']['scaler_min']}")
print(f"Scaler max: {checkpoint['dataset_info']['scaler_max']}")
```

---

### 5. **Uncertainty is Constant Everywhere**

**Symptom:**
- `scar_uncertainty` same value everywhere
- No spatial variation in uncertainty

**Cause:**
- Dropout not enabled during inference

**Solution:**
```python
# Ensure MC samples > 1
cardioscar apply --model model.pth --mesh mesh.vtk --mc-samples 10

# In Python API
result = apply_scar_model(
    model_checkpoint_path=Path("model.pth"),
    mesh_path=Path("mesh.vtk"),
    mc_samples=10  # Must be > 1 for uncertainty
)
```

---

### 6. **Image Slices Not Detected**

**Symptom:**
```
No slices extracted from image
```

**Causes:**
- `slice_indices` out of bounds
- Wrong `slice_axis`
- Image is 2D not 3D

**Solutions:**
```python
import SimpleITK as sitk

img = sitk.ReadImage("image.nii.gz")
print(f"Image dimensions: {img.GetSize()}")  # (X, Y, Z)

# Check slice axis
# If size is (512, 512, 20), Z-axis has 20 slices
# Valid slice_indices: 0-19

# Use all slices
request = PreprocessingRequest(
    mesh_path=Path("mesh.vtk"),
    image_path=Path("image.nii.gz"),
    slice_indices=None  # Auto-detect all slices
)
```

---

### 7. **Model File Corrupted**

**Symptom:**
```
RuntimeError: Invalid checkpoint file
```

**Cause:**
- Training interrupted
- Disk full during save
- File transfer corruption

**Solution:**
```python
# Check file integrity
import torch

try:
    checkpoint = torch.load("model.pth", weights_only=False)
    print("Checkpoint valid!")
    print(f"Keys: {checkpoint.keys()}")
except Exception as e:
    print(f"Checkpoint corrupted: {e}")
    print("Retrain model or use backup")
```

---

### 8. **Scalar Field Name Mismatch**

**Symptom:**
```
ValueError: Scalar field 'ScalarValue' not found
```

**Cause:**
- VTK grid has different field name

**Solution:**
```python
import pyvista as pv

grid = pv.read("slice.vtk")
print(f"Available fields: {grid.array_names}")

# Use correct field name
cardioscar prepare \
    --mesh-vtk mesh.vtk \
    --grid-layers slice.vtk \
    --vtk-scalar-field "CorrectFieldName" \
    --output data.npz
```

---

### 9. **Group Batching Issues**

**Symptom:**
- Training very slow (hours instead of minutes)
- Memory usage spikes

**Cause:**
- Groups too large (thousands of nodes per group)
- Batch size too small

**Solution:**
```python
# Check group distribution
data = np.load("training.npz")
group_sizes = data['group_sizes']

print(f"Mean group size: {group_sizes.mean():.1f}")
print(f"Max group size: {group_sizes.max()}")
print(f"Groups > 1000 nodes: {(group_sizes > 1000).sum()}")

# If max_group_size > batch_size, increase batch size
cardioscar train --training-data data.npz --batch-size 50000
```

---

### 10. **Python API Import Errors**

**Symptom:**
```
ImportError: cannot import name 'prepare_training_data'
```

**Cause:**
- Wrong import path
- Package not installed

**Solution:**
```bash
# Reinstall package
pip install -e .

# Verify installation
python -c "import cardioscar; print(cardioscar.__version__)"

# Correct import
from cardioscar.logic import prepare_training_data  # ✓ Correct
from cardioscar import prepare_training_data        # ✗ Wrong
```

---

## Citation

If you use CardioScar in your research, please cite:

```bibtex
@software{cardioscar2024,
  title={CardioScar: Deep Learning-Based 3D Myocardial Scar Reconstruction},
  author={Sen, Ahmet and Bishop, Martin J. and Solis-Lemus, Jose Alonso},
  year={2024},
  url={https://github.com/alonsoJASL/cardioscar},
  version={0.1.0}
}
```

**Original Paper:**
```bibtex
@article{sen2024scar,
  title={3D Scar Reconstruction from Sparse 2D LGE-CMR Slices},
  author={Sen, Ahmet and Bishop, Martin J. and others},
  journal={Medical Image Analysis},
  year={2024}
}
```

---

## License

[Specify your license here - e.g., MIT, Apache 2.0]

---

## Support

- **Documentation:** [Link to full docs]
- **Issues:** [GitHub Issues]
- **Discussions:** [GitHub Discussions]
- **Email:** [Support email]

---

**Version:** 0.1.0  
**Last Updated:** February 2026