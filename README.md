# CardioScar

Deep learning-based 3D myocardial scar reconstruction from sparse 2D Late Gadolinium-Enhanced Cardiac MRI (LGE-CMR).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CardioScar implements a **coordinate-based Bayesian neural network** that reconstructs continuous 3D scar probability fields from sparse 2D MRI slices. It addresses the challenge of low through-plane resolution (8-10mm) in LGE-CMR by learning smooth anatomically plausible interpolations that preserve narrow conducting isthmuses critical for arrhythmia prediction.

**Key Features:**
- **Faster training** than legacy implementation 
- **Bayesian uncertainty quantification** via Monte Carlo Dropout
- **Smaller model** (330k → 50k parameters) with equivalent accuracy
- **Production-ready architecture** - type-safe contracts, CLI tooling, comprehensive testing
- **Oblique image support** - correct handling of arbitrarily oriented medical images
- **Flexible input** - works with VTK grid slices or NIfTI/NRRD volumes
- 
---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Method Overview](#method-overview)
- [Examples](#examples)
- [Performance](#performance)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-enabled GPU recommended (CPU supported)
- pycemrg suite dependencies

### Setup

```bash
# 1. Clone repositories
git clone https://github.com/alonsoJASL/cardioscar.git
cd cardioscar

# Clone pycemrg dependencies (if not already installed)
git clone https://github.com/OpenHeartDevelopers/pycemrg.git ../pycemrg
git clone https://github.com/OpenHeartDevelopers/pycemrg-image-analysis.git ../pycemrg-image-analysis
git clone https://github.com/OpenHeartDevelopers/pycemrg-model-creation.git ../pycemrg-model-creation

# 2. Create environment
conda create -n cardioscar python=3.11 -y
conda activate cardioscar

# 3. Install dependencies
pip install -e ../pycemrg
pip install -e ../pycemrg-image-analysis
pip install -e ../pycemrg-model-creation
pip install -e .

# 4. (Optional) Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
cardioscar --version
python -c "import cardioscar; print('✓ CardioScar installed successfully')"
```

---

## Quick Start

CardioScar provides a unified CLI with three main commands: **`prepare`**, **`train`**, and **`apply`**.

### Complete Workflow (CLI)

```bash
# 1. Prepare training data from NIfTI image
cardioscar prepare \
    --mesh-vtk data/lv_mesh.vtk \
    --image data/lge_scan.nii.gz \
    --output data/training.npz

# 2. Train model
cardioscar train \
    --training-data data/training.npz \
    --output models/patient_001.pth

# 3. Apply to mesh
cardioscar apply \
    --model models/patient_001.pth \
    --mesh data/lv_mesh.vtk \
    --output results/scar_predictions.vtk \
    --mc-samples 20
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
    mesh_path=Path("data/lv_mesh.vtk"),
    image_path=Path("data/lge_scan.nii.gz"),
    slice_axis='z'
)
result = prepare_training_data(request)
save_preprocessing_result(result, Path("data/training.npz"))

# 2. Train model
config = TrainingConfig(max_epochs=10000, early_stopping_patience=500)
checkpoint = train_scar_model(Path("data/training.npz"), config)
save_trained_model(checkpoint, Path("models/patient_001.pth"))

# 3. Apply model
inference_result = apply_scar_model(
    model_checkpoint_path=Path("models/patient_001.pth"),
    mesh_path=Path("data/lv_mesh.vtk"),
    mc_samples=20
)
save_inference_result(
    inference_result, 
    Path("data/lv_mesh.vtk"), 
    Path("results/scar_predictions.vtk")
)
```

---

## Command Line Interface

### `cardioscar prepare`

Prepare training data by mapping 2D image intensities to 3D mesh nodes.

#### From NIfTI/NRRD Images (Recommended)

```bash
cardioscar prepare \
    --mesh-vtk MESH.vtk \
    --image IMAGE.nii.gz \
    --output TRAINING.npz \
    [--slice-axis {x,y,z}] \
    [--slice-indices "2,5,8,11"]
```

**Example:**
```bash
# Use all slices
cardioscar prepare \
    --mesh-vtk heart.vtk \
    --image lge_scan.nii.gz \
    --output training_data.npz

# Use specific slices only
cardioscar prepare \
    --mesh-vtk heart.vtk \
    --image lge_scan.nii.gz \
    --slice-indices "5,10,15,20" \
    --output training_data.npz
```

#### From VTK Grid Slices (Legacy)

```bash
cardioscar prepare \
    --mesh-vtk MESH.vtk \
    --grid-layers SLICE1.vtk [SLICE2.vtk ...] \
    --output TRAINING.npz \
    [--vtk-scalar-field FIELD_NAME]
```

**Example:**
```bash
cardioscar prepare \
    --mesh-vtk heart.vtk \
    --grid-layers slice_*.vtk \
    --vtk-scalar-field scar_probability \
    --output training_data.npz
```

**Key Options:**

| Option            | Description                          | Default    |
| ----------------- | ------------------------------------ | ---------- |
| `--mesh-vtk`      | Path to 3D target mesh (VTK)         | Required   |
| `--image`         | Path to medical image (NIfTI, NRRD)  | Required*  |
| `--grid-layers`   | Paths to VTK grid files              | Required*  |
| `--slice-axis`    | Axis to slice along (x/y/z)          | `z`        |
| `--slice-indices` | Comma-separated slice indices        | All slices |
| `--output`        | Output path for training data (.npz) | Required   |

*Must provide either `--image` OR `--grid-layers`

---

### `cardioscar train`

Train Bayesian neural network on prepared data.

```bash
cardioscar train \
    --training-data TRAINING.npz \
    --output MODEL.pth \
    [--batch-size SIZE] \
    [--max-epochs N] \
    [--early-stopping-patience N] \
    [--mc-samples N] \
    [--cpu]
```

**Example:**
```bash
# Default settings (recommended)
cardioscar train \
    --training-data training_data.npz \
    --output model.pth

# High-quality (slower)
cardioscar train \
    --training-data training_data.npz \
    --output model.pth \
    --mc-samples 5 \
    --early-stopping-patience 1000

# Fast prototyping
cardioscar train \
    --training-data training_data.npz \
    --output model.pth \
    --max-epochs 1000 \
    --early-stopping-patience 100
```

**Key Options:**

| Option                      | Description                                | Default         |
| --------------------------- | ------------------------------------------ | --------------- |
| `--training-data`           | Path to training data (.npz)               | Required        |
| `--output`                  | Output path for trained model (.pth)       | Required        |
| `--batch-size`              | Target batch size                          | 10000           |
| `--max-epochs`              | Maximum training epochs                    | 10000           |
| `--early-stopping-patience` | Epochs without improvement before stopping | 500             |
| `--mc-samples`              | MC Dropout samples during training         | 3               |
| `--cpu`                     | Force CPU usage                            | Auto-detect GPU |

---

### `cardioscar apply`

Apply trained model to predict scar probability on mesh.

```bash
cardioscar apply \
    --model MODEL.pth \
    --mesh MESH.vtk \
    --output OUTPUT.vtk \
    [--mc-samples N] \
    [--threshold VALUE] \
    [--batch-size SIZE]
```

**Example:**
```bash
# Basic inference
cardioscar apply \
    --model model.pth \
    --mesh heart.vtk \
    --output scar_predictions.vtk

# High-quality uncertainty estimation
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

**Key Options:**

| Option         | Description                                  | Default  |
| -------------- | -------------------------------------------- | -------- |
| `--model`      | Path to trained model (.pth)                 | Required |
| `--mesh`       | Path to input mesh (VTK)                     | Required |
| `--output`     | Output path for augmented mesh (.vtk)        | Required |
| `--mc-samples` | MC Dropout samples for uncertainty           | 10       |
| `--threshold`  | Optional threshold for binary classification | None     |
| `--batch-size` | Batch size for inference                     | 50000    |

**Output Fields:**

The output mesh contains three scalar fields:
- **`scar_probability`**: Mean scar probability per node [0, 1]
- **`scar_uncertainty`**: Uncertainty (standard deviation) per node
- **`scar_binary`**: Binary classification (if `--threshold` provided)

---

## Python API

For detailed API documentation, see the [API Reference](docs/API_REFERENCE.md).

### Core Functions

```python
# Data preparation
from cardioscar.logic import prepare_training_data, PreprocessingRequest

# Training
from cardioscar.logic import train_scar_model, TrainingConfig

# Inference
from cardioscar.logic import apply_scar_model

# I/O
from cardioscar.logic import (
    save_preprocessing_result,
    save_trained_model,
    save_inference_result
)
```

### Example: Batch Processing

```python
from pathlib import Path
from cardioscar.logic import apply_scar_model, save_inference_result

def process_patient_cohort(patient_ids, model_path, output_dir):
    """Process multiple patients with same trained model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for patient_id in patient_ids:
        print(f"Processing {patient_id}...")
        
        result = apply_scar_model(
            model_checkpoint_path=model_path,
            mesh_path=Path(f"data/{patient_id}/mesh.vtk"),
            mc_samples=10
        )
        
        save_inference_result(
            result,
            Path(f"data/{patient_id}/mesh.vtk"),
            output_dir / f"{patient_id}_scar.vtk"
        )
        
        print(f"  Mean scar: {result.mean_scar_probability:.3f}")
        print(f"  Uncertainty: {result.mean_uncertainty:.3f}")

# Usage
process_patient_cohort(
    patient_ids=['P001', 'P002', 'P003'],
    model_path=Path("models/scar.pth"),
    output_dir=Path("results/")
)
```

---

## Method Overview

### Problem Statement

**Input:**
- Sparse 2D LGE-CMR slices (typically 8-10mm apart) with scar intensity/segmentation
- Dense 3D left ventricular mesh (thousands of nodes, ~0.5mm resolution)

**Output:**
- Continuous scar probability at every 3D mesh node
- Per-node uncertainty estimates

### Approach

1. **Spatial Constraint Mapping**  
   Each 2D image pixel extends through slice thickness as a rectangular prism. All 3D mesh nodes within this prism form a "group".

2. **Physical Constraint**  
   The mean prediction across all nodes in a group must equal the observed 2D pixel value.

3. **Network Architecture**  
   Coordinate-based MLP: (X, Y, Z) → scar probability
   - Input: 3D coordinates (normalized to [0, 1])
   - Architecture: 4 hidden layers × 128 neurons
   - Dropout: 10% for uncertainty estimation
   - Output: Sigmoid activation → [0, 1] probability

4. **Loss Function**  
   Group-based reconstruction loss:
   ```
   L = Σ_groups (pixel_value - mean(group_predictions))²
   ```

5. **Optimization**  
   - Adam optimizer with cyclical learning rate (1e-3 to 1e-2)
   - Monte Carlo Dropout (3-5 samples during training)
   - Complete-group mini-batching (groups never split across batches)
   - Early stopping with patience

### Key Innovation: Complete-Group Batching

Traditional mini-batching would split spatial groups across batches, violating the physical constraint. Our implementation:
- Pre-sorts nodes by group ID
- Ensures batches contain only complete groups
- Batch sizes vary slightly (~±10%) to maintain exact constraints
- Enables 5× speedup over naive full-batch training while preserving accuracy

### Bayesian Uncertainty Quantification

Monte Carlo Dropout provides:
- **Epistemic uncertainty**: Model uncertainty about scar location
- **High uncertainty regions**: Areas where model is less confident (e.g., sparse data, ambiguous boundaries)
- **Quality control**: Flag regions requiring manual review or additional imaging

---

## Citation

If you use CardioScar in your research, please cite:


### Original Research

```bibtex
@article{SEN2025111219,
  title = {Weakly supervised learning for scar reconstruction in personalized cardiac models: Integrating 2D MRI to 3D anatomical models},
  journal = {Computers in Biology and Medicine},
  volume = {198},
  pages = {111219},
  year = {2025},
  issn = {0010-4825},
  doi = {https://doi.org/10.1016/j.compbiomed.2025.111219},
  url = {https://www.sciencedirect.com/science/article/pii/S0010482525015720},
  author = {Ahmet SEN and Ursula Rohrer and Pranav Bhagirath and Reza Razavi and Mark O'Neill and John Whitaker and Martin Bishop},
  keywords = {Myocardial scar segmentation, Late gadolinium-enhanced cardiac MRI, Deep learning-based interpolation, Deep learning for medical imaging, Monte Carlo Dropout}
}
```

### Software

```bibtex
@software{cardioscar2024,
  title={CardioScar: Deep Learning-Based 3D Myocardial Scar Reconstruction},
  author={Sen, Ahmet and Bishop, Martin J. and Solis-Lemus, Jose Alonso},
  year={2024},
  url={https://github.com/alonsoJASL/cardioscar},
  version={0.1.0}
}
```
---

## Acknowledgments

**Original Research:** Ahmet Sen, Martin J. Bishop (King's College London)  
**Engineering & pycemrg Integration:** Jose Alonso Solis-Lemus (Imperial College London)  
**Collaborators:** Ursula Rohrer, Pranav Bhagirath, Reza Razavi, Mark O'Neill, John Whitaker

This work builds upon:
- Original TensorFlow implementation by Ahmet Sen
- pycemrg suite for cardiac image analysis and mesh processing
- SimpleITK for medical image coordinate transforms

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Support & Contributing

- **Documentation:** [Full API Reference](docs/API_REFERENCE.md)
- **Issues:** [GitHub Issues](https://github.com/alonsoJASL/cardioscar/issues)
- **Discussions:** [GitHub Discussions](https://github.com/alonsoJASL/cardioscar/discussions)
- **Email:** j.solis-lemus [at] imperial.ac.uk

---

## Related Projects

- [pycemrg](https://github.com/OpenHeartDevelopers/pycemrg) - Core utilities for cardiac image analysis
- [pycemrg-model-creation](https://github.com/OpenHeartDevelopers/pycemrg-model-creation) - Mesh processing and UVC coordinates
- [pycemrg-image-analysis](https://github.com/OpenHeartDevelopers/pycemrg-image-analysis) - Medical image preprocessing
- [pycemrg-interpolation](https://github.com/OpenHeartDevelopers/pycemrg-interpolation) - Volumetric super-resolution

---

**Version:** 0.1.0  
**Last Updated:** February 2026s