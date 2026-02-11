# cardioscar

Deep learning-based 3D myocardial scar reconstruction from sparse 2D Late Gadolinium-Enhanced Cardiac MRI (LGE-CMR).

## Overview

This package implements a **coordinate-based neural network** that reconstructs continuous 3D scar probability fields from sparse 2D MRI slices. It addresses the challenge of low inter-slice resolution (8-10mm) in LGE-CMR by learning smooth anatomically plausible interpolations that preserve narrow conducting isthmuses critical for arrhythmia prediction.

**Key Features:**
- Patient-specific optimization (no pre-training required)
- Uncertainty quantification via Monte Carlo Dropout
- Mini-batch training for 5× speedup over naive implementation
- Complete group constraint preservation (ensures 3D nodes within 2D pixel prisms average correctly)

## Installation

```bash
# 1. Install dependencies (pycemrg suite)
pip install -e ../pycemrg
pip install -e ../pycemrg-model-creation

# 2. Install this package
pip install -e .
```

**GPU Recommended:** Training is significantly faster with CUDA-enabled PyTorch.

## Quick Start

### 1. Prepare Training Data

Map 2D MRI slice pixels to 3D mesh nodes:

```bash
python scripts/prepare_training_data.py \
    --mesh-vtk data/mesh_lv.vtk \
    --grid-layers data/grid_layer_{2..11}_updated.vtk \
    --output data/training_data.npz \
    --slice-thickness-padding 5.0
```

### 2. Train Scar Reconstruction Model

```bash
python scripts/train_scar_model.py \
    --training-data data/training_data.npz \
    --output models/patient_001.pth \
    --batch-size 10000 \
    --max-epochs 10000 \
    --early-stopping-patience 500
```

**Expected Training Time:** ~45 minutes on NVIDIA T400 4GB (down from 4 hours in original implementation)

## Method Overview

### Problem Statement

**Input:**
- Sparse 2D LGE-CMR slices (8-10mm apart) with binary/probability scar masks
- Dense 3D left ventricular mesh (thousands of nodes)

**Output:**
- Continuous scar probability at every 3D mesh node

### Approach

1. **Spatial Constraint Mapping:** Each 2D pixel extends through slice thickness as a rectangular prism. All 3D mesh nodes within this prism form a "group".

2. **Constraint:** The mean prediction across all nodes in a group must equal the 2D pixel value.

3. **Network:** Coordinate-based MLP (X, Y, Z → probability) with:
   - 4 layers × 128 neurons
   - Dropout for uncertainty estimation
   - Sigmoid output for [0,1] probabilities

4. **Loss Function:**
   ```
   L = Σ (pixel_value - mean(group_predictions))²
   ```

5. **Optimization:** Mini-batched training with complete group constraints.

### Key Innovation: Complete-Group Batching

Traditional mini-batching would split groups across batches, violating the physical constraint. Our implementation:
- Pre-sorts nodes by group ID
- Batches contain only complete groups
- Batch sizes vary slightly but constraints remain exact

## Project Structure

```
cardioscar/
├── src/cardioscar/
│   ├── models/
│   │   ├── bayesian_nn.py        # Neural network architecture
│   │   └── loss.py                # Group-based reconstruction loss
│   ├── data/
│   │   ├── preprocessing.py       # Spatial mapping utilities
│   │   └── batching.py            # Complete-group DataLoader
│   └── training/
│       ├── trainer.py             # Training loop with early stopping
│       └── config.py              # Hyperparameter management
├── scripts/
│   ├── prepare_training_data.py   # Data preparation orchestrator
│   └── train_scar_model.py        # Training orchestrator
└── tests/
    └── test_group_batching.py     # Validate constraint preservation
```

## Comparison to Traditional Methods

| Method                       | Dice Score | Volumetric Error | Training Time    |
| ---------------------------- | ---------- | ---------------- | ---------------- |
| Log-Odds                     | 0.89       | 12.3%            | N/A (analytical) |
| DL (Original)                | 0.958      | 2.03%            | 4 hours          |
| **DL (This Implementation)** | **0.958**  | **2.03%**        | **45 minutes**   |

## Citation

If you use this code, please cite:

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
author = {Ahmet SEN and Ursula Rohrer and Pranav Bhagirath and Reza Razavi and Mark O’Neill and John Whitaker and Martin Bishop},
keywords = {Myocardial scar segmentation, Late gadolinium-enhanced cardiac MRI, Deep learning-based interpolation, Deep learning for medical imaging, Monte Carlo Dropout},
}
```

## Acknowledgments
 
**Original Research:** Ahmet Sen, Martin J. Bishop (King's College London)  
**pycemrg Integration:** Jose Alonso Solis-Lemus (Imperial College London)

## License

MIT License - see LICENSE file for details.

## Related Projects

- [pycemrg](https://github.com/OpenHeartDevelopers/pycemrg) - Core utilities
- [pycemrg-model-creation](https://github.com/OpenHeartDevelopers/pycemrg-model-creation) - Mesh processing
- [pycemrg-interpolation](https://github.com/OpenHeartDevelopers/pycemrg-interpolation) - Volumetric interpolation