"""
pycemrg-scar-reconstruction

Deep learning-based 3D myocardial scar reconstruction from sparse 2D LGE-CMR.

This package provides tools for patient-specific optimization of coordinate-based
neural networks that reconstruct continuous 3D scar probability fields from
sparse 2D MRI slices.

Quick Start:
    1. Prepare training data:
       python scripts/prepare_training_data.py --mesh-vtk mesh.vtk --grid-layers *.vtk
    
    2. Train model:
       python scripts/train_scar_model.py --training-data data.npz --output model.pth

Authors:
    - Jose Alonso Solis-Lemus (pycemrg integration)
    - Martin J. Bishop (original research)

License: MIT
"""

__version__ = "0.1.0"
__author__ = "Ahmet Sen, Jose Alonso Solis-Lemus, Martin J. Bishop"
__license__ = "MIT"

# Currently this package is script-based (no library API exposed)
# Future versions may expose:
# - BayesianNN model class
# - ScarReconstructionDataset
# - Group loss functions
# - Inference utilities
