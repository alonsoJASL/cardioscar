import logging
import argparse
from pathlib import Path

import torch 

from pycemrg.core.logs import setup_logging

from cardioscar.logic.orchestrators import prepare_training_data, save_preprocessing_result
from cardioscar.logic.contracts import PreprocessingRequest

setup_logging()
logger = logging.getLogger("PrepareTrainingData")

def main(args) : 
    # Parse slice indices if provided
    parsed_indices = None
    if args.slice_indices:
        parsed_indices = [int(i.strip()) for i in args.slice_indices.split(',')]
    
    # Build request
    request = PreprocessingRequest(
        mesh_path=args.mesh_vtk,
        # VTK input
        vtk_grid_paths=list(args.grid_layers) if args.grid_layers else None,
        vtk_scalar_field=args.vtk_scalar_field,
        # Image input
        image_path=args.image,
        slice_axis=args.slice_axis,
        slice_indices=parsed_indices,
        # Common
        slice_thickness_padding=args.slice_thickness_padding
    )
    
    result = prepare_training_data(request)
    save_preprocessing_result(result, args.output)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Standalone script to prepare data")
    parser.add_argument(
        '--mesh-vtk', 
        type=Path, 
        required=True,
        help='Path to 3D target mesh (VTK)'
    )
    
    # VTK grid input
    parser.add_argument(
        '--grid-layers',
        type=Path, 
        nargs='+',
        help='Paths to 2D grid layers (VTK). Can specify multiple times.'
    )
    parser.add_argument(
        '--vtk-scalar-field', 
        type=str, 
        default='ScalarValue',
        help='Scalar field name in VTK grids'
    )
    # Image input 
    parser.add_argument(
        '--image', 
        type=Path,
        help='Path to medical image (NIfTI, NRRD)'
    )
    parser.add_argument(
        '--slice-axis', 
        choices=['x', 'y', 'z'], 
        default='z',
        help='Axis to slice along (default: z)'
    )
    parser.add_argument(
        '--slice-indices', 
        type=str,
        help='Comma-separated slice indices (e.g., "2,5,8,11"). If not provided, uses ALL slices.'
    )
    # Common options
    parser.add_argument(
        '--output', 
        type=Path, 
        required=True,
        help='Output path for training data (.npz)'
    )
    parser.add_argument(
        '--slice-thickness-padding', 
        type=float, 
        default=5.0,
        help='Z-direction padding (mm) for slice thickness'
    )

    args = parser.parse_args()
    main(args)