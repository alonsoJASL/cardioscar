#!/usr/bin/env python3
"""
Generate VTK image planes from NIfTI for ParaView overlay.

Correctly handles oblique images by transforming the 2D slice plane
into physical space using the image's direction matrix.
"""
import logging 
import argparse
import numpy as np
import SimpleITK as sitk
import pyvista as pv
from pathlib import Path

from pycemrg.core.logs import setup_logging
setup_logging()
logger = logging.getLogger("GenerateSlicePlanes")

def extract_slice_as_structured_grid(
    image: sitk.Image,
    slice_idx: int,
    slice_axis: str = 'z'
) -> pv.StructuredGrid:
    """
    Extract a 2D slice and create a StructuredGrid positioned in physical space.
    
    Uses StructuredGrid (not ImageData) because we need to apply the full
    affine transform (origin + direction matrix) to position the oblique slice.
    """
    # Get metadata
    size = image.GetSize()  # (X, Y, Z)
    
    # Extract 2D array
    img_array = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[slice_axis]
    
    if slice_axis == 'z':
        slice_2d = img_array[slice_idx, :, :]  # (Y, X)
        grid_shape = (size[0], size[1])  # (nx, ny) for meshgrid
        # Create index arrays for this slice
        x_indices = np.arange(size[0])
        y_indices = np.arange(size[1])
        xx, yy = np.meshgrid(x_indices, y_indices, indexing='xy')
        zz = np.full_like(xx, slice_idx)
        
        # Stack as (n_points, 3) in (X, Y, Z) order
        voxel_indices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
    elif slice_axis == 'y':
        slice_2d = img_array[:, slice_idx, :]  # (Z, X)
        grid_shape = (size[0], size[2])
        x_indices = np.arange(size[0])
        z_indices = np.arange(size[2])
        xx, zz = np.meshgrid(x_indices, z_indices, indexing='xy')
        yy = np.full_like(xx, slice_idx)
        voxel_indices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
    else:  # x
        slice_2d = img_array[:, :, slice_idx]  # (Z, Y)
        grid_shape = (size[1], size[2])
        y_indices = np.arange(size[1])
        z_indices = np.arange(size[2])
        yy, zz = np.meshgrid(y_indices, z_indices, indexing='xy')
        xx = np.full_like(yy, slice_idx)
        voxel_indices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # Transform voxel indices to physical coordinates
    # For each voxel index (x, y, z), compute:
    #   physical = origin + direction @ (index * spacing)
    physical_coords = np.array([
        image.TransformIndexToPhysicalPoint(tuple(int(i) for i in idx))
        for idx in voxel_indices
    ])
    
    # Create StructuredGrid
    # Reshape coords to grid shape
    nx, ny = grid_shape
    x = physical_coords[:, 0].reshape((ny, nx))
    y = physical_coords[:, 1].reshape((ny, nx))
    z = physical_coords[:, 2].reshape((ny, nx))
    
    grid = pv.StructuredGrid(x, y, z)
    
    # Add intensity as point data
    # Transpose slice_2d to match (ny, nx) → (nx, ny) then flatten
    grid.point_data['intensity'] = slice_2d.T.ravel()
    
    return grid


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading image: {args.image}")
    image = sitk.ReadImage(str(args.image))
    
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape(3, 3)
    
    logger.info(f"  Size: {size}")
    logger.info(f"  Spacing: {spacing}")
    logger.info(f"  Origin: {origin}")
    logger.info(f"  Direction matrix:")
    logger.info(f"    {direction[0]}")
    logger.info(f"    {direction[1]}")
    logger.info(f"    {direction[2]}")
    
    # Parse slice indices
    if args.slice_indices:
        slice_indices = [int(x.strip()) for x in args.slice_indices.split(',')]
    else:
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        n_slices = size[axis_map[args.slice_axis]]
        slice_indices = list(range(0, n_slices, 1))
    
    logger.info(f"\nGenerating {len(slice_indices)} slice planes along {args.slice_axis}-axis...")
    
    for slice_idx in slice_indices:
        logger.info(f"  Slice {slice_idx}/{size[{'x':0,'y':1,'z':2}[args.slice_axis]]-1}")
        
        grid = extract_slice_as_structured_grid(
            image=image,
            slice_idx=slice_idx,
            slice_axis=args.slice_axis
        )
        
        output_path = output_dir / f"{args.slice_name_prefix}_{args.slice_axis}{slice_idx:03d}.vtk"
        grid.save(str(output_path))
        
        # Print bounds for verification
        bounds = grid.bounds
        logger.info(f"    Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] "
              f"Y[{bounds[2]:.1f}, {bounds[3]:.1f}] "
              f"Z[{bounds[4]:.1f}, {bounds[5]:.1f}]")
    
    logger.info(f"\n✓ Slice planes saved to: {output_dir}")
    logger.info("\nParaView workflow:")
    logger.info("  1. Load all slice_*.vtk + input_model.vtk")
    logger.info("  2. Color slices by 'intensity'")
    logger.info("  3. Adjust slice opacity to see mesh through them")
    logger.info("  4. Verify: Does bright ring in LGE align with LV cavity?")
    logger.info("  5. Verify: Does scar (bright in LGE) appear in LV or RV wall?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VTK slice planes from NIfTI (handles oblique images)"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice-axis", type=str, default='z',
                        choices=['x', 'y', 'z'])
    parser.add_argument("--slice-indices", type=str,
                        help="Comma-separated indices (e.g., '0,3,6,9,11')")
    parser.add_argument("--slice-name-prefix", type=str, default='slice',
                        help="Prefix for output slice files (default: 'slice')")
    
    args = parser.parse_args()
    main(args)