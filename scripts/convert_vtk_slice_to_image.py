# scripts/convert_vtk_slice_to_image.py

"""
Convert VTK grid slice to NIfTI image for testing.

Creates a "flat" 3D image (1 voxel thick in Z direction) that matches
the VTK slice geometry exactly.
"""
import logging
import argparse

import numpy as np
import pyvista as pv
import SimpleITK as sitk
from pathlib import Path

from pycemrg.core.logs import setup_logging

setup_logging()
logger = logging.getLogger('ConvertVtkSlice')

def vtk_slice_to_image(
    vtk_path: Path,
    output_path: Path,
    scalar_field_name: str = "ScalarValue",
    z_position: float = None
) -> None:
    """
    Convert VTK grid slice to NIfTI image.
    
    Args:
        vtk_path: Path to VTK grid file
        output_path: Path for output NIfTI (.nii.gz)
        scalar_field_name: Name of scalar field to extract
        z_position: Z-coordinate for the slice (auto-detected if None)
    """
    # 1. Load VTK grid
    grid = pv.read(vtk_path)
    
    # 2. Get scalar values
    if scalar_field_name not in grid.array_names:
        raise ValueError(
            f"Scalar field '{scalar_field_name}' not found. "
            f"Available: {grid.array_names}"
        )
    
    scalar_values = grid[scalar_field_name]
    
    # 3. Extract cell centers (2D grid positions)
    cell_centers = []
    for cell_idx in range(grid.n_cells):
        cell = grid.get_cell(cell_idx)
        cell_points = grid.points[cell.point_ids]
        center = cell_points.mean(axis=0)
        cell_centers.append(center)
    
    cell_centers = np.array(cell_centers)  # (N, 3)
    
    # 4. Determine image geometry
    # Find bounding box
    x_coords = cell_centers[:, 0]
    y_coords = cell_centers[:, 1]
    z_coords = cell_centers[:, 2]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    if z_position is None:
        z_position = z_coords.mean()  # Use mean Z
    
    # Estimate spacing from cell distances
    unique_x = np.unique(np.round(x_coords, decimals=3))
    unique_y = np.unique(np.round(y_coords, decimals=3))
    
    if len(unique_x) > 1:
        spacing_x = np.diff(unique_x).min()
    else:
        spacing_x = 1.0
    
    if len(unique_y) > 1:
        spacing_y = np.diff(unique_y).min()
    else:
        spacing_y = 1.0
    
    spacing_z = 8.0  # Typical LGE slice thickness
    
    print(f"Detected spacing: ({spacing_x:.2f}, {spacing_y:.2f}, {spacing_z:.2f}) mm")
    
    # 5. Create image array
    # Calculate image dimensions
    nx = int(np.ceil((x_max - x_min) / spacing_x)) + 1
    ny = int(np.ceil((y_max - y_min) / spacing_y)) + 1
    nz = 1  # Single slice
    
    print(f"Image size: ({nx}, {ny}, {nz})")
    
    # Initialize array (Z, Y, X convention)
    image_array = np.zeros((nz, ny, nx), dtype=np.float32)
    
    # 6. Map VTK cells to image voxels
    for cell_idx, (center, value) in enumerate(zip(cell_centers, scalar_values)):
        # Convert physical coordinates to voxel indices
        i = int(np.round((center[0] - x_min) / spacing_x))
        j = int(np.round((center[1] - y_min) / spacing_y))
        k = 0  # Single Z slice
        
        # Bounds check
        if 0 <= i < nx and 0 <= j < ny:
            image_array[k, j, i] = value
    
    print(f"Filled {np.sum(image_array > 0)} voxels")
    
    # 7. Create SimpleITK image
    img = sitk.GetImageFromArray(image_array)
    img.SetSpacing((spacing_x, spacing_y, spacing_z))
    img.SetOrigin((x_min, y_min, z_position - spacing_z/2))  # Center the slice
    
    # 8. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
    
    print(f"Saved image to: {output_path}")
    print(f"  Origin: {img.GetOrigin()}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Size: {img.GetSize()}")
    print(f"  Value range: [{image_array.min():.3f}, {image_array.max():.3f}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert VTK grid slice to NIfTI image"
    )
    parser.add_argument("vtk_slice", type=Path, help="Input VTK grid file")
    parser.add_argument("output_nifti", type=Path, help="Output NIfTI file (.nii.gz)")
    parser.add_argument(
        "--scalar-field",
        type=str,
        default="ScalarValue",
        help="Name of scalar field to extract"
    )
    parser.add_argument(
        "--z-position",
        type=float,
        default=None,
        help="Z-coordinate for the slice (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    vtk_slice_to_image(
        vtk_path=args.vtk_slice,
        output_path=args.output_nifti,
        scalar_field_name=args.scalar_field,
        z_position=args.z_position
    )