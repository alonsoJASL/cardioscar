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
    z_thickness: float = 8.0
) -> None:
    """
    Convert VTK grid slice to NIfTI image.
    
    Args:
        vtk_path: Path to VTK grid file
        output_path: Path for output NIfTI (.nii.gz)
        scalar_field_name: Name of scalar field to extract
        z_thickness: Slice thickness in mm (for spacing[2])
    """
    # 1. Load VTK grid
    logger.info(f"Loading VTK grid: {vtk_path}")
    grid = pv.read(vtk_path)
    
    # 2. Get scalar values
    if scalar_field_name not in grid.array_names:
        raise ValueError(
            f"Scalar field '{scalar_field_name}' not found. "
            f"Available: {grid.array_names}"
        )
    
    scalar_values = grid[scalar_field_name]
    logger.info(f"  Cells: {grid.n_cells}")
    logger.info(f"  Scalar range: [{scalar_values.min():.3f}, {scalar_values.max():.3f}]")
    
    # 3. Get VTK grid's actual bounds
    bounds = grid.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    logger.info(f"  VTK bounds: {bounds}")
    
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    z_min, z_max = bounds[4], bounds[5]
    
    # 4. Extract cell centers and determine spacing
    cell_centers = []
    for cell_idx in range(grid.n_cells):
        cell = grid.get_cell(cell_idx)
        cell_points = grid.points[cell.point_ids]
        center = cell_points.mean(axis=0)
        cell_centers.append(center)
    
    cell_centers = np.array(cell_centers)  # (N, 3)
    
    # Estimate in-plane spacing from cell distances
    x_coords = cell_centers[:, 0]
    y_coords = cell_centers[:, 1]
    
    unique_x = np.unique(np.round(x_coords, decimals=2))
    unique_y = np.unique(np.round(y_coords, decimals=2))
    
    if len(unique_x) > 1:
        spacing_x = np.median(np.diff(unique_x))
    else:
        spacing_x = x_max - x_min  # Single column
    
    if len(unique_y) > 1:
        spacing_y = np.median(np.diff(unique_y))
    else:
        spacing_y = y_max - y_min  # Single row
    
    logger.info(f"\nDetected in-plane spacing:")
    logger.info(f"  X: {spacing_x:.3f} mm")
    logger.info(f"  Y: {spacing_y:.3f} mm")
    logger.info(f"  Z: {z_thickness:.3f} mm (specified)")
    
    # 5. Build a regular grid from cell centers
    x_centers = np.unique(np.round(cell_centers[:, 0], decimals=2))
    y_centers = np.unique(np.round(cell_centers[:, 1], decimals=2))

    x_centers_sorted = np.sort(x_centers)
    y_centers_sorted = np.sort(y_centers)

    nx = len(x_centers_sorted)
    ny = len(y_centers_sorted)
    nz = 1

    logger.info(f"\nImage dimensions: ({nx}, {ny}, {nz})")

    # Spacing from actual center separations
    if nx > 1:
        spacing_x = np.median(np.diff(x_centers_sorted))
    else:
        spacing_x = 1.0

    if ny > 1:
        spacing_y = np.median(np.diff(y_centers_sorted))
    else:
        spacing_y = 1.0

    logger.info(f"\nSpacing:")
    logger.info(f"  X: {spacing_x:.3f} mm")
    logger.info(f"  Y: {spacing_y:.3f} mm")
    logger.info(f"  Z: {z_thickness:.3f} mm")

    # Origin: first voxel center should be at min cell center
    # origin + 0.5 * spacing = first_center
    origin_x = x_centers_sorted[0] - 0.5 * spacing_x
    origin_y = y_centers_sorted[0] - 0.5 * spacing_y
    origin_z = cell_centers[:, 2].mean() - 0.5 * z_thickness

    logger.info(f"\nImage origin: ({origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f})")

    # 6. Create image array
    image_array = np.zeros((nz, ny, nx), dtype=np.float32)

    # 7. Map cells to voxels using lookup
    x_to_idx = {round(x, 2): i for i, x in enumerate(x_centers_sorted)}
    y_to_idx = {round(y, 2): j for j, y in enumerate(y_centers_sorted)}

    mapped_count = 0
    for cell_idx, (center, value) in enumerate(zip(cell_centers, scalar_values)):
        x_rounded = round(center[0], 2)
        y_rounded = round(center[1], 2)

        if x_rounded in x_to_idx and y_rounded in y_to_idx:
            i = x_to_idx[x_rounded]
            j = y_to_idx[y_rounded]
            k = 0

            image_array[k, j, i] = value
            mapped_count += 1

    logger.info(f"\nMapped {mapped_count}/{grid.n_cells} cells to voxels")
    
    # 8. Create SimpleITK image
    img = sitk.GetImageFromArray(image_array)
    img.SetSpacing((spacing_x, spacing_y, z_thickness))
    img.SetOrigin((origin_x, origin_y, origin_z))
    
    # 9. Verify voxel centers align with VTK cell centers
    logger.info("\nVerifying alignment (first 5 cells):")
    for cell_idx in range(min(5, grid.n_cells)):
        vtk_center = cell_centers[cell_idx]
        
        # Use the SAME lookup method as mapping
        x_rounded = round(vtk_center[0], 2)
        y_rounded = round(vtk_center[1], 2)
        
        if x_rounded in x_to_idx and y_rounded in y_to_idx:
            i = x_to_idx[x_rounded]
            j = y_to_idx[y_rounded]
            k = 0
            
            # Compute voxel center in physical space
            voxel_center = img.TransformIndexToPhysicalPoint((i, j, k))
            
            diff = np.array(vtk_center) - np.array(voxel_center)
            logger.info(f"  Cell {cell_idx}: VTK={vtk_center[:2]}, Voxel={voxel_center[:2]}, Diff={diff[:2]}")
        else:
            logger.warning(f"  Cell {cell_idx}: VTK center {vtk_center[:2]} not in lookup!")
        
    # 10. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
    
    logger.info(f"\n✓ Saved image to: {output_path}")
    logger.info(f"  Origin: {img.GetOrigin()}")
    logger.info(f"  Spacing: {img.GetSpacing()}")
    logger.info(f"  Size: {img.GetSize()}")
    logger.info(f"  Bounds: [{origin_x:.1f}, {origin_x + nx*spacing_x:.1f}, "
          f"{origin_y:.1f}, {origin_y + ny*spacing_y:.1f}, "
          f"{origin_z:.1f}, {origin_z + z_thickness:.1f}]")
    logger.info(f"  Value range: [{image_array.min():.3f}, {image_array.max():.3f}]")


if __name__ == "__main__":
    import argparse
    
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
        "--z-thickness",
        type=float,
        default=8.0,
        help="Slice thickness in mm (default: 8.0)"
    )
    
    args = parser.parse_args()
    
    vtk_slice_to_image(
        vtk_path=args.vtk_slice,
        output_path=args.output_nifti,
        scalar_field_name=args.scalar_field,
        z_thickness=args.z_thickness
    )
