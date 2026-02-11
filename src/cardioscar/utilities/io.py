# src/cardioscar/utilities/io.py

"""
I/O Utilities for Scar Reconstruction

Handles reading VTK meshes and medical images, converting to numpy arrays
for downstream processing. Thin wrappers around PyVista and SimpleITK.

Following pycemrg principles:
- Explicit I/O (no hidden file operations)
- Convert to standard numpy arrays
- Minimal dependencies on external formats
"""

import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from pycemrg_image_analysis.utilities.io import load_image
from pycemrg_image_analysis.utilities.spatial import (
    extract_slice_voxels,
    get_voxel_physical_bounds
)

# =============================================================================
# VTK MESH I/O
# =============================================================================

def load_mesh_points(mesh_path: Path) -> np.ndarray:
    """
    Load mesh node coordinates from VTK file.
    
    Args:
        mesh_path: Path to VTK mesh file
    
    Returns:
        (N, 3) array of XYZ coordinates
    
    Example:
        >>> coords = load_mesh_points(Path("mesh.vtk"))
        >>> coords.shape
        (33202, 3)
    """
    mesh = pv.read(mesh_path)
    return np.array(mesh.points)


def load_grid_layer_data(grid_path: Path, scalar_field_name: str = 'ScalarValue') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 2D grid layer with cell bounds and scalar values.
    
    Args:
        grid_path: Path to VTK grid file
        scalar_field_name: Name of scalar field containing scar probabilities
    
    Returns:
        Tuple of (cell_bounds, scalar_values)
        - cell_bounds: (M, 6) bounding boxes [xmin, xmax, ymin, ymax, zmin, zmax]
        - scalar_values: (M,) scar probability per cell
    
    Raises:
        ValueError: If scalar field not found in grid
    
    Example:
        >>> bounds, scalars = load_grid_layer_data(Path("slice.vtk"))
        >>> bounds.shape
        (751, 6)
        >>> scalars.shape
        (751,)
    """
    grid = pv.read(grid_path)
    
    # Validate scalar field exists
    if scalar_field_name not in grid.array_names:
        raise ValueError(
            f"Scalar field '{scalar_field_name}' not found in {grid_path.name}. "
            f"Available fields: {grid.array_names}"
        )
    
    scalar_values = grid[scalar_field_name]
    n_cells = grid.n_cells
    
    # Compute bounding box for each cell
    cell_bounds = []
    for cell_idx in range(n_cells):
        cell = grid.get_cell(cell_idx)
        cell_points = grid.points[cell.point_ids]
        
        bounds = np.array([
            cell_points[:, 0].min(), cell_points[:, 0].max(),
            cell_points[:, 1].min(), cell_points[:, 1].max(),
            cell_points[:, 2].min(), cell_points[:, 2].max()
        ])
        cell_bounds.append(bounds)
    
    return np.array(cell_bounds), scalar_values


def save_mesh_with_scalars(
    mesh_path: Path,
    output_path: Path,
    scalar_fields: Dict[str, np.ndarray]
) -> None:
    """
    Add scalar fields to mesh and save.
    
    Args:
        mesh_path: Path to input VTK mesh
        output_path: Path for output mesh with scalars
        scalar_fields: Dictionary mapping field names to (N,) arrays
    
    Example:
        >>> scalars = {
        ...     'scar_probability': np.random.rand(33202),
        ...     'scar_uncertainty': np.random.rand(33202)
        ... }
        >>> save_mesh_with_scalars(
        ...     Path("mesh.vtk"),
        ...     Path("mesh_with_scar.vtk"),
        ...     scalars
        ... )
    """
    mesh = pv.read(mesh_path)
    
    # Add each scalar field
    for field_name, field_data in scalar_fields.items():
        if len(field_data) != mesh.n_points:
            raise ValueError(
                f"Scalar field '{field_name}' length ({len(field_data)}) "
                f"does not match mesh points ({mesh.n_points})"
            )
        mesh[field_name] = field_data
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(output_path)


# =============================================================================
# FUTURE: MEDICAL IMAGE I/O (SimpleITK)
# =============================================================================

def extract_image_slice_data(
    image_path: Path,
    slice_axis: str = 'z',
    slice_indices: Optional[List[int]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract voxel bounds and intensities from image slices.
    
    Args:
        image_path: Path to medical image (NIfTI, NRRD)
        slice_axis: Axis to slice along ('x', 'y', 'z')
        slice_indices: List of slice indices. If None, extracts ALL slices.
    
    Returns:
        List of (voxel_bounds, voxel_intensities) tuples, one per slice:
        - voxel_bounds: (N, 6) array [xmin, ymin, zmin, xmax, ymax, zmax]
        - voxel_intensities: (N,) array of intensity values
    
    Example:
        >>> slice_data = extract_image_slice_data(
        ...     Path("lge.nii.gz"),
        ...     slice_axis='z',
        ...     slice_indices=[2, 5, 8, 11]
        ... )
        >>> len(slice_data)
        4
        >>> slice_data[0][0].shape  # First slice, bounds
        (1523, 6)
    """
    
    # Load image
    img = load_image(image_path)
    
    # Determine slice indices
    if slice_indices is None:
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[slice_axis]
        n_slices = img.GetSize()[axis_idx]
        slice_indices = list(range(n_slices))
    
    # Extract data from each slice
    slice_data = []
    for slice_idx in slice_indices:
        voxel_indices, voxel_values = extract_slice_voxels(
            image=img,
            slice_index=slice_idx,
            slice_axis=slice_axis,
            label=None
        )
        
        if len(voxel_indices) == 0:
            continue
        
        # Get physical bounds - returns [xmin, ymin, zmin, xmax, ymax, zmax]
        voxel_bounds, _ = get_voxel_physical_bounds(img, voxel_indices)
        
        # ✅ Convert to VTK format: [xmin, xmax, ymin, ymax, zmin, zmax]
        voxel_bounds_vtk = np.column_stack([
            voxel_bounds[:, 0],  # xmin
            voxel_bounds[:, 3],  # xmax
            voxel_bounds[:, 1],  # ymin
            voxel_bounds[:, 4],  # ymax
            voxel_bounds[:, 2],  # zmin
            voxel_bounds[:, 5],  # zmax
        ])
        
        slice_data.append((voxel_bounds_vtk, voxel_values))
    
    return slice_data