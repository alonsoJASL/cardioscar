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
from typing import Tuple, Dict

from pycemrg_image_analysis.utilities.io import load_image

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

def load_image_slice(image_path: Path, slice_index: int, axis: str = 'z') -> np.ndarray:
    """
    Load a single slice from a 3D medical image.
    
    Args:
        image_path: Path to NIfTI or other medical image
        slice_index: Index of slice to extract
        axis: Axis to slice along ('x', 'y', or 'z')
    
    Returns:
        (H, W) 2D array of image intensities
    
    Note:
        Currently not implemented. Add when migrating from VTK grid layers
        to direct NIfTI reading. Will use SimpleITK or pycemrg-image-analysis.
    """
    raise NotImplementedError(
        "Direct image slice loading not yet implemented. "
        "Currently using VTK grid layers as input. "
        "To add: use pycemrg_image_analysis.utilities.io.load_image()"
    )