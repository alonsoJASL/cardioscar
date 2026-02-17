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

import logging
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from pathlib import Path
from typing import Tuple, Dict, List, Optional

from pycemrg_image_analysis.utilities.io import load_image
from pycemrg_image_analysis.utilities.spatial import sample_image_at_points

logger = logging.getLogger(__name__)
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
    mesh_coords: np.ndarray,
    precise: bool = True,
    slice_thickness_padding: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample image intensities at mesh node locations.

    Replaces the previous bounding-box-based approach, which produced
    incorrect results for oblique images (bounding boxes spanning the
    entire physical volume). Delegates all coordinate transforms to
    SimpleITK via sample_image_at_points(), which handles arbitrary
    image orientations correctly.

    Nodes outside the image volume are silently excluded. Groups are
    defined by shared voxel: all mesh nodes that fall inside the same
    image voxel receive the same group ID. This preserves the
    group-based loss structure used by the training pipeline.

    Args:
        image_path: Path to the NIfTI image (LGE scan or segmentation).
        mesh_coords: (N, 3) array of mesh node physical coordinates in
            (X, Y, Z) convention, as returned by pyvista mesh.points.
        precise: If True (default), uses per-point
            TransformPhysicalPointToContinuousIndex. Recommended for
            oblique images where the vectorized affine inverse may
            disagree at voxel boundaries. If False, uses vectorized
            affine inverse (~1000x faster, suitable for axis-aligned
            images).
        slice_thickness_padding: Deprecated. Has no effect. Retained
            for backwards compatibility with existing CLI calls. Will
            be removed in a future release.

    Returns:
        Tuple of three aligned (M,) arrays:
        - node_indices: (M,) integer indices into mesh_coords
          identifying which input nodes were successfully sampled.
        - group_ids: (M,) integer group ID per sampled node. Nodes
          sharing the same voxel receive the same group ID.
        - intensities: (M,) float32 intensity values normalised to
          [0, 1] from the raw image range.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If mesh_coords is not shape (N, 3).

    Example:
        >>> mesh = pv.read("mesh.vtk")
        >>> mesh_coords = np.array(mesh.points)  # (N, 3) XYZ
        >>> node_indices, group_ids, intensities = extract_image_slice_data(
        ...     image_path=Path("lge_scan.nii.gz"),
        ...     mesh_coords=mesh_coords,
        ... )
        >>> # Map back to full mesh for visualisation
        >>> field = np.zeros(len(mesh_coords))
        >>> field[node_indices] = intensities
    """
    if slice_thickness_padding is not None:
        import warnings
        warnings.warn(
            "slice_thickness_padding is deprecated and has no effect. "
            "Physical slice coverage is now handled implicitly by "
            "sample_image_at_points(). Remove this argument.",
            DeprecationWarning,
            stacklevel=2,
        )

    if mesh_coords.ndim != 2 or mesh_coords.shape[1] != 3:
        raise ValueError(
            f"mesh_coords must have shape (N, 3), got {mesh_coords.shape}"
        )

    logger.info(f"Loading image: {image_path}")
    image = load_image(image_path)
    logger.info(
        f"  Size: {image.GetSize()}, "
        f"Spacing: {[f'{s:.2f}' for s in image.GetSpacing()]} mm"
    )

    # Delegate all oblique/coordinate math to pycemrg-image-analysis.
    # precise=True is the safe default for LGE images, which are
    # typically heavily oblique (~45 degrees).
    sampled_indices, raw_values = sample_image_at_points(
        image=image,
        physical_points=mesh_coords,
        precise=precise,
    )

    if len(sampled_indices) == 0:
        logger.warning(
            "No mesh nodes fall inside the image volume. "
            "Verify the mesh is registered to the image space."
        )
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, empty_int, np.empty(0, dtype=np.float32)

    logger.info(
        f"  Sampled {len(sampled_indices):,} / {len(mesh_coords):,} nodes"
    )

    # Normalize to [0, 1] using global range of sampled values.
    global_min = float(raw_values.min())
    global_max = float(raw_values.max())

    if global_max > global_min:
        intensities = (raw_values - global_min) / (global_max - global_min)
        logger.info(
            f"  Intensity range: [{global_min:.1f}, {global_max:.1f}] → [0, 1]"
        )
    else:
        intensities = np.zeros_like(raw_values)
        logger.warning("Flat image detected - all intensities set to 0.")

    # Derive group IDs: nodes mapping to the same voxel share a group.
    # Flat voxel index: z * (sY * sX) + y * sX + x
    size_x, size_y, _ = image.GetSize()
    voxel_indices = _physical_to_voxel_indices(
        image=image,
        physical_points=mesh_coords[sampled_indices],
        precise=precise,
    )
    group_ids = (
        voxel_indices[:, 2] * (size_y * size_x)
        + voxel_indices[:, 1] * size_x
        + voxel_indices[:, 0]
    )

    logger.info(f"  Unique groups (voxels): {len(np.unique(group_ids)):,}")

    return (
        sampled_indices.astype(np.int64),
        group_ids.astype(np.int64),
        intensities.astype(np.float32),
    )

def _physical_to_voxel_indices(
    image: sitk.Image,
    physical_points: np.ndarray,
    precise: bool,
) -> np.ndarray:
    """
    Convert physical coordinates to voxel indices.

    All input points are assumed to be inside the image volume.
    Matches the transform mode used by sample_image_at_points so
    group boundaries are consistent with sampled values.

    Args:
        image: SimpleITK image.
        physical_points: (M, 3) physical coordinates in (X, Y, Z).
        precise: If True, uses per-point TransformPhysicalPointToIndex.
                 If False, uses vectorized affine inverse.

    Returns:
        (M, 3) integer voxel indices in (X, Y, Z) order.
    """
    if precise:
        return np.array([
            image.TransformPhysicalPointToIndex(pt.tolist())
            for pt in physical_points
        ], dtype=np.int64)

    # Vectorized affine inverse - matches _sample_fast in pycemrg
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    direction_inv = np.linalg.inv(direction)
    shifted = physical_points - origin
    indices_float = (direction_inv @ shifted.T).T / spacing
    
    return np.floor(indices_float).astype(np.int64)









































