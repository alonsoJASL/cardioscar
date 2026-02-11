# src/cardioscar/utilities/preprocessing.py

"""
Spatial Preprocessing Utilities

Pure functions for mapping 3D mesh nodes to 2D slice pixel groups.
Handles spatial intersection tests and coordinate normalization.

These are stateless utilities that operate on numpy arrays.
File I/O is handled by orchestrators.
"""
import logging
import numpy as np

from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def find_nodes_in_cell_bounds(
    mesh_coords: np.ndarray,
    cell_bounds: np.ndarray,
    z_padding: float = 5.0
) -> np.ndarray:
    """
    Find mesh nodes within a cell's spatial extent.
    
    Args:
        mesh_coords: (N, 3) XYZ coordinates of all mesh nodes
        cell_bounds: (6,) [xmin, xmax, ymin, ymax, zmin, zmax]
        z_padding: Additional padding in Z direction (mm) for slice thickness
    
    Returns:
        Boolean mask of shape (N,) indicating nodes within bounds
    
    Example:
        >>> mesh_coords = np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10]])
        >>> cell_bounds = np.array([0, 6, 0, 6, 0, 6])
        >>> mask = find_nodes_in_cell_bounds(mesh_coords, cell_bounds, z_padding=1.0)
        >>> mask
        array([ True,  True, False])
    """
    xmin, xmax, ymin, ymax, zmin, zmax = cell_bounds
    
    # Apply Z-padding to account for slice thickness
    zmin -= z_padding
    zmax += z_padding
    
    mask = (
        (mesh_coords[:, 0] >= xmin) & (mesh_coords[:, 0] <= xmax) &
        (mesh_coords[:, 1] >= ymin) & (mesh_coords[:, 1] <= ymax) &
        (mesh_coords[:, 2] >= zmin) & (mesh_coords[:, 2] <= zmax)
    )
    
    return mask


def create_group_mapping(
    mesh_coords: np.ndarray,
    grid_cells_bounds: np.ndarray,
    grid_cells_scalars: np.ndarray,
    z_padding: float = 5.0,
    layer_id: int = 0
) -> np.ndarray:
    """
    Create group assignments for mesh nodes from a single grid layer.
    
    Args:
        mesh_coords: (N, 3) mesh node coordinates
        grid_cells_bounds: (M, 6) bounding boxes for M grid cells
        grid_cells_scalars: (M,) scalar values (scar probabilities) per cell
        z_padding: Z-direction padding for slice thickness
        layer_id: Identifier for this grid layer (for unique group IDs)
    
    Returns:
        (K, 5) array where K <= N, columns are:
            [X, Y, Z, scalar_value, group_id]
        Only includes nodes that fall within at least one cell.
    
    Example:
        >>> mesh_coords = np.random.rand(100, 3) * 10
        >>> cell_bounds = np.array([[0, 5, 0, 5, 0, 5], [5, 10, 5, 10, 5, 10]])
        >>> cell_scalars = np.array([0.3, 0.7])
        >>> mapping = create_group_mapping(mesh_coords, cell_bounds, cell_scalars)
        >>> mapping.shape[1]
        5
    """
    all_mappings = []
    group_counter = 0
    
    for cell_idx in range(len(grid_cells_bounds)):
        # Find nodes within this cell
        mask = find_nodes_in_cell_bounds(
            mesh_coords,
            grid_cells_bounds[cell_idx],
            z_padding
        )
        
        matching_nodes = mesh_coords[mask]
        
        if len(matching_nodes) == 0:
            continue
        
        # Assign scalar value and group ID
        scalar_value = grid_cells_scalars[cell_idx]
        
        for node_coord in matching_nodes:
            all_mappings.append([
                node_coord[0], node_coord[1], node_coord[2],
                scalar_value,
                group_counter
            ])
        
        group_counter += 1
    
    return np.array(all_mappings) if all_mappings else np.empty((0, 5))


def remove_duplicate_nodes(data: np.ndarray) -> np.ndarray:
    """
    Remove duplicate nodes (same XYZ coordinates) keeping first occurrence.
    
    Args:
        data: (N, 5) array [X, Y, Z, scalar, group_id]
    
    Returns:
        (M, 5) array with M <= N, duplicates removed
    
    Notes:
        Duplicates can occur when a node falls within multiple overlapping
        grid cells. We keep the first assignment.
    """
    # Find unique rows based on first 4 columns (XYZ + scalar)
    unique_indices = np.unique(data[:, :4], axis=0, return_index=True)[1]
    unique_indices = np.sort(unique_indices)
    
    return data[unique_indices]


def compute_group_sizes(group_ids: np.ndarray) -> np.ndarray:
    """
    Compute the number of nodes in each unique group.
    
    Args:
        group_ids: (N,) array of group assignments
    
    Returns:
        (M,) array where M is number of unique groups
    
    Example:
        >>> group_ids = np.array([0, 0, 0, 1, 1, 2])
        >>> sizes = compute_group_sizes(group_ids)
        >>> sizes
        array([3, 2, 1])
    """
    _, counts = np.unique(group_ids, return_counts=True)
    return counts


def normalize_coordinates(
    coords: np.ndarray
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize coordinates to [0, 1] range using MinMaxScaler.
    
    Args:
        coords: (N, 3) raw coordinates
    
    Returns:
        Tuple of (normalized_coords, fitted_scaler)
        - normalized_coords: (N, 3) coordinates in [0, 1]
        - fitted_scaler: sklearn scaler (save for inference)
    
    Example:
        >>> coords = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        >>> norm_coords, scaler = normalize_coordinates(coords)
        >>> norm_coords.min(), norm_coords.max()
        (0.0, 1.0)
    """
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(coords)
    return normalized, scaler


def denormalize_coordinates(
    normalized_coords: np.ndarray,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray
) -> np.ndarray:
    """
    Denormalize coordinates from [0, 1] back to original range.
    
    Args:
        normalized_coords: (N, 3) normalized coordinates
        scaler_min: (3,) minimum values from training scaler
        scaler_max: (3,) maximum values from training scaler
    
    Returns:
        (N, 3) coordinates in original range
    
    Example:
        >>> norm_coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> scaler_min = np.array([0, 0, 0])
        >>> scaler_max = np.array([10, 10, 10])
        >>> coords = denormalize_coordinates(norm_coords, scaler_min, scaler_max)
        >>> coords
        array([[ 0., 0., 0.], [10., 10., 10.]])
    """
    # Reconstruct the scaling transformation
    scaler = MinMaxScaler()
    scaler.data_min_ = scaler_min
    scaler.data_max_ = scaler_max
    scaler.data_range_ = scaler_max - scaler_min
    scaler.scale_ = 1.0 / scaler.data_range_
    scaler.min_ = -scaler_min * scaler.scale_
    scaler.n_samples_seen_ = 1
    scaler.n_features_in_ = 3
    
    return scaler.inverse_transform(normalized_coords)

def process_vtk_grid_data(
    mesh_coords: np.ndarray,
    grid_layers_data: List[Tuple[np.ndarray, np.ndarray]],  # ✅ Data, not Paths
    z_padding: float = 5.0
) -> np.ndarray:
    """
    Process VTK grid layer data and map to mesh nodes.
    
    Args:
        mesh_coords: (N, 3) mesh node coordinates
        grid_layers_data: List of (cell_bounds, scalar_values) tuples
            - cell_bounds: (M, 6) bounding boxes per layer
            - scalar_values: (M,) scalar values per layer
        z_padding: Z-direction padding for slice thickness
    
    Returns:
        (K, 5) array [X, Y, Z, scalar_value, group_id] with unique nodes
    
    Example:
        >>> # Orchestrator loads the data first
        >>> layer_data = []
        >>> for path in grid_paths:
        ...     bounds, values = load_grid_layer_data(path)
        ...     layer_data.append((bounds, values))
        >>> 
        >>> # Then utility processes pure data
        >>> from cardioscar.utilities import process_vtk_grid_data
        >>> result = process_vtk_grid_data(mesh_coords, layer_data)
    """
    logger.info(f"Processing {len(grid_layers_data)} VTK grid layers")
    
    all_mappings = []
    group_counter = 0
    
    for layer_idx, (cell_bounds, scalar_values) in enumerate(grid_layers_data):
        logger.info(f"  Layer {layer_idx + 1}/{len(grid_layers_data)}")
        
        mapping = create_group_mapping(
            mesh_coords=mesh_coords,
            grid_cells_bounds=cell_bounds,
            grid_cells_scalars=scalar_values,
            z_padding=z_padding,
            layer_id=layer_idx
        )
        
        if len(mapping) > 0:
            # Offset group IDs to make them globally unique
            mapping[:, 4] += group_counter
            group_counter = int(mapping[:, 4].max()) + 1
            all_mappings.append(mapping)
        else:
            logger.warning(f"  No mesh nodes found in layer {layer_idx + 1}")
    
    if not all_mappings:
        raise ValueError(
            "No valid mappings found. "
            "Check that grid layers overlap with mesh coordinates."
        )
    
    combined_data = np.vstack(all_mappings)
    unique_data = remove_duplicate_nodes(combined_data)
    
    logger.info(f"Combined {len(all_mappings)} layers → {len(unique_data)} unique nodes")
    
    return unique_data


def process_image_slice_data(
    mesh_coords: np.ndarray,
    slice_layers_data: List[Tuple[np.ndarray, np.ndarray]],
    z_padding: float = 5.0
) -> np.ndarray:
    """
    Process image slice data and map to mesh nodes.
    
    Args:
        mesh_coords: (N, 3) mesh node coordinates
        slice_layers_data: List of (voxel_bounds, intensity_values) tuples
            - voxel_bounds: (M, 6) physical bounds [xmin, ymin, zmin, xmax, ymax, zmax]
            - intensity_values: (M,) intensity values per voxel
        z_padding: Padding in slice direction (mm)
    
    Returns:
        (K, 5) array [X, Y, Z, intensity, group_id] with unique nodes
    
    Example:
        >>> # Orchestrator extracts slice data from image
        >>> slice_data = extract_image_slice_data(image_path, slice_axis='z')
        >>> 
        >>> # Utility processes pure data
        >>> result = process_image_slice_data(mesh_coords, slice_data, z_padding=5.0)
    """
    logger.info(f"Processing {len(slice_layers_data)} image slices")
    
    all_mappings = []
    group_counter = 0
    
    for layer_idx, (voxel_bounds, voxel_values) in enumerate(slice_layers_data):
        logger.info(f"  Slice {layer_idx + 1}/{len(slice_layers_data)}")
        
        # voxel_bounds is already in the format we need for create_group_mapping
        # Shape: (M, 6) where each row is [xmin, ymin, zmin, xmax, ymax, zmax]
        
        mapping = create_group_mapping(
            mesh_coords=mesh_coords,
            grid_cells_bounds=voxel_bounds,  # Voxel bounds work same as VTK cell bounds
            grid_cells_scalars=voxel_values,
            z_padding=z_padding,
            layer_id=layer_idx
        )
        
        if len(mapping) > 0:
            # Offset group IDs to make them globally unique
            mapping[:, 4] += group_counter
            group_counter = int(mapping[:, 4].max()) + 1
            all_mappings.append(mapping)
        else:
            logger.warning(f"  No mesh nodes found in slice {layer_idx + 1}")
    
    if not all_mappings:
        raise ValueError(
            "No valid mappings found. "
            "Check that image slices overlap with mesh coordinates."
        )
    
    combined_data = np.vstack(all_mappings)
    unique_data = remove_duplicate_nodes(combined_data)
    
    logger.info(f"Combined {len(all_mappings)} slices → {len(unique_data)} unique nodes")
    
    return unique_data