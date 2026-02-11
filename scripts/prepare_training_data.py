#!/usr/bin/env python3
"""
Spatial Mapping Preparation for Scar Reconstruction

This script maps 2D LGE-CMR grid pixels to 3D mesh nodes by determining
which nodes fall within each pixel's spatial extent (accounting for slice thickness).

Orchestrator Responsibilities:
- File I/O (load VTK grids and mesh)
- Path management
- Logging configuration
- Data persistence (.npz output)

Mathematical Logic (delegated to utilities):
- Spatial intersection tests
- Group assignment
- Coordinate normalization

Usage:
    python scripts/prepare_training_data.py \\
        --mesh-vtk data/rotated_2.vtk \\
        --grid-layers data/grid_layer_{2..11}_updated.vtk \\
        --output data/training_data.npz \\
        --slice-thickness-padding 5.0

Output Format (.npz):
    - 'coordinates': (N, 3) - XYZ mesh node coordinates
    - 'intensities': (N, 1) - Target scar probabilities from 2D slices
    - 'group_ids': (N, 2) - [slice_id, pixel_id] for constraint grouping
    - 'group_sizes': (M,) - Number of nodes in each unique group
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler

from pycemrg.core import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger("PrepareTrainingData")


def load_mesh_coordinates(mesh_path: Path) -> np.ndarray:
    """
    Load 3D mesh node coordinates.
    
    Args:
        mesh_path: Path to VTK mesh file
        
    Returns:
        (N, 3) array of XYZ coordinates
    """
    logger.info(f"Loading target mesh: {mesh_path}")
    mesh = pv.read(mesh_path)
    coords = np.array(mesh.points)
    logger.info(f"  Loaded {len(coords)} mesh nodes")
    return coords


def load_grid_layer(grid_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single 2D grid layer with spatial bounds and scalar values.
    
    Args:
        grid_path: Path to VTK grid file
        
    Returns:
        Tuple of (cell_coords, cell_bounds, scalar_values)
        - cell_coords: (n_cells, 3) - centroid of each cell
        - cell_bounds: (n_cells, 6) - [xmin, xmax, ymin, ymax, zmin, zmax]
        - scalar_values: (n_cells,) - scar probability per cell
    """
    grid = pv.read(grid_path)
    
    # Extract scalar values (assumes 'ScalarValue' field)
    if 'ScalarValue' not in grid.array_names:
        raise ValueError(f"Grid {grid_path.name} missing 'ScalarValue' field")
    
    scalar_values = grid['ScalarValue']
    
    # Compute cell bounds
    n_cells = grid.n_cells
    cell_coords = []
    cell_bounds = []
    
    for cell_idx in range(n_cells):
        cell = grid.get_cell(cell_idx)
        cell_point_ids = cell.point_ids
        
        # Get coordinates of cell vertices
        cell_points = grid.points[cell_point_ids]
        
        # Bounding box
        x_coords = cell_points[:, 0]
        y_coords = cell_points[:, 1]
        z_coords = cell_points[:, 2]
        
        bounds = np.array([
            x_coords.min(), x_coords.max(),
            y_coords.min(), y_coords.max(),
            z_coords.min(), z_coords.max()
        ])
        
        cell_bounds.append(bounds)
        cell_coords.append(cell_points.mean(axis=0))
    
    return (
        np.array(cell_coords),
        np.array(cell_bounds),
        scalar_values
    )


def find_nodes_in_cell(
    mesh_coords: np.ndarray,
    cell_bounds: np.ndarray,
    z_padding: float = 5.0
) -> np.ndarray:
    """
    Find mesh nodes within a cell's spatial extent.
    
    Args:
        mesh_coords: (N, 3) XYZ coordinates
        cell_bounds: (6,) [xmin, xmax, ymin, ymax, zmin, zmax]
        z_padding: Additional padding in Z direction (slice thickness buffer)
        
    Returns:
        Boolean mask of shape (N,) indicating nodes within bounds
    """
    xmin, xmax, ymin, ymax, zmin, zmax = cell_bounds
    
    # Apply Z-padding
    zmin -= z_padding
    zmax += z_padding
    
    mask = (
        (mesh_coords[:, 0] >= xmin) & (mesh_coords[:, 0] <= xmax) &
        (mesh_coords[:, 1] >= ymin) & (mesh_coords[:, 1] <= ymax) &
        (mesh_coords[:, 2] >= zmin) & (mesh_coords[:, 2] <= zmax)
    )
    
    return mask


def create_training_data(
    mesh_coords: np.ndarray,
    grid_layers: List[Path],
    z_padding: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset by mapping 2D grid cells to 3D mesh nodes.
    
    Args:
        mesh_coords: (N, 3) target mesh coordinates
        grid_layers: List of paths to grid layer VTK files
        z_padding: Z-direction padding for slice thickness
        
    Returns:
        Tuple of:
        - coordinates: (M, 3) - filtered mesh coords with group assignments
        - intensities: (M, 1) - target scar values
        - group_ids: (M, 2) - [layer_idx, cell_idx]
        - group_sizes: (K,) - nodes per unique group
    """
    all_data = []
    group_counter = 0
    
    for layer_idx, grid_path in enumerate(grid_layers):
        logger.info(f"Processing layer {layer_idx + 1}/{len(grid_layers)}: {grid_path.name}")
        
        cell_coords, cell_bounds, scalar_values = load_grid_layer(grid_path)
        
        for cell_idx in range(len(cell_bounds)):
            # Find nodes within this cell
            mask = find_nodes_in_cell(mesh_coords, cell_bounds[cell_idx], z_padding)
            matching_nodes = mesh_coords[mask]
            
            if len(matching_nodes) == 0:
                continue
            
            # Assign group ID and target intensity
            intensity = scalar_values[cell_idx]
            
            for node_coord in matching_nodes:
                all_data.append([
                    node_coord[0], node_coord[1], node_coord[2],  # XYZ
                    intensity,                                      # Target
                    group_counter                                   # Group ID
                ])
            
            group_counter += 1
    
    # Convert to arrays
    all_data = np.array(all_data)
    
    # Remove duplicate nodes (same XYZ but multiple group assignments)
    # Keep first occurrence
    unique_indices = np.unique(all_data[:, :4], axis=0, return_index=True)[1]
    unique_indices = np.sort(unique_indices)
    all_data = all_data[unique_indices]
    
    coordinates = all_data[:, 0:3]
    intensities = all_data[:, 3:4]
    group_ids = all_data[:, 4].astype(int)
    
    # Compute group sizes
    unique_groups, group_sizes = np.unique(group_ids, return_counts=True)
    
    logger.info(f"Dataset created:")
    logger.info(f"  Total nodes: {len(coordinates)}")
    logger.info(f"  Unique groups: {len(unique_groups)}")
    logger.info(f"  Avg nodes/group: {group_sizes.mean():.1f}")
    logger.info(f"  Min nodes/group: {group_sizes.min()}")
    logger.info(f"  Max nodes/group: {group_sizes.max()}")
    
    return coordinates, intensities, group_ids, group_sizes


def normalize_coordinates(coords: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize coordinates to [0, 1] range.
    
    Args:
        coords: (N, 3) raw coordinates
        
    Returns:
        Tuple of (normalized_coords, scaler)
    """
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(coords)
    return normalized, scaler


def main(args: argparse.Namespace) -> None:
    """Main orchestration."""
    
    # 1. Load mesh
    mesh_coords = load_mesh_coordinates(args.mesh_vtk)
    
    # 2. Parse grid layer paths
    grid_layers = sorted(args.grid_layers)
    logger.info(f"Found {len(grid_layers)} grid layers")
    
    # 3. Create spatial mapping
    coordinates, intensities, group_ids, group_sizes = create_training_data(
        mesh_coords,
        grid_layers,
        z_padding=args.slice_thickness_padding
    )
    
    # 4. Normalize coordinates
    normalized_coords, scaler = normalize_coordinates(coordinates)
    
    # 5. Save training data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        args.output,
        coordinates=normalized_coords.astype(np.float32),
        intensities=intensities.astype(np.float32),
        group_ids=group_ids.astype(np.int32),
        group_sizes=group_sizes.astype(np.int32),
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_
    )
    
    logger.info(f"Training data saved to: {args.output}")
    logger.info(f"File size: {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training data for scar reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mesh-vtk",
        type=Path,
        required=True,
        help="Path to 3D target mesh (VTK format)"
    )
    
    parser.add_argument(
        "--grid-layers",
        type=Path,
        nargs='+',
        required=True,
        help="Paths to 2D grid layer files (VTK format)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data.npz"),
        help="Output path for training data"
    )
    
    parser.add_argument(
        "--slice-thickness-padding",
        type=float,
        default=5.0,
        help="Z-direction padding (mm) to account for slice thickness"
    )
    
    args = parser.parse_args()
    main(args)
