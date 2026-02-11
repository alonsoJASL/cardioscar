# src/cardioscar/utilities/__init__.py

"""
Utilities Public API

Exposes preprocessing, batching, and I/O utilities.
"""

from cardioscar.utilities.io import (
    load_mesh_points,
    load_grid_layer_data,
    save_mesh_with_scalars
)

from cardioscar.utilities.preprocessing import (
    find_nodes_in_cell_bounds,
    create_group_mapping,
    remove_duplicate_nodes,
    compute_group_sizes,
    normalize_coordinates,
    denormalize_coordinates
)

from cardioscar.utilities.batching import (
    ScarReconstructionDataset,
    create_complete_group_batches,
    get_batch_statistics
)

__all__ = [
    # I/O
    "load_mesh_points",
    "load_grid_layer_data",
    "save_mesh_with_scalars",
    # Preprocessing
    "find_nodes_in_cell_bounds",
    "create_group_mapping",
    "remove_duplicate_nodes",
    "compute_group_sizes",
    "normalize_coordinates",
    "denormalize_coordinates",
    # Batching
    "ScarReconstructionDataset",
    "create_complete_group_batches",
    "get_batch_statistics",
]