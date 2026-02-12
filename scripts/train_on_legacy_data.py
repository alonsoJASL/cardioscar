# scripts/train_on_legacy_data.py
"""
Train CardioScar model using legacy Intersect_data.npy
for direct comparison.
"""

import logging
import argparse

import numpy as np

from pathlib import Path

from pycemrg.core.logs import setup_logging

from cardioscar.logic.orchestrators import save_preprocessing_result
from cardioscar.logic.contracts import PreprocessingResult
from cardioscar.utilities.preprocessing import (
    normalize_coordinates,
    compute_group_sizes,
)


def main(args):
    legacy_data_file = args.legacy_coordinates_file
    output_dir = legacy_data_file.parent
    # Load legacy data
    data = np.load(legacy_data_file)

    # Extract components
    coordinates = data[:, 0:3]
    intensities = data[:, 3:4]
    group_ids = data[:, 4].astype(np.int32)

    # Normalize (legacy does this at runtime)
    normalized_coords, scaler = normalize_coordinates(coordinates)
    group_sizes = compute_group_sizes(group_ids)

    # Create PreprocessingResult
    result = PreprocessingResult(
        coordinates=normalized_coords.astype(np.float32),
        intensities=intensities.astype(np.float32),
        group_ids=group_ids,
        group_sizes=group_sizes.astype(np.int32),
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_,
        n_nodes=len(coordinates),
        n_groups=len(group_sizes),
    )

    # Save as CardioScar format
    save_preprocessing_result(result, output_dir / "legacy_data.npz")
    print(f"Converted legacy data: {result.n_nodes} nodes, {result.n_groups} groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "legacy_coordinates_file",
        type=Path,
    )

    args = parser.parse_args()
    main(args)
