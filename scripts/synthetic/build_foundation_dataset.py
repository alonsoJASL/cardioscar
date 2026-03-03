"""
build_foundation_dataset.py

Scans a directory of cardioscar `.npz` training files produced by `cardioscar prepare`,
groups them by subset number, assigns subsets to train/finetune/test splits, and
concatenates the training split into a single `foundation_training.npz` with correct
group ID offsetting.

Fine-tune and test files are listed in the manifest but NOT concatenated.

Usage
-----
python scripts/build_foundation_dataset.py \\
    --input-dir /path/to/prepare_outputs \\
    --output-dir /path/to/foundation \\
    --train-subsets 11 \\
    --finetune-subsets 2 \\
    --test-subsets 2 \\
    --expected-subsets 180 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename pattern
# Matches files like:
#   subset_013_n10_f2_s3.0_xy2p0mm_z8p0mm.nii.gz_training.npz
#   subset_013_n10_f2_s3.0_xy2p0mm_z8p0mm_training.npz
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(
    r"^subset_(?P<subset>\d+)"
    r"_n(?P<neighbours>\d+)"
    r"_f(?P<falloff>\d+)"
    r"_s(?P<sigma>[\d.]+)"
    r"_xy[\dp]+mm"
    r"_z[\dp]+mm"
    r"(?:\.nii\.gz)?"
    r"_training\.npz$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def scan_files(input_dir: Path) -> Dict[int, List[Path]]:
    """
    Scan *input_dir* for matching `.npz` files.

    Returns
    -------
    dict mapping subset_number (int) -> sorted list of matching Path objects.
    """
    grouped: Dict[int, List[Path]] = {}
    unmatched: List[str] = []

    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        m = _FILENAME_RE.match(f.name)
        if m:
            subset_id = int(m.group("subset"))
            grouped.setdefault(subset_id, []).append(f)
        elif f.suffix == ".npz":
            unmatched.append(f.name)

    if unmatched:
        log.warning(
            "%d .npz file(s) did not match the expected naming pattern and will be ignored:",
            len(unmatched),
        )
        for name in unmatched:
            log.warning("  %s", name)

    return grouped


def assign_splits(
    subset_ids: List[int],
    n_train: int,
    n_finetune: int,
    n_test: int,
    rng: np.random.Generator,
) -> Dict[str, List[int]]:
    """
    Randomly assign subset IDs to splits without replacement.

    Raises
    ------
    ValueError if the total requested exceeds available subsets.
    """
    total_requested = n_train + n_finetune + n_test
    available = len(subset_ids)

    if total_requested > available:
        raise ValueError(
            f"Requested {total_requested} subsets ({n_train} train + "
            f"{n_finetune} finetune + {n_test} test) but only "
            f"{available} subsets are available."
        )

    shuffled = list(subset_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    return {
        "train": sorted(shuffled[:n_train]),
        "finetune": sorted(shuffled[n_train : n_train + n_finetune]),
        "test": sorted(shuffled[n_train + n_finetune : total_requested]),
        "unallocated": sorted(shuffled[total_requested:]),
    }


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load a .npz file and return its arrays as a plain dict."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def concatenate_with_group_offset(
    file_lists: List[List[Path]],
) -> Dict[str, np.ndarray]:
    """
    Concatenate multiple .npz files into a single dataset, offsetting group IDs
    so that groups from different cases do not collide in the loss function.

    Each .npz is expected to contain at minimum:
        - ``coordinates``  : (N, 3) float array
        - ``targets``      : (N,)   float array
        - ``group_ids``    : (N,)   int array   (0-based within each file)
        - ``scaler_min``   : (3,)   float array (coordinate normalisation)
        - ``scaler_max``   : (3,)   float array

    The scaler from the first file encountered is used for the combined dataset
    (all cases share the same mesh, so coordinate bounds are identical).

    Returns
    -------
    dict with keys ``coordinates``, ``targets``, ``group_ids``,
    ``scaler_min``, ``scaler_max``.
    """
    all_coords: List[np.ndarray] = []
    all_intensities: List[np.ndarray] = []
    all_group_sizes: List[np.ndarray] = []
    all_group_ids: List[np.ndarray] = []
    scaler_min: Optional[np.ndarray] = None
    scaler_max: Optional[np.ndarray] = None

    group_offset = 0
    files_processed = 0

    flat_files = [f for sublist in file_lists for f in sublist]

    for path in flat_files:
        data = load_npz(path)

        for key in ("coordinates", "intensities", "group_ids"):
            if key not in data:
                raise KeyError(
                    f"Expected key '{key}' not found in {path.name}. "
                    f"Keys present: {list(data.keys())}"
                )

        if scaler_min is None:
            scaler_min = data.get("scaler_min")
            scaler_max = data.get("scaler_max")
            if scaler_min is None:
                log.warning(
                    "scaler_min/scaler_max not found in %s; "
                    "combined .npz will not contain scaler arrays.",
                    path.name,
                )

        group_ids = data["group_ids"].astype(np.int64)
        n_groups_this_file = int(group_ids.max()) + 1

        all_coords.append(data["coordinates"])
        all_intensities.append(data["intensities"])
        if "group_sizes" in data:
            all_group_sizes.append(data["group_sizes"])
        all_group_ids.append(group_ids + group_offset)

        group_offset += n_groups_this_file
        files_processed += 1

    log.info(
        "Concatenated %d files -> %d total nodes, %d total groups.",
        files_processed,
        sum(a.shape[0] for a in all_coords),
        group_offset,
    )

    combined: Dict[str, np.ndarray] = {
        "coordinates": np.concatenate(all_coords, axis=0),
        "intensities": np.concatenate(all_intensities, axis=0),
        "group_ids": np.concatenate(all_group_ids, axis=0),
    }
    if all_group_sizes:
        combined["group_sizes"] = np.concatenate(all_group_sizes, axis=0)

    if scaler_min is not None:
        combined["scaler_min"] = scaler_min
        combined["scaler_max"] = scaler_max  # type: ignore[assignment]

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build(
    input_dir: Path,
    output_dir: Path,
    n_train: int,
    n_finetune: int,
    n_test: int,
    expected_subsets: int,
    seed: int,
) -> None:
    """Full pipeline: scan -> split -> concatenate -> save."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # --- Scan ---
    log.info("Scanning input directory: %s", input_dir)
    grouped = scan_files(input_dir)
    found_subset_ids = sorted(grouped.keys())
    log.info("Found %d distinct subset(s) across %d file(s).",
             len(found_subset_ids),
             sum(len(v) for v in grouped.values()))

    # --- Missing expected ---
    all_expected_ids = list(range(expected_subsets))
    missing_expected = sorted(set(all_expected_ids) - set(found_subset_ids))
    if missing_expected:
        log.warning(
            "%d subset(s) expected but not found (likely incomplete transfer): %s",
            len(missing_expected),
            missing_expected,
        )

    # --- Assign splits ---
    rng = np.random.default_rng(seed)
    splits = assign_splits(found_subset_ids, n_train, n_finetune, n_test, rng)

    log.info(
        "Split assignment (seed=%d): train=%s  finetune=%s  test=%s  unallocated=%s",
        seed,
        splits["train"],
        splits["finetune"],
        splits["test"],
        splits["unallocated"],
    )

    # --- Build file lists per split ---
    split_files: Dict[str, Dict[int, List[str]]] = {}
    for split_name, subset_list in splits.items():
        split_files[split_name] = {
            sid: [str(p) for p in grouped[sid]]
            for sid in subset_list
            if sid in grouped
        }

    # --- Save manifest ---
    manifest = {
        "seed": seed,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "expected_subsets": expected_subsets,
        "found_subsets": found_subset_ids,
        "missing_expected": missing_expected,
        "splits": split_files,
        "foundation_training_npz": str(output_dir / "foundation_training.npz"),
    }

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to %s", manifest_path)

    # --- Concatenate training split ---
    train_file_lists = [grouped[sid] for sid in splits["train"] if sid in grouped]

    if not train_file_lists:
        log.error("No training files to concatenate. Aborting.")
        sys.exit(1)

    log.info("Concatenating %d training subset(s)...", len(train_file_lists))
    combined = concatenate_with_group_offset(train_file_lists)

    out_npz = output_dir / "foundation_training.npz"
    np.savez_compressed(out_npz, **combined)
    log.info("Foundation training dataset saved to %s", out_npz)
    log.info(
        "  coordinates : %s  dtype=%s", combined["coordinates"].shape, combined["coordinates"].dtype
    )
    log.info(
        "  targets      : %s  dtype=%s", combined["targets"].shape, combined["targets"].dtype
    )
    log.info(
        "  group_ids    : %s  dtype=%s  max_group=%d",
        combined["group_ids"].shape,
        combined["group_ids"].dtype,
        int(combined["group_ids"].max()),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a foundation model dataset from cardioscar .npz training files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .npz files from `cardioscar prepare`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the foundation dataset and manifest will be written.",
    )
    parser.add_argument(
        "--train-subsets",
        type=int,
        default=11,
        help="Number of subsets to assign to the training split.",
    )
    parser.add_argument(
        "--finetune-subsets",
        type=int,
        default=2,
        help="Number of subsets to assign to the fine-tune split.",
    )
    parser.add_argument(
        "--test-subsets",
        type=int,
        default=2,
        help="Number of subsets to assign to the test split.",
    )
    parser.add_argument(
        "--expected-subsets",
        type=int,
        required=True,
        help=(
            "Total number of subsets expected (e.g. 180). Used to compute "
            "missing_expected in the manifest."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split assignment.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    build(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_train=args.train_subsets,
        n_finetune=args.finetune_subsets,
        n_test=args.test_subsets,
        expected_subsets=args.expected_subsets,
        seed=args.seed,
    )