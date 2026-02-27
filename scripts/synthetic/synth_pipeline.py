#!/usr/bin/env python3
# scripts/synthetic/synth_pipeline.py

"""
Synthetic Scar Dataset Generation Pipeline

Three-stage pipeline for generating a diverse synthetic training dataset
for CardioScar validation:

  1. generate-seeds  - Randomly sample subsets from a hand-picked coordinate
                       file to use as seed inputs for syntheticScarVolume.
  2. rasterize       - Convert a VTK mesh (with synthetic_scar point scalar)
                       into a NIfTI volume on an axis-aligned bounding-box grid.
  3. generate-slices - Produce multi-resolution sparse-slice NIfTI volumes
                       from a rasterized NIfTI, simulating LGE-CMR acquisition.

Typical workflow
----------------
# Stage 1: generate seed subsets
python synth_pipeline.py generate-seeds \\
    --pts-file all_coords.pts \\
    --output-dir seeds/ \\
    --n-subsets 10

# Stage 2: user runs syntheticScarVolume for each subset × parameter variant
# Example bash loop (not managed by this script):
#   for pts in seeds/*.pts; do
#     stem=$(basename $pts .pts)
#     /path/to/syntheticScarVolume -i mesh.vtk -pts $pts -n 15 -falloff 1 -o scars/${stem}_n15_f1.vtk
#     /path/to/syntheticScarVolume -i mesh.vtk -pts $pts -n 20 -falloff 2 -o scars/${stem}_n20_f2.vtk
#   done

# Stage 3a: rasterize each scar mesh → NIfTI (run in loop)
#   for vtk in scars/*.vtk; do
#     python synth_pipeline.py rasterize --mesh-vtk $vtk --output-dir niftis/
#   done

# Stage 3b: generate multi-resolution slice volumes
#   for nii in niftis/*.nii.gz; do
#     python synth_pipeline.py generate-slices --nifti $nii --output-dir slices/
#   done

Authors:
    Jose Alonso Solis-Lemus
"""

import logging
import math
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from pycemrg.core.logs import setup_logging
from pycemrg_image_analysis.utilities.io import array_to_image, save_image
from pycemrg_image_analysis.utilities.artifact_simulation import downsample_volume

setup_logging()
logger = logging.getLogger("SynthPipeline")


# =============================================================================
# CLI GROUP
# =============================================================================


@click.group()
def cli() -> None:
    """Synthetic scar dataset generation pipeline for CardioScar validation."""


# =============================================================================
# SUBCOMMAND: generate-seeds
# =============================================================================


@cli.command("generate-seeds")
@click.option(
    "--pts-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to coordinate file with one 'x y z' triplet per line.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write seed subset .pts files.",
)
@click.option(
    "--n-subsets",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of random seed subsets to generate.",
)
@click.option(
    "--min-seeds",
    default=15,
    show_default=True,
    type=click.IntRange(min=1),
    help="Minimum number of seed points per subset.",
)
@click.option(
    "--max-seeds",
    default=25,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of seed points per subset.",
)
@click.option(
    "--random-seed",
    default=42,
    show_default=True,
    type=int,
    help="NumPy random seed for reproducibility.",
)
def generate_seeds(
    pts_file: Path,
    output_dir: Path,
    n_subsets: int,
    min_seeds: int,
    max_seeds: int,
    random_seed: int,
) -> None:
    """
    Randomly sample N subsets from a hand-picked coordinate file.

    Each subset is written as a .pts file compatible with syntheticScarVolume's
    -pts argument (one 'x y z' triplet per line, space-separated).

    Subsets are sampled without replacement. Subset size is drawn uniformly
    from [min-seeds, max-seeds] for each subset independently.
    """
    if min_seeds > max_seeds:
        raise click.BadParameter(
            f"--min-seeds ({min_seeds}) must be <= --max-seeds ({max_seeds})"
        )

    coords = _load_pts_file(pts_file)
    n_available = len(coords)
    logger.info(f"Loaded {n_available} coordinates from {pts_file}")

    if max_seeds > n_available:
        raise click.BadParameter(
            f"--max-seeds ({max_seeds}) exceeds available coordinates ({n_available})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    for i in range(n_subsets):
        subset_size = int(rng.integers(min_seeds, max_seeds + 1))
        indices = rng.choice(n_available, size=subset_size, replace=False)
        subset = coords[indices]

        out_path = output_dir / f"subset_{i:03d}.pts"
        np.savetxt(out_path, subset, fmt="%.6f", delimiter=" ")
        logger.info(f"  subset_{i:03d}.pts → {subset_size} points")

    logger.info(f"Generated {n_subsets} seed files in {output_dir}")


# =============================================================================
# SUBCOMMAND: rasterize
# =============================================================================


@cli.command("rasterize")
@click.option(
    "--mesh-vtk",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a single VTK/VTU mesh with a point scalar field. "
        "For multiple meshes, invoke this command in a shell loop: "
        "for f in scars/*.vtk; do python synth_pipeline.py rasterize --mesh-vtk $f ...; done"
    ),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write the output NIfTI volume.",
)
@click.option(
    "--scalar-field",
    default="synthetic_scar",
    show_default=True,
    help="Name of the point scalar array on the mesh to rasterize.",
)
@click.option(
    "--voxel-size",
    default=1.0,
    show_default=True,
    type=float,
    help="Isotropic voxel size in mm for the output NIfTI grid.",
)
@click.option(
    "--padding",
    default=2.0,
    show_default=True,
    type=float,
    help="Physical padding in mm added around the mesh bounding box on all sides.",
)
@click.option(
    "--base-value",
    default=0.1,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help=(
        "Baseline intensity assigned to all voxels inside the mesh but carrying "
        "no scar signal. Scar values are added on top, so the effective range "
        "is [base-value, base-value + max_scar], clipped to [0, 1]. "
        "Set to 0.0 to disable (background-only masking without a baseline)."
    ),
)
def rasterize(
    mesh_vtk: Path,
    output_dir: Path,
    scalar_field: str,
    voxel_size: float,
    padding: float,
    base_value: float,
) -> None:
    """
    Convert a VTK mesh with a point scalar to an axis-aligned NIfTI volume.

    The output grid origin is set to the padded mesh bounding box minimum,
    with identity direction matrix (axis-aligned). This guarantees that the
    mesh and image share the same physical coordinate frame, making
    registration trivially exact for synthetic validation.

    Voxels inside the mesh but carrying no scar signal receive --base-value,
    simulating the background myocardial signal in LGE-CMR. Scar signal is
    added on top of this baseline. True background (outside the mesh) is 0.
    """
    logger.info(f'Rasterising mesh [{mesh_vtk.name}]')
    mesh = pv.read(mesh_vtk)

    if scalar_field not in mesh.point_data.keys():
        raise click.BadParameter(
            f"Scalar field '{scalar_field}' not found in {mesh_vtk.name}. "
            f"Available fields: {list(mesh.point_data.keys())}"
        )

    logger.info(f"Rasterizing: {mesh_vtk.name}")
    logger.info(f"  Mesh nodes: {mesh.n_points}, cells: {mesh.n_cells}")
    logger.info(f"  Scalar field: '{scalar_field}', base value: {base_value:.3f}")

    volume, origin, spacing = _probe_mesh_to_volume(
        mesh=mesh,
        scalar_field=scalar_field,
        voxel_size=voxel_size,
        padding=padding,
        base_value=base_value,
    )

    logger.info(f"  Output grid shape (Z, Y, X): {volume.shape}")
    logger.info(f"  Origin (X, Y, Z): {origin}")
    logger.info(f"  Spacing: {spacing[0]:.2f} mm isotropic")
    logger.info(
        f"  Non-zero voxels: {np.count_nonzero(volume):,} "
        f"/ {volume.size:,} ({100 * np.count_nonzero(volume) / volume.size:.1f}%)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (mesh_vtk.stem + ".nii.gz")

    image = array_to_image(volume, origin=origin, spacing=spacing)
    save_image(image, out_path)
    logger.info(f"  Saved → {out_path}")


# =============================================================================
# SUBCOMMAND: generate-slices
# =============================================================================


@cli.command("generate-slices")
@click.option(
    "--nifti",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a single NIfTI volume (typically output of rasterize). "
        "For multiple volumes, invoke in a shell loop."
    ),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write the output sparse-slice NIfTI volumes.",
)
@click.option(
    "--xy-spacings",
    default=[1.0, 1.5, 2.0],
    multiple=True,
    type=float,
    show_default=True,
    help=(
        "In-plane (X/Y) voxel spacings in mm. One output volume is produced "
        "per value. Pass multiple times: --xy-spacings 1.0 --xy-spacings 1.5"
    ),
)
@click.option(
    "--z-spacing",
    default=8.0,
    show_default=True,
    type=float,
    help="Through-plane (Z) slice spacing in mm, simulating LGE-CMR acquisition.",
)
def generate_slices(
    nifti: Path,
    output_dir: Path,
    xy_spacings: Tuple[float, ...],
    z_spacing: float,
) -> None:
    """
    Produce multi-resolution sparse-slice NIfTI volumes from a rasterized mesh.

    Simulates LGE-CMR acquisition geometry: thick z-spacing (8-10mm) with
    varying in-plane resolution. One output NIfTI is produced per xy-spacing
    value. Direction matrix and origin are preserved from the input.

    Output naming: {input_stem}_xy{xy}mm_z{z}mm.nii.gz
    """
    if not xy_spacings:
        raise click.BadParameter("At least one --xy-spacings value is required.")

    image = sitk.ReadImage(str(nifti))
    original_spacing = image.GetSpacing()  # (sx, sy, sz) in mm
    logger.info(f"Generating slices for: {nifti.name}")
    logger.info(f"  Input spacing (X, Y, Z): {[f'{s:.2f}' for s in original_spacing]} mm")
    logger.info(f"  Target Z spacing: {z_spacing:.1f} mm")

    output_dir.mkdir(parents=True, exist_ok=True)

    for xy in xy_spacings:
        target_spacing = (xy, xy, z_spacing)

        if z_spacing <= original_spacing[2]:
            logger.warning(
                f"  Target Z spacing ({z_spacing:.1f} mm) <= input Z spacing "
                f"({original_spacing[2]:.2f} mm). This would upsample in Z. "
                f"Skipping xy={xy:.1f}mm variant."
            )
            continue

        downsampled = downsample_volume(image, target_spacing=target_spacing)

        out_spacing = downsampled.GetSpacing()
        out_size = downsampled.GetSize()
        logger.info(
            f"  xy={xy:.1f}mm → size {out_size}, "
            f"achieved spacing ({out_spacing[0]:.2f}, {out_spacing[1]:.2f}, {out_spacing[2]:.2f}) mm"
        )

        xy_tag = f"{xy:.1f}".replace(".", "p")
        z_tag = f"{z_spacing:.1f}".replace(".", "p")
        stem = nifti.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}_xy{xy_tag}mm_z{z_tag}mm.nii.gz"

        sitk.WriteImage(downsampled, str(out_path))
        logger.info(f"  Saved → {out_path.name}")

    logger.info(f"Done: {len(xy_spacings)} resolution variants for {nifti.name}")


# =============================================================================
# INTERNAL UTILITIES
# =============================================================================


def _load_pts_file(pts_file: Path) -> np.ndarray:
    """
    Load a coordinate file with one 'x y z' triplet per line.

    Args:
        pts_file: Path to the coordinate file.

    Returns:
        (N, 3) float64 array of coordinates.

    Raises:
        ValueError: If file cannot be parsed as (N, 3) numeric data.
    """
    try:
        coords = np.loadtxt(pts_file)
    except Exception as exc:
        raise ValueError(f"Could not parse {pts_file} as numeric data: {exc}") from exc

    if coords.ndim == 1:
        coords = coords.reshape(1, 3)

    if coords.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns (x y z) in {pts_file}, got {coords.shape[1]}"
        )

    return coords


def _probe_mesh_to_volume(
    mesh: pv.DataSet,
    scalar_field: str,
    voxel_size: float,
    padding: float,
    base_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Sample a mesh point scalar onto a regular bounding-box grid.

    Constructs a uniform ImageData grid over the padded mesh bounding box,
    then uses PyVista's probe() to interpolate scalar values from mesh cells
    onto grid points.

    A temporary all-ones scalar ('_probe_mask') is added to a copy of the
    mesh to distinguish interior voxels (inside mesh cells) from true
    background (outside all cells). Interior voxels with no scar signal
    receive base_value; interior voxels with scar signal receive
    base_value + scar_value, clipped to [0, 1]. The original mesh is
    never mutated.

    Args:
        mesh: PyVista mesh with the target scalar in point_data.
        scalar_field: Name of the point scalar array to rasterize.
        voxel_size: Isotropic grid spacing in mm.
        padding: Physical margin added around the bounding box on all sides.
        base_value: Baseline intensity for all voxels inside the mesh.
            0.0 disables the baseline (scar-only rasterization).

    Returns:
        Tuple of:
        - volume: (nz, ny, nx) float32 NumPy array in Z-Y-X order.
        - origin: (ox, oy, oz) origin in physical coordinates (XYZ), for
                  use with array_to_image() (which expects XYZ origin).
        - spacing: (voxel_size, voxel_size, voxel_size) isotropic spacing.

    Notes:
        PyVista ImageData stores scalars with X varying fastest (VTK convention).
        The flat scalar array therefore reshapes to (nz, ny, nx) with standard
        C-order (numpy default), matching the (Z, Y, X) convention used
        throughout pycemrg-image-analysis.
    """
    _MASK_FIELD = "_probe_mask"

    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    ox = xmin - padding
    oy = ymin - padding
    oz = zmin - padding

    # Number of grid points along each axis (include endpoint)
    nx = math.ceil((xmax - xmin + 2 * padding) / voxel_size) + 1
    ny = math.ceil((ymax - ymin + 2 * padding) / voxel_size) + 1
    nz = math.ceil((zmax - zmin + 2 * padding) / voxel_size) + 1

    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(voxel_size, voxel_size, voxel_size),
        origin=(ox, oy, oz),
    )

    # Work on a shallow copy so the caller's mesh is never mutated.
    # Add all-ones mask scalar to identify interior grid points after probing.
    mesh_copy = mesh.copy(deep=False)
    mesh_copy.point_data[_MASK_FIELD] = np.ones(mesh_copy.n_points, dtype=np.float32)

    # Probe: interpolate both scalars from mesh cells onto grid points.
    # Grid points outside all mesh cells receive the VTK fill value, which
    # probe() maps to 0.0 for float arrays (mask will also be 0 there).
    result = grid.sample(mesh_copy)

    # VTK flat index ordering: ix + iy*nx + iz*nx*ny  (X varies fastest).
    # C-order reshape to (nz, ny, nx) is correct: index = iz*(ny*nx) + iy*nx + ix
    def _to_volume(field: str) -> np.ndarray:
        arr = np.array(result.point_data[field], dtype=np.float32)
        vol = arr.reshape((nz, ny, nx))
        return np.nan_to_num(vol, nan=0.0)

    scar_volume = _to_volume(scalar_field)
    mask_volume = _to_volume(_MASK_FIELD)  # > 0 where inside mesh cells

    # Compose: background=0, interior=base_value+scar, clipped to [0,1]
    interior = mask_volume > 0.0
    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    volume[interior] = np.clip(base_value + scar_volume[interior], 0.0, 1.0)

    logger.debug(
        f"  Interior voxels: {interior.sum():,} / {interior.size:,} "
        f"({100 * interior.sum() / interior.size:.1f}%)"
    )

    origin = (float(ox), float(oy), float(oz))
    spacing = (float(voxel_size), float(voxel_size), float(voxel_size))

    return volume, origin, spacing


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    cli()