#!/usr/bin/env python3
"""
Flip mesh 180° along the LGE image's slice-stacking direction.

This accounts for the segmentation/scan slice order reversal.
"""
import logging
import argparse
import pyvista as pv
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from pycemrg.core.logs import setup_logging
setup_logging()
logger = logging.getLogger("FlipMeshImageAxis")

def flip_mesh_along_image_axis(
    mesh_path: Path,
    image_path: Path,
    output_path: Path,
    flip_axis: str = 'z'
):
    """
    Flip mesh 180° along one of the image's intrinsic axes.
    
    Args:
        mesh_path: Input mesh
        image_path: Reference image (to get direction matrix)
        output_path: Output flipped mesh
        flip_axis: 'x', 'y', or 'z' in image coordinate system
    """
    # Load mesh
    mesh = pv.read(str(mesh_path))
    mesh_center = mesh.center
    
    # Load image to get direction matrix
    image = sitk.ReadImage(str(image_path))
    direction = np.array(image.GetDirection()).reshape(3, 3)
    
    logger.info(f"Image direction matrix:")
    logger.info(direction)
    
    # Get the axis to flip around
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[flip_axis]
    flip_direction = direction[:, axis_idx]  # Column vector
    
    logger.info(f"\nFlipping along image {flip_axis}-axis:")
    logger.info(f"  Direction vector: {flip_direction}")
    
    # Create reflection matrix across plane perpendicular to flip_direction
    # Reflection formula: I - 2 * n ⊗ n  where n is unit normal
    n = flip_direction / np.linalg.norm(flip_direction)
    reflection_matrix = np.eye(3) - 2 * np.outer(n, n)
    
    logger.info(f"  Reflection matrix:")
    logger.info(reflection_matrix)
    
    # Apply reflection around mesh center
    mesh.points -= mesh_center
    mesh.points = (reflection_matrix @ mesh.points.T).T
    mesh.points += mesh_center
    
    mesh.save(str(output_path))
    
    logger.info(f"\n✓ Flipped mesh saved to: {output_path}")
    logger.info(f"  Original center: {mesh_center}")
    logger.info(f"  New center: {mesh.center}")  # Should be unchanged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", type=str, default='z', choices=['x', 'y', 'z'])
    
    args = parser.parse_args()
    
    flip_mesh_along_image_axis(
        mesh_path=args.mesh,
        image_path=args.image,
        output_path=args.output,
        flip_axis=args.axis
    )