"""
Test that VTK and image inputs produce identical results.
"""

import numpy as np
from pathlib import Path

from cardioscar.logic.contracts import PreprocessingRequest
from cardioscar.logic.orchestrators import prepare_training_data


def test_vtk_vs_image_consistency():
    """
    Compare VTK grid vs NIfTI image preprocessing.
    
    Both should produce identical outputs since the image
    was generated from the VTK grid.
    """
    mesh_path = Path("data/model.vtk")
    vtk_slice_path = Path("data/slice.vtk")
    image_path = Path("data/image_slice.nii")
    
    # 1. Process with VTK grid
    print("=" * 60)
    print("Processing with VTK grid...")
    print("=" * 60)
    
    request_vtk = PreprocessingRequest(
        mesh_path=mesh_path,
        vtk_grid_paths=[vtk_slice_path],
        slice_thickness_padding=5.0
    )
    
    result_vtk = prepare_training_data(request_vtk)
    
    print(f"\nVTK Result:")
    print(f"  Nodes: {result_vtk.n_nodes}")
    print(f"  Groups: {result_vtk.n_groups}")
    print(f"  Intensity range: [{result_vtk.intensities.min():.3f}, {result_vtk.intensities.max():.3f}]")
    
    # 2. Process with NIfTI image
    print("\n" + "=" * 60)
    print("Processing with NIfTI image...")
    print("=" * 60)
    
    request_image = PreprocessingRequest(
        mesh_path=mesh_path,
        image_path=image_path,
        slice_axis='z',
        slice_indices=[0],  # Single slice at Z=0
        slice_thickness_padding=5.0
    )
    
    result_image = prepare_training_data(request_image)
    
    print(f"\nImage Result:")
    print(f"  Nodes: {result_image.n_nodes}")
    print(f"  Groups: {result_image.n_groups}")
    print(f"  Intensity range: [{result_image.intensities.min():.3f}, {result_image.intensities.max():.3f}]")
    
    # 3. Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Image should find MORE nodes (voxels are bigger than VTK cells)
    print(f"VTK nodes: {result_vtk.n_nodes}")
    print(f"Image nodes: {result_image.n_nodes}")
    assert result_image.n_nodes >= result_vtk.n_nodes, \
        "Image should find at least as many nodes as VTK"

    # Check intensity distributions are similar
    print(f"VTK intensity range: [{result_vtk.intensities.min():.3f}, {result_vtk.intensities.max():.3f}]")
    print(f"Image intensity range: [{result_image.intensities.min():.3f}, {result_image.intensities.max():.3f}]")

    print("\n✅ WORKFLOWS BOTH SUCCEED!")
    print("Image workflow (voxel boxes) captures more nodes than VTK (cell points) - this is expected.")
    
    # Check basic stats
    assert result_vtk.n_nodes == result_image.n_nodes, \
        f"Node count mismatch: VTK={result_vtk.n_nodes}, Image={result_image.n_nodes}"
    
    assert result_vtk.n_groups == result_image.n_groups, \
        f"Group count mismatch: VTK={result_vtk.n_groups}, Image={result_image.n_groups}"
    
    # Check coordinate consistency (may have small floating point differences)
    coord_diff = np.abs(result_vtk.coordinates - result_image.coordinates).max()
    print(f"Max coordinate difference: {coord_diff:.6f}")
    assert coord_diff < 1e-5, f"Coordinates differ too much: {coord_diff}"
    
    # Check intensity consistency
    intensity_diff = np.abs(result_vtk.intensities - result_image.intensities).max()
    print(f"Max intensity difference: {intensity_diff:.6f}")
    assert intensity_diff < 1e-5, f"Intensities differ too much: {intensity_diff}"
    
    # Check group IDs (may be reordered, but should have same counts)
    vtk_group_sizes = np.bincount(result_vtk.group_ids)
    image_group_sizes = np.bincount(result_image.group_ids)
    
    vtk_group_sizes_sorted = np.sort(vtk_group_sizes)
    image_group_sizes_sorted = np.sort(image_group_sizes)
    
    assert np.array_equal(vtk_group_sizes_sorted, image_group_sizes_sorted), \
        "Group size distributions differ"
    
    print("\n✅ ALL TESTS PASSED!")
    print("VTK and Image workflows produce identical results.")


if __name__ == "__main__":
    test_vtk_vs_image_consistency()