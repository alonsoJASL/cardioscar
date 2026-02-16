# scripts/inspect_real_data.py (CORRECTED)
import pyvista as pv
import SimpleITK as sitk
import numpy as np

print("=" * 60)
print("REAL DATA INSPECTION")
print("=" * 60)

# 1. Mesh
mesh = pv.read("data/real_data_test/input_model.vtk")
print(f"\n1. Input Mesh:")
print(f"   Nodes: {mesh.n_points:,}")
print(f"   Cells: {mesh.n_cells:,}")
print(f"   Bounds: {mesh.bounds}")

# 2. LGE Image
lge = sitk.ReadImage("data/real_data_test/lge_scan.nii.gz")
print(f"\n2. LGE Scan:")
print(f"   Size: {lge.GetSize()} (X, Y, Z)")
print(f"   Spacing: {lge.GetSpacing()} mm")
print(f"   Origin: {lge.GetOrigin()}")
print(f"   Number of slices: {lge.GetSize()[2]}")

# Check intensity range
lge_array = sitk.GetArrayFromImage(lge)
print(f"   Intensity range: [{lge_array.min():.1f}, {lge_array.max():.1f}]")
print(f"   Mean: {lge_array.mean():.1f}, Std: {lge_array.std():.1f}")

# 3. Segmentation
seg = sitk.ReadImage("data/real_data_test/lge_segmentation.nii.gz")
print(f"\n3. LGE Segmentation:")
print(f"   Size: {seg.GetSize()}")
print(f"   Spacing: {seg.GetSpacing()} mm")

seg_array = sitk.GetArrayFromImage(seg)
unique_labels = np.unique(seg_array)
print(f"   Labels: {unique_labels}")
for label in unique_labels:
    count = (seg_array == label).sum()
    pct = 100 * count / seg_array.size
    print(f"     Label {label}: {count:,} voxels ({pct:.1f}%)")

# 4. Baseline Projection - CORRECTED
baseline = pv.read("data/real_data_test/scar3d_naive_proj_output.vtk")
print(f"\n4. Baseline Projection:")
print(f"   Available fields: {baseline.array_names}")

# Use 'scar' field specifically
if 'scar' in baseline.array_names:
    scar_values = baseline['scar']
    print(f"   Field 'scar':")
    print(f"     Range: [{scar_values.min():.3f}, {scar_values.max():.3f}]")
    print(f"     Mean: {scar_values.mean():.3f}")
    print(f"     Std: {scar_values.std():.3f}")
    print(f"     >0.5: {(scar_values > 0.5).sum()} nodes ({100*(scar_values > 0.5).sum()/len(scar_values):.1f}%)")
    print(f"     >0.8: {(scar_values > 0.8).sum()} nodes ({100*(scar_values > 0.8).sum()/len(scar_values):.1f}%)")
else:
    print(f"   ⚠️ Warning: 'scar' field not found!")
    print(f"   Available: {baseline.array_names}")

# 5. Check Alignment
print(f"\n5. Coordinate Alignment Check:")
mesh_z_range = [mesh.bounds[4], mesh.bounds[5]]
lge_z_range = [
    lge.GetOrigin()[2],
    lge.GetOrigin()[2] + lge.GetSize()[2] * lge.GetSpacing()[2]
]
print(f"   Mesh Z: [{mesh_z_range[0]:.1f}, {mesh_z_range[1]:.1f}] mm")
print(f"   LGE Z:  [{lge_z_range[0]:.1f}, {lge_z_range[1]:.1f}] mm")

overlap_min = max(mesh_z_range[0], lge_z_range[0])
overlap_max = min(mesh_z_range[1], lge_z_range[1])
overlap = overlap_max - overlap_min

if overlap > 0:
    print(f"   ✓ Overlap: {overlap:.1f} mm")
else:
    print(f"   ✗ NO OVERLAP! Coordinates misaligned.")