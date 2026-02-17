# scripts/inspection/inspect_real_data.py
import argparse
import pyvista as pv
import SimpleITK as sitk
import numpy as np
from pathlib import Path


def check_axis_alignment(mesh_min, mesh_max, img_min, img_max, axis_name):
    """Check alignment for a single axis and print results."""
    overlap_min = max(mesh_min, img_min)
    overlap_max = min(mesh_max, img_max)
    overlap = overlap_max - overlap_min

    mesh_span = mesh_max - mesh_min
    img_span = img_max - img_min

    if overlap > 0:
        overlap_pct_mesh = 100 * overlap / mesh_span
        overlap_pct_img = 100 * overlap / img_span
        print(f"   {axis_name}: Mesh [{mesh_min:.1f}, {mesh_max:.1f}] | "
              f"LGE [{img_min:.1f}, {img_max:.1f}] | "
              f"Overlap: {overlap:.1f} mm "
              f"({overlap_pct_mesh:.0f}% of mesh, {overlap_pct_img:.0f}% of image)")
    else:
        gap = overlap_min - overlap_max
        print(f"   {axis_name}: Mesh [{mesh_min:.1f}, {mesh_max:.1f}] | "
              f"LGE [{img_min:.1f}, {img_max:.1f}] | "
              f"✗ NO OVERLAP (gap: {gap:.1f} mm)")


def print_field_stats(name: str, values: np.ndarray):
    """Print standard statistics for a scalar field."""
    print(f"   Field '{name}':")
    print(f"     Range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"     Mean:  {values.mean():.3f}")
    print(f"     Std:   {values.std():.3f}")
    print(f"     >0.5:  {(values > 0.5).sum():,} nodes "
          f"({100*(values > 0.5).sum()/len(values):.1f}%)")
    print(f"     >0.8:  {(values > 0.8).sum():,} nodes "
          f"({100*(values > 0.8).sum()/len(values):.1f}%)")


def main(args):
    print("=" * 60)
    print("REAL DATA INSPECTION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Input Mesh
    # ------------------------------------------------------------------
    mesh = pv.read(args.input_mesh)
    print(f"\n1. Input Mesh: {args.input_mesh.name}")
    print(f"   Nodes: {mesh.n_points:,}")
    print(f"   Cells: {mesh.n_cells:,}")
    print(f"   Bounds:")
    print(f"     X: [{mesh.bounds[0]:.1f}, {mesh.bounds[1]:.1f}] mm")
    print(f"     Y: [{mesh.bounds[2]:.1f}, {mesh.bounds[3]:.1f}] mm")
    print(f"     Z: [{mesh.bounds[4]:.1f}, {mesh.bounds[5]:.1f}] mm")
    print(f"   Available fields: {mesh.array_names}")

    if args.input_mesh_fieldname in mesh.array_names:
        print(f"\n   Prediction field:")
        print_field_stats(args.input_mesh_fieldname, mesh[args.input_mesh_fieldname])
    else:
        print(f"   ⚠️ Field '{args.input_mesh_fieldname}' not found in input mesh.")
        print(f"   Available: {mesh.array_names}")

    # ------------------------------------------------------------------
    # 2. LGE Image
    # ------------------------------------------------------------------
    lge = sitk.ReadImage(str(args.ref_scan))
    origin = lge.GetOrigin()
    spacing = lge.GetSpacing()
    size = lge.GetSize()
    direction = lge.GetDirection()

    lge_array = sitk.GetArrayFromImage(lge)

    print(f"\n2. LGE Scan: {args.ref_scan.name}")
    print(f"   Size:    {size} (X, Y, Z)")
    print(f"   Spacing: ({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}) mm")
    print(f"   Origin:  ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}) mm")
    print(f"   Number of slices: {size[2]}")
    print(f"   Intensity range: [{lge_array.min():.1f}, {lge_array.max():.1f}]")
    print(f"   Mean: {lge_array.mean():.1f}, Std: {lge_array.std():.1f}")

    # Direction matrix - flag if oblique
    dir_matrix = np.array(direction).reshape(3, 3)
    print(f"   Direction matrix:")
    for row in dir_matrix:
        print(f"     [{row[0]:6.3f}, {row[1]:6.3f}, {row[2]:6.3f}]")

    is_oblique = not np.allclose(np.abs(dir_matrix), np.eye(3), atol=0.01)
    if is_oblique:
        print(f"   ⚠️ Image is OBLIQUE - may affect coordinate alignment!")
    else:
        print(f"   ✓ Image is axis-aligned")

    # ------------------------------------------------------------------
    # 3. Segmentation
    # ------------------------------------------------------------------
    seg = sitk.ReadImage(str(args.ref_segmentation))
    seg_array = sitk.GetArrayFromImage(seg)
    unique_labels = np.unique(seg_array)

    print(f"\n3. LGE Segmentation: {args.ref_segmentation.name}")
    print(f"   Size:    {seg.GetSize()}")
    print(f"   Spacing: {seg.GetSpacing()} mm")
    print(f"   Labels:")
    for label in unique_labels:
        count = (seg_array == label).sum()
        pct = 100 * count / seg_array.size
        print(f"     Label {label}: {count:,} voxels ({pct:.1f}%)")

    # Check seg/image consistency
    if seg.GetSize() != lge.GetSize():
        print(f"   ⚠️ Segmentation size {seg.GetSize()} != LGE size {size}!")
    else:
        print(f"   ✓ Segmentation size matches LGE")

    if not np.allclose(seg.GetOrigin(), lge.GetOrigin(), atol=0.01):
        print(f"   ⚠️ Segmentation origin != LGE origin!")
    else:
        print(f"   ✓ Segmentation origin matches LGE")

    # ------------------------------------------------------------------
    # 4. Reference Model (Baseline)
    # ------------------------------------------------------------------
    baseline = pv.read(str(args.ref_model))
    print(f"\n4. Reference Model: {args.ref_model.name}")
    print(f"   Nodes: {baseline.n_points:,}")
    print(f"   Available fields: {baseline.array_names}")

    if args.ref_model_fieldname in baseline.array_names:
        print_field_stats(args.ref_model_fieldname,
                          baseline[args.ref_model_fieldname])
    else:
        print(f"   ⚠️ Field '{args.ref_model_fieldname}' not found!")

    # Node count consistency
    if baseline.n_points != mesh.n_points:
        print(f"   ⚠️ Baseline nodes ({baseline.n_points:,}) != "
              f"Input mesh nodes ({mesh.n_points:,})!")
    else:
        print(f"   ✓ Baseline and input mesh have same node count")

    # ------------------------------------------------------------------
    # 5. Coordinate Alignment (All 3 Axes)
    # ------------------------------------------------------------------
    print(f"\n5. Coordinate Alignment Check:")

    # Compute LGE physical bounds
    lge_x_min = origin[0]
    lge_x_max = origin[0] + size[0] * spacing[0]
    lge_y_min = origin[1]
    lge_y_max = origin[1] + size[1] * spacing[1]
    lge_z_min = origin[2]
    lge_z_max = origin[2] + size[2] * spacing[2]

    check_axis_alignment(mesh.bounds[0], mesh.bounds[1],
                         lge_x_min, lge_x_max, "X")
    check_axis_alignment(mesh.bounds[2], mesh.bounds[3],
                         lge_y_min, lge_y_max, "Y")
    check_axis_alignment(mesh.bounds[4], mesh.bounds[5],
                         lge_z_min, lge_z_max, "Z")

    # ------------------------------------------------------------------
    # 6. Scar Field Comparison (if input mesh has predictions)
    # ------------------------------------------------------------------
    if (args.input_mesh_fieldname in mesh.array_names and
            args.ref_model_fieldname in baseline.array_names and
            mesh.n_points == baseline.n_points):

        print(f"\n6. Scar Field Comparison:")

        pred = mesh[args.input_mesh_fieldname]
        ref = baseline[args.ref_model_fieldname]

        from scipy.stats import pearsonr

        corr, _ = pearsonr(pred, ref)
        mae = np.abs(pred - ref).mean()

        # Dice at threshold 0.5
        pred_bin = (pred >= 0.5).astype(int)
        ref_bin = (ref >= 0.5).astype(int)
        intersection = (pred_bin * ref_bin).sum()
        dice = 2 * intersection / (pred_bin.sum() + ref_bin.sum() + 1e-8)

        print(f"   Correlation (r):  {corr:.4f}")
        print(f"   MAE:              {mae:.4f}")
        print(f"   Dice (t=0.5):     {dice:.4f}")
        print(f"   Predicted scar:   {pred_bin.sum():,} nodes "
              f"({100*pred_bin.sum()/len(pred_bin):.1f}%)")
        print(f"   Reference scar:   {ref_bin.sum():,} nodes "
              f"({100*ref_bin.sum()/len(ref_bin):.1f}%)")
    else:
        print(f"\n6. Scar Field Comparison: skipped "
              f"(missing fields or node count mismatch)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect and compare real cardiac data files."
    )
    parser.add_argument('--input-mesh', type=Path, required=True,
                        help="CardioScar output mesh with predictions")
    parser.add_argument('--input-mesh-fieldname', type=str,
                        default='scar_probability',
                        help="Scalar field name in input mesh")
    parser.add_argument('--ref-scan', type=Path, required=True,
                        help="LGE NIfTI image")
    parser.add_argument('--ref-segmentation', type=Path, required=True,
                        help="LGE segmentation NIfTI image")
    parser.add_argument('--ref-model', type=Path, required=True,
                        help="Baseline projection VTK mesh")
    parser.add_argument('--ref-model-fieldname', type=str,
                        default='scar',
                        help="Scalar field name in reference model")

    args = parser.parse_args()
    main(args)