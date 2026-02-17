import argparse 

import SimpleITK as sitk
import pyvista as pv
import numpy as np

from pathlib import Path 

def main(args) : 

    mesh = pv.read(args.mesh)
    lge = sitk.ReadImage(args.ref_scan)
    seg = sitk.ReadImage(args.ref_seegmentation)

    # Get physical position of a known scar voxel
    # Label 3 = scar in segmentation
    seg_array = sitk.GetArrayFromImage(seg)

    # Find first scar voxel (remember: SimpleITK array is Z,Y,X)
    scar_voxels = np.argwhere(seg_array == 3)
    print(f"Scar voxels (Z,Y,X array indices): {scar_voxels[:3]}")

    # Convert to physical using SimpleITK's own method
    for voxel in scar_voxels[:3]:
        z, y, x = voxel  # Array indexing is Z,Y,X
        physical = lge.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        print(f"  Voxel ({x},{y},{z}) → Physical {physical}")

    # Now check: do these physical coords fall inside the mesh?
    print(f"\nMesh Z range: [{mesh.bounds[4]:.1f}, {mesh.bounds[5]:.1f}]")
    print(f"Scar physical Z values: {[lge.TransformIndexToPhysicalPoint((int(v[2]), int(v[1]), int(v[0])))[2] for v in scar_voxels[:5]]}")

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--ref-scan', type=Path)
    parser.add_argument('--ref-segmentation', type=Path)

    args = parser.parse_args()
    main(args)