# CardioScar Scripts

Utility scripts for data processing, analysis, and validation.

## Utilities (`utilities/`)

**Data preprocessing and format conversion:**
- `fix_scan_intensity_normalisation.py` - Normalize LGE intensities in .npz files
<!-- - `create_binary_scar.py` - Convert multi-label segmentation to binary scar -->
- `convert_vtk_slice_to_image.py` - Convert VTK grids to NIfTI for testing

**Usage:**
```bash
python scripts/utilities/fix_lge_normalization.py --input PATH [--output PATH]
```

## Analysis (`analysis/`)

**Model comparison and validation:**
<!-- - `compare_all_methods.py` - Compare baseline vs CardioScar variants -->
- `comparison_with_leagcy.py` - Validate against legacy TensorFlow
<!-- - `visualize_comparison.py` - PyVista visualization of results -->

## Inspection (`inspection/`)

**Dataset exploration:**
- `inspect_real_data.py` - Inspect mesh, images, segmentation alignment
<!-- - `quality_control_check.py` - Uncertainty-based QC metrics -->

**Usage:**
```bash
python scripts/inspection/inspect_real_data.py
```

---

See `examples/` for complete workflow examples.  
See `tests/integration/` for automated validation tests.