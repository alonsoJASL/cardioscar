#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <nifti_file> [nifti_file ...]" >&2
    echo  " scripts/synthetic/generate_training_slices.sh"
    echo " "
    echo  " For each input NIfTI:"
    echo  "   1. Generate multi-resolution sparse-slice volumes (generate-slices)"
    echo  "   2. Convert each slice volume to VTK slice planes (generate_slice_planes.py)"
    echo " "
    echo  " Usage (single file):"
    echo  "   ./generate_training_slices.sh /data/.../subset_013_n10_f2_s3.0.nii.gz"
    echo " "
    echo  " Usage (batch, leave running):"
    echo  "   for f in /data/.../niftis/*.nii.gz; do"
    echo  "     ./generate_training_slices.sh "$f""
    echo  "   done"
    echo " "
    echo  "   Or in one call:"
    echo  "   ./generate_training_slices.sh /data/.../niftis/*.nii.gz"
    exit 1
fi

# ---------------------------------------------------------------------------
# Config - adjust paths to your environment
# ---------------------------------------------------------------------------
SYNTH_PIPELINE="scripts/synthetic/synth_pipeline.py"
SLICE_PLANES="scripts/utilities/generate_slice_planes.py"

XY_SPACINGS=(1.0 1.5 2.0)
Z_SPACING=8.0

# ---------------------------------------------------------------------------
# Argument check
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <nifti_file> [nifti_file ...]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Main loop over input NIfTIs
# ---------------------------------------------------------------------------
for INPUT_NIFTI in "$@"; do

    if [[ ! -f "$INPUT_NIFTI" ]]; then
        echo "[WARN] File not found, skipping: $INPUT_NIFTI" >&2
        continue
    fi

    OUTPUT_DIR="$(dirname "$INPUT_NIFTI")"

    # Strip .nii.gz or .nii suffix to get the base stem
    BASENAME="$(basename "$INPUT_NIFTI")"
    STEM="${BASENAME%.nii.gz}"
    STEM="${STEM%.nii}"

    echo "------------------------------------------------------------"
    echo "[INFO] Processing: $STEM"
    echo "------------------------------------------------------------"

    # ------------------------------------------------------------------
    # Stage 1: generate-slices
    # ------------------------------------------------------------------
    echo "[INFO] Generating slice volumes..."
    python "$SYNTH_PIPELINE" generate-slices \
        --nifti "$INPUT_NIFTI" \
        --output-dir "$OUTPUT_DIR" \
        --xy-spacings 1.0 \
        --xy-spacings 1.5 \
        --xy-spacings 2.0 \
        --z-spacing "$Z_SPACING"

    # ------------------------------------------------------------------
    # Stage 2: generate_slice_planes for each resolution variant
    # ------------------------------------------------------------------
    for XY in "${XY_SPACINGS[@]}"; do
        # Reconstruct the filename using the same tag format as synth_pipeline.py
        XY_TAG="$(printf "%.1f" "$XY" | tr '.' 'p')"
        Z_TAG="$(printf "%.1f" "$Z_SPACING" | tr '.' 'p')"

        SLICE_NIFTI="${OUTPUT_DIR}/${STEM}_xy${XY_TAG}mm_z${Z_TAG}mm.nii.gz"
        SLICE_STEM="${STEM}_xy${XY_TAG}mm_z${Z_TAG}mm"
        PLANES_OUTPUT_DIR="${OUTPUT_DIR}/${SLICE_STEM}"

        if [[ ! -f "$SLICE_NIFTI" ]]; then
            echo "[WARN] Expected slice volume not found, skipping: $SLICE_NIFTI" >&2
            continue
        fi

        echo "[INFO] Generating slice planes: $SLICE_STEM"
        python "$SLICE_PLANES" \
            --image "$SLICE_NIFTI" \
            --output-dir "$PLANES_OUTPUT_DIR"
    done

    echo "[INFO] Done: $STEM"
    echo ""

done

echo "All inputs processed."