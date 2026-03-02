#!/usr/bin/env bash
# scripts/run_pipeline.sh
#
# Full CardioScar pipeline: prepare → train → apply
#
# Usage:
#   ./run_pipeline.sh <mesh_vtk> <image_nifti> <output_dir> [log_file]
#
# Example:
#   ./run_pipeline.sh \
#     /mnt/data1/jsolisle/cardioscar/synthetic_data/single_test/input_model.vtk \
#     /mnt/data1/jsolisle/cardioscar/synthetic_data/single_test/subset_013_n10_f2_s3.0_xy2p0mm_z8p0mm.nii.gz \
#     /mnt/data1/jsolisle/cardioscar/synthetic_data/single_test/outputs \
#     pipeline.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <mesh_vtk> <image_nifti> <output_dir> [log_file]" >&2
    exit 1
fi

MESH_VTK="$1"
IMAGE_NIFTI="$2"
OUTPUT_DIR="$3"
LOG_FILE="${4:-pipeline_$(date +%Y%m%d_%H%M%S).log}"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
IMAGE_STEM="$(basename "$IMAGE_NIFTI" .nii.gz)"
TRAINING_DATA="${OUTPUT_DIR}/${IMAGE_STEM}_training_data.npz"
MODEL_WEIGHTS="${OUTPUT_DIR}/${IMAGE_STEM}_model_weights.pth"
PREDICTION="${OUTPUT_DIR}/${IMAGE_STEM}_prediction.vtk"

# ---------------------------------------------------------------------------
# Setup: tee all output to log file
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "CardioScar Pipeline"
echo "Started: $(date)"
echo "============================================================"
echo "Mesh:        $MESH_VTK"
echo "Image:       $IMAGE_NIFTI"
echo "Output dir:  $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "------------------------------------------------------------"

# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------
echo ""
echo "[STAGE 1/3] Preparing training data..."
echo "$(date)"
cardioscar prepare \
    --mesh-vtk "$MESH_VTK" \
    --image "$IMAGE_NIFTI" \
    --output "$TRAINING_DATA"

echo "[STAGE 1/3] Done. Output: $TRAINING_DATA"

# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------
echo ""
echo "[STAGE 2/3] Training model..."
echo "$(date)"
cardioscar train \
    --training-data "$TRAINING_DATA" \
    --output "$MODEL_WEIGHTS" \
    --max-epochs 5000 \
    --early-stopping-patience 1000

echo "[STAGE 2/3] Done. Output: $MODEL_WEIGHTS"

# ---------------------------------------------------------------------------
# Stage 3: apply
# ---------------------------------------------------------------------------
echo ""
echo "[STAGE 3/3] Applying model..."
echo "$(date)"
cardioscar apply \
    --model "$MODEL_WEIGHTS" \
    --mesh "$MESH_VTK" \
    --output "$PREDICTION"

echo "[STAGE 3/3] Done. Output: $PREDICTION"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "  Training data: $TRAINING_DATA"
echo "  Model:         $MODEL_WEIGHTS"
echo "  Prediction:    $PREDICTION"
echo "  Log:           $LOG_FILE"
echo "============================================================"
