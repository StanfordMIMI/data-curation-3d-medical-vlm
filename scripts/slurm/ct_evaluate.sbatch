#!/bin/bash
#SBATCH --job-name=ct_evaluate_clip3d
#SBATCH --output=logs/ct_evaluate_%j.out
#SBATCH --error=logs/ct_evaluate_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#
# Zero-shot evaluation of a trained CT CLIP checkpoint over the 30 abdominal
# findings — the Merlin test split for in-distribution results, or the
# Tri-Valley metadata for out-of-distribution results.
#
# Every variable can be overridden from the environment, e.g.
#   CHECKPOINT=/runs/ct-rrf-30pct/clip3d_..._best_epoch_99.pt \
#   METADATA=/data/merlin_labels.csv \
#   IMAGE_DIR=/data/ct_scans_merlin/ \
#   sbatch scripts/slurm/ct_evaluate.sbatch

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the model .pt file}
METADATA=${METADATA:?set METADATA to the evaluation metadata CSV}
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the directory with .nii.gz volumes}
SPLIT=${SPLIT:-test}
CACHE_DIR=${CACHE_DIR:-}
BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT=${OUTPUT:-}

EXTRA_ARGS=()
[[ -n "${CACHE_DIR}" ]] && EXTRA_ARGS+=(--cache-dir "${CACHE_DIR}")
[[ -n "${OUTPUT}" ]] && EXTRA_ARGS+=(--output "${OUTPUT}")

python -m ct.evaluate \
    --checkpoint "${CHECKPOINT}" \
    --metadata "${METADATA}" \
    --image-dir "${IMAGE_DIR}" \
    --split "${SPLIT}" \
    --batch-size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"
