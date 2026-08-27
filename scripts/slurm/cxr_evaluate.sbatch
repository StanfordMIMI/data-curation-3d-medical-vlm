#!/bin/bash
#SBATCH --job-name=cxr_eval_siglip
#SBATCH --output=logs/cxr_eval_%j.out
#SBATCH --error=logs/cxr_eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#
# Zero-shot evaluation of a trained checkpoint on the CheXpert Plus test set
# (in distribution) and the MIMIC-CXR test set (out of distribution).
#
#   CHECKPOINT=/runs/siglip-rrf-30pct \
#   sbatch scripts/slurm/cxr_evaluate.sbatch

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the trained model directory}
EVAL_DATA=${EVAL_DATA:?set EVAL_DATA to the CheXpert Plus test .jsonl}
ZERO_SHOT_LABELS=${ZERO_SHOT_LABELS:?set ZERO_SHOT_LABELS to the CheXpert test label .json}
ZERO_SHOT_SPLIT=${ZERO_SHOT_SPLIT:-test}
OUTPUT_DIR=${OUTPUT_DIR:-${CHECKPOINT}/eval_test_set}

MIMIC_TEST_FILENAMES=${MIMIC_TEST_FILENAMES:-}
MIMIC_CXR_ROOT=${MIMIC_CXR_ROOT:-}
MIMIC_CHEXPERT_CSV=${MIMIC_CHEXPERT_CSV:-}

VISION_MODEL=${VISION_MODEL:-google/siglip-large-patch16-384}
TEXT_MODEL=${TEXT_MODEL:-google/siglip-large-patch16-384}
IMAGE_SIZE=${IMAGE_SIZE:-512}
N_QUERIES=${N_QUERIES:-256}
BATCH_SIZE=${BATCH_SIZE:-8}

MIMIC_ARGS=()
if [[ -n "${MIMIC_TEST_FILENAMES}" && -n "${MIMIC_CXR_ROOT}" && -n "${MIMIC_CHEXPERT_CSV}" ]]; then
    MIMIC_ARGS=(
        --mimic_test_filenames_path "${MIMIC_TEST_FILENAMES}"
        --mimic_cxr_root "${MIMIC_CXR_ROOT}"
        --mimic_chexpert_csv_path "${MIMIC_CHEXPERT_CSV}"
        --mimic_metric_prefix mimic_test
    )
fi

python -m cxr.evaluate \
    --checkpoint "${CHECKPOINT}" \
    --vision_model "${VISION_MODEL}" \
    --text_model "${TEXT_MODEL}" \
    --eval_data "${EVAL_DATA}" \
    --labels "${ZERO_SHOT_LABELS}" \
    --split "${ZERO_SHOT_SPLIT}" \
    --image_size "${IMAGE_SIZE}" \
    --n_queries "${N_QUERIES}" \
    --batch_size "${BATCH_SIZE}" \
    --fp16 \
    --output_dir "${OUTPUT_DIR}" \
    "${MIMIC_ARGS[@]}"
