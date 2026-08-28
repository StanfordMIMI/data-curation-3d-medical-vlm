#!/bin/bash
#SBATCH --job-name=ct_train_clip3d
#SBATCH --output=logs/ct_train_%j.out
#SBATCH --error=logs/ct_train_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#
# Train the 3D CT CLIP model (I3ResNet-152 + Clinical Longformer) on a
# curated (or randomly sampled) subset.
#
# The hyperparameters below are the ones used for the CT experiments in the
# paper: per-device batch 18, 100 epochs, lr 1e-5 with a cosine schedule,
# fp16 mixed precision.
#
# Every variable can be overridden from the environment, e.g.
#   TRAIN_METADATA=/data/curated_rrf.csv \
#   NUM_TRAIN_SAMPLES=4599 \
#   SAVE_DIR=/runs/ct-rrf-30pct \
#   sbatch scripts/slurm/ct_train.sbatch
#
# Authentication: export HF_TOKEN and WANDB_API_KEY in your shell (or use
# `huggingface-cli login` / `wandb login`) before submitting. Never commit them.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

# ----- data -----------------------------------------------------------------
TRAIN_METADATA=${TRAIN_METADATA:?set TRAIN_METADATA to the curated training CSV}
VAL_METADATA=${VAL_METADATA:?set VAL_METADATA to the metadata CSV holding the validation split}
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the directory with .nii.gz volumes}
# Second image directory for the Mixed (Merlin + Tri-Valley) pool; leave
# empty for the single-source pool.
IMAGE_DIR_2=${IMAGE_DIR_2:-}
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR to the checkpoint output directory}
CACHE_DIR=${CACHE_DIR:-}
# Training budget: top-N by curation score (paper: 1533 = 10%, 4599 = 30% of
# the Merlin training pool). Leave empty to train on the full CSV.
NUM_TRAIN_SAMPLES=${NUM_TRAIN_SAMPLES:-}

# ----- model and optimisation ----------------------------------------------
BATCH_PER_GPU=${BATCH_PER_GPU:-18}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
RUN_NAME=${RUN_NAME:-}
WANDB_PROJECT=${WANDB_PROJECT:-}

GPUS_PER_NODE=${GPUS_PER_NODE:-2}

EXTRA_ARGS=()
[[ -n "${IMAGE_DIR_2}" ]] && EXTRA_ARGS+=(--image-dir "${IMAGE_DIR_2}")
[[ -n "${CACHE_DIR}" ]] && EXTRA_ARGS+=(--cache-dir "${CACHE_DIR}")
[[ -n "${NUM_TRAIN_SAMPLES}" ]] && EXTRA_ARGS+=(--num-train-samples "${NUM_TRAIN_SAMPLES}")
[[ -n "${RUN_NAME}" ]] && EXTRA_ARGS+=(--run-name "${RUN_NAME}")
[[ -n "${WANDB_PROJECT}" ]] && EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}")

accelerate launch \
    --num_processes "${GPUS_PER_NODE}" \
    --mixed_precision fp16 \
    -m ct.train \
    --train-metadata "${TRAIN_METADATA}" \
    --val-metadata "${VAL_METADATA}" \
    --image-dir "${IMAGE_DIR}" \
    --save-dir "${SAVE_DIR}" \
    --batch-size "${BATCH_PER_GPU}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    "${EXTRA_ARGS[@]}"
