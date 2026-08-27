#!/bin/bash
#SBATCH --job-name=cxr_train_siglip
#SBATCH --output=logs/cxr_train_%j.out
#SBATCH --error=logs/cxr_train_%j.err
#SBATCH --time=12:55:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#
# Train the CXR SigLIP model on a curated (or randomly sampled) subset.
#
# The hyperparameters below are the ones used for the CXR experiments in the
# paper: SigLIP-large at 512 px, effective batch 128 (64 per GPU x 2 GPUs),
# 10 epochs, lr 1e-5 with 1000 warmup steps and a cosine schedule, fp16.
#
# Every variable can be overridden from the environment, e.g.
#   TRAIN_DATA=/data/curated_rrf_top17400.jsonl \
#   OUTPUT_DIR=/runs/siglip-rrf-30pct \
#   sbatch scripts/slurm/cxr_train.sbatch
#
# Authentication: export HF_TOKEN and WANDB_API_KEY in your shell (or use
# `huggingface-cli login` / `wandb login`) before submitting. Never commit them.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

# ----- data -----------------------------------------------------------------
TRAIN_DATA=${TRAIN_DATA:?set TRAIN_DATA to the curated training .jsonl}
EVAL_DATA=${EVAL_DATA:?set EVAL_DATA to the CheXpert Plus validation .jsonl}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to the run directory}
ZERO_SHOT_LABELS=${ZERO_SHOT_LABELS:?set ZERO_SHOT_LABELS to the CheXpert label .json}
ZERO_SHOT_SPLIT=${ZERO_SHOT_SPLIT:-valid}

# Optional out-of-distribution evaluation during training.
MIMIC_VAL_FILENAMES=${MIMIC_VAL_FILENAMES:-}
MIMIC_CXR_ROOT=${MIMIC_CXR_ROOT:-}
MIMIC_CHEXPERT_CSV=${MIMIC_CHEXPERT_CSV:-}

# ----- model and optimisation ----------------------------------------------
VISION_MODEL=${VISION_MODEL:-google/siglip-large-patch16-384}
TEXT_MODEL=${TEXT_MODEL:-google/siglip-large-patch16-384}
IMAGE_SIZE=${IMAGE_SIZE:-512}
N_QUERIES=${N_QUERIES:-256}
BATCH_PER_GPU=${BATCH_PER_GPU:-64}
EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
# 0.01 for the curated and same-budget random-sampling runs; the 100% baseline
# was trained with 0.0.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-6036}

MIMIC_ARGS=()
if [[ -n "${MIMIC_VAL_FILENAMES}" && -n "${MIMIC_CXR_ROOT}" && -n "${MIMIC_CHEXPERT_CSV}" ]]; then
    MIMIC_ARGS=(
        --mimic_val_filenames_path "${MIMIC_VAL_FILENAMES}"
        --mimic_cxr_root "${MIMIC_CXR_ROOT}"
        --mimic_chexpert_csv_path "${MIMIC_CHEXPERT_CSV}"
    )
fi

torchrun \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m cxr.train \
    --train_data_path "${TRAIN_DATA}" \
    --eval_data_path "${EVAL_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --vision_model_name_or_path "${VISION_MODEL}" \
    --text_model_name_or_path "${TEXT_MODEL}" \
    --image_size "${IMAGE_SIZE}" \
    --n_queries "${N_QUERIES}" \
    --per_device_train_batch_size "${BATCH_PER_GPU}" \
    --per_device_eval_batch_size "${BATCH_PER_GPU}" \
    --gradient_accumulation_steps 1 \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --lr_scheduler_type cosine \
    --weight_decay "${WEIGHT_DECAY}" \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --fp16 True \
    --bf16 False \
    --remove_unused_columns False \
    --chexpert_zero_shot_labels "${ZERO_SHOT_LABELS}" \
    --chexpert_zero_shot_split "${ZERO_SHOT_SPLIT}" \
    "${MIMIC_ARGS[@]}" \
    --do_train True \
    --do_eval True
