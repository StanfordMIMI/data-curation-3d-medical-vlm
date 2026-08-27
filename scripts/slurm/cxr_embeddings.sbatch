#!/bin/bash
#SBATCH --job-name=cxr_embeddings
#SBATCH --output=logs/cxr_embeddings_%j.out
#SBATCH --error=logs/cxr_embeddings_%j.err
#SBATCH --time=23:55:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#
# Encode the CheXpert Plus pool and reference set with DreamSim (vision) and
# Clinical Longformer (text). This is the one-time cost of the pipeline; every
# curation configuration afterwards reuses these embeddings.
#
# Embeddings are written in the batched layout (1000 per file), which keeps the
# 58k-study pool to ~60 files instead of ~58k.
#
#   ENCODER=dreamsim   CURATION_DIR=... IMAGE_DIR=... EMBEDDING_DIR=... \
#     sbatch scripts/slurm/cxr_embeddings.sbatch
#   ENCODER=longformer CURATION_DIR=... EMBEDDING_DIR=... \
#     sbatch scripts/slurm/cxr_embeddings.sbatch

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

ENCODER=${ENCODER:?set ENCODER to 'dreamsim' or 'longformer'}
CURATION_DIR=${CURATION_DIR:?set CURATION_DIR to the directory holding pool.csv and reference.csv}
EMBEDDING_DIR=${EMBEDDING_DIR:?set EMBEDDING_DIR to the embedding output directory}
BATCH_SIZE=${BATCH_SIZE:-1000}

case "${ENCODER}" in
  dreamsim)
    # Encodes the source DICOMs; point IMAGE_DIR at the DICOM root and use the
    # dcm_file column. Use --image-column image_file for rendered PNG/JPEG.
    IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the CheXpert Plus DICOM root}
    IMAGE_COLUMN=${IMAGE_COLUMN:-dcm_file}
    for SPLIT in pool reference; do
        python scripts/generate_dreamsim_embeddings.py \
            --metadata "${CURATION_DIR}/${SPLIT}.csv" \
            --image-dir "${IMAGE_DIR}" \
            --image-column "${IMAGE_COLUMN}" \
            --output-dir "${EMBEDDING_DIR}" \
            --modality cxr \
            --batch-size "${BATCH_SIZE}" \
            --batch-prefix "${SPLIT}"
    done
    ;;
  longformer)
    for SPLIT in pool reference; do
        python scripts/generate_longformer_embeddings.py \
            --metadata "${CURATION_DIR}/${SPLIT}.csv" \
            --output-dir "${EMBEDDING_DIR}" \
            --text-column findings \
            --batch-size "${BATCH_SIZE}" \
            --batch-prefix "${SPLIT}"
    done
    ;;
  *)
    echo "ENCODER must be 'dreamsim' or 'longformer', got '${ENCODER}'" >&2
    exit 1
    ;;
esac
