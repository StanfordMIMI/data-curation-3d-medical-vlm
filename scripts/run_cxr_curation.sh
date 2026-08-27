#!/bin/bash
#
# Run the full CXR curation chain: vision alignment, text alignment, rank
# fusion, top-N selection, and export to training JSONL.
#
# Given precomputed embeddings this is CPU-only and takes well under an hour,
# so it is normally run on a login/interactive node rather than submitted.
#
# The k values below are the ones used for the paper's CXR experiments:
#
#   Budget   Method        k_vision   k_text
#   10%      vision-only        100        -
#   10%      text-only            -      125
#   10%      RRF                100      125
#   30%      vision-only        400        -
#   30%      text-only            -      400
#   30%      RRF                200      200
#
# Usage:
#   CURATION_DIR=/data/chexpertplus/curation \
#   VISION_EMB=/data/embeddings/dreamsim_cxr \
#   TEXT_EMB=/data/embeddings/longformer_cxr \
#   IMAGE_DIR=/data/chexpertplus/PNG \
#   K_VISION=200 K_TEXT=200 TOP_N=17402 TAG=rrf-30pct \
#     bash scripts/run_cxr_curation.sh

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${REPO_ROOT}"

CURATION_DIR=${CURATION_DIR:?set CURATION_DIR (holds pool.csv and reference.csv)}
VISION_EMB=${VISION_EMB:?set VISION_EMB to the DreamSim embedding directory}
TEXT_EMB=${TEXT_EMB:?set TEXT_EMB to the Clinical Longformer embedding directory}
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the rendered PNG/JPEG root}

K_VISION=${K_VISION:-200}
K_TEXT=${K_TEXT:-200}
TOP_N=${TOP_N:-17402}          # 30% of the 58,006-study pool
ETA=${ETA:-60}
W_VISION=${W_VISION:-1.0}
W_TEXT=${W_TEXT:-1.0}
TAG=${TAG:-rrf}
OUT=${OUT:-${CURATION_DIR}/${TAG}}

mkdir -p "${OUT}"

echo "==> Vision alignment (DreamSim, k=${K_VISION})"
python scripts/run_alignment.py \
    --pool-metadata "${CURATION_DIR}/pool.csv" \
    --ref-metadata  "${CURATION_DIR}/reference.csv" \
    --method vision_only \
    --vision-embeddings-dir "${VISION_EMB}" \
    --k "${K_VISION}" \
    --output "${OUT}/aligned_vision_k${K_VISION}.csv"

echo "==> Text alignment (Clinical Longformer, k=${K_TEXT})"
python scripts/run_alignment.py \
    --pool-metadata "${CURATION_DIR}/pool.csv" \
    --ref-metadata  "${CURATION_DIR}/reference.csv" \
    --method text_only \
    --text-embeddings-dir "${TEXT_EMB}" \
    --k "${K_TEXT}" \
    --output "${OUT}/aligned_text_k${K_TEXT}.csv"

echo "==> Reciprocal rank fusion (eta=${ETA}, w=${W_VISION}/${W_TEXT})"
python scripts/run_rrf.py \
    --vision-scores "${OUT}/aligned_vision_k${K_VISION}.csv" \
    --text-scores   "${OUT}/aligned_text_k${K_TEXT}.csv" \
    --output        "${OUT}/rrf.csv" \
    --eta "${ETA}" \
    --w-vision "${W_VISION}" \
    --w-text "${W_TEXT}"

echo "==> Selecting top ${TOP_N}"
python scripts/select_top_n.py \
    --input  "${OUT}/rrf.csv" \
    --output "${OUT}/curated_top${TOP_N}.csv" \
    --n "${TOP_N}" \
    --score-column rrf_score

echo "==> Exporting training JSONL"
python scripts/export_training_jsonl.py \
    --input  "${OUT}/curated_top${TOP_N}.csv" \
    --output "${OUT}/curated_top${TOP_N}.jsonl" \
    --image-dir "${IMAGE_DIR}" \
    --score-column rrf_score

echo
echo "Curated training set: ${OUT}/curated_top${TOP_N}.jsonl"
echo "Train with:  TRAIN_DATA=${OUT}/curated_top${TOP_N}.jsonl sbatch scripts/slurm/cxr_train.sbatch"
