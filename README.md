# Reference-Guided Data Curation for 3D Medical Vision-Language Pretraining

<p align="center">
  <img src="assets/method_rrf.png" width="60%" alt="Reference-guided curation pipeline: embedding generation, alignment, and RRF." />
</p>

This repository contains the data curation pipeline described in the paper
"Reference-Guided Data Curation for 3D Medical Vision-Language Pretraining",
together with the downstream training and evaluation code for both the 3D CT
and the chest X-ray experiments.

The pipeline filters a large, uncurated pool of medical scan–report pairs
(D₀) by aligning it to a small, high-quality reference set (D_ref),
producing a curated subset D_C ⊆ D₀ that maximises downstream
vision-language model performance.

The same curation code covers both modalities in the paper:

| Modality | Pool | Reference | Downstream model |
|---|---|---|---|
| **3D abdominal CT** | Merlin / mixed 35k | Merlin validation split | I3ResNet-152 + Clinical Longformer (included, see [`docs/CT.md`](docs/CT.md)) |
| **2D chest X-ray** | CheXpert Plus, 58k studies | CheXpert Plus validation split | SigLIP ViT (included, see [`docs/CXR.md`](docs/CXR.md)) |

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Format](#data-format)
- [Pipeline](#pipeline)
  - [Step 1 — Embedding Generation](#step-1--embedding-generation)
  - [Step 2 — CLIP Score Filtering (baseline)](#step-2--clip-score-filtering-baseline)
  - [Step 3 — Alignment-Based Filtering](#step-3--alignment-based-filtering)
  - [Step 4 — Reciprocal Rank Fusion](#step-4--reciprocal-rank-fusion)
  - [Step 5 — Select Top-N Samples](#step-5--select-top-n-samples)
- [3D CT Experiments](#3d-ct-experiments)
- [Chest X-ray Experiments](#chest-x-ray-experiments)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview

The curation strategy uses the reference set as an anchor for two independent
quality signals:

| Signal | Embedding model | Similarity measure |
|--------|----------------|--------------------|
| **Vision** | DreamSim (ensemble) | Slice-aligned cosine similarity |
| **Text** | Clinical Longformer | Cosine similarity |

The two signals are combined with **Reciprocal Rank Fusion (RRF)**:

$$\text{Score}_\text{RRF}(x) = \frac{w_1}{\eta + r_V(x)} + \frac{w_2}{\eta + r_T(x)}$$

where $r_V$, $r_T$ are the vision / text ranks, $\eta = 60$, and
$w_1 = w_2 = 1.0$ by default.

Four filtering methods are available:

| Method | Embeddings used |
|--------|----------------|
| `vision_only` | DreamSim |
| `text_only` | Clinical Longformer |
| `early_fusion` | DreamSim + Longformer (concatenated) |
| `rrf` | vision_only + text_only → RRF |

As a reference-free baseline, **CLIP-score filtering** selects samples by the
cosine similarity between the CLIP vision and text embeddings of the *same*
scan–report pair (no reference set required).

Embeddings are stored as `(S, D)` arrays, where `S` is the number of encoded
views per sample — one row per extracted axial slice for a CT volume, exactly
one row for a chest radiograph. Similarity is the average slice-aligned cosine
similarity, which reduces to plain cosine similarity when `S = 1`, so both
modalities run through the same alignment code.

---

## Installation

```bash
git clone https://github.com/faidrapts/rg-curation-3d-medical-vlm.git
cd rg-curation-3d-medical-vlm
pip install -r requirements.txt
```

---

## Data Format

Both the pool and reference CSVs must contain at minimum:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique identifier for the scan–report pair |
| `image_file` | Image path (relative to `--image-dir`, or a bare filename) |
| `findings` | Radiology report findings section |

The column names are configurable via `--id-column`, `--image-column`, and
`--text-column`.

**CT volumes** should be stored as `.nii.gz` files.
**CXR images** may be `.png`, `.jpg`, or `.dcm` (DICOM is decoded and rescaled
to 8-bit RGB automatically).

For CheXpert Plus, `scripts/prepare_chexpert_plus.py` generates both CSVs
directly from the released metadata.

---

## Pipeline

### Step 1 — Embedding Generation

Generate DreamSim (vision) and Clinical Longformer (text) embeddings for
both the pool and the reference set.

**DreamSim embeddings (CT)**

```bash
python scripts/generate_dreamsim_embeddings.py \
    --metadata    /data/pool.csv \
    --image-dir   /data/ct_scans/ \
    --output-dir  /data/embeddings/dreamsim/ \
    --modality    ct
```

CT volumes are preprocessed with the fixed `transforms_image` pipeline defined in
`rg_curation/utils/monai_transforms.py` (RAS orientation, 1.5 × 1.5 × 3 mm spacing,
HU clipped to [−1000, 1000] → [0, 1], padded/cropped to 224 × 224 × 160).

**DreamSim embeddings (CXR)**

```bash
python scripts/generate_dreamsim_embeddings.py \
    --metadata    /data/pool.csv \
    --image-dir   /data/cxr_images/ \
    --output-dir  /data/embeddings/dreamsim/ \
    --modality    cxr
```

**Longformer text embeddings**

```bash
python scripts/generate_longformer_embeddings.py \
    --metadata    /data/pool.csv \
    --output-dir  /data/embeddings/longformer/
```

> Run the same commands for the reference set, saving to the same embedding
> directories (sample IDs are used as the lookup key, so pool and reference
> embeddings coexist without conflict).

**Storage layout.** By default each sample is written to its own
`{sample_id}.pt`. For large pools, `--batch-size 1000` instead writes 1000
embeddings per file, keeping a 58k-sample pool to ~60 files rather than ~58k —
which matters on a parallel filesystem. Both layouts are detected and read
automatically downstream, and both are resumable.

---

### Step 2 — CLIP Score Filtering (baseline)

Compute a per-sample CLIP alignment score without using a reference set.

```bash
python scripts/compute_clip_scores.py \
    --metadata  /data/pool.csv \
    --image-dir /data/ct_scans/ \
    --output    /data/clip_scores.csv \
    --modality  ct
```

Then select the top-N samples (see [Step 5](#step-5--select-top-n-samples)).

---

### Step 3 — Alignment-Based Filtering

Run kNN alignment between the reference set and the pool.

**Vision-only**

```bash
python scripts/run_alignment.py \
    --pool-metadata  /data/pool.csv \
    --ref-metadata   /data/reference.csv \
    --method         vision_only \
    --vision-embeddings-dir /data/embeddings/dreamsim/ \
    --k   5 \
    --output /data/aligned_vision_k5.csv
```

**Text-only**

```bash
python scripts/run_alignment.py \
    --pool-metadata  /data/pool.csv \
    --ref-metadata   /data/reference.csv \
    --method         text_only \
    --text-embeddings-dir /data/embeddings/longformer/ \
    --k   5 \
    --output /data/aligned_text_k5.csv
```

**Early fusion (DreamSim + Longformer)**

```bash
python scripts/run_alignment.py \
    --pool-metadata  /data/pool.csv \
    --ref-metadata   /data/reference.csv \
    --method         early_fusion \
    --vision-embeddings-dir /data/embeddings/dreamsim/ \
    --text-embeddings-dir   /data/embeddings/longformer/ \
    --k   5 \
    --output /data/aligned_early_fusion_k5.csv
```

> The `--top-n` flag can be added to any `run_alignment.py` call to
> combine Steps 3 and 5 in one command.

Each reference sample contributes at most `k` neighbours, so the candidate
union is bounded by `(k_V + k_T) · |D_ref|`. Choose `k` so the deduplicated
union exceeds the target budget — smaller reference sets need larger `k`.

---

### Step 4 — Reciprocal Rank Fusion

Combine vision and text alignment scores with RRF.

```bash
python scripts/run_rrf.py \
    --vision-scores /data/aligned_vision_k5.csv \
    --text-scores   /data/aligned_text_k5.csv \
    --output        /data/rrf_k5.csv \
    --eta   60 \
    --w-vision 1.0 \
    --w-text   1.0
```

The output covers the union of both candidate sets. Metadata is taken from the
vision ranking and filled from the text ranking for samples only the text
ranking retrieved, so every row carries its image path and report text.

---

### Step 5 — Select Top-N Samples

Filter any scored CSV to the top-N samples:

```bash
# From RRF output
python scripts/select_top_n.py \
    --input        /data/rrf_k5.csv \
    --output       /data/curated_10k.csv \
    --n            10000 \
    --score-column rrf_score

# From alignment output
python scripts/select_top_n.py \
    --input        /data/aligned_vision_k5.csv \
    --output       /data/curated_10k.csv \
    --n            10000 \
    --score-column similarity_score

# From CLIP scores
python scripts/select_top_n.py \
    --input        /data/clip_scores.csv \
    --output       /data/curated_clip_10k.csv \
    --n            10000 \
    --score-column similarity_score
```

---

## 3D CT Experiments

The CT setting trains a CLIP-style 3D model — an I3ResNet-152 vision encoder
inflated from ImageNet weights paired with Clinical Longformer — on the
curated subset, and evaluates zero-shot classification of 30 abdominal
findings (in distribution on the Merlin test split, out of distribution on
Tri-Valley). The curation stages are the shared code above run with
`--modality ct`.

```bash
# 1. Pool and reference CSVs from the dataset metadata (splits 0-2 / 3)
# 2. Embeddings (one-time cost, reused by every configuration)
python scripts/generate_dreamsim_embeddings.py   ... --modality ct
python scripts/generate_longformer_embeddings.py ...

# 3. Curate: alignment (vision k=8, text k=7) -> RRF
python scripts/run_alignment.py ...
python scripts/run_rrf.py ...

# 4. Train (budget applied as top-N by curation score) and 5. evaluate
sbatch scripts/slurm/ct_train.sbatch
sbatch scripts/slurm/ct_evaluate.sbatch
```

**[`docs/CT.md`](docs/CT.md)** gives the full walkthrough, the data format,
the training hyperparameters, and the zero-shot evaluation protocol.

---

## Chest X-ray Experiments

The CXR setting adds metadata preparation, DICOM decoding, and a downstream
SigLIP ViT — the curation stages themselves are the shared code above.

```bash
# 1. Pool (58,006 studies) and reference set from the CheXpert Plus release
python scripts/prepare_chexpert_plus.py \
    --metadata   /data/chexpertplus/df_chexpert_plus_240401.csv \
    --output-dir /data/chexpertplus/curation/

# 2. Embeddings (one-time cost, reused by every configuration)
ENCODER=dreamsim   ... sbatch scripts/slurm/cxr_embeddings.sbatch
ENCODER=longformer ... sbatch scripts/slurm/cxr_embeddings.sbatch

# 3. Curate: alignment -> RRF -> top-N -> training JSONL
K_VISION=200 K_TEXT=200 TOP_N=17402 bash scripts/run_cxr_curation.sh

# 4. Build the evaluation lists (in and out of distribution)
python scripts/export_eval_jsonl.py ...
python scripts/prepare_mimic_filenames.py ...

# 5. Train and 6. evaluate
sbatch scripts/slurm/cxr_train.sbatch
sbatch scripts/slurm/cxr_evaluate.sbatch
```

**[`docs/CXR.md`](docs/CXR.md)** gives the full walkthrough, the per-budget `k`
values, the training hyperparameters, and the zero-shot evaluation protocol.

---

## Repository Structure

```
rg-curation-3d-medical-vlm/
│
├── rg_curation/                  # Curation library (modality-agnostic)
│   ├── embeddings/
│   │   ├── dreamsim.py           # DreamSim embedding generation (CT & CXR)
│   │   ├── longformer.py         # Clinical Longformer text embeddings
│   │   ├── clip_score.py         # CLIP vision–text alignment scores
│   │   └── store.py              # Batched / per-sample embedding I/O
│   ├── filtering/
│   │   ├── alignment.py          # kNN alignment (vision, text, early fusion)
│   │   └── rrf.py                # Reciprocal Rank Fusion
│   ├── datasets/
│   │   └── chexpert_plus.py      # CheXpert Plus metadata -> pool / reference
│   └── utils/
│       ├── ct_preprocessing.py   # CT slice extraction & center-mass crop
│       ├── monai_transforms.py   # MONAI transform pipeline for CT loading
│       ├── cxr_preprocessing.py  # DICOM decoding & image path resolution
│       ├── clip_utils.py         # Legacy CLIP text/image utilities (deprecated)
│       └── text_utils.py         # Text truncation & memory utilities
│
├── ct/                           # Downstream 3D CT vision-language model
│   ├── models/
│   │   ├── inflate.py            # 2D -> 3D kernel inflation (vendored, see NOTICE)
│   │   ├── i3res.py              # Inflated 3D ResNet vision encoder
│   │   └── clip3d.py             # Contrastive vision-language model
│   ├── data.py                   # Metadata CSVs -> MONAI datasets
│   ├── sections.py               # Organ-system report sectioning
│   ├── prompts.py                # Prompt bank for the 30 abdominal findings
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Zero-shot evaluation entry point
│   └── zero_shot.py              # 30-finding zero-shot protocol
│
├── cxr/                          # Downstream CXR vision-language model
│   ├── models/
│   │   ├── visual.py             # SigLIP vision tower (vendored, see NOTICE)
│   │   └── siglip.py             # Contrastive vision-language model
│   ├── data.py                   # Dataset and collator
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Zero-shot evaluation entry point
│   └── zero_shot.py              # CheXpert / MIMIC-CXR zero-shot protocol
│
├── scripts/                      # End-to-end pipeline scripts
│   ├── prepare_chexpert_plus.py
│   ├── generate_dreamsim_embeddings.py
│   ├── generate_longformer_embeddings.py
│   ├── compute_clip_scores.py
│   ├── run_alignment.py
│   ├── run_rrf.py
│   ├── select_top_n.py
│   ├── export_training_jsonl.py
│   ├── export_eval_jsonl.py      # Evaluation image list from a label file
│   ├── prepare_mimic_filenames.py # MIMIC-CXR split filename lists
│   ├── run_cxr_curation.sh       # Curation chain for the CXR setting
│   └── slurm/                    # SLURM submission scripts
│
├── docs/
│   ├── CT.md                     # 3D CT walkthrough
│   └── CXR.md                    # Chest X-ray walkthrough
│
├── examples/
│   ├── ct/                       # Example CT volumes (.nii.gz)
│   └── cxr/                      # Example CXR images (.png)
│
├── requirements.txt
├── NOTICE                        # Third-party attribution
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{patsatzi2026rgcurationvlm,
  title     = {Reference-Guided Data Curation for 3D Medical Vision-Language Pretraining},
  author    = {Patsatzi, Faidra Anastasia},
  url       = {https://github.com/faidrapts/rg-curation-3d-medical-vlm},
  year      = {2026},
}
```
