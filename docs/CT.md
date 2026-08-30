# 3D CT Pipeline (Merlin / TV)

End-to-end walkthrough for the 3D abdominal CT experiments: curating a
training subset from the CT pool, training the CLIP-style 3D model
(I3ResNet-152 + Clinical Longformer) on it, and evaluating zero-shot
classification of 30 abdominal findings in and out of distribution.

The curation stages are the same code as the CXR pipeline —
`run_alignment.py`, `run_rrf.py`, and `select_top_n.py` are shared. Only the
metadata, the volume preprocessing, and the downstream model differ.

## Setup

| | |
|---|---|
| **Pool** (D₀) | *Merlin* regime: the Merlin Abdominal CT training split, 15k scan–report pairs. *Mixed* regime: Merlin training split + TV, 35k pairs (the paper's usual setting) |
| **Reference** (D_ref) | Merlin validation split, 5k pairs |
| **Budgets** | 10% (1,533), 30% (4,599), and 100% of the Merlin training split |
| **Model** | I3ResNet-152 (inflated from ImageNet) + Clinical Longformer |
| **ID test set** | Merlin test split, 5k scans |
| **OOD test set** | TV test split (a different institution) |

## Data format

One metadata CSV per dataset with at least:

| Column | Description |
|--------|-------------|
| `anon_accession` | Unique identifier for the scan–report pair |
| `image_file` | CT volume filename (`.nii.gz`; resolved by filename against `--image-dir`) |
| `findings` | Radiology report findings section |
| `split` | 0–2 = train, 3 = val, 4 = test |

Volumes are preprocessed on the fly with the fixed pipeline in
`rg_curation/utils/monai_transforms.py` (RAS orientation, 1.5 × 1.5 × 3 mm
spacing, HU clipped to [−1000, 1000] → [0, 1], padded/cropped to
224 × 224 × 160). Pass `--cache-dir` to every entry point so each volume is
preprocessed only once across all runs.

## Step 1 — Prepare the pool and reference

The pool is the training split(s), the reference is the Merlin validation
split:

```python
import pandas as pd

merlin = pd.read_csv("/data/merlin_labels.csv")
tv = pd.read_csv("/data/tv_labels.csv")

# Mixed 35k pool (use merlin alone for the single-source regime)
pool = pd.concat([
    merlin[merlin.split.isin([0, 1, 2])],
    tv[tv.split.isin([0, 1, 2])],
])
pool.to_csv("/data/curation/pool.csv", index=False)

merlin[merlin.split == 3].to_csv("/data/curation/reference.csv", index=False)
```

## Step 2 — Generate embeddings

This is the one-time cost; every curation configuration below reuses it.

```bash
# Vision — DreamSim over 10 axial slices per volume
for CSV in pool reference; do
python scripts/generate_dreamsim_embeddings.py \
    --metadata    /data/curation/${CSV}.csv \
    --image-dir   /data/ct_scans/ \
    --output-dir  /data/embeddings/dreamsim_ct/ \
    --modality    ct \
    --id-column   anon_accession
done

# Text — Clinical Longformer over the findings sections
for CSV in pool reference; do
python scripts/generate_longformer_embeddings.py \
    --metadata    /data/curation/${CSV}.csv \
    --output-dir  /data/embeddings/longformer_ct/ \
    --id-column   anon_accession
done
```

Both jobs are resumable: rerunning skips samples already present.

## Step 3 — Curate

Alignment (per modality), then rank fusion. The paper's headline
configuration is **vision k = 8, text k = 7**, with RRF η = 60 and equal
weights; single-modality k values between 5 and 15 perform similarly
(paper, Appendix on k-sensitivity).

```bash
python scripts/run_alignment.py \
    --pool-metadata /data/curation/pool.csv \
    --ref-metadata  /data/curation/reference.csv \
    --method        vision_only \
    --vision-embeddings-dir /data/embeddings/dreamsim_ct/ \
    --id-column     anon_accession \
    --k 8 \
    --output /data/curation/aligned_vision_k8.csv

python scripts/run_alignment.py \
    --pool-metadata /data/curation/pool.csv \
    --ref-metadata  /data/curation/reference.csv \
    --method        text_only \
    --text-embeddings-dir /data/embeddings/longformer_ct/ \
    --id-column     anon_accession \
    --k 7 \
    --output /data/curation/aligned_text_k7.csv

python scripts/run_rrf.py \
    --vision-scores /data/curation/aligned_vision_k8.csv \
    --text-scores   /data/curation/aligned_text_k7.csv \
    --id-column     anon_accession \
    --output        /data/curation/rrf_v8_t7.csv
```

The alignment output keeps every pool metadata column (including `split`),
so the curated CSV feeds the trainer directly — the training budget is
applied there as top-N by score, and no separate `select_top_n.py` call is
needed. For a single-modality run, pass the alignment CSV as the training
metadata instead.

## Step 4 — Train

```bash
TRAIN_METADATA=/data/curation/rrf_v8_t7.csv \
VAL_METADATA=/data/merlin_labels.csv \
IMAGE_DIR=/data/ct_scans_merlin/ \
IMAGE_DIR_2=/data/ct_scans_tv/ \
NUM_TRAIN_SAMPLES=4599 \
SAVE_DIR=/runs/ct-rrf-30pct \
CACHE_DIR=/data/ct_preproc_cache/ \
  sbatch scripts/slurm/ct_train.sbatch
```

Hyperparameters (defaults in the sbatch and `ct/train.py`, all overridable):

| | |
|---|---|
| Vision encoder | I3ResNet-152, inflated from ImageNet weights |
| Text encoder | `yikuan8/Clinical-Longformer`, CLS token → linear 768 → 512 |
| Objective | CLIP loss, learnable temperature initialized to log 10 |
| Batch size | 18 per GPU |
| Epochs | 100 |
| Optimiser | AdamW, lr 1e-5, weight decay 0, cosine schedule |
| Precision | fp16 |
| Validation | every 10 epochs (best + last checkpoints saved) |

The text side of each training batch alternates between the full findings
report and one organ-system section extracted from it (`ct/sections.py`),
exposing the model to both global and organ-level supervision.

The random-sampling baseline at the same budget uses the same command with
the uncurated pool CSV as `TRAIN_METADATA`; without a curation score column
the budget is applied uniformly at random (seeded).

## Step 5 — Evaluate

In distribution (Merlin test split):

```bash
CHECKPOINT=/runs/ct-rrf-30pct/clip3d_..._best_epoch_99.pt \
METADATA=/data/merlin_labels.csv \
IMAGE_DIR=/data/ct_scans_merlin/ \
OUTPUT=/runs/ct-rrf-30pct/zero_shot_merlin_test.json \
  sbatch scripts/slurm/ct_evaluate.sbatch
```

Out of distribution (TV test split): same command with the
TV metadata CSV and image directory.

### Protocol

Each of the 30 abdominal findings carries a bank of positive and negative
report phrases (`ct/prompts.py`) that serves double duty:

- **Labels.** The evaluation cohort for a finding is built by
  case-insensitive substring matching of those phrases against the test
  reports; the larger class is subsampled so the cohort is balanced.
- **Prompts.** The volume embedding is scored against the embedded positive
  and negative phrases, and the higher mean similarity wins — no decision
  threshold is tuned.

All findings with at least one positive and one negative example are
included. Metrics are per-finding F1 and the macro average, with 95%
confidence intervals from 1000-iteration bootstrap resampling (AUPRC and
AUROC on the continuous pos-minus-neg score are reported alongside).

## Notes

- **Checkpoint compatibility.** `ct.models.clip3d.Clip3D` uses the original
  training code's parameter names, so checkpoints from the paper's training
  runs load with `strict=True`.
- **Mixed pool.** Pass every volume directory via repeated `--image-dir`
  flags (or `IMAGE_DIR`/`IMAGE_DIR_2` in the sbatch); rows are matched to
  files by filename, so Merlin and TV volumes can live in separate
  trees.
