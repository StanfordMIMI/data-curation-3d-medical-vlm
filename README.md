# Reference-Guided Data Curation for 3D Medical Vision-Language Pretraining

Code for the paper *"Reference-Guided Data Curation for 3D Medical
Vision-Language Pretraining"*. A large, uncurated pool of medical
scan–report pairs is filtered by kNN-aligning it to a small, high-quality
reference set; the curated subset then trains a vision-language model. The
repository contains the modality-agnostic curation pipeline and the
downstream training and zero-shot evaluation code for both settings in the
paper: **3D abdominal CT** and **2D chest X-ray**.

## Installation

```bash
git clone https://github.com/faidrapts/rg-curation-3d-medical-vlm.git
cd rg-curation-3d-medical-vlm
pip install -r requirements.txt
```

## Getting started

Pool and reference CSVs need three columns: a sample id, `image_file`, and
`findings` (names configurable via `--id-column`, `--image-column`,
`--text-column`). The pipeline is embeddings → alignment → rank fusion →
train → evaluate:

```bash
# 1. Embeddings for the pool and the reference set (run for both CSVs)
python scripts/generate_dreamsim_embeddings.py \
    --metadata pool.csv --image-dir /data/images/ \
    --output-dir emb/vision/ --modality ct        # or: cxr
python scripts/generate_longformer_embeddings.py \
    --metadata pool.csv --output-dir emb/text/

# 2. kNN-align the pool to the reference, per modality
python scripts/run_alignment.py --method vision_only --k 8 \
    --pool-metadata pool.csv --ref-metadata reference.csv \
    --vision-embeddings-dir emb/vision/ --output aligned_vision.csv
python scripts/run_alignment.py --method text_only --k 7 \
    --pool-metadata pool.csv --ref-metadata reference.csv \
    --text-embeddings-dir emb/text/ --output aligned_text.csv

# 3. Fuse the two rankings (Reciprocal Rank Fusion)
python scripts/run_rrf.py --vision-scores aligned_vision.csv \
    --text-scores aligned_text.csv --output curated.csv

# 4. Train on the curated subset and 5. evaluate zero-shot (CT shown)
accelerate launch --mixed_precision=fp16 -m ct.train \
    --train-metadata curated.csv --num-train-samples 4599 ...
python -m ct.evaluate --checkpoint <run>/..._best_epoch_99.pt ...
```

Other filtering methods: `--method early_fusion` (concatenated embeddings),
`scripts/compute_clip_scores.py` + `scripts/select_top_n.py` (reference-free
CLIP-score baseline), and omitting the score column entirely (seeded random
baseline at the same budget).

The full walkthroughs with the paper's exact configurations, data
preparation, SLURM scripts, and evaluation protocols:

- **[`docs/CT.md`](docs/CT.md)** — 3D CT: I3ResNet-152 + Clinical Longformer
- **[`docs/CXR.md`](docs/CXR.md)** — chest X-ray: SigLIP ViT on CheXpert Plus

## Model weights

Trained CT checkpoints are available on the Hugging Face Hub
([`faidrapts/rg-curation-3d-medical-vlm`](https://huggingface.co/faidrapts/rg-curation-3d-medical-vlm)):

```python
import torch
from huggingface_hub import hf_hub_download
from ct.models.clip3d import Clip3D

path = hf_hub_download(
    "faidrapts/rg-curation-3d-medical-vlm", "clip3d_rrf_mixed35k_30pct.pt"
)
model = Clip3D()
model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
```

Pass a downloaded checkpoint to `python -m ct.evaluate --checkpoint <path>`.

## Repository structure

```
rg_curation/       # Curation library (modality-agnostic):
                   #   embeddings (DreamSim, Longformer, CLIP score),
                   #   kNN alignment, RRF, CT/CXR preprocessing
ct/                # Downstream 3D CT model, training, zero-shot evaluation
cxr/               # Downstream CXR model, training, zero-shot evaluation
scripts/           # Pipeline entry points and SLURM submission scripts
docs/              # CT.md and CXR.md walkthroughs
examples/          # Small example inputs (pool/reference CSVs, CXR images)
```

## Citation

```bibtex
@inproceedings{patsatzi2026rgcurationvlm,
  title     = {Reference-Guided Data Curation for 3D Medical Vision-Language Pretraining},
  author    = {Patsatzi, Faidra Anastasia},
  url       = {https://github.com/faidrapts/rg-curation-3d-medical-vlm},
  year      = {2026},
}
```
