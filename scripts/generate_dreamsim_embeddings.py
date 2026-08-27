#!/usr/bin/env python3
"""
Step 1a — Generate DreamSim embeddings for vision-guided alignment.

For 3D CT volumes (``--modality ct``) the encoder is applied to
``--num-slices`` center slices extracted with a center-of-mass crop.  Each
slice embedding is stored separately so that the alignment step can perform
slice-aligned cosine similarity.

For 2D chest X-rays (``--modality cxr``) the encoder is applied directly to
the full image and a single embedding is stored.  DICOM inputs are recognised
by their ``.dcm`` extension and rescaled to 8-bit RGB before encoding.

Two on-disk layouts are available:

  --batch-size 0   one file per sample, ``{output_dir}/{sample_id}.pt``
                   (default; convenient for the CT pool)
  --batch-size N   N samples per file, ``{output_dir}/{prefix}_batch_{n}.pt``
                   (use for large 2D pools — 58k separate files is hard on a
                   parallel filesystem)

Both layouts are read transparently by ``run_alignment.py``.

Resumable in either layout: samples already present are skipped.

Usage
-----
# Abdominal CT (3D) — uses the fixed transforms_image pipeline
python scripts/generate_dreamsim_embeddings.py \\
    --metadata /data/pool.csv \\
    --image-dir /data/ct_scans/ \\
    --output-dir /data/embeddings/dreamsim/ \\
    --modality ct

# Chest X-ray (2D), encoding the source DICOMs into batched files
python scripts/generate_dreamsim_embeddings.py \\
    --metadata /data/chexpert_pool.csv \\
    --image-dir /data/chexpertplus/DICOM/Uncompressed/ \\
    --image-column dcm_file \\
    --output-dir /data/embeddings/dreamsim_cxr/ \\
    --modality cxr \\
    --batch-size 1000 \\
    --batch-prefix pool
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.embeddings import store
from rg_curation.embeddings.dreamsim import (
    compute_dreamsim_embedding_2d,
    compute_dreamsim_embeddings_ct,
    setup_dreamsim_model,
)
from rg_curation.utils.ct_preprocessing import extract_center_slices, get_ct_transforms
from rg_curation.utils.cxr_preprocessing import load_cxr_image, resolve_image_path
from rg_curation.utils.text_utils import get_memory_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DreamSim embeddings for a medical imaging dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="CSV file with at least an ID column and an image path column.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing the image files referenced in --metadata.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where embedding files will be written.",
    )
    parser.add_argument(
        "--modality",
        choices=["ct", "cxr"],
        default="ct",
        help="Image modality.  'ct' expects .nii.gz volumes; 'cxr' expects 2D images or DICOMs.",
    )
    parser.add_argument(
        "--id-column",
        default="sample_id",
        help="Name of the unique sample identifier column in the metadata CSV.",
    )
    parser.add_argument(
        "--image-column",
        default="image_file",
        help="Name of the image path column.  Use 'dcm_file' to encode CheXpert Plus DICOMs.",
    )
    # Storage layout
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Embeddings per output file.  0 writes one file per sample.",
    )
    parser.add_argument(
        "--batch-prefix",
        default="embeddings",
        help="Filename prefix for batched output (ignored when --batch-size is 0).",
    )
    # CT-specific slice extraction
    parser.add_argument(
        "--num-slices",
        type=int,
        default=10,
        help="[CT only] Number of center slices to extract per volume.",
    )
    parser.add_argument(
        "--slice-spacing",
        type=int,
        default=2,
        help="[CT only] Spacing between consecutive extracted slices.",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.85,
        help="[CT only] Center-of-mass crop ratio (0 < ratio < 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(df)} records from {args.metadata}")

    for col in (args.id_column, args.image_column):
        if col not in df.columns:
            logger.error(f"Column '{col}' not found.  Available: {df.columns.tolist()}")
            sys.exit(1)

    # Skip samples that already have embeddings, in whichever layout is present.
    existing = store.existing_sample_ids(args.output_dir)
    before = len(df)
    df = df[~df[args.id_column].astype(str).isin(existing)]
    logger.info(
        f"Skipping {before - len(df)} samples (already processed). Remaining: {len(df)}"
    )

    if df.empty:
        logger.info("All samples already processed.")
        return

    model, preprocess, device = setup_dreamsim_model()

    if args.modality == "ct":
        transforms = get_ct_transforms()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    writer = (
        store.BatchedEmbeddingWriter(
            args.output_dir, prefix=args.batch_prefix, batch_size=args.batch_size
        )
        if args.batch_size > 0
        else None
    )

    processed = errors = 0

    try:
        for i, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="DreamSim embeddings")
        ):
            sample_id = str(row[args.id_column])
            raw_value = str(row[args.image_column])

            if args.modality == "ct":
                img_path = os.path.join(args.image_dir, os.path.basename(raw_value))
                if not os.path.isfile(img_path):
                    img_path = None
            else:
                img_path = resolve_image_path(args.image_dir, raw_value)

            if img_path is None:
                logger.warning(f"Image not found for {sample_id}: {raw_value}")
                errors += 1
                continue

            try:
                if args.modality == "ct":
                    volume = transforms({"image": img_path})["image"]
                    cropped_slices, slice_indices = extract_center_slices(
                        volume,
                        num_slices=args.num_slices,
                        slice_spacing=args.slice_spacing,
                        crop_ratio=args.crop_ratio,
                    )
                    if len(cropped_slices) == 0:
                        logger.warning(f"No slices extracted for {sample_id}")
                        errors += 1
                        continue
                    embs = compute_dreamsim_embeddings_ct(
                        cropped_slices, model, preprocess, device
                    )
                    views = [
                        {"embedding": embs[j], "slice_idx": int(slice_indices[j])}
                        for j in range(len(slice_indices))
                    ]
                else:
                    image = load_cxr_image(img_path)
                    emb = compute_dreamsim_embedding_2d(image, model, preprocess, device)
                    views = [{"embedding": emb[0], "slice_idx": 0}]

                if writer is not None:
                    # Batched entries are flat: one row per sample, so stack the
                    # per-view tensors into a single (S, D) tensor.
                    stacked = torch.stack([v["embedding"] for v in views])
                    writer.add(
                        {
                            "id": sample_id,
                            "embedding": stacked.squeeze(0) if len(views) == 1 else stacked,
                        }
                    )
                else:
                    store.save_per_sample(views, args.output_dir, sample_id)

                processed += 1

            except Exception as exc:
                logger.error(f"Error processing {sample_id}: {exc}")
                errors += 1
                continue

            if device.type == "cuda" and i % 20 == 0:
                torch.cuda.empty_cache()

            if processed % 500 == 0 and processed > 0:
                logger.info(f"Processed {processed} samples | RAM: {get_memory_usage():.0f} MB")
    finally:
        if writer is not None:
            writer.flush()

    logger.info(
        f"Done — processed: {processed}, errors/skipped: {errors}, "
        f"total attempted: {len(df)}"
    )
    logger.info(f"Embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
