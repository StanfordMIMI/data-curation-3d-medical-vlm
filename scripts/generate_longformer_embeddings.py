#!/usr/bin/env python3
"""
Step 1b — Generate Clinical Longformer text embeddings.

Encodes radiology report findings using the ``yikuan8/Clinical-Longformer``
model (4096 token context).  The CLS token is extracted and L2-normalised.

Two on-disk layouts are available:

  --batch-size 0   one file per sample, ``{output_dir}/{sample_id}.pt``
                   (default)
  --batch-size N   N samples per file, ``{output_dir}/{prefix}_batch_{n}.pt``
                   (use for large pools such as the 58k CXR training pool)

Both layouts are read transparently by ``run_alignment.py``.

Resumable in either layout: samples already present are skipped.

Usage
-----
python scripts/generate_longformer_embeddings.py \\
    --metadata /data/pool_metadata.csv \\
    --output-dir /data/embeddings/longformer/ \\
    --id-column sample_id \\
    --text-column findings
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.embeddings import store
from rg_curation.embeddings.longformer import compute_longformer_embedding, setup_longformer_model
from rg_curation.utils.text_utils import get_memory_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Clinical Longformer text embeddings for a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="CSV file with at least an ID column and a text (findings) column.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where embedding files will be written.",
    )
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
    parser.add_argument(
        "--id-column",
        default="sample_id",
        help="Name of the unique sample identifier column.",
    )
    parser.add_argument(
        "--text-column",
        default="findings",
        help="Name of the radiology report text column.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum token length for Clinical Longformer (default: 4096).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(df)} records")

    for col in (args.id_column, args.text_column):
        if col not in df.columns:
            logger.error(f"Column '{col}' not found.  Available: {df.columns.tolist()}")
            sys.exit(1)

    # Drop rows with missing text (cannot produce a meaningful embedding).
    before = len(df)
    df = df.dropna(subset=[args.text_column])
    if before - len(df):
        logger.info(f"Dropped {before - len(df)} rows with missing '{args.text_column}'")

    # Skip samples that already have embeddings, in whichever layout is present.
    existing = store.existing_sample_ids(args.output_dir)
    before = len(df)
    df = df[~df[args.id_column].astype(str).isin(existing)]
    logger.info(f"Skipping {before - len(df)} already-processed samples.  "
                f"Remaining: {len(df)}")

    if df.empty:
        logger.info("All samples already processed.")
        return

    model, tokenizer, device = setup_longformer_model()

    writer = (
        store.BatchedEmbeddingWriter(
            args.output_dir, prefix=args.batch_prefix, batch_size=args.batch_size
        )
        if args.batch_size > 0
        else None
    )

    processed = errors = 0

    try:
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Longformer embeddings"
        ):
            sample_id = str(row[args.id_column])
            findings_text = row[args.text_column]

            try:
                emb = compute_longformer_embedding(
                    findings_text, model, tokenizer, device, max_length=args.max_length
                )
                record = {
                    "id": sample_id,
                    "embedding": emb,
                    "findings": str(findings_text),
                }
                if writer is not None:
                    writer.add(record)
                else:
                    store.save_per_sample(record, args.output_dir, sample_id)
                processed += 1

            except Exception as exc:
                logger.error(f"Error processing {sample_id}: {exc}")
                errors += 1
                continue

            if processed % 1000 == 0 and processed > 0:
                logger.info(f"Processed {processed} | RAM: {get_memory_usage():.0f} MB")
    finally:
        if writer is not None:
            writer.flush()

    logger.info(
        f"Done — processed: {processed}, errors: {errors}, "
        f"total attempted: {len(df)}"
    )
    logger.info(f"Embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
