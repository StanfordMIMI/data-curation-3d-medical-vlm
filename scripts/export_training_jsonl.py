#!/usr/bin/env python3
"""
Step 6 (CXR) — Export a curated CSV to the JSONL format the trainer reads.

Takes any scored/curated CSV produced by the pipeline (alignment output, RRF
output, or a top-N selection) and writes one JSON object per line:

    {"sample_id": ..., "image_path": ..., "text": ..., "similarity_score": ...}

``image_path`` is resolved against ``--image-dir``, so the trainer never has
to know how the metadata stored its paths.  Rows without usable text or a
resolvable image are dropped and counted.

Usage
-----
python scripts/export_training_jsonl.py \\
    --input     /data/curated_rrf_top17400.csv \\
    --output    /data/curated_rrf_top17400.jsonl \\
    --image-dir /data/chexpertplus/PNG/
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.datasets.chexpert_plus import is_valid_findings
from rg_curation.utils.cxr_preprocessing import resolve_image_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a curated CSV to training JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Curated/scored CSV.")
    parser.add_argument("--output", required=True, help="Destination .jsonl path.")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Root directory holding the rendered PNG/JPEG radiographs.",
    )
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--image-column", default="image_file")
    parser.add_argument("--text-column", default="findings")
    parser.add_argument(
        "--score-column",
        default="rrf_score",
        help="Optional score column to carry through, if present in the CSV.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Drop rows whose image cannot be resolved instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    for col in (args.image_column, args.text_column):
        if col not in df.columns:
            logger.error(f"Column '{col}' not found.  Available: {df.columns.tolist()}")
            sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    written = no_text = no_image = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            text = row[args.text_column]
            if not is_valid_findings(text):
                no_text += 1
                continue

            raw_image = row[args.image_column]
            if pd.isna(raw_image):
                no_image += 1
                continue

            image_path = resolve_image_path(args.image_dir, str(raw_image))
            if image_path is None:
                no_image += 1
                if not args.skip_missing_images:
                    logger.error(
                        f"Image not found under {args.image_dir}: {raw_image}. "
                        "Pass --skip-missing-images to drop these rows instead."
                    )
                    sys.exit(1)
                continue

            record = {"image_path": image_path, "text": str(text).strip()}
            if args.id_column in df.columns:
                record["sample_id"] = str(row[args.id_column])
            if args.score_column in df.columns and pd.notna(row[args.score_column]):
                record["similarity_score"] = float(row[args.score_column])

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        f"Wrote {written} records to {args.output} "
        f"(dropped {no_text} without text, {no_image} without a resolvable image)"
    )


if __name__ == "__main__":
    main()
