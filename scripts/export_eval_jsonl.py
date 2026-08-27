#!/usr/bin/env python3
"""
Build the evaluation image list consumed by training and evaluation.

Zero-shot classification scores images against label prompts and never reads
the report text, so the evaluation annotations are simply the set of images
that have labels. This script derives that set from the CheXpert label file:
every entry with a ``path_to_image`` becomes one record

    {"image_path": "<image-dir>/<path_to_image>", "text": "<placeholder>"}

The placeholder keeps the record shape identical to the training JSONL so a
single dataset class reads both.

Usage
-----
# In-distribution test set
python scripts/export_eval_jsonl.py \\
    --labels    /data/chexpertplus/test.json \\
    --image-dir /data/chexpertplus/PNG/ \\
    --output    /data/chexpertplus/test.jsonl \\
    --split     test

# Validation set used for in-training zero-shot evaluation
python scripts/export_eval_jsonl.py \\
    --labels    /data/chexpertplus/findings_fixed.json \\
    --image-dir /data/chexpertplus/PNG/ \\
    --output    /data/chexpertplus/valid.jsonl \\
    --split     valid
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.utils.cxr_preprocessing import resolve_image_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evaluation image list from a CheXpert label file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="CheXpert label file: one JSON object per line with a 'path_to_image' key.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Root directory holding the rendered radiographs.",
    )
    parser.add_argument("--output", required=True, help="Destination .jsonl path.")
    parser.add_argument(
        "--split",
        default=None,
        help="Keep only paths under this split prefix (train/valid/test). "
             "Omit to keep every labelled image.",
    )
    parser.add_argument(
        "--placeholder-text",
        default="chest xray",
        help="Text stored in each record. Unused by the zero-shot protocol, but "
             "records without text are dropped by the dataset.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Drop entries whose image cannot be resolved instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = f"{args.split.strip().strip('/')}/" if args.split else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    total = kept = wrong_split = missing = 0
    seen = set()

    with open(args.labels, "r", encoding="utf-8") as src, \
         open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1

            rel_path = json.loads(line).get("path_to_image", "")
            if not rel_path:
                continue

            normalised = str(rel_path).replace("\\", "/").lstrip("/")
            if prefix and not normalised.lower().startswith(prefix.lower()):
                wrong_split += 1
                continue

            image_path = resolve_image_path(args.image_dir, normalised)
            if image_path is None:
                missing += 1
                if not args.skip_missing_images:
                    logger.error(
                        f"Image not found under {args.image_dir}: {rel_path}. "
                        "Pass --skip-missing-images to drop these entries instead."
                    )
                    sys.exit(1)
                continue

            if image_path in seen:
                continue
            seen.add(image_path)

            dst.write(
                json.dumps(
                    {"image_path": image_path, "text": args.placeholder_text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    logger.info(
        f"Read {total} label entries; wrote {kept} to {args.output} "
        f"({wrong_split} outside --split, {missing} without a resolvable image)"
    )
    if kept == 0:
        logger.error("No records written — check --image-dir and --split.")
        sys.exit(1)


if __name__ == "__main__":
    main()
