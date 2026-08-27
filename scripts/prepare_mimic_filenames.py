#!/usr/bin/env python3
"""
Build the MIMIC-CXR filename lists used for out-of-distribution evaluation.

``cxr.evaluate`` and the in-training MIMIC callback take a file listing one
relative JPG path per line. This script produces that list by intersecting the
release's ``IMAGE_FILENAMES`` manifest with the requested split(s) from
``mimic-cxr-2.0.0-split.csv``, matching on ``dicom_id``.

Note that CheXpert labels for MIMIC-CXR are study-level. Evaluation therefore
normally runs with one image per study (``--mimic_eval_unique_studies``); this
script emits every matching image and leaves that reduction to the evaluator.

Usage
-----
python scripts/prepare_mimic_filenames.py \\
    --split-csv       /data/mimic-cxr/mimic-cxr-2.0.0-split.csv \\
    --image-filenames /data/mimic-cxr/IMAGE_FILENAMES \\
    --splits validate \\
    --output          /data/mimic-cxr/VAL_FILENAMES

python scripts/prepare_mimic_filenames.py \\
    --split-csv       /data/mimic-cxr/mimic-cxr-2.0.0-split.csv \\
    --image-filenames /data/mimic-cxr/IMAGE_FILENAMES \\
    --splits test \\
    --output          /data/mimic-cxr/TEST_FILENAMES
"""

import argparse
import csv
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_STUDY_RE = re.compile(r"(p\d+)/(s\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the MIMIC-CXR image manifest down to one or more splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        help="Path to mimic-cxr-2.0.0-split.csv (columns: dicom_id, study_id, subject_id, split).",
    )
    parser.add_argument(
        "--image-filenames",
        required=True,
        help="Path to the IMAGE_FILENAMES manifest shipped with MIMIC-CXR-JPG.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validate"],
        help="Split names to keep, as they appear in the CSV (e.g. train validate test).",
    )
    parser.add_argument("--output", required=True, help="Destination filename list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted_splits = {s.strip().lower() for s in args.splits}

    dicom_ids = set()
    with open(args.split_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "dicom_id" not in reader.fieldnames:
            logger.error(f"{args.split_csv} has no 'dicom_id' column.")
            sys.exit(1)
        for row in reader:
            if (row.get("split") or "").strip().lower() in wanted_splits:
                dicom_ids.add(row["dicom_id"].strip())

    logger.info(f"Split(s) {sorted(wanted_splits)}: {len(dicom_ids)} dicom_ids")
    if not dicom_ids:
        logger.error("No rows matched --splits; check the split names in the CSV.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    total = kept = 0
    studies = set()
    with open(args.image_filenames, "r", encoding="utf-8") as src, \
         open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            # ".../p10/p10003502/s51180958/<dicom_id>.jpg"
            dicom_id = os.path.splitext(os.path.basename(line))[0]
            if dicom_id not in dicom_ids:
                continue
            dst.write(line + "\n")
            kept += 1
            match = _STUDY_RE.search(line)
            if match:
                studies.add(match.groups())

    logger.info(
        f"Scanned {total} manifest entries; wrote {kept} images "
        f"({len(studies)} unique studies) to {args.output}"
    )
    if kept == 0:
        logger.error("No manifest entries matched; check --image-filenames.")
        sys.exit(1)


if __name__ == "__main__":
    main()
