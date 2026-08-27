#!/usr/bin/env python3
"""
Step 0 (CXR) — Turn the CheXpert Plus release into a pool / reference CSV pair.

Reads ``df_chexpert_plus_240401.csv``, keeps the studies with a usable
findings section, and writes the two CSVs the rest of the pipeline consumes:

  pool.csv        the ``train`` split — the uncurated pool D0 (58,006 studies)
  reference.csv   the ``valid`` split — the reference set D_ref

Both files carry the same canonical ``sample_id``, which is what lets the
vision and text rankings be fused later without reconciling identifiers.

Usage
-----
python scripts/prepare_chexpert_plus.py \\
    --metadata  /data/chexpertplus/df_chexpert_plus_240401.csv \\
    --output-dir /data/chexpertplus/curation/
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.datasets.chexpert_plus import load_chexpert_plus, split_pool_and_reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CheXpert Plus pool and reference CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to the released df_chexpert_plus_240401.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the generated pool.csv and reference.csv.",
    )
    parser.add_argument(
        "--pool-split",
        default="train",
        help="Split used as the uncurated pool (D0).",
    )
    parser.add_argument(
        "--reference-split",
        default="valid",
        help="Split used as the reference set (D_ref).",
    )
    parser.add_argument(
        "--keep-empty-findings",
        action="store_true",
        help="Keep studies whose findings section is missing or a placeholder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_chexpert_plus(args.metadata, require_findings=not args.keep_empty_findings)
    pool_df, ref_df = split_pool_and_reference(
        df, pool_split=args.pool_split, reference_split=args.reference_split
    )

    pool_path = os.path.join(args.output_dir, "pool.csv")
    ref_path = os.path.join(args.output_dir, "reference.csv")
    pool_df.to_csv(pool_path, index=False)
    ref_df.to_csv(ref_path, index=False)

    logger.info(f"Wrote pool      ({len(pool_df):>6} samples) to {pool_path}")
    logger.info(f"Wrote reference ({len(ref_df):>6} samples) to {ref_path}")


if __name__ == "__main__":
    main()
