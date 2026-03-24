#!/usr/bin/env python3
"""
Step 4 — Reciprocal Rank Fusion (RRF) over vision and text alignment scores.

Combines two independently-scored CSVs (produced by run_alignment.py with
``--method vision_only`` and ``--method text_only`` respectively) into a
single ranked list using the RRF formula:

    Score_RRF(x) = w1 / (η + r_V(x))  +  w2 / (η + r_T(x))

Samples present in only one list receive a contribution of 0 from the
missing modality (infinite rank penalty).

Output is a CSV sorted by ``rrf_score`` descending with added columns:
  rrf_score, final_rank, rank_vision, rank_text

Usage
-----
python scripts/run_rrf.py \\
    --vision-scores /data/aligned_vision_k5.csv \\
    --text-scores   /data/aligned_text_k5.csv \\
    --output        /data/rrf_k5.csv \\
    --top-n         10000
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.filtering.rrf import compute_rrf_scores, find_similarity_column

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reciprocal Rank Fusion of vision and text alignment scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vision-scores", required=True,
                        help="CSV with vision alignment scores (from run_alignment.py --method vision_only).")
    parser.add_argument("--text-scores", required=True,
                        help="CSV with text alignment scores (from run_alignment.py --method text_only).")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for RRF-ranked samples.")
    parser.add_argument("--id-column", default="sample_id",
                        help="Unique sample identifier column in both input CSVs.")
    parser.add_argument("--w-vision", type=float, default=1.0,
                        help="Weight for the vision ranking term.")
    parser.add_argument("--w-text", type=float, default=1.0,
                        help="Weight for the text ranking term.")
    parser.add_argument("--eta", type=int, default=60,
                        help="RRF smoothing constant η (default: 60).")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the top-N samples by RRF score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_vision = pd.read_csv(args.vision_scores)
    df_text = pd.read_csv(args.text_scores)
    logger.info(f"Vision scores: {len(df_vision)} samples | Text scores: {len(df_text)} samples")

    for df, name in ((df_vision, "vision"), (df_text, "text")):
        if args.id_column not in df.columns:
            logger.error(
                f"ID column '{args.id_column}' not found in {name} CSV. "
                f"Available: {df.columns.tolist()}"
            )
            sys.exit(1)

    try:
        vis_score_col = find_similarity_column(df_vision, "vision")
        txt_score_col = find_similarity_column(df_text, "text")
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    rrf_df = compute_rrf_scores(
        df_vision=df_vision,
        df_text=df_text,
        id_column=args.id_column,
        vision_score_col=vis_score_col,
        text_score_col=txt_score_col,
        w_vision=args.w_vision,
        w_text=args.w_text,
        eta=args.eta,
    )

    if args.top_n is not None and args.top_n < len(rrf_df):
        rrf_df = rrf_df.head(args.top_n).copy()
        logger.info(f"Filtered to top {args.top_n} samples by RRF score.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    rrf_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(rrf_df)} RRF-ranked samples to: {args.output}")

    # Show top-10 preview
    display_cols = [c for c in [args.id_column, "rrf_score", "final_rank",
                                 "rank_vision", "rank_text"] if c in rrf_df.columns]
    logger.info(f"\nTop 10 samples:\n{rrf_df[display_cols].head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
