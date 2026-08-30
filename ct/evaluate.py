#!/usr/bin/env python3
"""Zero-shot evaluation of a trained CT CLIP checkpoint.

Evaluates all 30 abdominal findings on the requested split of a metadata CSV
— the Merlin test split for in-distribution results, or the TV
metadata for out-of-distribution results.

Usage
-----
python -m ct.evaluate \\
    --checkpoint /runs/ct/clip3d_..._best_epoch_99.pt \\
    --metadata   /data/merlin_labels.csv \\
    --image-dir  /data/ct_scans_merlin/ \\
    --split      test \\
    --output     /runs/ct/zero_shot_test.json
"""

import argparse
import json

import torch

from ct import data as ct_data
from ct.models.clip3d import Clip3D
from ct.zero_shot import zero_shot_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata CSV with anon_accession, image_file, findings, split.",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        required=True,
        dest="image_dirs",
        help="Directory with .nii.gz volumes. May be repeated.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Metadata split to evaluate on.",
    )
    parser.add_argument("--cache-dir", default=None, help="Preprocessing cache directory.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON results path.")
    return parser.parse_args()


def load_model(checkpoint_path: str) -> Clip3D:
    print(f"Loading checkpoint: {checkpoint_path}")
    model = Clip3D()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    model = load_model(args.checkpoint)
    metadata_df = ct_data.load_metadata(args.metadata, args.image_dirs, args.split)
    print(f"Evaluating {len(metadata_df)} volumes from the {args.split} split.")

    results = zero_shot_classification(
        model,
        metadata_df,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )

    print(f"\nSummary for checkpoint: {args.checkpoint}")
    if "macro_average" in results:
        macro = results["macro_average"]
        print(
            f"Average F1 Score: {macro['f1']:.4f} "
            f"[{macro['ci_lower']:.4f}, {macro['ci_upper']:.4f}]"
        )
    for disease in sorted(results):
        if disease != "macro_average":
            print(f"  {disease}: {results[disease]['f1']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
