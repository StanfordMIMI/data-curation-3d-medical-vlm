#!/usr/bin/env python3
"""
Zero-shot evaluation of a trained CXR SigLIP checkpoint.

Runs the in-distribution CheXpert Plus test set and, when the three
``--mimic_*`` paths are given, the out-of-distribution MIMIC-CXR test set.
Metrics from both are merged and written to ``zero_shot_results.json`` in
``--output_dir``.

Usage
-----
python -m cxr.evaluate \\
    --checkpoint /runs/siglip-rrf-30pct \\
    --eval_data  /data/chexpertplus/test.jsonl \\
    --labels     /data/chexpertplus/test.json \\
    --split      test \\
    --output_dir /runs/siglip-rrf-30pct/eval_test_set \\
    --mimic_test_filenames_path /data/mimic-cxr/TEST_FILENAMES \\
    --mimic_cxr_root            /data/mimic-cxr/mimic-cxr-jpg/2.1.0 \\
    --mimic_chexpert_csv_path   /data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, set_seed

from cxr.data import CXRSiglipDataset, resolve_image_path
from cxr.models.siglip import VisionOnlySiglipModel, apply_interpolation_patch
from cxr.models.visual import TransformCXR
from cxr.zero_shot import (
    MimicChexpertEvalDataset,
    run_chexpert_zero_shot_eval,
    run_mimic_chexpert_zero_shot_eval,
)

apply_interpolation_patch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation for a trained CXR SigLIP checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--checkpoint", required=True, help="Directory holding the trained weights.")
    parser.add_argument("--vision_model", default="google/siglip-large-patch16-384")
    parser.add_argument("--text_model", default="google/siglip-large-patch16-384")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--n_queries", type=int, default=256)
    # In-distribution data
    parser.add_argument("--eval_data", required=True, help="Evaluation annotations (.jsonl).")
    parser.add_argument("--labels", required=True, help="CheXpert label JSONL.")
    parser.add_argument("--split", default="test", help="Split prefix to keep.")
    parser.add_argument("--image_key", default="image_path")
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--image_root", default=None, help="Prefix for relative image paths.")
    # Inference
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None, help="Where to write zero_shot_results.json.")
    # Out-of-distribution data (MIMIC-CXR)
    parser.add_argument("--mimic_test_filenames_path", default=None)
    parser.add_argument("--mimic_cxr_root", default=None)
    parser.add_argument("--mimic_chexpert_csv_path", default=None)
    parser.add_argument("--mimic_eval_unique_studies", action="store_true",
                        help="Evaluate one image per study rather than every image.")
    parser.add_argument("--mimic_eval_max_samples", type=int, default=None)
    parser.add_argument("--mimic_metric_prefix", default="mimic_test")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_dir: str) -> None:
    """Load trained weights from a Trainer output directory.

    Args:
        model: Freshly constructed model to populate.
        checkpoint_dir: Directory holding ``model.safetensors`` or
            ``pytorch_model.bin``.

    Raises:
        FileNotFoundError: If neither weight file is present.
    """
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
        )

    model.load_state_dict(state_dict, strict=True)


def _report(title: str, metrics: dict) -> None:
    """Print a metrics block in a stable, sorted order."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if not metrics:
        print("No metrics computed — check the overlap between the data and the labels.")
        return
    for key in sorted(metrics):
        print(f"{key:55s}: {metrics[key]:8.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    tokenizer.padding_side = "right"

    model = VisionOnlySiglipModel(
        vision_model_name_or_path=args.vision_model,
        text_model_name_or_path=args.text_model,
        image_size=args.image_size,
        n_queries=args.n_queries,
        projection_dim=None,
        freeze_text_encoder=True,
        temperature_init=0.07,
    )
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    if args.fp16:
        model = model.half()
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    eval_transform = TransformCXR(image_size=args.image_size, training=False)
    eval_dataset = CXRSiglipDataset(
        data_path=args.eval_data,
        tokenizer=tokenizer,
        image_transform=eval_transform,
        text_key=args.text_key,
        image_key=args.image_key,
        image_root=args.image_root,
        max_text_length=args.max_text_length,
    )
    print(f"In-distribution samples: {len(eval_dataset)}")

    chexpert_metrics = run_chexpert_zero_shot_eval(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        label_json_path=args.labels,
        split_prefix=args.split,
        batch_size=args.batch_size,
        max_text_length=args.max_text_length,
        metric_prefix="eval",
        resolve_image_path_fn=resolve_image_path,
    )
    _report("ZERO-SHOT RESULTS — CheXpert Plus (in distribution)", chexpert_metrics)

    mimic_metrics = {}
    if all((args.mimic_test_filenames_path, args.mimic_cxr_root, args.mimic_chexpert_csv_path)):
        try:
            mimic_dataset = MimicChexpertEvalDataset(
                filenames_path=args.mimic_test_filenames_path,
                mimic_root=args.mimic_cxr_root,
                labels_csv_path=args.mimic_chexpert_csv_path,
                image_transform=eval_transform,
                unique_studies=args.mimic_eval_unique_studies,
                max_samples=args.mimic_eval_max_samples,
            )
            print(f"\nOut-of-distribution samples: {len(mimic_dataset)}")
            mimic_metrics = run_mimic_chexpert_zero_shot_eval(
                model=model,
                tokenizer=tokenizer,
                mimic_eval_dataset=mimic_dataset,
                batch_size=args.batch_size,
                max_text_length=args.max_text_length,
                metric_prefix=args.mimic_metric_prefix,
            )
            _report("ZERO-SHOT RESULTS — MIMIC-CXR (out of distribution)", mimic_metrics)
        except Exception as exc:
            print(f"MIMIC-CXR evaluation failed: {exc}")

    all_metrics = {**chexpert_metrics, **mimic_metrics}
    if not all_metrics:
        print("\nNo metrics computed.")
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "zero_shot_results.json")
        with open(output_file, "w") as handle:
            json.dump(all_metrics, handle, indent=2)
        print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    main()
