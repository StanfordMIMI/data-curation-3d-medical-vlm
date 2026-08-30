#!/usr/bin/env python3
"""Train the 3D CT CLIP model on a curated (or uncurated) training subset.

The subset is chosen entirely upstream by the curation pipeline — this script
reads whichever metadata CSV it is given (optionally applying a top-N budget
by curation score), so every budget and curation method is compared under an
identical recipe.

Usage
-----
accelerate launch --mixed_precision=fp16 -m ct.train \\
    --train-metadata /data/curated_rrf.csv \\
    --val-metadata   /data/pool_labels.csv \\
    --image-dir      /data/ct_scans_merlin/ \\
    --image-dir      /data/ct_scans_tv/ \\
    --num-train-samples 4599 \\
    --save-dir       /runs/ct \\
    ... (see scripts/slurm/ct_train.sbatch for the full invocation)
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from open_clip.loss import ClipLoss
from torch import optim

from ct import data as ct_data
from ct.models.clip3d import Clip3D
from ct.sections import pick_training_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-metadata",
        required=True,
        help="Training metadata CSV (curated subset or uncurated pool).",
    )
    parser.add_argument(
        "--val-metadata",
        required=True,
        help="Metadata CSV providing the validation split (split == 3).",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        required=True,
        dest="image_dirs",
        help="Directory with .nii.gz volumes. Repeat for the Mixed pool.",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Training budget: keep the top-N samples by curation score "
        "(uniform random for a CSV without a score column). Default: all rows.",
    )
    parser.add_argument("--cache-dir", default=None, help="Preprocessing cache directory.")
    parser.add_argument("--save-dir", required=True, help="Checkpoint output directory.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run identifier used in checkpoint filenames. Default: a timestamp.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=18, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--validate-every-epochs",
        type=int,
        default=10,
        help="Validation (and checkpointing) interval.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=17,
        help="Seed for the uniform random baseline's subsampling.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Log to this Weights & Biases project (stdout-only when unset).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, base_lr: float, warmup_steps: int, total_steps: int):
    """Per-step cosine learning-rate schedule with linear warmup.

    From mlfoundations/open_clip. Returns a callable applied once per
    optimizer step.
    """

    def _assign(new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _adjust(step):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = 0.5 * (1 + math.cos(math.pi * progress)) * base_lr
        _assign(lr)
        return lr

    return _adjust


def recall_at_1(logits: torch.Tensor) -> float:
    """Fraction of rows whose highest logit is the matching pair (diagonal)."""
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_epoch(
    model, train_loader, criterion, optimizer, scheduler, epoch, accelerator
):
    model.train()
    epoch_loss = 0.0
    image_recalls, text_recalls = [], []
    steps_per_epoch = len(train_loader)

    for step, batch in enumerate(train_loader):
        images = batch["image"]
        # One findings list per sample (the loader collates them transposed).
        findings_lists = [list(texts) for texts in zip(*batch["findings_list"])]
        texts = pick_training_texts(findings_lists, step)

        image_features, text_features, logits_per_image, logits_per_text, logit_scale = model(
            images, texts
        )
        loss = criterion(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )

        accelerator.backward(loss)
        optimizer.step()
        scheduler(step + 1 + epoch * steps_per_epoch)
        optimizer.zero_grad()

        epoch_loss += loss.item()
        image_recall = recall_at_1(logits_per_image.detach())
        text_recall = recall_at_1(logits_per_text.detach())
        image_recalls.append(image_recall)
        text_recalls.append(text_recall)

        accelerator.log(
            {
                "train/loss": epoch_loss / (step + 1),
                "train/image_recall": image_recall,
                "train/text_recall": text_recall,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        sys.stdout.flush()

    accelerator.log(
        {
            "train/epoch": epoch + 1,
            "train/epoch_loss": epoch_loss / steps_per_epoch,
            "train/epoch_image_recall": float(np.mean(image_recalls)),
            "train/epoch_text_recall": float(np.mean(text_recalls)),
        }
    )
    print(f"Epoch {epoch + 1}: average train loss {epoch_loss / steps_per_epoch:.4f}")


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, accelerator):
    model.eval()
    running_loss = 0.0
    image_recalls, text_recalls = [], []

    for step, batch in enumerate(val_loader):
        images = batch["image"]
        texts = batch["findings"]
        image_features, text_features, logits_per_image, logits_per_text, logit_scale = model(
            images, texts
        )
        loss = criterion(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )
        running_loss += loss.item()
        image_recalls.append(recall_at_1(logits_per_image))
        text_recalls.append(recall_at_1(logits_per_text))

    val_loss = running_loss / (step + 1)
    accelerator.log(
        {
            "val/epoch": epoch + 1,
            "val/loss": val_loss,
            "val/image_recall": float(np.mean(image_recalls)),
            "val/text_recall": float(np.mean(text_recalls)),
        }
    )
    print(f"Epoch {epoch + 1}: average validation loss {val_loss:.4f}")
    return val_loss


def save_checkpoint(model, accelerator, save_dir, run_name, tag, epoch):
    unwrapped = accelerator.unwrap_model(model)
    path = os.path.join(save_dir, f"{run_name}_{tag}_epoch_{epoch}.pt")
    accelerator.save(unwrapped.state_dict(), path)
    print(f"Saved {tag} checkpoint to {path}")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    run_name = args.run_name or "clip3d_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    os.makedirs(args.save_dir, exist_ok=True)

    train_df = ct_data.load_metadata(args.train_metadata, args.image_dirs, "train")
    train_df = ct_data.select_training_subset(
        train_df, args.num_train_samples, seed=args.sampling_seed
    )
    val_df = ct_data.load_metadata(args.val_metadata, args.image_dirs, "val")

    train_loader = ct_data.build_dataloader(
        ct_data.build_datalist(train_df, include_sections=True),
        batch_size=args.batch_size,
        shuffle=True,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )
    val_loader = ct_data.build_dataloader(
        ct_data.build_datalist(val_df),
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )

    model = Clip3D(init_logit_scale=math.log(10))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    # The auxiliary classification head takes no gradient from the
    # contrastive loss, hence find_unused_parameters.
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, static_graph=True
    )
    accelerator = Accelerator(
        log_with="wandb" if args.wandb_project else None,
        kwargs_handlers=[ddp_kwargs],
    )
    if args.wandb_project:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": run_name}},
        )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    scheduler = cosine_lr(
        optimizer,
        args.learning_rate,
        args.warmup_steps,
        len(train_loader) * args.epochs,
    )
    criterion = ClipLoss(
        local_loss=True,
        gather_with_grad=True,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    print(f"Run name: {run_name}")
    print(f"Config: {vars(args)}")
    print(f"Processes: {accelerator.num_processes}")

    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, accelerator
        )
        if (epoch + 1) % args.validate_every_epochs == 0:
            val_loss = validate(model, val_loader, criterion, epoch, accelerator)
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    model, accelerator, args.save_dir, run_name, "best", epoch
                )
            save_checkpoint(model, accelerator, args.save_dir, run_name, "last", epoch)

    accelerator.end_training()


if __name__ == "__main__":
    main()
