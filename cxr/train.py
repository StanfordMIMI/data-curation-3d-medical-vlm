#!/usr/bin/env python3
"""
Train the CXR SigLIP model on a curated (or uncurated) training subset.

The subset is chosen entirely upstream by the curation pipeline — this script
just reads whichever annotation file it is given, so every budget and curation
method is compared under an identical recipe.

Zero-shot classification runs during training as a callback: CheXpert Plus
in-distribution, and optionally MIMIC-CXR out-of-distribution when the three
``--mimic_*`` paths are supplied.

Usage
-----
torchrun --nproc_per_node 2 -m cxr.train \\
    --train_data_path /data/curated_rrf_top17400.jsonl \\
    --eval_data_path  /data/chexpert_valid.jsonl \\
    --output_dir      /runs/siglip-rrf-30pct \\
    --chexpert_zero_shot_labels /data/chexpertplus/findings_fixed.json \\
    ... (see scripts/slurm/train_cxr_siglip.sbatch for the full invocation)
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from cxr.data import CXRSiglipDataset, SiglipDataCollator
from cxr.models.siglip import VisionOnlySiglipModel, apply_interpolation_patch
from cxr.models.visual import TransformCXR
from cxr.zero_shot import (
    MimicChexpertEvalCallback,
    MimicChexpertEvalDataset,
    ZeroShotEvalCallback,
    evaluate_with_zero_shot,
    run_mimic_chexpert_zero_shot_eval,
)

# SigLIP position embeddings are interpolated so the tower can run at a
# resolution other than the one it was pretrained at (512 vs. 384 here).
apply_interpolation_patch()


@dataclass
class VisionModelArguments:
    """Model configuration."""

    vision_model_name_or_path: str = field(
        default="google/siglip-large-patch16-384",
        metadata={"help": "Vision backbone to fine-tune."},
    )
    text_model_name_or_path: str = field(
        default="google/siglip-large-patch16-384",
        metadata={"help": "Text encoder paired with the vision tower."},
    )
    image_size: int = field(default=512, metadata={"help": "Input resolution for the ViT."})
    n_queries: int = field(default=256, metadata={"help": "Number of projection-head queries."})
    projection_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Projected embedding dim. Defaults to the text encoder hidden size."},
    )
    freeze_text_encoder: bool = field(default=False)
    temperature_init: float = field(
        default=0.07,
        metadata={"help": "Initial temperature for the contrastive loss."},
    )


@dataclass
class DataArguments:
    """Dataset configuration."""

    train_data_path: str = field(
        metadata={"help": "Curated training subset (.jsonl/.json/.csv)."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Validation annotations."}
    )
    image_key: str = field(default="image_path", metadata={"help": "Image path key."})
    text_key: str = field(default="text", metadata={"help": "Report text key."})
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Prefix for relative image paths."}
    )
    max_text_length: int = field(default=64, metadata={"help": "Max text tokens."})
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Cap on training samples."}
    )
    use_top_similarity_samples: bool = field(
        default=False,
        metadata={"help": "Apply the cap by similarity_score instead of uniformly."},
    )
    sampling_seed: int = field(default=17, metadata={"help": "Seed for uniform subsampling."})
    preload_images_into_memory: bool = field(
        default=False,
        metadata={"help": "Decode all images once at startup (faster epochs, more RAM)."},
    )
    # In-distribution zero-shot evaluation (CheXpert Plus)
    chexpert_zero_shot_labels: Optional[str] = field(
        default=None, metadata={"help": "Path to the CheXpert Plus label JSONL."}
    )
    chexpert_zero_shot_split: str = field(
        default="valid", metadata={"help": "Split prefix to keep (train/valid/test)."}
    )
    # Out-of-distribution zero-shot evaluation (MIMIC-CXR)
    mimic_val_filenames_path: Optional[str] = field(
        default=None,
        metadata={"help": "File listing one relative MIMIC-CXR jpg path per line."},
    )
    mimic_cxr_root: Optional[str] = field(
        default=None, metadata={"help": "Directory containing the MIMIC-CXR 'files/' tree."}
    )
    mimic_chexpert_csv_path: Optional[str] = field(
        default=None, metadata={"help": "Path to mimic-cxr-2.0.0-chexpert.csv."}
    )
    mimic_eval_unique_studies: bool = field(
        default=True,
        metadata={"help": "Evaluate one image per study, since labels are study-level."},
    )
    mimic_eval_max_samples: Optional[int] = field(
        default=None, metadata={"help": "Cap on MIMIC-CXR evaluation samples."}
    )
    mimic_metric_prefix: str = field(
        default="mimic", metadata={"help": "Metric name prefix for MIMIC-CXR results."}
    )


@dataclass
class VisionTrainingArguments(TrainingArguments):
    """Training arguments with the defaults this project needs."""

    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Required so Trainer keeps the custom tensors."}
    )


def compute_contrastive_metrics(eval_pred):
    """Image-to-text and text-to-image retrieval accuracy within each batch.

    The correct pairings lie on the diagonal of the similarity matrix, so a
    row (column) is correct when its argmax is its own index.

    Args:
        eval_pred: Trainer prediction output whose ``predictions`` hold the
            batch similarity logits.

    Returns:
        Dict with ``i2t_accuracy``, ``t2i_accuracy``, ``contrastive_accuracy``.
    """
    import numpy as np

    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    batch_size = logits.shape[0]
    if batch_size == 0:
        return {"i2t_accuracy": 0.0, "t2i_accuracy": 0.0, "contrastive_accuracy": 0.0}

    targets = np.arange(batch_size)
    i2t = float((np.argmax(logits, axis=1) == targets).sum() / batch_size)
    t2i = float((np.argmax(logits, axis=0) == targets).sum() / batch_size)

    return {
        "i2t_accuracy": i2t,
        "t2i_accuracy": t2i,
        "contrastive_accuracy": (i2t + t2i) / 2.0,
    }


class EvalLossCallback(TrainerCallback):
    """Compute a validation loss when the Trainer does not report one.

    The model returns a loss from ``forward``, but prediction-only evaluation
    loops can omit it. This recomputes the loss over a fixed-size prefix of the
    validation set and reduces it across ranks.

    Args:
        eval_dataset: Validation dataset.
        max_eval_samples: Number of validation samples to average over.
    """

    def __init__(self, eval_dataset, max_eval_samples: int = 200) -> None:
        self.eval_dataset = eval_dataset
        self.max_eval_samples = max_eval_samples
        self.trainer = None

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if metrics is None or self.eval_dataset is None or "eval_loss" in metrics:
            return

        import torch.distributed as dist
        from torch.utils.data import Subset

        model.eval()
        subset = (
            Subset(self.eval_dataset, range(self.max_eval_samples))
            if len(self.eval_dataset) > self.max_eval_samples
            else self.eval_dataset
        )

        total = torch.tensor(0.0, device=args.device)
        batches = torch.tensor(0.0, device=args.device)

        with torch.no_grad():
            for batch in self.trainer.get_eval_dataloader(subset):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                if isinstance(outputs, dict) and "loss" in outputs:
                    total += outputs["loss"].detach()
                    batches += 1.0

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(batches, op=dist.ReduceOp.SUM)

        if batches.item() > 0:
            metrics["eval_loss"] = (total / batches).item()
            if self.trainer.is_world_process_zero():
                self.trainer.log({"eval_loss": metrics["eval_loss"]})


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the most recently modified ``checkpoint-*`` directory, if any."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("checkpoint-")
    ]
    return sorted(checkpoints, key=os.path.getmtime)[-1] if checkpoints else None


def main() -> None:
    parser = HfArgumentParser((VisionModelArguments, DataArguments, VisionTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    tokenizer.padding_side = "right"

    train_transform = TransformCXR(image_size=model_args.image_size, training=True)
    eval_transform = TransformCXR(image_size=model_args.image_size, training=False)

    common = dict(
        tokenizer=tokenizer,
        max_text_length=data_args.max_text_length,
        image_key=data_args.image_key,
        text_key=data_args.text_key,
        image_root=data_args.image_root,
        preload_into_memory=data_args.preload_images_into_memory,
    )

    train_dataset = CXRSiglipDataset(
        data_path=data_args.train_data_path,
        image_transform=train_transform,
        max_samples=data_args.max_train_samples,
        use_top_similarity=data_args.use_top_similarity_samples,
        sampling_seed=data_args.sampling_seed,
        **common,
    )
    print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = CXRSiglipDataset(
            data_path=data_args.eval_data_path,
            image_transform=eval_transform,
            **common,
        )
    elif data_args.chexpert_zero_shot_labels:
        print(
            "chexpert_zero_shot_labels was set but eval_data_path was not; "
            "in-distribution zero-shot evaluation will be skipped."
        )

    mimic_eval_dataset = None
    if all(
        (
            data_args.mimic_val_filenames_path,
            data_args.mimic_cxr_root,
            data_args.mimic_chexpert_csv_path,
        )
    ):
        try:
            mimic_eval_dataset = MimicChexpertEvalDataset(
                filenames_path=data_args.mimic_val_filenames_path,
                mimic_root=data_args.mimic_cxr_root,
                labels_csv_path=data_args.mimic_chexpert_csv_path,
                image_transform=eval_transform,
                unique_studies=data_args.mimic_eval_unique_studies,
                max_samples=data_args.mimic_eval_max_samples,
            )
        except Exception as exc:
            print(f"Could not build the MIMIC-CXR eval dataset: {exc}")

    model = VisionOnlySiglipModel(
        vision_model_name_or_path=model_args.vision_model_name_or_path,
        text_model_name_or_path=model_args.text_model_name_or_path,
        image_size=model_args.image_size,
        n_queries=model_args.n_queries,
        projection_dim=model_args.projection_dim,
        freeze_text_encoder=model_args.freeze_text_encoder,
        temperature_init=model_args.temperature_init,
    )

    callbacks = []
    eval_loss_callback = None
    zero_shot_callback = None
    mimic_callback = None

    if training_args.do_train and eval_dataset is not None:
        eval_loss_callback = EvalLossCallback(eval_dataset)
        callbacks.append(eval_loss_callback)

    if eval_dataset is not None and data_args.chexpert_zero_shot_labels:
        zero_shot_callback = ZeroShotEvalCallback(
            eval_dataset, tokenizer, data_args, training_args
        )
        callbacks.append(zero_shot_callback)

    if mimic_eval_dataset is not None and training_args.do_train:
        mimic_callback = MimicChexpertEvalCallback(
            mimic_eval_dataset, tokenizer, data_args, training_args
        )
        callbacks.append(mimic_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SiglipDataCollator(),
        compute_metrics=compute_contrastive_metrics,
        callbacks=callbacks,
    )

    for callback in (eval_loss_callback, zero_shot_callback, mimic_callback):
        if callback is not None:
            callback.trainer = trainer

    if training_args.do_train:
        # Rank 0 picks the checkpoint and broadcasts it, so every rank resumes
        # from the same place.
        resume = training_args.resume_from_checkpoint
        if resume is None and training_args.local_rank in (-1, 0):
            resume = _latest_checkpoint(training_args.output_dir)
            if resume:
                print(f"Auto-resuming from {resume}")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            container = [resume]
            torch.distributed.broadcast_object_list(container, src=0)
            resume = container[0]

        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()

    elif training_args.do_eval and eval_dataset is not None:
        evaluate_with_zero_shot(
            trainer=trainer,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            metric_prefix="eval",
        )
        if mimic_eval_dataset is not None and trainer.is_world_process_zero():
            metrics = run_mimic_chexpert_zero_shot_eval(
                model=trainer.model,
                tokenizer=tokenizer,
                mimic_eval_dataset=mimic_eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                max_text_length=data_args.max_text_length,
                metric_prefix=data_args.mimic_metric_prefix or "mimic",
            )
            if metrics:
                trainer.log(metrics)


if __name__ == "__main__":
    main()
