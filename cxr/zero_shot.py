"""
Zero-shot classification evaluation for CheXpert Plus and MIMIC-CXR.

Each finding is scored by comparing the image embedding against a positive and
a negative text prompt, and the higher similarity wins, so no decision
threshold is tuned. CheXpert Plus provides the in-distribution test set;
MIMIC-CXR provides the out-of-distribution one, with study-level CheXpert
labels matched via ``study_id``. Reported metrics are macro-averaged over the
findings that have both positive and negative support, with 95% confidence
intervals from 1000-iteration bootstrap resampling.

Uncertain CheXpert labels (-1) are treated as negative.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import TrainerCallback
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from cxr.data import resolve_image_path

ZERO_SHOT_DEBUG = str(os.environ.get("RG_CURATION_ZERO_SHOT_DEBUG", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}




# ========== Data Utilities ==========

def normalize_label_key(path: str) -> str:
    """Normalize CheXpert image path to match label lookup format."""
    path_norm = path.replace("\\", "/").lower()
    for marker in ("/train/", "/valid/", "/test/"):
        idx = path_norm.find(marker)
        if idx != -1:
            path_norm = path_norm[idx + 1:]
            break
    if not any(path_norm.startswith(prefix) for prefix in ("train/", "valid/", "test/")):
        parts = path_norm.split("/")
        if len(parts) >= 2:
            path_norm = "/".join(parts[-2:])
        else:
            path_norm = parts[-1]
    path_norm = os.path.splitext(path_norm)[0]
    return path_norm.lstrip("/")


def format_zero_shot_prompt(label: str) -> str:
    """Convert label name to natural language prompt for zero-shot classification."""
    name = label.lower()
    if name in {"no finding", "no_finding"}:
        return "A normal chest X-ray without abnormal findings."
    return f"A chest X-ray showing {name}."


def format_zero_shot_prompt_pair(label: str) -> tuple[str, str]:
    """Return positive/negative prompts for a label."""
    name = label.lower()
    if name in {"no finding", "no_finding"}:
        pos = "A normal chest X-ray without abnormal findings."
        neg = "A chest X-ray showing abnormal findings."
        return pos, neg
    pos = f"A chest X-ray showing {name}."
    neg = f"A chest X-ray without {name}."
    return pos, neg


ZERO_SHOT_TEMPLATE_PAIRS = [
    ("{}", "no {}"),
]

ZERO_SHOT_BOOTSTRAP_SAMPLES = 1000


def load_chexpert_label_lookup(label_path: str, split_prefix: Optional[str] = None):
    """
    Load CheXpert Plus labels from JSONL file.
    
    Returns:
        tuple: (label_names, lookup_dict) where lookup_dict maps image_path -> label_vector
    """
    label_names = None
    lookup: Dict[str, List[Optional[float]]] = {}
    split_prefix = (split_prefix or "").strip().lower().rstrip("/")
    
    with open(label_path, "r") as handle:
        for line in handle:
            record = json.loads(line)
            rel_key = normalize_label_key(record.get("path_to_image", ""))
            if not rel_key:
                continue
            # No split-prefix filtering (allow any split)
            
            if label_names is None:
                label_names = [k for k in record.keys() if k != "path_to_image"]
            
            label_vector: List[Optional[float]] = []
            has_value = False
            for name in label_names:
                value = record.get(name)
                try:
                    as_float = None if value is None else float(value)
                except (TypeError, ValueError):
                    as_float = None
                if as_float is not None and as_float < 0:
                    as_float = 0.0  # Treat uncertain (-1) as negative (0.0)
                if as_float is None:
                    label_vector.append(None)
                else:
                    label_vector.append(as_float)
                    has_value = True
            
            if has_value:
                lookup[rel_key] = label_vector
    
    if label_names is None:
        raise ValueError(f"No labels found in {label_path}")
    
    if ZERO_SHOT_DEBUG:
        print(f"DEBUG: Loaded {len(lookup)} label entries from {label_path}")
        print(f"DEBUG: First 3 label keys:")
        for i, key in enumerate(list(lookup.keys())[:3]):
            print(f"  {i}: {key}")
    
    return label_names, lookup


# ========== Dataset ==========

class ChexpertZeroShotDataset(Dataset):
    """Wraps a base dataset with CheXpert labels for zero-shot evaluation."""
    
    def __init__(self, base_dataset, indices, labels, masks):
        self.base_dataset = base_dataset
        self.indices = indices
        self.labels = torch.stack(labels)
        self.masks = torch.stack(masks)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[self.indices[idx]]
        return {
            "pixel_values": sample["pixel_values"],
            "labels": self.labels[idx],
            "label_mask": self.masks[idx],
        }


def build_zero_shot_dataset(eval_dataset, label_lookup, resolve_image_path_fn):
    """
    Build zero-shot dataset by matching eval samples with CheXpert labels.
    
    Args:
        eval_dataset: Base dataset containing images
        label_lookup: Dict mapping normalized paths to label vectors
        resolve_image_path_fn: Function to convert image paths to normalized format
    
    Returns:
        ChexpertZeroShotDataset or None if no matches found
    """
    indices = []
    label_tensors = []
    mask_tensors = []
    
    for idx, sample in enumerate(eval_dataset.samples):
        image_path = sample.get(eval_dataset.image_key)
        if image_path is None:
            continue
        resolved_path = resolve_image_path_fn(image_path)
        key = normalize_label_key(resolved_path)
        label_values = label_lookup.get(key)
        if label_values is None:
            continue
        
        mask = [0.0 if value is None else 1.0 for value in label_values]
        if sum(mask) == 0:
            continue
        # For labels: None/masked values can be any placeholder (they're excluded by mask)
        # 0.0 = negative (finding absent), 1.0 = positive (finding present)
        labels = []
        for value in label_values:
            if value is None:
                labels.append(0.0)
                continue
            as_float = float(value)
            if as_float < 0.0:
                as_float = 0.0  # Treat CheXpert uncertain (-1) as negative for eval
            labels.append(as_float)
        
        indices.append(idx)
        label_tensors.append(torch.tensor(labels, dtype=torch.float32))
        mask_tensors.append(torch.tensor(mask, dtype=torch.float32))
    
    if not indices:
        return None
    return ChexpertZeroShotDataset(eval_dataset, indices, label_tensors, mask_tensors)


# ========== MIMIC-CXR (CheXpert Labels, matched by study_id) ==========

def parse_mimic_patient_study_from_relpath(rel_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse patient_id and study_id from a MIMIC-CXR relative path like:
      files/p10/p10003502/s51180958/<dicom>.jpg

    Returns:
      (patient_id, study_id) without 'p'/'s' prefixes.
    """
    if not rel_path:
        return None, None
    parts = str(rel_path).replace("\\", "/").strip("/").split("/")

    # Fast-path matching the common layout used in your alignment scripts.
    if len(parts) >= 4 and parts[2].startswith("p") and parts[3].startswith("s"):
        patient_id = parts[2][1:]
        study_id = parts[3][1:]
        return patient_id or None, study_id or None

    # Fallback: search segments.
    patient_id = None
    study_id = None
    for p in parts:
        if patient_id is None and p.startswith("p") and p[1:].isdigit() and len(p) >= 5:
            # Exclude shard folders like "p10" by preferring longer IDs (e.g. p10003502).
            if len(p) >= 7:
                patient_id = p[1:]
        if study_id is None and p.startswith("s") and p[1:].isdigit():
            study_id = p[1:]
        if patient_id is not None and study_id is not None:
            break
    return patient_id, study_id


def load_mimic_chexpert_study_label_lookup(csv_path: str) -> tuple[List[str], Dict[str, List[Optional[float]]]]:
    """
    Load MIMIC-CXR CheXpert labels CSV and return a lookup dict by study_id (string).

    Notes:
    - Uncertain (-1) is treated as missing (None) and masked out for evaluation.
    - Missing values are represented as None and masked out.
    """
    df = pd.read_csv(csv_path, dtype={"study_id": "string"})
    if "study_id" not in df.columns:
        raise ValueError(f"Expected 'study_id' column in {csv_path}")

    label_cols = [c for c in df.columns if c not in {"subject_id", "study_id"}]
    if not label_cols:
        raise ValueError(f"No label columns found in {csv_path}")

    lookup: Dict[str, List[Optional[float]]] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        sid = row_dict.get("study_id")
        if sid is None:
            continue
        sid = str(sid).strip()
        if not sid:
            continue

        vec: List[Optional[float]] = []
        has_value = False
        for c in label_cols:
            v = row_dict.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vec.append(None)
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                vec.append(None)
                continue
            if fv < 0:
                # Treat uncertain (-1) as missing for MIMIC-CXR evaluation
                vec.append(None)
                continue
            vec.append(float(fv))
            has_value = True
        if has_value:
            lookup[sid] = vec

    return label_cols, lookup


class MimicChexpertEvalDataset(Dataset):
    """
    Image-only dataset for MIMIC-CXR validation with CheXpert labels matched by study_id.
    Returns pixel_values + (labels, label_mask) for zero-shot classification evaluation.
    """

    def __init__(
        self,
        filenames_path: str,
        mimic_root: str,
        labels_csv_path: str,
        image_transform,
        unique_studies: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.mimic_root = str(Path(mimic_root).expanduser())
        if not os.path.isdir(self.mimic_root):
            raise FileNotFoundError(f"mimic_cxr_root does not exist or is not a directory: {self.mimic_root}")

        self.label_names, self._label_lookup = load_mimic_chexpert_study_label_lookup(labels_csv_path)
        self.image_transform = image_transform

        with open(filenames_path, "r", encoding="utf-8") as fp:
            rel_paths = [line.strip() for line in fp if line.strip()]

        samples = []
        seen_studies = set()
        skipped_no_labels = 0
        skipped_bad_path = 0
        for rel in rel_paths:
            _, study_id = parse_mimic_patient_study_from_relpath(rel)
            if not study_id:
                skipped_bad_path += 1
                continue
            if unique_studies:
                if study_id in seen_studies:
                    continue
                seen_studies.add(study_id)

            label_vec = self._label_lookup.get(str(study_id))
            if label_vec is None:
                skipped_no_labels += 1
                continue
            mask = [0.0 if v is None else 1.0 for v in label_vec]
            if sum(mask) == 0:
                skipped_no_labels += 1
                continue

            abs_path = os.path.join(self.mimic_root, rel)
            samples.append(
                {
                    "image_path": abs_path,
                    "study_id": str(study_id),
                    "labels": label_vec,
                    "mask": mask,
                }
            )
            if max_samples is not None and len(samples) >= int(max_samples):
                break

        self.samples = samples
        print(
            f"MIMIC-CXR eval: built {len(self.samples)} samples "
            f"(unique_studies={unique_studies}, max_samples={max_samples}). "
            f"Skipped {skipped_no_labels} without labels, {skipped_bad_path} with unparseable paths."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # IMPORTANT: MIMIC-CXR images are stored as .jpg on disk (do NOT apply CheXpert .jpg→.png mapping).
        image_path = os.path.expandvars(os.path.expanduser(sample["image_path"]))
        if not os.path.exists(image_path):
            # Be slightly forgiving if upstream code accidentally swapped extensions.
            root, ext = os.path.splitext(image_path)
            if ext.lower() == ".png":
                alt = root + ".jpg"
            elif ext.lower() == ".jpg":
                alt = root + ".png"
            else:
                alt = None
            if alt and os.path.exists(alt):
                image_path = alt
        try:
            image = Image.open(image_path)
            # Apply min-max normalization before converting to RGB
            img_array = np.array(image, dtype=np.float32)
            img_min = img_array.min()
            img_max = img_array.max()
            if img_max > img_min:  # Avoid division by zero
                img_array = (img_array - img_min) / (img_max - img_min)
                # Convert back to uint8 for PIL
                img_array = (img_array * 255).astype(np.uint8)
                image = Image.fromarray(img_array)
            # Now convert to RGB after normalization
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            # Return a sentinel that can be filtered out by a custom collate_fn.
            # NOTE: We do NOT raise here because a single bad file would crash the whole eval.
            return None

        pixel_values = self.image_transform(image)
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values.float()

        labels = []
        for v in sample["labels"]:
            labels.append(0.0 if v is None else float(v))

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "label_mask": torch.tensor(sample["mask"], dtype=torch.float32),
        }


def run_mimic_chexpert_zero_shot_eval(
    model,
    tokenizer,
    mimic_eval_dataset: MimicChexpertEvalDataset,
    batch_size: int,
    max_text_length: int,
    metric_prefix: str,
):
    """
    Run CheXpert-style multi-label zero-shot evaluation on MIMIC-CXR using study-level labels.
    """
    if hasattr(model, "module"):
        model = model.module

    if mimic_eval_dataset is None or len(mimic_eval_dataset) == 0:
        return {}

    label_names = list(mimic_eval_dataset.label_names)

    def _metric_name(suffix: str) -> str:
        return f"{metric_prefix}_{suffix}"

    effective_bs = max(1, batch_size or 1)

    def _collate_skip_failed(features):
        # features can contain `None` entries when an image failed to load.
        num_failed = 0
        kept = []
        for f in features:
            if f is None:
                num_failed += 1
                continue
            kept.append(f)
        if not kept:
            return {"_num_load_failures": num_failed, "_all_failed": True}
        pixel_values = torch.stack([f["pixel_values"] for f in kept])
        labels = torch.stack([f["labels"] for f in kept])
        label_mask = torch.stack([f["label_mask"] for f in kept])
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "label_mask": label_mask,
            "_num_load_failures": num_failed,
        }

    dataloader = DataLoader(
        mimic_eval_dataset,
        batch_size=effective_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_skip_failed,
    )

    device = next(model.parameters()).device

    # Reuse the same template strategy as the built-in CheXpert zero-shot eval.
    template_pairs = [("{}", "no {}")]
    prompts = []
    pos_prompt_indices = []
    neg_prompt_indices = []
    for name in label_names:
        label_pos_indices = []
        label_neg_indices = []
        for pos_template, neg_template in template_pairs:
            pos_idx = len(prompts)
            prompts.append(pos_template.format(name))
            neg_idx = len(prompts)
            prompts.append(neg_template.format(name))
            label_pos_indices.append(pos_idx)
            label_neg_indices.append(neg_idx)
        pos_prompt_indices.append(label_pos_indices)
        neg_prompt_indices.append(label_neg_indices)

    text_tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    if "attention_mask" not in text_tokens:
        pad_token_id = tokenizer.pad_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        text_tokens["attention_mask"] = (text_tokens["input_ids"] != pad_token_id).long()
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

    was_training = model.training
    model.eval()
    with torch.no_grad():
        text_embeds = model._encode_text(text_tokens["input_ids"], text_tokens["attention_mask"])
        if next(model.parameters()).dtype == torch.float16:
            text_embeds = text_embeds.half()

    preds, targets, masks = [], [], []
    image_load_failures = 0
    logit_scale = model.logit_scale.exp()
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            image_load_failures += int(batch.pop("_num_load_failures", 0))
            if bool(batch.pop("_all_failed", False)):
                continue
            pixel_values = batch["pixel_values"].to(device)
            image_embeds = model._encode_image(pixel_values)
            logits = logit_scale * (image_embeds @ text_embeds.t())
            scores_per_label = []
            for pos_idx_list, neg_idx_list in zip(pos_prompt_indices, neg_prompt_indices):
                pos_scores = logits[:, pos_idx_list].mean(dim=1)
                neg_scores = logits[:, neg_idx_list].mean(dim=1)
                probs = torch.sigmoid(pos_scores - neg_scores)
                scores_per_label.append(probs)
            scores = torch.stack(scores_per_label, dim=1)
            preds.append(scores.cpu())
            targets.append(batch["labels"].cpu())
            masks.append(batch["label_mask"].cpu())

    if was_training:
        model.train()

    if not preds:
        metrics: Dict[str, float] = {
            _metric_name("zero_shot_samples"): 0.0,
            _metric_name("zero_shot_labels_total"): float(len(label_names)),
            _metric_name("zero_shot_image_load_failures"): float(image_load_failures),
        }
        print(f"MIMIC-CXR zero-shot eval: image load failures skipped = {image_load_failures}")
        return metrics

    scores = torch.cat(preds, dim=0).numpy()
    label_tensor = torch.cat(targets, dim=0).numpy()
    mask_tensor = torch.cat(masks, dim=0).numpy()

    metrics: Dict[str, float] = {
        _metric_name("zero_shot_samples"): float(scores.shape[0]),
        _metric_name("zero_shot_labels_total"): float(len(label_names)),
    }
    metrics[_metric_name("zero_shot_image_load_failures")] = float(image_load_failures)
    print(f"MIMIC-CXR zero-shot eval: image load failures skipped = {image_load_failures}")

    per_class_aucs = []
    per_class_auprc = []
    per_class_f1 = []
    for idx, name in enumerate(label_names):
        normalized_name = name.strip().lower().replace(" ", "_")
        # Keep consistent with CheXpert benchmark evaluation
        if normalized_name in {"no_finding", "nofinding", "pleural_other"}:
            per_class_aucs.append(math.nan)
            per_class_auprc.append(math.nan)
            per_class_f1.append(math.nan)
            continue

        valid = mask_tensor[:, idx] > 0.5
        if valid.sum() < 2:
            auc = auprc = best_f1 = math.nan
        else:
            s = scores[valid, idx]
            y = label_tensor[valid, idx]
            if y.sum() == 0 or y.sum() == len(y):
                auc = auprc = best_f1 = math.nan
            else:
                auc = float(roc_auc_score(y, s))
                auprc, best_f1 = compute_auprc_and_best_f1(s, y)

        per_class_aucs.append(auc)
        per_class_auprc.append(auprc)
        per_class_f1.append(best_f1)

        metrics[_metric_name(f"zero_shot_auc_{normalized_name}")] = auc
        metrics[_metric_name(f"zero_shot_auprc_{normalized_name}")] = auprc
        metrics[_metric_name(f"zero_shot_f1_{normalized_name}")] = best_f1

    finite_aucs = [v for v in per_class_aucs if not math.isnan(v)]
    finite_auprc = [v for v in per_class_auprc if not math.isnan(v)]
    finite_f1 = [v for v in per_class_f1 if not math.isnan(v)]

    if finite_aucs:
        metrics[_metric_name("zero_shot_auc_avg")] = float(sum(finite_aucs) / len(finite_aucs))
        metrics[_metric_name("zero_shot_labels_used")] = float(len(finite_aucs))
    else:
        metrics[_metric_name("zero_shot_labels_used")] = 0.0
    if finite_auprc:
        metrics[_metric_name("zero_shot_auprc_avg")] = float(sum(finite_auprc) / len(finite_auprc))
    if finite_f1:
        metrics[_metric_name("zero_shot_f1_avg")] = float(sum(finite_f1) / len(finite_f1))

    # Bootstrap CIs for macro AUC only (MIMIC-CXR)
    # (We intentionally do NOT bootstrap AUPRC/F1 per your request.)
    valid_scores = []
    valid_labels = []
    for idx, name in enumerate(label_names):
        normalized_name = name.strip().lower().replace(" ", "_")
        if normalized_name in {"no_finding", "nofinding", "pleural_other"}:
            continue
        valid = mask_tensor[:, idx] > 0.5
        if valid.sum() < 2:
            continue
        s = scores[valid, idx]
        y = label_tensor[valid, idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        valid_scores.append(s)
        valid_labels.append(y)

    if valid_scores:
        auc_mean, auc_low, auc_high = bootstrap_macro_auc_only(valid_scores, valid_labels, seed=42)
        if not math.isnan(auc_mean):
            metrics[_metric_name("zero_shot_auc_avg_boot_mean")] = float(auc_mean)
            metrics[_metric_name("zero_shot_auc_avg_ci_low")] = float(auc_low)
            metrics[_metric_name("zero_shot_auc_avg_ci_high")] = float(auc_high)

    return metrics


# ========== Metrics ==========

def compute_auprc_and_best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if scores.size == 0:
        return math.nan, math.nan
    if labels.sum() == 0:
        return math.nan, math.nan
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    denom = precision + recall
    denom[denom == 0] = 1e-8
    f1_scores = 2 * precision * recall / denom
    best_f1 = float(np.max(f1_scores))
    return float(pr_auc), best_f1


def bootstrap_macro_auc_only(
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap macro AUC over a set of per-class score/label arrays.

    For each bootstrap iteration, resample examples WITH replacement within each class,
    compute per-class AUC when defined, then average across classes (skipping undefined).

    Returns:
      (auc_mean, auc_ci_low, auc_ci_high)
    """
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(ZERO_SHOT_BOOTSTRAP_SAMPLES):
        per_iter = []
        for scores, labels in zip(scores_list, labels_list):
            n = len(scores)
            if n < 2:
                continue
            idx = rng.randint(0, n, n)
            s = scores[idx]
            y = labels[idx]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            per_iter.append(float(roc_auc_score(y, s)))
        if per_iter:
            aucs.append(float(np.mean(per_iter)))
    if not aucs:
        return math.nan, math.nan, math.nan
    aucs = np.asarray(aucs, dtype=np.float64)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_metrics(scores: np.ndarray, labels: np.ndarray, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(scores)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    aucs = []
    auprcs = []
    f1s = []
    for _ in range(ZERO_SHOT_BOOTSTRAP_SAMPLES):
        idx = rng.randint(0, n, n)
        s = scores[idx]
        y = labels[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(roc_auc_score(y, s))
        pr, f1 = compute_auprc_and_best_f1(s, y)
        auprcs.append(pr)
        f1s.append(f1)
    if not aucs:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    def _ci(vals):
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    auc_low, auc_high = _ci(aucs)
    pr_low, pr_high = _ci(auprcs)
    f1_low, f1_high = _ci(f1s)
    return (float(np.mean(aucs)), auc_low, auc_high,
            float(np.mean(auprcs)), pr_low, pr_high,
            float(np.mean(f1s)), f1_low, f1_high)


def bootstrap_macro_metrics(scores_list: List[np.ndarray], labels_list: List[np.ndarray], seed: int = 42):
    rng = np.random.RandomState(seed)
    aucs = []
    auprcs = []
    f1s = []
    for _ in range(ZERO_SHOT_BOOTSTRAP_SAMPLES):
        auc_vals = []
        pr_vals = []
        f1_vals = []
        for scores, labels in zip(scores_list, labels_list):
            n = len(scores)
            idx = rng.randint(0, n, n)
            s = scores[idx]
            y = labels[idx]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            auc_vals.append(roc_auc_score(y, s))
            pr, f1 = compute_auprc_and_best_f1(s, y)
            pr_vals.append(pr)
            f1_vals.append(f1)
        if auc_vals:
            aucs.append(np.mean(auc_vals))
        if pr_vals:
            auprcs.append(np.mean(pr_vals))
        if f1_vals:
            f1s.append(np.mean(f1_vals))
    def _ci(vals):
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    if aucs:
        auc_low, auc_high = _ci(aucs)
        auc_mean = float(np.mean(aucs))
    else:
        auc_mean = auc_low = auc_high = math.nan
    if auprcs:
        pr_low, pr_high = _ci(auprcs)
        pr_mean = float(np.mean(auprcs))
    else:
        pr_mean = pr_low = pr_high = math.nan
    if f1s:
        f1_low, f1_high = _ci(f1s)
        f1_mean = float(np.mean(f1s))
    else:
        f1_mean = f1_low = f1_high = math.nan
    return (auc_mean, auc_low, auc_high, pr_mean, pr_low, pr_high, f1_mean, f1_low, f1_high)


# ========== Main Evaluation Function ==========

def run_chexpert_zero_shot_eval(
    model,
    tokenizer,
    eval_dataset,
    label_json_path: str,
    split_prefix: str,
    batch_size: int,
    max_text_length: int,
    metric_prefix: str,
    resolve_image_path_fn,
):
    """
    Run zero-shot CheXpert classification evaluation.
    
    Args:
        model: VisionOnlySiglipModel instance
        tokenizer: Text tokenizer
        eval_dataset: CXRSiglipDataset instance
        label_json_path: Path to CheXpert labels JSONL
        split_prefix: Which split to evaluate ('train', 'valid', 'test')
        batch_size: Batch size for inference
        max_text_length: Max tokens for text prompts
        metric_prefix: Prefix for logged metrics (e.g., 'eval')
        resolve_image_path_fn: Function to normalize image paths
    
    Returns:
        dict: Computed metrics
    """
    # Unwrap DDP/FSDP wrapper if needed
    if hasattr(model, "module"):
        model = model.module
    
    if not label_json_path:
        return {}
    
    try:
        label_names, label_lookup = load_chexpert_label_lookup(label_json_path, split_prefix)
    except FileNotFoundError:
        print(f"CheXpert label file {label_json_path} not found; skipping zero-shot eval.")
        return {}
    except ValueError as exc:
        print(f"Failed to parse {label_json_path}: {exc}")
        return {}
    
    zero_shot_dataset = build_zero_shot_dataset(eval_dataset, label_lookup, resolve_image_path_fn)
    if zero_shot_dataset is None:
        print("No overlapping CheXpert samples found for zero-shot evaluation; skipping.")
        return {}
    
    print(
        f"Running zero-shot CheXpert evaluation on {len(zero_shot_dataset)} images "
        f"with {len(label_names)} labels."
    )
    
    effective_bs = max(1, batch_size or 1)
    dataloader = DataLoader(
        zero_shot_dataset,
        batch_size=effective_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    device = next(model.parameters()).device
    
    # Encode label prompts (CheXzero-style pos/neg template pairs per label)
    prompts = []
    pos_prompt_indices = []
    neg_prompt_indices = []
    for name in label_names:
        label_pos_indices = []
        label_neg_indices = []
        for pos_template, neg_template in ZERO_SHOT_TEMPLATE_PAIRS:
            pos_idx = len(prompts)
            prompts.append(pos_template.format(name))
            neg_idx = len(prompts)
            prompts.append(neg_template.format(name))
            label_pos_indices.append(pos_idx)
            label_neg_indices.append(neg_idx)
        pos_prompt_indices.append(label_pos_indices)
        neg_prompt_indices.append(label_neg_indices)
    text_tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    if "attention_mask" not in text_tokens:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        text_tokens["attention_mask"] = (text_tokens["input_ids"] != pad_token_id).long()
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    
    was_training = model.training
    model.eval()
    with torch.no_grad():
        text_embeds = model._encode_text(text_tokens["input_ids"], text_tokens["attention_mask"])
        # Ensure text_embeds matches model dtype (for FP16 compatibility)
        if next(model.parameters()).dtype == torch.float16:
            text_embeds = text_embeds.half()
    
    # Encode images and compute similarities
    preds, targets, masks = [], [], []
    logit_scale = model.logit_scale.exp()
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            image_embeds = model._encode_image(pixel_values)
            logits = logit_scale * (image_embeds @ text_embeds.t())
            # Convert template logits into per-label probabilities
            scores_per_label = []
            for pos_idx_list, neg_idx_list in zip(pos_prompt_indices, neg_prompt_indices):
                pos_scores = logits[:, pos_idx_list].mean(dim=1)
                neg_scores = logits[:, neg_idx_list].mean(dim=1)
                probs = torch.sigmoid(pos_scores - neg_scores)
                scores_per_label.append(probs)
            scores = torch.stack(scores_per_label, dim=1)
            preds.append(scores.cpu())
            targets.append(batch["labels"].cpu())
            masks.append(batch["label_mask"].cpu())
    
    if was_training:
        model.train()
    
    scores = torch.cat(preds, dim=0)
    label_tensor = torch.cat(targets, dim=0)
    mask_tensor = torch.cat(masks, dim=0)
    
    def _metric_name(suffix: str) -> str:
        return f"{metric_prefix}_{suffix}"
    
    metrics: Dict[str, float] = {
        _metric_name("zero_shot_samples"): float(scores.size(0)),
    }
    
    # Per-class metrics (excluding "No Finding" to match benchmark evaluation)
    per_class_aucs = []
    per_class_auprc = []
    per_class_f1 = []
    for idx, name in enumerate(label_names):
        # Skip "No Finding" and "Pleural Other" - consistent with benchmark tasks
        normalized_name = name.strip().lower().replace(" ", "_")
        if normalized_name in {"no_finding", "nofinding", "pleural_other"}:
        # if normalized_name == "pleural_other":
            per_class_aucs.append(math.nan)
            per_class_auprc.append(math.nan)
            per_class_f1.append(math.nan)
            continue
        
        valid = mask_tensor[:, idx] > 0.5
        if valid.sum().item() < 2:
            auc = math.nan
            auprc = math.nan
            best_f1 = math.nan
        else:
            class_scores = scores[valid, idx].numpy()
            class_labels = label_tensor[valid, idx].numpy()

            if ZERO_SHOT_DEBUG:
                # Debug: print class distribution and score stats
                pos_count = (class_labels > 0.5).sum().item()
                neg_count = (class_labels <= 0.5).sum().item()
                print(
                    f"DEBUG {name}: pos={pos_count}, neg={neg_count}, "
                    f"pos_ratio={pos_count/(pos_count+neg_count):.3f}"
                )
                print(
                    f"DEBUG {name}: scores mean={class_scores.mean():.3f}, std={class_scores.std():.3f}, "
                    f"min={class_scores.min():.3f}, max={class_scores.max():.3f}"
                )
            
            if class_labels.sum() == 0 or class_labels.sum() == len(class_labels):
                auc = math.nan
                auprc = math.nan
                best_f1 = math.nan
            else:
                auc = float(roc_auc_score(class_labels, class_scores))
                auprc, best_f1 = compute_auprc_and_best_f1(class_scores, class_labels)
            
            if ZERO_SHOT_DEBUG:
                print(f"DEBUG {name}: AUC={auc:.3f}, AUPRC={auprc:.3f}, F1={best_f1:.3f}")
        per_class_aucs.append(auc)
        per_class_auprc.append(auprc)
        per_class_f1.append(best_f1)
        # Per-disease metrics (same as MIMIC-CXR) for CheXpert Plus
        metrics[_metric_name(f"zero_shot_auc_{normalized_name}")] = auc if not math.isnan(auc) else float("nan")
        metrics[_metric_name(f"zero_shot_auprc_{normalized_name}")] = auprc if not math.isnan(auprc) else float("nan")
        metrics[_metric_name(f"zero_shot_f1_{normalized_name}")] = best_f1 if not math.isnan(best_f1) else float("nan")
    
    # Aggregate metrics
    finite_aucs = [value for value in per_class_aucs if not math.isnan(value)]
    finite_aucs_no_support = []
    for idx, (value, name) in enumerate(zip(per_class_aucs, label_names)):
        if math.isnan(value):
            continue
        normalized_name = name.strip().lower().replace(" ", "_")
        if normalized_name in {"support_devices"}:
            continue
        finite_aucs_no_support.append(value)
    
    if finite_aucs:
        macro_auc = float(sum(finite_aucs) / len(finite_aucs))
        metrics[_metric_name("zero_shot_auc_avg")] = macro_auc
        metrics[_metric_name("zero_shot_labels_used")] = float(len(finite_aucs))
    else:
        metrics[_metric_name("zero_shot_labels_used")] = 0.0
    
    if finite_aucs_no_support:
        macro_auc_no_support = float(sum(finite_aucs_no_support) / len(finite_aucs_no_support))
        metrics[_metric_name("zero_shot_auc_avg_no_support")] = macro_auc_no_support
        metrics[_metric_name("zero_shot_labels_used_no_support")] = float(len(finite_aucs_no_support))
    
    finite_auprc = [value for value in per_class_auprc if not math.isnan(value)]
    if finite_auprc:
        macro_auprc = float(sum(finite_auprc) / len(finite_auprc))
        metrics[_metric_name("zero_shot_auprc_avg")] = macro_auprc
    
    finite_f1 = [value for value in per_class_f1 if not math.isnan(value)]
    if finite_f1:
        macro_f1 = float(sum(finite_f1) / len(finite_f1))
        metrics[_metric_name("zero_shot_f1_avg")] = macro_f1

    # Bootstrap CIs for macro metrics (sklearn-based)
    valid_scores = []
    valid_labels = []
    valid_scores_no_support = []
    valid_labels_no_support = []
    for idx, name in enumerate(label_names):
        normalized_name = name.strip().lower().replace(" ", "_")
        if normalized_name in {"no_finding", "nofinding", "pleural_other"}:
        # if normalized_name == "pleural_other":
            continue
        valid = mask_tensor[:, idx] > 0.5
        if valid.sum().item() < 2:
            continue
        s = scores[valid, idx].numpy()
        y = label_tensor[valid, idx].numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        valid_scores.append(s)
        valid_labels.append(y)
        
        # Also collect without support devices
        if normalized_name not in {"support_devices"}:
            valid_scores_no_support.append(s)
            valid_labels_no_support.append(y)

    if valid_scores:
        auc_mean, auc_low, auc_high, pr_mean, pr_low, pr_high, f1_mean, f1_low, f1_high = bootstrap_macro_metrics(
            valid_scores, valid_labels
        )
        if not math.isnan(auc_mean):
            metrics[_metric_name("zero_shot_auc_avg_boot_mean")] = auc_mean
            metrics[_metric_name("zero_shot_auc_avg_ci_low")] = auc_low
            metrics[_metric_name("zero_shot_auc_avg_ci_high")] = auc_high
        if not math.isnan(pr_mean):
            metrics[_metric_name("zero_shot_auprc_avg_boot_mean")] = pr_mean
            metrics[_metric_name("zero_shot_auprc_avg_ci_low")] = pr_low
            metrics[_metric_name("zero_shot_auprc_avg_ci_high")] = pr_high
        if not math.isnan(f1_mean):
            metrics[_metric_name("zero_shot_f1_avg_boot_mean")] = f1_mean
            metrics[_metric_name("zero_shot_f1_avg_ci_low")] = f1_low
            metrics[_metric_name("zero_shot_f1_avg_ci_high")] = f1_high
    
    # Bootstrap CIs excluding Support Devices
    if valid_scores_no_support:
        auc_mean_ns, auc_low_ns, auc_high_ns, _, _, _, _, _, _ = bootstrap_macro_metrics(
            valid_scores_no_support, valid_labels_no_support
        )
        if not math.isnan(auc_mean_ns):
            metrics[_metric_name("zero_shot_auc_avg_no_support_boot_mean")] = auc_mean_ns
            metrics[_metric_name("zero_shot_auc_avg_no_support_ci_low")] = auc_low_ns
            metrics[_metric_name("zero_shot_auc_avg_no_support_ci_high")] = auc_high_ns
    
    return metrics


# ========== Trainer Integration ==========

class ZeroShotEvalCallback(TrainerCallback):
    """Callback to run zero-shot evaluation after each regular evaluation."""
    
    def __init__(self, eval_dataset, tokenizer, data_args, training_args):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.trainer = None
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation - add zero-shot metrics and log to wandb."""
        if not self.data_args.chexpert_zero_shot_labels:
            return
        
        model = kwargs.get("model")
        if model is None:
            return

        # Run only on rank 0 to avoid duplicating work and duplicated prints.
        if self.trainer is not None and not self.trainer.is_world_process_zero():
            return
        
        metric_prefix = "eval"
        
        zero_shot_metrics = run_chexpert_zero_shot_eval(
            model,
            self.tokenizer,
            self.eval_dataset,
            self.data_args.chexpert_zero_shot_labels,
            self.data_args.chexpert_zero_shot_split,
            self.training_args.per_device_eval_batch_size,
            self.data_args.max_text_length,
            metric_prefix,
            resolve_image_path,
        )
        
        if zero_shot_metrics:
            if metrics is not None:
                metrics.update(zero_shot_metrics)
            if self.trainer is not None:
                self.trainer.log(zero_shot_metrics)


class MimicChexpertEvalCallback(TrainerCallback):
    """Callback to run MIMIC-CXR (CheXpert labels) zero-shot eval after each evaluation."""

    def __init__(self, mimic_eval_dataset, tokenizer, data_args, training_args):
        self.mimic_eval_dataset = mimic_eval_dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.trainer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.mimic_eval_dataset is None:
            return

        model = kwargs.get("model")
        if model is None:
            return

        # Run only on rank 0 to avoid duplicating work.
        if self.trainer is not None and not self.trainer.is_world_process_zero():
            return

        metric_prefix = (getattr(self.data_args, "mimic_metric_prefix", None) or "mimic").strip()
        mimic_metrics = run_mimic_chexpert_zero_shot_eval(
            model=model,
            tokenizer=self.tokenizer,
            mimic_eval_dataset=self.mimic_eval_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            max_text_length=self.data_args.max_text_length,
            metric_prefix=metric_prefix,
        )
        if mimic_metrics:
            if metrics is not None:
                metrics.update(mimic_metrics)
            if self.trainer is not None:
                self.trainer.log(mimic_metrics)


def evaluate_with_zero_shot(
    trainer,
    eval_dataset,
    tokenizer,
    data_args,
    training_args,
    metric_prefix: str,
    resolve_image_path_fn=None,
):
    """Run evaluation with optional zero-shot classification metrics."""
    # Use module-level resolve_image_path if not provided
    if resolve_image_path_fn is None:
        resolve_image_path_fn = resolve_image_path
    
    metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_prefix)
    
    if data_args.chexpert_zero_shot_labels and trainer.is_world_process_zero():
        zero_shot_metrics = run_chexpert_zero_shot_eval(
            trainer.model,
            tokenizer,
            eval_dataset,
            data_args.chexpert_zero_shot_labels,
            data_args.chexpert_zero_shot_split,
            training_args.per_device_eval_batch_size,
            data_args.max_text_length,
            metric_prefix,
            resolve_image_path_fn,
        )
        metrics.update(zero_shot_metrics)
        if zero_shot_metrics:
            trainer.log(zero_shot_metrics)
    
    if trainer.is_world_process_zero():
        trainer.log_metrics(metric_prefix, metrics)
        trainer.save_metrics(metric_prefix, metrics)
    
    return metrics
