"""Zero-shot classification of 30 abdominal findings from CT volumes.

Protocol (as in the paper):

1. Every evaluation volume is embedded once with the vision tower.
2. For each finding, ground-truth cohorts are built by matching the finding's
   positive and negative phrases against the evaluation reports
   (case-insensitive substring match), then balanced by subsampling the
   larger class.
3. The image embedding is compared against the embedded positive and negative
   prompts; the higher mean similarity wins, so no decision threshold is
   tuned.
4. Per-finding F1 with a 1000-iteration bootstrap CI, macro-averaged over all
   findings (AUPRC/AUROC on the continuous score are reported alongside).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from ct import data as ct_data
from ct.prompts import DiseasePrompts, disease_prompts

N_BOOTSTRAP = 1000


def compute_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    true_pos = float(((preds == 1) & (labels == 1)).sum())
    false_pos = float(((preds == 1) & (labels == 0)).sum())
    false_neg = float(((preds == 0) & (labels == 1)).sum())

    if true_pos + false_pos == 0 or true_pos + false_neg == 0:
        return 0.0
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bootstrap_f1(
    preds: np.ndarray, labels: np.ndarray, seed: int = 0
) -> Tuple[float, float, List[float]]:
    """95% bootstrap CI for the F1 score.

    Returns:
        ``(lower, upper, scores)`` where ``scores`` holds all bootstrap
        iterations (reused for the macro-average CI).
    """
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(N_BOOTSTRAP):
        indices = rng.randint(0, len(preds), len(preds))
        scores.append(compute_f1(preds[indices], labels[indices]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5), scores


def bootstrap_score_metric(
    scores: np.ndarray, labels: np.ndarray, metric_fn, seed: int = 0
) -> List[float]:
    """Bootstrap distribution of a threshold-free metric (AUPRC/AUROC).

    Iterations whose resample contains a single class (possible for rare
    findings) are skipped.
    """
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(N_BOOTSTRAP):
        indices = rng.randint(0, len(scores), len(scores))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(metric_fn(sampled_labels, scores[indices]))
    return values


def build_cohort(
    labels_df: pd.DataFrame, prompts: DiseasePrompts
) -> Tuple[List[str], List[str]]:
    """Match a finding's phrases against the reports to label the cohort.

    Returns:
        ``(positive_accessions, negative_accessions)``; a report matching any
        phrase of a list joins that cohort.
    """

    def matching_accessions(phrases: Sequence[str]) -> List[str]:
        accessions = set()
        for phrase in phrases:
            matches = labels_df[
                labels_df["findings"].str.contains(
                    phrase, case=False, regex=False, na=False
                )
            ]
            accessions.update(matches["anon_accession"].unique())
        return list(accessions)

    return (
        matching_accessions(prompts.positive_prompts),
        matching_accessions(prompts.negative_prompts),
    )


@torch.no_grad()
def embed_images(model, loader) -> Dict[str, np.ndarray]:
    """Embed every volume in the loader, keyed by accession."""
    device = next(model.parameters()).device
    embeddings = {}
    for batch in tqdm(loader, desc="Embedding evaluation volumes"):
        image_features = model.encode_image(batch["image"].to(device))[0]
        for accession, features in zip(batch["anon_accession"], image_features):
            embeddings[accession] = features.detach().cpu().numpy()
    return embeddings


@torch.no_grad()
def zero_shot_classification(
    model,
    metadata_df: pd.DataFrame,
    batch_size: int = 4,
    cache_dir: str = None,
    num_workers: int = 0,
    balance_seed: int = 42,
) -> Dict[str, dict]:
    """Run the full 30-finding zero-shot evaluation.

    Args:
        model: A ``Clip3D`` model in eval mode.
        metadata_df: Evaluation split with resolved image paths and
            ``anon_accession``/``findings`` columns.
        batch_size: Inference batch size.
        cache_dir: Optional preprocessing cache directory.
        num_workers: Dataloader workers.
        balance_seed: Seed for the per-finding class balancing (42 reproduces
            the paper's cohorts).

    Returns:
        Per-finding results plus a ``macro_average`` entry; each finding maps
        to ``{"f1", "ci_lower", "ci_upper", "auprc", "auroc", "num_examples"}``.
    """
    model.eval()
    np.random.seed(balance_seed)

    loader = ct_data.build_dataloader(
        ct_data.build_datalist(metadata_df),
        batch_size=batch_size,
        cache_dir=cache_dir,
        num_workers=num_workers,
    )
    embeddings = embed_images(model, loader)

    results = {}
    f1_bootstraps, auprc_bootstraps, auroc_bootstraps = {}, {}, {}

    for disease, prompts in disease_prompts.items():
        print(f"\nFinding: {disease}")

        text_features = model.encode_text(
            prompts.positive_prompts + prompts.negative_prompts
        )
        num_positive_prompts = len(prompts.positive_prompts)

        pos_accessions, neg_accessions = build_cohort(metadata_df, prompts)
        pos_accessions = [a for a in pos_accessions if a in embeddings]
        neg_accessions = [a for a in neg_accessions if a in embeddings]

        # Balance the classes by subsampling the larger cohort.
        cohort_size = min(len(pos_accessions), len(neg_accessions))
        if len(pos_accessions) > cohort_size:
            pos_accessions = np.random.choice(pos_accessions, cohort_size, replace=False)
        elif len(neg_accessions) > cohort_size:
            neg_accessions = np.random.choice(neg_accessions, cohort_size, replace=False)

        print(f"Positive/negative examples: {len(pos_accessions)}/{len(neg_accessions)}")
        if cohort_size == 0:
            print(f"Skipping {disease} - no positive or no negative examples.")
            continue

        image_features = torch.tensor(
            np.array(
                [embeddings[a] for a in neg_accessions]
                + [embeddings[a] for a in pos_accessions]
            ),
            device=text_features.device,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = 100.0 * image_features @ text_features.T
        positive_score = similarity[:, :num_positive_prompts].mean(dim=1)
        negative_score = similarity[:, num_positive_prompts:].mean(dim=1)

        preds = (positive_score > negative_score).long().cpu().numpy()
        labels = np.array([0] * len(neg_accessions) + [1] * len(pos_accessions))
        # Continuous pos-minus-neg score for the threshold-free metrics.
        cont_scores = (positive_score - negative_score).cpu().numpy()

        f1 = compute_f1(preds, labels)
        ci_lower, ci_upper, f1_bootstraps[disease] = bootstrap_f1(preds, labels)
        auprc = average_precision_score(labels, cont_scores)
        auroc = roc_auc_score(labels, cont_scores)
        auprc_bootstraps[disease] = bootstrap_score_metric(
            cont_scores, labels, average_precision_score
        )
        auroc_bootstraps[disease] = bootstrap_score_metric(
            cont_scores, labels, roc_auc_score
        )

        print(f"F1: {f1:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        results[disease] = {
            "f1": f1,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "auprc": auprc,
            "auroc": auroc,
            "num_examples": 2 * cohort_size,
        }

    if not results:
        print("WARNING: no finding had both positive and negative examples.")
        return {}

    results["macro_average"] = _macro_average(
        results, f1_bootstraps, auprc_bootstraps, auroc_bootstraps
    )
    return results


def _macro_average(results, f1_bootstraps, auprc_bootstraps, auroc_bootstraps):
    """Macro-average the per-finding metrics, with bootstrap CIs."""
    findings = list(f1_bootstraps)
    macro_f1 = float(np.mean([results[d]["f1"] for d in findings]))

    # Average across findings within each bootstrap iteration. AUPRC/AUROC
    # iterations with a single-class resample were skipped per finding, so
    # index modulo the available count.
    f1_avgs = [
        np.mean([f1_bootstraps[d][i] for d in findings]) for i in range(N_BOOTSTRAP)
    ]
    macro = {
        "f1": macro_f1,
        "ci_lower": float(np.percentile(f1_avgs, 2.5)),
        "ci_upper": float(np.percentile(f1_avgs, 97.5)),
        "num_findings": len(findings),
    }
    print(
        f"\nZero-shot F1 macro average: {macro['f1']:.5f} "
        f"[{macro['ci_lower']:.5f}, {macro['ci_upper']:.5f}] "
        f"over {len(findings)} findings"
    )

    for name, boot in (("auprc", auprc_bootstraps), ("auroc", auroc_bootstraps)):
        macro[name] = float(np.mean([results[d][name] for d in findings]))
        avgs = [
            np.mean([boot[d][i % len(boot[d])] for d in findings])
            for i in range(N_BOOTSTRAP)
        ]
        macro[f"{name}_ci_lower"] = float(np.percentile(avgs, 2.5))
        macro[f"{name}_ci_upper"] = float(np.percentile(avgs, 97.5))
        print(
            f"Zero-shot {name.upper()} macro average: {macro[name]:.5f} "
            f"[{macro[f'{name}_ci_lower']:.5f}, {macro[f'{name}_ci_upper']:.5f}]"
        )

    return macro
