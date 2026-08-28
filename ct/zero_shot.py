"""Zero-shot classification of 30 abdominal findings from CT volumes."""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from ct.prompts import DiseasePrompts

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
