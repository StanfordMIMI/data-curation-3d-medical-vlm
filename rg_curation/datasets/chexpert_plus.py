"""
CheXpert Plus metadata handling.

Turns the released ``df_chexpert_plus_240401.csv`` into the pool / reference
CSV pair the curation pipeline expects.  The paper's CXR experiments use the
single-source regime: the pool is the ``train`` split restricted to studies
with a usable findings section (58,006 studies), and the reference set is the
``valid`` split under the same restriction.

Every sample gets one canonical ``sample_id``, used as the embedding filename
key for both modalities.  Deriving it in a single place is what lets the
vision and text rankings be fused later without any key reconciliation.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns required in the released CheXpert Plus CSV.
REQUIRED_COLUMNS = (
    "deid_patient_id",
    "patient_report_date_order",
    "path_to_image",
    "section_findings",
    "split",
)

#: Placeholders that appear in ``section_findings`` in place of real text.
_EMPTY_FINDINGS = {"", "nan", "/"}


def is_valid_findings(value) -> bool:
    """True if a findings cell holds usable report text.

    Rejects nulls, empty strings, the literal string ``"nan"``, and the
    single-slash placeholder used for missing sections in CheXpert Plus.

    Args:
        value: Raw cell value from ``section_findings``.

    Returns:
        Whether the value is usable as report text.
    """
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in _EMPTY_FINDINGS


def build_sample_id(row: pd.Series) -> str:
    """Derive the canonical sample identifier for one CheXpert Plus row.

    The identifier is ``{patient}_study{order}_{view}``, e.g.
    ``patient00005_study4_view2_frontal``.  It is stable across the vision and
    text pipelines, so DreamSim and Longformer embeddings for the same study
    share a key and their rankings can be fused directly.

    Args:
        row: A row with ``deid_patient_id``, ``patient_report_date_order``,
            and ``path_to_image``.

    Returns:
        The sample identifier.
    """
    view = os.path.splitext(os.path.basename(str(row["path_to_image"])))[0]
    return f"{row['deid_patient_id']}_study{row['patient_report_date_order']}_{view}"


def load_chexpert_plus(csv_path: str, require_findings: bool = True) -> pd.DataFrame:
    """Load the CheXpert Plus metadata CSV into the pipeline's schema.

    Args:
        csv_path: Path to ``df_chexpert_plus_240401.csv``.
        require_findings: Drop rows whose findings section is unusable.

    Returns:
        DataFrame with ``sample_id``, ``image_file``, ``findings``, ``split``,
        and — when present in the source — ``dcm_file``, ``frontal_lateral``,
        and ``ap_pa``.

    Raises:
        ValueError: If the CSV lacks a required column.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CheXpert Plus CSV is missing required column(s): {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    if require_findings:
        before = len(df)
        df = df[df["section_findings"].apply(is_valid_findings)].copy()
        logger.info(
            f"Kept {len(df)} of {before} rows with a usable findings section "
            f"({before - len(df)} dropped)"
        )

    out = pd.DataFrame(
        {
            "sample_id": df.apply(build_sample_id, axis=1),
            "image_file": df["path_to_image"].astype(str),
            "findings": df["section_findings"].astype(str),
            "split": df["split"].astype(str),
        }
    )

    # Carried through when available: the DICOM path drives embedding
    # generation, the view columns are useful for stratified analysis.
    for source, target in (
        ("path_to_dcm", "dcm_file"),
        ("frontal_lateral", "frontal_lateral"),
        ("ap_pa", "ap_pa"),
    ):
        if source in df.columns:
            out[target] = df[source].values

    duplicates = out["sample_id"].duplicated().sum()
    if duplicates:
        logger.warning(
            f"{duplicates} duplicate sample_id(s) after construction; keeping the first of each"
        )
        out = out.drop_duplicates(subset=["sample_id"], keep="first")

    return out.reset_index(drop=True)


def split_pool_and_reference(
    df: pd.DataFrame,
    pool_split: str = "train",
    reference_split: str = "valid",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the prepared metadata into the training pool and reference set.

    Args:
        df: Output of :func:`load_chexpert_plus`.
        pool_split: Split name used as the uncurated pool (D0).
        reference_split: Split name used as the reference set (D_ref).

    Returns:
        ``(pool_df, reference_df)``.

    Raises:
        ValueError: If either split is empty.
    """
    pool = df[df["split"] == pool_split].reset_index(drop=True)
    reference = df[df["split"] == reference_split].reset_index(drop=True)

    for name, frame in ((pool_split, pool), (reference_split, reference)):
        if frame.empty:
            raise ValueError(
                f"Split '{name}' is empty. Available splits: "
                f"{sorted(df['split'].unique())}"
            )

    logger.info(f"Pool ({pool_split}): {len(pool)} | Reference ({reference_split}): {len(reference)}")
    return pool, reference
