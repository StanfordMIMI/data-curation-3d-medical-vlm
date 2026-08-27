"""
Reciprocal Rank Fusion (RRF) for combining vision and text alignment scores.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_ETA = 60
_DEFAULT_WEIGHT = 1.0


def compute_rrf_scores(
    df_vision: pd.DataFrame,
    df_text: pd.DataFrame,
    id_column: str = "sample_id",
    vision_score_col: str = "similarity_score",
    text_score_col: str = "similarity_score",
    w_vision: float = _DEFAULT_WEIGHT,
    w_text: float = _DEFAULT_WEIGHT,
    eta: int = _DEFAULT_ETA,
) -> pd.DataFrame:
    """Combine two scored DataFrames using Reciprocal Rank Fusion.

    Ranks are computed within each DataFrame independently (higher score →
    lower rank number → better).  The two DataFrames are then outer-joined on
    ``id_column`` and the RRF score is computed for each sample.

    Args:
        df_vision: DataFrame with vision alignment scores.  Must contain
            ``id_column`` and ``vision_score_col``.
        df_text: DataFrame with text alignment scores.  Must contain
            ``id_column`` and ``text_score_col``.
        id_column: Name of the unique sample identifier column
            (default: ``"sample_id"``).
        vision_score_col: Name of the similarity score column in *df_vision*
            (default: ``"similarity_score"``).
        text_score_col: Name of the similarity score column in *df_text*
            (default: ``"similarity_score"``).
        w_vision: Weight for the vision ranking term (default: 1.0).
        w_text: Weight for the text ranking term (default: 1.0).
        eta: RRF smoothing constant η (default: 60).

    Returns:
        DataFrame sorted by ``rrf_score`` descending, covering the union of
        both candidate sets.  Metadata columns are taken from *df_vision* and
        filled from *df_text* for samples the vision ranking did not retrieve,
        so every row carries its image path and report text.  Adds
        ``rrf_score``, ``final_rank``, ``rank_vision``, ``rank_text``,
        ``similarity_score_vision``, and ``similarity_score_text``.
    """
    for df, name, col in (
        (df_vision, "vision", vision_score_col),
        (df_text, "text", text_score_col),
    ):
        if id_column not in df.columns:
            raise ValueError(f"'{id_column}' not found in {name} DataFrame")
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in {name} DataFrame")

    # Each sample must appear once per ranking, otherwise duplicates multiply
    # through the merge below.  Keep the highest-scoring occurrence.
    df_vision = (
        df_vision.sort_values(vision_score_col, ascending=False)
        .drop_duplicates(subset=[id_column], keep="first")
        .reset_index(drop=True)
    )
    df_text = (
        df_text.sort_values(text_score_col, ascending=False)
        .drop_duplicates(subset=[id_column], keep="first")
        .reset_index(drop=True)
    )

    # Compute per-list ranks (ascending=False: higher score → rank 1).
    v_sub = df_vision[[id_column, vision_score_col]].copy()
    v_sub["rank_vision"] = v_sub[vision_score_col].rank(ascending=False, method="min")
    v_sub = v_sub.rename(columns={vision_score_col: "similarity_score_vision"})

    t_sub = df_text[[id_column, text_score_col]].copy()
    t_sub["rank_text"] = t_sub[text_score_col].rank(ascending=False, method="min")
    t_sub = t_sub.rename(columns={text_score_col: "similarity_score_text"})

    logger.info(
        f"Vision ranks: {v_sub['rank_vision'].min():.0f} – "
        f"{v_sub['rank_vision'].max():.0f}  ({len(v_sub)} samples)"
    )
    logger.info(
        f"Text ranks: {t_sub['rank_text'].min():.0f} – "
        f"{t_sub['rank_text'].max():.0f}  ({len(t_sub)} samples)"
    )

    # Ranks over the union of both candidate sets.
    merged = v_sub.merge(t_sub, on=id_column, how="outer")

    # Metadata: prefer the vision row, fall back to the text row for samples
    # only the text ranking retrieved.  Without this fallback those samples
    # reach training with an empty image path and empty findings.
    meta_columns = [
        c for c in df_vision.columns
        if c not in (id_column, vision_score_col, "similarity_score_vision")
    ]
    v_meta = df_vision.set_index(id_column)[meta_columns]
    t_meta = df_text.set_index(id_column).reindex(columns=meta_columns)
    metadata = v_meta.combine_first(t_meta)

    merged = merged.merge(
        metadata.reset_index().rename(columns={"index": id_column}),
        on=id_column,
        how="left",
    )

    # Missing modality contribution is 0 (infinite rank penalty).
    term_v = (w_vision / (eta + merged["rank_vision"])).fillna(0.0)
    term_t = (w_text / (eta + merged["rank_text"])).fillna(0.0)

    merged["rrf_score"] = term_v + term_t
    merged = merged.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    merged["final_rank"] = merged.index + 1

    # Coverage statistics
    both = (merged["rank_vision"].notna() & merged["rank_text"].notna()).sum()
    only_v = (merged["rank_vision"].notna() & merged["rank_text"].isna()).sum()
    only_t = (merged["rank_vision"].isna() & merged["rank_text"].notna()).sum()
    n = len(merged)
    logger.info(
        f"Dataset coverage — both: {both} ({both/n*100:.1f}%), "
        f"vision only: {only_v} ({only_v/n*100:.1f}%), "
        f"text only: {only_t} ({only_t/n*100:.1f}%)"
    )
    logger.info(
        f"RRF score — mean: {merged['rrf_score'].mean():.6f}, "
        f"max: {merged['rrf_score'].max():.6f}"
    )
    return merged


def find_similarity_column(df: pd.DataFrame, label: str = "") -> str:
    """Auto-detect the similarity score column in a DataFrame.

    Looks for columns containing the substring ``"similarity_score"``
    (case-insensitive).

    Args:
        df: Input DataFrame.
        label: Descriptive label used in error messages.

    Returns:
        Name of the first matching column.

    Raises:
        ValueError: If no matching column is found, or if multiple matches
            are present.
    """
    cols = [c for c in df.columns if "similarity_score" in c.lower()]
    if not cols:
        raise ValueError(f"No similarity_score column found{' in ' + label if label else ''}")
    if len(cols) > 1:
        raise ValueError(f"Multiple similarity columns in {label}: {cols}. Using {cols[0]}")
    return cols[0]
