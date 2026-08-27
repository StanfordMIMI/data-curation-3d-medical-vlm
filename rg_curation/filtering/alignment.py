"""
Reference-guided kNN alignment for filtering a training pool.

Given pre-computed embeddings for both a training pool (D0) and a small
high-quality reference set (D_ref), this module finds the k nearest
neighbours in D0 for each reference sample, then deduplicates so that each
pool sample is retained at most once — with the similarity score equal to
its maximum cosine similarity to any reference sample.

Embeddings are represented uniformly as ``(S, D)`` arrays, where ``S`` is the
number of encoded views per sample: one row per extracted axial slice for CT
volumes, exactly one row for 2D chest X-rays.  Similarity is therefore the
average slice-aligned cosine similarity in general, which reduces to plain
cosine similarity when ``S = 1``.  Single-view pools take a vectorised path
(one matrix product for the whole pool); multi-view pools fall back to a
per-sample comparison.

Three alignment strategies are supported:

``vision_only``
    DreamSim embeddings.

``text_only``
    Clinical Longformer embeddings (always single-view).

``early_fusion``
    Per-view DreamSim embeddings concatenated with the (broadcast)
    Longformer embedding.  Each modality is L2-normalised before
    concatenation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from rg_curation.embeddings import store

logger = logging.getLogger(__name__)

METHODS = ("vision_only", "text_only", "early_fusion")


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def load_dreamsim_embedding(sample_id: str, embeddings_dir: str) -> np.ndarray:
    """Load DreamSim embeddings for a single sample.

    Args:
        sample_id: Unique sample identifier.
        embeddings_dir: Directory of ``.pt`` files, in either supported
            layout (see :mod:`rg_curation.embeddings.store`).

    Returns:
        Float32 array of shape ``(S, D)``.
    """
    return store.load_embeddings(embeddings_dir, [sample_id], desc="")[sample_id]


def load_text_embedding(sample_id: str, embeddings_dir: str) -> np.ndarray:
    """Load the Clinical Longformer embedding for a single sample.

    Args:
        sample_id: Unique sample identifier.
        embeddings_dir: Directory of ``.pt`` files.

    Returns:
        Float32 array of shape ``(1, D)``.
    """
    return store.load_embeddings(embeddings_dir, [sample_id], desc="")[sample_id]


# ---------------------------------------------------------------------------
# Normalisation and similarity
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Row-normalise an array of embeddings to unit L2 norm.

    Args:
        embeddings: Array of shape ``(N, D)`` or ``(D,)``.

    Returns:
        L2-normalised array of the same shape.
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / max(norm, 1e-12)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def compute_slice_aligned_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Average slice-aligned cosine similarity between two samples.

    Views are compared in order up to the minimum view count of the two
    samples.  Both inputs must already be L2-normalised.  For single-view
    embeddings this is plain cosine similarity.

    Args:
        emb_a: Normalised array of shape ``(S_a, D)``.
        emb_b: Normalised array of shape ``(S_b, D)``.

    Returns:
        Mean cosine similarity.
    """
    n = min(len(emb_a), len(emb_b))
    if n == 0:
        return 0.0
    return float(np.mean(np.sum(emb_a[:n] * emb_b[:n], axis=1)))


def build_early_fusion_embedding(
    vision_emb: np.ndarray,
    text_emb: np.ndarray,
) -> np.ndarray:
    """Concatenate normalised vision and text embeddings (early fusion).

    The text embedding is broadcast across the vision views so that
    slice-aligned similarity can be computed on the joint representation.

    Args:
        vision_emb: Array of shape ``(S, D_v)``.
        text_emb: Array of shape ``(1, D_t)`` or ``(D_t,)``.

    Returns:
        Float32 array of shape ``(S, D_v + D_t)`` with unit-norm rows.
    """
    vision_norm = normalize_embeddings(vision_emb)
    text_norm = normalize_embeddings(np.asarray(text_emb).reshape(-1))
    text_broadcast = np.tile(text_norm, (len(vision_emb), 1))
    fused = np.concatenate([vision_norm, text_broadcast], axis=1).astype(np.float32)
    return normalize_embeddings(fused)


# ---------------------------------------------------------------------------
# Assembling per-method embeddings
# ---------------------------------------------------------------------------

def _build_method_cache(
    sample_ids: Sequence[str],
    method: str,
    vision_dir: Optional[str],
    text_dir: Optional[str],
    label: str,
) -> Dict[str, np.ndarray]:
    """Load and combine the embeddings required by *method*.

    Args:
        sample_ids: IDs to load.
        method: One of :data:`METHODS`.
        vision_dir: DreamSim embedding directory (vision/early-fusion only).
        text_dir: Longformer embedding directory (text/early-fusion only).
        label: Human-readable name used in log lines.

    Returns:
        Mapping of sample ID to a normalised ``(S, D)`` array.  IDs whose
        embeddings are absent are omitted.
    """
    use_vision = method in ("vision_only", "early_fusion")
    use_text = method in ("text_only", "early_fusion")

    vision = (
        store.load_embeddings(vision_dir, sample_ids, desc=f"{label} vision embeddings")
        if use_vision
        else {}
    )
    text = (
        store.load_embeddings(text_dir, sample_ids, desc=f"{label} text embeddings")
        if use_text
        else {}
    )

    cache: Dict[str, np.ndarray] = {}
    for sample_id in sample_ids:
        if use_vision and use_text:
            if sample_id not in vision or sample_id not in text:
                continue
            cache[sample_id] = normalize_embeddings(
                build_early_fusion_embedding(vision[sample_id], text[sample_id])
            )
        elif use_vision:
            if sample_id not in vision:
                continue
            cache[sample_id] = normalize_embeddings(vision[sample_id])
        else:
            if sample_id not in text:
                continue
            cache[sample_id] = normalize_embeddings(text[sample_id])

    missing = len(sample_ids) - len(cache)
    if missing:
        logger.warning(f"{label}: {missing} of {len(sample_ids)} samples have no embedding")
    return cache


def _is_single_view(cache: Dict[str, np.ndarray]) -> bool:
    """True if every embedding in *cache* has exactly one view."""
    return all(emb.shape[0] == 1 for emb in cache.values())


# ---------------------------------------------------------------------------
# k-NN search
# ---------------------------------------------------------------------------

def _knn_matrix(
    query: np.ndarray,
    pool_matrix: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k cosine similarity against a stacked pool matrix.

    Both inputs must be L2-normalised, so the dot product is the cosine.

    Args:
        query: Normalised query of shape ``(D,)``.
        pool_matrix: Normalised pool of shape ``(N, D)``.
        k: Number of neighbours.

    Returns:
        ``(indices, similarities)``, sorted descending by similarity.
    """
    sims = pool_matrix @ query
    k = min(k, len(sims))
    top_k = np.argpartition(-sims, k - 1)[:k]
    top_k = top_k[np.argsort(-sims[top_k])]
    return top_k, sims[top_k]


def _knn_multi_view(
    query: np.ndarray,
    pool_cache: Dict[str, np.ndarray],
    pool_ids: List[str],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k slice-aligned similarity for multi-view (volumetric) embeddings.

    Args:
        query: Normalised query of shape ``(S_q, D)``.
        pool_cache: Normalised pool embeddings by ID.
        pool_ids: Ordered IDs defining index positions.
        k: Number of neighbours.

    Returns:
        ``(indices, similarities)``, sorted descending by similarity.
    """
    sims = np.array(
        [compute_slice_aligned_similarity(query, pool_cache[sid]) for sid in pool_ids]
    )
    k = min(k, len(sims))
    top_k = np.argpartition(-sims, k - 1)[:k]
    top_k = top_k[np.argsort(-sims[top_k])]
    return top_k, sims[top_k]


# ---------------------------------------------------------------------------
# Main alignment entry point
# ---------------------------------------------------------------------------

def run_knn_alignment(
    pool_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    method: str,
    k: int,
    id_column: str = "sample_id",
    vision_embeddings_dir: Optional[str] = None,
    text_embeddings_dir: Optional[str] = None,
    pool_vision_embeddings_dir: Optional[str] = None,
    pool_text_embeddings_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run reference-guided kNN alignment to filter a training pool.

    For each reference sample the function retrieves the *k* most similar
    candidates from the pool.  The pool is then deduplicated: each sample
    is kept once with the similarity score equal to its best match across
    all reference queries.

    Args:
        pool_df: DataFrame for the training pool (D0).  Must contain
            ``id_column``.
        ref_df: DataFrame for the reference set (D_ref).  Must contain
            ``id_column``.
        method: One of :data:`METHODS`.
        k: Number of nearest neighbours per reference sample.
        id_column: Name of the unique sample identifier column.
        vision_embeddings_dir: DreamSim directory for the *reference* set.
            Falls back to ``pool_vision_embeddings_dir``.
        text_embeddings_dir: Longformer directory for the *reference* set.
            Falls back to ``pool_text_embeddings_dir``.
        pool_vision_embeddings_dir: DreamSim directory for the *pool*.
            Defaults to ``vision_embeddings_dir``.
        pool_text_embeddings_dir: Longformer directory for the *pool*.
            Defaults to ``text_embeddings_dir``.

    Returns:
        Deduplicated DataFrame of aligned pool samples, sorted by descending
        score.  Added columns:

        - ``similarity_score``: max cosine similarity to any reference sample.
        - ``ref_{id_column}``: identifier of the closest reference match.
        - ``neighbor_rank``: rank within its matched reference query
          (1 = closest).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(METHODS)}")

    # Reference dirs default to the pool dirs and vice versa, so a single
    # --vision-embeddings-dir / --text-embeddings-dir covers the common case
    # where pool and reference embeddings live together.
    ref_vision_dir = vision_embeddings_dir or pool_vision_embeddings_dir
    ref_text_dir = text_embeddings_dir or pool_text_embeddings_dir
    pool_vision_dir = pool_vision_embeddings_dir or vision_embeddings_dir
    pool_text_dir = pool_text_embeddings_dir or text_embeddings_dir

    pool_ids = pool_df[id_column].astype(str).tolist()
    ref_ids = ref_df[id_column].astype(str).tolist()

    logger.info(f"Loading pool embeddings ({method})…")
    pool_cache = _build_method_cache(pool_ids, method, pool_vision_dir, pool_text_dir, "Pool")
    if not pool_cache:
        raise RuntimeError("No pool embeddings could be loaded.")

    logger.info(f"Loading reference embeddings ({method})…")
    ref_cache = _build_method_cache(ref_ids, method, ref_vision_dir, ref_text_dir, "Reference")
    if not ref_cache:
        raise RuntimeError("No reference embeddings could be loaded.")

    valid_pool_ids = [sid for sid in pool_ids if sid in pool_cache]

    # Single-view pools (chest X-rays, and text embeddings for any modality)
    # collapse to one matrix product per reference sample.  Multi-view pools
    # (CT volumes) need the slice-aligned comparison.
    single_view = _is_single_view(pool_cache) and _is_single_view(ref_cache)
    if single_view:
        pool_matrix = np.vstack([pool_cache[sid] for sid in valid_pool_ids])
        logger.info(f"Single-view pool: {pool_matrix.shape[0]} x {pool_matrix.shape[1]}")
    else:
        pool_matrix = None
        views = {emb.shape[0] for emb in pool_cache.values()}
        logger.info(f"Multi-view pool: {len(valid_pool_ids)} samples, views per sample {sorted(views)}")

    logger.info(
        f"Running {method} kNN alignment (k={k}) over {len(ref_cache)} reference samples…"
    )
    aligned_records = []

    for ref_id in tqdm(ref_ids, desc="Reference kNN"):
        if ref_id not in ref_cache:
            continue
        ref_emb = ref_cache[ref_id]

        if single_view:
            nn_indices, nn_sims = _knn_matrix(ref_emb.reshape(-1), pool_matrix, k)
        else:
            nn_indices, nn_sims = _knn_multi_view(ref_emb, pool_cache, valid_pool_ids, k)

        for rank, (idx, sim) in enumerate(zip(nn_indices, nn_sims), start=1):
            aligned_records.append(
                {
                    id_column: valid_pool_ids[idx],
                    f"ref_{id_column}": ref_id,
                    "similarity_score": float(sim),
                    "neighbor_rank": rank,
                }
            )

    if not aligned_records:
        logger.error("No aligned records produced.")
        return pd.DataFrame()

    aligned_df = pd.DataFrame(aligned_records)
    logger.info(f"Pre-deduplication: {len(aligned_df)} (pool_id, ref_id) pairs")

    # Keep each pool sample once, at its best (lowest) neighbour rank.
    aligned_df = aligned_df.sort_values([id_column, "neighbor_rank"])
    aligned_df = aligned_df.drop_duplicates(subset=[id_column], keep="first")
    aligned_df = aligned_df.sort_values("similarity_score", ascending=False)
    aligned_df = aligned_df.reset_index(drop=True)

    # Attach the pool metadata columns.
    pool_meta = pool_df.copy()
    pool_meta[id_column] = pool_meta[id_column].astype(str)
    pool_meta = pool_meta.set_index(id_column)
    aligned_df = aligned_df.join(pool_meta, on=id_column, how="left")

    logger.info(f"Post-deduplication: {len(aligned_df)} unique pool samples retained")
    logger.info(
        f"Similarity score — mean: {aligned_df['similarity_score'].mean():.4f}, "
        f"min: {aligned_df['similarity_score'].min():.4f}, "
        f"max: {aligned_df['similarity_score'].max():.4f}"
    )
    return aligned_df
