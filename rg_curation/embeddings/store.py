"""
On-disk storage for pre-computed embeddings.

Two layouts are supported, and the reader auto-detects which one a directory
uses:

``per_sample``
    One file per sample, named ``{sample_id}.pt``.  Convenient for the CT
    pool (15k-35k volumes) and for resuming interrupted jobs.

``batched``
    Many samples per file, named ``{prefix}_batch_{n:04d}.pt``.  Each file
    holds a list of dicts with at least ``"id"`` and ``"embedding"`` keys.
    Preferred for the CXR pool (58k images), where one file per sample would
    put ~58k inodes on a parallel filesystem.

Both layouts decode to the same in-memory representation: a float32 array of
shape ``(S, D)``, where ``S`` is the number of encoded views per sample.  CT
volumes have one row per extracted axial slice; 2D chest X-rays have exactly
one row.  Keeping the shape uniform lets the alignment code treat both
modalities identically.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

LAYOUT_PER_SAMPLE = "per_sample"
LAYOUT_BATCHED = "batched"

_BATCH_GLOB = "*_batch_*.pt"
_DEFAULT_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_layout(embeddings_dir: str) -> str:
    """Determine whether *embeddings_dir* uses the batched or per-sample layout.

    Args:
        embeddings_dir: Directory containing ``.pt`` embedding files.

    Returns:
        Either :data:`LAYOUT_BATCHED` or :data:`LAYOUT_PER_SAMPLE`.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no ``.pt``
            files.
    """
    if not os.path.isdir(embeddings_dir):
        raise FileNotFoundError(f"Embedding directory does not exist: {embeddings_dir}")

    if glob.glob(os.path.join(embeddings_dir, _BATCH_GLOB)):
        return LAYOUT_BATCHED

    if glob.glob(os.path.join(embeddings_dir, "*.pt")):
        return LAYOUT_PER_SAMPLE

    raise FileNotFoundError(f"No .pt embedding files found in: {embeddings_dir}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _to_2d_float32(value) -> np.ndarray:
    """Coerce a stored embedding payload into a float32 ``(S, D)`` array.

    Accepts the three shapes that occur on disk: a list of per-slice dicts
    (CT), a single dict with an ``"embedding"`` key (text), or a bare
    tensor/array (batched entries).
    """
    if isinstance(value, list):
        # Per-sample DreamSim file: [{"embedding": Tensor(D,), "slice_idx": int}, ...]
        rows = [_to_2d_float32(entry) for entry in value]
        return np.concatenate(rows, axis=0)

    if isinstance(value, dict):
        if "embedding" not in value:
            raise ValueError(f"Embedding payload has no 'embedding' key: {sorted(value)}")
        return _to_2d_float32(value["embedding"])

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported embedding shape: {array.shape}")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_embeddings(
    embeddings_dir: str,
    sample_ids: Optional[Sequence[str]] = None,
    layout: Optional[str] = None,
    desc: str = "Loading embeddings",
) -> Dict[str, np.ndarray]:
    """Load embeddings from *embeddings_dir* into memory.

    Args:
        embeddings_dir: Directory holding the ``.pt`` files.
        sample_ids: Restrict loading to these IDs.  For the batched layout
            every file must still be opened, but unwanted entries are dropped
            immediately rather than retained.  ``None`` loads everything
            present (batched) or nothing (per-sample, where IDs are required).
        layout: Force a layout instead of auto-detecting it.
        desc: Progress-bar label.

    Returns:
        Mapping of sample ID to a float32 array of shape ``(S, D)``.

    Raises:
        ValueError: If the per-sample layout is used without ``sample_ids``.
    """
    layout = layout or detect_layout(embeddings_dir)
    wanted: Optional[Set[str]] = set(sample_ids) if sample_ids is not None else None

    if layout == LAYOUT_BATCHED:
        return _load_batched(embeddings_dir, wanted, desc)

    if wanted is None:
        raise ValueError(
            "sample_ids is required for the per-sample layout "
            f"(directory: {embeddings_dir})"
        )
    return _load_per_sample(embeddings_dir, wanted, desc)


def _load_batched(
    embeddings_dir: str,
    wanted: Optional[Set[str]],
    desc: str,
) -> Dict[str, np.ndarray]:
    """Read every ``*_batch_*.pt`` file, keeping the requested IDs."""
    batch_files = sorted(glob.glob(os.path.join(embeddings_dir, _BATCH_GLOB)))
    logger.info(f"Reading {len(batch_files)} batch files from {embeddings_dir}")

    out: Dict[str, np.ndarray] = {}
    corrupt = 0

    for path in tqdm(batch_files, desc=desc):
        try:
            entries = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(f"Skipping unreadable batch file {path}: {exc}")
            corrupt += 1
            continue

        for entry in entries:
            sample_id = entry.get("id")
            if sample_id is None:
                continue
            if wanted is not None and sample_id not in wanted:
                continue
            out[sample_id] = _to_2d_float32(entry["embedding"])

    if corrupt:
        logger.warning(f"{corrupt} batch file(s) could not be read")
    logger.info(f"Loaded {len(out)} embeddings from {embeddings_dir}")
    return out


def _load_per_sample(
    embeddings_dir: str,
    wanted: Set[str],
    desc: str,
) -> Dict[str, np.ndarray]:
    """Read one ``{sample_id}.pt`` file per requested ID."""
    out: Dict[str, np.ndarray] = {}
    missing: List[str] = []

    for sample_id in tqdm(sorted(wanted), desc=desc):
        path = os.path.join(embeddings_dir, f"{sample_id}.pt")
        if not os.path.exists(path):
            missing.append(sample_id)
            continue
        try:
            out[sample_id] = _to_2d_float32(
                torch.load(path, map_location="cpu", weights_only=False)
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load embedding for {sample_id}: {exc}") from exc

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} embedding file(s) missing in {embeddings_dir} "
            f"(e.g. {preview}). Generate them before running alignment."
        )
    return out


def existing_sample_ids(embeddings_dir: str) -> Set[str]:
    """Return the IDs already present in *embeddings_dir*.

    Used by the generation scripts to make embedding jobs resumable.  A
    missing directory yields an empty set rather than an error.
    """
    if not os.path.isdir(embeddings_dir):
        return set()

    batch_files = glob.glob(os.path.join(embeddings_dir, _BATCH_GLOB))
    if batch_files:
        ids: Set[str] = set()
        for path in batch_files:
            try:
                entries = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                logger.warning(f"Skipping unreadable batch file {path}: {exc}")
                continue
            ids.update(e["id"] for e in entries if e.get("id") is not None)
        return ids

    return {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(embeddings_dir, "*.pt"))
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class BatchedEmbeddingWriter:
    """Accumulate embedding records and flush them in fixed-size ``.pt`` files.

    Files are named ``{prefix}_batch_{n:04d}.pt`` and hold a list of the dicts
    passed to :meth:`add`.  Use as a context manager so the final partial
    batch is always flushed::

        with BatchedEmbeddingWriter(out_dir, prefix="train") as writer:
            for sample_id, emb in ...:
                writer.add({"id": sample_id, "embedding": emb})

    Args:
        output_dir: Destination directory; created if absent.
        prefix: Filename prefix, typically the split name.
        batch_size: Records per file (default: 1000).
        start_index: First batch number.  Set this past any existing batches
            when appending to a directory, so nothing is overwritten.
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str = "embeddings",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        start_index: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)

        self._buffer: List[dict] = []
        self._index = start_index if start_index is not None else self._next_free_index()
        self.files_written = 0
        self.records_written = 0

    def _next_free_index(self) -> int:
        """Return the first batch number not already used by *prefix*."""
        existing = glob.glob(os.path.join(self.output_dir, f"{self.prefix}_batch_*.pt"))
        if not existing:
            return 0
        numbers = []
        for path in existing:
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                numbers.append(int(stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 0

    def add(self, record: dict) -> None:
        """Queue one record, flushing automatically at ``batch_size``."""
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Write any buffered records to a new batch file."""
        if not self._buffer:
            return
        path = os.path.join(self.output_dir, f"{self.prefix}_batch_{self._index:04d}.pt")
        torch.save(self._buffer, path)
        logger.info(f"Wrote {len(self._buffer)} embeddings to {os.path.basename(path)}")
        self.records_written += len(self._buffer)
        self.files_written += 1
        self._buffer = []
        self._index += 1

    def __enter__(self) -> "BatchedEmbeddingWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()


def save_per_sample(record: dict, output_dir: str, sample_id: str) -> str:
    """Write a single ``{sample_id}.pt`` file and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{sample_id}.pt")
    torch.save(record, path)
    return path
