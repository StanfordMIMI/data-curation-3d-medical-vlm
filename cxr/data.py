"""
Dataset and collator for CXR SigLIP training.

Consumes the JSONL written by ``scripts/export_training_jsonl.py`` (one object
per line with ``image_path`` and ``text``), and also accepts the curated CSVs
directly.  Which subset of the pool a run trains on is decided entirely by the
curation pipeline; this module only reads whatever file it is handed, so every
budget and curation method shares one training path.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# Radiographs routinely exceed Pillow's decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None

#: Keys checked, in order, when locating the image path in a record.
IMAGE_KEYS = ("image_path", "image_file", "path_to_image")

#: Keys checked, in order, when locating the report text in a record.
TEXT_KEYS = ("text", "findings", "section_findings")


def resolve_image_path(image_path: str) -> str:
    """Resolve an image reference to the file that actually exists on disk.

    CheXpert Plus metadata refers to ``.jpg`` files while the training and
    validation splits are distributed as rendered PNGs; the test split keeps
    its original extension.  Reproducing the paper's evaluation requires the
    same substitution, so it is applied here rather than in the metadata.

    Args:
        image_path: Path as stored in the annotation file.

    Returns:
        The path with environment variables and ``~`` expanded, and with a
        ``.jpg`` suffix rewritten to ``.png`` outside the test split.
    """
    image_path = os.path.expandvars(os.path.expanduser(image_path))
    root, ext = os.path.splitext(image_path)

    if ext.lower() == ".jpg" and "test" not in image_path.lower():
        return root + ".png"

    return image_path


def load_records(path: str) -> List[dict]:
    """Load image-text pairs from a ``.jsonl``, ``.json``, or ``.csv`` file.

    Args:
        path: Annotation file path.

    Returns:
        List of record dicts.

    Raises:
        ValueError: If the file does not decode to a list of records.
    """
    lower = path.lower()

    if lower.endswith(".csv"):
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = [
                row for row in reader
                if any((value or "").strip() for value in row.values())
            ]
    elif lower.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)

    if isinstance(records, dict):
        for key in ("data", "train", "samples"):
            if key in records:
                records = records[key]
                break

    if not isinstance(records, list):
        raise ValueError(f"Unsupported annotation format in {path}")
    return records


def _first_present(sample: dict, keys) -> Optional[str]:
    """Return the first non-empty string value among *keys*."""
    for key in keys:
        value = sample.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            return value
    return None


class CXRSiglipDataset(Dataset):
    """Chest radiographs paired with the findings section of their report.

    Args:
        data_path: Annotation file (``.jsonl``, ``.json``, or ``.csv``).
        tokenizer: SigLIP tokenizer.
        image_transform: Augmentation/normalisation callable
            (:class:`cxr.models.visual.TransformCXR`).
        max_text_length: Token budget for the report text.
        image_key: Preferred image-path key; other common keys are tried as a
            fallback.
        text_key: Preferred text key; other common keys are tried as a
            fallback.
        image_root: Prefix applied to relative image paths.
        max_samples: Optional cap on dataset size, applied after filtering.
        use_top_similarity: If ``True``, the cap keeps the highest
            ``similarity_score`` records; otherwise it samples uniformly.
            Curated files are already ordered by the pipeline, so runs that
            train on a curated subset normally leave this ``False`` and pass a
            pre-truncated file.
        sampling_seed: Seed for the uniform sample.
        preload_into_memory: Decode every image once at construction instead
            of per epoch.  Faster epochs, substantially more RAM.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_transform,
        max_text_length: int = 64,
        image_key: str = "image_path",
        text_key: str = "text",
        image_root: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_top_similarity: bool = False,
        sampling_seed: int = 17,
        preload_into_memory: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_text_length = max_text_length
        self.image_key = image_key
        self.text_key = text_key
        self.image_root = Path(image_root).expanduser().resolve() if image_root else None

        self._image_keys = (image_key,) + tuple(k for k in IMAGE_KEYS if k != image_key)
        self._text_keys = (text_key,) + tuple(k for k in TEXT_KEYS if k != text_key)

        records = load_records(data_path)

        # Normalise every record onto the configured keys so downstream
        # components (notably the zero-shot evaluator) can rely on them.
        self.samples: List[dict] = []
        dropped = 0
        for record in records:
            text = _first_present(record, self._text_keys)
            image = _first_present(record, self._image_keys)
            if text is None or image is None:
                dropped += 1
                continue
            record[self.text_key] = text
            record[self.image_key] = image
            self.samples.append(record)

        if dropped:
            print(
                f"{data_path}: dropped {dropped} records without both an image "
                f"and text; {len(self.samples)} remain"
            )

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self._subsample(max_samples, use_top_similarity, sampling_seed, data_path)

        self._cache = {}
        if preload_into_memory:
            self._preload()

    def _subsample(
        self,
        max_samples: int,
        use_top_similarity: bool,
        seed: int,
        data_path: str,
    ) -> List[dict]:
        """Reduce the sample list to *max_samples* records."""
        if use_top_similarity:
            ranked = sorted(
                self.samples,
                key=lambda s: s.get("similarity_score", -float("inf")),
                reverse=True,
            )
            print(f"{data_path}: kept top {max_samples} records by similarity_score")
            return ranked[:max_samples]

        print(f"{data_path}: randomly sampled {max_samples} records (seed={seed})")
        return random.Random(seed).sample(self.samples, max_samples)

    def _preload(self) -> None:
        """Decode all images once, keeping them in memory."""
        paths = {self._image_path(s) for s in self.samples}
        print(f"Preloading {len(paths)} images into memory…")
        for i, path in enumerate(sorted(paths), start=1):
            try:
                self._cache[path] = Image.open(path).convert("RGB")
            except Exception as exc:
                print(f"Warning: could not preload {path}: {exc}")
            if i % 5000 == 0:
                print(f"  {i}/{len(paths)}")
        print(f"Preloaded {len(self._cache)} images.")

    def _image_path(self, sample: dict) -> str:
        """Absolute, extension-corrected path for one sample."""
        raw = sample[self.image_key]
        path = Path(raw)
        if not path.is_absolute() and self.image_root is not None:
            raw = str(self.image_root / str(raw).lstrip("/"))
        return resolve_image_path(str(raw))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        path = self._image_path(sample)

        image = self._cache.get(path)
        if image is None:
            image = Image.open(path).convert("RGB")

        pixel_values = self.image_transform(image)
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.from_numpy(pixel_values)

        tokenized = self.tokenizer(
            sample[self.text_key],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = (
                tokenized["input_ids"] != self.tokenizer.pad_token_id
            ).long()

        return {
            "pixel_values": pixel_values.float(),
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }


class SiglipDataCollator:
    """Stack per-sample tensors into a batch."""

    def __call__(self, features: List[dict]) -> dict:
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
