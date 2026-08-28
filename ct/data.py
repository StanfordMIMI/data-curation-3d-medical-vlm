"""Dataset construction for CT CLIP training and evaluation.

The metadata CSV needs ``anon_accession``, ``image_file``, ``findings`` and
``split`` columns (splits 0-2 = train, 3 = val, 4 = test). A curated training
CSV additionally carries the curation score (``rrf_score`` or
``similarity_score``) used to rank samples when a training budget is applied.

Image paths are resolved by filename against one or more image directories,
so a CSV written on another machine keeps working — and the Mixed pool
(Merlin + Tri-Valley) is handled by simply passing both directories.
"""

import glob
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

import monai
import pandas as pd
import torch
from monai.data import DataLoader
from monai.data.utils import SUPPORTED_PICKLE_MOD
from monai.utils import look_up_option
from tqdm import tqdm

from ct.sections import build_findings_list
from rg_curation.utils.monai_transforms import transforms_image

SPLIT_INDICES = {"train": (0, 1, 2), "val": (3,), "test": (4,)}

# Curation score columns, in lookup order. `rrf_score` wins when both are
# present because RRF outputs may also carry the per-modality scores.
SCORE_COLUMNS = ("rrf_score", "similarity_score")


def resolve_image_paths(
    data_frame: pd.DataFrame, image_dirs: Sequence[str]
) -> pd.DataFrame:
    """Replace ``image_file`` with actual paths found in ``image_dirs``.

    Matching is by filename, and rows whose image is not found in any
    directory are dropped (with a warning).
    """
    data_frame = data_frame.copy()
    data_frame["filename"] = data_frame["image_file"].map(os.path.basename)

    found = []
    for image_dir in image_dirs:
        found.extend(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    found_df = pd.DataFrame({"image_file": found})
    found_df["filename"] = found_df["image_file"].map(os.path.basename)

    resolved = pd.merge(
        data_frame.drop(columns=["image_file"]), found_df, on="filename", how="inner"
    ).drop(columns=["filename"])

    missing = len(data_frame) - len(resolved)
    if missing > 0:
        print(f"Warning: dropped {missing} rows whose image file was not found.")
    return resolved


def load_metadata(
    metadata_path: str, image_dirs: Sequence[str], split: str
) -> pd.DataFrame:
    """Load a metadata CSV, resolve image paths, and filter to one split.

    A CSV without a ``split`` column (e.g. a curated training CSV that holds
    only training rows) is used as-is.
    """
    data_frame = pd.read_csv(metadata_path)
    data_frame = resolve_image_paths(data_frame, image_dirs)
    if "split" in data_frame.columns:
        data_frame = data_frame[data_frame["split"].isin(SPLIT_INDICES[split])]
    else:
        print(f"No 'split' column in {metadata_path}; using all rows as {split}.")
    return data_frame.reset_index(drop=True)


def select_training_subset(
    data_frame: pd.DataFrame, num_samples: Optional[int], seed: int = 17
) -> pd.DataFrame:
    """Apply the training budget.

    With a curation score column present, the top ``num_samples`` rows by
    score are kept; otherwise (an uncurated pool) the subset is sampled
    uniformly at random with ``seed`` — the paper's random baseline.
    """
    if num_samples is None or num_samples >= len(data_frame):
        return data_frame

    for score_column in SCORE_COLUMNS:
        if score_column in data_frame.columns:
            print(f"Selecting top {num_samples} samples by {score_column}.")
            return (
                data_frame.sort_values(by=score_column, ascending=False)
                .head(num_samples)
                .reset_index(drop=True)
            )

    print(f"No curation score column found; sampling {num_samples} rows at random.")
    return data_frame.sample(n=num_samples, random_state=seed).reset_index(drop=True)


def build_datalist(data_frame: pd.DataFrame, include_sections: bool = False) -> List[dict]:
    """Convert a metadata frame into MONAI-style data dictionaries.

    Args:
        data_frame: Frame with resolved image paths.
        include_sections: Whether to precompute the per-sample
            ``findings_list`` (full report + organ-system sections) used by
            the training-time text sampling schedule.
    """
    datalist = []
    rows = data_frame.itertuples(index=False)
    if include_sections:
        rows = tqdm(rows, total=len(data_frame), desc="Parsing findings sections")
    for row in rows:
        item = {
            "image": row.image_file,
            "findings": row.findings,
            "anon_accession": row.anon_accession,
        }
        if include_sections:
            item["findings_list"] = build_findings_list(row.findings)
        datalist.append(item)
    print(f"Number of samples in datalist: {len(datalist)}")
    return datalist


class CTPersistentDataset(monai.data.PersistentDataset):
    """PersistentDataset that caches the transformed volume per image path.

    The cache key is derived from the image path alone (not the transform
    hash), so a cache directory is shared across runs that use the same
    preprocessing — which all experiments in the paper do.
    """

    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        print(f"Size of dataset: {len(self)}\n")

    def _cachecheck(self, item_transformed):
        item = deepcopy(item_transformed)
        image_data = {"image": item_transformed.get("image")}

        hashfile = None
        if self.cache_dir is not None:
            key = self.hash_func(image_data).decode("utf-8")
            hashfile = self.cache_dir / f"{key}.pt"

        if hashfile is not None and hashfile.is_file():
            item["image"] = torch.load(hashfile)
            return item

        transformed_image = self._pre_transform(image_data)["image"]
        item["image"] = transformed_image
        if hashfile is None:
            return item

        try:
            # Write to a temporary file and rename, so a killed process never
            # leaves a partially written cache entry behind.
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=transformed_image,
                    f=temp_hash_file,
                    pickle_module=look_up_option(
                        self.pickle_module, SUPPORTED_PICKLE_MOD
                    ),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:
            pass
        return item

    def _transform(self, index: int):
        return self._post_transform(self._cachecheck(self.data[index]))


def build_dataloader(
    datalist: List[dict],
    batch_size: int,
    shuffle: bool = False,
    cache_dir: Optional[str] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a dataloader over CT volumes with the fixed transform pipeline."""
    dataset = CTPersistentDataset(
        data=datalist, transform=transforms_image, cache_dir=cache_dir
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
