"""
Legacy CLIP text/image utilities.

.. deprecated::
    This module is superseded by :mod:`rg_curation.embeddings.clip_score`.
    It is retained for reference only and is not used by the main pipeline.
"""

import warnings

warnings.warn(
    "rg_curation.utils.clip_utils is deprecated and not used by the main pipeline. "
    "Use rg_curation.embeddings.clip_score instead.",
    DeprecationWarning,
    stacklevel=2,
)

import gc
import logging
import os
import sys

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def truncate_text(text, max_words=120):
    """Truncate text to a maximum number of words."""
    if not text or pd.isna(text):
        return "No impressions available"

    words = str(text).split()
    if len(words) <= max_words:
        return text

    return ' '.join(words[:max_words]) + '...'


def extract_all_slices(ct_volume):
    """
    Extract ALL 2D slices from 3D CT volume for comprehensive CLIP processing.
    Returns all axial, sagittal, and coronal slices.
    """
    if ct_volume.ndim == 4:
        ct_volume = ct_volume[0]

    h, w, d = ct_volume.shape

    axial_slices = []
    sagittal_slices = []
    coronal_slices = []

    for z in range(d):
        axial_slices.append(ct_volume[:, :, z])

    for x in range(h):
        sagittal_slices.append(ct_volume[x, :, :])

    for y in range(w):
        coronal_slices.append(ct_volume[:, y, :])

    return axial_slices, sagittal_slices, coronal_slices


def convert_to_rgb_image(slice_2d):
    """
    Convert a 2D CT slice to RGB PIL Image for CLIP processing.
    """
    slice_norm = np.clip(slice_2d, 0, 1)
    slice_uint8 = (slice_norm * 255).astype(np.uint8)
    slice_rgb = np.stack([slice_uint8, slice_uint8, slice_uint8], axis=-1)
    pil_image = Image.fromarray(slice_rgb, 'RGB')
    return pil_image
