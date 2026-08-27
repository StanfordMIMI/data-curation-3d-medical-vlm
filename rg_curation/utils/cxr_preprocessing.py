"""
Image loading helpers for 2D chest radiographs.

CheXpert Plus ships radiographs both as DICOM and as rendered PNG/JPEG.  The
curation pipeline encodes the DICOMs (so the embeddings are computed from the
source pixel data), while downstream training reads the rendered images.
:func:`load_cxr_image` handles either, dispatching on file extension, so the
same metadata CSV can drive both stages by pointing at a different column.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Chest radiographs routinely exceed Pillow's decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None

_DICOM_SUFFIXES = (".dcm", ".dicom")


def load_dicom_image(dcm_path: str) -> Image.Image:
    """Load a DICOM radiograph as an 8-bit RGB image.

    Pixel values are min-max rescaled to ``[0, 255]``; an all-zero image is
    passed through rather than divided by zero.  Photometric interpretation is
    not inverted here, matching the encoder inputs used for the paper.

    Args:
        dcm_path: Path to a ``.dcm`` file.

    Returns:
        RGB :class:`PIL.Image.Image`.
    """
    import pydicom

    dcm = pydicom.dcmread(dcm_path)
    pixels = dcm.pixel_array.astype(np.float32)

    pixels -= pixels.min()
    peak = pixels.max()
    pixels = (pixels / peak * 255.0) if peak > 0 else pixels
    pixels = pixels.astype(np.uint8)

    return Image.fromarray(pixels).convert("RGB")


def load_cxr_image(image_path: str) -> Image.Image:
    """Load a chest radiograph as RGB, from DICOM or a standard image format.

    Args:
        image_path: Path to a ``.dcm``, ``.png``, or ``.jpg`` file.

    Returns:
        RGB :class:`PIL.Image.Image`.
    """
    if image_path.lower().endswith(_DICOM_SUFFIXES):
        return load_dicom_image(image_path)
    return Image.open(image_path).convert("RGB")


def resolve_image_path(image_dir: str, value: str) -> Optional[str]:
    """Resolve a metadata image reference against a root directory.

    CheXpert Plus stores nested relative paths
    (``train/patient00005/study4/view1_frontal.jpg``) whereas the CT metadata
    stores bare filenames.  The nested path is tried first; if it does not
    exist, the basename is looked up directly in *image_dir*.

    Args:
        image_dir: Root directory holding the images.
        value: Path reference from the metadata CSV.

    Returns:
        An existing path, or ``None`` if neither candidate resolves.
    """
    if os.path.isabs(value) and os.path.isfile(value):
        return value

    nested = os.path.join(image_dir, str(value).lstrip("/"))
    if os.path.isfile(nested):
        return nested

    flat = os.path.join(image_dir, os.path.basename(str(value)))
    if os.path.isfile(flat):
        return flat

    return None
