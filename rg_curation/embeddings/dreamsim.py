"""
DreamSim embedding generation for CT volumes and 2D medical images.

For 3D CT volumes the encoder is applied independently to multiple center
slices.  Each resulting embedding is stored as a separate entry so that the
downstream alignment step can perform slice-aligned cosine similarity.

For 2D images (e.g. chest X-rays) a single embedding is produced and stored in
the same format as a one-element list, keeping the interface uniform across
modalities.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def setup_dreamsim_model(
    dreamsim_type: str = "ensemble",
    compile_model: bool = True,
) -> Tuple:
    """Load and configure the DreamSim model.

    Args:
        dreamsim_type: DreamSim variant to load (default: ``"ensemble"``).
        compile_model: Whether to apply ``torch.compile`` for GPU inference
            (default: ``True``).  Silently skipped on CPU or if compilation
            fails.

    Returns:
        Tuple of ``(model, preprocess, device)``.
    """
    from dreamsim import dreamsim as _dreamsim

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess = _dreamsim(pretrained=True, device=device, dreamsim_type=dreamsim_type)

    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            logger.info("torch.compile applied to DreamSim model")
        except Exception as exc:
            logger.warning(f"torch.compile skipped: {exc}")

    model.eval()
    logger.info(f"DreamSim ({dreamsim_type}) loaded on {device}")
    return model, preprocess, device


def _slices_to_tensors(
    slices: np.ndarray,
    preprocess,
    convert_fn,
) -> torch.Tensor:
    """Convert a stack of 2D NumPy slices to a single batched tensor.

    Args:
        slices: Array of shape ``(N, H, W)``.
        preprocess: DreamSim preprocessing callable.
        convert_fn: Function mapping a 2D NumPy array to a PIL RGB image.

    Returns:
        Float tensor of shape ``(N, C, H, W)``.
    """
    return torch.stack([preprocess(convert_fn(s)) for s in slices])


def compute_dreamsim_embeddings_ct(
    slices: np.ndarray,
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Compute DreamSim embeddings for a stack of 2D CT slice arrays.

    Args:
        slices: NumPy array of shape ``(N, H, W)`` containing pre-processed
            (e.g. center-mass-cropped) CT slices.
        model: DreamSim model with an ``embed`` method.
        preprocess: DreamSim preprocessing transform.
        device: Target device.
        batch_size: Number of slices per GPU batch (default: 128).

    Returns:
        Float16 tensor of shape ``(N, D)`` where ``D`` is the embedding
        dimension.
    """
    from rg_curation.utils.ct_preprocessing import convert_slice_to_rgb

    all_embeddings: List[torch.Tensor] = []
    n = len(slices)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = slices[start : start + batch_size]
            imgs = _slices_to_tensors(batch, preprocess, convert_slice_to_rgb)

            # DreamSim may return a (B, 1, C, H, W) tensor
            if imgs.dim() == 5:
                b, _, c, h, w = imgs.shape
                imgs = imgs.view(b, c, h, w)

            imgs = imgs.to(device, non_blocking=True)
            embs = model.embed(imgs).to(torch.float16)
            all_embeddings.append(embs.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_dreamsim_embedding_2d(
    image: "str | Image.Image",
    model,
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    """Compute a single DreamSim embedding for a 2D image (e.g. chest X-ray).

    Args:
        image: An already-loaded PIL image, or a path to a PNG/JPEG file.
            Chest X-rays stored as DICOM should be loaded first with
            :func:`rg_curation.utils.cxr_preprocessing.load_cxr_image`.
        model: DreamSim model with an ``embed`` method.
        preprocess: DreamSim preprocessing transform.
        device: Target device.

    Returns:
        Float16 tensor of shape ``(1, D)``.
    """
    img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
    tensor = preprocess(img)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.embed(tensor).to(torch.float16)

    return emb.cpu()
