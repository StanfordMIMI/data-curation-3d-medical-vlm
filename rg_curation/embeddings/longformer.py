"""
Clinical Longformer text embedding generation.

Uses the `yikuan8/Clinical-Longformer` model to encode radiology report
findings sections.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

_MODEL_NAME = "yikuan8/Clinical-Longformer"
_MAX_TOKENS = 4096


def setup_longformer_model() -> Tuple:
    """Load the Clinical Longformer tokenizer and model.

    Returns:
        Tuple of ``(model, tokenizer, device)``.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    logger.info(f"Clinical Longformer loaded on {device} (max context: {_MAX_TOKENS} tokens)")
    return model, tokenizer, device


def compute_longformer_embedding(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = _MAX_TOKENS,
) -> torch.Tensor:
    """Encode a single radiology report finding with Clinical Longformer.

    The CLS token hidden state is extracted and L2-normalised.

    Args:
        text: Radiology findings text.  Empty or ``NaN`` values are replaced
            with a placeholder string.
        model: Clinical Longformer model.
        tokenizer: Corresponding tokenizer.
        device: Target device.
        max_length: Maximum token length (default: 4096).

    Returns:
        1-D CPU tensor of shape ``(D,)`` with unit L2 norm.
    """
    with torch.no_grad():
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**tokens)
        # CLS token (first token) as the sequence representation
        cls_emb = outputs.last_hidden_state[0, 0, :]
        cls_emb = cls_emb / cls_emb.norm()

    return cls_emb.cpu()
