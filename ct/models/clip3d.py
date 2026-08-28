"""CLIP-style 3D CT vision-language model.

The paper's downstream CT model (the "Merlin" architecture): an I3ResNet-152
vision encoder inflated from ImageNet weights, paired with a Clinical
Longformer text encoder whose CLS embedding is projected to the shared
512-dim space.

Module and parameter names intentionally match the original training code
(``encode_image.i3_resnet.*``, ``encode_text.text_encoder.*``,
``encode_text.linear_layer.*``, ``logit_scale``), so checkpoints trained with
it load with ``strict=True``.
"""

import math

import torch
import torchvision
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ct.models.i3res import I3ResNet

TEXT_ENCODER_NAME = "yikuan8/Clinical-Longformer"
MAX_TEXT_TOKENS = 1024


class ImageEncoder(nn.Module):
    """I3ResNet-152 vision tower, inflated from ImageNet-pretrained weights."""

    def __init__(self, class_nb: int = 1692):
        super().__init__()
        resnet = torchvision.models.resnet152(
            weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        )
        self.i3_resnet = I3ResNet(resnet, class_nb=class_nb)

    def forward(self, image):
        """Return ``(contrastive_features, class_logits)`` for a CT batch."""
        return self.i3_resnet(image)


class TextEncoder(nn.Module):
    """Clinical Longformer text tower with a 768 -> 512 projection."""

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_ENCODER_NAME)
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, texts):
        """Embed a list of report strings into the shared 512-dim space."""
        texts = [text.lower() for text in texts]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        cls_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        return self.linear_layer(cls_embeddings)


class Clip3D(nn.Module):
    """Contrastive 3D CT vision-language model.

    Args:
        init_logit_scale: Initial value of the learnable logit scale (the
            paper's runs use ``log(10)``).
        class_nb: Output dimension of the vision tower's auxiliary
            classification head (kept for checkpoint compatibility).
    """

    def __init__(self, init_logit_scale: float = math.log(10), class_nb: int = 1692):
        super().__init__()
        self.encode_image = ImageEncoder(class_nb=class_nb)
        self.encode_text = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, texts):
        """Encode both modalities and return normalized features and logits.

        Args:
            image: CT batch of shape ``(B, 1, H, W, D)``.
            texts: List of ``B`` report strings.

        Returns:
            Tuple ``(image_features, text_features, logits_per_image,
            logits_per_text, logit_scale)``; features are L2-normalized.
        """
        image_features, _ = self.encode_image(image)
        text_features = self.encode_text(texts)

        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return (
            image_features,
            text_features,
            logits_per_image,
            logits_per_text,
            logit_scale,
        )
