"""SigLIP-based vision-language model for chest X-ray analysis.

A SigLIP vision tower is trained to align with a SigLIP text encoder under
the sigmoid (binary cross-entropy) contrastive loss. This is the downstream
model used for the CXR experiments; the curated training subset selected by
``rg_curation`` is what varies between runs.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from cxr.models.visual import CLIPModel


def apply_interpolation_patch():
    """Patch CLIPModel.forward to enable position embedding interpolation for arbitrary image sizes."""
    original_forward = CLIPModel.forward
    
    def patched_forward(self, x):
        output = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            x = output.hidden_states[-1]
        else:
            x = output.last_hidden_state
        x = self.forward_resampler(x)
        return x
    
    CLIPModel.forward = patched_forward


class VisionOnlySiglipModel(nn.Module):
    """
    Vision-language model using SigLIP architecture for contrastive learning.
    
    Trains a vision encoder to align with text embeddings from a pre-trained
    SigLIP text encoder using binary cross-entropy loss.
    """
    
    def __init__(
        self,
        vision_model_name_or_path: str,
        text_model_name_or_path: str,
        image_size: int,
        n_queries: int,
        projection_dim: int = None,
        freeze_text_encoder: bool = False,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        
        # Initialize text encoder to determine embedding dimension
        self.text_encoder = AutoModel.from_pretrained(text_model_name_or_path).text_model
        text_hidden = self._get_text_hidden_size()
        
        # Auto-configure projection dimension if not specified
        if projection_dim is None:
            projection_dim = text_hidden
        
        # Initialize vision encoder with resampler
        self.visual = CLIPModel(
            image_size=image_size,
            n_queries=n_queries,
            output_dim=projection_dim,
            vision_model_name_or_path=vision_model_name_or_path,
        )
        self.visual.model.config.output_hidden_states = True
        
        # Setup text projection layer (Identity if dimensions match, else Linear)
        if projection_dim != text_hidden:
            print(f"Projecting text embeddings: {text_hidden}d → {projection_dim}d")
            self.text_proj = nn.Linear(text_hidden, projection_dim, bias=False)
        else:
            self.text_proj = nn.Identity()
        
        # Optionally freeze text encoder, by default not frozen
        if freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
        
        # Learnable temperature parameter for contrastive loss
        temperature = torch.tensor(1.0 / max(temperature_init, 1e-4))
        self.logit_scale = nn.Parameter(torch.log(temperature))
    
    def _get_text_hidden_size(self) -> int:
        """Extract hidden size from text encoder config."""
        text_hidden = getattr(self.text_encoder.config, "hidden_size", None)
        if text_hidden is None and hasattr(self.text_encoder.config, "text_config"):
            text_hidden = getattr(self.text_encoder.config.text_config, "hidden_size", None)
        if text_hidden is None:
            raise AttributeError(
                "Text encoder config does not expose hidden_size. "
                "Please update the adapter logic for this backbone."
            )
        return text_hidden
    
    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized embeddings."""
        image_tokens = self.visual(pixel_values)
        image_embeddings = image_tokens.mean(dim=1)
        return F.normalize(image_embeddings, dim=-1)
    
    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to normalized embeddings using mean pooling."""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        # Mean pooling with attention mask
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        text_embeddings = self.text_proj(pooled)
        return F.normalize(text_embeddings, dim=-1)
    
    def forward(self, pixel_values, input_ids, attention_mask, **kwargs):
        """
        Forward pass computing SigLIP contrastive loss.
        
        Returns:
            dict: Contains 'loss', 'logits', 'image_embeds', and 'text_embeds'
        """
        image_embeds = self._encode_image(pixel_values)
        text_embeds = self._encode_text(input_ids, attention_mask)
        
        # Compute scaled similarity logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_embeds @ text_embeds.t())
        
        # SigLIP uses binary cross-entropy with diagonal targets
        labels = torch.eye(logits.size(0), device=logits.device, dtype=logits.dtype)
        loss_i2t = F.binary_cross_entropy_with_logits(logits, labels)
        loss_t2i = F.binary_cross_entropy_with_logits(logits.t(), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        
        return {
            "loss": loss,
            "logits": logits,
            "image_embeds": image_embeds.detach(),
            "text_embeds": text_embeds.detach(),
        }
