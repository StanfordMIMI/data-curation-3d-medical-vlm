"""
Vision backbone for the CXR SigLIP model.

Vendored from the CheXagent project (Stanford AIMI), trimmed to the two
classes this repository uses: :class:`TransformCXR`, the albumentations
augmentation stack applied to radiographs, and :class:`CLIPModel`, a SigLIP
vision tower followed by a sinusoidal-position-embedded MLP projection head.

Source: https://github.com/Stanford-AIMI/CheXagent
        ``standalone/chexagent_v3/models/modeling_visual.py``

Padding arguments use the albumentations >= 2.0 names (``border_mode`` /
``fill``); upstream passes the pre-2.0 ``value`` / ``pad_mode`` / ``pad_val``,
which 2.x silently ignores. The values are the same constant black border, so
this is a rename rather than a behaviour change.

Dropped relative to upstream: XrayReader (arrow-file loader), Resampler
(unused — CLIPModel projects with an MLP instead), DINOv2Model,
VisionEnsembleModel, MultiScaleCLIPModel, and CLIPModel.load_image / .encode
(the dataset performs its own image loading and transforms).

See the NOTICE file at the repository root for attribution.
"""

import math
from functools import partial

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoProcessor
from transformers.activations import ACT2FN


class TransformCXR(object):
    def __init__(
            self,
            image_size=448,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            allow_shift=True,
            training=True,
            normalize=True
    ):

        resize_size = image_size
        p_train = 0.5
        shift_limit = (-0.0, 0.0)
        scale_limit = (-0.1, -0.02)
        rotate_limit = 5
        scale = (0.00, 0.01)
        brightness_limit = (-0.15, 0.15)
        contrast_limit = (-0.05, 0.05)
        pad_mode = cv2.BORDER_CONSTANT
        pad_val = (0, 0, 0)

        if training:
            if allow_shift:
                transform_list = [
                    A.ShiftScaleRotate(
                        shift_limit=shift_limit, scale_limit=scale_limit,
                        rotate_limit=rotate_limit, border_mode=pad_mode, fill=pad_val,
                        p=p_train
                    ),
                    A.Perspective(
                        scale=scale, border_mode=pad_mode, fill=pad_val, p=p_train
                    ),
                    A.Resize(height=resize_size, width=resize_size, interpolation=cv2.INTER_CUBIC),
                    A.RandomCrop(height=image_size, width=image_size),
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit, contrast_limit=contrast_limit,
                        p=p_train
                    )
                ]
            else:
                transform_list = [
                    A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit, contrast_limit=contrast_limit,
                        p=p_train
                    )
                ]
        else:
            transform_list = [
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC)
            ]

        if normalize:
            transform_list += [A.Normalize(mean=mean, std=std), ToTensorV2(transpose_mask=True)]

        self.transforms = A.Compose(transform_list)

    def __call__(self, image):
        image = np.array(image)
        return self.transforms(image=image)['image']


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class CLIPModel(nn.Module):

    def __init__(
            self,
            image_size: int,
            n_queries: int = 256,
            output_dim: int = 512,
            vision_model_name_or_path: str = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli",
            **kwargs
    ):
        super().__init__()
        # load model and processor
        self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model
        self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor

        # set constants
        self.image_height, self.image_width = self.image_size = (image_size, image_size)
        width = self.model.config.hidden_size
        patch_height, patch_width = self.model.embeddings.patch_embedding.kernel_size
        self.grid_size = (self.image_height // patch_height, self.image_width // patch_width)
        self.output_dim = output_dim

        # Transforms
        self.mean = self.processor.image_mean
        self.std = self.processor.image_std
        self.image_transform_train = TransformCXR(image_size=image_size, mean=self.mean, std=self.std, training=True)
        self.image_transform_train_no_shift = TransformCXR(
            image_size=image_size, mean=self.mean, std=self.std, allow_shift=False, training=True
        )
        self.image_transform_val = TransformCXR(image_size=image_size, mean=self.mean, std=self.std, training=False)
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])


        # MLP
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(width, self.grid_size[0])).float()
        ).requires_grad_(False)
        self.attn_pool = nn.Sequential(
            nn.Linear(width, output_dim * 4, bias=True),
            ACT2FN["gelu"],
            nn.Linear(output_dim * 4, output_dim, bias=True)
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter((output_dim ** -0.5) * torch.randn(output_dim, output_dim), requires_grad=True)


    def forward_resampler(self, x):
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))
        x = x + pos_embed.unsqueeze(0)
        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def forward(self, x: torch.Tensor):
        # get feature
        x = self.model(x, output_hidden_states=True).hidden_states[-1]

        x = self.forward_resampler(x)
        return x

