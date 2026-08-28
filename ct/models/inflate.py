"""Inflate 2D convolutional network layers into their 3D counterparts.

Vendored from hassony2/inflated_convnets_pytorch (MIT License, see NOTICE),
trimmed to the helpers used by :mod:`ct.models.i3res`. The numerical behavior
is kept identical to the upstream code so that pretrained 2D weights inflate
to exactly the initialization used for the paper's experiments.
"""

import torch
from torch.nn import Parameter


def inflate_conv(
    conv2d,
    time_dim=3,
    time_padding=0,
    time_stride=1,
    time_dilation=1,
    center=False,
):
    # The 7x7 stem convolution gets a fixed (3, 7, 7) kernel with spatial
    # stride 2 and no temporal stride, matching the upstream I3D recipe.
    if conv2d.kernel_size[0] == 7:
        kernel_dim = (3, 7, 7)
        padding = (1, 3, 3)
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
    else:
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])

    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride,
    )

    weight_2d = conv2d.weight.data
    if center:
        # Place the 2D filter on the central temporal slice, zero elsewhere.
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_batch_norm(batch2d):
    # BatchNorm is dimension-agnostic apart from the input-dimension check,
    # so the 2D module is reused with the 3D check patched in. (Kept as
    # upstream wrote it; the state dict is identical to BatchNorm3d.)
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(
    pool2d,
    time_dim=1,
    time_padding=0,
    time_stride=None,
    time_dilation=1,
):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        return torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
    padding = (time_padding, pool2d.padding, pool2d.padding)
    if time_stride is None:
        time_stride = time_dim
    stride = (time_stride, pool2d.stride, pool2d.stride)
    if isinstance(pool2d, torch.nn.MaxPool2d):
        dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
        return torch.nn.MaxPool3d(
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
            ceil_mode=pool2d.ceil_mode,
        )
    if isinstance(pool2d, torch.nn.AvgPool2d):
        return torch.nn.AvgPool3d(kernel_dim, stride=stride)
    raise ValueError(f"{type(pool2d)} is not among known pooling classes")
