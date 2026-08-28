"""Inflated 3D ResNet (I3ResNet) vision encoder.

Adapted from hassony2/inflated_convnets_pytorch (MIT License, see NOTICE):
a 2D torchvision ResNet is "inflated" into a 3D network by repeating each
convolution kernel along the depth axis, so the 3D encoder starts from
ImageNet-pretrained weights.

The variant kept here is the configuration used for the paper's CT
experiments (``conv_class`` in the upstream code): the pooled features feed
a 512-dim contrastive head and an auxiliary linear classification head.
Parameter names match the original training code so published checkpoints
load without remapping.
"""

import torch
import torch.utils.checkpoint as checkpoint

from ct.models import inflate


class Bottleneck3d(torch.nn.Module):
    """A torchvision ResNet bottleneck block, inflated to 3D."""

    def __init__(self, bottleneck2d):
        super().__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = _inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        if x.requires_grad:
            out = checkpoint.checkpoint(run_function, x, use_reentrant=False)
        else:
            out = run_function(x)

        out = out + residual
        out = self.relu(out)
        return out


def _inflate_reslayer(reslayer2d):
    return torch.nn.Sequential(*[Bottleneck3d(layer2d) for layer2d in reslayer2d])


def _inflate_downsample(downsample2d, time_stride=1):
    return torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        inflate.inflate_batch_norm(downsample2d[1]),
    )


class I3ResNet(torch.nn.Module):
    """3D ResNet inflated from a 2D torchvision ResNet.

    Args:
        resnet2d: The pretrained 2D ResNet to inflate (e.g. ResNet-152).
        class_nb: Output dimension of the auxiliary classification head. The
            head is part of the published checkpoints, so it is kept even
            though the contrastive objective does not train it.
        hidden_dim: Channel width of the final residual stage (2048 for
            bottleneck ResNets).
    """

    def __init__(self, resnet2d, class_nb: int = 1692, hidden_dim: int = 2048):
        super().__init__()

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True
        )
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
        )

        self.layer1 = _inflate_reslayer(resnet2d.layer1)
        self.layer2 = _inflate_reslayer(resnet2d.layer2)
        self.layer3 = _inflate_reslayer(resnet2d.layer3)
        self.layer4 = _inflate_reslayer(resnet2d.layer4)

        self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
        self.classifier = torch.nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=class_nb,
            kernel_size=(1, 1, 1),
            bias=True,
        )
        self.contrastive_head = torch.nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=512,
            kernel_size=(1, 1, 1),
            bias=True,
        )

    def forward(self, x):
        """Encode a batch of CT volumes.

        Args:
            x: Tensor of shape ``(B, 1, H, W, D)`` as produced by the MONAI
                transform pipeline.

        Returns:
            Tuple ``(contrastive_features, class_logits)`` of shapes
            ``(B, 512)`` and ``(B, class_nb)``.
        """
        # (B, 1, H, W, D) -> (B, 3, D, H, W): depth becomes the temporal axis
        # and the single CT channel is repeated to match the RGB stem.
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = checkpoint.checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.layer2, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.layer3, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.layer4, x, use_reentrant=False)

        x_features = self.avgpool(x)

        class_logits = self.classifier(x_features).squeeze(3).squeeze(3).mean(2)
        contrastive = self.contrastive_head(x_features).squeeze(3).squeeze(3).mean(2)
        return contrastive, class_logits
