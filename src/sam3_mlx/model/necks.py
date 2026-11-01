# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""ViTDet feature-pyramid neck that turns the ViT trunk output into SAM 3 features.

Layout: the trunk output and the scale heads are NHWC (MLX convolutions are
channels-last); the neck returns NCHW levels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.vitdet import ViT
from sam3_mlx.precision_layers import AutocastConvTranspose2d


NeckOutput: TypeAlias = tuple[
    list[mx.array],
    list[mx.array],
    list[mx.array] | None,
    list[mx.array] | None,
]


def _upsample_2x(layer: nn.ConvTranspose2d, x: mx.array) -> mx.array:
    """Apply a 2x2, stride-2 ``nn.ConvTranspose2d`` as one matmul plus a pixel shuffle.

    Each input pixel maps to its own 2x2 output block. Preserve the explicit
    input cast of the BF16 autocast layer before the matrix multiplication.
    """
    weight = layer.weight  # (out, 2, 2, in)
    if isinstance(layer, AutocastConvTranspose2d):
        x = x.astype(weight.dtype)
    out_channels, kh, kw, in_channels = weight.shape
    batch, height, width = x.shape[:3]
    y = mx.reshape(x, (-1, in_channels)) @ mx.reshape(weight, (-1, in_channels)).T
    y = mx.reshape(y, (batch, height, width, out_channels, kh, kw))
    y = mx.reshape(mx.transpose(y, (0, 1, 4, 2, 5, 3)), (batch, height * kh, width * kw, out_channels))
    return y + layer.bias


class Scale4FN(nn.Module):
    """Scale head with 4x upsampling: two stride-2 transposed convs, then 1x1 and 3x3 convs.

    Input is NHWC with ``in_channels`` channels; output is NHWC ``(B, 4H, 4W,
    d_model)``.
    """

    def __init__(self, in_channels: int, d_model: int) -> None:
        """Configure the fourfold feature upsampling and projection layers."""
        super().__init__()
        self.dconv_2x2_0 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        # GELU and MaxPool2d leave ``__call__`` unannotated.
        self.gelu: Callable[[mx.array], mx.array] = nn.GELU()
        self.dconv_2x2_1 = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 4, kernel_size=2, stride=2
        )
        self.conv_1x1 = nn.Conv2d(in_channels // 4, d_model, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample features fourfold and project pyramid channels."""
        x = _upsample_2x(self.dconv_2x2_0, x)
        x = self.gelu(x)
        x = _upsample_2x(self.dconv_2x2_1, x)
        return self.conv_3x3(self.conv_1x1(x))


class Scale2FN(nn.Module):
    """Scale head with 2x upsampling (NHWC ``(B, H, W, C)`` to ``(B, 2H, 2W, d_model)``)."""

    def __init__(self, in_channels: int, d_model: int) -> None:
        """Configure the twofold feature upsampling and projection layers."""
        super().__init__()
        self.dconv_2x2 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv_1x1 = nn.Conv2d(in_channels // 2, d_model, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample features twofold and project pyramid channels."""
        x = _upsample_2x(self.dconv_2x2, x)
        return self.conv_3x3(self.conv_1x1(x))


class Scale1FN(nn.Module):
    """Scale head that keeps the resolution (NHWC ``(B, H, W, C)`` to ``(B, H, W, d_model)``)."""

    def __init__(self, in_channels: int, d_model: int) -> None:
        """Configure same-resolution pyramid projection layers."""
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Project features without changing their spatial resolution."""
        return self.conv_3x3(self.conv_1x1(x))


class Scale0_5FN(nn.Module):
    """Scale head with 2x max-pool downsampling (NHWC ``(B, H, W, C)`` to ``(B, H/2, W/2, d_model)``)."""

    def __init__(self, in_channels: int, d_model: int) -> None:
        """Configure pooled half-resolution pyramid projection layers."""
        super().__init__()
        self.maxpool_2x2: Callable[[mx.array], mx.array] = nn.MaxPool2d(
            kernel_size=2, stride=2
        )
        self.conv_1x1 = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Pool features by two and project pyramid channels."""
        x = self.maxpool_2x2(x)
        return self.conv_3x3(self.conv_1x1(x))


ScaleHead: TypeAlias = Scale4FN | Scale2FN | Scale1FN | Scale0_5FN
_SCALE_HEADS: dict[float, type[ScaleHead]] = {
    4.0: Scale4FN,
    2.0: Scale2FN,
    1.0: Scale1FN,
    0.5: Scale0_5FN,
}


class Sam3DualViTDetNeck(nn.Module):
    """Simple feature pyramid over the trunk feature map.

    One set of scale heads (``convs``) builds the SAM 3 pyramid; with
    ``add_sam2_neck`` a second, independently parameterized set (``sam2_convs``)
    builds the pyramid consumed by the interactive (SAM 1-style) predictor.

    Args:
        trunk: ViT backbone; its NHWC output feeds every head.
        position_encoding: Sine encoding of an NCHW shape, matching each level.
        d_model: Output channel count of every level.
        scale_factors: Per-level resampling; each must be one of 4.0, 2.0, 1.0,
            0.5.
        add_sam2_neck: Also build the interactive-predictor pyramid.
    """

    def __init__(
        self,
        trunk: ViT,
        position_encoding: PositionEmbeddingSine,
        d_model: int,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ) -> None:
        """Configure the visual trunk and SAM 3 and optional SAM 2 pyramid heads."""
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        dim = trunk.embed_dim

        self.convs: list[ScaleHead] = [
            _SCALE_HEADS[scale](dim, d_model) for scale in scale_factors
        ]
        self.sam2_convs: list[ScaleHead] | None = None
        if add_sam2_neck:
            self.sam2_convs = [
                _SCALE_HEADS[scale](dim, d_model) for scale in scale_factors
            ]

    def _apply_heads(
        self, x_nhwc: mx.array, convs: Sequence[ScaleHead]
    ) -> tuple[list[mx.array], list[mx.array]]:
        """Apply pyramid heads and produce matching positional encodings."""
        out: list[mx.array] = []
        pos: list[mx.array] = []
        for conv in convs:
            head_out = conv(x_nhwc)
            batch, height, width, channels = head_out.shape
            out.append(mx.transpose(head_out, axes=(0, 3, 1, 2)))
            pos.append(
                self.position_encoding((batch, channels, height, width)).astype(
                    head_out.dtype
                )
            )
        return out, pos

    def __call__(self, x: mx.array, *, scalp: int = 0) -> NeckOutput:
        """Run the trunk and the scale heads on an NCHW image batch.

        Returns ``(sam3_features, sam3_pos, sam2_features, sam2_pos)``. Each
        list has one NCHW ``(B, d_model, H_i, W_i)`` array per kept level, ordered
        as ``scale_factors``; ``*_pos`` are matching position encodings in the
        feature dtype. The last two entries are ``None`` unless the interactive
        neck exists. ``scalp`` drops that many trailing levels without running
        their heads.
        """
        x = self.trunk(x)
        keep = len(self.convs) - scalp
        sam3_out, sam3_pos = self._apply_heads(x, self.convs[:keep])
        if self.sam2_convs is None:
            return sam3_out, sam3_pos, None, None
        sam2_out, sam2_pos = self._apply_heads(x, self.sam2_convs[:keep])
        return sam3_out, sam3_pos, sam2_out, sam2_pos
