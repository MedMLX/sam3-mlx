# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""NCHW layers shared by the SAM prompt encoder and mask decoder.

Feature maps here are NCHW (PyTorch convention) so the ported SAM code keeps its
original axis arithmetic. MLX convolutions are channels-last, so ``ConvNCHW``
transposes around them; their kernels are stored in MLX layout
``(out, kH, kW, in)``.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


class LayerNorm2d(nn.Module):
    """Normalize NCHW channels at each pixel with learned channel scale and bias.

    Uses the population variance and adds ``eps`` inside the square root.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initialize channel scale, bias and normalization epsilon."""
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Normalize NCHW channels at each pixel and apply learned scale and bias."""
        mean = mx.mean(x, axis=1, keepdims=True)
        variance = mx.mean((x - mean) ** 2, axis=1, keepdims=True)
        x = (x - mean) / mx.sqrt(variance + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class ConvNCHW(nn.Module):
    """Run an MLX (transposed) convolution, stored as ``conv``, on NCHW tensors."""

    def __init__(self, conv: nn.Conv2d | nn.ConvTranspose2d) -> None:
        """Wrap a channels-last convolution."""
        super().__init__()
        self.conv = conv

    def __call__(self, x: mx.array) -> mx.array:
        """Convert NCHW to NHWC for the convolution and restore NCHW layout."""
        channels_last = mx.transpose(x, axes=(0, 2, 3, 1))
        return mx.transpose(self.conv(channels_last), axes=(0, 3, 1, 2))
