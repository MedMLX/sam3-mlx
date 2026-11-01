"""Checkpoint layout normalization for the SAM 3 image model.

A checkpoint is the flat ``{tensor name: mx.array}`` mapping that ``mx.load``
returns for a ``.safetensors`` file. Image checkpoints are already named for the
image model and only get their convolution kernels moved to MLX's channels-last
layout. The interactive head checkpoint is already in final MLX form.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias, cast

import mlx.core as mx
import mlx.utils as mlx_utils
from mlx import nn

NormalizedWeights: TypeAlias = dict[str, mx.array]

# PyTorch to MLX kernel layout: Conv2d (out, in, kH, kW) -> (out, kH, kW, in) and
# ConvTranspose2d (in, out, kH, kW) -> (out, kH, kW, in).
_CONV2D_FROM_TORCH = (0, 2, 3, 1)
_CONV_TRANSPOSE2D_FROM_TORCH = (1, 2, 3, 0)


def flatten_model_parameters(model: nn.Module) -> NormalizedWeights:
    """Return the model's parameters as a flat ``{dotted_name: mx.array}`` dict."""
    return cast(
        NormalizedWeights,
        getattr(mlx_utils, "tree_flatten")(model.parameters(), destination={}),
    )


def normalize_sam3_image_weights(
    weights: Mapping[str, mx.array], model: nn.Module
) -> NormalizedWeights:
    """Return image checkpoint ``weights`` with PyTorch-layout conv kernels in MLX layout.

    A kernel is transposed only when ``model`` has an ``nn.Conv2d`` or
    ``nn.ConvTranspose2d`` at that key and the tensor has the PyTorch layout of
    the module's MLX weight shape. Every other tensor is returned unchanged.
    """
    model_shapes = {
        key: value.shape for key, value in flatten_model_parameters(model).items()
    }
    normalized = dict(weights)
    for name, module in cast(Iterable[tuple[str, nn.Module]], model.named_modules()):
        if isinstance(module, nn.ConvTranspose2d):
            permutation = _CONV_TRANSPOSE2D_FROM_TORCH
        elif isinstance(module, nn.Conv2d):
            permutation = _CONV2D_FROM_TORCH
        else:
            continue
        key = f"{name}.weight"
        value = normalized.get(key)
        if value is None or value.ndim != 4 or value.shape == model_shapes[key]:
            continue
        if tuple(value.shape[axis] for axis in permutation) == model_shapes[key]:
            normalized[key] = mx.transpose(value, permutation)
    return normalized
