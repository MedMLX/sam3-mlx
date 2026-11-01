# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""The ``FindStage`` batch selector and a bilinear ``interpolate``."""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass
class FindStage:
    """Which image and text prompt each batch row uses.

    ``img_ids`` and ``text_ids`` are ``(B,)`` int64 indices into the backbone's
    images and text prompts. Box and point prompts travel separately as a
    ``geometry_encoders.Prompt``.
    """

    img_ids: mx.array
    text_ids: mx.array


def interpolate(input: mx.array, size: tuple[int, int]) -> mx.array:
    """Bilinearly resize an NCHW array to ``size`` ``(H, W)``.

    Matches ``torch.nn.functional.interpolate`` with ``mode="bilinear"`` and
    ``align_corners=False``. An empty input yields zeros of the output shape.

    Raises:
        ValueError: If both the batch and channel dimensions are empty.
    """
    out_h, out_w = size
    if input.size == 0:
        if input.ndim == 4 and input.shape[0] == 0 and input.shape[1] == 0:
            raise ValueError(
                "interpolate does not support tensors with both empty batch "
                "and channel dimensions."
            )
        return mx.zeros((*input.shape[:-2], out_h, out_w), dtype=input.dtype)

    in_h, in_w = int(input.shape[-2]), int(input.shape[-1])
    x = mx.transpose(input, (0, 2, 3, 1))
    upsample = nn.Upsample(
        scale_factor=(out_h / in_h, out_w / in_w), mode="linear", align_corners=False
    )
    x = upsample(x)
    # nn.Upsample derives its output size from the scale factor and can differ by
    # a pixel from the requested size; pad or crop to the exact size.
    got_h, got_w = x.shape[-3], x.shape[-2]
    if got_h != out_h or got_w != out_w:
        pad_h = max(out_h - got_h, 0)
        pad_w = max(out_w - got_w, 0)
        if pad_h or pad_w:
            x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
        x = x[:, :out_h, :out_w, :]
    return mx.transpose(x, (0, 3, 1, 2))
