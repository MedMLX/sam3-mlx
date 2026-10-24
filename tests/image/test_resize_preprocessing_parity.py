"""Resize rounding agrees with the original Apple Silicon FP32 TorchVision path."""

from __future__ import annotations

import importlib

from hypothesis import given, settings, strategies as st
import numpy as np
import numpy.typing as npt
from PIL import Image
import pytest

from scripts._resize_reference import resize_reference_float32
from sam3_mlx.model.sam3_image_processor import (
    _resize_uint8_bilinear_like_torchvision,  # pyright: ignore[reportPrivateUsage]
)


def official_resize(
    pixels: npt.NDArray[np.uint8], size: tuple[int, int]
) -> npt.NDArray[np.uint8]:
    v2 = importlib.import_module("torchvision.transforms.v2")
    image = v2.functional.to_image(Image.fromarray(pixels))
    resized = resize_reference_float32(image, size)
    return np.asarray(resized.permute(1, 2, 0).numpy(), dtype=np.uint8)


@pytest.mark.parametrize("shape", [(720, 1280), (1280, 720), (1200, 1600)])
def test_real_inference_resize_matches_torchvision_exactly(
    shape: tuple[int, int],
) -> None:
    pixels = np.random.default_rng(731).integers(0, 256, (*shape, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        _resize_uint8_bilinear_like_torchvision(pixels, (1008, 1008)),
        official_resize(pixels, (1008, 1008)),
    )


@given(
    height=st.integers(2, 96),
    width=st.integers(2, 96),
    out_height=st.integers(2, 96),
    out_width=st.integers(2, 96),
    seed=st.integers(0, 1000),
)
@settings(max_examples=30, deadline=None)
def test_resize_preserves_uint8_contract_across_random_aspect_ratios(
    height: int, width: int, out_height: int, out_width: int, seed: int
) -> None:
    pixels = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    size = (out_height, out_width)
    resized = _resize_uint8_bilinear_like_torchvision(pixels, size)
    assert resized.shape == (*size, 3)
    assert resized.dtype == np.uint8
    np.testing.assert_array_equal(resized, official_resize(pixels, size))
