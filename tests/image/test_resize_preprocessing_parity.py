"""Resize rounding agrees with the original Apple Silicon FP32 TorchVision path."""

from __future__ import annotations


from hypothesis import given, settings, strategies as st
import numpy as np
import numpy.typing as npt
from PIL import Image

from tests._resize_reference import official_resize
from tests._paths import MEDICAL_FIXTURE_ROOT
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_image_processor import transform


def transform_bytes(pixels: npt.NDArray[np.uint8], resolution: int) -> npt.NDArray[np.uint8]:
    """Recover the rounded RGB levels from the public normalized transform."""
    normalized = to_numpy(transform(Image.fromarray(pixels), resolution=resolution), dtype=np.float32)
    return np.rint((normalized.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)


def test_real_image_resize_matches_torchvision_exactly() -> None:
    """Compare resize on the repository's sourced chest X-ray."""
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    np.testing.assert_array_equal(
        transform_bytes(pixels, 1008),
        official_resize(pixels, (1008, 1008)),
    )


@given(
    height=st.integers(2, 96),
    width=st.integers(2, 96),
    resolution=st.sampled_from([14, 28, 42, 56, 70, 84, 98]),
    x=st.integers(0, 100),
    y=st.integers(0, 100),
)
@settings(max_examples=30, deadline=None)
def test_resize_preserves_uint8_contract_across_random_aspect_ratios(
    height: int, width: int, resolution: int, x: int, y: int
) -> None:
    """Compare real-image crops of varied aspect ratios against TorchVision."""
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)[y:y + height, x:x + width]
    size = (resolution, resolution)
    resized = transform_bytes(pixels, resolution)
    assert resized.shape == (*size, 3)
    assert resized.dtype == np.uint8
    np.testing.assert_array_equal(resized, official_resize(pixels, size))
