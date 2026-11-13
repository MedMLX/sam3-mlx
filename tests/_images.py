"""Sourced chest radiograph frames for image, video, and session tests."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tests._paths import MEDICAL_FIXTURE_ROOT


def chest_frame(
    size: tuple[int, int], offset: int = 0, *, marker: int | None = None,
) -> Image.Image:
    """Crop and resize the radiograph, optionally marking one pixel for frame stubs."""
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as source:
        frame = source.convert("RGB").crop((offset, offset, offset + 200, offset + 200)).resize(size)
    if marker is not None:
        frame.putpixel((0, 0), (marker, 0, 0))
    return frame


def chest_input(size: tuple[int, int]) -> NDArray[np.float32]:
    """Return a normalized RGB radiograph in the visual trunk's NCHW format."""
    pixels = np.asarray(chest_frame(size), dtype=np.float32)
    return (pixels.transpose(2, 0, 1)[None] / np.float32(127.5) - np.float32(1)).copy()
