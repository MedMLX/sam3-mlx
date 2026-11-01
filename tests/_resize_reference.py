"""Execute the original Apple Silicon FP32 resize with patched TorchVision."""

from collections.abc import Callable
import importlib
from typing import Protocol, Self, cast

import numpy as np
import numpy.typing as npt
from PIL import Image


class ResizeTensor(Protocol):
    """Tensor operations used at the independent resize reference boundary."""

    def to(self, *, dtype: object) -> Self: ...

    def round(self) -> Self: ...

    def permute(self, *dims: int) -> Self: ...

    def numpy(self) -> npt.NDArray[np.generic]: ...


def resize_reference_float32(image: object, size: tuple[int, int]) -> ResizeTensor:
    """Resize CHW pixels in FP32, then round back to uint8 as TorchVision 0.24 did."""
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torchvision.transforms.v2.functional")
    resize = cast(Callable[[object, list[int]], ResizeTensor], functional.resize)
    # TorchVision 0.28 selects a different native uint8 kernel on ARM. Explicit
    # FP32 retains the checkpoint preprocessing contract and its exact fixtures.
    source = cast(ResizeTensor, image).to(dtype=torch.float32)
    return resize(source, list(size)).round().to(dtype=torch.uint8)


def official_resize(
    pixels: npt.NDArray[np.uint8], size: tuple[int, int]
) -> npt.NDArray[np.uint8]:
    """Apply the FP32 TorchVision reference to HWC RGB pixels and return uint8 HWC."""
    v2 = importlib.import_module("torchvision.transforms.v2")
    image = v2.functional.to_image(Image.fromarray(pixels))
    resized = resize_reference_float32(image, size)
    return np.asarray(resized.permute(1, 2, 0).numpy(), dtype=np.uint8)
