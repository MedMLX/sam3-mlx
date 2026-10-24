# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Model utility helpers ported from the official SAM3 ``sam3/model/utils``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sam3_mlx.model.utils.misc import copy_data_to_device
    from sam3_mlx.model.utils.sam1_utils import SAM2Transforms

__all__ = [
    "SAM2Transforms",
    "copy_data_to_device",
]


def __getattr__(name: str) -> object:
    """Import the exported helpers lazily (PEP 562) to keep package import cheap."""
    if name == "copy_data_to_device":
        from sam3_mlx.model.utils.misc import copy_data_to_device

        return copy_data_to_device
    if name == "SAM2Transforms":
        from sam3_mlx.model.utils.sam1_utils import SAM2Transforms

        return SAM2Transforms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
