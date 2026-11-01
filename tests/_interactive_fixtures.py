"""Synthetic interactive-neck features, so prompt, box and mask paths run without a backbone."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from sam3_mlx.model.sam1_task_predictor import (
    FEATURE_SIZES,
    SAM3InteractiveImagePredictor,
)

# Channels of the projected pyramid that Sam3Processor.set_image stores.
_CHANNELS = (32, 64, 256)


def interactive_predictor(seed: int = 731) -> SAM3InteractiveImagePredictor:
    """Build a seeded, randomly initialized interactive head."""
    mx.random.seed(seed)
    return SAM3InteractiveImagePredictor()


def interactive_features(
    seed: int = 731,
    batch_size: int = 1,
    sizes: tuple[tuple[int, ...], ...] = FEATURE_SIZES,
) -> list[mx.array]:
    """Return random NCHW levels shaped like the projected interactive pyramid."""
    rng = np.random.default_rng(seed)
    return [
        mx.array(rng.normal(size=(batch_size, channels, *size)).astype(np.float32))
        for channels, size in zip(_CHANNELS, sizes, strict=True)
    ]


def interactive_state(
    features: list[mx.array], original_size: tuple[int, int]
) -> dict[str, object]:
    """Wrap features in the parts of a ``Sam3Processor.set_image`` state that ``predict_inst`` reads."""
    height, width = original_size
    return {
        "original_height": height,
        "original_width": width,
        "backbone_out": {"sam2_backbone_out": {"backbone_fpn": features}},
    }
