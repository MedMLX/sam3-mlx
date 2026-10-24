"""Experimental APIs that are not part of the stable 0.1.x contract.

Tracker construction and SAM 3.1 multiplex / temporal tracking builders live
here until their parity contracts are demonstrated. Prefer the selected-frame
SAM 3 surface for supported 0.1.x workflows.
"""

from __future__ import annotations

from sam3_mlx.model_builder import (
    build_tracker,
    build_sam3_multiplex_video_model,
    build_sam3_multiplex_video_predictor,
)

__all__ = [
    "build_tracker",
    "build_sam3_multiplex_video_model",
    "build_sam3_multiplex_video_predictor",
]
