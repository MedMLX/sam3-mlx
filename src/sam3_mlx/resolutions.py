"""ViT window-aligned image resolutions."""

from __future__ import annotations

# ViT patch edge in input pixels.
PATCH_SIZE = 14
# Input size in pixels at which the token grid divides evenly into the ViT's
# 24-token attention windows.
DEFAULT_IMAGE_RESOLUTION = 1008


def window_layout_is_exact(height: int, width: int, window_size: int) -> bool:
    """Return whether a token grid tiles exactly into windows (no padding needed).

    ``height`` and ``width`` are token counts of the ViT feature grid, and
    ``window_size`` is in tokens and must be nonzero when both dimensions are
    positive; zero raises ``ZeroDivisionError``.
    """
    return (
        height > 0
        and width > 0
        and height % window_size == 0
        and width % window_size == 0
    )
