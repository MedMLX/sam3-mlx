"""ViT window-aligned image resolutions."""

from __future__ import annotations

# ViT patch edge in input pixels.
PATCH_SIZE = 14
# Local-attention window edge in patch tokens (not pixels).
WINDOW_SIZE = 24
# Input size in pixels at which the token grid divides evenly into windows.
ALIGNED_WINDOW_STRIDE = PATCH_SIZE * WINDOW_SIZE
CANONICAL_ALIGNED_RESOLUTIONS = (336, 672, 1008)
DEFAULT_IMAGE_RESOLUTION = 1008
# Optional low-latency tier; not part of the release parity matrix.
FAST_TIER_RESOLUTION = 336


def is_patch_aligned(resolution: int) -> bool:
    """Return whether a square input size in pixels is a positive multiple of 14."""
    return resolution > 0 and resolution % PATCH_SIZE == 0


def is_window_aligned(resolution: int) -> bool:
    """Return whether a square input size in pixels is a positive multiple of 336."""
    return resolution > 0 and resolution % ALIGNED_WINDOW_STRIDE == 0


def window_layout_is_exact(
    height: int,
    width: int,
    window_size: int = WINDOW_SIZE,
) -> bool:
    """Return whether a token grid tiles exactly into windows (no padding needed).

    ``height`` and ``width`` are token counts of the ViT feature grid, and
    ``window_size`` is in tokens.
    """
    return (
        height > 0
        and width > 0
        and height % window_size == 0
        and width % window_size == 0
    )
