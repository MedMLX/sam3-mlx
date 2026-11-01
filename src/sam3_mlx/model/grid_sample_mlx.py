"""Metal-kernel bilinear ``grid_sample`` for channels-last MLX arrays."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, cast

import mlx.core as mx


LaunchShape = tuple[int, int, int]
KernelTemplate = tuple[str, mx.Dtype]


class _MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[mx.array],
        template: Sequence[KernelTemplate],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[mx.Dtype],
        grid: LaunchShape,
        threadgroup: LaunchShape,
    ) -> Sequence[mx.array]:
        """Launch the compiled Metal kernel with specified inputs and output shapes."""
        ...


_SOURCE = """
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;
        int gH = grid_shape[1];
        int gW = grid_shape[2];
        uint grid_idx = elem / C * 2;
        float gx = float(grid[grid_idx]);
        float gy = float(grid[grid_idx + 1]);
        // Beyond [-3, 3], no pixel can contribute even for a size-one axis.
        // Return before coordinate multiplication or float-to-int conversion can
        // overflow for large finite values. Keep non-finite behavior unchanged.
        if (isfinite(gx) && isfinite(gy) && (fabs(gx) > 3.0f || fabs(gy) > 3.0f)) {
            out[elem] = T(0);
            return;
        }
        float ix = ((gx + 1) * W - 1) / 2;
        float iy = ((gy + 1) * H - 1) / 2;
        int ix_nw = floor(ix);
        int iy_nw = floor(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        float nw = (ix_se - ix)    * (iy_se - iy);
        float ne = (ix    - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix)    * (iy    - iy_ne);
        float se = (ix    - ix_nw) * (iy    - iy_nw);
        int batch_idx = elem / C / gH / gW * b_stride;
        int channel_idx = elem % C;
        int base_idx = batch_idx + channel_idx;
        bool nw_in_bounds = iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W;
        bool ne_in_bounds = iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W;
        bool sw_in_bounds = iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W;
        bool se_in_bounds = iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W;
        T I_nw = T(0);
        T I_ne = T(0);
        T I_sw = T(0);
        T I_se = T(0);
        if (nw_in_bounds) {
            I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        }
        if (ne_in_bounds) {
            I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        }
        if (sw_in_bounds) {
            I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        }
        if (se_in_bounds) {
            I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];
        }
        out[elem] = T(
            nw * float(I_nw) + ne * float(I_ne)
            + sw * float(I_sw) + se * float(I_se)
        );
"""


@lru_cache(maxsize=1)
def _kernel() -> _MetalKernel:
    """Construct or retrieve the cached grid-sampling Metal kernel."""
    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name="grid_sample",
            input_names=("x", "grid"),
            output_names=("out",),
            source=_SOURCE,
        ),
    )


def grid_sample(x: mx.array, grid: mx.array) -> mx.array:
    """Bilinearly sample ``x`` at normalized ``grid`` locations.

    For finite floating coordinates, uses the bilinear, zeros-padding,
    ``align_corners=False`` convention of ``torch.nn.functional.grid_sample``,
    but in channels-last layout. Non-finite grid values are not validated and
    are outside that comparison; NaN and infinity produce NaN in the tested
    Metal runtime.

    Args:
        x: Input of shape ``(B, H, W, C)`` (NHWC, not torch's NCHW).
        grid: Sampling locations of shape ``(B, gH, gW, 2)``. The last axis is
            normalized ``(x, y)``: -1 and 1 are the outer edges of the first and
            last pixels, and coordinates outside that range are allowed. Each
            out-of-bounds neighbor contributes zero, so a sample near an edge
            can mix valid pixels with zero padding.

    Returns:
        Array of shape ``(B, gH, gW, C)`` with the dtype of ``x``.

    Raises:
        ValueError: If ``x`` or ``grid`` is not 4-D, the batch sizes differ, or
            the last axis of ``grid`` is not 2.
    """
    if x.ndim != 4 or grid.ndim != 4:
        raise ValueError(f"`x` and `grid` must be 4D, got {x.shape} and {grid.shape}.")
    batch, _, _, channels = x.shape
    grid_batch, grid_height, grid_width, coords = grid.shape
    if grid_batch != batch:
        raise ValueError(
            f"`grid` batch dimension must match `x`: got {grid_batch} and {batch}."
        )
    if coords != 2:
        raise ValueError(f"Last dimension of `grid` must be size 2, got {coords}.")
    outputs = _kernel()(
        inputs=[x, grid],
        template=[("T", x.dtype)],
        output_shapes=[(batch, grid_height, grid_width, channels)],
        output_dtypes=[x.dtype],
        grid=(batch * grid_height * grid_width * channels, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return outputs[0]
