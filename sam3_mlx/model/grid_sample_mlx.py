"""Metal-kernel bilinear ``grid_sample`` for channels-last MLX arrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import mlx.core as mx


Shape4D = tuple[int, int, int, int]
LaunchShape = tuple[int, int, int]
KernelTemplate = tuple[str, mx.Dtype]


class _ArrayMetadata(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> mx.Dtype: ...


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
        init_value: int | float | None = None,
    ) -> Sequence[mx.array]: ...


class _MetalKernelFactory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
    ) -> _MetalKernel: ...


_metal_kernel = cast(_MetalKernelFactory, mx.fast.metal_kernel)
_METAL_KERNEL_CACHE: dict[str, _MetalKernel] = {}


def _array_metadata(array: mx.array) -> _ArrayMetadata:
    return cast(_ArrayMetadata, array)


def _shape4(array: mx.array, *, name: str) -> Shape4D:
    metadata = _array_metadata(array)
    if metadata.ndim != 4:
        raise ValueError(f"`{name}` must be 4D, got shape {metadata.shape}.")
    shape = metadata.shape
    if len(shape) != 4:
        raise ValueError(f"`{name}` must be 4D, got shape {shape}.")
    return shape


def _dtype(array: mx.array) -> mx.Dtype:
    return _array_metadata(array).dtype


def _validate_grid_sample_inputs(
    x: mx.array,
    grid: mx.array,
) -> tuple[Shape4D, Shape4D]:
    x_shape = _shape4(x, name="x")
    grid_shape = _shape4(grid, name="grid")
    if grid_shape[0] != x_shape[0]:
        raise ValueError(
            "`grid` batch dimension must match `x`: "
            f"got {grid_shape[0]} and {x_shape[0]}."
        )
    if grid_shape[-1] != 2:
        raise ValueError(
            f"Last dimension of `grid` must be size 2, got {grid_shape[-1]}."
        )
    return x_shape, grid_shape


def _cached_metal_kernel(
    cache_key: str,
    *,
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    atomic_outputs: bool = False,
) -> _MetalKernel:
    kernel = _METAL_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        kernel = _metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
            atomic_outputs=atomic_outputs,
        )
        _METAL_KERNEL_CACHE[cache_key] = kernel
    return kernel


def grid_sample(x: mx.array, grid: mx.array) -> mx.array:
    """Bilinearly sample ``x`` at normalized ``grid`` locations.

    Follows the semantics of ``torch.nn.functional.grid_sample`` with its default
    arguments (bilinear, zeros padding, ``align_corners=False``), but in
    channels-last layout.

    Args:
        x: Input of shape ``(B, H, W, C)`` (NHWC, not torch's NCHW).
        grid: Sampling locations of shape ``(B, gH, gW, 2)``. The last axis is
            ``(x, y)`` in ``[-1, 1]``, where -1 and 1 are the outer edges of the
            first and last pixels. Samples outside the image read zero.

    Returns:
        Array of shape ``(B, gH, gW, C)`` with the dtype of ``x``.

    Raises:
        ValueError: If ``x`` or ``grid`` is not 4-D, the batch sizes differ, or
            the last axis of ``grid`` is not 2.
    """

    x_shape, grid_shape = _validate_grid_sample_inputs(x, grid)
    batch, _, _, channels = x_shape
    _, grid_height, grid_width, _ = grid_shape
    out_shape: Shape4D = (batch, grid_height, grid_width, channels)
    out_size = batch * grid_height * grid_width * channels

    source = """
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
        float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;
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
    kernel = _cached_metal_kernel(
        "grid_sample",
        name="grid_sample",
        input_names=("x", "grid"),
        output_names=("out",),
        source=source,
    )
    x_dtype = _dtype(x)
    outputs = kernel(
        inputs=[x, grid],
        template=[("T", x_dtype)],
        output_shapes=[out_shape],
        output_dtypes=[x_dtype],
        grid=(out_size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return outputs[0]
