"""MLX port of TorchVision's ``roi_align`` with adaptive sampling (NCHW features, bilinear sampling)."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from sam3_mlx.mlx_runtime import evaluate_boundary, scalar_int


ADAPTIVE_ROI_CHUNK_SIZE = 4
ADAPTIVE_SAMPLE_CHUNK_SIZE = 8


def _bilinear_interpolate(
    input: mx.array,  # [N, C, H, W]
    roi_batch_ind: mx.array,  # [K]
    y: mx.array,  # [K, PH, IY]
    x: mx.array,  # [K, PW, IX]
) -> mx.array:
    """Sample NCHW features at ROI coordinates with source-compatible boundary handling.

    Returns ``[K, C, PH, PW, IY, IX]`` samples.
    """
    _, channels, height, width = input.shape
    # TorchVision zeros samples more than one pixel outside the feature map,
    # while coordinates on the inclusive [-1, size] boundary clamp to an edge.
    y_valid = (y >= -1.0) & (y <= height)
    x_valid = (x >= -1.0) & (x <= width)
    y = mx.clip(y, a_min=0, a_max=None)
    x = mx.clip(x, a_min=0, a_max=None)

    y_low = y.astype(mx.int32)
    x_low = x.astype(mx.int32)
    y_high = mx.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = mx.where(y_low >= height - 1, height - 1, y_low)
    y = mx.where(y_low >= height - 1, y_low.astype(y.dtype), y)

    x_high = mx.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = mx.where(x_low >= width - 1, width - 1, x_low)
    x = mx.where(x_low >= width - 1, x_low.astype(x.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    hy = hy * y_valid
    ly = ly * y_valid
    hx = hx * x_valid
    lx = lx * x_valid

    # Clamped indices are always in range, so padded adaptive samples gather
    # real pixels here and are zeroed by the caller.
    def gather(y_indices: mx.array, x_indices: mx.array) -> mx.array:
        """Gather feature values at ``[K, PH, IY]`` rows and ``[K, PW, IX]`` columns."""
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            mx.arange(channels)[None, :, None, None, None, None],
            y_indices[:, None, :, None, :, None],
            x_indices[:, None, None, :, None, :],
        ]  # [K, C, PH, PW, IY, IX]

    v1 = gather(y_low, x_low)
    v2 = gather(y_low, x_high)
    v3 = gather(y_high, x_low)
    v4 = gather(y_high, x_high)

    def outer_prod(y_weights: mx.array, x_weights: mx.array) -> mx.array:
        """Combine vertical and horizontal interpolation weights."""
        return (
            y_weights[:, None, :, None, :, None] * x_weights[:, None, None, :, None, :]
        )

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


def roi_align(
    input: mx.array,
    boxes: Sequence[mx.array],
    height: int,
    width: int,
) -> mx.array:
    """Pool each region of interest to a fixed ``height`` x ``width`` grid.

    Mirrors ``torchvision.ops.roi_align`` with ``spatial_scale=1``,
    ``sampling_ratio=-1`` (adaptive sample count from each ROI's size) and
    ``aligned=False`` (average of bilinear samples per bin; ROI sides are
    clamped to at least 1), except that the output size is passed as separate
    ``height`` and ``width``.

    Args:
        input: Features of shape ``(N, C, H, W)`` (channels-first), floating
            dtype.
        boxes: Nonempty sequence of at most ``N`` floating ``(L_i, 4)`` xyxy
            arrays in feature-map pixels; entry ``i`` holds the boxes of batch
            element ``i``.
        height: Output bin rows.
        width: Output bin columns.

    Returns:
        ``(sum L_i, C, height, width)`` array with the dtype of ``input``.

    This call evaluates the graph once, to check that box coordinates are
    finite and to read the largest ROI extents, so it synchronizes with the
    device. Box coordinate ordering is not checked.

    Raises:
        TypeError: If ``input`` or a box array is not floating.
        ValueError: On a wrong rank or box shape, a non-positive output size,
            an empty or longer-than-batch ``boxes``, or a non-finite box
            coordinate.
    """
    batch_size, channels, _, _ = input.shape
    input_dtype = input.dtype
    if not mx.issubdtype(input_dtype, mx.floating) or not all(
        mx.issubdtype(image_boxes.dtype, mx.floating) for image_boxes in boxes
    ):
        raise TypeError("`input` and `boxes` must have floating-point dtypes.")
    if height < 1 or width < 1:
        raise ValueError(f"Output size must be positive; got {(height, width)}.")
    if not boxes or len(boxes) > batch_size:
        raise ValueError(f"`boxes` must hold 1 to {batch_size} per-image arrays.")
    if any(image_boxes.ndim != 2 or image_boxes.shape[1] != 4 for image_boxes in boxes):
        raise ValueError("Each `boxes` entry must have shape (L, 4).")
    rois = mx.concat(list(boxes), axis=0)  # [K, 4]
    roi_count = rois.shape[0]
    if roi_count == 0:
        return mx.zeros((0, channels, height, width), dtype=input_dtype)
    if rois.dtype == mx.bfloat16:
        rois = rois.astype(mx.float32)
    roi_batch_ind = mx.array(
        [index for index, image_boxes in enumerate(boxes) for _ in range(image_boxes.shape[0])],
        dtype=mx.int32,
    )

    roi_start_w = rois[:, 0]
    roi_start_h = rois[:, 1]
    roi_width = mx.clip(rois[:, 2] - roi_start_w, a_min=1.0, a_max=None)
    roi_height = mx.clip(rois[:, 3] - roi_start_h, a_min=1.0, a_max=None)
    bin_size_h = roi_height / height
    bin_size_w = roi_width / width

    # One device sync validates the boxes and bounds the adaptive sample grid by
    # the largest ROI instead of the feature-map size; smaller per-chunk grids
    # contribute masked zeros.
    finite = mx.all(mx.isfinite(rois))
    max_grid_h = mx.ceil(mx.max(roi_height) / height)
    max_grid_w = mx.ceil(mx.max(roi_width) / width)
    evaluate_boundary(finite, max_grid_h, max_grid_w)
    if not finite.item():
        raise ValueError("ROI box coordinates must be finite.")
    grid_h_limit = max(scalar_int(max_grid_h), 1)
    grid_w_limit = max(scalar_int(max_grid_w), 1)

    ph = mx.arange(height)
    pw = mx.arange(width)
    coordinate_dtype = rois.dtype
    outputs: list[mx.array] = []
    for roi_offset in range(0, roi_count, ADAPTIVE_ROI_CHUNK_SIZE):
        roi_stop = min(roi_offset + ADAPTIVE_ROI_CHUNK_SIZE, roi_count)
        chunk = slice(roi_offset, roi_stop)
        chunk_grid_h = mx.ceil(roi_height[chunk] / height)
        chunk_grid_w = mx.ceil(roi_width[chunk] / width)
        count = mx.clip(chunk_grid_h * chunk_grid_w, a_min=1, a_max=None)
        safe_grid_h = mx.clip(chunk_grid_h, a_min=1, a_max=None)
        safe_grid_w = mx.clip(chunk_grid_w, a_min=1, a_max=None)

        chunk_bin_h = bin_size_h[chunk]
        chunk_bin_w = bin_size_w[chunk]
        accumulated = mx.zeros(
            (roi_stop - roi_offset, channels, height, width), dtype=input_dtype
        )

        x_chunks: list[tuple[mx.array, mx.array]] = []
        for sample_x in range(0, grid_w_limit, ADAPTIVE_SAMPLE_CHUNK_SIZE):
            ix = mx.arange(sample_x, min(sample_x + ADAPTIVE_SAMPLE_CHUNK_SIZE, grid_w_limit))
            xmask = ix[None, :] < chunk_grid_w[:, None]
            x = (
                roi_start_w[chunk, None, None]
                + pw[None, :, None] * chunk_bin_w[:, None, None]
                + (ix[None, None, :] + 0.5).astype(coordinate_dtype)
                * chunk_bin_w[:, None, None]
                / safe_grid_w[:, None, None]
            )
            x_chunks.append((x, xmask))

        for sample_y in range(0, grid_h_limit, ADAPTIVE_SAMPLE_CHUNK_SIZE):
            iy = mx.arange(sample_y, min(sample_y + ADAPTIVE_SAMPLE_CHUNK_SIZE, grid_h_limit))
            ymask = iy[None, :] < chunk_grid_h[:, None]
            y = (
                roi_start_h[chunk, None, None]
                + ph[None, :, None] * chunk_bin_h[:, None, None]
                + (iy[None, None, :] + 0.5).astype(coordinate_dtype)
                * chunk_bin_h[:, None, None]
                / safe_grid_h[:, None, None]
            )
            for x, xmask in x_chunks:
                values = _bilinear_interpolate(input, roi_batch_ind[chunk], y, x)
                values = mx.where(ymask[:, None, None, None, :, None], values, 0)
                values = mx.where(xmask[:, None, None, None, None, :], values, 0)
                accumulated = accumulated + values.sum(axis=(-1, -2))

        outputs.append(accumulated / count[:, None, None, None])

    return mx.concat(outputs, axis=0).astype(input_dtype)
