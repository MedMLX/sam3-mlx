"""MLX port of TorchVision's ``roi_align`` (NCHW features, bilinear sampling)."""

from __future__ import annotations

from typing import cast

import mlx.core as mx

from sam3_mlx.mlx_runtime import scalar_int


Shape2D = tuple[int, int]
Shape4D = tuple[int, int, int, int]
BoxesList = list[mx.array] | tuple[mx.array, ...]
BoxesArg = mx.array | BoxesList
ADAPTIVE_ROI_CHUNK_SIZE = 4
ADAPTIVE_SAMPLE_CHUNK_SIZE = 8


def _shape2(array: mx.array, *, name: str) -> Shape2D:
    shape = array.shape
    if len(shape) != 2:
        raise ValueError(f"`{name}` must be 2D; got shape {shape}.")
    return shape


def _shape4(array: mx.array, *, name: str) -> Shape4D:
    shape = array.shape
    if len(shape) != 4:
        raise ValueError(f"`{name}` must be 4D; got shape {shape}.")
    return shape


def _dtype(array: mx.array) -> mx.Dtype:
    return array.dtype


def _astype(array: mx.array, dtype: mx.Dtype) -> mx.array:
    return array.astype(dtype)


def _validate_floating_array(array: mx.array, *, name: str) -> None:
    if not mx.issubdtype(_dtype(array), mx.floating):
        raise TypeError(f"`{name}` must have a floating-point dtype.")


def _sum_last_two_axes(array: mx.array) -> mx.array:
    return array.sum(axis=(-1, -2))


# All inputs are MLX arrays.
def _bilinear_interpolate(
    input: mx.array,  # [N, C, H, W]
    roi_batch_ind: mx.array,  # [K]
    y: mx.array,  # [K, PH, IY]
    x: mx.array,  # [K, PW, IX]
    ymask: mx.array | None,  # [K, IY]
    xmask: mx.array | None,  # [K, IX]
) -> mx.array:
    _, channels, height, width = _shape4(input, name="input")
    coordinate_dtype = _dtype(y)
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
    y = mx.where(y_low >= height - 1, _astype(y_low, coordinate_dtype), y)

    x_high = mx.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = mx.where(x_low >= width - 1, width - 1, x_low)
    x = mx.where(x_low >= width - 1, _astype(x_low, coordinate_dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    hy = hy * y_valid
    ly = ly * y_valid
    hx = hx * x_valid
    lx = lx * x_valid

    # Respect adaptive-sampling masks before indexing padded interpolation bins.
    def masked_index(
        y_indices: mx.array,  # [K, PH, IY]
        x_indices: mx.array,  # [K, PW, IX]
    ) -> mx.array:
        if ymask is not None:
            if xmask is None:
                raise ValueError("xmask is required when ymask is provided.")
            y_indices = mx.where(ymask[:, None, :], y_indices, 0)
            x_indices = mx.where(xmask[:, None, :], x_indices, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            mx.arange(channels)[None, :, None, None, None, None],
            y_indices[:, None, :, None, :, None],  # prev [K, PH, IY]
            x_indices[:, None, None, :, None, :],  # prev [K, PW, IX]
        ]  # [K, C, PH, PW, IY, IX]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    # all ws preemptively [K, C, PH, PW, IY, IX]
    def outer_prod(y_weights: mx.array, x_weights: mx.array) -> mx.array:
        return (
            y_weights[:, None, :, None, :, None] * x_weights[:, None, None, :, None, :]
        )

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


def _adaptive_roi_align(
    input: mx.array,
    roi_batch_ind: mx.array,
    roi_start_h: mx.array,
    roi_start_w: mx.array,
    roi_height: mx.array,
    roi_width: mx.array,
    bin_size_h: mx.array,
    bin_size_w: mx.array,
    pooled_height: int,
    pooled_width: int,
) -> mx.array:
    """Evaluate official adaptive sampling with bounded ROI/sample chunks."""

    roi_count = int(roi_batch_ind.shape[0])
    _, channels, _, _ = _shape4(input, name="input")
    input_dtype = _dtype(input)
    coordinate_dtype = _dtype(roi_start_h)
    if roi_count == 0:
        return mx.zeros(
            (0, channels, pooled_height, pooled_width),
            dtype=input_dtype,
        )

    ph = mx.arange(pooled_height)
    pw = mx.arange(pooled_width)
    # Synchronize ROI maxima once to bound adaptive samples instead of using the
    # full feature-map size; smaller per-chunk grids contribute masked zeros.
    max_grid_h = max(scalar_int(mx.ceil(mx.max(roi_height) / pooled_height)), 1)
    max_grid_w = max(scalar_int(mx.ceil(mx.max(roi_width) / pooled_width)), 1)
    outputs: list[mx.array] = []
    for roi_offset in range(0, roi_count, ADAPTIVE_ROI_CHUNK_SIZE):
        roi_stop = min(roi_offset + ADAPTIVE_ROI_CHUNK_SIZE, roi_count)
        chunk = slice(roi_offset, roi_stop)
        chunk_size = roi_stop - roi_offset
        chunk_grid_h = mx.ceil(roi_height[chunk] / pooled_height)
        chunk_grid_w = mx.ceil(roi_width[chunk] / pooled_width)
        count = mx.clip(chunk_grid_h * chunk_grid_w, a_min=1, a_max=None)
        safe_grid_h = mx.clip(chunk_grid_h, a_min=1, a_max=None)
        safe_grid_w = mx.clip(chunk_grid_w, a_min=1, a_max=None)

        chunk_bin_h = bin_size_h[chunk]
        chunk_bin_w = bin_size_w[chunk]
        accumulated = mx.zeros(
            (chunk_size, channels, pooled_height, pooled_width),
            dtype=input_dtype,
        )

        x_chunks: list[tuple[mx.array, mx.array]] = []
        for sample_x in range(0, max_grid_w, ADAPTIVE_SAMPLE_CHUNK_SIZE):
            ix = mx.arange(
                sample_x,
                min(sample_x + ADAPTIVE_SAMPLE_CHUNK_SIZE, max_grid_w),
            )
            xmask = ix[None, :] < chunk_grid_w[:, None]
            x = (
                roi_start_w[chunk, None, None]
                + pw[None, :, None] * chunk_bin_w[:, None, None]
                + _astype(ix[None, None, :] + 0.5, coordinate_dtype)
                * chunk_bin_w[:, None, None]
                / safe_grid_w[:, None, None]
            )
            x_chunks.append((x, xmask))

        for sample_y in range(0, max_grid_h, ADAPTIVE_SAMPLE_CHUNK_SIZE):
            iy = mx.arange(
                sample_y,
                min(sample_y + ADAPTIVE_SAMPLE_CHUNK_SIZE, max_grid_h),
            )
            ymask = iy[None, :] < chunk_grid_h[:, None]
            y = (
                roi_start_h[chunk, None, None]
                + ph[None, :, None] * chunk_bin_h[:, None, None]
                + _astype(iy[None, None, :] + 0.5, coordinate_dtype)
                * chunk_bin_h[:, None, None]
                / safe_grid_h[:, None, None]
            )
            for x, xmask in x_chunks:
                values = _bilinear_interpolate(
                    input,
                    roi_batch_ind[chunk],
                    y,
                    x,
                    ymask,
                    xmask,
                )
                values = mx.where(ymask[:, None, None, None, :, None], values, 0)
                values = mx.where(xmask[:, None, None, None, None, :], values, 0)
                accumulated = accumulated + _sum_last_two_axes(values)

        outputs.append(accumulated / count[:, None, None, None])

    return mx.concat(outputs, axis=0).astype(input_dtype)


def convert_boxes_to_roi_format(boxes: BoxesList) -> mx.array:
    """Stack per-image ``(L_i, 4)`` boxes into ``(sum L_i, 5)`` ROI rows.

    Each row is ``(image_index, x0, y0, x1, y1)``, where ``image_index`` is the
    position of the box's array in ``boxes``.

    Raises:
        ValueError: If ``boxes`` is empty.
    """
    if not boxes:
        raise ValueError(
            "`boxes` must contain at least one per-image array; "
            "an empty outer sequence is unsupported."
        )
    concat_boxes = mx.concat(list(boxes), axis=0)
    batch_ids: list[mx.array] = []
    for i, b in enumerate(boxes):
        batch_ids.append(mx.full(b[:, :1].shape, i))
    ids = mx.concat(batch_ids, axis=0)
    rois = mx.concat([ids, concat_boxes], axis=1)
    return rois


def check_roi_boxes_shape(boxes: BoxesArg) -> None:
    """Validate the ``boxes`` argument of ``roi_align``.

    Accepts either a ``(K, 5)`` floating array of ``(image_index, x0, y0, x1,
    y1)`` rows or a list/tuple of floating ``(L, 4)`` arrays, one per image.

    Raises:
        TypeError: On a wrong container, element type or non-floating dtype.
        ValueError: On a wrong trailing dimension.
    """
    if isinstance(boxes, (list, tuple)):
        for _array in boxes:
            if not isinstance(cast(object, _array), mx.array):
                raise TypeError("Each boxes-list element must be an MLX array.")
            if _shape2(_array, name="boxes element")[1] != 4:
                raise ValueError("Each boxes-list element must have shape (L, 4).")
            _validate_floating_array(_array, name="boxes element")
    else:
        if not isinstance(cast(object, boxes), mx.array):
            raise TypeError(
                "`boxes` must be an MLX array or a list/tuple of MLX arrays."
            )
        if _shape2(boxes, name="boxes")[1] != 5:
            raise ValueError("The boxes array must have shape (K, 5).")
        _validate_floating_array(boxes, name="boxes")


def _validate_output_dimension(value: int, *, name: str) -> None:
    runtime_value = cast(object, value)
    if isinstance(runtime_value, bool) or not isinstance(runtime_value, int):
        raise TypeError(f"`{name}` must be an integer; got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"`{name}` must be positive; got {value}.")


def _validate_batch_indices(rois: mx.array, *, batch_size: int) -> None:
    if rois.shape[0] == 0:
        return
    batch_indices = rois[:, 0]
    valid = mx.all(
        mx.isfinite(batch_indices)
        & (batch_indices == mx.floor(batch_indices))
        & (batch_indices >= 0)
        & (batch_indices < batch_size)
    )
    if not bool(valid.item()):
        raise ValueError(
            f"ROI batch indices must be finite integers in [0, {batch_size - 1}]."
        )


def _roi_align(
    input: mx.array,
    rois: mx.array,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int,
    aligned: bool,
) -> mx.array:
    orig_dtype = _dtype(input)

    _shape4(input, name="input")
    if _shape2(rois, name="rois")[1] != 5:
        raise ValueError("`rois` must have shape (K, 5).")
    if _dtype(rois) == mx.bfloat16:
        rois = _astype(rois, mx.float32)

    # inputs: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = _astype(rois[:, 0], mx.int32)
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

    roi_width = roi_end_w - roi_start_w  # [K]
    roi_height = roi_end_h - roi_start_h  # [K]
    if not aligned:
        roi_width = mx.clip(roi_width, a_min=1.0, a_max=None)
        roi_height = mx.clip(roi_height, a_min=1.0, a_max=None)

    bin_size_h = roi_height / pooled_height  # [K]
    bin_size_w = roi_width / pooled_width  # [K]

    exact_sampling = sampling_ratio > 0

    if exact_sampling:
        count = max(sampling_ratio * sampling_ratio, 1)
        ph = mx.arange(pooled_height)
        pw = mx.arange(pooled_width)
        iy = mx.arange(sampling_ratio)
        ix = mx.arange(sampling_ratio)
    else:
        return _adaptive_roi_align(
            input,
            roi_batch_ind,
            roi_start_h,
            roi_start_w,
            roi_height,
            roi_width,
            bin_size_h,
            bin_size_w,
            pooled_height,
            pooled_width,
        )

    def from_K(t: mx.array) -> mx.array:
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + ph[None, :, None] * from_K(bin_size_h)
        + _astype(iy[None, None, :] + 0.5, _dtype(rois))
        * from_K(bin_size_h)
        / sampling_ratio
    )  # [K, PH, IY]
    x = (
        from_K(roi_start_w)
        + pw[None, :, None] * from_K(bin_size_w)
        + _astype(ix[None, None, :] + 0.5, _dtype(rois))
        * from_K(bin_size_w)
        / sampling_ratio
    )  # [K, PW, IX]
    val = _bilinear_interpolate(
        input, roi_batch_ind, y, x, None, None
    )  # [K, C, PH, PW, IY, IX]

    output = _sum_last_two_axes(val)  # remove IY, IX ~> [K, C, PH, PW]
    output /= count

    output = output.astype(orig_dtype)

    return output


def roi_align(
    input: mx.array,
    boxes: BoxesArg,
    height: int,
    width: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> mx.array:
    """Pool each region of interest to a fixed ``height`` x ``width`` grid.

    Mirrors ``torchvision.ops.roi_align`` (average of bilinear samples per bin),
    except that the output size is passed as separate ``height`` and ``width``.

    Args:
        input: Features of shape ``(N, C, H, W)`` (channels-first), floating
            dtype.
        boxes: Either ``(K, 5)`` rows of ``(batch_index, x0, y0, x1, y1)`` or a
            list/tuple of ``(L_i, 4)`` xyxy arrays, one per batch element.
            Coordinates are in input-image pixels before ``spatial_scale``.
        height: Output bin rows (positive).
        width: Output bin columns (positive).
        spatial_scale: Multiplier mapping box coordinates to feature-map
            coordinates.
        sampling_ratio: Samples per bin side; ``-1`` (or any value <= 0) picks
            an adaptive count from each ROI's size.
        aligned: If True, shift box coordinates by -0.5 pixel and do not force
            ROI width/height to be at least 1; False keeps the legacy behavior.

    Returns:
        ``(K, C, height, width)`` array with the dtype of ``input``.

    This call evaluates the graph to validate batch indices and, with adaptive
    sampling, to read the ROI extents, so it synchronizes with the device.

    Raises:
        TypeError: If ``input`` or ``boxes`` is not floating, or ``height`` /
            ``width`` is not an int.
        ValueError: On wrong ranks or shapes, non-positive output size, or
            batch indices that are not integers in ``[0, N)``.
    """
    batch_size, _, _, _ = _shape4(input, name="input")
    _validate_floating_array(input, name="input")
    _validate_output_dimension(height, name="height")
    _validate_output_dimension(width, name="width")
    check_roi_boxes_shape(boxes)
    rois = (
        convert_boxes_to_roi_format(boxes)
        if isinstance(boxes, (list, tuple))
        else boxes
    )
    _validate_batch_indices(rois, batch_size=batch_size)
    return _roi_align(
        input, rois, spatial_scale, height, width, sampling_ratio, aligned
    )
