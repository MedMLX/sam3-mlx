# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Image-prompt processor: preprocessing, prompt state and result post-processing.

``Sam3Processor`` resizes images to a square ``resolution`` x ``resolution`` grid
(without keeping the aspect ratio), converts them to RGB float32 CHW and
normalizes with mean 0.5 and std 0.5 (values in [-1, 1]). Prompt boxes are
normalized (cx, cy, w, h) and points normalized (x, y), both in [0, 1] relative to
the original image. Predicted boxes are returned as (x0, y0, x1, y1) in original
image pixels.
"""

from collections.abc import Callable, Mapping, Sequence
import os
from functools import lru_cache, partial
from numbers import Integral
from typing import Literal, NoReturn, Protocol, TypeAlias, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from PIL import Image

from sam3_mlx.mlx_runtime import scalar_float
from sam3_mlx.mlx_runtime import scalar_int
from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model import box_ops
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.data_misc import (
    FindStage,
    IndexArray,
    ResizeWeights,
    WeightArray,
    interpolate,
)
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.sam3_image import (
    Output,
    RawPrediction,
    raw_prediction_from_output,
)
from sam3_mlx.precision import cast_visual_input, model_precision, parse_precision
from sam3_mlx.resolutions import DEFAULT_IMAGE_RESOLUTION, PATCH_SIZE

PREPROCESSED_LAYOUTS = ("nchw", "nhwc")
PREPROCESSED_VALUE_CONTRACT = "normalized-minus1-plus1"
DEFAULT_TEXT_CACHE_SIZE = 8
ImageSource: TypeAlias = str | os.PathLike[str] | Image.Image | npt.NDArray[np.generic]
NativeImageSource: TypeAlias = ImageSource | mx.array
PreprocessingMode: TypeAlias = Literal["parity", "fast", "native"]
PREPROCESSING_MODES: tuple[PreprocessingMode, ...] = ("parity", "fast", "native")


def _raise_processor_unsupported(
    feature: str,
    *,
    reason: str,
    detail: str,
    alternative: str | None = None,
) -> NoReturn:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


def _score_keep_indices(scores: mx.array, threshold: float) -> mx.array:
    keep = scores > threshold
    # MLX has no boolean indexing/nonzero yet. Sync only the scalar count, then
    # build the ordered variable-length index vector on device.
    keep_count = scalar_int(mx.sum(keep))
    if keep_count == 0:
        return mx.array([], dtype=mx.int64)
    positions = mx.arange(keep.shape[0], dtype=mx.int64)
    sentinel = mx.array(keep.shape[0], dtype=mx.int64)
    return mx.sort(mx.where(keep, positions, sentinel))[:keep_count]


def single_image_keep_indices(out_probs: mx.array, threshold: float) -> mx.array:
    """Return ascending indices of the first batch row of ``out_probs`` ``(1, Q)`` above ``threshold``.

    The count of kept scores is read back to the host (a device synchronization).
    """
    return _score_keep_indices(out_probs[0], threshold)


def _presence_weighted_scores(outputs: Mapping[str, object]) -> mx.array:
    out_logits = cast(mx.array, outputs["pred_logits"])
    presence_logit = cast(mx.array, outputs["presence_logit_dec"])
    if out_logits.dtype != mx.bfloat16:
        out_logits = out_logits.astype(mx.float32)
        presence_logit = presence_logit.astype(mx.float32)
    out_probs = mx.sigmoid(out_logits.astype(mx.float32)).astype(out_logits.dtype)
    presence_score = mx.sigmoid(presence_logit.astype(mx.float32)).astype(presence_logit.dtype)
    return (out_probs * presence_score[:, None]).squeeze(-1).astype(mx.float32)


def _filter_single_image_detections(
    out_probs: mx.array,
    out_masks: mx.array,
    out_bbox: mx.array,
    *,
    threshold: float,
) -> tuple[mx.array, mx.array, mx.array]:
    keep_indices = single_image_keep_indices(out_probs, threshold)
    return (
        out_probs[0][keep_indices],
        out_masks[0][keep_indices],
        out_bbox[0][keep_indices],
    )


def _boxes_to_original_xyxy(out_bbox: mx.array, img_h: int, img_w: int) -> mx.array:
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox.astype(mx.float32))
    scale_fct = mx.array([img_w, img_h, img_w, img_h], dtype=mx.float32)
    return boxes * scale_fct[None, :]


def _upsample_spatial(array: mx.array, img_h: int, img_w: int) -> mx.array:
    return interpolate(
        array,
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    )


def upsample_and_activate_masks(
    out_masks: mx.array, img_h: int, img_w: int
) -> mx.array:
    """Bilinearly upsample mask logits ``(K, H, W)`` to ``img_h`` x ``img_w`` and apply a sigmoid.

    Returns probabilities of shape ``(K, 1, img_h, img_w)`` in float32. Sigmoid is
    applied after the resize, so interpolation acts on logits.
    """
    return mx.sigmoid(
        _upsample_spatial(out_masks.astype(mx.float32)[:, None], img_h, img_w)
    )


def filter_and_convert_single_image(
    outputs: Mapping[str, object],
    *,
    threshold: float,
    img_h: int,
    img_w: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return kept scores, model-grid mask logits, and source-pixel XYXY boxes.

    Scores multiply sigmoid query and image-presence probabilities and must
    strictly exceed ``threshold``. Masks retain their model-grid shape and
    logits; only boxes are scaled to ``img_w``/``img_h``. Use
    ``upsample_and_activate_masks`` for source-resolution mask probabilities.
    """

    out_probs, out_masks, out_bbox = _filter_single_image_detections(
        _presence_weighted_scores(outputs),
        cast(mx.array, outputs["pred_masks"]),
        cast(mx.array, outputs["pred_boxes"]),
        threshold=threshold,
    )
    return out_probs, out_masks, _boxes_to_original_xyxy(out_bbox, img_h, img_w)


def _validate_processor_resolution(resolution: object) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, Integral):
        raise ValueError(
            "Processor resolution must be a positive integer multiple of "
            f"{PATCH_SIZE}, got {resolution!r}."
        )
    resolution = int(resolution)
    if resolution <= 0 or resolution % PATCH_SIZE != 0:
        raise ValueError(
            "Processor resolution must be a positive integer multiple of "
            f"{PATCH_SIZE}, got {resolution}."
        )
    return resolution


def coerce_preprocessed_image(
    tensor: object,
    *,
    resolution: int,
    layout: str = "nchw",
    value_contract: str = PREPROCESSED_VALUE_CONTRACT,
) -> mx.array:
    """Accept a caller-normalized FP32 image and return batched NCHW.

    The input is one RGB image as a float32 MLX or NumPy array with finite values in
    [-1, 1] (a tolerance of 1e-4 is allowed), laid out as ``(3, H, W)`` / ``(1, 3, H,
    W)`` for ``layout="nchw"`` or ``(H, W, 3)`` / ``(1, H, W, 3)`` for ``"nhwc"``,
    with ``H == W == resolution``.

    Raises:
        TypeError: If the input is neither an MLX nor a NumPy array.
        ValueError: On an unknown ``layout`` or ``value_contract``, a non-float32
            dtype, non-finite or out-of-range values, a wrong shape, a batch other
            than one, or a spatial size different from ``resolution``.
    """

    if layout not in PREPROCESSED_LAYOUTS:
        raise ValueError(
            f"layout must be one of {PREPROCESSED_LAYOUTS}; got {layout!r}."
        )
    if value_contract != PREPROCESSED_VALUE_CONTRACT:
        raise ValueError(
            "value_contract must be "
            f"{PREPROCESSED_VALUE_CONTRACT!r}; got {value_contract!r}."
        )
    if isinstance(tensor, mx.array):
        array = tensor
    elif isinstance(tensor, np.ndarray):
        array = mx.array(tensor)
    else:
        raise TypeError("Preprocessed image must be an MLX or NumPy array.")
    if array.dtype != mx.float32:
        raise ValueError(
            "Preprocessed image must be float32; normalize in FP32, then cast "
            "once at the visual-model boundary."
        )
    if not bool(mx.all(mx.isfinite(array)).item()):
        raise ValueError("Preprocessed image must contain only finite values.")
    if array.size and (
        scalar_float(mx.min(array)) < -1.0 - 1e-4
        or scalar_float(mx.max(array)) > 1.0 + 1e-4
    ):
        raise ValueError(
            "Preprocessed image must follow the "
            f"{PREPROCESSED_VALUE_CONTRACT} contract."
        )
    if layout == "nchw":
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or int(array.shape[1]) != 3:
            raise ValueError(
                "nchw preprocessed images must have shape (3, H, W) or "
                f"(N, 3, H, W); got {tuple(int(dim) for dim in array.shape)}."
            )
    else:
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or int(array.shape[-1]) != 3:
            raise ValueError(
                "nhwc preprocessed images must have shape (H, W, 3) or "
                f"(N, H, W, 3); got {tuple(int(dim) for dim in array.shape)}."
            )
        array = mx.transpose(array, (0, 3, 1, 2))
    if int(array.shape[0]) != 1:
        raise ValueError(
            "set_preprocessed_image accepts a single image; got batch "
            f"{int(array.shape[0])}."
        )
    if int(array.shape[2]) != resolution or int(array.shape[3]) != resolution:
        raise ValueError(
            "Preprocessed spatial size must match processor resolution "
            f"{resolution}; got {int(array.shape[2])}x{int(array.shape[3])}."
        )
    return array


def _model_raw_prediction(
    model: object,
    *,
    backbone_out: Mapping[str, object],
    find_input: object,
    find_target: object | None,
    geometric_prompt: Prompt,
) -> RawPrediction:
    predict_raw = getattr(model, "predict_raw", None)
    if callable(predict_raw):
        return cast(
            RawPrediction,
            predict_raw(
                backbone_out,
                find_input,
                find_target,
                geometric_prompt,
            ),
        )
    forward = getattr(model, "forward_grounding", None)
    if not callable(forward):
        raise TypeError("Processor model must define predict_raw or forward_grounding.")
    output = forward(
        backbone_out,
        find_input,
        find_target,
        geometric_prompt,
    )
    if not isinstance(output, Mapping):
        raise TypeError("Processor forward_grounding must return a mapping.")
    return raw_prediction_from_output(cast(Mapping[str, object], output))


def _readonly_array[DTypeT: np.generic](
    value: npt.NDArray[DTypeT],
) -> npt.NDArray[DTypeT]:
    value.setflags(write=False)
    return value


@lru_cache(maxsize=128)
def _resize_weights_1d(in_size: int, out_size: int) -> ResizeWeights:
    scale = np.float32(in_size / out_size)
    support = max(scale, np.float32(1.0))
    invscale = np.float32(1.0 / float(scale)) if scale >= 1 else np.float32(1.0)
    weights_by_output: list[tuple[IndexArray, WeightArray]] = []
    for out_index in range(out_size):
        # ATen's antialias kernel stores scales, centers, and totals in FP32.
        center = np.float32(float(scale) * (out_index + 0.5))
        start = max(int(float(center - support) + 0.5), 0)
        stop = min(int(float(center + support) + 0.5), in_size)
        indices = np.arange(start, stop, dtype=np.int64)
        offsets = np.asarray(indices - center, dtype=np.float32)
        distances = np.asarray(
            (offsets.astype(np.float64) + 0.5) * float(invscale), dtype=np.float32
        )
        weights = np.maximum(np.float32(1.0) - np.abs(distances), np.float32(0.0))
        total = np.float32(0.0)
        for weight in weights:
            total = np.float32(total + weight)
        weights /= total
        weights_by_output.append((_readonly_array(indices), _readonly_array(weights)))
    return tuple(weights_by_output)


def _fused_multiply_add_float32(
    multiplicand: object, multiplier: object, addend: object
) -> npt.NDArray[np.float32]:
    return np.asarray(
        np.asarray(multiplicand, dtype=np.float64)
        * np.asarray(multiplier, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _resample_antialiased_axis(
    image: npt.NDArray[np.float32], weights: ResizeWeights, *, axis: int
) -> npt.NDArray[np.float32]:
    taps = max(len(indices) for indices, _ in weights)
    indices = np.zeros((len(weights), taps), dtype=np.int64)
    coefficients = np.zeros((len(weights), taps), dtype=np.float32)
    for row, (source_indices, source_weights) in enumerate(weights):
        indices[row, : len(source_indices)] = source_indices
        coefficients[row, : len(source_weights)] = source_weights
    shape = (-1, 1, 1) if axis == 0 else (1, -1, 1)
    output = np.take(image, indices[:, 0], axis=axis) * coefficients[:, 0].reshape(
        shape
    )
    for tap in range(1, taps):
        output = _fused_multiply_add_float32(
            np.take(image, indices[:, tap], axis=axis),
            coefficients[:, tap].reshape(shape),
            output,
        )
    return output


def _resize_uint8_bilinear_like_torchvision(
    image: npt.NDArray[np.uint8],
    size: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Resize HWC uint8 RGB data like Torchvision tensor Resize(bilinear)."""

    out_h, out_w = size
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Torchvision-style resize expects an RGB image array.")
    in_h, in_w, channels = image.shape
    if (in_h, in_w) == (out_h, out_w):
        return image.copy()

    image_f = image.astype(np.float32)
    if out_h >= in_h and out_w >= in_w:
        scale_y = np.float32(in_h / out_h)
        scale_x = np.float32(in_w / out_w)
        x_source = (
            np.arange(out_w, dtype=np.float32) + np.float32(0.5)
        ) * scale_x - np.float32(0.5)
        x0_raw = np.floor(x_source).astype(np.int64)
        x_weight = (x_source - x0_raw.astype(np.float32)).astype(np.float32)
        x0 = np.clip(x0_raw, 0, in_w - 1)
        x1 = np.clip(x0_raw + 1, 0, in_w - 1)
        inv_x_weight = np.float32(1.0) - x_weight

        resized = np.empty((out_h, out_w, channels), dtype=np.float32)
        for out_y in range(out_h):
            y_source = (np.float32(out_y) + np.float32(0.5)) * scale_y - np.float32(0.5)
            y0_raw = int(np.floor(y_source))
            y_weight = np.float32(y_source - np.float32(y0_raw))
            y0 = np.clip(y0_raw, 0, in_h - 1)
            y1 = np.clip(y0_raw + 1, 0, in_h - 1)
            top = _fused_multiply_add_float32(
                image_f[y0, x1],
                x_weight[:, None],
                np.float32(image_f[y0, x0] * inv_x_weight[:, None]),
            )
            bottom = _fused_multiply_add_float32(
                image_f[y1, x1],
                x_weight[:, None],
                np.float32(image_f[y1, x0] * inv_x_weight[:, None]),
            )
            resized[out_y] = _fused_multiply_add_float32(
                bottom,
                y_weight,
                np.float32(top * (np.float32(1.0) - y_weight)),
            )
        return np.rint(resized).clip(0, 255).astype(np.uint8)

    # ATen applies horizontal then vertical passes with FP32 fused accumulation.
    tmp = _resample_antialiased_axis(image_f, _resize_weights_1d(in_w, out_w), axis=1)
    resized = _resample_antialiased_axis(tmp, _resize_weights_1d(in_h, out_h), axis=0)
    return np.rint(resized).clip(0, 255).astype(np.uint8)


def transform(
    image_path_or_pil: ImageSource,
    resolution: object,
) -> mx.array:
    """Preprocess one image in ``"parity"`` mode: ``(3, resolution, resolution)`` float32 in [-1, 1].

    The image is converted to 8-bit RGB, resized on the host with a re-implementation
    of TorchVision's uint8 bilinear ``Resize`` (antialiased when shrinking, rounded to
    uint8 after resizing), scaled by 1/255 and normalized with mean 0.5 and std 0.5.
    The rounding to uint8 after the resize is deliberate: it follows TorchVision's
    uint8 resize.
    """
    resolution = _validate_processor_resolution(resolution)
    img = _as_pil_rgb_image(image_path_or_pil)

    img_np = np.asarray(img, dtype=np.uint8)
    img_np = _resize_uint8_bilinear_like_torchvision(
        img_np,
        (resolution, resolution),
    )
    # ToDtype scales by the FP32 reciprocal; division changes BF16 rounding at 127.
    img_mx = mx.array(img_np, dtype=mx.float32) * (1.0 / 255.0)  # [H, W, C]
    img_mx = (img_mx - 0.5) / 0.5
    return mx.transpose(img_mx, (2, 0, 1))  # [H, W, C] -> [C, H, W]


def transform_fast(
    image_path_or_pil: ImageSource,
    resolution: object,
) -> mx.array:
    """Resize and normalize on MLX (antialiased bilinear), rounding to uint8 levels like parity mode.

    The output layout and value range match ``transform``. A test on seeded random
    images bounds the maximum difference from ``transform`` to 2/255 in normalized
    units (one 8-bit level).
    """

    resolution = _validate_processor_resolution(resolution)
    img = _as_pil_rgb_image(image_path_or_pil)
    image = mx.array(np.asarray(img, dtype=np.uint8), dtype=mx.uint8)
    return _resize_normalize_mlx(image, resolution=resolution)


def _resize_normalize_mlx(image: mx.array, *, resolution: int) -> mx.array:
    """Apply the shared device-side HWC uint8 resize and normalization."""

    image = image.astype(mx.float32)
    image = mx.transpose(image, (2, 0, 1))[None]
    image = interpolate(
        image,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    # Round to uint8 levels as TorchVision's uint8 resize does; unrounded inputs
    # shift text-prompt scores past the release threshold.
    image = mx.clip(mx.round(image), 0, 255) * (1.0 / 255.0)
    return (image - 0.5) / 0.5


def _native_numpy_to_rgb(value: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
    array = np.asarray(value)
    if array.ndim not in (2, 3):
        raise ValueError("Native image arrays must use HxW or HxWxC layout.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("Native image height and width must be positive.")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        raise ValueError("Native HWC image arrays must have 1, 3, or 4 channels.")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError("Native NumPy images must use an integer or floating dtype.")
    array = cast(npt.NDArray[np.integer | np.floating], array)
    if not np.isfinite(array).all():
        raise ValueError("Native image arrays must contain only finite values.")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and array.min() >= 0.0 and array.max() <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[:, :, :3]
    return array


def _native_mlx_to_rgb(value: mx.array) -> mx.array:
    if value.ndim not in (2, 3):
        raise ValueError("Native image arrays must use HxW or HxWxC layout.")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("Native image height and width must be positive.")
    if value.ndim == 3 and value.shape[-1] not in (1, 3, 4):
        raise ValueError("Native HWC image arrays must have 1, 3, or 4 channels.")
    if not (
        mx.issubdtype(value.dtype, mx.integer)
        or mx.issubdtype(value.dtype, mx.floating)
    ):
        raise TypeError("Native MLX images must use an integer or floating dtype.")
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError("Native image arrays must contain only finite values.")
    image = value
    if mx.issubdtype(image.dtype, mx.floating) and image.size:
        minimum = scalar_float(mx.min(image))
        maximum = scalar_float(mx.max(image))
        if minimum >= 0.0 and maximum <= 1.0:
            image = image * 255.0
    image = mx.clip(image, 0, 255).astype(mx.uint8)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[-1] == 1:
        image = mx.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
    return image


def _native_hwc_rgb(image: object) -> mx.array:
    if isinstance(image, mx.array):
        return _native_mlx_to_rgb(image)
    if isinstance(image, np.ndarray):
        return mx.array(
            _native_numpy_to_rgb(cast(npt.NDArray[np.generic], image)), dtype=mx.uint8
        )
    if isinstance(image, (str, os.PathLike)):
        with Image.open(os.fspath(image)) as opened:
            if opened.mode not in {"L", "RGB", "RGBA"}:
                raise ValueError(
                    "Native path images must decode as grayscale, RGB, or RGBA."
                )
            return mx.array(
                _native_numpy_to_rgb(np.asarray(opened).copy()), dtype=mx.uint8
            )
    if isinstance(image, Image.Image):
        if image.mode not in {"L", "RGB", "RGBA"}:
            raise ValueError("Native PIL images must be grayscale, RGB, or RGBA.")
        return mx.array(_native_numpy_to_rgb(np.asarray(image)), dtype=mx.uint8)
    raise ValueError(
        "Native image must be a filesystem path, PIL image, NumPy array, or MLX array."
    )


def _transform_native_with_size(
    image: object, *, resolution: int
) -> tuple[mx.array, int, int]:
    rgb = _native_hwc_rgb(image)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return _resize_normalize_mlx(rgb, resolution=resolution), height, width


def transform_native(
    image: NativeImageSource,
    resolution: object,
) -> mx.array:
    """Resize and normalize supported HWC inputs without a PIL pixel round-trip.

    Accepts a path, PIL image, NumPy array or MLX array of shape ``(H, W)`` or
    ``(H, W, C)`` with 1, 3 or 4 channels (alpha is dropped, one channel is
    repeated). Floating-point inputs whose values all lie in [0, 1] are treated as
    unit-range and scaled by 255; all other values are clipped to [0, 255].
    Returns ``(3, resolution, resolution)`` float32 in [-1, 1].
    """

    validated_resolution = _validate_processor_resolution(resolution)
    transformed, _, _ = _transform_native_with_size(
        image, resolution=validated_resolution
    )
    return transformed


def _validate_preprocessing_mode(value: object) -> PreprocessingMode:
    if not isinstance(value, str) or value not in PREPROCESSING_MODES:
        raise ValueError(
            f"preprocessing must be one of {PREPROCESSING_MODES}; got {value!r}."
        )
    return value


def _as_pil_rgb_image(
    image: object,
) -> Image.Image:
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(os.fspath(image)) as opened:
            # convert() may return self for RGB; copy so close() cannot drop pixels.
            return opened.convert("RGB").copy()
    if isinstance(image, np.ndarray):
        array = np.asarray(cast(npt.NDArray[np.generic], image))
        if array.ndim not in (2, 3):
            raise ValueError("Image NumPy arrays must have shape HxW or HxWxC.")
        if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
            raise ValueError("Image NumPy arrays must have 1, 3, or 4 channels.")
        if not np.isfinite(array).all():
            raise ValueError("Image NumPy arrays must contain only finite values.")
        if np.issubdtype(array.dtype, np.floating):
            array = cast(npt.NDArray[np.floating], array)
            if array.size and array.min() >= 0.0 and array.max() <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array).convert("RGB")
    raise ValueError("Image must be a filesystem path, PIL image, or NumPy array.")


_IMAGE_RESULT_KEYS = (
    "geometric_prompt",
    "boxes",
    "masks",
    "masks_logits",
    "mask_logits",
    "semantic_seg",
    "scores",
)
_IMAGE_SIZE_KEYS = (
    "original_height",
    "original_width",
    "original_heights",
    "original_widths",
)
_IMAGE_LANGUAGE_KEYS = (
    "language_features",
    "language_mask",
    "language_embeds",
)


ProcessorState = dict[str, object]
OriginalSizes = list[tuple[int, int]]


def _validate_original_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"original_{name} must be an integer.")
    dimension = int(value)
    if dimension < 1:
        raise ValueError(f"original_{name} must be positive.")
    return dimension


def _validate_original_size(value: object) -> tuple[int, int]:
    """Validate pair arity before reusing the scalar dimension contract."""

    if not isinstance(value, (tuple, list)):
        raise ValueError("original_size must be (height, width).")
    size = cast(tuple[object, ...] | list[object], value)
    if len(size) != 2:
        raise ValueError("original_size must be (height, width).")
    return (
        _validate_original_dimension(size[0], "height"),
        _validate_original_dimension(size[1], "width"),
    )


def _require_int_list(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of integers.")
    result: list[int] = []
    for item in cast(list[object], value):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"{name} must be a list of integers.")
        result.append(int(item))
    return result


def _require_backbone_out(state: ProcessorState) -> dict[str, object]:
    value = state.get("backbone_out")
    if not isinstance(value, dict):
        raise TypeError("processor state backbone_out must be a dictionary.")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError("processor state backbone_out must use string keys.")
    return cast(dict[str, object], mapping)


def _require_geometric_prompt(state: ProcessorState) -> Prompt:
    value = state.get("geometric_prompt")
    if not isinstance(value, Prompt):
        raise TypeError("processor state geometric_prompt must be a Prompt.")
    return value


def _batch_original_sizes(state: ProcessorState) -> OriginalSizes | None:
    has_heights = "original_heights" in state
    has_widths = "original_widths" in state
    if has_heights != has_widths:
        raise ValueError(
            "Batch state must contain both original_heights and original_widths."
        )
    if not has_heights:
        return None
    heights = _require_int_list(
        state["original_heights"],
        name="original_heights",
    )
    widths = _require_int_list(
        state["original_widths"],
        name="original_widths",
    )
    if len(heights) != len(widths):
        raise ValueError(
            "original_heights and original_widths must have the same length."
        )
    if len(heights) == 0:
        raise ValueError("Batch state must contain at least one original image size.")
    return list(zip(heights, widths))


def _clear_prompt_and_result_keys(state: ProcessorState) -> None:
    """Remove prompts and prediction outputs from a mutable processor state."""
    if "backbone_out" in state:
        backbone_out = _require_backbone_out(state)
        for key in _IMAGE_LANGUAGE_KEYS:
            backbone_out.pop(key, None)

    for key in _IMAGE_RESULT_KEYS:
        state.pop(key, None)


def _clear_image_dependent_state(state: ProcessorState) -> None:
    """Invalidate prompts, outputs, and size metadata before replacing an image.

    Reused states must not retain geometric prompts, opposite batch size keys, or
    stale outputs from a previous single-image or batch call.
    """
    _clear_prompt_and_result_keys(state)
    for key in _IMAGE_SIZE_KEYS:
        state.pop(key, None)


def _normalize_processor_device(device: object) -> str:
    if is_mlx_runtime_device(device):
        return "mlx"
    _raise_processor_unsupported(
        f"sam3_mlx.model.sam3_image_processor.Sam3Processor(device={device!r})",
        reason="unsupported-device",
        detail=(
            "sam3_mlx only runs on the explicit MLX runtime. Non-MLX "
            "device strings are not accepted as aliases."
        ),
        alternative="device='mlx'",
    )


class _ArrayLayer(Protocol):
    def __call__(self, value: mx.array) -> mx.array: ...


class _MaskDecoder(Protocol):
    conv_s0: _ArrayLayer
    conv_s1: _ArrayLayer


class _InteractiveModel(Protocol):
    sam_mask_decoder: _MaskDecoder


class _InteractivePredictor(Protocol):
    model: _InteractiveModel


class _ForwardImage(Protocol):
    def __call__(self, image: mx.array, /) -> object: ...


class _ForwardText(Protocol):
    def __call__(
        self,
        prompts: list[str],
        /,
        *,
        device: str | None = None,
    ) -> object: ...


class _ProcessorModel(Protocol):
    @property
    def backbone(self) -> object: ...

    @property
    def inst_interactive_predictor(self) -> object | None: ...

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt: ...

    def forward_grounding(
        self,
        backbone_out: Mapping[str, object],
        find_input: object,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> Output: ...


def _forward_backbone_image(
    backbone: object,
    image: mx.array,
    *,
    precision: object = "fp32",
) -> dict[str, object]:
    forward_value = getattr(backbone, "forward_image", None)
    if not callable(forward_value):
        raise TypeError("Processor model backbone must define forward_image.")
    forward = cast(_ForwardImage, forward_value)
    output = forward(cast_visual_input(image, parse_precision(precision)))
    if not isinstance(output, Mapping):
        raise TypeError("Processor backbone forward_image must return a mapping.")
    string_output: dict[str, object] = {}
    for key, value in cast(Mapping[object, object], output).items():
        if not isinstance(key, str):
            raise TypeError("Processor backbone output keys must be strings.")
        string_output[key] = value
    return string_output


def _validate_text_outputs(value: object) -> dict[str, mx.array]:
    if not isinstance(value, Mapping):
        raise TypeError("Processor text output must be a mapping.")
    text_output: dict[str, mx.array] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not isinstance(item, mx.array):
            raise TypeError("Processor text output must map string keys to MLX arrays.")
        text_output[key] = item
    return text_output


def normalize_text_prompt_key(prompt: object) -> str:
    """Return the exact whitespace-normalized cache key for one prompt."""

    if not isinstance(prompt, str):
        raise TypeError("Text prompt must be a string.")
    return " ".join(prompt.split())


def _copy_text_outputs(value: Mapping[str, mx.array]) -> dict[str, mx.array]:
    return {key: item for key, item in value.items()}


def _validate_text_cache_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("text_cache_size must be a non-negative integer.")
    return int(value)


def _forward_backbone_text(
    backbone: object, prompts: list[str], *, device: str
) -> dict[str, mx.array]:
    forward_value = getattr(backbone, "forward_text", None)
    if not callable(forward_value):
        raise TypeError("Processor model backbone must define forward_text.")
    forward = cast(_ForwardText, forward_value)
    return _validate_text_outputs(forward(prompts, device=device))


def _evaluate_processor_state(state: ProcessorState) -> None:
    evaluate = cast(Callable[[object], None], getattr(mx, "eval"))
    evaluate(state)


def _model_dummy_prompt(
    model: _ProcessorModel,
    *,
    num_prompts: int = 1,
) -> Prompt:
    factory = getattr(model, "_get_dummy_prompt")
    return factory(num_prompts=num_prompts)


def _parse_prompt_batch(
    coordinates: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[Sequence[bool]],
    *,
    batch_size: int,
    width: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pack boundary inputs into the canonical sequence-first Prompt layout."""
    if len(coordinates) != batch_size or len(labels) != batch_size:
        raise ValueError("Prompt coordinates and labels must have one entry per image.")
    count = max(map(len, coordinates), default=0)
    values = np.zeros((count, batch_size, width), dtype=np.float32)
    label_values = np.zeros((count, batch_size), dtype=np.int64)
    mask = np.ones((batch_size, count), dtype=bool)
    for image_idx, (image_coords, image_labels) in enumerate(zip(coordinates, labels)):
        length = len(image_coords)
        if len(image_labels) != length:
            raise ValueError(
                f"Prompt labels must match coordinates for image {image_idx}."
            )
        if length == 0:
            continue
        parsed = np.asarray(image_coords, dtype=np.float32)
        if parsed.shape != (length, width):
            raise ValueError(f"Prompt coordinates must have shape (N, {width}).")
        if not np.isfinite(parsed).all() or np.any((parsed < 0) | (parsed > 1)):
            raise ValueError(
                "Prompt coordinates must be finite and normalized to [0, 1]."
            )
        if any(
            not isinstance(cast(object, label), (bool, np.bool_))
            for label in image_labels
        ):
            raise TypeError("Prompt labels must be positive/negative booleans.")
        values[:length, image_idx] = parsed
        label_values[:length, image_idx] = image_labels
        mask[image_idx, :length] = False
    return mx.array(values), mx.array(label_values), mx.array(mask)


class Sam3Processor:
    """Stateful helper that runs the SAM 3 image model from images and prompts.

    Usage: ``set_image`` (or ``set_image_batch``) returns a state dictionary; add a
    text prompt with ``set_text_prompt`` and/or geometric prompts with
    ``add_geometric_prompt`` / ``add_point_prompt``; each prompt call re-runs
    grounding unless ``run_grounding=False``. When replacing an image in a
    previously returned state, the previous image, prompts, and results remain
    available if preprocessing, encoding, evaluation, or interactive preparation
    fails. A successful replacement updates the same dictionary and clears its
    previous prompts and results.

    Result keys written into the state by ``predict``, for a single image: ``boxes``
    ``(K, 4)`` xyxy in original-image pixels, ``scores`` ``(K,)``, ``masks`` ``(K, 1,
    H, W)`` bool at the original size, and ``masks_logits`` (same shape, but holding
    sigmoid probabilities, not logits) plus ``semantic_seg`` when the model
    produces it. For a batch state each of these is a list with one entry per image.
    Detections are kept when the product of the query probability and the image
    presence probability is strictly above ``confidence_threshold``.

    Args:
        model: The SAM 3 image model.
        resolution: Square input size in pixels; a positive multiple of the patch
            size.
        device: Only the MLX runtime is supported.
        confidence_threshold: Score threshold in [0, 1].
        text_cache_size: Number of text-encoder outputs to cache by normalized
            prompt; 0 disables caching.
        preprocessing: ``"parity"`` (host TorchVision-style resize), ``"fast"`` (MLX
            resize) or ``"native"`` (MLX resize from raw HWC pixels).
    """

    def __init__(
        self,
        model: _ProcessorModel,
        resolution: object = DEFAULT_IMAGE_RESOLUTION,
        device: object = "mlx",
        confidence_threshold: float = 0.5,
        text_cache_size: object = DEFAULT_TEXT_CACHE_SIZE,
        preprocessing: object = "parity",
    ) -> None:
        runtime_device = _normalize_processor_device(device)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        cache_size = _validate_text_cache_size(text_cache_size)
        preprocessing_mode = _validate_preprocessing_mode(preprocessing)
        self.model = model
        self.resolution = _validate_processor_resolution(resolution)
        self.device = runtime_device
        self.confidence_threshold = confidence_threshold
        self.text_cache_size = cache_size
        self.preprocessing = preprocessing_mode
        self._text_cache: BoundedLRUCache[str, dict[str, mx.array]] | None = (
            None if cache_size == 0 else BoundedLRUCache(cache_size)
        )
        transform_image = {
            "parity": transform,
            "fast": transform_fast,
            "native": transform_native,
        }[preprocessing_mode]
        self.transform = partial(transform_image, resolution=self.resolution)

        self.find_stage = FindStage(
            img_ids=mx.array([0], dtype=mx.int64),
            text_ids=mx.array([0], dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _find_stage_for_state(self, state: ProcessorState) -> FindStage:
        sizes = _batch_original_sizes(state)
        if sizes is None:
            return self.find_stage
        batch_size = len(sizes)
        return FindStage(
            img_ids=mx.arange(batch_size, dtype=mx.int64),
            text_ids=mx.zeros((batch_size,), dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def clear_text_cache(self) -> None:
        """Drop every cached text-encoder output for this processor."""

        if self._text_cache is not None:
            self._text_cache.clear()

    def _text_outputs_for_prompt(self, prompt: str) -> dict[str, mx.array]:
        cache = self._text_cache
        if cache is None:
            return _forward_backbone_text(
                self.model.backbone,
                [prompt],
                device=self.device,
            )
        key = normalize_text_prompt_key(prompt)
        cached = cache.get(key)
        if cached is not None:
            return _copy_text_outputs(cached)
        encoded = _forward_backbone_text(
            self.model.backbone,
            [key],
            device=self.device,
        )
        cache[key] = encoded
        return _copy_text_outputs(encoded)

    def _patch_interactive_backbone_features(
        self, backbone_out: Mapping[str, object]
    ) -> None:
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if not inst_interactivity_en or "sam2_backbone_out" not in backbone_out:
            return
        predictor = cast(_InteractivePredictor, self.model.inst_interactive_predictor)
        sam2_backbone_out = cast(
            Mapping[Literal["backbone_fpn"], list[mx.array]],
            backbone_out["sam2_backbone_out"],
        )
        backbone_fpn = sam2_backbone_out["backbone_fpn"]
        backbone_fpn[0] = predictor.model.sam_mask_decoder.conv_s0(backbone_fpn[0])
        backbone_fpn[1] = predictor.model.sam_mask_decoder.conv_s1(backbone_fpn[1])

    def _replace_image_state(
        self,
        image_tensor: mx.array,
        *,
        new_state: ProcessorState,
        state: ProcessorState | None,
    ) -> ProcessorState:
        backbone_out = _forward_backbone_image(
            self.model.backbone,
            image_tensor,
            precision=model_precision(self.model),
        )
        new_state["backbone_out"] = backbone_out
        _evaluate_processor_state(new_state)
        self._patch_interactive_backbone_features(backbone_out)
        if state is None:
            return new_state
        _clear_image_dependent_state(state)
        state.update(new_state)
        return state

    def _apply_visual_tensor(
        self,
        image_tensor: mx.array,
        *,
        original_height: int,
        original_width: int,
        state: ProcessorState | None,
    ) -> ProcessorState:
        return self._replace_image_state(
            image_tensor,
            new_state={
                "original_height": original_height,
                "original_width": original_width,
            },
            state=state,
        )

    def set_image(
        self,
        image: NativeImageSource,
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        """Encode one path, PIL image, or NumPy array (MLX arrays too in ``"native"`` mode).

        A passed ``state`` is reused after dropping its previous prompts and results.
        The original image size is stored for rescaling boxes and masks. The backbone
        outputs are evaluated before returning.
        """

        if self.preprocessing == "native":
            image_tensor, height, width = _transform_native_with_size(
                image, resolution=self.resolution
            )
        else:
            pil_image = _as_pil_rgb_image(image)
            width, height = pil_image.size
            image_tensor = self.transform(pil_image)
        return self._apply_visual_tensor(
            image_tensor[None],
            original_height=height,
            original_width=width,
            state=state,
        )

    def set_preprocessed_image(
        self,
        tensor: object,
        *,
        original_size: tuple[int, int] | list[int],
        layout: str = "nchw",
        value_contract: str = PREPROCESSED_VALUE_CONTRACT,
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        """Encode an image that the caller already preprocessed (see ``coerce_preprocessed_image``).

        ``original_size`` is the ``(height, width)`` of the source image, used to
        rescale predictions.
        """
        height, width = _validate_original_size(original_size)
        image_tensor = coerce_preprocessed_image(
            tensor,
            resolution=self.resolution,
            layout=layout,
            value_contract=value_contract,
        )
        return self._apply_visual_tensor(
            image_tensor,
            original_height=height,
            original_width=width,
            state=state,
        )

    def set_image_batch(
        self,
        images: object,
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        """Encode a batch of paths, PIL images, or NumPy arrays."""

        new_state: ProcessorState = {}
        if not isinstance(images, list):
            raise ValueError(
                "Images must be a list of filesystem paths, PIL images, or NumPy arrays."
            )
        typed_images = cast(list[NativeImageSource], images)
        if len(typed_images) == 0:
            raise ValueError("Images list must not be empty.")
        if self.preprocessing == "native":
            transformed = [
                _transform_native_with_size(image, resolution=self.resolution)
                for image in typed_images
            ]
            new_state["original_heights"] = [height for _, height, _ in transformed]
            new_state["original_widths"] = [width for _, _, width in transformed]
            image_batch = mx.stack([tensor for tensor, _, _ in transformed], axis=0)
        else:
            pil_images = [_as_pil_rgb_image(image) for image in typed_images]
            new_state["original_heights"] = [image.height for image in pil_images]
            new_state["original_widths"] = [image.width for image in pil_images]
            image_batch = mx.stack(
                [self.transform(image) for image in pil_images], axis=0
            )
        return self._replace_image_state(
            image_batch,
            new_state=new_state,
            state=state,
        )

    def set_text_prompt(
        self,
        prompt: str,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
        text_outputs: object | None = None,
    ) -> ProcessorState:
        """Set the text prompt for the image(s) in ``state`` and, by default, run grounding.

        ``text_outputs`` may supply precomputed text-encoder outputs instead of
        encoding ``prompt``. Raises ``ValueError`` if no image has been set.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        validated_text_outputs = (
            self._text_outputs_for_prompt(prompt)
            if text_outputs is None
            else _validate_text_outputs(text_outputs)
        )
        # Reset the previous text prompt.
        backbone_out = _require_backbone_out(state)
        backbone_out.update(validated_text_outputs)
        if "geometric_prompt" not in state:
            sizes = _batch_original_sizes(state)
            num_prompts = len(sizes) if sizes is not None else 1
            state["geometric_prompt"] = _model_dummy_prompt(
                self.model,
                num_prompts=num_prompts,
            )
        return self._forward_grounding(state) if run_grounding else state

    def add_geometric_prompt(
        self,
        box: list[float],
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt (the text
        prompt "visual" is used when none was given).
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        For a batch state, the same normalized prompt is appended to every image.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_geometric_prompt")
        sizes = _batch_original_sizes(state)
        batch_size = len(sizes) if sizes is not None else 1

        backbone_out = _require_backbone_out(state)
        if "language_features" not in backbone_out:
            # Use "visual" when no text prompt is present so geometry-only
            # prompts work.
            backbone_out.update(self._text_outputs_for_prompt("visual"))

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _model_dummy_prompt(
                self.model, num_prompts=batch_size
            )

        boxes = mx.broadcast_to(
            mx.reshape(mx.array(box, dtype=mx.float32), (1, 1, 4)),
            (1, batch_size, 4),
        )
        labels = mx.full((1, batch_size), label, dtype=mx.bool_)
        geometric_prompt = cast(Prompt, state["geometric_prompt"])
        geometric_prompt.append_boxes(boxes, labels)

        return self._forward_grounding(state) if run_grounding else state

    def add_point_prompt(
        self,
        point: list[float],
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Adds a point prompt and run inference on the current image.

        The point is expected in normalized ``[x, y]`` image coordinates.
        For a batch state, the same normalized prompt is appended to every image.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")
        sizes = _batch_original_sizes(state)
        batch_size = len(sizes) if sizes is not None else 1

        backbone_out = _require_backbone_out(state)
        if "language_features" not in backbone_out:
            backbone_out.update(self._text_outputs_for_prompt("visual"))

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _model_dummy_prompt(
                self.model, num_prompts=batch_size
            )

        points = mx.broadcast_to(
            mx.reshape(mx.array(point, dtype=mx.float32), (1, 1, 2)),
            (1, batch_size, 2),
        )
        labels = mx.full((1, batch_size), label, dtype=mx.bool_)
        geometric_prompt = cast(Prompt, state["geometric_prompt"])
        geometric_prompt.append_points(points, labels)
        return self._forward_grounding(state) if run_grounding else state

    def add_geometric_prompt_batch(
        self,
        boxes: Sequence[Sequence[Sequence[float]]],
        labels: Sequence[Sequence[bool]],
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Append per-image normalized cxcywh boxes, allowing unequal counts."""
        return self._append_prompt_batch(
            boxes, labels, state, kind="box", run_grounding=run_grounding
        )

    def add_point_prompt_batch(
        self,
        points: Sequence[Sequence[Sequence[float]]],
        labels: Sequence[Sequence[bool]],
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Append per-image normalized xy points, allowing unequal counts."""
        return self._append_prompt_batch(
            points, labels, state, kind="point", run_grounding=run_grounding
        )

    def _append_prompt_batch(
        self,
        coordinates: Sequence[Sequence[Sequence[float]]],
        labels: Sequence[Sequence[bool]],
        state: ProcessorState,
        *,
        kind: Literal["box", "point"],
        run_grounding: bool,
    ) -> ProcessorState:
        backbone_out = _require_backbone_out(state)
        sizes = _batch_original_sizes(state)
        batch_size = len(sizes) if sizes is not None else 1
        values, prompt_labels, mask = _parse_prompt_batch(
            coordinates, labels, batch_size=batch_size, width=4 if kind == "box" else 2
        )
        if "language_features" not in backbone_out:
            backbone_out.update(self._text_outputs_for_prompt("visual"))
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _model_dummy_prompt(
                self.model, num_prompts=batch_size
            )
        prompt = _require_geometric_prompt(state)
        if kind == "box":
            prompt.append_boxes(values, prompt_labels, mask)
        else:
            prompt.append_points(values, prompt_labels, mask)
        return self._forward_grounding(state) if run_grounding else state

    def run_grounding(self, state: ProcessorState) -> ProcessorState:
        """Execute one grounding pass after prompt mutation is complete (same as ``predict``)."""
        return self.predict(state)

    def predict_raw(self, state: ProcessorState) -> RawPrediction:
        """Fixed-shape device arrays before filtering or original-res upsample.

        Masks and boxes are on the model grid / normalized coordinates and every
        query is present; nothing is thresholded.
        """

        return _model_raw_prediction(
            self.model,
            backbone_out=_require_backbone_out(state),
            find_input=self._find_stage_for_state(state),
            find_target=None,
            geometric_prompt=_require_geometric_prompt(state),
        )

    def predict(self, state: ProcessorState) -> ProcessorState:
        """Filter, upsample, and write host-facing boxes/masks/scores into state."""

        return self._materialize_prediction(state, self.predict_raw(state))

    def reset_all_prompts(self, state: ProcessorState) -> None:
        """Removes all the prompts and results (the encoded image is kept)."""
        _clear_prompt_and_result_keys(state)

    def set_confidence_threshold(
        self, threshold: float, state: ProcessorState | None = None
    ) -> ProcessorState | None:
        """Set the score threshold; re-run grounding on ``state`` if it already has results."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        self.confidence_threshold = float(threshold)
        if state is not None and "boxes" in state:
            return self._forward_grounding(state)
        return state

    def _forward_grounding(self, state: ProcessorState) -> ProcessorState:
        return self.predict(state)

    def _materialize_prediction(
        self, state: ProcessorState, outputs: Mapping[str, object]
    ) -> ProcessorState:
        batch_sizes = _batch_original_sizes(state)

        out_bbox = cast(mx.array, outputs["pred_boxes"])
        out_masks = cast(mx.array, outputs["pred_masks"])
        out_probs = _presence_weighted_scores(outputs)

        if batch_sizes is not None:
            return self._forward_grounding_batch_outputs(
                state,
                outputs=outputs,
                out_bbox=out_bbox,
                out_masks=out_masks,
                out_probs=out_probs,
                original_sizes=batch_sizes,
            )

        if out_probs.shape[0] != 1:
            _raise_processor_unsupported(
                "sam3_mlx.model.sam3_image_processor.Sam3Processor._forward_grounding(batch_output)",
                reason="image-interactivity",
                detail=(
                    "Batch grounding is not supported in the MLX port yet; "
                    f"received batch size {out_probs.shape[0]}."
                ),
                alternative="set_image for a single image",
            )

        img_h = _validate_original_dimension(state.get("original_height"), "height")
        img_w = _validate_original_dimension(state.get("original_width"), "width")
        out_probs, out_masks, boxes = filter_and_convert_single_image(
            outputs,
            threshold=self.confidence_threshold,
            img_h=img_h,
            img_w=img_w,
        )
        out_masks = upsample_and_activate_masks(out_masks, img_h, img_w)
        seg_mask = cast(mx.array | None, outputs.get("semantic_seg"))
        if seg_mask is not None:
            state["semantic_seg"] = _upsample_spatial(seg_mask, img_h, img_w)
        # Named "masks_logits" for parity with the official state dict key, but
        # upsample_and_activate_masks already applied sigmoid, so this holds
        # probabilities (hence the > 0.5 threshold below), not logits.
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def _forward_grounding_batch_outputs(
        self,
        state: ProcessorState,
        *,
        outputs: Mapping[str, object],
        out_bbox: mx.array,
        out_masks: mx.array,
        out_probs: mx.array,
        original_sizes: OriginalSizes,
    ) -> ProcessorState:
        batch_size = len(original_sizes)
        if out_probs.shape[0] != batch_size:
            raise ValueError(
                "Batch grounding output batch size must match original image sizes; "
                f"got outputs batch {out_probs.shape[0]} and {batch_size} sizes."
            )

        boxes_by_image: list[mx.array] = []
        masks_logits_by_image: list[mx.array] = []
        masks_by_image: list[mx.array] = []
        scores_by_image: list[mx.array] = []
        semantic_seg = cast(mx.array | None, outputs.get("semantic_seg"))
        semantic_seg_by_image: list[mx.array] | None = (
            [] if semantic_seg is not None else None
        )

        for batch_idx, (img_h, img_w) in enumerate(original_sizes):
            keep_indices = _score_keep_indices(
                out_probs[batch_idx], self.confidence_threshold
            )
            image_scores = out_probs[batch_idx][keep_indices]
            image_masks = out_masks[batch_idx][keep_indices]
            image_boxes = out_bbox[batch_idx][keep_indices]
            boxes = _boxes_to_original_xyxy(image_boxes, img_h, img_w)
            image_masks = upsample_and_activate_masks(image_masks, img_h, img_w)

            boxes_by_image.append(boxes)
            masks_logits_by_image.append(image_masks)
            masks_by_image.append(image_masks > 0.5)
            scores_by_image.append(image_scores)

            if semantic_seg_by_image is not None:
                assert semantic_seg is not None
                semantic_seg_by_image.append(
                    _upsample_spatial(
                        semantic_seg[batch_idx : batch_idx + 1], img_h, img_w
                    )
                )

        if semantic_seg_by_image is not None:
            state["semantic_seg"] = semantic_seg_by_image
        # See the single-image branch above: "masks_logits" holds probabilities,
        # matching the official state dict key name.
        state["masks_logits"] = masks_logits_by_image
        state["masks"] = masks_by_image
        state["boxes"] = boxes_by_image
        state["scores"] = scores_by_image
        return state
