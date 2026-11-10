# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Image-prompt processor: preprocessing, prompt state and result post-processing.

``Sam3Processor`` resizes 8-bit images to a square ``resolution`` x ``resolution``
grid (without keeping the aspect ratio) on the host with the TorchVision-parity
FP32 bilinear resize, converts them to RGB float32 CHW and normalizes with mean
0.5 and std 0.5 (values in [-1, 1]). Prompt boxes are normalized (cx, cy, w, h)
and points normalized (x, y), both in [0, 1] relative to the original image.
Predicted boxes are returned as (x0, y0, x1, y1) in original image pixels.
"""

from collections.abc import Mapping, Sequence
import os
from functools import lru_cache
from numbers import Integral
from typing import Literal, Protocol, TypeAlias, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from PIL import Image

from sam3_mlx.mlx_runtime import evaluate_boundary, scalar_int
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy
from sam3_mlx.model.data_misc import FindStage, interpolate
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.io_utils import EIGHT_BIT_HINT, reject_high_bit_depth
from sam3_mlx.model.sam3_image import RawPrediction
from sam3_mlx.precision import cast_visual_input, model_precision
from sam3_mlx.resolutions import DEFAULT_IMAGE_RESOLUTION, PATCH_SIZE

DEFAULT_TEXT_CACHE_SIZE = 8
IndexArray: TypeAlias = npt.NDArray[np.int64]
WeightArray: TypeAlias = npt.NDArray[np.float32]
ResizeWeights: TypeAlias = tuple[IndexArray, WeightArray]
ImageSource: TypeAlias = str | os.PathLike[str] | Image.Image | npt.NDArray[np.generic]
ProcessorState = dict[str, object]


def _score_keep_indices(scores: mx.array, threshold: float) -> mx.array:
    """Return ascending query indices whose scores strictly exceed the threshold."""
    keep = scores > threshold
    # MLX has no boolean indexing/nonzero yet. Sync only the scalar count, then
    # build the ordered variable-length index vector on device.
    keep_count = scalar_int(mx.sum(keep))
    if keep_count == 0:
        return mx.array([], dtype=mx.int64)
    positions = mx.arange(keep.shape[0], dtype=mx.int64)
    sentinel = mx.array(keep.shape[0], dtype=mx.int64)
    return mx.sort(mx.where(keep, positions, sentinel))[:keep_count]


def _presence_weighted_scores(outputs: RawPrediction) -> mx.array:
    """Multiply sigmoid query and image-presence scores, returning ``(B, Q)`` float32."""
    out_logits = outputs["pred_logits"]
    presence_logit = outputs["presence_logit_dec"]
    if out_logits.dtype != mx.bfloat16:
        out_logits = out_logits.astype(mx.float32)
        presence_logit = presence_logit.astype(mx.float32)
    # BF16 probabilities are rounded to BF16 before thresholding, as upstream.
    out_probs = mx.sigmoid(out_logits.astype(mx.float32)).astype(out_logits.dtype)
    presence_score = mx.sigmoid(presence_logit.astype(mx.float32)).astype(presence_logit.dtype)
    return (out_probs * presence_score[:, None]).squeeze(-1).astype(mx.float32)


def _boxes_to_original_xyxy(out_bbox: mx.array, img_h: int, img_w: int) -> mx.array:
    """Scale normalized center-format boxes to original-image pixel XYXY coordinates."""
    boxes = box_cxcywh_to_xyxy(out_bbox.astype(mx.float32))
    scale_fct = mx.array([img_w, img_h, img_w, img_h], dtype=mx.float32)
    return boxes * scale_fct[None, :]


def _upsample_spatial(array: mx.array, img_h: int, img_w: int) -> mx.array:
    """Bilinearly resize spatial arrays to the requested image dimensions."""
    return interpolate(array, size=(img_h, img_w))


def _upsample_boolean_masks(
    out_masks: mx.array, img_h: int, img_w: int, batch_bytes: int
) -> mx.array:
    """Resize logits, apply the FP32 sigmoid and threshold in bounded mask groups.

    ``batch_bytes`` bounds one FP32 resized group's payload, with at least one
    mask per group. The returned bool tensor still holds every retained mask.
    """
    count = out_masks.shape[0]
    group = max(1, batch_bytes // (4 * img_h * img_w))
    if count <= group:
        return (
            mx.sigmoid(
                _upsample_spatial(out_masks.astype(mx.float32)[:, None], img_h, img_w)
            )
            > 0.5
        )
    result = mx.zeros((count, 1, img_h, img_w), dtype=mx.bool_)
    for start in range(0, count, group):
        stop = min(start + group, count)
        probabilities = mx.sigmoid(
            _upsample_spatial(
                out_masks[start:stop].astype(mx.float32)[:, None], img_h, img_w
            )
        )
        result[start:stop] = probabilities > 0.5
        evaluate_boundary(result)
    return result


def _validate_processor_resolution(resolution: object) -> int:
    """Require a positive integer resolution divisible by the model patch size."""
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


def _readonly_array[DTypeT: np.generic](
    value: npt.NDArray[DTypeT],
) -> npt.NDArray[DTypeT]:
    """Mark a cached NumPy array read-only and return it."""
    value.setflags(write=False)
    return value


@lru_cache(maxsize=128)
def _resize_weights_1d(in_size: int, out_size: int) -> ResizeWeights:
    """Cache immutable packed antialias indices and FP32 weights for one resize axis."""
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
    taps = max(len(indices) for indices, _ in weights_by_output)
    packed_indices = np.zeros((out_size, taps), dtype=np.int64)
    coefficients = np.zeros((out_size, taps), dtype=np.float32)
    for row, (source_indices, source_weights) in enumerate(weights_by_output):
        packed_indices[row, : len(source_indices)] = source_indices
        coefficients[row, : len(source_weights)] = source_weights
    return _readonly_array(packed_indices), _readonly_array(coefficients)


def _fused_multiply_add_float32(
    multiplicand: object, multiplier: object, addend: object
) -> npt.NDArray[np.float32]:
    """Approximate fused multiply-add by evaluating in float64 and rounding to float32."""
    return np.asarray(
        np.asarray(multiplicand, dtype=np.float64)
        * np.asarray(multiplier, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _resample_antialiased_axis(
    image: npt.NDArray[np.float32], weights: ResizeWeights, *, axis: int
) -> npt.NDArray[np.float32]:
    """Apply cached weights in strips, preserving each pixel's FP32 FMA sequence.

    Each FP64 image temporary is at most 4 MiB, except when a single output
    strip is larger. The float32 destination contains the complete resized image.
    """
    indices, coefficients = weights
    taps = indices.shape[1]
    shape = (-1, 1, 1) if axis == 0 else (1, -1, 1)
    output_shape = list(image.shape)
    output_shape[axis] = len(indices)
    output = np.empty(output_shape, dtype=np.float32)
    across = 1 - axis
    strip_size = max(1, (4 * 1024**2) // (8 * len(indices) * image.shape[2]))
    for start in range(0, image.shape[across], strip_size):
        window = [slice(None)] * 3
        window[across] = slice(start, start + strip_size)
        region = tuple(window)
        source = image[region]
        tile = np.take(source, indices[:, 0], axis=axis) * coefficients[:, 0].reshape(
            shape
        )
        for tap in range(1, taps):
            tile = _fused_multiply_add_float32(
                np.take(source, indices[:, tap], axis=axis),
                coefficients[:, tap].reshape(shape),
                tile,
            )
        output[region] = tile
    return output


def _resize_uint8_bilinear_like_torchvision(
    image: npt.NDArray[np.uint8],
    size: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Resize HWC uint8 RGB with the FP32 bilinear reference and uint8 rounding."""

    out_h, out_w = size
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Torchvision-style resize expects an RGB image array.")
    in_h, in_w, _ = image.shape
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
        y_source = (
            np.arange(out_h, dtype=np.float32) + np.float32(0.5)
        ) * scale_y - np.float32(0.5)
        y0_raw = np.floor(y_source).astype(np.int64)
        y_weight = (y_source - y0_raw.astype(np.float32))[:, None, None]
        y0 = np.clip(y0_raw, 0, in_h - 1)
        y1 = np.clip(y0_raw + 1, 0, in_h - 1)

        # Interpolate each source row along x once, then blend the two source
        # rows of every output row; each element sees the same FP32 operations.
        rows = _fused_multiply_add_float32(
            image_f[:, x1],
            x_weight[:, None],
            np.float32(image_f[:, x0] * inv_x_weight[:, None]),
        )
        top = rows[y0]
        resized = _fused_multiply_add_float32(
            rows[y1],
            y_weight,
            np.float32(top * (np.float32(1.0) - y_weight)),
        )
        return np.rint(resized).clip(0, 255).astype(np.uint8)

    # ATen applies horizontal then vertical passes with FP32 fused accumulation.
    tmp = _resample_antialiased_axis(image_f, _resize_weights_1d(in_w, out_w), axis=1)
    resized = _resample_antialiased_axis(tmp, _resize_weights_1d(in_h, out_h), axis=0)
    return np.rint(resized).clip(0, 255).astype(np.uint8)


def _preprocess_rgb(image: Image.Image, resolution: int) -> mx.array:
    """Resize and normalize an 8-bit RGB PIL image to ``(3, resolution, resolution)``."""
    img_np = _resize_uint8_bilinear_like_torchvision(
        np.asarray(image, dtype=np.uint8),
        (resolution, resolution),
    )
    # ToDtype scales by the FP32 reciprocal; division changes BF16 rounding at 127.
    img_mx = mx.array(img_np, dtype=mx.float32) * (1.0 / 255.0)  # [H, W, C]
    img_mx = (img_mx - 0.5) / 0.5
    return mx.transpose(img_mx, (2, 0, 1))  # [H, W, C] -> [C, H, W]


def transform(image: ImageSource, resolution: object) -> mx.array:
    """Preprocess one image: ``(3, resolution, resolution)`` float32 in [-1, 1].

    The image is converted to 8-bit RGB, resized on the host with the FP32
    bilinear reference (antialiased when shrinking and rounded to uint8 after
    resizing), scaled by 1/255 and normalized with mean 0.5 and std 0.5.
    Newer TorchVision versions may use a different native uint8 resize kernel.

    Raises:
        ValueError: For an invalid ``resolution`` or an unsupported image (see
            ``Sam3Processor.set_image``).
    """
    return _preprocess_rgb(
        _as_pil_rgb_image(image), _validate_processor_resolution(resolution)
    )


def _numpy_to_uint8(array: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
    """Validate an HxW or HxWxC array and convert it to uint8 without clipping data."""
    if array.ndim not in (2, 3):
        raise ValueError("Image NumPy arrays must have shape HxW or HxWxC.")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        raise ValueError("Image NumPy arrays must have 1, 3, or 4 channels.")
    is_integer = np.issubdtype(array.dtype, np.integer)
    if not (is_integer or np.issubdtype(array.dtype, np.floating)):
        raise TypeError("Image NumPy arrays must have an integer or floating dtype.")
    values = cast(npt.NDArray[np.integer | np.floating], array)
    if not np.isfinite(values).all():
        raise ValueError("Image NumPy arrays must contain only finite values.")
    if values.size:
        low, high = float(values.min()), float(values.max())
        if not is_integer and low >= 0.0 and high <= 1.0:
            # Upstream's float path, v2.ToDtype(torch.uint8, scale=True), multiplies
            # by 255 + 1 - 1e-3 in the input dtype and truncates, so 0.999 maps to 255
            # rather than 254. Narrower floats scale in float32, where the factor
            # stays below 256.
            wide = np.float64 if values.dtype == np.float64 else np.float32
            values = values.astype(wide) * wide(255.0 + 1.0 - 1e-3)
        elif low < 0.0 or high > 255.0:
            raise ValueError(
                f"Image NumPy array values span [{low:g}, {high:g}], outside the "
                f"8-bit range [0, 255]; {EIGHT_BIT_HINT}."
            )
    return values.astype(np.uint8)


def _as_pil_rgb_image(image: object) -> Image.Image:
    """Load a path, PIL image or NumPy array as an 8-bit PIL RGB image.

    Floating NumPy arrays whose values all lie in [0, 1] are converted as
    TorchVision's ``ToDtype(torch.uint8, scale=True)`` does.
    Pillow modes ``I``, ``I;16*`` and ``F`` and arrays with values outside
    [0, 255] are rejected rather than clipped.
    """
    if isinstance(image, Image.Image):
        reject_high_bit_depth(image, source="The image")
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(os.fspath(image)) as opened:
            reject_high_bit_depth(opened, source=os.fspath(image))
            # convert() may return self for RGB; copy so close() cannot drop pixels.
            return opened.convert("RGB").copy()
    if isinstance(image, np.ndarray):
        array = _numpy_to_uint8(cast(npt.NDArray[np.generic], image))
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array).convert("RGB")
    raise ValueError("Image must be a filesystem path, PIL image, or NumPy array.")


_IMAGE_STATE_KEYS = (
    "geometric_prompt",
    "boxes",
    "masks",
    "semantic_seg",
    "scores",
    "original_height",
    "original_width",
    "original_heights",
    "original_widths",
)


def _batch_original_sizes(state: ProcessorState) -> list[tuple[int, int]] | None:
    """Pair the per-image original heights and widths of a batch state, or return None."""
    if "original_heights" not in state:
        return None
    heights = cast(list[int], state["original_heights"])
    widths = cast(list[int], state["original_widths"])
    return list(zip(heights, widths, strict=True))


def _image_count(state: ProcessorState) -> int:
    """Return the number of images encoded in a single-image or batch state."""
    sizes = _batch_original_sizes(state)
    return 1 if sizes is None else len(sizes)


def _empty_prompt(batch_size: int) -> Prompt:
    """Return a geometry prompt with zero boxes (and points) for ``batch_size`` images."""
    return Prompt.empty(batch_size)


def _parse_prompt(
    coordinates: Sequence[float], label: bool, width: int
) -> tuple[mx.array, mx.array]:
    """Validate one normalized box or point and return ``(1, 1, width)`` values and a ``(1, 1)`` label."""
    values = np.asarray(coordinates, dtype=np.float32)
    if values.shape != (width,):
        raise ValueError(f"Prompt coordinates must have {width} values.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Prompt coordinates must be finite and normalized to [0, 1].")
    if not isinstance(cast(object, label), (bool, np.bool_)):
        raise TypeError("Prompt labels must be positive/negative booleans.")
    return mx.array(values.reshape(1, 1, width)), mx.array([[bool(label)]])


def normalize_text_prompt_key(prompt: object) -> str:
    """Return the exact whitespace-normalized cache key for one prompt."""

    if not isinstance(prompt, str):
        raise TypeError("Text prompt must be a string.")
    return " ".join(prompt.split())


def _validate_text_cache_size(value: object) -> int:
    """Require a nonnegative, non-boolean integer cache capacity."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("text_cache_size must be a non-negative integer.")
    return int(value)


class _ArrayLayer(Protocol):
    def __call__(self, value: mx.array) -> mx.array:
        """Apply an MLX layer to one tensor."""
        ...


class _MaskDecoder(Protocol):
    conv_s0: _ArrayLayer
    conv_s1: _ArrayLayer


class _InteractiveModel(Protocol):
    sam_mask_decoder: _MaskDecoder


class _InteractivePredictor(Protocol):
    model: _InteractiveModel


class _ProcessorBackbone(Protocol):
    def forward_image(self, samples: mx.array, /) -> Mapping[str, object]:
        """Encode an NCHW image batch into backbone features."""
        ...

    def forward_text(self, captions: list[str], /) -> Mapping[str, mx.array]:
        """Encode text prompts into ``language_*`` features."""
        ...


class ProcessorModel(Protocol):
    """Image-model operations required by the prompt processor."""

    @property
    def backbone(self) -> _ProcessorBackbone:
        """Return the image and text backbone."""
        ...

    @property
    def inst_interactive_predictor(self) -> object | None:
        """Return the optional interactive predictor."""
        ...

    def predict_raw(
        self,
        backbone_out: Mapping[str, object],
        find_input: FindStage,
        geometric_prompt: Prompt,
    ) -> RawPrediction:
        """Run grounding from image, text and geometry inputs."""
        ...


class Sam3Processor:
    """Stateful helper that runs the SAM 3 image model from images and prompts.

    Usage: ``set_image`` (or ``set_image_batch``) returns a state dictionary; add a
    text prompt with ``set_text_prompt`` and/or geometric prompts with
    ``add_geometric_prompt`` / ``add_point_prompt``; each prompt call runs
    ``predict`` unless ``run_grounding=False``. When replacing an image in a
    previously returned state, the previous image, prompts, and results remain
    available if preprocessing, encoding, evaluation, or interactive preparation
    fails. A successful replacement updates the same dictionary and clears its
    previous prompts and results.

    Result keys written into the state by ``predict``, for a single image: ``boxes``
    ``(K, 4)`` xyxy in original-image pixels, ``scores`` ``(K,)``, ``masks`` ``(K, 1,
    H, W)`` bool at the original size, and ``semantic_seg`` ``(1, 1, H, W)``
    logits when the model produces it. For a batch state each of these is a list
    with one entry per image. Detections are kept when the product of the query
    probability and the image presence probability is strictly above
    ``confidence_threshold``.

    Args:
        model: The SAM 3 image model.
        resolution: Square input size in pixels; a positive multiple of the patch
            size.
        confidence_threshold: Score threshold in [0, 1].
        text_cache_size: Number of text-encoder outputs to cache by normalized
            prompt; 0 disables caching.
        mask_batch_bytes: FP32 resized-mask payload per group; one mask is the
            minimum group even when it exceeds this budget.
    """

    def __init__(
        self,
        model: ProcessorModel,
        resolution: object = DEFAULT_IMAGE_RESOLUTION,
        confidence_threshold: float = 0.5,
        text_cache_size: object = DEFAULT_TEXT_CACHE_SIZE,
        mask_batch_bytes: object = 64 * 1024**2,
    ) -> None:
        """Validate the resolution, threshold and text-cache settings."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        cache_size = _validate_text_cache_size(text_cache_size)
        if (
            isinstance(mask_batch_bytes, bool)
            or not isinstance(mask_batch_bytes, int)
            or mask_batch_bytes < 1
        ):
            raise ValueError("mask_batch_bytes must be a positive integer")
        self.model = model
        self.resolution = _validate_processor_resolution(resolution)
        self.confidence_threshold = confidence_threshold
        self.mask_batch_bytes = mask_batch_bytes
        self._text_cache: BoundedLRUCache[str, Mapping[str, mx.array]] | None = (
            None if cache_size == 0 else BoundedLRUCache(cache_size)
        )

    def _text_outputs_for_prompt(self, prompt: str) -> Mapping[str, mx.array]:
        """Return the prompt's text features, reusing the bounded cache by normalized prompt."""
        cache = self._text_cache
        if cache is None:
            return self.model.backbone.forward_text([prompt])
        key = normalize_text_prompt_key(prompt)
        cached = cache.get(key)
        if cached is None:
            cached = self.model.backbone.forward_text([key])
            cache[key] = cached
        return cached

    def _patch_interactive_backbone_features(
        self, backbone_out: Mapping[str, object]
    ) -> None:
        """Apply interactive decoder projections to cached feature pyramid levels."""
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
        image_batch: mx.array,
        *,
        new_state: ProcessorState,
        state: ProcessorState | None,
    ) -> ProcessorState:
        """Encode and evaluate ``image_batch`` before replacing image-dependent state."""
        backbone_out = dict(
            self.model.backbone.forward_image(
                cast_visual_input(image_batch, model_precision(self.model))
            )
        )
        new_state["backbone_out"] = backbone_out
        evaluate_boundary(new_state)
        self._patch_interactive_backbone_features(backbone_out)
        if state is None:
            return new_state
        for key in _IMAGE_STATE_KEYS:
            state.pop(key, None)
        state.update(new_state)
        return state

    def set_image(
        self,
        image: ImageSource,
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        """Encode one 8-bit image: a path, PIL image or NumPy array.

        A passed ``state`` is reused after dropping its previous prompts and results.
        The original image size is stored for rescaling boxes and masks. The
        backbone features are evaluated before the state is replaced; with an
        interactive predictor, its feature projections are then added lazily.
        Floating NumPy arrays whose values all lie in [0, 1] are converted as
        TorchVision's ``ToDtype(torch.uint8, scale=True)`` does.

        Raises:
            ValueError: For Pillow modes ``I``, ``I;16*`` or ``F``, or NumPy
                arrays with values outside [0, 255]: convert such images to 8-bit
                first (for example, window a CT scan). Also for unsupported array
                layouts, non-finite values or input types.
            TypeError: For NumPy arrays that are neither integer nor floating.
        """
        pil_image = _as_pil_rgb_image(image)
        return self._replace_image_state(
            _preprocess_rgb(pil_image, self.resolution)[None],
            new_state={
                "original_height": pil_image.height,
                "original_width": pil_image.width,
            },
            state=state,
        )

    def set_image_batch(
        self,
        images: Sequence[ImageSource],
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        """Encode a nonempty list of images, each accepted as in ``set_image``.

        Each image's original size is retained for its own output coordinates.
        A supplied state is reused after its previous prompts and results are cleared.
        """
        if isinstance(images, str) or len(images) == 0:
            raise ValueError(
                "Images must be a nonempty list of filesystem paths, PIL images, "
                "or NumPy arrays."
            )
        pil_images = [_as_pil_rgb_image(image) for image in images]
        return self._replace_image_state(
            mx.stack(
                [_preprocess_rgb(image, self.resolution) for image in pil_images],
                axis=0,
            ),
            new_state={
                "original_heights": [image.height for image in pil_images],
                "original_widths": [image.width for image in pil_images],
            },
            state=state,
        )

    def set_text_prompt(
        self,
        prompt: str,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Set the text prompt for the image(s) in ``state`` and, by default, run ``predict``.

        Raises ``ValueError`` if no image has been set.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        language = self._text_outputs_for_prompt(prompt)
        cast(ProcessorState, state["backbone_out"]).update(language)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _empty_prompt(_image_count(state))
        return self.predict(state) if run_grounding else state

    def _add_geometry(
        self,
        kind: Literal["box", "point"],
        coordinates: Sequence[float],
        label: bool,
        state: ProcessorState,
        run_grounding: bool,
    ) -> ProcessorState:
        """Validate one box or point, append it for every image and optionally run ``predict``."""
        if "backbone_out" not in state:
            method = "add_geometric_prompt" if kind == "box" else "add_point_prompt"
            raise ValueError(f"You must call set_image before {method}")
        width = 4 if kind == "box" else 2
        values, labels = _parse_prompt(coordinates, label, width)
        batch_size = _image_count(state)
        backbone_out = cast(ProcessorState, state["backbone_out"])
        if "language_features" not in backbone_out:
            # Geometry-only prompts use the text prompt "visual".
            backbone_out.update(self._text_outputs_for_prompt("visual"))
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _empty_prompt(batch_size)
        geometric_prompt = cast(Prompt, state["geometric_prompt"])
        values = mx.broadcast_to(values, (1, batch_size, width))
        labels = mx.broadcast_to(labels, (1, batch_size))
        if kind == "box":
            geometric_prompt.append_boxes(values, labels)
        else:
            geometric_prompt.append_points(values, labels)
        return self.predict(state) if run_grounding else state

    def add_geometric_prompt(
        self,
        box: Sequence[float],
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Add a box prompt and, by default, run ``predict``.

        The image needs to be set, but not necessarily the text prompt (the text
        prompt "visual" is used when none was given).
        The box is ``[center_x, center_y, width, height]``, with finite coordinates
        normalized to ``[0, 1]``. The label is a boolean: True for a positive box,
        False for a negative box. For a batch state, the same prompt is appended
        to every image. Invalid prompts are rejected before the state is changed.
        """
        return self._add_geometry("box", box, label, state, run_grounding)

    def add_point_prompt(
        self,
        point: Sequence[float],
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Add a point prompt and, by default, run ``predict``.

        The point is finite normalized ``[x, y]`` image coordinates in ``[0, 1]``;
        the label is a boolean. As with boxes, the text prompt "visual" is used
        when none was given. For a batch state, the same prompt is appended to
        every image. Invalid prompts are rejected before the state is changed.
        """
        return self._add_geometry("point", point, label, state, run_grounding)

    def predict_raw(self, state: ProcessorState) -> RawPrediction:
        """Fixed-shape device arrays before filtering or original-res upsample.

        Masks and boxes are on the model grid / normalized coordinates and every
        query is present; nothing is thresholded.

        Raises:
            ValueError: If no image or no prompt has been set.
        """
        if "backbone_out" not in state or "geometric_prompt" not in state:
            raise ValueError("Set an image and a text, box or point prompt before predicting.")
        batch_size = _image_count(state)
        find_stage = FindStage(
            img_ids=mx.arange(batch_size, dtype=mx.int64),
            text_ids=mx.zeros((batch_size,), dtype=mx.int64),
        )
        return self.model.predict_raw(
            cast(ProcessorState, state["backbone_out"]),
            find_stage,
            cast(Prompt, state["geometric_prompt"]),
        )

    def predict(self, state: ProcessorState) -> ProcessorState:
        """Filter and write MLX detections into ``state`` at original image size.

        ``boxes`` are pixel XYXY coordinates, ``scores`` strictly exceed the
        confidence threshold, and ``masks`` are Boolean arrays of shape
        ``(N, 1, height, width)``. Batch states hold one array per image for each
        field, preserving each image's dimensions.
        """
        outputs = self.predict_raw(state)
        batch_sizes = _batch_original_sizes(state)
        sizes = batch_sizes or [
            (cast(int, state["original_height"]), cast(int, state["original_width"]))
        ]
        scores = _presence_weighted_scores(outputs)
        if scores.shape[0] != len(sizes):
            raise ValueError(
                f"Grounding returned {scores.shape[0]} images for a state with "
                f"{len(sizes)} image sizes."
            )
        semantic_seg = outputs["semantic_seg"]
        results: dict[str, list[mx.array]] = {
            "boxes": [], "masks": [], "scores": [], "semantic_seg": []
        }
        for index, (img_h, img_w) in enumerate(sizes):
            keep = _score_keep_indices(scores[index], self.confidence_threshold)
            masks = _upsample_boolean_masks(
                outputs["pred_masks"][index][keep], img_h, img_w, self.mask_batch_bytes
            )
            results["boxes"].append(
                _boxes_to_original_xyxy(outputs["pred_boxes"][index][keep], img_h, img_w)
            )
            results["masks"].append(masks)
            results["scores"].append(scores[index][keep])
            results["semantic_seg"].append(
                _upsample_spatial(semantic_seg[index : index + 1], img_h, img_w)
            )
        for key, values in results.items():
            state[key] = values if batch_sizes is not None else values[0]
        return state
