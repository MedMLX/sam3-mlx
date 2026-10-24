"""Multiplex video tracking sessions: prompting, propagation and object removal.

``Sam3MultiplexTracking`` turns ``Sam3MultiplexBase`` into a session API. ``init_state``
loads a video, ``add_prompt`` (text or normalized boxes) clears earlier prompts and
detects on one frame, ``propagate_in_video`` yields ``(frame_idx, PostprocessedOutput)``
frame by frame, and ``remove_object`` drops an identity. ``Sam3MultiplexTrackingProd``
keeps the propagation generator state across calls, and
``Sam3MultiplexTrackingWithInteractivity`` adds point prompts, per-object refinement and
propagation types derived from an action history.

Public outputs are host NumPy arrays at the original video resolution; internally
masks are MLX boolean ``(1, H, W)`` arrays per object id. Distributed execution is not
supported (see ``Sam3MultiplexBase``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import fields, is_dataclass
from os import PathLike

from PIL import Image
from typing import (
    Literal,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    overload,
)

import mlx.core as mx
import numpy as np
from numpy.typing import DTypeLike, NDArray

from sam3_mlx.mlx_runtime import mlx_array
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import (
    BatchedDatapoint,
    BatchedPointer,
    FindStage,
    MyTensor,
    interpolate,
)
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.io_utils import load_resource_as_video_frames
from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    raise_unsupported_multiplex_runtime,
)
from sam3_mlx.model.sam3_multiplex_base import (
    FeatureCache,
    MultiplexBaseOptions,
    MaskletConfirmationStatus,
    Sam3MultiplexBase,
    TrackerObjectId,
    TrackerState,
    require_propagate_call,
    text_outputs_for_batch,
)
from sam3_mlx.model.multiplex_inference_state import (
    FrameDimensions, MultiplexInferenceState, TrackingAction,
)
from sam3_mlx.model.multiplex_metadata import (
    TrackerMetadata, InitializedTrackerMetadata, MultiplexTrackerMetadata, GpuMetadata, HotstartGpuMetadata,
    Rank0TrackerMetadata, InitializedRank0Metadata, ConfirmedRank0Metadata, frame_scores_for_key,
)
from sam3_mlx.model.sam3_tracking_predictor import ArrayInput
from sam3_mlx.model.sam3_tracker_base import NO_OBJ_SCORE
from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3_mlx.model.sam3_video_base import EvaluatorPredictions
from sam3_mlx.model import box_ops
from sam3_mlx.mlx_runtime import is_mlx_array as _is_mlx_array
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.perflib import masks_ops


DUMMY_OUTPUT: Literal["DUMMY_OUTPUT"] = "DUMMY_OUTPUT"
NumpyArray: TypeAlias = NDArray[np.generic]
BoolArray: TypeAlias = NDArray[np.bool_]
FloatArray: TypeAlias = NDArray[np.float32]
TrackerArray: TypeAlias = mx.array | NumpyArray
BoxCoordinatesInput: TypeAlias = (
    TrackerArray | list[Sequence[float]] | tuple[Sequence[float], ...]
)
BoxLabelsInput: TypeAlias = TrackerArray | list[int] | tuple[int, ...]
ObjectIdCollection: TypeAlias = Sequence[int] | set[int]
RecursiveValue = TypeVar("RecursiveValue")
PointCoordinatesInput: TypeAlias = TrackerArray | Sequence[object]


ResourceInput: TypeAlias = str | PathLike[str] | Sequence[Image.Image]


class TrackingOptions(MultiplexBaseOptions):
    """Constructor options of ``Sam3MultiplexTracking``: the base options plus the input image size and normalization, the compile flag and the post-processing batch size."""

    image_size: NotRequired[int]
    image_mean: NotRequired[tuple[float, float, float]]
    image_std: NotRequired[tuple[float, float, float]]
    compile_model: NotRequired[bool]
    postprocess_batch_size: NotRequired[int]


class TrackingBackboneOutput(TypedDict):
    """Detector inputs of a session: the ``img_batch_all_stages`` frames plus the text ``language_features`` and ``language_mask``."""

    img_batch_all_stages: MyTensor
    language_features: TrackerArray
    language_mask: TrackerArray


class _SliceableValue(Protocol):
    def __getitem__(self, key: slice) -> _SliceableValue: ...


type StoredArray = TrackerArray | _SliceableValue


class _MuxTransform(Protocol):
    def mux(self, value: StoredArray) -> StoredArray: ...


class _DemuxTransform(Protocol):
    def demux(self, value: StoredArray) -> StoredArray: ...


class SingletonFrameOutput(TypedDict, total=False):
    """Fields retained when one object is extracted from a packed tracker."""

    pred_masks: StoredArray | None
    object_score_logits: StoredArray
    image_features: StoredArray | None
    image_pos_enc: StoredArray | None
    local_obj_id_to_idx: dict[int, int]
    maskmem_features: StoredArray | None
    maskmem_pos_enc: list[StoredArray | None] | None
    obj_ptr: StoredArray | None
    conditioning_objects: set[int]


type FrameStorage = Literal["cond_frame_outputs", "non_cond_frame_outputs"]
type PerObjectFrameOutputs = dict[int, SingletonFrameOutput]


class RawFrameOutput(TypedDict):
    """Per-frame result before post-processing.

    ``obj_id_to_mask`` maps object ids to bool ``(1, H, W)`` masks at the video size;
    ``obj_id_to_score`` and ``obj_id_to_sam2_score`` hold detection and tracker
    probabilities. ``removed_obj_ids``, ``suppressed_obj_ids`` (this frame) and
    ``unconfirmed_obj_ids`` are hidden from the public output.
    """

    obj_id_to_mask: dict[int, object]
    obj_id_to_score: Mapping[int, object]
    obj_id_to_sam2_score: NotRequired[Mapping[int, object]]
    removed_obj_ids: NotRequired[ObjectIdCollection]
    suppressed_obj_ids: NotRequired[ObjectIdCollection]
    frame_stats: NotRequired[object]
    unconfirmed_obj_ids: NotRequired[ObjectIdCollection]


class PostprocessedOutput(TypedDict):
    """Public per-frame output as host NumPy arrays for ``K`` visible objects.

    ``out_obj_ids`` ``(K,)`` int64 ascending, ``out_probs`` ``(K,)`` float32 detection
    scores, ``out_boxes_xywh`` ``(K, 4)`` normalized ``(x0, y0, w, h)``,
    ``out_binary_masks`` ``(K, H, W)`` bool at the video size and, only when
    ``running_in_prod``, ``out_centers`` ``(K, 2)`` normalized mask centers ``(x, y)``.
    """

    out_obj_ids: NDArray[np.int64]
    out_probs: FloatArray
    out_boxes_xywh: FloatArray
    out_binary_masks: BoolArray
    frame_stats: object
    out_centers: NotRequired[FloatArray | None]


PostprocessBatchItem: TypeAlias = tuple[
    RawFrameOutput,
    ObjectIdCollection | None,
    ObjectIdCollection | None,
    ObjectIdCollection | None,
]
PendingPropagationOutput: TypeAlias = tuple[int, RawFrameOutput, set[int]]
PropagationOutput: TypeAlias = PostprocessedOutput | Literal["DUMMY_OUTPUT"]
PropagationYield: TypeAlias = tuple[int, PropagationOutput]


class GeneratorState(TypedDict):
    """Propagation-generator state kept between calls: frames waiting out the hotstart delay, ids removed meanwhile, unconfirmed ids per frame and outputs awaiting post-processing."""

    hotstart_buffer: list[tuple[int, RawFrameOutput]]
    hotstart_removed_obj_ids: set[int]
    unconfirmed_obj_ids_per_frame: dict[int, ObjectIdCollection]
    postprocess_yield_list: list[PendingPropagationOutput]


class _MlxUnaryArrayCall(Protocol):
    def __call__(self, value: mx.array) -> object: ...


class _MlxMasksToBoxesCall(Protocol):
    def __call__(self, masks: mx.array, obj_ids: list[int]) -> object: ...


class _NonOverlappingConstraintsCall(Protocol):
    def __call__(
        self,
        masks: mx.array,
        scores: mx.array,
        *,
        background_value: int,
    ) -> object: ...


class _NoArgCall(Protocol):
    def __call__(self) -> object: ...


class _IterableCall(Protocol):
    def __call__(self, values: object) -> object: ...


class _KeywordCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class _ClearPointsCall(Protocol):
    def __call__(
        self,
        inference_state: TrackerState,
        frame_idx: int,
        obj_id: int,
        need_output: bool,
    ) -> object: ...


class _AddMaskCall(Protocol):
    def __call__(
        self,
        inference_state: TrackerState,
        frame_idx: int,
        obj_id: int,
        mask: object,
    ) -> object: ...


class _PreflightCall(Protocol):
    def __call__(
        self,
        inference_state: TrackerState,
        run_mem_encoder: bool = True,
    ) -> object: ...


class _ControllerStateCall(Protocol):
    def __call__(
        self,
        *,
        num_valid_entries: int,
        device: object,
        dtype: mx.Dtype,
        random: bool,
        object_ids: list[int],
    ) -> MultiplexState[int]: ...


class _RemoveObjectCall(Protocol):
    def __call__(
        self,
        inference_state: TrackerState,
        obj_id: int,
        *,
        strict: bool,
        need_output: bool,
    ) -> tuple[Sequence[TrackerObjectId], object]: ...


class _MaterializedVideoFrames(Protocol):
    images: object

    def __len__(self) -> int: ...


class _ShapedValue(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class _SizedValue(Protocol):
    def __len__(self) -> int: ...


def _runtime_type(value: object) -> object:
    return type(value)


def _require_materialized_video_frames(value: object) -> _MaterializedVideoFrames:
    length_method: object = getattr(value, "__len__", None)
    if not callable(length_method) or not hasattr(value, "images"):
        raise TypeError("loaded video frames must expose length and images")
    return cast(_MaterializedVideoFrames, value)


def _integer_value(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return value.item()
    raise TypeError(f"{name} must be an integer")


def _mapping_view(value: object, *, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[object, object], value)


def _require_feature_cache(value: object) -> FeatureCache:
    if not isinstance(value, dict):
        raise TypeError(
            "feature_cache must be a dictionary with string or integer keys"
        )
    mapping = cast(dict[object, object], value)
    if not all(
        isinstance(key, str) or (not isinstance(key, bool) and isinstance(key, int))
        for key in mapping
    ):
        raise TypeError(
            "feature_cache must be a dictionary with string or integer keys"
        )
    return cast(FeatureCache, value)


def _require_tracker_state(value: object) -> TrackerState:
    if not isinstance(value, dict):
        raise TypeError("tracker state must be a dictionary with string keys")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError("tracker state must be a dictionary with string keys")
    return cast(TrackerState, value)


def _require_tracker_state_list(value: object) -> list[TrackerState]:
    if not isinstance(value, list):
        raise TypeError("sam2_inference_states must be a list")
    for state in cast(list[object], value):
        _require_tracker_state(state)
    return cast(list[TrackerState], value)


def _object_id_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("tracker state obj_ids must be a sequence")
    return cast(Sequence[object], value)


def _object_id_collection(value: object) -> Sequence[object] | set[object]:
    if isinstance(value, set):
        return cast(set[object], value)
    return _object_id_sequence(value)


def _optional_text(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError("text_str must be a string when provided.")
    return value


def _first_axis_size(value: object, *, name: str) -> int:
    shape = getattr(value, "shape", None)
    if not isinstance(shape, Sequence) or isinstance(shape, str) or not shape:
        raise TypeError(f"{name} must expose a non-empty shape")
    shape_values = cast(Sequence[object], shape)
    return _integer_value(shape_values[0], name=f"{name} first dimension")


def _tracker_prompt_result(
    value: object,
) -> tuple[int, list[int], object, TrackerArray | None]:
    if not isinstance(value, tuple):
        raise TypeError("tracker prompt methods must return a four-item tuple")
    tuple_value = cast(tuple[object, ...], value)
    if len(tuple_value) != 4:
        raise TypeError("tracker prompt methods must return a four-item tuple")
    frame_idx_raw, obj_ids_raw, low_res_masks, video_res_masks = tuple_value
    frame_idx = _integer_value(frame_idx_raw, name="tracker prompt frame_idx")
    obj_ids = [
        _integer_value(obj_id, name="tracker prompt obj_ids")
        for obj_id in _object_id_sequence(obj_ids_raw)
    ]
    if video_res_masks is not None and not (
        _is_mlx_array(video_res_masks) or isinstance(video_res_masks, np.ndarray)
    ):
        raise TypeError("tracker prompt video_res_masks must be an array or None")
    return frame_idx, obj_ids, low_res_masks, cast(TrackerArray | None, video_res_masks)


def _tracker_object_ids(value: object) -> list[TrackerObjectId]:
    if _is_mlx_array(value):
        raw_ids = cast(list[object], _array_to_numpy(value).reshape(-1).tolist())
    elif isinstance(value, np.ndarray):
        raw_ids = cast(list[object], value.reshape(-1).tolist())
    elif isinstance(value, Sequence) and not isinstance(value, str):
        raw_ids = list(cast(Sequence[object], value))
    else:
        raise TypeError("tracker propagation obj_ids must be a sequence or array")
    object_ids: list[TrackerObjectId] = []
    for obj_id in raw_ids:
        if isinstance(obj_id, str):
            object_ids.append(obj_id)
        else:
            object_ids.append(_integer_value(obj_id, name="tracker propagation obj_id"))
    return object_ids


def _int_keyed_object_dict(value: object, *, name: str) -> dict[int, object]:
    source = _mapping_view(value, name=name)
    return {
        _integer_value(key, name=f"{name} keys"): item for key, item in source.items()
    }


def _call_mlx_unary_array(target: object, name: str, value: mx.array) -> mx.array:
    method: object = getattr(target, name, None)
    if not callable(method):
        raise TypeError(f"{name} must be callable")
    result: object = cast(_MlxUnaryArrayCall, method)(value)
    if not _is_mlx_array(result):
        raise TypeError(f"{name} must return an MLX array for MLX inputs")
    return result


def _masks_to_boxes(masks: mx.array, obj_ids: list[int]) -> mx.array:
    method: object = getattr(masks_ops, "masks_to_boxes", None)
    if not callable(method):
        raise TypeError("masks_to_boxes must be callable")
    result: object = cast(_MlxMasksToBoxesCall, method)(masks, obj_ids)
    if not _is_mlx_array(result):
        raise TypeError("masks_to_boxes must return an MLX array for MLX inputs")
    return result


def _apply_non_overlapping_constraints(
    tracker: object,
    masks: mx.array,
    scores: mx.array,
) -> mx.array:
    method: object = getattr(
        tracker,
        "_apply_object_wise_non_overlapping_constraints",
        None,
    )
    if not callable(method):
        raise TypeError(
            "_apply_object_wise_non_overlapping_constraints must be callable"
        )
    result: object = cast(_NonOverlappingConstraintsCall, method)(
        masks,
        scores,
        background_value=0,
    )
    if not _is_mlx_array(result):
        raise TypeError(
            "_apply_object_wise_non_overlapping_constraints must return an MLX array"
        )
    return result


def require_postprocessed_output(
    output: PropagationOutput,
) -> PostprocessedOutput:
    """Return ``output``, or raise ``RuntimeError`` if it is the dummy output of a non-zero rank."""
    if output == DUMMY_OUTPUT:
        raise RuntimeError("rank-zero propagation returned a dummy output")
    return output


def _mlx_to(data: mx.array, *args: object, **kwargs: object) -> mx.array:
    """Apply an MLX dtype target while rejecting unsupported devices."""
    device = kwargs.pop("device", None)
    dtype = kwargs.pop("dtype", None)
    if len(args) > 1:
        raise TypeError("recursive_to accepts at most one positional target.")
    if args:
        arg = args[0]
        if arg is None or isinstance(arg, str):
            device = arg
        else:
            dtype = arg
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported recursive_to kwargs for MLX array: {names}.")
    if device not in (None, "mlx"):
        raise_unsupported(
            f"sam3_mlx.model.sam3_multiplex_tracking.recursive_to(device={device!r})",
            reason="unsupported-device",
            detail="recursive_to only supports the explicit MLX device.",
            alternative="device='mlx' or device=None",
        )
    return data.astype(cast(mx.Dtype, dtype)) if dtype is not None else data


def _array_to_numpy(value: object, *, dtype: DTypeLike | None = None) -> NumpyArray:
    return to_numpy(value, dtype=dtype, copy=False)


HOTSTART_GPU_METADATA_KEYS = (
    "obj_first_frame",
    "consecutive_unmatch_count",
    "trk_keep_alive",
    "removed_mask",
    "overlap_pair_counts",
    "last_occluded_tensor",
)


@overload
def _copy_first_axis_slice(value: mx.array, start: int, stop: int) -> mx.array: ...


@overload
def _copy_first_axis_slice[ScalarT: np.generic](
    value: NDArray[ScalarT], start: int, stop: int
) -> NDArray[ScalarT]: ...


@overload
def _copy_first_axis_slice(
    value: _SliceableValue, start: int, stop: int
) -> _SliceableValue: ...


def _copy_first_axis_slice(value: StoredArray, start: int, stop: int) -> StoredArray:
    """Copy a tensor slice while preserving its array or clone/copy backend."""
    sliced = value[start:stop]
    if _is_mlx_array(sliced):
        return mx.array(sliced)
    if isinstance(sliced, np.ndarray):
        return sliced.copy()
    clone: object = getattr(sliced, "clone", None)
    if callable(clone):
        return cast(StoredArray, cast(_NoArgCall, clone)())
    copy: object = getattr(sliced, "copy", None)
    return cast(StoredArray, cast(_NoArgCall, copy)() if callable(copy) else sliced)


def _demux_and_slice_first_axis(
    value: StoredArray | None, multiplex_state: object | None, obj_idx: int
) -> StoredArray | None:
    """Copy object ``obj_idx`` from a per-object array, demultiplexing first when a multiplex state is given.

    Returns None if the value is missing or cannot be demultiplexed.
    """
    if value is None:
        return None
    if multiplex_state is not None:
        try:
            value = cast(_DemuxTransform, multiplex_state).demux(value)
        except (AssertionError, IndexError, ValueError):
            return None
    return _copy_first_axis_slice(value, obj_idx, obj_idx + 1)


def _score_to_float(value: object) -> float:
    array = _array_to_numpy(value, dtype=np.float32)
    return float(array.reshape(()))


def _mask_to_mlx_bool(mask: object) -> mx.array:
    """Convert one object's boolean mask to shape (1, H, W)."""
    mask = mask if _is_mlx_array(mask) else mx.array(cast(ArrayInput, mask))
    if mask.dtype != mx.bool_:
        raise TypeError("Sam3MultiplexTracking postprocess expects boolean masks.")
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "Each obj_id_to_mask entry must have shape (1, H, W) or (H, W), "
            f"got {mask.shape}."
        )
    return mask


def _empty_postprocessed_output(
    height: int,
    width: int,
    frame_stats: object,
    *,
    include_prod_outputs: bool,
) -> PostprocessedOutput:
    output = PostprocessedOutput(
        out_obj_ids=np.zeros(0, dtype=np.int64),
        out_probs=np.zeros(0, dtype=np.float32),
        out_boxes_xywh=np.zeros((0, 4), dtype=np.float32),
        out_binary_masks=np.zeros((0, height, width), dtype=bool),
        frame_stats=frame_stats,
    )
    if include_prod_outputs:
        output["out_centers"] = np.zeros((0, 2), dtype=np.float32)
    return output


def _hidden_obj_ids(
    removed_obj_ids: Sequence[int] | set[int] | None,
    suppressed_obj_ids: Sequence[int] | set[int] | None,
    unconfirmed_obj_ids: Sequence[int] | set[int] | None,
) -> set[int]:
    """Union of removed, suppressed and unconfirmed ids (the ids hidden from public output)."""
    hidden: set[int] = set()
    if suppressed_obj_ids is not None:
        hidden.update(suppressed_obj_ids)
    if removed_obj_ids is not None:
        hidden.update(removed_obj_ids)
    if unconfirmed_obj_ids is not None:
        hidden.update(unconfirmed_obj_ids)
    return hidden


def _mask_centers(binary_masks: TrackerArray) -> FloatArray:
    """Centers of mass of ``(N, H, W)`` masks as float32 ``(N, 2)`` ``(x, y)``, each coordinate divided by the width or height (pixel indices are used, not pixel centers)."""
    binary_masks = (
        binary_masks if _is_mlx_array(binary_masks) else mlx_array(binary_masks)
    )
    if binary_masks.shape[0] == 0:
        return cast(FloatArray, np.zeros((0, 2), dtype=np.float32))
    height, width = binary_masks.shape[-2:]
    masks_f32 = binary_masks.astype(mx.float32)
    y = mx.reshape(mx.arange(height, dtype=mx.float32), (1, height, 1))
    x = mx.reshape(mx.arange(width, dtype=mx.float32), (1, 1, width))
    mass = mx.maximum(mx.sum(masks_f32, axis=(1, 2)), mx.array(1e-6, dtype=mx.float32))
    center_x = mx.sum(masks_f32 * x, axis=(1, 2)) / mass / width
    center_y = mx.sum(masks_f32 * y, axis=(1, 2)) / mass / height
    return cast(
        FloatArray,
        _array_to_numpy(mx.stack([center_x, center_y], axis=1), dtype=np.float32),
    )


def _point_count(points: TrackerArray | Sequence[object]) -> int:
    """Number of points in a prompt: the second-to-last axis for arrays with 2 or 3 coordinates, else the first axis; 0 for scalars."""
    if _is_mlx_array(points):
        shape = points.shape
        if len(shape) == 0:
            return 0
        if len(shape) >= 2 and shape[-1] in (2, 3):
            return int(shape[-2])
        return int(shape[0])
    return len(points)


def recursive_to(
    data: RecursiveValue, *args: object, **kwargs: object
) -> RecursiveValue:
    """Apply an MLX dtype to every MLX array inside nested containers.

    Mappings, tuples, sequences and dataclasses are rebuilt with converted values;
    NumPy arrays and other values are returned unchanged. The target is one positional
    dtype or ``dtype=``; ``device`` may only be None or ``'mlx'``.

    Raises:
        TypeError: For several positional targets, unsupported keyword arguments or
            containers whose type cannot be rebuilt.
        Sam3MlxUnsupportedError: For any other device.
    """
    if _is_mlx_array(data):
        return cast(RecursiveValue, _mlx_to(data, *args, **kwargs))
    if isinstance(data, np.ndarray):
        return cast(RecursiveValue, cast(NumpyArray, data))
    if isinstance(data, Mapping):
        source = cast(Mapping[object, object], data)
        constructor = _runtime_type(source)
        if not callable(constructor):
            raise TypeError("recursive_to mapping types must be callable")
        ret_value = cast(_NoArgCall, constructor)()
        if not isinstance(ret_value, MutableMapping):
            raise TypeError("recursive_to mapping constructors must return mappings")
        ret = cast(MutableMapping[object, object], ret_value)
        for key, value in source.items():
            ret[key] = recursive_to(value, *args, **kwargs)
        return cast(RecursiveValue, ret)
    if isinstance(data, tuple):
        values = cast(tuple[object, ...], data)
        return cast(
            RecursiveValue,
            tuple(recursive_to(value, *args, **kwargs) for value in values),
        )
    if isinstance(data, Sequence) and not isinstance(data, str):
        values = cast(Sequence[object], data)
        constructor = _runtime_type(values)
        if not callable(constructor):
            raise TypeError("recursive_to sequence types must be callable")
        constructed = cast(_IterableCall, constructor)(
            recursive_to(value, *args, **kwargs) for value in values
        )
        if not isinstance(constructed, Sequence) or isinstance(constructed, str):
            raise TypeError("recursive_to sequence constructors must return sequences")
        return cast(RecursiveValue, constructed)
    if is_dataclass(data):
        ret_cls = cast(_KeywordCall, type(data))
        ret_fields = {
            field.name: recursive_to(getattr(data, field.name), *args, **kwargs)
            for field in fields(data)
        }
        return cast(RecursiveValue, ret_cls(**ret_fields))
    return data


class Sam3MultiplexTracking(Sam3MultiplexBase):
    """Multiplex video tracking session API on top of ``Sam3MultiplexBase``.

    A session is a ``MultiplexInferenceState`` created by ``init_state``. ``add_prompt``
    clears earlier prompts and tracker state and detects on one frame;
    ``propagate_in_video`` yields ``(frame_idx, PostprocessedOutput)``. When
    ``hotstart_delay > 0``, up to that many frame outputs are held back so objects
    removed during the hotstart window are already hidden when a frame is yielded.

    Args:
        image_size: Side length in pixels of the square frames fed to the models.
        image_mean: Per-channel RGB mean used to normalize frames.
        image_std: Per-channel RGB standard deviation used to normalize frames.
        compile_model: Forwarded to the detector's ``compile_model`` when present;
            ``mx.compile`` is not used here.
        postprocess_batch_size: Number of frames post-processed together (>= 1).
        **kwargs: ``MultiplexBaseOptions``.

    Raises:
        ValueError: If ``postprocess_batch_size`` is below 1.
    """

    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        image_size: int = 1008,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        compile_model: bool = False,
        postprocess_batch_size: int = 1,
        **kwargs: Unpack[MultiplexBaseOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        if postprocess_batch_size < 1:
            raise ValueError("postprocess_batch_size must be >= 1.")
        self.compile_model = compile_model
        if hasattr(self.detector, "compile_model"):
            setattr(self.detector, "compile_model", compile_model)
        self.postprocess_batch_size = postprocess_batch_size
        self._compiled_for_propagation = False

    def _construct_initial_input_batch(
        self,
        inference_state: dict[str, object],
        images: _MaterializedVideoFrames,
    ) -> None:
        """Create the session's batched detector input and its empty per-frame prompt bookkeeping.

        Raises:
            ValueError: If the loaded frames have no normalized MLX ``images``.
            TypeError: If ``images`` is not an MLX array.
        """
        num_frames = len(images)
        image_tensors_raw = images.images
        if image_tensors_raw is None:
            raise ValueError("Loaded video frames must expose normalized MLX images.")
        if not _is_mlx_array(image_tensors_raw):
            raise TypeError("Loaded video frame images must be an MLX array.")
        image_tensors = image_tensors_raw

        find_text_batch = ["<text placeholder>", "visual", "geometric"]
        dummy_ptrs = BatchedPointer(
            stage_ids=mx.array([], dtype=mx.int64),
            query_ids=mx.array([], dtype=mx.int64),
            object_ids=mx.array([], dtype=mx.int64),
            ptr_mask=mx.array([], dtype=mx.bool_),
            ptr_types=mx.array([], dtype=mx.int64),
        )
        stages = [
            FindStage(
                img_ids=mx.array([stage_id], dtype=mx.int64),
                img_ids_np=np.array([stage_id], dtype=np.int64),
                text_ids=mx.array([self.TEXT_ID_FOR_TEXT], dtype=mx.int64),
                input_boxes=mx.zeros((258, 1), dtype=mx.float32),
                input_boxes_before_embed=mx.zeros((0, 1, 4), dtype=mx.float32),
                input_boxes_mask=mx.zeros((1, 0), dtype=mx.bool_),
                input_boxes_label=mx.zeros((0, 1), dtype=mx.int64),
                input_points=mx.zeros((1, 0, 257), dtype=mx.float32),
                input_points_before_embed=mx.zeros((1, 0, 3), dtype=mx.float32),
                input_points_mask=mx.zeros((1, 0), dtype=mx.bool_),
                ptrs=dummy_ptrs,
                ptrs_seg=dummy_ptrs,
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]

        input_batch = BatchedDatapoint(
            img_batch=image_tensors,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            find_metadatas=[None] * num_frames,
            get_queries=None,
        )
        inference_state["input_batch"] = input_batch
        constants = cast(dict[str, Prompt], inference_state["constants"])
        constants["empty_geometric_prompt"] = Prompt(
            box_embeddings=mx.zeros((0, 1, 4), dtype=mx.float32),
            box_mask=mx.zeros((1, 0), dtype=mx.bool_),
            box_labels=mx.zeros((0, 1), dtype=mx.int64),
            point_embeddings=mx.zeros((0, 1, 2), dtype=mx.float32),
            point_mask=mx.zeros((1, 0), dtype=mx.bool_),
            point_labels=mx.zeros((0, 1), dtype=mx.int64),
        )
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    @staticmethod
    def _clear_find_input_geometry(find_input: FindStage) -> None:
        """Reset the frame's box and point geometry to empty."""
        find_input.input_boxes_before_embed = mx.zeros((0, 1, 4), dtype=mx.float32)
        find_input.input_boxes_mask = mx.zeros((1, 0), dtype=mx.bool_)
        find_input.input_boxes_label = mx.zeros((0, 1), dtype=mx.int64)
        find_input.input_points_before_embed = mx.zeros((1, 0, 3), dtype=mx.float32)
        find_input.input_points_mask = mx.zeros((1, 0), dtype=mx.bool_)

    @staticmethod
    def _set_find_input_box_geometry(
        find_input: FindStage,
        boxes_cxcywh: mx.array,
        box_labels: mx.array,
    ) -> None:
        """Store box geometry in the detector's sequence-first layout."""
        num_boxes = int(boxes_cxcywh.shape[0])
        find_input.input_boxes_before_embed = mx.reshape(
            boxes_cxcywh, (num_boxes, 1, 4)
        )
        find_input.input_boxes_mask = mx.zeros((1, num_boxes), dtype=mx.bool_)
        find_input.input_boxes_label = mx.reshape(box_labels, (num_boxes, 1))

    def _get_visual_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        boxes_cxcywh: BoxCoordinatesInput,
        box_labels: BoxLabelsInput,
    ) -> tuple[mx.array, mx.array, Prompt]:
        """Build an MLX geometry prompt from center-format boxes and labels."""
        del frame_idx
        boxes_cxcywh = (
            mlx_array(boxes_cxcywh, dtype=mx.float32)
            if isinstance(boxes_cxcywh, np.ndarray)
            else mx.array(boxes_cxcywh, dtype=mx.float32)
        )
        box_labels = (
            mlx_array(box_labels, dtype=mx.int64)
            if isinstance(box_labels, np.ndarray)
            else mx.array(box_labels, dtype=mx.int64)
        )
        batch_size = 1
        geometric_prompt = Prompt(
            box_embeddings=mx.zeros((0, batch_size, 4), dtype=mx.float32),
            box_mask=mx.zeros((batch_size, 0), dtype=mx.bool_),
            point_embeddings=None,
            point_mask=None,
        )
        geometric_prompt.append_boxes(
            boxes=mx.reshape(boxes_cxcywh, (-1, batch_size, 4)),
            labels=mx.reshape(box_labels, (-1, batch_size)),
        )
        return boxes_cxcywh, box_labels, geometric_prompt

    def _cache_frame_outputs[MaskValue](
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        obj_id_to_mask: dict[int, MaskValue],
        suppressed_obj_ids: ObjectIdCollection | None = None,
        removed_obj_ids: ObjectIdCollection | None = None,
        unconfirmed_obj_ids: ObjectIdCollection | None = None,
    ) -> None:
        """Cache the frame's masks for later fetches, leaving out suppressed, removed and unconfirmed objects."""
        if "cached_frame_outputs" not in inference_state:
            inference_state["cached_frame_outputs"] = {}
        filtered_obj_id_to_mask = obj_id_to_mask.copy()
        objects_to_exclude: set[int] = set()
        if suppressed_obj_ids is not None:
            objects_to_exclude.update(suppressed_obj_ids)
        if removed_obj_ids is not None:
            objects_to_exclude.update(removed_obj_ids)
        if unconfirmed_obj_ids is not None:
            objects_to_exclude.update(unconfirmed_obj_ids)
        for obj_id in objects_to_exclude:
            filtered_obj_id_to_mask.pop(obj_id, None)
        inference_state["cached_frame_outputs"][frame_idx] = cast(dict[int, object], filtered_obj_id_to_mask)

    def _build_sam2_output(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        refined_obj_id_to_mask: Mapping[int, object] | None = None,
    ) -> dict[int, object]:
        """Cached masks of ``frame_idx`` (empty if none) overridden by ``refined_obj_id_to_mask``.

        Raises:
            ValueError: If a refined mask is None.
        """
        cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
        if frame_idx not in cached_frame_outputs:
            return {}

        obj_id_to_mask = cached_frame_outputs[frame_idx].copy()
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                if refined_mask is None:
                    raise ValueError(
                        f"Refined mask data must be provided for obj_id {obj_id}."
                    )
                obj_id_to_mask[obj_id] = refined_mask
        return obj_id_to_mask

    def _postprocess_output(
        self,
        inference_state: FrameDimensions,
        out: RawFrameOutput,
        removed_obj_ids: ObjectIdCollection | None = None,
        suppressed_obj_ids: ObjectIdCollection | None = None,
        unconfirmed_obj_ids: ObjectIdCollection | None = None,
    ) -> PostprocessedOutput:
        """Turn a raw frame output into the public output.

        Objects are ordered by id; hidden ones (removed, suppressed, unconfirmed) and
        those with empty masks are dropped. Probabilities are the detection scores.
        Boxes come from the kept masks before any non-overlap constraint and are
        normalized by the frame size. When several objects are kept and the tracker
        supports it, masks are then made non-overlapping using the tracker scores
        (``obj_id_to_sam2_score``, 0 if missing). ``frame_stats`` is passed through.
        """
        obj_id_to_mask = out["obj_id_to_mask"]
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        height = int(inference_state["orig_height"])
        width = int(inference_state["orig_width"])
        frame_stats = out.get("frame_stats", None)
        if len(curr_obj_ids) == 0:
            return _empty_postprocessed_output(
                height,
                width,
                frame_stats,
                include_prod_outputs=self.running_in_prod,
            )

        masks = mx.concat(
            [_mask_to_mlx_bool(obj_id_to_mask[obj_id]) for obj_id in curr_obj_ids],
            axis=0,
        )
        hidden = _hidden_obj_ids(
            removed_obj_ids,
            suppressed_obj_ids,
            unconfirmed_obj_ids,
        )
        # Python-metadata boundary: object ids, hidden ids, and public output
        # ordering are Python-owned, so materialize keep indices once here.
        has_area = _array_to_numpy(mx.any(masks, axis=(1, 2)), dtype=bool)
        keep = np.array(
            [
                bool(has_area[index]) and obj_id not in hidden
                for index, obj_id in enumerate(curr_obj_ids)
            ],
            dtype=bool,
        )
        if not keep.any():
            return _empty_postprocessed_output(
                height,
                width,
                frame_stats,
                include_prod_outputs=self.running_in_prod,
            )

        keep_indices = np.flatnonzero(keep).astype(np.int64)
        kept_obj_ids = [curr_obj_ids[int(index)] for index in keep_indices]
        kept_masks = mx.take(masks, mx.array(keep_indices, dtype=mx.int64), axis=0)

        kept_probs = np.array(
            [
                _score_to_float(out["obj_id_to_score"][obj_id])
                for obj_id in kept_obj_ids
            ],
            dtype=np.float32,
        )
        sam2_score_by_obj_id = out.get("obj_id_to_sam2_score", {})
        kept_sam2_probs = mx.array(
            [
                _score_to_float(sam2_score_by_obj_id.get(obj_id, 0.0))
                for obj_id in kept_obj_ids
            ],
            dtype=mx.float32,
        )

        out_boxes_xyxy = _masks_to_boxes(kept_masks, kept_obj_ids)
        out_boxes_xywh = _call_mlx_unary_array(
            box_ops, "box_xyxy_to_xywh", out_boxes_xyxy
        ) / mx.array(
            [width, height, width, height],
            dtype=mx.float32,
        )

        if kept_masks.shape[0] > 1 and hasattr(
            self.tracker,
            "_apply_object_wise_non_overlapping_constraints",
        ):
            constrained = _apply_non_overlapping_constraints(
                self.tracker,
                kept_masks[:, None, :, :],
                kept_sam2_probs[:, None],
            )
            kept_masks = (constrained[:, 0, :, :] > 0).astype(mx.bool_)

        out_centers = _mask_centers(kept_masks) if self.running_in_prod else None
        out_binary_masks = cast(
            BoolArray,
            _array_to_numpy(kept_masks, dtype=bool),
        )
        outputs = PostprocessedOutput(
            out_obj_ids=np.array(kept_obj_ids, dtype=np.int64),
            out_probs=kept_probs,
            out_boxes_xywh=cast(
                FloatArray,
                _array_to_numpy(out_boxes_xywh, dtype=np.float32),
            ),
            out_binary_masks=out_binary_masks,
            frame_stats=frame_stats,
        )
        if self.running_in_prod:
            outputs["out_centers"] = out_centers
        return outputs

    def _postprocess_output_batched(
        self,
        height: int,
        width: int,
        batched_outs: Sequence[PostprocessBatchItem],
    ) -> list[PostprocessedOutput]:
        """Post-process several frames of one ``height`` x ``width`` video with ``_postprocess_output``."""
        inference_state: FrameDimensions = {
            "orig_height": height,
            "orig_width": width,
        }
        return [
            self._postprocess_output(
                inference_state,
                out,
                removed_obj_ids=removed_obj_ids,
                suppressed_obj_ids=suppressed_obj_ids,
                unconfirmed_obj_ids=unconfirmed_obj_ids,
            )
            for out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids in batched_outs
        ]

    def _compile_model(self) -> None:
        """Mark the model as compiled for propagation; ``mx.compile`` is not used in the MLX runtime."""
        self._compiled_for_propagation = True

    def _new_generator_state(self) -> GeneratorState:
        """Empty propagation-generator state."""
        return {
            "hotstart_buffer": [],
            "hotstart_removed_obj_ids": set(),
            "unconfirmed_obj_ids_per_frame": {},
            "postprocess_yield_list": [],
        }

    def _unconfirmed_ids_for_yield_frame(
        self,
        *,
        yield_frame_idx: int,
        num_frames: int,
        reverse: bool,
        unconfirmed_status_delay: int,
        unconfirmed_obj_ids_per_frame: Mapping[int, ObjectIdCollection],
    ) -> ObjectIdCollection | None:
        """Unconfirmed ids to hide on a yielded frame.

        Uses the ids recorded as unconfirmed ``unconfirmed_status_delay`` frames later in
        the tracking direction (clamped to the video).
        """
        unconfirmed_status_frame_idx = (
            yield_frame_idx + unconfirmed_status_delay
            if not reverse
            else yield_frame_idx - unconfirmed_status_delay
        )
        unconfirmed_status_frame_idx = max(
            0,
            min(unconfirmed_status_frame_idx, num_frames - 1),
        )
        return unconfirmed_obj_ids_per_frame.get(unconfirmed_status_frame_idx, None)

    def _postprocess_propagation_batch(
        self,
        inference_state: MultiplexInferenceState,
        batch_to_process: Sequence[PendingPropagationOutput],
        *,
        reverse: bool,
        unconfirmed_status_delay: int,
        unconfirmed_obj_ids_per_frame: Mapping[int, ObjectIdCollection],
    ) -> list[PropagationYield]:
        """Cache and post-process pending frame outputs; returns ``(frame_idx, output)`` pairs.

        Non-zero ranks get ``DUMMY_OUTPUT`` for every frame.
        """
        if self.rank != 0:
            return [
                (yield_frame_idx, DUMMY_OUTPUT)
                for yield_frame_idx, _, _ in batch_to_process
            ]

        height = int(inference_state["orig_height"])
        width = int(inference_state["orig_width"])
        num_frames = int(inference_state["num_frames"])
        batched_outs: list[PostprocessBatchItem] = []
        frame_indices: list[int] = []
        for yield_frame_idx, yield_out, removed_obj_ids_snapshot in batch_to_process:
            suppressed_obj_ids = yield_out.get("suppressed_obj_ids", None)
            unconfirmed_obj_ids = self._unconfirmed_ids_for_yield_frame(
                yield_frame_idx=yield_frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                unconfirmed_status_delay=unconfirmed_status_delay,
                unconfirmed_obj_ids_per_frame=unconfirmed_obj_ids_per_frame,
            )
            batched_outs.append(
                (
                    yield_out,
                    removed_obj_ids_snapshot,
                    suppressed_obj_ids,
                    unconfirmed_obj_ids,
                )
            )
            frame_indices.append(yield_frame_idx)
            self._cache_frame_outputs(
                inference_state,
                yield_frame_idx,
                yield_out["obj_id_to_mask"],
                suppressed_obj_ids=suppressed_obj_ids,
                removed_obj_ids=removed_obj_ids_snapshot,
                unconfirmed_obj_ids=unconfirmed_obj_ids,
            )

        if self.postprocess_batch_size > 1:
            postprocessed_outs = self._postprocess_output_batched(
                height,
                width,
                batched_outs,
            )
        else:
            postprocessed_outs = [
                self._postprocess_output(
                    inference_state,
                    yield_out,
                    removed_obj_ids_snapshot,
                    suppressed_obj_ids,
                    unconfirmed_obj_ids,
                )
                for (
                    yield_out,
                    removed_obj_ids_snapshot,
                    suppressed_obj_ids,
                    unconfirmed_obj_ids,
                ) in batched_outs
            ]
        return list(zip(frame_indices, postprocessed_outs))

    def _record_bucket_utilization(self, inference_state: MultiplexInferenceState) -> None:
        """For multiplex trackers, store bucket statistics in ``bucket_utilization_stats``.

        Records the object and bucket totals, the utilization (objects over
        ``buckets * bucket_capacity``) and the subscription (objects per bucket), both in
        percent. Asserts that each state's object count equals its ``total_valid_entries``.
        """
        if not self.is_multiplex:
            return
        total_valid_objects = 0
        total_num_buckets = 0
        states_raw = cast(object, inference_state.get("sam2_inference_states", []))
        if not isinstance(states_raw, Sequence) or isinstance(states_raw, str):
            raise TypeError("sam2_inference_states must be a sequence")
        for state_raw in cast(Sequence[object], states_raw):
            if not isinstance(state_raw, Mapping):
                continue
            state = cast(Mapping[object, object], state_raw)
            obj_ids_raw = state.get("obj_ids", [])
            obj_ids_length: object = getattr(obj_ids_raw, "__len__", None)
            if isinstance(obj_ids_raw, str) or not callable(obj_ids_length):
                raise TypeError("tracker state obj_ids must be sized")
            obj_ids_count = len(cast(_SizedValue, obj_ids_raw))
            multiplex_state = state.get("multiplex_state")
            if multiplex_state is None:
                continue
            total_valid_entries_raw: object = getattr(
                multiplex_state,
                "total_valid_entries",
                None,
            )
            if total_valid_entries_raw is not None:
                total_valid_entries = _integer_value(
                    total_valid_entries_raw,
                    name="total_valid_entries",
                )
                assert obj_ids_count == total_valid_entries
            num_buckets_raw: object = getattr(multiplex_state, "num_buckets", 0)
            num_buckets = _integer_value(num_buckets_raw, name="num_buckets")
            total_valid_objects += obj_ids_count
            total_num_buckets += num_buckets
        if total_num_buckets == 0:
            return
        inference_state["bucket_utilization_stats"] = {
            "total_valid_objects": total_valid_objects,
            "total_num_buckets": total_num_buckets,
            "bucket_utilization_rate": (
                total_valid_objects / (total_num_buckets * self.bucket_capacity)
            )
            * 100,
            "subscription_rate": (total_valid_objects / total_num_buckets) * 100,
        }

    def _run_single_frame_inference(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        reverse: bool,
        is_instance_processing: bool = False,
    ) -> RawFrameOutput:
        """Detect and track one frame and return its raw output.

        Updates ``sam2_inference_states``, ``tracker_metadata`` and ``previous_stages_out``
        in ``inference_state`` and caches the frame's masks. On rank 0 the output also lists
        removed ids, ids suppressed on this frame, unconfirmed ids and ``frame_stats``.
        ``is_instance_processing`` is unused.
        """
        del is_instance_processing
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["sam2_inference_states"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=inference_state["is_image_only"],
        )
        inference_state["sam2_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        if self.rank == 0:
            self._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)

        out: RawFrameOutput = {
            "obj_id_to_mask": cast(dict[int, object], obj_id_to_mask),
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_sam2_score": frame_scores_for_key(tracker_metadata_new, "obj_id_to_sam2_score_frame_wise")[frame_idx],
        }
        if self.rank == 0:
            rank0_metadata = cast(InitializedRank0Metadata, cast(Rank0TrackerMetadata, tracker_metadata_new)["rank0_metadata"])
            out["removed_obj_ids"] = rank0_metadata["removed_obj_ids"]
            out["suppressed_obj_ids"] = rank0_metadata["suppressed_obj_ids"][frame_idx]
            out["frame_stats"] = frame_stats
            if self.masklet_confirmation_enable:
                status = cast(ConfirmedRank0Metadata, rank0_metadata)["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                obj_ids_all_gpu = cast(
                    NDArray[np.int64],
                    cast(InitializedTrackerMetadata, tracker_metadata_new)["obj_ids_all_gpu"],
                )
                out["unconfirmed_obj_ids"] = obj_ids_all_gpu[is_unconfirmed].tolist()
            else:
                out["unconfirmed_obj_ids"] = []
        return out

    def _propagate_in_video_impl(
        self,
        inference_state: MultiplexInferenceState,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        is_instance_processing: bool = False,
        generator_state: GeneratorState | None = None,
        flush_hotstart_at_end: bool = True,
        persist_generator_state: bool = False,
    ) -> Iterator[PropagationYield]:
        """Generate ``(frame_idx, output)`` for ``propagate_in_video``.

        Runs ``_run_single_frame_inference`` over ``_get_processing_order``. With
        ``hotstart_delay > 0`` frames wait in a buffer of up to that many entries (all are
        released at the end when ``flush_hotstart_at_end``), and each yielded frame is
        post-processed with a snapshot of the ids removed so far. Outputs are post-processed
        in groups of ``postprocess_batch_size``. With ``persist_generator_state`` the
        remaining buffers are stored back into ``generator_state`` for the next call.
        """
        self._compile_model()
        processing_order, end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )
        feature_cache = inference_state.setdefault("feature_cache", {})
        feature_cache["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }

        if generator_state is None:
            generator_state = self._new_generator_state()
        hotstart_buffer = generator_state["hotstart_buffer"]
        hotstart_removed_obj_ids = generator_state["hotstart_removed_obj_ids"]
        unconfirmed_obj_ids_per_frame = generator_state["unconfirmed_obj_ids_per_frame"]
        postprocess_yield_list = generator_state.get("postprocess_yield_list", [])
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1

        for frame_idx in processing_order:
            out = self._run_single_frame_inference(
                inference_state,
                frame_idx,
                reverse,
                is_instance_processing=is_instance_processing,
            )
            if self.rank == 0:
                unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                if unconfirmed_obj_ids is not None:
                    unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids

            if self.hotstart_delay > 0:
                hotstart_buffer.append((frame_idx, out))
                if self.rank == 0:
                    hotstart_removed_obj_ids.update(out.get("removed_obj_ids", set()))

                if frame_idx == end_frame_idx and flush_hotstart_at_end:
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.hotstart_delay:
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    yield_list = []
            else:
                yield_list = [(frame_idx, out)]

            for yield_frame_idx, yield_out in yield_list:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )

            while len(postprocess_yield_list) >= self.postprocess_batch_size:
                batch_to_process = postprocess_yield_list[: self.postprocess_batch_size]
                postprocess_yield_list = postprocess_yield_list[
                    self.postprocess_batch_size :
                ]
                yield from self._postprocess_propagation_batch(
                    inference_state,
                    batch_to_process,
                    reverse=reverse,
                    unconfirmed_status_delay=unconfirmed_status_delay,
                    unconfirmed_obj_ids_per_frame=unconfirmed_obj_ids_per_frame,
                )

        if flush_hotstart_at_end and hotstart_buffer:
            for yield_frame_idx, yield_out in hotstart_buffer:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )
            hotstart_buffer = []

        if postprocess_yield_list:
            yield from self._postprocess_propagation_batch(
                inference_state,
                postprocess_yield_list,
                reverse=reverse,
                unconfirmed_status_delay=unconfirmed_status_delay,
                unconfirmed_obj_ids_per_frame=unconfirmed_obj_ids_per_frame,
            )
            postprocess_yield_list = []

        if persist_generator_state:
            generator_state["hotstart_buffer"] = hotstart_buffer
            generator_state["hotstart_removed_obj_ids"] = hotstart_removed_obj_ids
            generator_state["unconfirmed_obj_ids_per_frame"] = (
                unconfirmed_obj_ids_per_frame
            )
            generator_state["postprocess_yield_list"] = postprocess_yield_list

        self._record_bucket_utilization(inference_state)

    def init_state(
        self,
        resource_path: ResourceInput,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        input_is_mp4: bool = False,
        *,
        offload_state_to_cpu: bool = False,
    ) -> MultiplexInferenceState:
        """Load a video and create an empty session state.

        ``resource_path`` is anything ``load_resource_as_video_frames`` accepts (video
        file, image folder, image or PIL sequence); frames become normalized MLX images
        (loaded in a background thread when ``async_loading_frames``). A single frame
        makes the session image-only
        (``is_image_only``). ``use_torchcodec`` selects the unsupported torchcodec loader,
        ``use_cv2`` has no effect (cv2 is the default) and ``input_is_mp4`` is ignored.

        Raises:
            UnsupportedMultiplexRuntimeError: If ``offload_state_to_cpu`` is set.
        """
        del input_is_mp4
        if offload_state_to_cpu:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.init_state(offload_state_to_cpu=True)"
            )
        if use_torchcodec:
            video_loader_type = "torchcodec"
        elif use_cv2:
            video_loader_type = "cv2"
        else:
            video_loader_type = "cv2"
        images = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        inference_state: dict[str, object] = {
            "image_size": self.image_size,
            "num_frames": len(images),
            "device": "mlx",
            "orig_height": images.orig_height,
            "orig_width": images.orig_width,
            "constants": {},
        }
        self._construct_initial_input_batch(
            inference_state,
            _require_materialized_video_frames(images),
        )
        inference_state["sam2_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["is_image_only"] = len(images) == 1
        return cast(MultiplexInferenceState, inference_state)

    def reset_state(self, inference_state: MultiplexInferenceState) -> None:
        """Clear prompts, tracker states, metadata and caches so the session can be prompted again; the loaded frames are kept."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for frame_idx in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[frame_idx].text_ids = mx.array(
                [self.TEXT_ID_FOR_TEXT],
                dtype=mx.int64,
            )
            self._clear_find_input_geometry(
                inference_state["input_batch"].find_inputs[frame_idx]
            )
            inference_state["previous_stages_out"][frame_idx] = None
            inference_state["per_frame_raw_point_input"][frame_idx] = None
            inference_state["per_frame_raw_box_input"][frame_idx] = None
            inference_state["per_frame_visual_prompt"][frame_idx] = None
            inference_state["per_frame_geometric_prompt"][frame_idx] = None
            inference_state["per_frame_cur_step"][frame_idx] = 0
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        inference_state["sam2_inference_states"].clear()
        cast(dict[str, object], inference_state["tracker_metadata"]).clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"] = {}
        return None

    def add_fake_objects_to_inference_state(
        self,
        inference_state: MultiplexInferenceState,
        num_objects: int,
        frame_idx: int,
    ) -> MultiplexInferenceState:
        """Synthesize warm-up objects by calling the tracker-add path directly.

        Adds ``num_objects`` objects with ids ``0..num_objects - 1`` at ``frame_idx``, each
        with an all-ones mask at the memory-encoder resolution and score 1.0, and replaces
        the session metadata (``max_obj_id`` is set to ``num_objects``).

        Raises:
            UnsupportedMultiplexRuntimeError: In distributed runs.
            ValueError: For a negative ``num_objects`` or an out-of-range ``frame_idx``.
        """
        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.add_fake_objects_to_inference_state(distributed)"
            )
        num_objects = int(num_objects)
        if num_objects < 0:
            raise ValueError("num_objects must be non-negative.")
        frame_idx = int(frame_idx)
        num_frames = int(inference_state["num_frames"])
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        mask_downsampler = getattr(
            getattr(self.tracker, "maskmem_backbone", None),
            "mask_downsampler",
            None,
        )
        interpol_size = getattr(mask_downsampler, "interpol_size", None)
        input_mask_size = getattr(self.tracker, "input_mask_size", None)
        if interpol_size is not None:
            high_res_h, high_res_w = (int(interpol_size[0]), int(interpol_size[1]))
        elif input_mask_size is not None:
            high_res_h = high_res_w = int(input_mask_size)
        else:
            high_res_h = int(inference_state["orig_height"])
            high_res_w = int(inference_state["orig_width"])

        new_obj_ids = np.arange(num_objects, dtype=np.int64)
        new_obj_masks = mx.ones(
            (num_objects, high_res_h, high_res_w),
            dtype=mx.float32,
        )
        inference_state["sam2_inference_states"] = self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=num_frames,
            new_obj_ids=new_obj_ids,
            new_obj_masks=new_obj_masks,
            tracker_states_local=inference_state.setdefault(
                "sam2_inference_states", []
            ),
            orig_vid_height=int(inference_state["orig_height"]),
            orig_vid_width=int(inference_state["orig_width"]),
            feature_cache=inference_state.setdefault("feature_cache", {}),
        )

        obj_id_to_mask: dict[int, object] = {}
        if num_objects > 0:
            video_res_masks = (
                interpolate(
                    new_obj_masks[:, None, :, :],
                    size=(
                        int(inference_state["orig_height"]),
                        int(inference_state["orig_width"]),
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                > 0
            )
            for obj_idx, obj_id in enumerate(new_obj_ids.tolist()):
                obj_id_to_mask[int(obj_id)] = video_res_masks[obj_idx]
        for cache_frame_idx in range(num_frames):
            self._cache_frame_outputs(
                inference_state,
                cache_frame_idx,
                obj_id_to_mask,
            )

        tracker_metadata = self._initialize_metadata()
        tracker_metadata["obj_ids_per_gpu"][0] = new_obj_ids.copy()
        tracker_metadata["obj_ids_all_gpu"] = new_obj_ids.copy()
        tracker_metadata["num_obj_per_gpu"][0] = num_objects
        tracker_metadata["max_obj_id"] = num_objects
        tracker_metadata["obj_id_to_score"] = {
            int(obj_id): 1.0 for obj_id in new_obj_ids.tolist()
        }
        rank0_metadata = cast(Rank0TrackerMetadata, tracker_metadata)["rank0_metadata"]
        rank0_metadata["obj_first_frame_idx"] = {
            int(obj_id): frame_idx for obj_id in new_obj_ids.tolist()
        }
        rank0_metadata["trk_keep_alive"] = defaultdict(
            int,
            {
                int(obj_id): int(self.init_trk_keep_alive)
                for obj_id in new_obj_ids.tolist()
            },
        )
        tracker_metadata["obj_id_to_last_occluded"] = {
            int(obj_id): -1 for obj_id in new_obj_ids.tolist()
        }
        if self.masklet_confirmation_enable:
            rank0_metadata["masklet_confirmation"] = {
                "status": np.zeros(num_objects, dtype=np.int64),
                "consecutive_det_num": np.zeros(num_objects, dtype=np.int64),
            }
        tracker_metadata["gpu_metadata"] = self._hotstart_gpu_metadata_from_rank0(
            tracker_metadata,
            frame_idx=frame_idx,
            num_objects=num_objects,
        )
        if self.is_multiplex:
            num_buc_per_gpu = cast(MultiplexTrackerMetadata, tracker_metadata)["num_buc_per_gpu"]
            num_buc_per_gpu[0] = self._count_buckets_in_states(
                inference_state["sam2_inference_states"]
            )
        inference_state["tracker_metadata"] = cast(TrackerMetadata, tracker_metadata)
        return inference_state

    def _hotstart_gpu_metadata_has_tensors(
        self, gpu_metadata: GpuMetadata | HotstartGpuMetadata
    ) -> bool:
        """Whether every hotstart array is present."""
        return all(key in gpu_metadata for key in HOTSTART_GPU_METADATA_KEYS)

    def _rebuild_hotstart_gpu_metadata_from_rank0(
        self,
        tracker_metadata: TrackerMetadata,
        *,
        frame_idx: int | None,
    ) -> HotstartGpuMetadata:
        """Rebuild hotstart arrays from the rank-0 records (empty metadata when there are no objects)."""
        num_objects = int(
            np.asarray(
                tracker_metadata.get("obj_ids_all_gpu", []),
                dtype=np.int64,
            ).size
        )
        if num_objects == 0:
            return self._empty_hotstart_gpu_metadata()
        return self._hotstart_gpu_metadata_from_rank0(
            tracker_metadata,
            frame_idx=0 if frame_idx is None else int(frame_idx),
            num_objects=num_objects,
        )

    def _sync_hotstart_gpu_metadata_after_removal(
        self,
        tracker_metadata: TrackerMetadata,
        *,
        previous_obj_ids: TrackerArray,
        keep_mask: TrackerArray,
        frame_idx: int | None,
    ) -> None:
        """Update the hotstart arrays after objects were removed.

        ``keep_mask`` is a bool vector aligned with ``previous_obj_ids``; rows of objects
        with a False entry are dropped. Does nothing for non-multiplex trackers or without
        ``gpu_metadata``.

        Raises:
            ValueError: If ``keep_mask`` and ``previous_obj_ids`` differ in shape.
        """
        if not self.is_multiplex or "gpu_metadata" not in tracker_metadata:
            return
        previous_obj_ids_np = np.asarray(previous_obj_ids, dtype=np.int64).reshape(-1)
        keep_mask_np = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if keep_mask_np.shape != previous_obj_ids_np.shape:
            raise ValueError(
                "hotstart gpu metadata removal mask must match previous object ids; "
                f"got {keep_mask_np.shape} and {previous_obj_ids_np.shape}."
            )
        gpu_metadata_prev = tracker_metadata.get("gpu_metadata", {})
        if self._hotstart_gpu_metadata_has_tensors(gpu_metadata_prev):
            previous_tracker_metadata = cast(TrackerMetadata, dict(tracker_metadata))
            previous_tracker_metadata["obj_ids_all_gpu"] = previous_obj_ids_np
            gpu_metadata = self._ensure_hotstart_gpu_metadata(
                previous_tracker_metadata,
                gpu_metadata_prev,
                frame_idx=0 if frame_idx is None else int(frame_idx),
                num_objects=int(previous_obj_ids_np.size),
            )
            gpu_metadata["removed_mask"] = gpu_metadata["removed_mask"] | mx.array(
                ~keep_mask_np, dtype=mx.bool_
            )
            tracker_metadata["gpu_metadata"] = self._compact_hotstart_gpu_metadata(
                gpu_metadata
            )
        else:
            tracker_metadata["gpu_metadata"] = (
                self._rebuild_hotstart_gpu_metadata_from_rank0(
                    tracker_metadata,
                    frame_idx=frame_idx,
                )
            )

    def _sync_hotstart_gpu_metadata_after_addition(
        self,
        tracker_metadata: TrackerMetadata,
        *,
        frame_idx: int,
        num_new_objects: int,
    ) -> None:
        """Extend the hotstart arrays for the ``num_new_objects`` most recently appended object ids.

        Does nothing for non-multiplex trackers or without ``gpu_metadata``.

        Raises:
            ValueError: If more new objects are given than there are object ids.
        """
        if (
            not self.is_multiplex
            or num_new_objects <= 0
            or "gpu_metadata" not in tracker_metadata
        ):
            return
        obj_ids_all = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        current_num = int(obj_ids_all.size)
        previous_num = current_num - int(num_new_objects)
        if previous_num < 0:
            raise ValueError(
                "hotstart gpu metadata addition count exceeds current object ids; "
                f"got {num_new_objects} new objects for {current_num} ids."
            )
        gpu_metadata_prev = tracker_metadata.get("gpu_metadata", {})
        if self._hotstart_gpu_metadata_has_tensors(gpu_metadata_prev):
            previous_tracker_metadata = cast(TrackerMetadata, dict(tracker_metadata))
            previous_tracker_metadata["obj_ids_all_gpu"] = obj_ids_all[:previous_num]
            gpu_metadata = self._ensure_hotstart_gpu_metadata(
                previous_tracker_metadata,
                gpu_metadata_prev,
                frame_idx=int(frame_idx),
                num_objects=previous_num,
            )
            tracker_metadata["gpu_metadata"] = (
                self._extend_hotstart_gpu_metadata_for_new_objects(
                    gpu_metadata,
                    frame_idx=int(frame_idx),
                    num_new_objects=int(num_new_objects),
                )
            )
        else:
            previous_tracker_metadata = cast(TrackerMetadata, dict(tracker_metadata))
            previous_tracker_metadata["obj_ids_all_gpu"] = obj_ids_all[:previous_num]
            gpu_metadata = (
                self._empty_hotstart_gpu_metadata()
                if previous_num == 0
                else self._hotstart_gpu_metadata_from_rank0(
                    previous_tracker_metadata,
                    frame_idx=int(frame_idx),
                    num_objects=previous_num,
                )
            )
            tracker_metadata["gpu_metadata"] = (
                self._extend_hotstart_gpu_metadata_for_new_objects(
                    gpu_metadata,
                    frame_idx=int(frame_idx),
                    num_new_objects=int(num_new_objects),
                )
            )

    def _fetch_removed_object_frame(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int | None,
    ) -> tuple[int | None, PropagationOutput | None]:
        """Public output of ``frame_idx`` from the cached frame outputs, for use after a removal.

        Returns ``(None, None)`` for a None frame, the dummy output on non-zero ranks and an
        empty output when the frame has nothing cached.
        """
        if frame_idx is None:
            return frame_idx, None
        if self.rank != 0:
            return frame_idx, DUMMY_OUTPUT
        if frame_idx in inference_state.get("cached_frame_outputs", {}):
            cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
            tracker_metadata = inference_state.get("tracker_metadata", {})
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})
            suppressed_by_frame = rank0_metadata.get("suppressed_obj_ids", {})
            obj_id_to_sam2_score = tracker_metadata.get(
                "obj_id_to_sam2_score_frame_wise",
                {},
            ).get(frame_idx, {})
            obj_id_to_score = tracker_metadata.get("obj_id_to_score", {})
            frame_scores = {
                obj_id: obj_id_to_sam2_score.get(
                    obj_id,
                    obj_id_to_score.get(obj_id, 0.0),
                )
                for obj_id in cached_frame_outputs[frame_idx]
            }
            return (
                frame_idx,
                self._postprocess_output(
                    inference_state,
                    {
                        "obj_id_to_mask": cached_frame_outputs[frame_idx],
                        "obj_id_to_score": {
                            obj_id: obj_id_to_score.get(obj_id, frame_scores[obj_id])
                            for obj_id in cached_frame_outputs[frame_idx]
                        },
                        "obj_id_to_sam2_score": frame_scores,
                    },
                    suppressed_obj_ids=suppressed_by_frame.get(frame_idx, set()),
                ),
            )
        return (
            frame_idx,
            _empty_postprocessed_output(
                int(inference_state["orig_height"]),
                int(inference_state["orig_width"]),
                frame_stats=None,
                include_prod_outputs=self.running_in_prod,
            ),
        )

    def _validate_removable_sam2_states(
        self,
        inference_state: MultiplexInferenceState,
    ) -> None:
        """Require every tracker state to be a mapping with ``obj_ids`` or packed multiplex object ids.

        Raises:
            UnsupportedMultiplexRuntimeError: Otherwise.
        """
        sam2_states_raw = cast(object, inference_state.get("sam2_inference_states", []))
        if not isinstance(sam2_states_raw, Sequence) or isinstance(
            sam2_states_raw,
            str,
        ):
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.remove_object(existing-tracker-states)"
            )
        for sam2_state_raw in cast(Sequence[object], sam2_states_raw):
            if not isinstance(sam2_state_raw, Mapping):
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexTracking.remove_object(existing-tracker-states)"
                )
            sam2_state = cast(Mapping[object, object], sam2_state_raw)
            multiplex_state = sam2_state.get("multiplex_state")
            has_packed_ids = (
                multiplex_state is not None
                and getattr(multiplex_state, "object_ids", None) is not None
            )
            if "obj_ids" not in sam2_state and not has_packed_ids:
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexTracking.remove_object(existing-tracker-states)"
                )

    def _get_gpu_id_by_obj_id(
        self,
        inference_state: MultiplexInferenceState,
        obj_id: TrackerObjectId,
    ) -> int | None:
        """Rank that owns ``obj_id`` according to ``obj_ids_per_gpu``, or None if it is unknown."""
        obj_id_int = int(obj_id)
        obj_ids_per_gpu = inference_state.get("tracker_metadata", {}).get(
            "obj_ids_per_gpu",
            [],
        )
        for rank, obj_ids in enumerate(obj_ids_per_gpu):
            if obj_id_int in np.asarray(obj_ids, dtype=np.int64).reshape(-1).tolist():
                return rank
        return None

    def _remove_object_from_sam2_states(
        self,
        inference_state: MultiplexInferenceState,
        obj_id: TrackerObjectId,
        frame_idx: int | None,
        strict: bool,
    ) -> tuple[int | None, PropagationOutput | None]:
        """Remove an object from the tracker states and from every session record.

        The object leaves the tracker states, per-rank metadata, hotstart arrays, cached
        frame outputs, scores and suppression sets, and is marked removed (also in
        ``generator_state`` when present) so later outputs hide it. Returns the output of
        ``frame_idx`` (see ``_fetch_removed_object_frame``). An unknown id raises
        ``ValueError`` when ``strict`` and otherwise just returns that output.

        Raises:
            UnsupportedMultiplexRuntimeError: In distributed runs or for unsupported state
                layouts.
            ValueError: For an out-of-range ``frame_idx`` or, when ``strict``, an unknown id.
            TypeError: If ``obj_id`` is not integer-compatible.
        """
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.remove_object(existing-tracker-states-distributed)"
            )
        self._validate_removable_sam2_states(inference_state)
        if frame_idx is not None:
            frame_idx = int(frame_idx)
            num_frames = int(inference_state["num_frames"])
            if not 0 <= frame_idx < num_frames:
                raise ValueError(
                    f"frame_idx={frame_idx} is out of range for {num_frames} frames."
                )
        try:
            obj_id_int = int(obj_id)
        except (TypeError, ValueError) as exc:
            raise TypeError("obj_id must be integer-compatible.") from exc

        tracker_metadata = cast(InitializedTrackerMetadata, inference_state.setdefault("tracker_metadata", {}))
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())
        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            cast(Rank0TrackerMetadata, self._initialize_metadata())["rank0_metadata"],
        )

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id_int)
        found_in_cache = any(
            isinstance(cast(object, frame_outputs), Mapping) and obj_id_int in frame_outputs
            for frame_outputs in inference_state.get(
                "cached_frame_outputs", {}
            ).values()
        )
        if obj_rank is None and not found_in_cache:
            if strict:
                raise ValueError(
                    f"Object id {obj_id_int} does not exist in the tracking state."
                )
            return self._fetch_removed_object_frame(inference_state, frame_idx)

        previous_obj_ids_all = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        keep_mask_all = previous_obj_ids_all != obj_id_int
        if obj_rank is not None:
            tracker_states_local = inference_state["sam2_inference_states"]
            self._tracker_remove_objects(tracker_states_local, [obj_id_int])

            obj_ids_per_gpu = tracker_metadata["obj_ids_per_gpu"]
            kept_ids = np.asarray(obj_ids_per_gpu[obj_rank], dtype=np.int64)
            kept_ids = kept_ids[kept_ids != obj_id_int]
            obj_ids_per_gpu[obj_rank] = kept_ids
            tracker_metadata["num_obj_per_gpu"][obj_rank] = int(kept_ids.size)
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(obj_ids_per_gpu)
            if "num_buc_per_gpu" in tracker_metadata:
                remaining_bucket_count = self._count_buckets_in_states(
                    tracker_states_local
                )
                tracker_metadata["num_buc_per_gpu"][obj_rank] = (
                    remaining_bucket_count
                    if remaining_bucket_count > 0
                    else int(np.ceil(kept_ids.size / max(int(self.bucket_capacity), 1)))
                    if kept_ids.size > 0
                    else 0
                )

        for frame_outputs in inference_state.get("cached_frame_outputs", {}).values():
            if isinstance(cast(object, frame_outputs), MutableMapping):
                cast(MutableMapping[object, object], frame_outputs).pop(
                    obj_id_int,
                    None,
                )

        rank0_metadata.setdefault("removed_obj_ids", set()).add(obj_id_int)
        for suppressed_obj_ids in rank0_metadata.setdefault(
            "suppressed_obj_ids",
            defaultdict(set),
        ).values():
            suppressed_obj_ids.discard(obj_id_int)
        rank0_metadata.get("obj_first_frame_idx", {}).pop(obj_id_int, None)
        rank0_metadata.get("trk_keep_alive", {}).pop(obj_id_int, None)
        rank0_metadata.get("unmatched_frame_inds", {}).pop(obj_id_int, None)
        if obj_rank is not None:
            self._sync_hotstart_gpu_metadata_after_removal(
                cast(TrackerMetadata, tracker_metadata),
                previous_obj_ids=previous_obj_ids_all,
                keep_mask=keep_mask_all,
                frame_idx=frame_idx,
            )
        if "generator_state" in inference_state:
            inference_state["generator_state"].setdefault(
                "hotstart_removed_obj_ids",
                set(),
            ).add(obj_id_int)

        tracker_metadata.get("obj_id_to_score", {}).pop(obj_id_int, None)
        tracker_metadata.get("obj_id_to_last_occluded", {}).pop(obj_id_int, None)
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            for frame_scores in tracker_metadata.get(score_key, {}).values():
                frame_scores.pop(obj_id_int, None)

        return self._fetch_removed_object_frame(inference_state, frame_idx)

    def remove_object(
        self,
        inference_state: MultiplexInferenceState,
        obj_id: TrackerObjectId,
        frame_idx: int | None = 0,
        strict: bool = False,
        is_user_action: bool = False,
    ) -> tuple[int | None, PropagationOutput | None]:
        """Remove an object from cached/image-only state or local SAM2 states.

        With tracker states this delegates to ``_remove_object_from_sam2_states``.
        Otherwise (image-only sessions) the object is dropped from the cached outputs and
        metadata and the result is the output of the remaining objects on ``frame_idx``
        with ``frame_stats`` ``{'num_obj_removed': 1, 'num_obj_remaining': n}``. An unknown
        id raises ``ValueError`` when ``strict`` and otherwise returns an empty output.
        ``is_user_action`` is unused here.

        Returns:
            ``(frame_idx, output)``.

        Raises:
            TypeError: If ``obj_id`` is not integer-compatible, or ``frame_idx`` is None
                without tracker states.
            ValueError: For an out-of-range ``frame_idx``.
        """
        del is_user_action
        if inference_state.get("sam2_inference_states"):
            return self._remove_object_from_sam2_states(
                inference_state,
                obj_id=obj_id,
                frame_idx=frame_idx,
                strict=strict,
            )
        if frame_idx is None:
            raise TypeError("frame_idx cannot be None without SAM2 tracker states")
        num_frames = int(inference_state["num_frames"])
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )
        try:
            obj_id_int = int(obj_id)
        except (TypeError, ValueError) as exc:
            raise TypeError("obj_id must be integer-compatible.") from exc

        tracker_metadata = cast(InitializedTrackerMetadata, inference_state.setdefault("tracker_metadata", {}))
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())

        cached_frame_outputs = inference_state.setdefault("cached_frame_outputs", {})
        active_ids = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        found_in_active = bool(np.any(active_ids == obj_id_int))
        found_in_cache = any(
            isinstance(cast(object, frame_outputs), Mapping) and obj_id_int in frame_outputs
            for frame_outputs in cached_frame_outputs.values()
        )
        if not found_in_active and not found_in_cache:
            if strict:
                raise ValueError(
                    f"Object id {obj_id_int} does not exist in the tracking state."
                )
            return (
                frame_idx,
                _empty_postprocessed_output(
                    int(inference_state["orig_height"]),
                    int(inference_state["orig_width"]),
                    frame_stats=None,
                    include_prod_outputs=self.running_in_prod,
                ),
            )

        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            cast(Rank0TrackerMetadata, self._initialize_metadata())["rank0_metadata"],
        )
        rank0_metadata.setdefault("removed_obj_ids", set()).add(obj_id_int)
        if "generator_state" in inference_state:
            inference_state["generator_state"].setdefault(
                "hotstart_removed_obj_ids",
                set(),
            ).add(obj_id_int)

        for frame_outputs in cached_frame_outputs.values():
            if isinstance(cast(object, frame_outputs), MutableMapping):
                cast(MutableMapping[object, object], frame_outputs).pop(
                    obj_id_int,
                    None,
                )

        keep_mask = active_ids != obj_id_int
        kept_ids = active_ids[keep_mask]
        tracker_metadata["obj_ids_all_gpu"] = kept_ids
        tracker_metadata["obj_ids_per_gpu"][0] = kept_ids
        tracker_metadata["num_obj_per_gpu"][0] = int(kept_ids.size)
        if self.is_multiplex:
            num_buc_per_gpu = cast(MultiplexTrackerMetadata, tracker_metadata)["num_buc_per_gpu"]
            num_buc_per_gpu[0] = (
                int(np.ceil(kept_ids.size / self.bucket_capacity))
                if kept_ids.size > 0
                else 0
            )
        tracker_metadata.get("obj_id_to_score", {}).pop(obj_id_int, None)
        tracker_metadata.get("obj_id_to_last_occluded", {}).pop(obj_id_int, None)
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            for frame_scores in tracker_metadata.get(score_key, {}).values():
                frame_scores.pop(obj_id_int, None)

        rank0_metadata.get("obj_first_frame_idx", {}).pop(obj_id_int, None)
        rank0_metadata.get("trk_keep_alive", {}).pop(obj_id_int, None)
        rank0_metadata.get("unmatched_frame_inds", {}).pop(obj_id_int, None)
        if (
            self.masklet_confirmation_enable
            and "masklet_confirmation" in rank0_metadata
        ):
            confirmation = rank0_metadata["masklet_confirmation"]
            for key in ("status", "consecutive_det_num"):
                values = confirmation.get(key)
                if values is not None and len(values) == len(active_ids):
                    confirmation[key] = np.asarray(values)[keep_mask]
        self._sync_hotstart_gpu_metadata_after_removal(
            cast(TrackerMetadata, tracker_metadata),
            previous_obj_ids=active_ids,
            keep_mask=keep_mask,
            frame_idx=frame_idx,
        )

        remaining_masks = _int_keyed_object_dict(
            cached_frame_outputs.get(frame_idx, {}),
            name="cached frame outputs",
        )
        framewise_scores: dict[int, object] = {}
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            scores_by_frame = _mapping_view(
                tracker_metadata.get(score_key, {}),
                name=score_key,
            )
            framewise_scores.update(
                _int_keyed_object_dict(
                    scores_by_frame.get(frame_idx, {}),
                    name=f"{score_key}[{frame_idx}]",
                )
            )
        score_map = _int_keyed_object_dict(
            tracker_metadata.get("obj_id_to_score", {}),
            name="obj_id_to_score",
        )
        out_scores = {
            remaining_obj_id: framewise_scores.get(
                remaining_obj_id,
                score_map.get(remaining_obj_id, 0.0),
            )
            for remaining_obj_id in remaining_masks
        }
        return (
            frame_idx,
            self._postprocess_output(
                inference_state,
                {
                    "obj_id_to_mask": remaining_masks,
                    "obj_id_to_score": out_scores,
                    "obj_id_to_sam2_score": framewise_scores,
                    "frame_stats": {
                        "num_obj_removed": 1,
                        "num_obj_remaining": len(remaining_masks),
                    },
                },
            ),
        )

    def _get_processing_order(
        self,
        inference_state: MultiplexInferenceState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> tuple[range, int]:
        """Return ``(frames to process, last frame index)`` for a propagation request.

        Forward covers ``start_frame_idx`` through ``min(start + max_frame_num_to_track,
        num_frames - 1)``. Reverse starts at ``start_frame_idx - 1`` (the start frame is
        not revisited) and goes down to ``max(start - max_frame_num_to_track, 0)``.
        ``start_frame_idx=None`` uses the earliest prompted frame and
        ``max_frame_num_to_track=None`` means all frames.

        Raises:
            RuntimeError: If no frame has a prompt and no start frame is given.
        """
        num_frames = int(inference_state["num_frames"])
        previous_stages_out_raw = cast(object, inference_state["previous_stages_out"])
        if not isinstance(previous_stages_out_raw, Sequence) or isinstance(
            previous_stages_out_raw,
            str,
        ):
            raise TypeError("previous_stages_out must be a sequence")
        previous_stages_out = cast(Sequence[object | None], previous_stages_out_raw)
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at "
                "least one frame before propagation."
            )
        if start_frame_idx is None:
            start_frame_idx = min(
                frame_idx
                for frame_idx, out in enumerate(previous_stages_out)
                if out is not None
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track,
                num_frames - 1,
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    def propagate_in_video(
        self,
        inference_state: MultiplexInferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        output_prob_thresh: float = 0.5,
        compute_stability_score: bool = False,
        is_instance_processing: bool = False,
    ) -> Iterator[PropagationYield]:
        """Propagate the prompts through the video, yielding ``(frame_idx, PostprocessedOutput)``.

        See ``_get_processing_order`` for the frames visited. ``output_prob_thresh`` and
        ``compute_stability_score`` are unused.

        Raises:
            TypeError: If ``inference_state`` is not a dict.
            RuntimeError: If no frame has a prompt and no start frame is given.
        """
        del output_prob_thresh, compute_stability_score
        inference_state_raw = cast(object, inference_state)
        if not isinstance(inference_state_raw, dict):
            raise TypeError("inference_state must be a dictionary")
        if start_frame_idx is not None:
            start_frame_idx = _integer_value(start_frame_idx, name="start_frame_idx")
        if max_frame_num_to_track is not None:
            max_frame_num_to_track = _integer_value(
                max_frame_num_to_track, name="max_frame_num_to_track"
            )

        yield from self._propagate_in_video_impl(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            is_instance_processing=is_instance_processing,
        )

    def _init_backbone_out(self, inference_state: MultiplexInferenceState) -> TrackingBackboneOutput:
        """Collect the session's frames and the text-encoder outputs for its current prompts."""
        input_batch_raw = cast(object, inference_state["input_batch"])
        if not isinstance(input_batch_raw, BatchedDatapoint):
            raise TypeError("input_batch must be a BatchedDatapoint")
        input_batch = input_batch_raw
        feature_cache = _require_feature_cache(
            inference_state.setdefault("feature_cache", {})
        )
        text_outputs = text_outputs_for_batch(
            self.detector,
            feature_cache,
            input_batch.find_text_batch,
            device=self.device,
        )
        return cast(TrackingBackboneOutput, {
            "img_batch_all_stages": input_batch.img_batch,
            **text_outputs,
        })

    @overload
    def add_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: None = None,
        point_labels: ArrayInput | None = None,
        boxes_xywh: BoxCoordinatesInput | None = None,
        box_labels: BoxLabelsInput | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
    ) -> tuple[int, PostprocessedOutput]: ...

    @overload
    def add_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: PointCoordinatesInput | None = None,
        point_labels: ArrayInput | None = None,
        boxes_xywh: BoxCoordinatesInput | None = None,
        box_labels: BoxLabelsInput | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
    ) -> tuple[int, PostprocessedOutput | None]: ...

    def add_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: PointCoordinatesInput | None = None,
        point_labels: ArrayInput | None = None,
        boxes_xywh: BoxCoordinatesInput | None = None,
        box_labels: BoxLabelsInput | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
    ) -> tuple[int, PostprocessedOutput | None]:
        """Prompt the session with text and/or boxes and run detection on ``frame_idx``.

        All earlier prompts and tracker state are cleared first (``reset_state``).
        ``boxes_xywh`` is ``(N, 4)`` normalized to [0, 1] as ``(x0, y0, w, h)`` with
        ``box_labels`` an ``(N,)`` integer vector; both must be given together. Point
        prompts and ``clear_old_points=False`` are not supported here.

        Returns:
            ``(frame_idx, output)`` for the prompted frame.

        Raises:
            ValueError: If no prompt is given, ``frame_idx`` is out of range, boxes are
                malformed or not normalized, or ``clear_old_boxes`` is False.
            UnsupportedMultiplexRuntimeError: For point prompts.
        """
        del output_prob_thresh
        frame_idx = int(frame_idx)
        num_frames = int(inference_state["num_frames"])
        if text_str is None and points is None and boxes_xywh is None:
            raise ValueError("add_prompt requires text_str, boxes_xywh, or points.")
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )
        if points is not None or point_labels is not None or not clear_old_points:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.add_prompt(point-prompts)"
            )
        if not clear_old_boxes:
            raise ValueError("clear_old_boxes must be True.")
        if (boxes_xywh is None) != (box_labels is None):
            raise ValueError("boxes_xywh and box_labels must be provided together.")
        text_str = _optional_text(text_str)

        self.reset_state(inference_state)

        if text_str is not None:
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            for find_input in inference_state["input_batch"].find_inputs:
                find_input.text_ids = mx.full(
                    cast(mx.array, find_input.text_ids).shape,
                    self.TEXT_ID_FOR_TEXT,
                    dtype=mx.int64,
                )

        if boxes_xywh is not None:
            boxes_xywh_mx = (
                mlx_array(boxes_xywh, dtype=mx.float32)
                if isinstance(boxes_xywh, np.ndarray)
                else mx.array(boxes_xywh, dtype=mx.float32)
            )
            box_labels_value = cast(BoxLabelsInput, box_labels)
            box_labels_mx = (
                mlx_array(box_labels_value, dtype=mx.int64)
                if isinstance(box_labels_value, np.ndarray)
                else mx.array(box_labels_value, dtype=mx.int64)
            )
            if boxes_xywh_mx.ndim != 2 or boxes_xywh_mx.shape[-1] != 4:
                raise ValueError(
                    f"boxes_xywh must have shape (N, 4), got {boxes_xywh_mx.shape}."
                )
            if boxes_xywh_mx.shape[0] == 0:
                raise ValueError("boxes_xywh must contain at least one box.")
            if (
                box_labels_mx.ndim != 1
                or box_labels_mx.shape[0] != boxes_xywh_mx.shape[0]
            ):
                raise ValueError(
                    "box_labels must have shape (N,) and match boxes_xywh; "
                    f"got {box_labels_mx.shape} for {boxes_xywh_mx.shape[0]} boxes."
                )
            boxes_cxcywh = _call_mlx_unary_array(
                box_ops, "box_xywh_to_cxcywh", boxes_xywh_mx
            )
            boxes_in_range = mx.all(
                (boxes_xywh_mx >= 0.0)
                & (boxes_xywh_mx <= 1.0)
                & (boxes_cxcywh >= 0.0)
                & (boxes_cxcywh <= 1.0)
            )
            if not bool(_array_to_numpy(boxes_in_range, dtype=bool).reshape(())):
                raise ValueError("boxes_xywh must be normalized to [0, 1].")
            inference_state["per_frame_raw_box_input"][frame_idx] = (
                boxes_cxcywh,
                box_labels_mx,
            )
            _, _, geometric_prompt = self._get_visual_prompt(
                inference_state,
                frame_idx,
                boxes_cxcywh,
                box_labels_mx,
            )
            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        out = self._run_single_frame_inference(
            inference_state,
            frame_idx,
            reverse=False,
        )
        return frame_idx, self._postprocess_output(inference_state, out)

    def forward(
        self,
        input: object | None = None,
        is_inference: bool = False,
    ) -> dict[int, EvaluatorPredictions]:
        """Evaluator entry point: track every text prompt of a ``BatchedDatapoint``.

        Each prompt is added on frame 0 and propagated (image-only inputs use the prompt
        output directly); object ids are offset per prompt so they stay unique. Rank and
        world size are forced to 0 and 1 on this object and the detector during the call
        and restored afterwards. Returns ``{video_id: predictions}`` from
        ``prep_for_evaluator``.

        Raises:
            TypeError: If ``input`` is not a ``BatchedDatapoint``.
            ValueError: If ``raw_images``, ``find_metadatas`` or the category ids are
                missing or too short.
        """
        del is_inference
        if not isinstance(input, BatchedDatapoint):
            raise TypeError(
                "Sam3MultiplexTracking.forward requires a BatchedDatapoint."
            )
        if input.raw_images is None:
            raise ValueError("Sam3MultiplexTracking.forward requires input.raw_images.")
        if not input.find_metadatas or input.find_metadatas[0] is None:
            raise ValueError(
                "Sam3MultiplexTracking.forward requires find_metadatas[0]."
            )

        metadata = input.find_metadatas[0]
        prompt_ids = _array_to_numpy(
            metadata.original_category_id,
            dtype=np.int64,
        ).reshape(-1)
        prompt_list = list(input.find_text_batch)
        if len(prompt_ids) < len(prompt_list):
            raise ValueError(
                "original_category_id must contain at least one id per text prompt."
            )

        original_rank = self.rank
        original_world_size = self.world_size
        detector_has_rank = hasattr(self.detector, "rank")
        detector_has_world_size = hasattr(self.detector, "world_size")
        original_detector_rank = getattr(self.detector, "rank", None)
        original_detector_world_size = getattr(self.detector, "world_size", None)

        self.rank = 0
        self.world_size = 1
        if detector_has_rank:
            setattr(self.detector, "rank", 0)
        if detector_has_world_size:
            setattr(self.detector, "world_size", 1)

        try:
            tracking_res: defaultdict[int, dict[int, BoolArray]] = defaultdict(dict)
            scores_labels: dict[int, tuple[object, int]] = {}
            inference_state = self.init_state(resource_path=input.raw_images)
            for prompt_id, prompt in zip(prompt_ids, prompt_list):
                _, prompt_out = self.add_prompt(
                    inference_state,
                    frame_idx=0,
                    text_str=prompt,
                )
                start_obj_id = max(scores_labels.keys(), default=-1) + 1
                obj_ids_this_prompt: set[int] = set()
                if inference_state["is_image_only"]:
                    prompt_outputs = [(0, prompt_out)]
                else:
                    prompt_outputs = self.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,
                        max_frame_num_to_track=inference_state["num_frames"],
                        reverse=False,
                    )
                for frame_idx, out in prompt_outputs:
                    postprocessed_out = require_postprocessed_output(out)
                    for obj_id, mask in zip(
                        postprocessed_out["out_obj_ids"],
                        postprocessed_out["out_binary_masks"],
                        strict=True,
                    ):
                        output_obj_id = int(obj_id) + start_obj_id
                        # evaluator-export-boundary: prep_for_evaluator builds COCO-style
                        # NumPy/RLE payloads, so the postprocessed masks remain on host.
                        tracking_res[int(frame_idx)][output_obj_id] = np.asarray(
                            mask,
                            dtype=bool,
                        )[None, :, :]
                        obj_ids_this_prompt.add(output_obj_id)

                obj_id_to_score = cast(InitializedTrackerMetadata, inference_state["tracker_metadata"])["obj_id_to_score"]
                for obj_id, score in obj_id_to_score.items():
                    output_obj_id = int(obj_id) + start_obj_id
                    if output_obj_id in obj_ids_this_prompt:
                        scores_labels[output_obj_id] = (score, int(prompt_id))
                self.reset_state(inference_state)

            video_id = int(
                _array_to_numpy(metadata.original_image_id, dtype=np.int64).reshape(-1)[
                    0
                ]
            )
            preds = self.prep_for_evaluator(
                input.raw_images,
                tracking_res,
                scores_labels,
            )
            return {video_id: preds}
        finally:
            self.rank = original_rank
            self.world_size = original_world_size
            if detector_has_rank:
                setattr(self.detector, "rank", original_detector_rank)
            if detector_has_world_size:
                setattr(self.detector, "world_size", original_detector_world_size)


class Sam3MultiplexTrackingProd(Sam3MultiplexTracking):
    """Variant that keeps a ``generator_state`` in the session across ``propagate_in_video`` calls, so propagation can be issued in batches (``is_last_batch`` flushes the hotstart buffer)."""

    def init_state(
        self,
        resource_path: ResourceInput,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        input_is_mp4: bool = False,
        *,
        offload_state_to_cpu: bool = False,
    ) -> MultiplexInferenceState:
        inference_state = super().init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
            input_is_mp4=input_is_mp4,
        )
        inference_state["generator_state"] = self._new_generator_state()
        return inference_state

    def reset_state(self, inference_state: MultiplexInferenceState) -> None:
        super().reset_state(inference_state)
        inference_state["generator_state"] = self._new_generator_state()

    def propagate_in_video(
        self,
        inference_state: MultiplexInferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        output_prob_thresh: float = 0.5,
        compute_stability_score: bool = False,
        is_instance_processing: bool = False,
        *,
        is_last_batch: bool = True,
    ) -> Iterator[PropagationYield]:
        del output_prob_thresh, compute_stability_score
        generator_state = inference_state.setdefault(
            "generator_state",
            self._new_generator_state(),
        )
        yield from self._propagate_in_video_impl(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            is_instance_processing=is_instance_processing,
            generator_state=generator_state,
            flush_hotstart_at_end=is_last_batch,
            persist_generator_state=True,
        )


class Sam3MultiplexTrackingWithInteractivity(Sam3MultiplexTracking):
    """Tracking session with point prompts, per-object refinement and an action history.

    ``add_prompt`` with ``points`` and ``obj_id`` adds or refines one object through the
    tracker; ``propagate_in_video`` chooses what to compute (full propagation, partial
    propagation of the edited objects, or fetching cached results) from the recorded
    actions.

    Args:
        use_prev_mem_frame: Forwarded to the tracker's ``add_new_points`` for point prompts.
        use_stateless_refinement: Remove an object that was never refined and re-add it
            before its first point refinement instead of refining its existing state.
        refinement_detector_cond_frame_removal_window: Radius in frames around a refinement
            within which detector-added conditioning inputs are cleared.
        **kwargs: ``TrackingOptions``.
    """

    def __init__(
        self,
        use_prev_mem_frame: bool = False,
        use_stateless_refinement: bool = False,
        refinement_detector_cond_frame_removal_window: int = 30 * 4,
        **kwargs: Unpack[TrackingOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.use_stateless_refinement = use_stateless_refinement
        self.refinement_detector_cond_frame_removal_window = (
            refinement_detector_cond_frame_removal_window
        )

    def init_state(
        self,
        resource_path: ResourceInput,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        input_is_mp4: bool = False,
        *,
        offload_state_to_cpu: bool = False,
    ) -> MultiplexInferenceState:
        """Create a session as ``Sam3MultiplexTracking.init_state`` does, plus an empty ``action_history`` (and one shared tracker state when ``per_obj_inference``)."""
        inference_state = super().init_state(
            resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
            input_is_mp4=input_is_mp4,
        )
        inference_state["action_history"] = []
        if bool(getattr(self.tracker, "per_obj_inference", False)):
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]
        return inference_state

    def reset_state(self, inference_state: MultiplexInferenceState) -> None:
        """Reset as the base class does and also clear ``action_history``."""
        super().reset_state(inference_state)
        inference_state.setdefault("action_history", []).clear()
        if bool(getattr(self.tracker, "per_obj_inference", False)):
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]

    def _init_new_sam2_state(self, inference_state: MultiplexInferenceState) -> TrackerState:
        """Create an empty tracker state for the video via the tracker's ``init_state``."""
        init_state = getattr(self.tracker, "init_state", None)
        if init_state is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity._init_new_sam2_state"
            )
        sam2_state_raw: object = cast(_KeywordCall, init_state)(
            cached_features=inference_state.get("feature_cache", {}),
            video_height=int(inference_state["orig_height"]),
            video_width=int(inference_state["orig_width"]),
            num_frames=int(inference_state["num_frames"]),
        )
        sam2_state = _require_tracker_state(sam2_state_raw)
        sam2_state.setdefault("obj_ids", [])
        return sam2_state

    def _prepare_backbone_feats(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        *,
        reverse: bool,
    ) -> None:
        """Ensure the tracker backbone features of ``frame_idx`` are cached by running the detector on it if needed.

        Raises:
            RuntimeError: If the detector did not cache them.
        """
        feature_cache = inference_state["feature_cache"]
        if frame_idx in feature_cache:
            return
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=int(inference_state["num_frames"]),
            reverse=reverse,
            input_batch=inference_state["input_batch"],
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            use_batched_grounding=False,
        )
        if frame_idx not in feature_cache:
            raise RuntimeError(
                f"Detector did not cache tracker backbone features for frame {frame_idx}."
            )

    def _create_singleton_multiplex_state(
        self,
        source_state: Mapping[str, object],
        obj_id: int,
    ) -> MultiplexState:
        """Build a one-object multiplex state, from the controller's ``get_state`` when it has one, else a single bucket with object 0 in slot 0."""
        controller = getattr(self.tracker, "multiplex_controller", None)
        get_state = getattr(controller, "get_state", None)
        if get_state is not None:
            return cast(_ControllerStateCall, get_state)(
                num_valid_entries=1,
                device=source_state.get("device", "mlx"),
                dtype=mx.float32,
                random=False,
                object_ids=[obj_id],
            )

        source_multiplex_state = source_state.get("multiplex_state")
        multiplex_count = int(
            getattr(
                source_multiplex_state,
                "multiplex_count",
                getattr(controller, "multiplex_count", 1),
            )
        )
        allowed_bucket_capacity = int(
            getattr(
                source_multiplex_state,
                "allowed_bucket_capacity",
                getattr(controller, "allowed_bucket_capacity", 1),
            )
        )
        multiplex_count = max(multiplex_count, 1)
        allowed_bucket_capacity = max(min(allowed_bucket_capacity, multiplex_count), 1)
        return MultiplexState(
            [[0] + [-1] * (multiplex_count - 1)],
            device=source_state.get("device", "mlx"),
            dtype=mx.float32,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=[obj_id],
        )

    def _extract_object_to_singleton_state(
        self,
        inference_state: MultiplexInferenceState,
        obj_id: TrackerObjectId,
        obj_rank: int,
    ) -> None:
        """Move one object of a multi-object tracker state into its own single-object state.

        Lets the object be refined independently. Copies the object's slice of the stored
        frame outputs, its point and mask inputs and its per-object outputs, removes it
        from the source state (dropping that state if it becomes empty) and appends the
        new state. Does nothing if the object is already alone or lives on another rank.

        Raises:
            ValueError: If the object is not in the local states or its stored outputs are
                incomplete.
            UnsupportedMultiplexRuntimeError: In distributed runs or if the tracker has no
                ``remove_object``.
        """
        if self.rank != obj_rank:
            return
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "_extract_object_to_singleton_state(distributed)"
            )

        obj_id_int = int(obj_id)
        tracker_states_local = _require_tracker_state_list(
            inference_state.setdefault("sam2_inference_states", [])
        )
        source_state: TrackerState | None = None
        source_state_idx: int | None = None
        for idx, state in enumerate(tracker_states_local):
            state_obj_ids = _object_id_sequence(state.get("obj_ids", []))
            if obj_id_int in state_obj_ids:
                source_state = state
                source_state_idx = idx
                break
        if source_state is None or source_state_idx is None:
            raise ValueError(
                f"Object id {obj_id_int} is not present in local SAM2 states."
            )
        if len(_object_id_sequence(source_state.get("obj_ids", []))) <= 1:
            return

        obj_idx_in_source = self._lookup_existing_obj_idx(source_state, obj_id_int)
        if obj_idx_in_source is None:
            raise ValueError(
                f"Object id {obj_id_int} is missing from source obj_id mapping."
            )
        multiplex_state = source_state.get("multiplex_state")
        singleton_consolidated_outputs: dict[FrameStorage, PerObjectFrameOutputs] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        source_output_dict = _mapping_view(
            source_state.get("output_dict", {}),
            name="source output_dict",
        )
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            source_outputs = _mapping_view(
                source_output_dict.get(storage_key, {}),
                name=f"source output_dict {storage_key}",
            )
            for output_frame_idx_raw, source_frame_out_raw in source_outputs.items():
                output_frame_idx = _integer_value(
                    output_frame_idx_raw,
                    name=f"{storage_key} frame index",
                )
                source_frame_out = cast(SingletonFrameOutput, _mapping_view(
                    source_frame_out_raw,
                    name=f"{storage_key} frame output",
                ))
                pred_masks = source_frame_out.get("pred_masks")
                object_score_logits = source_frame_out.get("object_score_logits")
                if pred_masks is None or object_score_logits is None:
                    raise ValueError(
                        "Packed singleton extraction requires pred_masks and "
                        f"object_score_logits for {storage_key} frame {output_frame_idx}."
                    )
                if (
                    _first_axis_size(pred_masks, name="pred_masks")
                    < obj_idx_in_source + 1
                ):
                    continue
                singleton_frame_out: SingletonFrameOutput = {
                    "pred_masks": _copy_first_axis_slice(
                        pred_masks,
                        obj_idx_in_source,
                        obj_idx_in_source + 1,
                    ),
                    "object_score_logits": _copy_first_axis_slice(
                        object_score_logits,
                        obj_idx_in_source,
                        obj_idx_in_source + 1,
                    ),
                    "image_features": source_frame_out.get("image_features"),
                    "image_pos_enc": source_frame_out.get("image_pos_enc"),
                    "local_obj_id_to_idx": {obj_id_int: 0},
                    "maskmem_features": _demux_and_slice_first_axis(
                        source_frame_out.get("maskmem_features"),
                        multiplex_state,
                        obj_idx_in_source,
                    ),
                }

                maskmem_pos_enc = cast(object, source_frame_out.get("maskmem_pos_enc"))
                if maskmem_pos_enc is None:
                    singleton_frame_out["maskmem_pos_enc"] = None
                else:
                    if not isinstance(maskmem_pos_enc, Sequence) or isinstance(
                        maskmem_pos_enc,
                        str,
                    ):
                        raise TypeError("maskmem_pos_enc must be a sequence")
                    singleton_frame_out["maskmem_pos_enc"] = [
                        _demux_and_slice_first_axis(
                            level_enc, multiplex_state, obj_idx_in_source
                        )
                        if level_enc is not None
                        else None
                        for level_enc in cast(Sequence[StoredArray | None], maskmem_pos_enc)
                    ]

                if "obj_ptr" in source_frame_out and getattr(
                    self.tracker, "use_obj_ptrs_in_encoder", False
                ):
                    singleton_frame_out["obj_ptr"] = _demux_and_slice_first_axis(
                        source_frame_out["obj_ptr"],
                        multiplex_state,
                        obj_idx_in_source,
                    )
                if "conditioning_objects" in source_frame_out:
                    conditioning_objects = _object_id_collection(
                        source_frame_out["conditioning_objects"]
                    )
                    singleton_frame_out["conditioning_objects"] = (
                        {0} if obj_idx_in_source in conditioning_objects else set()
                    )
                singleton_consolidated_outputs[storage_key][output_frame_idx] = (
                    singleton_frame_out
                )

        point_inputs_per_obj = _mapping_view(
            source_state.get("point_inputs_per_obj", {}),
            name="point_inputs_per_obj",
        )
        extracted_point_inputs = dict(
            _mapping_view(
                point_inputs_per_obj.get(obj_idx_in_source, {}),
                name="object point inputs",
            )
        )
        mask_inputs_per_obj = _mapping_view(
            source_state.get("mask_inputs_per_obj", {}),
            name="mask_inputs_per_obj",
        )
        extracted_mask_inputs = dict(
            _mapping_view(
                mask_inputs_per_obj.get(obj_idx_in_source, {}),
                name="object mask inputs",
            )
        )

        extracted_obj_cond_outputs: PerObjectFrameOutputs = {}
        extracted_obj_non_cond_outputs: PerObjectFrameOutputs = {}
        output_dict_per_obj = _mapping_view(
            source_state.get("output_dict_per_obj", {}),
            name="output_dict_per_obj",
        )
        obj_output_dict_raw = output_dict_per_obj.get(obj_idx_in_source)
        if obj_output_dict_raw is not None:
            obj_output_dict = _mapping_view(
                obj_output_dict_raw,
                name="object output_dict",
            )
            cond_input_keys = (
                extracted_point_inputs.keys() | extracted_mask_inputs.keys()
            )
            cond_outputs = cast(Mapping[int, SingletonFrameOutput], _mapping_view(
                obj_output_dict.get("cond_frame_outputs", {}),
                name="object cond_frame_outputs",
            ))
            extracted_obj_cond_outputs = {
                frame_key: frame_out
                for frame_key, frame_out in cond_outputs.items()
                if frame_key in cond_input_keys
            }
            extracted_obj_non_cond_outputs = cast(PerObjectFrameOutputs, dict(
                _mapping_view(
                    obj_output_dict.get("non_cond_frame_outputs", {}),
                    name="object non_cond_frame_outputs",
                )
            ))

        extracted_temp_cond_outputs: PerObjectFrameOutputs = {}
        extracted_temp_non_cond_outputs: PerObjectFrameOutputs = {}
        temp_output_dict_per_obj = _mapping_view(
            source_state.get("temp_output_dict_per_obj", {}),
            name="temp_output_dict_per_obj",
        )
        temp_obj_output_dict_raw = temp_output_dict_per_obj.get(obj_idx_in_source)
        if temp_obj_output_dict_raw is not None:
            temp_obj_output_dict = _mapping_view(
                temp_obj_output_dict_raw,
                name="object temp_output_dict",
            )
            extracted_temp_cond_outputs = cast(PerObjectFrameOutputs, dict(
                _mapping_view(
                    temp_obj_output_dict.get("cond_frame_outputs", {}),
                    name="object temp cond_frame_outputs",
                )
            ))
            extracted_temp_non_cond_outputs = cast(PerObjectFrameOutputs, dict(
                _mapping_view(
                    temp_obj_output_dict.get("non_cond_frame_outputs", {}),
                    name="object temp non_cond_frame_outputs",
                )
            ))

        remove_object = getattr(self.tracker, "remove_object", None)
        if remove_object is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "_extract_object_to_singleton_state(remove_object)"
            )
        remaining_obj_ids, _ = cast(_RemoveObjectCall, remove_object)(
            source_state,
            obj_id_int,
            strict=False,
            need_output=False,
        )

        new_sam2_state = self._init_new_sam2_state(inference_state)
        new_sam2_state["obj_id_to_idx"] = {obj_id_int: 0}
        new_sam2_state["obj_idx_to_id"] = {0: obj_id_int}
        new_sam2_state["obj_ids"] = [obj_id_int]
        new_sam2_state["point_inputs_per_obj"] = {0: extracted_point_inputs}
        new_sam2_state["mask_inputs_per_obj"] = {0: extracted_mask_inputs}
        new_sam2_state["output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_obj_cond_outputs,
                "non_cond_frame_outputs": extracted_obj_non_cond_outputs,
            }
        }
        new_sam2_state["temp_output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_temp_cond_outputs,
                "non_cond_frame_outputs": extracted_temp_non_cond_outputs,
            }
        }

        new_multiplex_state = self._create_singleton_multiplex_state(
            source_state,
            obj_id_int,
        )
        new_sam2_state["multiplex_state"] = new_multiplex_state
        if getattr(self.tracker, "use_obj_ptrs_in_encoder", False):
            for storage_outputs in singleton_consolidated_outputs.values():
                for frame_out in storage_outputs.values():
                    pointer = frame_out.get("obj_ptr")
                    if pointer is not None:
                        frame_out["obj_ptr"] = cast(_MuxTransform, new_multiplex_state).mux(pointer)
        new_sam2_state["output_dict"] = singleton_consolidated_outputs

        for key in ("first_ann_frame_idx", "tracking_has_started"):
            if key in source_state:
                new_sam2_state[key] = source_state[key]
        new_sam2_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }

        tracker_states_local.append(new_sam2_state)
        if len(remaining_obj_ids) == 0:
            tracker_states_local.pop(source_state_idx)

    def add_action_history(
        self,
        inference_state: MultiplexInferenceState,
        action_type: str,
        frame_idx: int | None = None,
        obj_ids: list[TrackerObjectId] | None = None,
    ) -> None:
        """Append an action to the session history.

        ``action_type`` is ``'add'``, ``'remove'`` or ``'refine'`` (instance actions), or
        ``'propagation_full'``, ``'propagation_partial'``, ``'propagation_fetch'`` or
        ``'propagation_cancel'``.

        Raises:
            ValueError: For any other action type.
        """
        instance_actions = {"add", "remove", "refine"}
        propagation_actions = {
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
            "propagation_cancel",
        }
        if action_type not in instance_actions | propagation_actions:
            valid = sorted(instance_actions | propagation_actions)
            raise ValueError(
                f"Invalid action type: {action_type}, must be one of {valid}."
            )
        inference_state.setdefault("action_history", []).append(
            {
                "type": action_type,
                "frame_idx": frame_idx,
                "obj_ids": obj_ids,
            }
        )

    @overload
    def add_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: None = None,
        point_labels: ArrayInput | None = None,
        boxes_xywh: BoxCoordinatesInput | None = None,
        box_labels: BoxLabelsInput | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: TrackerObjectId | None = None,
        rel_coordinates: bool = True,
    ) -> tuple[int, PostprocessedOutput]: ...

    @overload
    def add_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: PointCoordinatesInput | None = None,
        point_labels: ArrayInput | None = None,
        boxes_xywh: BoxCoordinatesInput | None = None,
        box_labels: BoxLabelsInput | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: TrackerObjectId | None = None,
        rel_coordinates: bool = True,
    ) -> tuple[int, PostprocessedOutput | None]: ...

    def add_prompt(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: PointCoordinatesInput | None = None,
        point_labels: ArrayInput | None = None,
        boxes_xywh: BoxCoordinatesInput | None = None,
        box_labels: BoxLabelsInput | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: TrackerObjectId | None = None,
        rel_coordinates: bool = True,
    ) -> tuple[int, PostprocessedOutput | None]:
        """Prompt the session with points for one object, or with text or boxes.

        With ``points``, ``obj_id`` and ``point_labels`` are required, ``text_str`` and
        ``boxes_xywh`` must be None and the call is handled by ``add_sam2_new_points``.
        Otherwise it behaves like ``Sam3MultiplexTracking.add_prompt`` (with batched
        grounding switched off for the call) and a caller-supplied ``obj_id`` is not
        supported.

        Raises:
            ValueError: If the point-prompt arguments are inconsistent.
            Sam3MlxUnsupportedError: If ``obj_id`` is given for a text or box prompt.
        """
        if points is not None:
            del output_prob_thresh, clear_old_boxes, box_labels
            if text_str is not None or boxes_xywh is not None:
                raise ValueError(
                    "When points are provided, text_str and boxes_xywh must be None."
                )
            if obj_id is None:
                raise ValueError("obj_id must be provided when points are provided.")
            if point_labels is None:
                raise ValueError(
                    "point_labels must be provided when points are provided."
                )
            return self.add_sam2_new_points(
                inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                clear_old_points=clear_old_points,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=self.use_prev_mem_frame,
            )
        if obj_id is not None:
            raise_unsupported(
                "Sam3MultiplexTrackingWithInteractivity.add_prompt(obj_id)",
                reason="video-multiplex",
                detail=(
                    "Text and box prompts create detector-managed identities; a "
                    "caller-supplied object ID is not supported on this branch."
                ),
                alternative="Omit obj_id or use the point-prompt branch.",
            )
        orig_use_batched_grounding = self.use_batched_grounding
        self.use_batched_grounding = False
        try:
            return super().add_prompt(
                inference_state,
                frame_idx=frame_idx,
                text_str=text_str,
                clear_old_points=clear_old_points,
                points=points,
                point_labels=point_labels,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
                clear_old_boxes=clear_old_boxes,
                output_prob_thresh=output_prob_thresh,
            )
        finally:
            self.use_batched_grounding = orig_use_batched_grounding

    def add_sam2_new_points(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        obj_id: TrackerObjectId,
        points: PointCoordinatesInput,
        labels: ArrayInput,
        clear_old_points: bool,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
    ) -> tuple[int, PostprocessedOutput | None]:
        """Add a new object or refine an existing one from point prompts.

        An unknown id starts a new tracker state (the shared one with ``per_obj_inference``)
        unless ``max_num_objects`` is reached, in which case ``(frame_idx, None)`` is
        returned. A known object that was never refined is first split into its own state
        (or removed and re-added with ``use_stateless_refinement``). The object's score is
        set to 1.0, it is cleared from the removed and suppressed sets and marked
        confirmed, and an 'add' or 'refine' action is recorded. On an image-only session an
        empty point list restores the object's original mask input through ``add_new_mask``.
        The resulting video-resolution mask is hole-filled and thresholded at 0 before it
        replaces the object's cached mask.

        Returns:
            ``(frame_idx, output)`` for the prompted frame.

        Raises:
            UnsupportedMultiplexRuntimeError: In distributed runs.
            ValueError: For an out-of-range ``frame_idx`` or inconsistent tracker states.
            TypeError: If a required tracker method is missing.
        """
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity.add_sam2_new_points(distributed)"
            )
        frame_idx = int(frame_idx)
        obj_id = _integer_value(obj_id, name="obj_id")
        num_frames = int(inference_state["num_frames"])
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        tracker_metadata = cast(InitializedTrackerMetadata, inference_state.setdefault("tracker_metadata", {}))
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())
        tracker_metadata.setdefault(
            "obj_id_to_sam2_score_frame_wise",
            defaultdict(dict),
        )
        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            cast(Rank0TrackerMetadata, self._initialize_metadata())["rank0_metadata"],
        )

        self._prepare_backbone_feats(
            inference_state,
            frame_idx,
            reverse=False,
        )

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
        if (
            obj_rank is not None
            and self.use_stateless_refinement
            and not object_has_been_refined
        ):
            self.remove_object(
                inference_state,
                obj_id=obj_id,
                frame_idx=None,
                strict=False,
                is_user_action=False,
            )
            obj_rank = None
        elif obj_rank is not None and not object_has_been_refined:
            if self.rank == obj_rank and not bool(
                getattr(self.tracker, "per_obj_inference", False)
            ):
                tracker_states = self._get_sam2_inference_states_by_obj_ids(
                    inference_state,
                    [obj_id],
                )
                if len(tracker_states) != 1:
                    raise ValueError(
                        "Expected exactly one SAM2 state for obj_id "
                        f"{obj_id}, found {len(tracker_states)}."
                    )
                if len(_object_id_sequence(tracker_states[0].get("obj_ids", []))) > 1:
                    self._extract_object_to_singleton_state(
                        inference_state,
                        obj_id,
                        obj_rank,
                    )
        if obj_rank is None:
            num_prev_obj = int(np.sum(tracker_metadata["num_obj_per_gpu"]))
            if num_prev_obj >= int(self.max_num_objects):
                return frame_idx, None
            obj_rank = 0
            tracker_states_local = inference_state.setdefault(
                "sam2_inference_states",
                [],
            )
            if bool(getattr(self.tracker, "per_obj_inference", False)):
                if not tracker_states_local:
                    tracker_states_local.append(
                        self._init_new_sam2_state(inference_state)
                    )
                sam2_state = tracker_states_local[0]
            else:
                sam2_state = self._init_new_sam2_state(inference_state)
                tracker_states_local.append(sam2_state)
            obj_ids_per_gpu = tracker_metadata["obj_ids_per_gpu"]
            obj_ids_per_gpu[obj_rank] = np.concatenate(
                [
                    np.asarray(obj_ids_per_gpu[obj_rank], dtype=np.int64),
                    np.array([int(obj_id)], dtype=np.int64),
                ]
            )
            tracker_metadata["num_obj_per_gpu"][obj_rank] = int(
                obj_ids_per_gpu[obj_rank].size
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(obj_ids_per_gpu)
            tracker_metadata["max_obj_id"] = max(
                int(tracker_metadata.get("max_obj_id", -1)),
                int(obj_id),
            )
            if "num_buc_per_gpu" in tracker_metadata:
                tracker_metadata["num_buc_per_gpu"][obj_rank] = int(
                    np.ceil(
                        tracker_metadata["num_obj_per_gpu"][obj_rank]
                        / max(int(self.bucket_capacity), 1)
                    )
                )
            rank0_metadata.setdefault("obj_first_frame_idx", {})[int(obj_id)] = (
                frame_idx
            )
            self._sync_hotstart_gpu_metadata_after_addition(
                cast(TrackerMetadata, tracker_metadata),
                frame_idx=frame_idx,
                num_new_objects=1,
            )
            self.add_action_history(
                inference_state,
                "add",
                frame_idx=frame_idx,
                obj_ids=[obj_id],
            )
        else:
            tracker_states = self._get_sam2_inference_states_by_obj_ids(
                inference_state,
                [obj_id],
            )
            if len(tracker_states) != 1:
                raise ValueError(
                    "Expected exactly one SAM2 state for obj_id "
                    f"{obj_id}, found {len(tracker_states)}."
                )
            sam2_state = tracker_states[0]
            self.add_action_history(
                inference_state,
                "refine",
                frame_idx=frame_idx,
                obj_ids=[obj_id],
            )

        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        frame_scores_for_key(tracker_metadata, "obj_id_to_sam2_score_frame_wise")[frame_idx][obj_id] = (
            mx.array(1.0, dtype=mx.float32)
        )
        rank0_metadata.setdefault("removed_obj_ids", set()).discard(obj_id)
        for suppressed_obj_ids in rank0_metadata.setdefault(
            "suppressed_obj_ids",
            defaultdict(set),
        ).values():
            suppressed_obj_ids.discard(obj_id)
        confirmation = rank0_metadata.get("masklet_confirmation")
        if confirmation is not None:
            obj_ids_all_gpu = np.asarray(
                tracker_metadata["obj_ids_all_gpu"],
                dtype=np.int64,
            )
            obj_indices = np.where(obj_ids_all_gpu == int(obj_id))[0]
            if obj_indices.size > 0:
                obj_idx = int(obj_indices[0])
                if obj_idx < len(confirmation["status"]):
                    confirmation["status"][obj_idx] = (
                        MaskletConfirmationStatus.CONFIRMED.value
                    )
                    confirmation["consecutive_det_num"][obj_idx] = (
                        self.masklet_confirmation_consecutive_det_thresh
                    )

        should_restore_original_mask = _point_count(points) == 0 and bool(
            inference_state.get("is_image_only", False)
        )
        mask_input = (
            self._get_mask_input(sam2_state, frame_idx, obj_id)
            if should_restore_original_mask
            else None
        )
        if mask_input is not None and 0 in tuple(cast(_ShapedValue, mask_input).shape):
            mask_input = None
        add_new_mask: object = getattr(self.tracker, "add_new_mask", None)
        if mask_input is not None and callable(add_new_mask):
            clear_points: object = getattr(
                self.tracker,
                "clear_all_points_in_frame",
                None,
            )
            if not callable(clear_points):
                raise TypeError("tracker.clear_all_points_in_frame must be callable")
            cast(_ClearPointsCall, clear_points)(
                sam2_state,
                frame_idx,
                obj_id,
                need_output=False,
            )
            prompt_result = cast(_AddMaskCall, add_new_mask)(
                sam2_state, frame_idx, obj_id, mask_input
            )
        else:
            add_new_points: object = getattr(self.tracker, "add_new_points", None)
            if not callable(add_new_points):
                raise TypeError("tracker.add_new_points must be callable")
            prompt_result = cast(_KeywordCall, add_new_points)(
                inference_state=sam2_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=clear_old_points,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=use_prev_mem_frame,
            )
        frame_idx, obj_ids, _low_res_masks, video_res_masks = _tracker_prompt_result(
            prompt_result
        )
        if video_res_masks is not None and len(video_res_masks) > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks,
                fill_hole_area=self.fill_hole_area,
                sprinkle_removal_area=self.sprinkle_removal_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
        self._sync_sam2_inputs_before_point_preflight(sam2_state)
        preflight: object = getattr(
            self.tracker,
            "propagate_in_video_preflight",
            None,
        )
        if not callable(preflight):
            raise TypeError("tracker.propagate_in_video_preflight must be callable")
        cast(_PreflightCall, preflight)(sam2_state, run_mem_encoder=True)
        if not inference_state.get("is_image_only", False):
            self.clear_detector_added_cond_frame_in_sam2(sam2_state, obj_id, frame_idx)

        if video_res_masks is None or len(video_res_masks) == 0:
            refined_obj_id_to_mask = None
        else:
            if obj_id not in obj_ids:
                raise ValueError(
                    f"Tracker add_new_points did not return obj_id {obj_id}."
                )
            new_mask_data = video_res_masks[list(obj_ids).index(obj_id)] > 0.0
            refined_obj_id_to_mask = {obj_id: new_mask_data}

        obj_id_to_mask = self._build_sam2_output(
            inference_state,
            frame_idx,
            refined_obj_id_to_mask,
        )
        suppressed_obj_ids = rank0_metadata.get("suppressed_obj_ids", {}).get(
            frame_idx,
            set(),
        )
        self._cache_frame_outputs(
            inference_state,
            frame_idx,
            obj_id_to_mask,
            suppressed_obj_ids=suppressed_obj_ids,
        )
        out_scores = {
            cached_obj_id: tracker_metadata["obj_id_to_score"].get(cached_obj_id, 0.0)
            for cached_obj_id in obj_id_to_mask
        }
        return frame_idx, self._postprocess_output(
            inference_state,
            {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": out_scores,
                "obj_id_to_sam2_score": frame_scores_for_key(
                    tracker_metadata, "obj_id_to_sam2_score_frame_wise"
                )[frame_idx],
            },
            suppressed_obj_ids=suppressed_obj_ids,
        )

    def _sync_sam2_inputs_before_point_preflight(
        self, sam2_state: TrackerState
    ) -> None:
        """Make the tracker state's mask inputs and consolidated frame indices consistent before preflight.

        Mask inputs on frames without point inputs are dropped, and
        ``consolidated_frame_inds`` is rebuilt from the frames that have inputs and are not
        only temporary outputs, split into conditioning and non-conditioning frames.
        """
        mask_inputs_per_obj = sam2_state.get("mask_inputs_per_obj")
        point_inputs_per_obj = sam2_state.get("point_inputs_per_obj")
        if not isinstance(mask_inputs_per_obj, Mapping) or not isinstance(
            point_inputs_per_obj,
            Mapping,
        ):
            return

        mask_inputs_by_obj = cast(Mapping[object, object], mask_inputs_per_obj)
        point_inputs_by_obj = cast(Mapping[object, object], point_inputs_per_obj)
        for obj_idx, mask_inputs_per_frame in list(mask_inputs_by_obj.items()):
            if not isinstance(mask_inputs_per_frame, dict):
                continue
            mutable_mask_inputs = cast(dict[object, object], mask_inputs_per_frame)
            point_inputs_per_frame = point_inputs_by_obj.get(obj_idx, {})
            if isinstance(point_inputs_per_frame, Mapping):
                point_frame_indices = set(
                    cast(Mapping[object, object], point_inputs_per_frame)
                )
            else:
                point_frame_indices: set[object] = set()
            for frame_id in list(mutable_mask_inputs):
                if frame_id not in point_frame_indices:
                    mutable_mask_inputs.pop(frame_id, None)

        input_frames: set[int] = set()
        for point_inputs_per_frame in point_inputs_by_obj.values():
            if isinstance(point_inputs_per_frame, Mapping):
                point_frames = cast(Mapping[object, object], point_inputs_per_frame)
                input_frames.update(
                    _integer_value(frame_id, name="point input frame index")
                    for frame_id in point_frames
                )
        for mask_inputs_per_frame in mask_inputs_by_obj.values():
            if isinstance(mask_inputs_per_frame, Mapping):
                mask_frames = cast(Mapping[object, object], mask_inputs_per_frame)
                input_frames.update(
                    _integer_value(frame_id, name="mask input frame index")
                    for frame_id in mask_frames
                )

        temp_frames: set[int] = set()
        temp_outputs_per_obj = _mapping_view(
            sam2_state.get("temp_output_dict_per_obj", {}),
            name="temp_output_dict_per_obj",
        )
        for temp_outputs in temp_outputs_per_obj.values():
            if not isinstance(temp_outputs, Mapping):
                continue
            temp_output_mapping = cast(Mapping[object, object], temp_outputs)
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                frame_outputs = temp_output_mapping.get(storage_key, {})
                if isinstance(frame_outputs, Mapping):
                    frame_output_mapping = cast(Mapping[object, object], frame_outputs)
                    temp_frames.update(
                        _integer_value(frame_id, name="temporary output frame index")
                        for frame_id in frame_output_mapping
                    )

        previous_input_frames = input_frames - temp_frames
        output_dict = _mapping_view(
            sam2_state.get("output_dict", {}),
            name="output_dict",
        )
        cond_outputs = _mapping_view(
            output_dict.get("cond_frame_outputs", {}),
            name="cond_frame_outputs",
        )
        cond_frame_outputs: set[int] = set()
        non_cond_frame_outputs: set[int] = set()
        for frame_id in previous_input_frames:
            if frame_id in cond_outputs:
                cond_frame_outputs.add(frame_id)
            else:
                non_cond_frame_outputs.add(frame_id)
        sam2_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": cond_frame_outputs,
            "non_cond_frame_outputs": non_cond_frame_outputs,
        }

    def _has_object_been_refined(
        self,
        inference_state: MultiplexInferenceState,
        obj_id: TrackerObjectId,
    ) -> bool:
        """Whether the action history records an 'add' or 'refine' of ``obj_id``."""
        for action in inference_state.get("action_history", []):
            if action["type"] in {"add", "refine"} and action.get("obj_ids"):
                if obj_id in cast(Sequence[TrackerObjectId], action["obj_ids"]):
                    return True
        return False

    def parse_action_history_for_propagation(
        self,
        inference_state: MultiplexInferenceState,
    ) -> tuple[str, list[TrackerObjectId] | None]:
        """Choose the propagation type and objects from the action history.

        Returns ``(propagation_type, obj_ids)``. A history of only a cancel gives
        ``propagation_full``; a trailing cancel repeats the action before it (skipping a
        fetch). Otherwise see ``_parse_action_history_for_propagation``.
        """
        action_history = inference_state.setdefault("action_history", [])
        if (
            len(action_history) == 1
            and action_history[0]["type"] == "propagation_cancel"
        ):
            return "propagation_full", None
        if (
            len(action_history) >= 2
            and action_history[-1]["type"] == "propagation_cancel"
        ):
            action_before_cancellation = action_history[-2]
            if (
                action_before_cancellation["type"] == "propagation_fetch"
                and len(action_history) >= 3
            ):
                action_before_cancellation = action_history[-3]
            return (
                action_before_cancellation["type"],
                action_before_cancellation.get("obj_ids", None),
            )
        return self._parse_action_history_for_propagation(
            action_history,
            int(inference_state["num_frames"]),
        )

    def _parse_action_history_for_propagation(
        self,
        action_history: list[TrackingAction],
        num_frames: int,
    ) -> tuple[str, list[TrackerObjectId] | None]:
        """Choose the propagation type for a history that does not end in a cancel.

        An empty history gives ``propagation_full``. After a fetch, ``propagation_fetch``.
        After a full or partial propagation, ``propagation_fetch`` if it was preceded by
        another propagation or started at the first or last frame, else the same type and
        objects again. After edits, ``propagation_partial`` over the sorted ids added or
        refined since the last propagation, or ``propagation_fetch`` when there are none.
        """
        if len(action_history) == 0:
            return "propagation_full", None

        last_action = action_history[-1]
        if "propagation" in last_action["type"]:
            if last_action["type"] == "propagation_fetch":
                return "propagation_fetch", None
            if last_action["type"] in {"propagation_partial", "propagation_full"}:
                has_previous_propagation = len(action_history) > 1 and action_history[
                    -2
                ]["type"] in {"propagation_partial", "propagation_full"}
                if has_previous_propagation or last_action["frame_idx"] in {
                    0,
                    num_frames - 1,
                }:
                    return "propagation_fetch", None
                return last_action["type"], last_action["obj_ids"]

        obj_ids: list[TrackerObjectId] = []
        for action in reversed(action_history):
            if "propagation" in action["type"]:
                break
            if action["type"] in {"add", "refine"}:
                obj_ids.extend(cast(Sequence[TrackerObjectId], action["obj_ids"]))
        if obj_ids:
            return "propagation_partial", sorted(set(obj_ids))
        return "propagation_fetch", None

    def fetch_and_process_single_frame_results(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
    ) -> tuple[int, PostprocessedOutput]:
        """Return the public output of ``frame_idx`` from the cached frame outputs.

        Raises:
            ValueError: If nothing is cached for the frame.
        """
        cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
        if frame_idx not in cached_frame_outputs:
            raise ValueError(
                f"No cached multiplex output is available for frame_idx={frame_idx}."
            )
        tracker_metadata = inference_state.get("tracker_metadata", {})
        rank0_metadata = tracker_metadata.get("rank0_metadata", {})
        suppressed_by_frame = rank0_metadata.get("suppressed_obj_ids", {})
        obj_id_to_sam2_score = tracker_metadata.get(
            "obj_id_to_sam2_score_frame_wise",
            {},
        ).get(frame_idx, {})
        obj_id_to_score = tracker_metadata.get("obj_id_to_score", {})
        frame_scores = {
            obj_id: obj_id_to_sam2_score.get(
                obj_id,
                obj_id_to_score.get(obj_id, 0.0),
            )
            for obj_id in cached_frame_outputs[frame_idx]
        }
        return (
            frame_idx,
            self._postprocess_output(
                inference_state,
                {
                    "obj_id_to_mask": cached_frame_outputs[frame_idx],
                    "obj_id_to_score": {
                        obj_id: obj_id_to_score.get(obj_id, frame_scores[obj_id])
                        for obj_id in cached_frame_outputs[frame_idx]
                    },
                    "obj_id_to_sam2_score": frame_scores,
                },
                suppressed_obj_ids=suppressed_by_frame.get(frame_idx, set()),
            ),
        )

    def cancel_propagation(self, inference_state: MultiplexInferenceState) -> None:
        """Record a ``propagation_cancel`` action and reset the generator state if the session has one."""
        self.add_action_history(
            inference_state,
            action_type="propagation_cancel",
            obj_ids=None,
            frame_idx=None,
        )
        if "generator_state" in inference_state:
            inference_state["generator_state"] = self._new_generator_state()
        return None

    def remove_object(
        self,
        inference_state: MultiplexInferenceState,
        obj_id: TrackerObjectId,
        frame_idx: int | None = 0,
        strict: bool = False,
        is_user_action: bool = False,
    ) -> tuple[int | None, PropagationOutput | None]:
        """Remove an object as the base class does; a user action is also recorded as a 'remove' in the history."""
        result = super().remove_object(
            inference_state,
            obj_id=obj_id,
            frame_idx=frame_idx,
            strict=strict,
            is_user_action=is_user_action,
        )
        if is_user_action:
            self.add_action_history(
                inference_state,
                action_type="remove",
                frame_idx=frame_idx,
                obj_ids=[int(obj_id)],
            )
        return result

    def _get_sam2_inference_states_by_obj_ids(
        self,
        inference_state: MultiplexInferenceState,
        obj_ids: Sequence[TrackerObjectId],
    ) -> list[TrackerState]:
        """Tracker states that contain at least one of ``obj_ids``."""
        requested_obj_ids = set(obj_ids)
        tracker_states = _require_tracker_state_list(
            inference_state.get("sam2_inference_states", [])
        )
        return [
            state
            for state in tracker_states
            if requested_obj_ids & set(_object_id_sequence(state.get("obj_ids", [])))
        ]

    def _lookup_existing_obj_idx(
        self,
        inference_state: TrackerState,
        obj_id: TrackerObjectId,
    ) -> int | None:
        """Index of ``obj_id`` inside a tracker state (from ``obj_id_to_idx`` or ``obj_ids``), or None."""
        obj_id_to_idx = cast(Mapping[TrackerObjectId, int], inference_state.get("obj_id_to_idx", {}))
        if obj_id in obj_id_to_idx:
            return int(obj_id_to_idx[obj_id])
        if "obj_ids" in inference_state:
            obj_id_matches = np.flatnonzero(
                np.asarray(inference_state["obj_ids"]) == obj_id
            )
            if obj_id_matches.size:
                return int(obj_id_matches[0])
        return None

    def _get_mask_input(
        self,
        inference_state: TrackerState,
        frame_idx: int,
        obj_id: TrackerObjectId,
    ) -> object:
        """Stored mask input of the object on ``frame_idx`` with leading singleton axes removed, or None."""
        obj_idx = self._lookup_existing_obj_idx(inference_state, obj_id)
        if obj_idx is None:
            return None

        mask_inputs = cast(Mapping[int, Mapping[int, object]], inference_state.get("mask_inputs_per_obj", {}))
        mask_inputs_per_frame = mask_inputs.get(obj_idx, {})
        if frame_idx not in mask_inputs_per_frame:
            return None

        mask_input = mask_inputs_per_frame[frame_idx]
        if _is_mlx_array(mask_input):
            if mask_input.ndim == 4 and mask_input.shape[:2] == (1, 1):
                return mask_input[0, 0]
            if mask_input.ndim == 3 and mask_input.shape[0] == 1:
                return mask_input[0]
            return mask_input

        mask_input_np = np.asarray(mask_input)
        if mask_input_np.ndim == 4 and mask_input_np.shape[:2] == (1, 1):
            return mask_input_np[0, 0]
        if mask_input_np.ndim == 3 and mask_input_np.shape[0] == 1:
            return mask_input_np[0]
        return mask_input

    @overload
    def _convert_low_res_mask_to_video_res(
        self, low_res_mask: None, *, orig_vid_height: int, orig_vid_width: int
    ) -> None: ...

    @overload
    def _convert_low_res_mask_to_video_res(
        self, low_res_mask: TrackerArray, *, orig_vid_height: int, orig_vid_width: int
    ) -> mx.array: ...

    def _convert_low_res_mask_to_video_res(
        self,
        low_res_mask: TrackerArray | None,
        *,
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> mx.array | None:
        """Resize a present low-resolution mask to a boolean video-frame mask."""
        if low_res_mask is None:
            return None
        return super()._convert_low_res_mask_to_video_res(
            low_res_mask,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
        )

    def _gather_obj_id_to_mask_across_gpus(
        self,
        inference_state: MultiplexInferenceState,
        obj_id_to_mask_local: Mapping[TrackerObjectId, TrackerArray],
    ) -> mx.array:
        """Stack local masks in metadata ID order, filling absent objects."""
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "_gather_obj_id_to_mask_across_gpus(distributed)"
            )

        tracker_metadata = inference_state.get("tracker_metadata", {})
        obj_ids_per_gpu = tracker_metadata.get("obj_ids_per_gpu", [])
        if self.rank >= len(obj_ids_per_gpu):
            raise ValueError(
                f"rank={self.rank} is out of range for obj_ids_per_gpu "
                f"with {len(obj_ids_per_gpu)} entries."
            )

        low_res_mask_size: object = getattr(
            self.tracker,
            "low_res_mask_size",
            None,
        )
        h_mask = w_mask = _integer_value(
            low_res_mask_size,
            name="tracker.low_res_mask_size",
        )
        low_res_masks_local: list[mx.array] = []
        local_obj_ids = _tracker_object_ids(obj_ids_per_gpu[self.rank])
        for obj_id in local_obj_ids:
            if obj_id in obj_id_to_mask_local:
                low_res_mask = obj_id_to_mask_local[obj_id]
                low_res_mask_mx = (
                    low_res_mask
                    if _is_mlx_array(low_res_mask)
                    else mlx_array(low_res_mask)
                )
                if low_res_mask_mx.shape != (h_mask, w_mask):
                    raise ValueError(
                        "Each low-res mask must have shape "
                        f"({h_mask}, {w_mask}), got {low_res_mask_mx.shape} "
                        f"for obj_id {obj_id}."
                    )
                low_res_masks_local.append(low_res_mask_mx.astype(mx.float32))
            else:
                low_res_masks_local.append(
                    mx.full((h_mask, w_mask), NO_OBJ_SCORE, dtype=mx.float32)
                )

        if not low_res_masks_local:
            return mx.zeros((0, h_mask, w_mask), dtype=mx.float32)
        return mx.stack(low_res_masks_local, axis=0)

    def _propogate_tracker_one_frame_local_gpu(
        self,
        tracker_states_local: Sequence[dict[str, object]],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = True,
    ) -> tuple[list[TrackerObjectId], list[mx.array], list[mx.array]]:
        """Propagate every given tracker state by one frame.

        Returns three lists over all objects in state order: ids, float32 ``(H, W)`` mask
        logits and tracker object-score logits (scalar when single-valued).

        Raises:
            ValueError: If the tracker returns another frame index, mismatched counts or
                mask shapes, or an id twice.
        """
        obj_ids_local: list[TrackerObjectId] = []
        low_res_masks_local: list[mx.array] = []
        sam2_scores_local: list[mx.array] = []
        seen_obj_ids: set[TrackerObjectId] = set()
        propagate = require_propagate_call(self.tracker)

        for sam2_state in tracker_states_local:
            frame_results = propagate(
                sam2_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                propagate_preflight=False,
            )
            for result in frame_results:
                (
                    result_frame_idx,
                    state_obj_ids,
                    state_low_res_masks,
                    _state_video_res_masks,
                    state_sam2_scores,
                ) = result
                if result_frame_idx != frame_idx:
                    raise ValueError(
                        "Tracker partial propagation returned frame_idx="
                        f"{result_frame_idx}, expected {frame_idx}."
                    )

                state_obj_ids_list = _tracker_object_ids(state_obj_ids)

                state_low_res_masks_mx = (
                    state_low_res_masks
                    if _is_mlx_array(state_low_res_masks)
                    else mlx_array(state_low_res_masks)
                )
                state_sam2_scores_mx = (
                    state_sam2_scores
                    if _is_mlx_array(state_sam2_scores)
                    else mlx_array(state_sam2_scores)
                )
                if state_low_res_masks_mx.shape[0] != len(state_obj_ids_list):
                    raise ValueError(
                        "Tracker low-res mask batch must match obj_ids; got "
                        f"{state_low_res_masks_mx.shape[0]} masks for "
                        f"{len(state_obj_ids_list)} object ids."
                    )
                if state_sam2_scores_mx.shape[0] != len(state_obj_ids_list):
                    raise ValueError(
                        "Tracker score batch must match obj_ids; got "
                        f"{state_sam2_scores_mx.shape[0]} scores for "
                        f"{len(state_obj_ids_list)} object ids."
                    )

                for obj_idx, obj_id in enumerate(state_obj_ids_list):
                    if obj_id in seen_obj_ids:
                        raise ValueError(
                            f"Duplicate obj_id {obj_id} in local SAM2 states."
                        )
                    seen_obj_ids.add(obj_id)
                    low_res_mask = state_low_res_masks_mx[obj_idx]
                    if low_res_mask.ndim == 3 and low_res_mask.shape[0] == 1:
                        low_res_mask = low_res_mask[0]
                    if low_res_mask.ndim != 2:
                        raise ValueError(
                            "Tracker low-res mask entries must have shape (H, W) "
                            f"or (1, H, W), got {low_res_mask.shape}."
                        )
                    obj_ids_local.append(obj_id)
                    low_res_masks_local.append(low_res_mask.astype(mx.float32))
                    sam2_score = state_sam2_scores_mx[obj_idx]
                    if int(np.prod(sam2_score.shape)) == 1:
                        sam2_score = mx.reshape(sam2_score, ())
                    sam2_scores_local.append(sam2_score)

        return obj_ids_local, low_res_masks_local, sam2_scores_local

    def clear_detector_added_cond_frame_in_sam2(
        self,
        sam2_state: TrackerState,
        obj_id: TrackerObjectId,
        refined_frame_idx: int,
    ) -> None:
        """Clear detector-added conditioning masks near a point refinement.

        Mask inputs without matching point inputs, within
        ``refinement_detector_cond_frame_removal_window`` frames of ``refined_frame_idx``,
        are cleared for every object in the state.

        Raises:
            TypeError: If the tracker's ``clear_all_points_in_frame`` is not callable.
        """
        obj_idx = self._lookup_existing_obj_idx(sam2_state, obj_id)
        if obj_idx is None:
            return

        mask_inputs_per_obj = cast(Mapping[int, Mapping[int, object]], sam2_state.get("mask_inputs_per_obj", {}))
        mask_inputs_per_frame = mask_inputs_per_obj.get(obj_idx, {})
        if not mask_inputs_per_frame:
            return

        point_inputs_per_obj = cast(Mapping[int, Mapping[int, object]], sam2_state.get("point_inputs_per_obj", {}))
        point_frame_indices = set(point_inputs_per_obj.get(obj_idx, {}))
        window = int(self.refinement_detector_cond_frame_removal_window)
        frame_indices_to_clear = [
            int(frame_idx)
            for frame_idx in mask_inputs_per_frame
            if frame_idx not in point_frame_indices
            and abs(int(frame_idx) - int(refined_frame_idx)) <= window
        ]

        obj_ids_on_state = list(cast(Mapping[TrackerObjectId, int], sam2_state.get("obj_id_to_idx", {})).keys())
        clear_points: object = getattr(
            self.tracker,
            "clear_all_points_in_frame",
            None,
        )
        if not callable(clear_points):
            raise TypeError("tracker.clear_all_points_in_frame must be callable")
        for frame_idx in frame_indices_to_clear:
            for obj_id_to_clear in obj_ids_on_state:
                cast(_ClearPointsCall, clear_points)(
                    sam2_state,
                    frame_idx,
                    _integer_value(obj_id_to_clear, name="tracker state obj_id"),
                    need_output=False,
                )

    def propagate_in_video(
        self,
        inference_state: MultiplexInferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        output_prob_thresh: float = 0.5,
        compute_stability_score: bool = False,
        is_instance_processing: bool = False,
        *,
        is_last_batch: bool = True,
    ) -> Iterator[PropagationYield]:
        """Propagate according to the action history, yielding ``(frame_idx, output)``.

        The type comes from ``parse_action_history_for_propagation`` and is recorded as a
        new action. ``propagation_full`` re-runs detection and tracking,
        ``propagation_fetch`` yields cached outputs and ``propagation_partial`` re-propagates
        only the tracker states of the edited objects and merges them into the cached
        masks. ``output_prob_thresh`` and ``compute_stability_score`` are unused.

        Raises:
            UnsupportedMultiplexRuntimeError: For partial propagation in distributed runs.
            ValueError: For an unexpected propagation type or missing object ids.
        """
        del output_prob_thresh, compute_stability_score
        propagation_type, obj_ids = self.parse_action_history_for_propagation(
            inference_state
        )
        self.add_action_history(
            inference_state,
            action_type=propagation_type,
            obj_ids=obj_ids,
            frame_idx=start_frame_idx,
        )

        if propagation_type == "propagation_full":
            yield from self._propagate_in_video_impl(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
                is_instance_processing=is_instance_processing,
                flush_hotstart_at_end=is_last_batch,
            )
            return

        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        if propagation_type == "propagation_fetch":
            for frame_idx in processing_order:
                if self.rank == 0:
                    yield self.fetch_and_process_single_frame_results(
                        inference_state,
                        frame_idx,
                    )
                else:
                    yield frame_idx, DUMMY_OUTPUT
            return

        if propagation_type != "propagation_partial":
            raise ValueError(f"Unexpected propagation_type={propagation_type!r}.")
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "propagate_in_video(propagation_partial-distributed)"
            )
        if obj_ids is None:
            raise ValueError("propagation_partial requires obj_ids.")

        tracker_states_local = self._get_sam2_inference_states_by_obj_ids(
            inference_state,
            obj_ids,
        )
        preflight: object = getattr(
            self.tracker,
            "propagate_in_video_preflight",
            None,
        )
        if not callable(preflight):
            raise TypeError("tracker.propagate_in_video_preflight must be callable")
        for sam2_state in tracker_states_local:
            cast(_PreflightCall, preflight)(
                sam2_state,
                run_mem_encoder=True,
            )

        tracker_metadata = inference_state["tracker_metadata"]
        framewise_sam2_scores = tracker_metadata.setdefault(
            "obj_id_to_sam2_score_frame_wise",
            defaultdict(dict),
        )
        for frame_idx in processing_order:
            self._prepare_backbone_feats(
                inference_state,
                frame_idx,
                reverse=reverse,
            )
            obj_ids_local, low_res_masks_local, sam2_scores_local = (
                self._propogate_tracker_one_frame_local_gpu(
                    tracker_states_local,
                    frame_idx=frame_idx,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
            )
            refined_obj_data = {
                cast(int, obj_id): (sam2_scores_local[obj_idx], low_res_masks_local[obj_idx])
                for obj_idx, obj_id in enumerate(obj_ids_local)
                if obj_id in obj_ids
            }
            frame_scores = framewise_sam2_scores.setdefault(frame_idx, {})
            for obj_id, (refined_score, _) in refined_obj_data.items():
                frame_scores[obj_id] = refined_score

            if self.rank != 0:
                yield frame_idx, DUMMY_OUTPUT
                continue

            refined_obj_id_to_mask = {
                obj_id: self._convert_low_res_mask_to_video_res(
                    refined_mask_low_res,
                    orig_vid_height=int(inference_state["orig_height"]),
                    orig_vid_width=int(inference_state["orig_width"]),
                )
                for obj_id, (_, refined_mask_low_res) in refined_obj_data.items()
            }
            obj_id_to_mask = self._build_sam2_output(
                inference_state,
                frame_idx,
                refined_obj_id_to_mask,
            )
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})
            suppressed_obj_ids = rank0_metadata.get("suppressed_obj_ids", {}).get(
                frame_idx,
                set(),
            )
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            obj_id_to_score = tracker_metadata.get("obj_id_to_score", {})
            yield (
                frame_idx,
                self._postprocess_output(
                    inference_state,
                    {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": {
                            obj_id: obj_id_to_score.get(
                                obj_id,
                                frame_scores.get(obj_id, 0.0),
                            )
                            for obj_id in obj_id_to_mask
                        },
                        "obj_id_to_sam2_score": frame_scores,
                    },
                    suppressed_obj_ids=suppressed_obj_ids,
                ),
            )
