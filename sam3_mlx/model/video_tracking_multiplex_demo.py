"""Interactive session API for the multiplex video tracker (upstream demo interface).

``VideoTrackingMultiplexDemo`` adds the stateful workflow of the SAM 3 demo to
``VideoTrackingDynamicMultiplex``: ``init_state`` loads a video (or sizes an empty
session), ``add_new_points`` and ``add_new_masks`` prompt objects on a frame,
``propagate_in_video`` tracks all or selected objects through the video, and
``clear_all_points_in_*`` and ``remove_object(s)`` edit the session.

The session is a ``MultiplexDemoState`` dict. It holds the packed per-frame outputs
(``output_dict``, one row per object in data space) and a per-object view
(``output_dict_per_obj``). Object ids are caller-chosen and map to dense indices in order
of first use. Masks returned to callers are logits (foreground where > 0) at the original
video size, shaped ``(num_objects, 1, video_height, video_width)``. CPU offloading of
frames or state and torchcodec decoding are not supported and raise
``UnsupportedMultiplexRuntimeError``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import wraps
from inspect import signature
from numbers import Integral, Number
from os import PathLike
from typing import TYPE_CHECKING, Protocol, Unpack, cast, overload

import mlx.core as mx
from PIL import Image

from sam3_mlx.mlx_runtime import evaluate_boundary, is_mlx_array, to_numpy
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.io_utils import (
    AsyncVideoFileFrames, NormalizedVideoFrameSequence, load_resource_as_video_frames,
)
from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    raise_unsupported_multiplex_runtime,
)
from sam3_mlx.model.multiplex_backbone_contract import PreparedBackboneOutput, RawBackboneOutput
from sam3_mlx.model.multiplex_tracking_state import MultiplexDemoState
from sam3_mlx.model.sam3_multiplex_base import FeatureCache
from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3_mlx.model.video_tracking_multiplex import (
    NO_OBJ_SCORE,
    DynamicMultiplexTrackingConstructorArgs,
    StageOutput,
    VideoTrackingDynamicMultiplex,
    concat_points,
    _reshape_spatial_feature,
)

if TYPE_CHECKING:
    from sam3_mlx.model.sam3_tracking_predictor import ArrayInput


# Propagation yields (frame_idx, object ids, low-res mask logits, video-res mask logits);
# the scored variant appends the object score logits.
type TrackingFrame = tuple[int, list[object], mx.array, mx.array]
type ScoredTrackingFrame = tuple[int, list[object], mx.array, mx.array, mx.array]


class _ImageSizeModel(Protocol):
    """A model exposing the side length ``image_size`` (pixels) of its square input."""

    @property
    def image_size(self) -> int: ...


def _eval_tree(*values: object) -> None:
    """Evaluate every MLX array found in the given values, descending into mappings, lists and tuples."""

    def _eval_value(value: object) -> None:
        if is_mlx_array(value):
            evaluate_boundary(value)
        elif isinstance(value, Mapping):
            for item in cast(Mapping[object, object], value).values():
                _eval_value(item)
        elif isinstance(value, (list, tuple)):
            for item in cast(Sequence[object], value):
                _eval_value(item)

    for value in values:
        _eval_value(value)


def _coerce_obj_id_list(obj_ids: object) -> list[object]:
    """Return object ids as a list.

    A string or bytes value is a single id, an MLX array is flattened to Python scalars,
    any other iterable is listed and a non-iterable is a single id.
    """
    if isinstance(obj_ids, (str, bytes)):
        return [obj_ids]
    if is_mlx_array(obj_ids):
        evaluate_boundary(obj_ids)
        return list[object](to_numpy(obj_ids).reshape(-1).tolist())
    if isinstance(obj_ids, Iterable):
        return list(cast(Iterable[object], obj_ids))
    return [obj_ids]


@overload
def _take_indices(value: None, indices: list[int]) -> None: ...


@overload
def _take_indices(value: mx.array, indices: list[int]) -> mx.array: ...


@overload
def _take_indices(value: list[mx.array], indices: list[int]) -> list[mx.array]: ...


def _take_indices(
    value: mx.array | list[mx.array] | None, indices: list[int]
) -> mx.array | list[mx.array] | None:
    """Select rows ``indices`` along axis 0 of an array or of each array in a list; None stays None."""
    if value is None:
        return None
    index = mx.array(indices, dtype=mx.int64)
    if isinstance(value, list):
        return [mx.take(item, index, axis=0) for item in value]
    return mx.take(value, index, axis=0)


def _replace_indices(
    value: mx.array, indices: list[int], replacement: mx.array
) -> mx.array:
    """Return ``value`` with the rows ``indices`` (axis 0) replaced by the rows of ``replacement``.

    Raises:
        ValueError: If ``indices`` and ``replacement`` differ in length.
    """
    if len(indices) == 0:
        return value
    if len(indices) != replacement.shape[0]:
        raise ValueError("indices length must match replacement batch")
    replacements: dict[int, mx.array] = {
        int(index): row for index, row in zip(indices, replacement)
    }
    rows: list[mx.array] = []
    for row_idx in range(value.shape[0]):
        if row_idx in replacements:
            rows.append(replacements[row_idx][None])
        else:
            rows.append(value[row_idx : row_idx + 1])
    return mx.concatenate(rows, axis=0)


def _init_multiplex_demo_state(
    *,
    video_height: int,
    video_width: int,
    num_frames: int,
    images: mx.array | Sequence[ArrayInput] | NormalizedVideoFrameSequence | None = None,
    cached_features: object | None = None,
) -> MultiplexDemoState:
    """Create an empty session state for a video of ``num_frames`` frames.

    ``video_height`` and ``video_width`` are the original frame size in pixels, which is
    the resolution of the masks returned to callers. ``images`` holds the normalized
    model-resolution frames (None when features come from ``cached_features``).
    ``cached_features`` is a dict keyed by frame index and is kept by reference.

    Raises:
        ValueError: If a size or the frame count is not a positive integer.
        TypeError: If ``cached_features`` is not a dict.
    """
    if (
        isinstance(video_height, bool)
        or isinstance(video_width, bool)
        or not isinstance(video_height, Integral)
        or not isinstance(video_width, Integral)
        or video_height <= 0
        or video_width <= 0
    ):
        raise ValueError("video_height and video_width must be positive integers.")
    if (
        isinstance(num_frames, bool)
        or not isinstance(num_frames, Integral)
        or num_frames <= 0
    ):
        raise ValueError("num_frames must be a positive integer.")
    if cached_features is None:
        cached_features = {}
    if not isinstance(cached_features, dict):
        raise TypeError("cached_features must be a dict keyed by frame index.")
    cached_features = cast(FeatureCache, cached_features)
    return {
        "images": images,
        "num_frames": int(num_frames),
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": int(video_height),
        "video_width": int(video_width),
        "device": "mlx",
        "storage_device": "mlx",
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": cached_features,
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        },
        "first_ann_frame_idx": None,
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        },
        "tracking_has_started": False,
        "frames_already_tracked": {},
        "multiplex_state": None,
        "user_refined_frames_per_obj": {},
    }


def _load_multiplex_demo_state_from_resource(
    model: _ImageSizeModel,
    *,
    video_path: str | PathLike[str] | Sequence[Image.Image],
    cached_features: object = None,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
    async_loading_frames: bool = False,
    use_torchcodec: bool = False,
    use_cv2: bool = False,
) -> MultiplexDemoState:
    """Load a video file, image folder or list of PIL frames and create the session state.

    Frames are resized to ``model.image_size`` and normalized by
    ``load_resource_as_video_frames`` (OpenCV loader); the original frame size is kept
    for the output masks and the frame source is stored as ``state['frames']``.
    ``use_cv2`` is ignored.

    Raises:
        UnsupportedMultiplexRuntimeError: If CPU offloading or torchcodec is requested.
        ValueError: If the loaded frames have no normalized images.
    """
    if offload_video_to_cpu or offload_state_to_cpu:
        raise_unsupported_multiplex_runtime(
            "VideoTrackingMultiplexDemo.init_state(offload)"
        )
    if use_torchcodec:
        raise_unsupported_multiplex_runtime(
            "VideoTrackingMultiplexDemo.init_state(use_torchcodec=True)"
        )
    del use_cv2
    loaded_frames = load_resource_as_video_frames(
        resource_path=video_path,
        image_size=int(model.image_size),
        offload_video_to_cpu=False,
        async_loading_frames=async_loading_frames,
        video_loader_type="cv2",
    )
    images = (
        NormalizedVideoFrameSequence(loaded_frames)
        if isinstance(loaded_frames, AsyncVideoFileFrames)
        else loaded_frames.images
    )
    if images is None:
        raise ValueError("Loaded video frames must expose normalized MLX images.")
    try:
        state = _init_multiplex_demo_state(
            video_height=int(loaded_frames.orig_height),
            video_width=int(loaded_frames.orig_width),
            num_frames=len(loaded_frames),
            images=images,
            cached_features=cached_features,
        )
    except (TypeError, ValueError):
        # Release the frame reader (e.g. background loader) when the state cannot be built.
        close = getattr(loaded_frames, "close", None)
        if callable(close):
            close()
        raise
    state["frames"] = loaded_frames
    return state


class VideoTrackingMultiplexDemo(VideoTrackingDynamicMultiplex):
    """Dynamic multiplex tracker with the interactive session API of the upstream demo.

    Args:
        clear_non_cond_mem_around_input: When propagating, drop non-conditioning outputs
            within ``memory_temporal_stride_for_eval * num_maskmem`` frames of each
            conditioning frame so those frames are tracked again.
        clear_non_cond_mem_for_multi_obj: Also do that when tracking more than one object.
        fill_hole_area: Area in pixels (at video resolution) of holes to fill and small
            foreground regions to remove in returned masks; 0 disables.
        always_start_from_first_ann_frame: Propagate from the first annotated frame
            regardless of ``start_frame_idx``.
        max_point_num_in_prompt_enc: Stored only; this class does not read it.
        non_overlap_masks_for_output: Keep each pixel in the mask of the object with the
            highest logit only, in returned masks.
        **kwargs: Passed to ``VideoTrackingDynamicMultiplex``.
    """

    def __init__(
        self,
        clear_non_cond_mem_around_input: bool = False,
        clear_non_cond_mem_for_multi_obj: bool = False,
        fill_hole_area: int = 0,
        always_start_from_first_ann_frame: bool = False,
        max_point_num_in_prompt_enc: int = 16,
        non_overlap_masks_for_output: bool = True,
        **kwargs: Unpack[DynamicMultiplexTrackingConstructorArgs],
    ):
        super().__init__(**kwargs)
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

    def init_state(
        self,
        video_path: str | PathLike[str] | Sequence[Image.Image],
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
    ) -> MultiplexDemoState:
        """Create a session for ``video_path`` (see ``_load_multiplex_demo_state_from_resource``).

        Raises:
            UnsupportedMultiplexRuntimeError: If offloading or torchcodec is requested.
        """
        return _load_multiplex_demo_state_from_resource(
            self,
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )

    def _get_obj_num(self, inference_state: MultiplexDemoState) -> int:
        """Number of tracked objects: the multiplex state's valid entries, else the registered ids."""
        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is not None:
            return int(multiplex_state.total_valid_entries)
        return len(inference_state.get("obj_ids", []))

    def _obj_id_to_idx(
        self,
        inference_state: MultiplexDemoState,
        obj_id: object,
        error_if_new: bool = False,
    ) -> int:
        """Map an object id to its dense index, registering a new id when allowed.

        A new id gets the next index and empty per-object containers. That is allowed when
        the model is dynamic or tracking has not started, and ``error_if_new`` is False.

        Raises:
            RuntimeError: If the id is unknown and cannot be added.
        """
        obj_id_to_idx = inference_state["obj_id_to_idx"]
        obj_idx = obj_id_to_idx.get(obj_id)
        if obj_idx is not None:
            return int(obj_idx)
        if (
            getattr(self, "is_dynamic_model", False)
            or not inference_state["tracking_has_started"]
        ) and not error_if_new:
            obj_idx = len(obj_id_to_idx)
            obj_id_to_idx[obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(obj_id_to_idx)
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        raise RuntimeError(
            f"Cannot add new object id {obj_id}. Existing object ids: "
            f"{inference_state['obj_ids']}."
        )

    def _select_propagation_output_obj_ids(
        self,
        inference_state: MultiplexDemoState,
        obj_ids: object,
    ) -> tuple[list[object], list[int] | None]:
        """Resolve the objects to report during propagation.

        Returns ``(ids, indices)``: all ids with ``None`` indices when ``obj_ids`` is None,
        otherwise the requested ids with their data-space indices.

        Raises:
            ValueError: For duplicate or unknown ids.
        """
        if obj_ids is None:
            return list(inference_state["obj_ids"]), None
        requested_obj_ids = _coerce_obj_id_list(obj_ids)
        seen: set[object] = set()
        duplicates: list[object] = []
        for obj_id in requested_obj_ids:
            if obj_id in seen:
                duplicates.append(obj_id)
            seen.add(obj_id)
        if duplicates:
            raise ValueError(f"obj_ids contains duplicate object ids: {duplicates}.")

        obj_id_to_idx = inference_state["obj_id_to_idx"]
        missing = [
            obj_id for obj_id in requested_obj_ids if obj_id not in obj_id_to_idx
        ]
        if missing:
            raise ValueError(
                f"Unknown obj_ids {missing}; active object ids are "
                f"{inference_state['obj_ids']}."
            )
        return requested_obj_ids, [
            int(obj_id_to_idx[obj_id]) for obj_id in requested_obj_ids
        ]

    def _get_processing_order(
        self,
        inference_state: MultiplexDemoState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> range | list[int]:
        """Frame indices to process, inclusive of both ends.

        Forward gives ``start`` to ``min(start + max_frame_num_to_track, num_frames - 1)``.
        Reverse gives ``start`` down to ``max(start - max_frame_num_to_track, 0)``, or just
        ``[0]`` when ``start`` is 0. ``start`` defaults to the first conditioning frame and
        is overridden by the first annotated frame with ``always_start_from_first_ann_frame``.
        """
        num_frames = int(inference_state["num_frames"])
        if self.always_start_from_first_ann_frame:
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(int(start_frame_idx) - int(max_frame_num_to_track), 0)
            if int(start_frame_idx) > 0:
                return range(int(start_frame_idx), end_frame_idx - 1, -1)
            return [0]
        end_frame_idx = min(
            int(start_frame_idx) + int(max_frame_num_to_track),
            num_frames - 1,
        )
        return range(int(start_frame_idx), end_frame_idx + 1)

    def _prepare_demo_backbone_features(self, backbone_out: object) -> PreparedBackboneOutput[mx.array]:
        """Return backbone features in prepared (flattened) form, flattening raw ones if needed."""
        if isinstance(backbone_out, dict):
            values = cast(Mapping[str, object], backbone_out)
            interactive = values.get("interactive")
            if (
                "sam2_backbone_out" in values
                and isinstance(interactive, dict)
                and "vision_feats" in interactive
            ):
                return cast(PreparedBackboneOutput[mx.array], backbone_out)
        return self._prepare_backbone_features(cast(RawBackboneOutput[mx.array], backbone_out))

    def _get_image_feature(
        self,
        inference_state: MultiplexDemoState,
        frame_idx: int,
        batch_size: int,
    ) -> tuple[mx.array, PreparedBackboneOutput[mx.array]]:
        """Return ``(image, prepared backbone features)`` for ``frame_idx``.

        Uses ``cached_features[frame_idx]``, an ``(image, backbone_out)`` pair. On a miss
        the backbone runs on the frame from ``inference_state['images']`` and
        ``cached_features`` is replaced by a new dict holding only that frame, so the
        caller's dict is not updated. ``batch_size`` is unused.

        Raises:
            RuntimeError: If the features are not cached and the state has no images.
            TypeError: If a cached entry is not an ``(image, backbone_out)`` pair.
        """
        cached: object = inference_state["cached_features"].get(frame_idx)
        if cached is None:
            images = inference_state.get("images")
            if images is None:
                raise RuntimeError(
                    f"Image features for frame {frame_idx} are not cached and "
                    "inference_state does not contain images."
                )
            if is_mlx_array(images):
                image = images[frame_idx : frame_idx + 1].astype(mx.float32)
            else:
                image = mx.array(images[frame_idx], dtype=mx.float32)
                if len(image.shape) == 3:
                    image = image[None]
            backbone_out = self.forward_image(
                image,
                need_sam3_out=True,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
            cached = (image, backbone_out)

        if not isinstance(cached, tuple):
            raise TypeError("cached_features values must be (image, backbone_out).")
        cached_items = cast(tuple[object, ...], cached)
        if len(cached_items) != 2:
            raise TypeError("cached_features values must be (image, backbone_out).")
        image, backbone_out = cast(tuple[mx.array, RawBackboneOutput[mx.array] | PreparedBackboneOutput[mx.array]], cached_items)
        features = self._prepare_demo_backbone_features(backbone_out)
        _eval_tree(image)
        return image, features

    def _run_memory_encoder(
        self,
        inference_state: MultiplexDemoState,
        frame_idx: int,
        batch_size: int,
        high_res_masks: mx.array,
        object_score_logits: mx.array,
        is_mask_from_pts: bool,
        conditioning_objects: set[int] | None = None,
    ) -> tuple[mx.array, list[mx.array] | None, mx.array, mx.array]:
        """Run the memory encoder on ``high_res_masks``.

        Ported from the official ``Sam3VideoTrackingMultiplexDemo._run_memory_encoder``.
        Re-encodes propagated masks into memory. Returns the multiplex
        4-tuple ``(maskmem_features, maskmem_pos_enc, image_features, image_pos_enc)``.
        The official Torch bf16 cast and ``storage_device`` transfers are dropped
        because MLX keeps unified-memory float32 arrays.
        """
        image, backbone_features = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        multiplex_state = inference_state["multiplex_state"]
        assert multiplex_state is not None
        backbone_features_propagation = backbone_features["sam2_backbone_out"]
        propagation_vision_feats = backbone_features_propagation["vision_feats"]
        if "vision_pos_embeds" not in backbone_features_propagation:
            raise KeyError("vision_pos_embeds")
        propagation_vision_pos_embeds = backbone_features_propagation[
            "vision_pos_embeds"
        ]
        propagation_feat_sizes = backbone_features_propagation["feat_sizes"]

        if conditioning_objects is None:
            output_dict = inference_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                storage = output_dict[storage_key]
                if frame_idx not in storage:
                    continue
                conditioning_objects = storage[frame_idx]["conditioning_objects"]
                break
            else:
                raise ValueError(
                    f"conditioning objects not found at frame_idx={frame_idx}"
                )

        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            image=image,
            current_vision_feats=propagation_vision_feats,
            feat_sizes=propagation_feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
            conditioning_objects=conditioning_objects,
            multiplex_state=multiplex_state,
        )
        # "maskmem_pos_enc" is the same across frames, so we cache one copy and
        # re-expand it to the current batch size.
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        image_features = propagation_vision_feats[-1]
        image_pos_enc = propagation_vision_pos_embeds[-1]
        return maskmem_features, maskmem_pos_enc, image_features, image_pos_enc

    def _get_maskmem_pos_enc(
        self,
        inference_state: MultiplexDemoState,
        current_out: Mapping[str, list[mx.array] | None],
    ) -> list[mx.array] | None:
        """Cache ``maskmem_pos_enc`` once and re-expand it to the batch size.

        ``maskmem_pos_enc`` does not depend on the object index, so the demo
        stores a single object's slice in ``inference_state["constants"]`` and
        broadcasts it back up to the actual batch size on each call.
        """
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if out_maskmem_pos_enc is None:
            return None
        if "maskmem_pos_enc" not in model_constants:
            # only keep the slice for one object, since it's the same across objects
            maskmem_pos_enc = [x[0:1] for x in out_maskmem_pos_enc]
            model_constants["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            maskmem_pos_enc = model_constants["maskmem_pos_enc"]
        batch_size = out_maskmem_pos_enc[0].shape[0]
        return [mx.broadcast_to(x, (batch_size, *x.shape[1:])) for x in maskmem_pos_enc]

    def _run_single_frame_inference(
        self,
        *,
        inference_state: MultiplexDemoState,
        output_dict: dict[str, dict[int, StageOutput]],
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: dict[str, mx.array] | None,
        mask_inputs: mx.array | None,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: mx.array | None = None,
        add_to_existing_state: bool = False,
        new_object_masks: mx.array | None = None,
        new_object_idxs: list[int] | None = None,
        new_object_ids: list[object] | None = None,
        are_new_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
        reconditioning: bool = False,
        objects_to_interact: list[int] | None = None,
    ) -> tuple[StageOutput, mx.array]:
        """Run tracking for one frame, or edit that frame's stored output, and return ``(output, pred_masks)``.

        By default this calls ``track_step`` for the whole session. With
        ``add_to_existing_state`` or ``reconditioning`` the existing output of ``frame_idx``
        (conditioning or not) is edited in place instead, adding (or replacing) the objects
        at ``new_object_idxs`` with integer ids ``new_object_ids``. Their masks are
        ``mask_inputs``, ``new_object_masks`` or, for a point prompt, the low-resolution
        masks of a SAM-head interaction on ``point_inputs`` (``prev_sam_mask_logits`` rows
        are selected by ``new_object_idxs`` when its batch differs from the points).
        Memory is encoded only if ``run_mem_encoder``. ``pred_masks`` are the
        low-resolution mask logits ``(N, 1, h, w)`` of all objects; the outputs are
        evaluated before returning. ``batch_size`` is only forwarded to the feature lookup,
        which ignores it.

        Raises:
            ValueError: If masks, points, indices or ids needed for an edit are missing.
            TypeError: For non-integer ids or a state without a ``MultiplexState``.
            RuntimeError: If the frame has no stored output to edit or no ``pred_masks``
                are produced.
        """
        image, backbone_features = self._get_image_feature(
            inference_state,
            frame_idx,
            batch_size,
        )
        if add_to_existing_state or reconditioning:
            if new_object_idxs is None:
                raise ValueError("new_object_idxs are required when editing state.")
            if new_object_ids is None:
                raise ValueError("new_object_ids are required when editing state.")
            validated_object_ids: list[object] = []
            for object_id in new_object_ids:
                if isinstance(object_id, bool) or not isinstance(object_id, Integral):
                    raise TypeError("new_object_ids must contain integer identifiers.")
                validated_object_ids.append(int(object_id))

            multiplex_state = inference_state.get("multiplex_state")
            if not isinstance(multiplex_state, MultiplexState):
                raise TypeError("Editing state requires a MultiplexState instance.")

            existing_out = output_dict["cond_frame_outputs"].get(frame_idx)
            if existing_out is None:
                existing_out = output_dict["non_cond_frame_outputs"].get(frame_idx)
            if existing_out is None:
                raise RuntimeError(
                    f"No existing output found for frame {frame_idx} in either storage."
                )

            interactive_features = backbone_features["interactive"]
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive_features["vision_feats"],
                interactive_features["feat_sizes"],
            )
            interactive_high_res_features = [
                _reshape_spatial_feature(x, feat_size)
                for x, feat_size in zip(
                    interactive_features["vision_feats"][:-1],
                    interactive_features["feat_sizes"][:-1],
                )
            ]
            propagation_features = backbone_features["sam2_backbone_out"]
            propagation_vision_feats = (
                propagation_features["vision_feats"] if run_mem_encoder else None
            )
            propagation_feat_sizes = (
                propagation_features["feat_sizes"] if run_mem_encoder else None
            )

            if (add_to_existing_state or reconditioning) and mask_inputs is None:
                if point_inputs is None:
                    raise ValueError("point_inputs are required to add point objects.")
                sam_mask_inputs = prev_sam_mask_logits
                if (
                    sam_mask_inputs is not None
                    and sam_mask_inputs.shape[0]
                    != point_inputs["point_coords"].shape[0]
                ):
                    sam_mask_inputs = _take_indices(sam_mask_inputs, new_object_idxs)
                interaction_out = self._forward_sam_heads(
                    backbone_features=interactive_pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=sam_mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    multimask_output=self._use_multimask(
                        is_init_cond_frame,
                        point_inputs=point_inputs,
                    ),
                    objects_to_interact=new_object_idxs,
                    multiplex_state=multiplex_state,
                )
                new_object_masks = interaction_out["low_res_masks"]
                are_new_masks_from_pts = True

            masks_to_edit = mask_inputs if mask_inputs is not None else new_object_masks
            if masks_to_edit is None:
                raise ValueError("new object masks are required when editing state.")
            if reconditioning:
                self.recondition_masks_in_existing_state(
                    interactive_pix_feat=interactive_pix_feat,
                    interactive_high_res_features=interactive_high_res_features,
                    propagation_vision_feats=propagation_vision_feats,
                    propagation_feat_sizes=propagation_feat_sizes,
                    new_masks=masks_to_edit,
                    obj_idxs_in_mask=new_object_idxs,
                    obj_ids_in_mask=validated_object_ids,
                    prev_output=existing_out,
                    multiplex_state=multiplex_state,
                    add_mask_to_memory=run_mem_encoder,
                    are_masks_from_pts=are_new_masks_from_pts,
                )
            else:
                self.add_new_masks_to_existing_state(
                    interactive_pix_feat=interactive_pix_feat,
                    interactive_high_res_features=interactive_high_res_features,
                    propagation_vision_feats=propagation_vision_feats,
                    propagation_feat_sizes=propagation_feat_sizes,
                    new_masks=masks_to_edit,
                    obj_idxs_in_mask=new_object_idxs,
                    obj_ids_in_mask=validated_object_ids,
                    prev_output=existing_out,
                    multiplex_state=multiplex_state,
                    add_mask_to_memory=run_mem_encoder,
                    are_masks_from_pts=are_new_masks_from_pts,
                    allow_new_buckets=allow_new_buckets,
                    prefer_new_buckets=prefer_new_buckets,
                )
            current_out = existing_out
        else:
            multiplex_state = inference_state["multiplex_state"]
            assert multiplex_state is not None
            current_out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                backbone_features_interactive=backbone_features["interactive"],
                backbone_features_propagation=backbone_features["sam2_backbone_out"],
                image=image,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                gt_masks=None,
                frames_to_add_correction_pt=[],
                output_dict=output_dict,
                num_frames=inference_state["num_frames"],
                track_in_reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                prev_sam_mask_logits=prev_sam_mask_logits,
                multiplex_state=multiplex_state,
                objects_to_interact=objects_to_interact,
                new_object_masks=new_object_masks,
                new_object_idxs=new_object_idxs,
                new_object_ids=new_object_ids,
                are_new_masks_from_pts=are_new_masks_from_pts,
                allow_new_buckets=allow_new_buckets,
                prefer_new_buckets=prefer_new_buckets,
                reconditioning=reconditioning,
            )
        pred_masks = current_out.get("pred_masks")
        if pred_masks is None:
            raise RuntimeError(
                "Tracking must produce pred_masks for the current frame."
            )
        _eval_tree(current_out, pred_masks)
        return current_out, pred_masks

    def _get_orig_video_res_output(
        self,
        inference_state: MultiplexDemoState,
        any_res_masks: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Return ``(any_res_masks, video_res_masks)`` for mask logits ``(N, 1, h, w)``.

        The video-resolution masks are the input bilinearly resized to the original
        ``(video_height, video_width)``, then, if configured, resolved so objects do not
        overlap (``non_overlap_masks_for_output``) and cleaned with ``fill_hole_area``.
        """
        video_h = int(inference_state["video_height"])
        video_w = int(inference_state["video_width"])
        if tuple(any_res_masks.shape[-2:]) == (video_h, video_w):
            video_res_masks = any_res_masks
        else:
            video_res_masks = interpolate(
                any_res_masks.astype(mx.float32),
                size=(video_h, video_w),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        if self.fill_hole_area > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks,
                self.fill_hole_area,
            )
        _eval_tree(any_res_masks, video_res_masks)
        return any_res_masks, video_res_masks

    def _add_output_per_object(
        self,
        inference_state: MultiplexDemoState,
        frame_idx: int,
        current_out: StageOutput,
        storage_key: str,
    ) -> None:
        """Store each object's row of a packed frame output in ``output_dict_per_obj``.

        Keeps ``pred_masks``, ``object_score_logits`` and, when present,
        ``pred_masks_video_res`` and (with memory selection) ``iou_score``, sliced at the
        object's index, under ``storage_key`` and ``frame_idx``.

        Raises:
            RuntimeError: If the output lacks ``pred_masks`` or ``object_score_logits``.
        """
        pred_masks = current_out.get("pred_masks")
        if pred_masks is None:
            raise RuntimeError("Per-object output requires pred_masks.")
        object_score_logits = current_out.get("object_score_logits")
        if object_score_logits is None:
            raise RuntimeError("Per-object output requires object_score_logits.")
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out: dict[str, mx.array] = {
                "pred_masks": pred_masks[obj_slice],
                "object_score_logits": object_score_logits[obj_slice],
            }
            pred_masks_video_res = current_out.get("pred_masks_video_res")
            if pred_masks_video_res is not None:
                obj_out["pred_masks_video_res"] = pred_masks_video_res[obj_slice]
            iou_score = current_out.get("iou_score")
            if self.use_memory_selection and iou_score is not None:
                obj_out["iou_score"] = iou_score[obj_slice]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _get_or_create_multiplex_state(
        self,
        inference_state: MultiplexDemoState,
        obj_ids: list[object],
        *,
        is_new_state: bool,
        reconditioning: bool,
    ) -> MultiplexState[object]:
        """Return the session's multiplex state, creating it for ``obj_ids`` if none exists.

        A new state is built by the controller (objects in slot order) or, without a
        controller, as one bucket. ``is_new_state`` is unused.

        Raises:
            AssertionError: If a state exists, ``reconditioning`` is off and the model is not
                dynamic (new objects are not allowed).
            ValueError: If ``reconditioning`` is requested without an existing state.
        """
        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is not None:
            if not reconditioning and not getattr(self, "is_dynamic_model", False):
                raise AssertionError("New objects are not allowed after state creation")
            return multiplex_state
        if reconditioning:
            raise ValueError("reconditioning requires an existing multiplex_state")
        controller = getattr(self, "multiplex_controller", None)
        if controller is not None:
            multiplex_state = controller.get_state(
                num_valid_entries=len(obj_ids),
                device=inference_state["device"],
                dtype=mx.float32,
                random=False,
                object_ids=obj_ids,
            )
        else:
            multiplex_state = MultiplexState(
                [list(range(len(obj_ids)))],
                dtype=mx.float32,
                allowed_bucket_capacity=len(obj_ids),
                object_ids=obj_ids,
            )
        inference_state["multiplex_state"] = multiplex_state
        return multiplex_state

    def add_new_points(
        self,
        inference_state: MultiplexDemoState,
        frame_idx: int,
        obj_id: object,
        points: ArrayInput,
        labels: ArrayInput,
        clear_old_points: bool,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
    ) -> tuple[int, list[object], None, mx.array]:
        """Add point prompts for one object on a frame and return its updated masks.

        ``points`` is ``(P, 2)`` or ``(1, P, 2)`` of ``(x, y)`` and ``labels`` ``(P,)`` or
        ``(1, P)`` with 1 for foreground and 0 for background. With ``rel_coordinates``
        the points are fractions of the frame and are scaled by ``image_size``; otherwise
        they are in model input pixels. ``clear_old_points`` replaces the points already
        given for this object on this frame, otherwise the new ones are appended.
        ``use_prev_mem_frame`` is unused.

        An unseen ``obj_id`` creates the session's multiplex state, or is added to the
        existing one (the model is dynamic). In a session with several objects, points for a
        known object on a frame that already has output recondition that object's rows in
        place; on a frame without output only that object is interacted with. Repeated
        refinements of one frame after tracking has started feed the object's previous mask
        logits (clipped to +/-32) back as mask input. The frame's memory is not encoded here
        (``run_mem_encoder=False``).

        Returns:
            ``(frame_idx, obj_ids, None, video_res_masks)``, the masks being logits
            ``(num_objects, 1, video_height, video_width)`` of the frame.

        Raises:
            ValueError: For a frame outside the video or inconsistent point and label shapes.
            UnsupportedMultiplexRuntimeError: For a point batch larger than 1.
            RuntimeError: If the object cannot be added or the model produces no masks.
        """
        del use_prev_mem_frame
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < int(inference_state["num_frames"]):
            raise ValueError(f"frame_idx {frame_idx} is outside the video frame range.")
        points = (
            points.astype(mx.float32)
            if is_mlx_array(points)
            else mx.array(
                points,
                dtype=mx.float32,
            )
        )
        labels = (
            labels.astype(mx.int32)
            if is_mlx_array(labels)
            else mx.array(
                labels,
                dtype=mx.int32,
            )
        )
        if len(points.shape) == 2:
            points = points[None]
        if len(labels.shape) == 1:
            labels = labels[None]
        if tuple(points.shape[:2]) != tuple(labels.shape):
            raise ValueError(
                "points must have shape [B, P, 2] and labels must have shape [B, P]."
            )
        if points.shape[0] != 1:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplexDemo.add_new_points(batch_size)"
            )
        if rel_coordinates:
            points = points * float(self.image_size)

        multiplex_state = inference_state.get("multiplex_state")
        state_obj_ids = (
            []
            if multiplex_state is None
            else list(getattr(multiplex_state, "object_ids", None) or [])
        )
        is_new_state = multiplex_state is None
        is_existing_object = obj_id in state_obj_ids
        is_dynamic_new_object = not is_new_state and not is_existing_object
        is_refinement = (
            not is_new_state
            and inference_state["tracking_has_started"]
            and is_existing_object
        )
        existing_multi_object_edit = False
        existing_multi_object_gap_fill = False
        if (
            not is_new_state
            and is_existing_object
            and self._get_obj_num(inference_state) != 1
        ):
            existing_multi_object_edit = True
            frame_has_existing_output = (
                frame_idx in inference_state["output_dict"]["cond_frame_outputs"]
                or frame_idx in inference_state["output_dict"]["non_cond_frame_outputs"]
            )
            if not frame_has_existing_output:
                existing_multi_object_gap_fill = True

        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        obj_ids = [obj_id]
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        old_point_inputs = (
            None if clear_old_points else point_inputs_per_frame.get(frame_idx)
        )
        point_inputs = concat_points(old_point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        if is_new_state:
            multiplex_state = self._get_or_create_multiplex_state(
                inference_state,
                obj_ids,
                is_new_state=True,
                reconditioning=False,
            )

        batch_size = self._get_obj_num(inference_state)
        run_batch_size = (
            1
            if (
                is_dynamic_new_object
                or (existing_multi_object_edit and not existing_multi_object_gap_fill)
            )
            else batch_size
        )
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        user_refined_frames = None
        if is_refinement and not is_init_cond_frame:
            user_refined_frames_map = inference_state.setdefault(
                "user_refined_frames_per_obj",
                {},
            )
            user_refined_frames = user_refined_frames_map.setdefault(obj_id, set())
        is_first_refinement = (
            user_refined_frames is not None and frame_idx not in user_refined_frames
        )
        run_is_init_cond_frame = (
            False
            if existing_multi_object_gap_fill
            else is_dynamic_new_object or is_init_cond_frame or is_first_refinement
        )
        is_cond = (
            is_init_cond_frame
            or is_refinement
            or self.add_all_frames_to_correct_as_cond
        )
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        if user_refined_frames is not None and not is_first_refinement:
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            search_keys = (
                storage_key,
                "cond_frame_outputs",
                "non_cond_frame_outputs",
            )
            prev_out = None
            for container in (obj_temp_output_dict, obj_output_dict):
                searched: set[str] = set()
                for candidate_key in search_keys:
                    if candidate_key in searched:
                        continue
                    searched.add(candidate_key)
                    prev_out = container[candidate_key].get(frame_idx)
                    if prev_out is not None:
                        break
                if prev_out is not None:
                    break
            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = mx.clip(prev_out["pred_masks"], -32.0, 32.0)

        current_out, pred_masks = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=run_batch_size,
            is_init_cond_frame=run_is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=False if is_dynamic_new_object else reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            add_to_existing_state=is_dynamic_new_object,
            new_object_idxs=[obj_idx]
            if is_dynamic_new_object
            or existing_multi_object_edit
            or existing_multi_object_gap_fill
            else None,
            new_object_ids=[obj_id]
            if is_dynamic_new_object
            or existing_multi_object_edit
            or existing_multi_object_gap_fill
            else None,
            allow_new_buckets=is_dynamic_new_object,
            prefer_new_buckets=is_dynamic_new_object,
            reconditioning=(
                existing_multi_object_edit and not existing_multi_object_gap_fill
            ),
            objects_to_interact=[obj_idx]
            if existing_multi_object_edit or existing_multi_object_gap_fill
            else None,
        )
        if "pred_masks" not in current_out:
            raise RuntimeError("Point interaction must produce pred_masks.")
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state,
            pred_masks,
        )
        current_out["pred_masks_video_res"] = video_res_masks
        current_out["local_obj_id_to_idx"] = OrderedDict(
            inference_state["obj_id_to_idx"]
        )

        if is_cond:
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(
                frame_idx, None
            )
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(frame_idx)
        inference_state["output_dict"][storage_key][frame_idx] = current_out
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = frame_idx
        if user_refined_frames is not None:
            user_refined_frames.add(frame_idx)

        self._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        obj_temp = inference_state["temp_output_dict_per_obj"][obj_idx][storage_key]
        obj_temp[frame_idx] = dict(
            inference_state["output_dict_per_obj"][obj_idx][storage_key][frame_idx]
        )
        _eval_tree(current_out, video_res_masks)
        return frame_idx, list(inference_state["obj_ids"]), None, video_res_masks

    @overload
    def add_new_masks(
        self,
        *,
        inference_state: MultiplexDemoState | dict[str, object],
        frame_idx: int,
        obj_ids: object,
        masks: ArrayInput,
        add_mask_to_memory: bool = False,
        are_masks_from_pts: bool = False,
    ) -> tuple[int, list[object], None, mx.array]: ...

    @overload
    def add_new_masks(
        self,
        inference_state: MultiplexDemoState | dict[str, object],
        frame_idx: int,
        obj_ids: object,
        masks: ArrayInput,
        add_mask_to_memory: bool = False,
        reconditioning: bool = False,
        *,
        are_masks_from_pts: bool = False,
    ) -> tuple[int, list[object], None, mx.array]: ...

    def add_new_masks(
        self,
        inference_state: MultiplexDemoState | dict[str, object],
        frame_idx: int,
        obj_ids: object,
        masks: ArrayInput,
        add_mask_to_memory: bool = False,
        reconditioning: bool = False,
        *,
        are_masks_from_pts: bool = False,
    ) -> tuple[int, list[object], None, mx.array]:
        """Add mask prompts for objects on a frame and return the frame's masks.

        ``masks`` is ``(N, H, W)`` at any resolution, one per id in ``obj_ids``. They are
        resized bilinearly to the model mask input size for the tracker, and a copy
        resized to the video size and thresholded at 0.5 is kept in
        ``mask_inputs_per_obj``. A first
        call creates the multiplex state; later calls add the objects to it. With
        ``reconditioning`` the ids must already exist and their masks are replaced.
        ``add_mask_to_memory`` also encodes the frame into memory and
        ``are_masks_from_pts`` marks masks that came from point prompts.

        Returns:
            ``(frame_idx, obj_ids, None, video_res_masks)`` with logits
            ``(num_objects, 1, video_height, video_width)``; the rows of the given objects
            are replaced by the prompt itself as +/-1024 (``NO_OBJ_SCORE``) logits.

        Raises:
            ValueError: For no ids, masks that are not ``(N, H, W)`` or a count mismatch.
            RuntimeError: For unknown ids with ``reconditioning`` or if no masks are produced.
        """
        inference_state = cast(MultiplexDemoState, inference_state)
        obj_ids = _coerce_obj_id_list(obj_ids)
        if len(obj_ids) == 0:
            raise ValueError("obj_ids must contain at least one object id.")
        obj_idxs = [
            self._obj_id_to_idx(
                inference_state,
                obj_id,
                error_if_new=reconditioning,
            )
            for obj_id in obj_ids
        ]
        masks = (
            masks.astype(mx.float32)
            if is_mlx_array(masks)
            else mx.array(
                masks,
                dtype=mx.float32,
            )
        )
        if len(masks.shape) != 3:
            raise ValueError("masks must have shape [N, H, W].")
        num_objects, mask_h, mask_w = masks.shape
        if num_objects != len(obj_ids):
            raise ValueError(
                "masks batch must match obj_ids length; "
                f"got {num_objects} masks for {len(obj_ids)} ids."
            )

        masks_orig = masks[:, None, :, :]
        input_mask_size = int(getattr(self, "input_mask_size", self.image_size))
        if (int(mask_h), int(mask_w)) == (input_mask_size, input_mask_size):
            mask_inputs = masks_orig
        else:
            mask_inputs = interpolate(
                masks_orig,
                size=(input_mask_size, input_mask_size),
                mode="bilinear",
                align_corners=False,
            )

        video_h = int(inference_state["video_height"])
        video_w = int(inference_state["video_width"])
        if (int(mask_h), int(mask_w)) == (video_h, video_w):
            mask_inputs_video_res = masks_orig
        else:
            mask_inputs_video_res = interpolate(
                masks_orig,
                size=(video_h, video_w),
                mode="bilinear",
                align_corners=False,
            )
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        multiplex_state = inference_state.get("multiplex_state")
        is_new_state = multiplex_state is None
        multiplex_state = self._get_or_create_multiplex_state(
            inference_state,
            obj_ids,
            is_new_state=is_new_state,
            reconditioning=reconditioning,
        )

        for mask_idx, obj_idx in enumerate(obj_idxs):
            inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = (
                mask_inputs_video_res[mask_idx : mask_idx + 1]
            )
            inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        allow_new_buckets = (
            not is_new_state
            and not reconditioning
            and multiplex_state.available_slots < num_objects
        )
        current_out, pred_masks = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=num_objects,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=add_mask_to_memory,
            add_to_existing_state=not is_new_state and not reconditioning,
            new_object_masks=(None if is_new_state else mask_inputs),
            new_object_idxs=(None if is_new_state else obj_idxs),
            new_object_ids=(None if is_new_state else obj_ids),
            are_new_masks_from_pts=are_masks_from_pts,
            allow_new_buckets=allow_new_buckets,
            reconditioning=reconditioning,
        )

        if "pred_masks" not in current_out:
            raise RuntimeError("Mask interaction must produce pred_masks.")
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state,
            pred_masks,
        )
        direct_logits = mx.where(
            mask_inputs_video_res,
            mx.array(-NO_OBJ_SCORE, dtype=video_res_masks.dtype),
            mx.array(NO_OBJ_SCORE, dtype=video_res_masks.dtype),
        )
        video_res_masks = _replace_indices(video_res_masks, obj_idxs, direct_logits)
        current_out["pred_masks_video_res"] = video_res_masks
        current_out["local_obj_id_to_idx"] = OrderedDict(
            inference_state["obj_id_to_idx"]
        )

        if is_cond:
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(
                frame_idx, None
            )
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(frame_idx)
        inference_state["output_dict"][storage_key][frame_idx] = current_out
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = frame_idx

        self._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        for obj_idx in obj_idxs:
            obj_temp = inference_state["temp_output_dict_per_obj"][obj_idx][storage_key]
            obj_temp[frame_idx] = dict(
                inference_state["output_dict_per_obj"][obj_idx][storage_key][frame_idx]
            )

        _eval_tree(current_out, video_res_masks)
        return frame_idx, list(inference_state["obj_ids"]), None, video_res_masks

    def propagate_in_video_preflight(
        self, sam2_state: object, run_mem_encoder: bool = True,
    ) -> None:
        """Commit pending prompt masks and encode their joint tracking memory.

        Unlike upstream's deferred prompt writer, this demo already maintains a
        packed frame output. Merge its per-object edits before marking the frame
        consolidated, otherwise propagation recomputes the prompted frame.
        """
        if not isinstance(sam2_state, dict):
            raise TypeError("SAM2 state must be a dictionary.")
        state = cast(MultiplexDemoState, sam2_state)
        state["tracking_has_started"] = True
        batch_size = self._get_obj_num(state)
        outputs = state["output_dict"]
        temporary = state["temp_output_dict_per_obj"]
        consolidated = state["consolidated_frame_inds"]
        for storage_key in ("non_cond_frame_outputs", "cond_frame_outputs"):
            pending = {
                frame_idx
                for object_outputs in temporary.values()
                for frame_idx in object_outputs[storage_key]
            }
            for frame_idx in sorted(pending):
                output = outputs[storage_key][frame_idx]
                masks = output.get("pred_masks")
                if masks is None:
                    raise RuntimeError("Pending prompt output must contain pred_masks.")
                for obj_idx, object_outputs in temporary.items():
                    object_output = object_outputs[storage_key].get(frame_idx)
                    if object_output is None:
                        existing = state["output_dict_per_obj"][obj_idx]
                        object_output = existing["cond_frame_outputs"].get(frame_idx)
                        if object_output is None:
                            object_output = existing["non_cond_frame_outputs"].get(frame_idx)
                    if object_output is None:
                        continue
                    video_mask = object_output.get("pred_masks_video_res")
                    object_mask = (
                        video_mask if video_mask is not None else object_output.get("pred_masks")
                    )
                    if object_mask is None:
                        raise RuntimeError("Pending object output must contain mask logits.")
                    if object_mask.shape[-2:] != masks.shape[-2:]:
                        object_mask = interpolate(
                            object_mask, size=(masks.shape[-2], masks.shape[-1]), mode="bilinear",
                            align_corners=False, antialias=video_mask is not None,
                        )
                    masks = _replace_indices(masks, [obj_idx], object_mask)
                output["pred_masks"] = masks
                output.pop("pred_masks_video_res", None)
                if run_mem_encoder:
                    object_scores = output.get("object_score_logits")
                    if object_scores is None:
                        raise RuntimeError("Pending prompt output must contain object scores.")
                    high_res_masks = interpolate(
                        masks, size=(self.image_size, self.image_size),
                        mode="bilinear", align_corners=False,
                    )
                    high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
                    memory, positions, features, feature_positions = self._run_memory_encoder(
                        inference_state=state, frame_idx=frame_idx, batch_size=batch_size,
                        high_res_masks=high_res_masks,
                        object_score_logits=object_scores,
                        is_mask_from_pts=True,
                        conditioning_objects=output["conditioning_objects"],
                    )
                    output["maskmem_features"] = memory
                    output["maskmem_pos_enc"] = positions
                    output["image_features"] = features
                    output["image_pos_enc"] = feature_positions
                _eval_tree(output)
                self._add_output_per_object(state, frame_idx, output, storage_key)
                consolidated[storage_key].add(frame_idx)
                if self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                ):
                    self._clear_non_cond_mem_around_input(state, frame_idx)
            for object_outputs in temporary.values():
                object_outputs[storage_key].clear()

        for frame_idx in outputs["cond_frame_outputs"]:
            outputs["non_cond_frame_outputs"].pop(frame_idx, None)
            consolidated["non_cond_frame_outputs"].discard(frame_idx)
        for object_outputs in state["output_dict_per_obj"].values():
            for frame_idx in object_outputs["cond_frame_outputs"]:
                object_outputs["non_cond_frame_outputs"].pop(frame_idx, None)
        input_frames = {
            frame_idx
            for inputs in (state["point_inputs_per_obj"], state["mask_inputs_per_obj"])
            for frames in inputs.values()
            for frame_idx in frames
        }
        if consolidated["cond_frame_outputs"] | consolidated["non_cond_frame_outputs"] != input_frames:
            raise ValueError("Consolidated frames must match point and mask input frames.")
        if state["first_ann_frame_idx"] not in outputs["cond_frame_outputs"]:
            state["first_ann_frame_idx"] = min(outputs["cond_frame_outputs"], default=None)

    def propagate_in_video(
        self,
        inference_state: MultiplexDemoState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        tqdm_disable: bool = False,
        obj_ids: object = None,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Iterator[TrackingFrame | ScoredTrackingFrame]:
        """Track the prompted objects through the video, yielding one result per frame.

        Frames in the order from ``_get_processing_order`` that already have a stored
        output are reused as they are (conditioning outputs created without memory stay
        that way); the others run the tracker (memory encoding follows
        ``run_mem_encoder``) and are stored as non-conditioning outputs. Each item is
        ``(frame_idx, obj_ids, low_res_masks, video_res_masks)``, restricted to ``obj_ids``
        when given. ``tqdm_disable`` is unused; ``propagate_preflight`` runs
        ``propagate_in_video_preflight`` first.

        Raises:
            RuntimeError: If no prompt has been added, or a stored output lacks masks.
            ValueError: If the session has no multiplex state, or ``obj_ids`` are
                duplicated or unknown.
        """
        del tqdm_disable
        if propagate_preflight:
            preflight = getattr(self, "propagate_in_video_preflight", None)
            if preflight is not None:
                preflight(inference_state, run_mem_encoder=run_mem_encoder)
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        if inference_state.get("multiplex_state") is None:
            raise ValueError("inference_state['multiplex_state'] is required.")
        inference_state["tracking_has_started"] = True

        batch_size = self._get_obj_num(inference_state)
        output_obj_ids, output_obj_indices = self._select_propagation_output_obj_ids(
            inference_state,
            obj_ids,
        )
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        processing_order = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        )
        for frame_idx in processing_order:
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out.get("pred_masks")
                if pred_masks is None:
                    raise RuntimeError("Stored tracking output is missing pred_masks.")
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out.get("pred_masks")
                if pred_masks is None:
                    raise RuntimeError("Stored tracking output is missing pred_masks.")
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                current_out["local_obj_id_to_idx"] = OrderedDict(
                    inference_state["obj_id_to_idx"]
                )
                output_dict[storage_key][frame_idx] = current_out

            self._add_output_per_object(
                inference_state,
                frame_idx,
                current_out,
                storage_key,
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state,
                pred_masks,
            )
            if output_obj_indices is not None:
                low_res_masks = _take_indices(low_res_masks, output_obj_indices)
                video_res_masks = _take_indices(video_res_masks, output_obj_indices)
                _eval_tree(low_res_masks, video_res_masks)
            yield frame_idx, output_obj_ids, low_res_masks, video_res_masks

    def clear_all_points_in_video(self, inference_state: MultiplexDemoState) -> None:
        """Forget all prompts, outputs and objects, leaving only the loaded video."""
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["multiplex_state"] = None
        inference_state.get("user_refined_frames_per_obj", {}).clear()

    def _reset_tracking_results(self, inference_state: MultiplexDemoState) -> None:
        """Clear prompts and all stored outputs and mark tracking as not started; objects and the multiplex state stay."""
        for value in inference_state["point_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["mask_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        for value in inference_state["temp_output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _clear_non_cond_mem_around_input(
        self,
        inference_state: MultiplexDemoState,
        frame_idx: int,
    ) -> None:
        """Drop non-conditioning outputs within ``memory_temporal_stride_for_eval * num_maskmem`` frames of ``frame_idx`` (both directions), packed and per object."""
        radius = int(self.memory_temporal_stride_for_eval) * int(self.num_maskmem)
        frame_idx_begin = int(frame_idx) - radius
        frame_idx_end = int(frame_idx) + radius
        output_dict = inference_state["output_dict"]
        for time_idx in range(frame_idx_begin, frame_idx_end + 1):
            output_dict["non_cond_frame_outputs"].pop(time_idx, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(time_idx, None)
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(time_idx)

    def clear_all_points_in_frame(
        self,
        inference_state: MultiplexDemoState,
        frame_idx: int,
        obj_id: object,
        need_output: bool = True,
        preserve_user_refined: bool = False,
    ) -> tuple[int, list[object], None, mx.array] | None:
        """Remove one object's prompts on a frame.

        If no other object has a prompt on the frame, its conditioning output becomes a
        non-conditioning one, and if no conditioning output is left all tracking results are
        reset. ``preserve_user_refined`` keeps the object's refined-frame record.
        With ``need_output`` returns ``(frame_idx, obj_ids, None, video_res_masks)`` built
        from the frame's stored predictions (not recomputed without the removed prompts),
        or None if the frame has no stored output.
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        if not preserve_user_refined:
            inference_state.get("user_refined_frames_per_obj", {}).get(
                obj_id,
                set(),
            ).discard(frame_idx)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        batch_size = len(inference_state["obj_ids"])
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"].get(obj_idx2, {}):
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"].get(obj_idx2, {}):
                frame_has_input = True
                break

        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                consolidated_frame_inds["non_cond_frame_outputs"].add(frame_idx)
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"].get(obj_idx2)
                if obj_output_dict is None:
                    continue
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return None
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            current_out = inference_state["output_dict"][storage_key].get(frame_idx)
            if current_out is not None:
                pred_masks = current_out.get("pred_masks")
                assert pred_masks is not None
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state,
                    pred_masks,
                )
                return (
                    frame_idx,
                    list(inference_state["obj_ids"]),
                    None,
                    video_res_masks,
                )
        return None

    def remove_object(
        self,
        inference_state: MultiplexDemoState,
        obj_id: object,
        strict: bool = False,
        need_output: bool = True,
        clear_user_refined_map: bool = True,
    ) -> tuple[list[object], list[tuple[int, mx.array]]]:
        """Remove one object; see ``remove_objects``."""
        return self.remove_objects(
            inference_state,
            [obj_id],
            strict=strict,
            need_output=need_output,
            clear_user_refined_map=clear_user_refined_map,
        )

    def remove_objects(
        self,
        inference_state: MultiplexDemoState,
        obj_ids: object,
        strict: bool = False,
        need_output: bool = True,
        clear_user_refined_map: bool = True,
    ) -> tuple[list[object], list[tuple[int, mx.array]]]:
        """Remove objects and renumber the remaining ones.

        The objects' prompts are cleared, the per-object records and packed outputs
        (mask rows, memory buckets, object pointers) are re-indexed, and the multiplex state
        drops the objects. Removing every object is the same as ``clear_all_points_in_video``.
        Unknown ids are ignored unless ``strict``. ``clear_user_refined_map`` also forgets
        the objects' refined frames.

        Returns:
            ``(remaining obj_ids, updated_frames)`` where, with ``need_output``,
            ``updated_frames`` lists ``(frame_idx, video_res_masks)`` for each frame on which
            a removed object had a prompt.

        Raises:
            ValueError: With ``strict`` for unknown ids, or if the session has no
                multiplex state.
        """
        requested_obj_ids = _coerce_obj_id_list(obj_ids)
        obj_id_to_idx = inference_state["obj_id_to_idx"]
        old_obj_idxs_to_rm = [
            obj_id_to_idx.get(obj_id, None) for obj_id in requested_obj_ids
        ]
        missing = [
            obj_id
            for obj_id, old_idx in zip(requested_obj_ids, old_obj_idxs_to_rm)
            if old_idx is None
        ]
        if missing and strict:
            raise ValueError(
                f"Object ids {missing} do not exist in the tracking state."
            )

        used_pairs = [
            (int(old_idx), obj_id)
            for old_idx, obj_id in zip(old_obj_idxs_to_rm, requested_obj_ids)
            if old_idx is not None
        ]
        updated_frames: list[tuple[int, mx.array]] = []
        if not used_pairs:
            return list(inference_state["obj_ids"]), updated_frames

        if len(used_pairs) == len(inference_state["obj_ids"]):
            self.clear_all_points_in_video(inference_state)
            return list(inference_state["obj_ids"]), updated_frames

        old_obj_idxs = [pair[0] for pair in used_pairs]
        removed_obj_ids = [pair[1] for pair in used_pairs]
        removed_obj_id_set = set(removed_obj_ids)

        if clear_user_refined_map:
            user_refined_map = inference_state.get("user_refined_frames_per_obj", {})
            for obj_id in removed_obj_ids:
                user_refined_map.pop(obj_id, None)

        all_obj_input_frames_inds: set[int] = set()
        for old_obj_idx, obj_id in used_pairs:
            point_frames = set(inference_state["point_inputs_per_obj"][old_obj_idx])
            mask_frames = set(inference_state["mask_inputs_per_obj"][old_obj_idx])
            obj_input_frames = point_frames | mask_frames
            for input_frame_idx in obj_input_frames:
                self.clear_all_points_in_frame(
                    inference_state,
                    input_frame_idx,
                    obj_id,
                    need_output=False,
                    preserve_user_refined=not clear_user_refined_map,
                )
            all_obj_input_frames_inds.update(obj_input_frames)

        old_obj_ids = list(inference_state["obj_ids"])
        remain_old_obj_inds = [
            old_idx
            for old_idx in range(len(old_obj_ids))
            if old_idx not in old_obj_idxs
        ]
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = OrderedDict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = OrderedDict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        def _map_keys[T](container: dict[int, T]) -> None:
            old_values = dict(container)
            container.clear()
            for old_idx in remain_old_obj_inds:
                if old_idx in old_values:
                    container[old_idx_to_new_idx[old_idx]] = old_values[old_idx]

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is None:
            raise ValueError("inference_state['multiplex_state'] is required.")
        buckets_to_keep = multiplex_state.remove_objects(old_obj_idxs, strict=True)

        def _slice_packed_output(out: StageOutput) -> None:
            pred_masks = out.get("pred_masks")
            object_score_logits = out.get("object_score_logits")
            assert pred_masks is not None and object_score_logits is not None
            if buckets_to_keep:
                out["maskmem_features"] = _take_indices(
                    out.get("maskmem_features"),
                    buckets_to_keep,
                )
                out["maskmem_pos_enc"] = _take_indices(
                    out.get("maskmem_pos_enc"),
                    buckets_to_keep,
                )
                out["obj_ptr"] = _take_indices(out.get("obj_ptr"), buckets_to_keep)
            else:
                out["maskmem_features"] = None
                out["maskmem_pos_enc"] = None
                out["obj_ptr"] = None

            local_obj_id_to_idx = OrderedDict(
                out.get("local_obj_id_to_idx", inference_state["obj_id_to_idx"])
            )
            keep_indices = [
                int(obj_idx)
                for obj_id, obj_idx in local_obj_id_to_idx.items()
                if obj_id not in removed_obj_id_set
                and 0 <= int(obj_idx) < pred_masks.shape[0]
                and 0 <= int(obj_idx) < object_score_logits.shape[0]
            ]
            old_to_new = {
                old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)
            }
            out["pred_masks"] = _take_indices(pred_masks, keep_indices)
            out["object_score_logits"] = _take_indices(
                object_score_logits,
                keep_indices,
            )
            if "iou_score" in out:
                out["iou_score"] = _take_indices(out["iou_score"], keep_indices)
            if "eff_iou_score" in out:
                out["eff_iou_score"] = _take_indices(
                    out["eff_iou_score"],
                    keep_indices,
                )
            if "pred_masks_video_res" in out:
                out["pred_masks_video_res"] = _take_indices(
                    out["pred_masks_video_res"],
                    keep_indices,
                )
            out["local_obj_id_to_idx"] = OrderedDict(
                (obj_id, old_to_new[int(obj_idx)])
                for obj_id, obj_idx in local_obj_id_to_idx.items()
                if obj_id not in removed_obj_id_set and int(obj_idx) in old_to_new
            )
            conditioning_objects = set(out.get("conditioning_objects", set()))
            out["conditioning_objects"] = {
                old_to_new[old_idx]
                for old_idx in conditioning_objects
                if old_idx in old_to_new
            }

        def _slice_state(storage_key: str) -> None:
            for frame_idx, out in inference_state["output_dict"][storage_key].items():
                _slice_packed_output(out)
                self._add_output_per_object(
                    inference_state,
                    frame_idx,
                    out,
                    storage_key,
                )

        _slice_state("cond_frame_outputs")
        _slice_state("non_cond_frame_outputs")

        if need_output:
            for frame_idx in sorted(all_obj_input_frames_inds):
                for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    out = inference_state["output_dict"][storage_key].get(frame_idx)
                    if out is None:
                        continue
                    pred_masks = out.get("pred_masks")
                    assert pred_masks is not None
                    _, video_res_masks = self._get_orig_video_res_output(
                        inference_state,
                        pred_masks,
                    )
                    updated_frames.append((frame_idx, video_res_masks))
                    break

        return list(inference_state["obj_ids"]), updated_frames


def _bind_demo_init_state(
    method: Callable[..., MultiplexDemoState],
) -> Callable[..., MultiplexDemoState]:
    """Let ``init_state`` accept both call styles of the upstream API.

    A non-numeric first positional argument is read as ``video_path`` followed by the
    ``VideoTrackingMultiplexDemo.init_state`` parameters (``cached_features`` is then
    keyword-only); otherwise the wrapped method's own order (video size first) applies.
    The wrapper also requires the offload and decoder flags to be bools.
    """
    dimensions_signature = signature(method)
    resource_signature = signature(VideoTrackingMultiplexDemo.init_state)

    @wraps(method)
    def bound_init_state(
        model: Sam3VideoTrackingMultiplexDemo, *args: object, **kwargs: object
    ) -> MultiplexDemoState:
        resource_first = (
            args and args[0] is not None and not isinstance(args[0], Number)
        )
        if resource_first:
            cached_features = kwargs.pop(
                "cached_features",
                dimensions_signature.parameters["cached_features"].default,
            )
            bound = resource_signature.bind_partial(model, *args, **kwargs)
            bound.apply_defaults()
            bound.arguments["cached_features"] = cached_features
        else:
            bound = dimensions_signature.bind(model, *args, **kwargs)
            bound.apply_defaults()
        values: dict[str, object] = {
            name: parameter.default
            for name, parameter in dimensions_signature.parameters.items()
            if name != "self"
        }
        values.update(bound.arguments)
        values.pop("self")
        for name in (
            "offload_video_to_cpu",
            "offload_state_to_cpu",
            "async_loading_frames",
            "use_torchcodec",
            "use_cv2",
        ):
            if not isinstance(values[name], bool):
                raise TypeError(f"{name} must be a bool.")
        return method(model, **values)

    return bound_init_state


class Sam3VideoTrackingMultiplexDemo(VideoTrackingMultiplexDemo):
    """Demo tracker whose ``init_state`` accepts either a video resource or explicit video dimensions, and whose propagation also yields object scores."""

    @overload
    def init_state(
        self,
        video_height: int | None = None,
        video_width: int | None = None,
        num_frames: int | None = None,
        cached_features: object = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        video_path: object = None,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
    ) -> MultiplexDemoState: ...

    @overload
    def init_state(
        self,
        video_path: str | PathLike[str] | Sequence[Image.Image],
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        *,
        cached_features: object = None,
    ) -> MultiplexDemoState: ...

    # Bind both positional contracts while preserving upstream signature inspection.
    @_bind_demo_init_state
    def init_state(
        self,
        video_height: int | None = None,
        video_width: int | None = None,
        num_frames: int | None = None,
        cached_features: object = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        video_path: str | PathLike[str] | Sequence[Image.Image] | None = None,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
    ) -> MultiplexDemoState:
        """Create a session from ``video_path``, or an empty one from the video dimensions.

        With ``video_path`` the video is loaded as in ``_load_multiplex_demo_state_from_resource``.
        Otherwise ``video_height``, ``video_width`` (original frame size, pixels) and
        ``num_frames`` are required and the session has no images, so frame features must
        come from ``cached_features``.

        Raises:
            UnsupportedMultiplexRuntimeError: For offloading, async frame loading without a
                path, or torchcodec.
            ValueError: If the dimensions are missing without a ``video_path``.
        """
        if video_path is not None:
            return _load_multiplex_demo_state_from_resource(
                self,
                video_path=video_path,
                cached_features=cached_features,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                async_loading_frames=async_loading_frames,
                use_torchcodec=use_torchcodec,
                use_cv2=use_cv2,
            )
        if offload_video_to_cpu or offload_state_to_cpu:
            raise_unsupported_multiplex_runtime(
                "Sam3VideoTrackingMultiplexDemo.init_state(offload)"
            )
        if async_loading_frames:
            raise_unsupported_multiplex_runtime(
                "Sam3VideoTrackingMultiplexDemo.init_state(async_loading_frames)"
            )
        if use_torchcodec:
            raise_unsupported_multiplex_runtime(
                "Sam3VideoTrackingMultiplexDemo.init_state(use_torchcodec=True)"
            )
        if video_height is None or video_width is None or num_frames is None:
            raise ValueError(
                "video_height, video_width, and num_frames are required when "
                "video_path is not provided."
            )
        del use_cv2
        return _init_multiplex_demo_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            cached_features=cached_features,
        )

    def propagate_in_video(
        self,
        inference_state: MultiplexDemoState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        tqdm_disable: bool = False,
        obj_ids: object = None,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Iterator[ScoredTrackingFrame]:
        """Like the base ``propagate_in_video`` but each item also ends with the frame's object score logits for the reported objects."""
        output_obj_indices = None
        base_outputs = cast(Iterator[TrackingFrame], super().propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            tqdm_disable=tqdm_disable,
            obj_ids=obj_ids,
            run_mem_encoder=run_mem_encoder,
            propagate_preflight=propagate_preflight,
        ))
        for frame_idx, output_obj_ids, low_res_masks, video_res_masks in base_outputs:
            if obj_ids is not None and output_obj_indices is None:
                _, output_obj_indices = self._select_propagation_output_obj_ids(
                    inference_state,
                    obj_ids,
                )
            output_dict = inference_state["output_dict"]
            current_out = output_dict["cond_frame_outputs"].get(frame_idx)
            if current_out is None:
                current_out = output_dict["non_cond_frame_outputs"][frame_idx]
            obj_scores = current_out.get("object_score_logits")
            assert obj_scores is not None
            if output_obj_indices is not None:
                obj_scores = _take_indices(obj_scores, output_obj_indices)
                _eval_tree(obj_scores)
            yield frame_idx, output_obj_ids, low_res_masks, video_res_masks, obj_scores
