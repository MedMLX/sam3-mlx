# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Independent SAM 3 image inference on selected video frames, with optional tracking.

Each session keeps one image processor, its frames and one prompt. Every requested
frame is encoded (with a bounded cache of frame encodings) and segmented with that
prompt on its own: the model carries nothing from one frame to the next, so object
ids are frame-local indices. A session opened with ``track=True`` instead links the
detections of propagated frames with ByteTrack (``sam3_mlx.tracking``) on the host
and returns persistent track ids.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping, Sequence
from typing import Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.io_utils import VideoFrameSource, load_video_frames
from sam3_mlx.tracking import LOW_SCORE_THRESH, VideoTracker

type ProcessorState = dict[str, object]
type PromptCoordinates = NDArray[np.float32]
type PromptLabels = NDArray[np.bool_]


class VideoOutputs(TypedDict):
    """Object ids, scores, normalized ``xywh`` boxes and boolean ``(N, H, W)`` masks.

    Ids are frame-local indices, or persistent track ids for frames propagated in
    a tracked session.
    """

    out_obj_ids: NDArray[np.int64]
    out_probs: NDArray[np.float32]
    out_boxes_xywh: NDArray[np.float32]
    out_binary_masks: NDArray[np.bool_]


class _ImageProcessor(Protocol):
    """The ``Sam3Processor`` methods a session uses; ``predict`` reads ``confidence_threshold``."""

    confidence_threshold: float

    def set_image(self, image: Image.Image) -> ProcessorState:
        """Encode one video frame as image-processor state."""
        ...

    def set_text_prompt(
        self, prompt: str, state: ProcessorState, *, run_grounding: bool = True,
    ) -> ProcessorState:
        """Attach a text prompt, optionally running grounding."""
        ...

    def add_geometric_prompt(
        self, box: list[float], label: bool, state: ProcessorState, *, run_grounding: bool = True,
    ) -> ProcessorState:
        """Append a labeled normalized ``cxcywh`` box prompt, optionally running grounding."""
        ...

    def add_point_prompt(
        self, point: list[float], label: bool, state: ProcessorState, *, run_grounding: bool = True,
    ) -> ProcessorState:
        """Append a labeled normalized ``xy`` point prompt, optionally running grounding."""
        ...

    def predict(self, state: ProcessorState) -> ProcessorState:
        """Run grounding and write ``masks``, pixel ``xyxy`` ``boxes`` and ``scores`` into the state."""
        ...


class ProcessorFactory(Protocol):
    """Build a session's image processor from a model and its inference settings."""

    def __call__(
        self, image_model: object, *, resolution: int, confidence_threshold: float,
    ) -> _ImageProcessor:
        """Construct an image processor for the selected resolution and threshold."""
        ...


class VideoInferenceState(TypedDict):
    """One session: its frames, processor, cached frame encodings and current prompt.

    ``box_prompt`` holds normalized ``cxcywh`` boxes and ``point_prompt``
    normalized ``xy`` points; ``prompt_frame_idx`` is the frame the prompt was
    added on, or None before the first prompt. ``tracker`` links propagated
    frames in a tracked session and is None otherwise.
    """

    frames: VideoFrameSource
    num_frames: int
    orig_height: int
    orig_width: int
    processor: _ImageProcessor
    frame_feature_cache: BoundedLRUCache[int, ProcessorState]
    text_prompt: str | None
    box_prompt: PromptCoordinates | None
    box_labels: PromptLabels | None
    point_prompt: PromptCoordinates | None
    point_labels: PromptLabels | None
    prompt_frame_idx: int | None
    tracker: VideoTracker | None


def _validate_frame_feature_cache_size(value: object) -> int:
    """Require a positive integer frame-feature cache capacity."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("frame_feature_cache_size must be a positive integer.")
    return value


def _validate_threshold(value: float) -> float:
    """Require a score threshold in [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
    return value


def _default_processor(
    image_model: object, *, resolution: int, confidence_threshold: float,
) -> _ImageProcessor:
    """Build a ``Sam3Processor`` for a SAM 3 image model."""
    from sam3_mlx.model.sam3_image_processor import ProcessorModel, Sam3Processor

    return Sam3Processor(
        cast(ProcessorModel, image_model),
        resolution=resolution,
        confidence_threshold=confidence_threshold,
    )


class Sam3VideoInference:
    """Run the image model independently on each requested frame.

    Args:
        image_model: ``Sam3Image`` run on each frame (any object the
            ``processor_factory`` accepts).
        resolution: Square model input size in pixels.
        confidence_threshold: Default score threshold for returned detections.
        processor_factory: Builds a session's image processor from ``(image_model,
            resolution=..., confidence_threshold=...)``; defaults to
            ``Sam3Processor``.
        frame_feature_cache_size: Number of frame encodings kept per session.

    Raises:
        ValueError: For an invalid cache size or threshold.
    """

    def __init__(
        self,
        image_model: object,
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        processor_factory: ProcessorFactory | None = None,
        frame_feature_cache_size: int = 4,
    ) -> None:
        """Configure independent-frame inference and its bounded feature cache."""
        self.image_model = image_model
        self.resolution = resolution
        self.confidence_threshold = _validate_threshold(confidence_threshold)
        self.processor_factory: ProcessorFactory = (
            _default_processor if processor_factory is None else processor_factory
        )
        self.frame_feature_cache_size = _validate_frame_feature_cache_size(
            frame_feature_cache_size
        )

    def init_state(
        self,
        resource_path: str | os.PathLike[str] | Sequence[Image.Image],
        track: bool = False,
    ) -> VideoInferenceState:
        """Open a video (file, image folder, image file or list of PIL images) as a session without a prompt.

        Frames come from ``load_video_frames`` and are resized by the session's
        image processor. With ``track``, ``propagate_in_video`` returns persistent
        track ids (see there).
        """
        processor = self.processor_factory(
            self.image_model,
            resolution=self.resolution,
            confidence_threshold=self.confidence_threshold,
        )
        frames = load_video_frames(resource_path)
        return {
            "frames": frames,
            "num_frames": len(frames),
            "orig_height": frames.orig_height,
            "orig_width": frames.orig_width,
            "processor": processor,
            "frame_feature_cache": BoundedLRUCache(self.frame_feature_cache_size),
            "text_prompt": None,
            "box_prompt": None,
            "box_labels": None,
            "point_prompt": None,
            "point_labels": None,
            "prompt_frame_idx": None,
            "tracker": VideoTracker() if track else None,
        }

    def add_prompt(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        boxes_xywh: object | None = None,
        box_labels: object | None = None,
        points: object | None = None,
        point_labels: object | None = None,
        rel_coordinates: bool = True,
        output_prob_thresh: float | None = None,
    ) -> tuple[int, VideoOutputs]:
        """Set the session's prompt, replacing any earlier one, and segment ``frame_idx``.

        Text, boxes and points may be combined; they are applied in that order. In a
        tracked session the new prompt also ends the current tracking run.

        Args:
            inference_state: Session from ``init_state``.
            frame_idx: Frame the prompt applies to.
            text_str: Text prompt.
            boxes_xywh: ``(N, 4)`` or ``(4,)`` boxes as (x, y, w, h) with (x, y) the
                top-left corner, normalized to [0, 1] when ``rel_coordinates`` else in
                pixels of the original frame.
            box_labels: ``(N,)`` booleans (True positive, False negative); all positive
                by default. Values are cast to bool, so any nonzero number is True.
            points: ``(N, 2)`` (x, y) points with the same units as the boxes.
            point_labels: ``(N,)`` booleans for the points; all positive by default.
            rel_coordinates: Whether box and point coordinates are normalized.
            output_prob_thresh: Score threshold for returned detections; defaults to
                ``confidence_threshold``.

        Returns:
            ``(frame_idx, outputs)``; object ids are frame-local indices.

        Raises:
            ValueError: If no prompt is given, shapes or labels are invalid,
                coordinates fall outside the image, or the threshold is outside [0, 1].
            IndexError: If ``frame_idx`` is out of range.
        """
        self._assert_frame_idx(inference_state, frame_idx, name="frame_index")
        if text_str is None and boxes_xywh is None and points is None:
            raise ValueError("add_prompt requires text, boxes_xywh, or points.")

        boxes_cxcywh, labels = _coerce_boxes(
            boxes_xywh=boxes_xywh,
            box_labels=box_labels,
            rel_coordinates=rel_coordinates,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )
        points_xy, labels_point = _coerce_points(
            points=points,
            point_labels=point_labels,
            rel_coordinates=rel_coordinates,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )

        inference_state["text_prompt"] = text_str
        inference_state["box_prompt"] = boxes_cxcywh
        inference_state["box_labels"] = labels
        inference_state["point_prompt"] = points_xy
        inference_state["point_labels"] = labels_point
        inference_state["prompt_frame_idx"] = frame_idx
        tracker = inference_state["tracker"]
        if tracker is not None:
            tracker.reset()
        outputs = self._run_frame_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            output_prob_thresh=output_prob_thresh,
        )
        return frame_idx, outputs

    def propagate_in_video(
        self,
        inference_state: VideoInferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
        reverse: bool = False,
    ) -> Generator[tuple[int, VideoOutputs], None, None]:
        """Yield ``(frame_idx, outputs)`` by re-running the prompt on each frame in order.

        Forward includes ``start_frame_idx`` (default: the prompt's frame) and up to
        ``max_frame_num_to_track`` subsequent frames; reverse excludes the start and
        yields up to that many preceding frames. Thus a zero maximum still yields the
        start in forward mode and no frames in reverse mode.

        In a tracked session each frame's detections pass through ByteTrack, and the
        outputs hold the confirmed tracks, in id order, each with the mask, box and
        score of the detection it matched on that frame. Detections scoring at
        least the output threshold (``output_prob_thresh`` or
        ``confidence_threshold``) start and rescue tracks; a tracked object also
        continues through a detection scoring down to ``LOW_SCORE_THRESH`` (0.1),
        so such a detection can be returned below the threshold. A new track is
        returned once it has matched on three more frames, except that tracks found
        on the first frame of a run are returned at once. A track without a match
        is not returned on that frame but keeps its id for 30 frames.
        Tracking runs over consecutive frames: a pass continues the run if its
        first frame follows the last tracked frame in the same direction, or
        follows the run's first frame in the opposite direction (the second half of
        a two-way propagation, which shares the start frame's tracks and ids).
        Any other start begins a new run with ids from 0, as does ``add_prompt``.

        Raises:
            RuntimeError: If no prompt was added.
            ValueError: If ``max_frame_num_to_track`` is negative.
            IndexError: If the start frame is out of range.
        """
        tracker = inference_state["tracker"]
        for frame_idx in self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            if tracker is None:
                yield frame_idx, self._run_frame_prompt(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    output_prob_thresh=output_prob_thresh,
                )
                continue
            threshold = self._output_threshold(output_prob_thresh)
            outputs = self._run_frame_prompt(
                inference_state=inference_state,
                frame_idx=frame_idx,
                output_prob_thresh=min(threshold, LOW_SCORE_THRESH),
            )
            yield frame_idx, _track_outputs(
                tracker,
                frame_idx,
                outputs,
                reverse=reverse,
                track_thresh=threshold,
                orig_height=inference_state["orig_height"],
                orig_width=inference_state["orig_width"],
            )

    def _get_processing_order(
        self,
        inference_state: VideoInferenceState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> range:
        """Select frame indices in the requested propagation direction and limit."""
        prompt_frame_idx = inference_state["prompt_frame_idx"]
        if prompt_frame_idx is None:
            raise RuntimeError(
                "No prompt has been added. Add a prompt on a frame before propagation."
            )
        start = prompt_frame_idx if start_frame_idx is None else start_frame_idx
        self._assert_frame_idx(inference_state, start, name="start_frame_index")
        num_frames = inference_state["num_frames"]
        max_frames = num_frames if max_frame_num_to_track is None else max_frame_num_to_track
        if max_frames < 0:
            raise ValueError("max_frame_num_to_track must be non-negative.")
        if reverse:
            end = max(start - max_frames, 0)
            return range(start - 1, end - 1, -1)
        end = min(start + max_frames, num_frames - 1)
        return range(start, end + 1)

    def _run_frame_prompt(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        output_prob_thresh: float | None,
    ) -> VideoOutputs:
        """Segment one frame with the session's text, box and point prompts."""
        processor = inference_state["processor"]
        frame_state = self._frame_image_state(inference_state, frame_idx)
        text_prompt = inference_state["text_prompt"]
        if text_prompt is not None:
            frame_state = processor.set_text_prompt(
                text_prompt, frame_state, run_grounding=False
            )
        boxes_cxcywh = inference_state["box_prompt"]
        box_labels = inference_state["box_labels"]
        if boxes_cxcywh is not None and box_labels is not None:
            for box, label in zip(boxes_cxcywh, box_labels, strict=True):
                frame_state = processor.add_geometric_prompt(
                    [float(value) for value in box],
                    bool(label),
                    frame_state,
                    run_grounding=False,
                )
        points = inference_state["point_prompt"]
        point_labels = inference_state["point_labels"]
        if points is not None and point_labels is not None:
            for point, label in zip(points, point_labels, strict=True):
                frame_state = processor.add_point_prompt(
                    [float(value) for value in point],
                    bool(label),
                    frame_state,
                    run_grounding=False,
                )
        processor.confidence_threshold = self._output_threshold(output_prob_thresh)
        frame_state = processor.predict(frame_state)
        return _state_to_video_outputs(
            frame_state,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )

    def _output_threshold(self, output_prob_thresh: float | None) -> float:
        """Return the requested score threshold, or the model default when None."""
        if output_prob_thresh is None:
            return self.confidence_threshold
        return _validate_threshold(output_prob_thresh)

    def _frame_image_state(
        self, inference_state: VideoInferenceState, frame_idx: int,
    ) -> ProcessorState:
        """Return a copy of the frame's cached image state, encoding the frame on a miss.

        The top-level dictionary and its ``backbone_out`` mapping are copied, so
        prompts never change the cached encoding; nested values are shared. A failed
        encoding does not insert a cache entry.
        """
        cache = inference_state["frame_feature_cache"]
        encoded = cache.get(frame_idx)
        if encoded is None:
            frame = inference_state["frames"][frame_idx]
            encoded = inference_state["processor"].set_image(frame)
            cache[frame_idx] = encoded
        frame_state = dict(encoded)
        if "backbone_out" in frame_state:
            frame_state["backbone_out"] = dict(
                cast(Mapping[str, object], frame_state["backbone_out"])
            )
        return frame_state

    @staticmethod
    def _assert_frame_idx(
        inference_state: VideoInferenceState,
        frame_idx: int,
        name: str,
    ) -> None:
        """Reject frame indices outside the loaded video."""
        num_frames = inference_state["num_frames"]
        if not 0 <= frame_idx < num_frames:
            raise IndexError(
                f"{name} {frame_idx} is out of range for {num_frames} frames."
            )


def _coerce_boxes(
    boxes_xywh: object | None,
    box_labels: object | None,
    rel_coordinates: bool,
    orig_height: int,
    orig_width: int,
) -> tuple[PromptCoordinates | None, PromptLabels | None]:
    """Validate (x, y, w, h) boxes and return normalized (cx, cy, w, h) boxes with boolean labels.

    Pixel boxes (``rel_coordinates=False``) are divided by the original width and
    height. Omitted labels default to True. Returns ``(None, None)`` when there
    are no boxes.

    Raises:
        ValueError: For labels without boxes, wrong shape, non-finite values, boxes
            outside the image or a label count that does not match.
    """
    if boxes_xywh is None:
        if box_labels is not None:
            raise ValueError("bounding_box_labels require bounding_boxes.")
        return None, None

    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"bounding_boxes must have shape (N, 4), got {boxes.shape}.")
    if not np.isfinite(boxes).all():
        raise ValueError("bounding_boxes must contain only finite values.")

    if rel_coordinates:
        normalized = boxes
    else:
        scale = np.array([orig_width, orig_height, orig_width, orig_height])
        normalized = boxes / scale[None, :]
    if (
        (normalized[:, :2] < 0).any()
        or (normalized[:, 2:] < 0).any()
        or (normalized[:, 0] + normalized[:, 2] > 1).any()
        or (normalized[:, 1] + normalized[:, 3] > 1).any()
    ):
        raise ValueError("bounding_boxes must be within the image bounds.")

    if box_labels is None:
        labels = np.ones((normalized.shape[0],), dtype=bool)
    else:
        labels = np.asarray(box_labels, dtype=bool)
        if labels.ndim == 0:
            labels = labels[None]
        if labels.shape != (normalized.shape[0],):
            raise ValueError("bounding_box_labels must have one label per box.")

    boxes_cxcywh = normalized.astype(np.float32)
    boxes_cxcywh[:, 0] = normalized[:, 0] + normalized[:, 2] / 2.0
    boxes_cxcywh[:, 1] = normalized[:, 1] + normalized[:, 3] / 2.0
    return boxes_cxcywh, labels


def _coerce_points(
    points: object | None,
    point_labels: object | None,
    rel_coordinates: bool,
    orig_height: int,
    orig_width: int,
) -> tuple[PromptCoordinates | None, PromptLabels | None]:
    """Validate ``(N, 2)`` (x, y) points and return normalized points with boolean labels.

    Pixel points (``rel_coordinates=False``) are divided by the original width and
    height; omitted labels default to all positive. Returns ``(None, None)`` when
    there are no points.

    Raises:
        ValueError: For labels without points, wrong shape, non-finite values, points
            outside the image or a label count that does not match.
    """
    if points is None:
        if point_labels is not None:
            raise ValueError("point_labels require points.")
        return None, None

    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim == 1:
        points_array = points_array[None, :]
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points_array.shape}.")
    if not np.isfinite(points_array).all():
        raise ValueError("points must contain only finite values.")

    if rel_coordinates:
        normalized = points_array
    else:
        scale = np.array([orig_width, orig_height], dtype=np.float32)
        normalized = points_array / scale[None, :]
    if (normalized < 0).any() or (normalized > 1).any():
        raise ValueError("points must be within the image bounds.")

    if point_labels is None:
        labels = np.ones((normalized.shape[0],), dtype=bool)
    else:
        labels = np.asarray(point_labels, dtype=bool)
        if labels.ndim == 0:
            labels = labels[None]
        if labels.shape != (normalized.shape[0],):
            raise ValueError("point_labels must have one label per point.")
    return normalized, labels


def _state_to_video_outputs(
    frame_state: Mapping[str, object],
    orig_height: int,
    orig_width: int,
) -> VideoOutputs:
    """Convert processor results (``masks``, pixel ``xyxy`` ``boxes``, ``scores``) to video outputs.

    Masks may be ``(N, 1, H, W)`` or ``(N, H, W)``; boxes are normalized to the
    original frame as top-left ``xywh``.

    Raises:
        ValueError: If the shapes do not describe the same ``N`` objects.
    """
    masks_value = frame_state["masks"]
    boxes_value = frame_state["boxes"]
    scores_value = frame_state["scores"]
    evaluate_boundary(masks_value, boxes_value, scores_value)

    masks = np.asarray(masks_value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
    count = masks.shape[0]
    scores = np.asarray(scores_value, dtype=np.float32).reshape(-1)
    if scores.shape != (count,):
        raise ValueError(f"scores must have shape ({count},), got {scores.shape}.")
    boxes = np.asarray(boxes_value, dtype=np.float32)
    if boxes.shape != (count, 4):
        raise ValueError(
            f"boxes must have shape ({count}, 4) to match the masks, got {boxes.shape}."
        )
    boxes_xywh = np.empty_like(boxes)
    boxes_xywh[:, 0] = boxes[:, 0] / orig_width
    boxes_xywh[:, 1] = boxes[:, 1] / orig_height
    boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / orig_width
    boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / orig_height
    return {
        "out_obj_ids": np.arange(count, dtype=np.int64),
        "out_probs": scores,
        "out_boxes_xywh": boxes_xywh,
        "out_binary_masks": masks.astype(bool, copy=False),
    }


def _track_outputs(
    tracker: VideoTracker,
    frame_idx: int,
    outputs: VideoOutputs,
    *,
    reverse: bool,
    track_thresh: float,
    orig_height: int,
    orig_width: int,
) -> VideoOutputs:
    """Track one frame's detections and keep those of confirmed tracks, labeled with track ids."""
    boxes = outputs["out_boxes_xywh"].astype(np.float64)
    scale = np.array([orig_width, orig_height, orig_width, orig_height], dtype=np.float64)
    boxes_xyxy = np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1) * scale
    assignment = tracker.update(
        frame_idx, boxes_xyxy, outputs["out_probs"], reverse=reverse, track_thresh=track_thresh,
    )
    kept = assignment.detection_indices
    return {
        "out_obj_ids": assignment.track_ids,
        "out_probs": outputs["out_probs"][kept],
        "out_boxes_xywh": outputs["out_boxes_xywh"][kept],
        "out_binary_masks": outputs["out_binary_masks"][kept],
    }
