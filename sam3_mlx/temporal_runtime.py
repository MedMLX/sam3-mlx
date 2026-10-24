"""Backend-neutral prompt scheduling for official and MLX temporal adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx.mask_metrics import as_binary_mask
from sam3_mlx.temporal_parity import (
    TemporalFrameOutput,
    TemporalRun,
    TemporalTrackOutput,
)


class TemporalModel[StateT: Mapping[str, object]](Protocol):
    """Predictor surface shared by the official PyTorch and MLX temporal adapters.

    Mirrors the upstream SAM 3.1 video predictor methods used by the pinned
    prompt timeline. Point and box coordinates are normalized to [0, 1].
    """

    def init_state(
        self,
        *,
        resource_path: list[Image.Image],
        offload_video_to_cpu: bool,
        async_loading_frames: bool,
        use_torchcodec: bool,
        use_cv2: bool,
        input_is_mp4: bool,
    ) -> StateT: ...

    def add_prompt(
        self,
        inference_state: StateT,
        /,
        *,
        frame_idx: int,
        obj_id: int | None = None,
        points: NDArray[np.float32] | None = None,
        point_labels: NDArray[np.int32] | None = None,
        clear_old_points: bool = True,
        rel_coordinates: bool = True,
        boxes_xywh: NDArray[np.float32] | None = None,
        box_labels: NDArray[np.int32] | None = None,
        clear_old_boxes: bool = True,
    ) -> object: ...

    def propagate_in_video(
        self,
        *,
        inference_state: StateT,
        start_frame_idx: int,
        max_frame_num_to_track: int,
        reverse: bool,
        is_last_batch: bool,
    ) -> Iterable[object]: ...

    def remove_object(
        self,
        inference_state: StateT,
        /,
        *,
        obj_id: int,
        frame_idx: int,
        strict: bool,
        is_user_action: bool,
    ) -> object: ...

    def reset_state(self, inference_state: StateT, /) -> object: ...


class PointArguments(TypedDict):
    obj_id: int
    points: NDArray[np.float32]
    point_labels: NDArray[np.int32]
    clear_old_points: bool
    rel_coordinates: bool


class BoxArguments(TypedDict):
    boxes_xywh: NDArray[np.float32]
    box_labels: NDArray[np.int32]
    clear_old_boxes: bool


@dataclass(frozen=True)
class TemporalPropagationObservation:
    """One propagation call: its frame range, frames actually produced, and wall time in seconds."""

    start_frame: int
    end_frame: int
    reverse: bool
    completed_frame_indices: tuple[int, ...]
    wall_seconds: float


def load_temporal_frames(
    sequence: Mapping[str, object], *, repo_root: Path
) -> list[Image.Image]:
    """Decode a manifest sequence's frames into PIL RGB images.

    ``sequence["asset"]`` is a path relative to ``repo_root`` of an ``.npz`` with a
    ``frames`` array of shape ``(T, H, W, 3)``, uint8 RGB. Pickles are not loaded.
    """

    asset = repo_root / str(sequence["asset"])
    with np.load(asset, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"], dtype=np.uint8)
    return [Image.fromarray(frame, mode="RGB") for frame in frames]


def _normal_output(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Temporal model output must be a mapping, got {type(value).__name__}."
        )
    return cast(Mapping[str, object], value)


def _unpack_result(value: object) -> tuple[int, Mapping[str, object]]:
    if not isinstance(value, tuple):
        raise TypeError("Temporal model results must be (frame_index, output) pairs.")
    values = cast(tuple[object, ...], value)
    if len(values) != 2:
        raise TypeError("Temporal model results must be (frame_index, output) pairs.")
    frame_index, output = values
    if not isinstance(frame_index, int):
        raise TypeError("Temporal model frame indices must be integers.")
    if output == "DUMMY_OUTPUT":
        raise RuntimeError("Temporal model returned distributed DUMMY_OUTPUT.")
    return frame_index, _normal_output(output)


def _output_arrays(
    output: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_ids = np.asarray(output.get("out_obj_ids"), dtype=np.int64)
    masks = as_binary_mask(
        np.asarray(output.get("out_binary_masks")), label="Temporal masks", ndim=3
    )
    scores = np.asarray(output.get("out_probs"), dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Temporal scores must contain only finite values.")
    if object_ids.ndim != 1 or scores.shape != object_ids.shape:
        raise ValueError("Temporal object IDs and scores must be aligned 1-D arrays.")
    if masks.ndim != 3 or masks.shape[0] != object_ids.shape[0]:
        raise ValueError(
            "Temporal masks must have shape (N, H, W) aligned to object IDs."
        )
    return object_ids, masks, scores


def _select_box_identity(
    output: Mapping[str, object],
    *,
    coordinates: Sequence[object],
    assigned_ids: set[int],
) -> int | None:
    # A box prompt does not say which runtime object it created. Pick the
    # unassigned object whose mask contains the box center (falling back to any
    # unassigned object), preferring the higher score, then the lower ID.
    object_ids, masks, scores = _output_arrays(output)
    if len(coordinates) != 4:
        raise ValueError("Box prompts must use normalized xywh coordinates.")
    numeric_coordinates: list[float] = []
    for value in coordinates:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("Box prompt coordinates must be numeric.")
        numeric_coordinates.append(float(value))
    x, y, width, height = numeric_coordinates
    center_x = min(max(x + width / 2.0, 0.0), 1.0)
    center_y = min(max(y + height / 2.0, 0.0), 1.0)
    candidates: list[tuple[int, float, bool]] = []
    for index, runtime_id in enumerate(object_ids.tolist()):
        if runtime_id in assigned_ids:
            continue
        row = min(int(center_y * masks.shape[1]), masks.shape[1] - 1)
        column = min(int(center_x * masks.shape[2]), masks.shape[2] - 1)
        candidates.append(
            (runtime_id, float(scores[index]), bool(masks[index, row, column]))
        )
    if not candidates:
        return None
    containing = [candidate for candidate in candidates if candidate[2]]
    ranked = containing or candidates
    return max(ranked, key=lambda candidate: (candidate[1], -candidate[0]))[0]


def _tracks_from_output(
    output: Mapping[str, object],
    *,
    semantic_to_runtime: Mapping[str, int],
) -> dict[str, TemporalTrackOutput]:
    object_ids, masks, scores = _output_arrays(output)
    index_by_runtime = {
        int(runtime_id): index for index, runtime_id in enumerate(object_ids.tolist())
    }
    tracks: dict[str, TemporalTrackOutput] = {}
    for semantic_id, runtime_id in semantic_to_runtime.items():
        index = index_by_runtime.get(runtime_id)
        if index is None:
            continue
        mask = masks[index]
        tracks[semantic_id] = TemporalTrackOutput(
            runtime_ids=(str(runtime_id),),
            mask=mask,
            confidence=float(scores[index]),
            logit_mean=None,
        )
    return tracks


def _point_arguments(event: Mapping[str, object], runtime_id: int) -> PointArguments:
    coordinates = np.asarray(event["coordinates"], dtype=np.float32)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    labels = np.asarray(event["labels"], dtype=np.int32)
    return {
        "obj_id": runtime_id,
        "points": coordinates,
        "point_labels": labels,
        "clear_old_points": event["operation"] in {"initial_point", "add_object_point"},
        "rel_coordinates": True,
    }


def _box_arguments(event: Mapping[str, object]) -> BoxArguments:
    coordinates = np.asarray(event["coordinates"], dtype=np.float32)
    if coordinates.shape != (4,):
        raise ValueError("Box prompts must contain one normalized xywh box.")
    return {
        "boxes_xywh": coordinates[None, :],
        "box_labels": np.asarray(event["labels"], dtype=np.int32),
        "clear_old_boxes": True,
    }


def _clear_tracking_bounds(inference_state: Mapping[str, object]) -> None:
    """Detach a new prompt from the preceding bounded propagation window."""

    feature_cache = inference_state.get("feature_cache")
    if isinstance(feature_cache, dict):
        cast(dict[object, object], feature_cache).pop("tracking_bounds", None)


def _empty_frame(frame_index: int, *, elapsed_ms: float = 0.0) -> TemporalFrameOutput:
    return TemporalFrameOutput(
        frame_index=frame_index,
        tracks={},
        inference_time_ms=elapsed_ms,
    )


def _record(
    records: dict[int, TemporalFrameOutput],
    result: object,
    *,
    semantic_to_runtime: Mapping[str, int],
    elapsed_ms: float | None,
    frame_observer: Callable[[TemporalFrameOutput], None] | None,
) -> tuple[int, Mapping[str, object]]:
    frame_index, output = _unpack_result(result)
    frame_output = TemporalFrameOutput(
        frame_index=frame_index,
        tracks=_tracks_from_output(output, semantic_to_runtime=semantic_to_runtime),
        inference_time_ms=elapsed_ms,
    )
    records[frame_index] = frame_output
    if frame_observer is not None:
        frame_observer(frame_output)
    return frame_index, output


def _propagate[StateT: Mapping[str, object]](
    model: TemporalModel[StateT],
    inference_state: StateT,
    records: dict[int, TemporalFrameOutput],
    semantic_to_runtime: Mapping[str, int],
    *,
    start_frame: int,
    end_frame: int,
    reverse: bool,
    synchronize: Callable[[], None] | None,
    frame_observer: Callable[[TemporalFrameOutput], None] | None,
    propagation_observer: Callable[[TemporalPropagationObservation], None] | None,
) -> None:
    if (not reverse and end_frame < start_frame) or (
        reverse and end_frame > start_frame
    ):
        return
    distance = abs(end_frame - start_frame)
    if synchronize is not None:
        synchronize()
    started_at = time.perf_counter()
    generator = iter(
        model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=distance,
            reverse=reverse,
            is_last_batch=True,
        )
    )
    completed_frame_indices: list[int] = []
    while True:
        try:
            result = next(generator)
        except StopIteration:
            break
        frame_index, _ = _record(
            records,
            result,
            semantic_to_runtime=semantic_to_runtime,
            elapsed_ms=None,
            frame_observer=frame_observer,
        )
        completed_frame_indices.append(frame_index)
    if synchronize is not None:
        synchronize()
    observation = TemporalPropagationObservation(
        start_frame=start_frame,
        end_frame=end_frame,
        reverse=reverse,
        completed_frame_indices=tuple(completed_frame_indices),
        wall_seconds=time.perf_counter() - started_at,
    )
    if propagation_observer is not None:
        propagation_observer(observation)


def run_temporal_model[StateT: Mapping[str, object]](
    model: TemporalModel[StateT],
    sequence: Mapping[str, object],
    frames: Sequence[Image.Image],
    *,
    offload_video_to_cpu: bool,
    raise_on_error: bool = False,
    synchronize: Callable[[], None] | None = None,
    frame_observer: Callable[[TemporalFrameOutput], None] | None = None,
    phase_observer: Callable[[str], None] | None = None,
    propagation_observer: Callable[[TemporalPropagationObservation], None]
    | None = None,
) -> TemporalRun:
    """Run the pinned prompt-event timeline through one backend.

    ``sequence`` supplies ``objects`` (with ``semantic_id``) and ``prompt_events``
    (frame index, operation, normalized coordinates and labels, propagation
    direction); ``frames`` are the decoded RGB frames. Objects are tracked by
    semantic ID, which is mapped to each backend's runtime object ID, so results
    from different backends can be compared by identity. ``synchronize`` is
    called before and after timed sections (use it to drain device work).

    A ``RuntimeError`` from the model yields a ``TemporalRun`` with
    ``state="failed"`` and the frames recorded so far, unless ``raise_on_error``;
    other exceptions propagate. Raises ``ValueError`` for malformed events.
    """

    objects = cast(list[dict[str, object]], sequence["objects"])
    assigned_runtime = {
        str(object_spec["semantic_id"]): 1000 + index
        for index, object_spec in enumerate(objects)
    }
    semantic_to_runtime: dict[str, int] = {}
    prompt_latencies: dict[str, float] = {}
    records: dict[int, TemporalFrameOutput] = {}
    try:
        inference_state = model.init_state(
            resource_path=list(frames),
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=False,
            use_torchcodec=False,
            use_cv2=False,
            input_is_mp4=False,
        )
        if phase_observer is not None:
            phase_observer("state-initialized")
        events = cast(list[dict[str, object]], sequence["prompt_events"])
        for event_index, event in enumerate(events):
            raw_frame_index = event["frame_index"]
            if not isinstance(raw_frame_index, int) or isinstance(
                raw_frame_index, bool
            ):
                raise ValueError("Temporal event frame_index must be an integer.")
            frame_index = raw_frame_index
            event_id = str(event["event_id"])
            semantic_id = str(event["semantic_id"])
            operation = str(event["operation"])
            if synchronize is not None:
                synchronize()
            start = time.perf_counter()
            if operation in {"remove_object", "reset"}:
                if operation == "remove_object":
                    runtime_id = semantic_to_runtime.get(semantic_id)
                    if runtime_id is None:
                        raise ValueError(
                            f"Cannot remove unbound semantic object {semantic_id}."
                        )
                    result = model.remove_object(
                        inference_state,
                        obj_id=runtime_id,
                        frame_idx=frame_index,
                        strict=True,
                        is_user_action=True,
                    )
                    semantic_to_runtime.pop(semantic_id)
                    if synchronize is not None:
                        synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    _record(
                        records,
                        result,
                        semantic_to_runtime=semantic_to_runtime,
                        elapsed_ms=elapsed_ms,
                        frame_observer=frame_observer,
                    )
                else:
                    model.reset_state(inference_state)
                    semantic_to_runtime.clear()
                    if synchronize is not None:
                        synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    records[frame_index] = _empty_frame(
                        frame_index,
                        elapsed_ms=elapsed_ms,
                    )
                    if frame_observer is not None:
                        frame_observer(records[frame_index])
                prompt_latencies[event_id] = elapsed_ms
            elif operation in {
                "initial_point",
                "add_object_point",
                "positive_negative_correction",
                "point_refinement",
            }:
                runtime_id = semantic_to_runtime.setdefault(
                    semantic_id, assigned_runtime[semantic_id]
                )
                point_arguments = _point_arguments(event, runtime_id)
                _clear_tracking_bounds(inference_state)
                result = model.add_prompt(
                    inference_state,
                    frame_idx=frame_index,
                    **point_arguments,
                )
                if synchronize is not None:
                    synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prompt_latencies[event_id] = elapsed_ms
                _record(
                    records,
                    result,
                    semantic_to_runtime=semantic_to_runtime,
                    elapsed_ms=elapsed_ms,
                    frame_observer=frame_observer,
                )
            elif operation in {"initial_box", "add_object_box", "box_refinement"}:
                box_arguments = _box_arguments(event)
                _clear_tracking_bounds(inference_state)
                result = model.add_prompt(
                    inference_state,
                    frame_idx=frame_index,
                    **box_arguments,
                )
                if synchronize is not None:
                    synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prompt_latencies[event_id] = elapsed_ms
                _, output = _unpack_result(result)
                assigned_ids = set(semantic_to_runtime.values())
                if operation == "box_refinement":
                    current_runtime_id = semantic_to_runtime.get(semantic_id)
                    if current_runtime_id is not None:
                        assigned_ids.discard(current_runtime_id)
                runtime_id = _select_box_identity(
                    output,
                    coordinates=cast(Sequence[object], event["coordinates"]),
                    assigned_ids=assigned_ids,
                )
                if runtime_id is not None:
                    semantic_to_runtime[semantic_id] = runtime_id
                _record(
                    records,
                    result,
                    semantic_to_runtime=semantic_to_runtime,
                    elapsed_ms=elapsed_ms,
                    frame_observer=frame_observer,
                )
            else:
                raise ValueError(
                    f"Unsupported temporal prompt operation {operation!r}."
                )
            if phase_observer is not None:
                phase_observer(f"prompt:{event_id}")
            propagation = str(event["propagation"])
            if propagation in {"backward", "both"} and frame_index > 0:
                _propagate(
                    model,
                    inference_state,
                    records,
                    semantic_to_runtime,
                    start_frame=frame_index,
                    end_frame=0,
                    reverse=True,
                    synchronize=synchronize,
                    frame_observer=frame_observer,
                    propagation_observer=propagation_observer,
                )
            next_frame = (
                cast(int, events[event_index + 1]["frame_index"])
                if event_index + 1 < len(events)
                else len(frames)
            )
            if operation == "reset":
                for empty_frame_index in range(frame_index, next_frame):
                    records[empty_frame_index] = _empty_frame(
                        empty_frame_index,
                        elapsed_ms=(
                            elapsed_ms if empty_frame_index == frame_index else 0.0
                        ),
                    )
                    if frame_observer is not None:
                        frame_observer(records[empty_frame_index])
            if propagation in {"forward", "both"} and next_frame > frame_index:
                _propagate(
                    model,
                    inference_state,
                    records,
                    semantic_to_runtime,
                    start_frame=frame_index,
                    end_frame=next_frame - 1,
                    reverse=False,
                    synchronize=synchronize,
                    frame_observer=frame_observer,
                    propagation_observer=propagation_observer,
                )
    except RuntimeError as exc:
        if raise_on_error:
            raise
        return TemporalRun(
            state="failed",
            frames=tuple(records[index] for index in sorted(records)),
            prompt_latencies_ms=prompt_latencies,
            reason=f"{type(exc).__name__}: {exc}",
        )
    if phase_observer is not None:
        phase_observer("complete")
    return TemporalRun(
        state="passed",
        frames=tuple(records[index] for index in sorted(records)),
        prompt_latencies_ms=prompt_latencies,
    )
