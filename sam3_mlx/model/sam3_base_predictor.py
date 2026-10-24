# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Session-based request dispatcher for the SAM 3 video predictors.

``Sam3BasePredictor`` owns the session lifecycle (create, expire, close, shut down),
serializes requests per session and forwards them to a model object that provides
``init_state``, ``add_prompt``, ``propagate_in_video`` and related methods. The
request and response dictionaries mirror the official SAM 3 predictor API.
"""

from __future__ import annotations

import gc
import inspect
import os
from threading import Event, RLock
import time
import uuid
from collections.abc import Callable, Generator, MutableMapping, Sequence
from typing import Literal, NotRequired, Protocol, TypedDict, cast, overload

from PIL import Image

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.api_contract import API_TARGET_REVISION


_CLEAR_CACHE_THRESHOLD = 80


class StartSessionRequest(TypedDict):
    """Open a session on a video file, a folder of frames or a list of PIL images."""

    type: Literal["start_session"]
    resource_path: str | os.PathLike[str] | Sequence[Image.Image]
    session_id: NotRequired[str | None]
    offload_video_to_cpu: NotRequired[bool]
    offload_state_to_cpu: NotRequired[bool]


class AddPromptRequest(TypedDict):
    """Add a text, box or point prompt on ``frame_index``.

    Boxes and points are forwarded to the model; with ``rel_coordinates`` they are
    normalized to [0, 1], otherwise in pixels of the original frame.
    """

    type: Literal["add_prompt"]
    session_id: str
    frame_index: int
    text: NotRequired[str | None]
    points: NotRequired[object]
    point_labels: NotRequired[object]
    clear_old_points: NotRequired[bool]
    bounding_boxes: NotRequired[object]
    bounding_box_labels: NotRequired[object]
    clear_old_boxes: NotRequired[bool]
    output_prob_thresh: NotRequired[float | None]
    obj_id: NotRequired[int | None]
    rel_coordinates: NotRequired[bool]


class AddMaskRequest(TypedDict):
    """Condition tracked object ``obj_id`` with a 2-D ``mask`` on ``frame_index``."""

    type: Literal["add_mask"]
    session_id: str
    frame_index: int
    obj_id: int
    mask: object


class RemoveObjectRequest(TypedDict):
    """Remove object ``obj_id`` from the session."""

    type: Literal["remove_object"]
    session_id: str
    obj_id: int
    frame_index: NotRequired[int]
    is_user_action: NotRequired[bool]


class ResetSessionRequest(TypedDict):
    """Clear the session's prompts and results, keeping the loaded video."""

    type: Literal["reset_session"]
    session_id: str


class CancelPropagationRequest(TypedDict):
    """Stop a running propagation stream on the session."""

    type: Literal["cancel_propagation"]
    session_id: str


class CloseSessionRequest(TypedDict):
    """Close the session; ``clear_cache_threshold`` is a percentage (0 to 100) of the memory budget in use above which the MLX cache is cleared."""

    type: Literal["close_session"]
    session_id: str
    run_gc_collect: NotRequired[bool]
    clear_cache_threshold: NotRequired[int]


class PropagateInVideoRequest(TypedDict):
    """Stream per-frame results; handled by ``handle_stream_request``."""

    type: Literal["propagate_in_video"]
    session_id: str
    propagation_direction: NotRequired[Literal["both", "forward", "backward"]]
    start_frame_index: NotRequired[int | None]
    max_frame_num_to_track: NotRequired[int | None]
    output_prob_thresh: NotRequired[float | None]
    force_tracker_propagation: NotRequired[bool]
    evict_cached_frame_outputs: NotRequired[bool]


type PredictorRequest = (
    StartSessionRequest
    | AddPromptRequest
    | AddMaskRequest
    | RemoveObjectRequest
    | ResetSessionRequest
    | CancelPropagationRequest
    | CloseSessionRequest
)


class _MlxMemorySnapshot(TypedDict):
    backend: str
    memory_scope: str
    total_bytes: int
    free_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    cache_bytes: int
    free_pct: float
    active_session_count: int


class PredictorModelState(TypedDict, total=False):
    """Fields the dispatcher owns or reads from an otherwise model-owned state."""

    frames: object
    orig_height: int
    orig_width: int
    num_frames: int
    cached_frame_outputs: MutableMapping[int, object]
    _cancel_event: Event


class FrameResponse(TypedDict):
    """Result for one frame: its index and the model's output dictionary."""

    frame_index: int
    outputs: object


class _FrameDimensions(TypedDict):
    orig_height: int
    orig_width: int


class CloseSessionResponse(TypedDict):
    """Close result; ``gpu_mem`` (a memory snapshot) is present when ``run_gc_collect`` ran."""

    is_success: bool
    gpu_mem: NotRequired[_MlxMemorySnapshot]


class PredictorSession(TypedDict):
    """Predictor-owned lifecycle data around a model-specific state dictionary."""

    state: PredictorModelState
    session_id: str
    created_monotonic: float
    last_used_monotonic: float
    lock: RLock
    cancel_event: Event
    state_version: int
    propagation_active: bool
    closing: bool
    closed: bool


class _SessionModel(Protocol):
    def init_state(self, **kwargs: object) -> PredictorModelState: ...

    def add_prompt(self, **kwargs: object) -> tuple[int, object]: ...

    def remove_object(
        self, *args: object, **kwargs: object
    ) -> object | tuple[object, object] | None: ...

    def cancel_propagation(self, inference_state: PredictorModelState) -> None: ...

    def propagate_in_video(
        self, **kwargs: object
    ) -> Generator[tuple[int, object], None, None]: ...

    def reset_state(self, inference_state: PredictorModelState) -> None: ...


class Sam3BasePredictor:
    """Torch-free request dispatcher matching the official SAM3 video API.

    Sessions are identified by string ids and each has its own lock. A session
    expires after ``session_expiration_sec`` of inactivity when that attribute is a
    positive number (checked when a session is looked up or started). While a
    propagation stream is active, calls that change the session raise
    ``RuntimeError``; ``cancel_propagation`` or finishing the stream releases it.
    ``model`` must be set by a subclass.
    """

    async_loading_frames: bool
    default_output_prob_thresh: float = 0.5
    session_expiration_sec: float | None
    video_loader_type: str

    def __init__(self) -> None:
        self.model: object | None = None
        self._all_inference_states: dict[str, PredictorSession] = {}
        self._sessions_lock = RLock()
        self._reserved_session_ids: set[str] = set()
        self._cancelled_session_ids: set[str] = set()
        self._predictor_shutdown = False

    @property
    def is_shutdown(self) -> bool:
        """Whether ``shutdown`` has been called."""
        with self._sessions_lock:
            return self._predictor_shutdown

    @overload
    def handle_request(self, request: StartSessionRequest) -> dict[str, str]: ...

    @overload
    def handle_request(
        self, request: AddPromptRequest | AddMaskRequest
    ) -> FrameResponse: ...

    @overload
    def handle_request(
        self, request: RemoveObjectRequest
    ) -> FrameResponse | dict[str, bool]: ...

    @overload
    def handle_request(
        self, request: ResetSessionRequest | CancelPropagationRequest
    ) -> dict[str, bool]: ...

    @overload
    def handle_request(self, request: CloseSessionRequest) -> CloseSessionResponse: ...

    def handle_request(
        self, request: PredictorRequest
    ) -> dict[str, str] | FrameResponse | dict[str, bool] | CloseSessionResponse:
        """Dispatch a tagged request using the native session lifecycle."""
        request_type = request["type"]
        if request["type"] == "start_session":
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id"),
                offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                offload_state_to_cpu=request.get("offload_state_to_cpu", False),
            )
        if request["type"] == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text"),
                points=request.get("points"),
                point_labels=request.get("point_labels"),
                clear_old_points=request.get("clear_old_points", True),
                bounding_boxes=request.get("bounding_boxes"),
                bounding_box_labels=request.get("bounding_box_labels"),
                clear_old_boxes=request.get("clear_old_boxes", True),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    self.default_output_prob_thresh,
                ),
                obj_id=request.get("obj_id"),
                rel_coordinates=request.get("rel_coordinates", True),
            )
        if request["type"] == "add_mask":
            return self.add_mask(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                obj_id=request["obj_id"],
                mask=request["mask"],
            )
        if request["type"] == "remove_object":
            return self.remove_object(
                session_id=request["session_id"],
                frame_idx=request.get("frame_index", 0),
                obj_id=request["obj_id"],
                is_user_action=request.get("is_user_action", True),
            )
        if request["type"] == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        if request["type"] == "cancel_propagation":
            return self.cancel_propagation(session_id=request["session_id"])
        if request["type"] == "close_session":
            return self.close_session(
                session_id=request["session_id"],
                run_gc_collect=request.get("run_gc_collect", True),
                clear_cache_threshold=request.get(
                    "clear_cache_threshold", _CLEAR_CACHE_THRESHOLD
                ),
            )
        raise RuntimeError(f"invalid request type: {request_type}")

    def handle_stream_request(
        self, request: PropagateInVideoRequest
    ) -> Generator[FrameResponse, None, None]:
        """Dispatch a ``propagate_in_video`` request and yield one ``FrameResponse`` per frame.

        Raises:
            RuntimeError: For an unknown request type.
        """
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_index"),
                max_frame_num_to_track=request.get("max_frame_num_to_track"),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    self.default_output_prob_thresh,
                ),
                force_tracker_propagation=request.get(
                    "force_tracker_propagation", False
                ),
                evict_cached_frame_outputs=request.get(
                    "evict_cached_frame_outputs", False
                ),
            )
            return
        raise RuntimeError(f"invalid request type: {request_type}")

    def start_session(
        self,
        resource_path: str | os.PathLike[str] | Sequence[Image.Image],
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        """Open a session and return ``{"session_id": ...}``.

        A new UUID is used when ``session_id`` is None. The id is reserved while the
        video loads, so a concurrent ``close_session`` on it cancels the start.

        Raises:
            ValueError: If ``session_id`` is already in use.
            RuntimeError: If the predictor is shut down, or was shut down or the id
                was closed while loading.
        """
        model = self._require_model()
        self._sweep_expired_sessions()
        if session_id is None:
            session_id = str(uuid.uuid4())
        with self._sessions_lock:
            if self._predictor_shutdown:
                raise RuntimeError(
                    "Predictor has been shut down and cannot start new sessions."
                )
            if (
                session_id in self._all_inference_states
                or session_id in self._reserved_session_ids
            ):
                raise ValueError(f"Session ID already exists: {session_id}")
            self._cancelled_session_ids.discard(session_id)
            self._reserved_session_ids.add(session_id)

        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        initialization_succeeded = False
        try:
            inference_state = model.init_state(**init_kwargs)
            initialization_succeeded = True
        finally:
            if not initialization_succeeded:
                with self._sessions_lock:
                    self._reserved_session_ids.discard(session_id)
                    self._cancelled_session_ids.discard(session_id)

        now = time.monotonic()
        cancel_event = Event()
        inference_state["_cancel_event"] = cancel_event
        session: PredictorSession = {
            "state": inference_state,
            "session_id": session_id,
            "created_monotonic": now,
            "last_used_monotonic": now,
            "lock": RLock(),
            "cancel_event": cancel_event,
            "state_version": 0,
            "propagation_active": False,
            "closing": False,
            "closed": False,
        }

        publish_error: RuntimeError | None = None
        with self._sessions_lock:
            self._reserved_session_ids.discard(session_id)
            cancelled = session_id in self._cancelled_session_ids
            self._cancelled_session_ids.discard(session_id)
            if self._predictor_shutdown:
                publish_error = RuntimeError(
                    "Predictor was shut down while the session was loading; "
                    "the loaded state was disposed."
                )
            elif cancelled:
                publish_error = RuntimeError(
                    f"Session {session_id} was closed while it was loading; "
                    "the loaded state was disposed."
                )
            elif session_id in self._all_inference_states:
                publish_error = RuntimeError(
                    f"Session ID became occupied while loading: {session_id}"
                )
            else:
                self._all_inference_states[session_id] = session

        if publish_error is not None:
            try:
                self._dispose_session(session)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to dispose unpublished session {session_id}."
                ) from exc
            raise publish_error
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: str | None = None,
        points: object | None = None,
        point_labels: object | None = None,
        clear_old_points: bool = True,
        bounding_boxes: object | None = None,
        bounding_box_labels: object | None = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float | None = None,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
    ) -> FrameResponse:
        """Forward a prompt to the model and return ``{"frame_index", "outputs"}``.

        Arguments the model's ``add_prompt`` does not accept are dropped; asking for
        ``clear_old_points=False``, ``clear_old_boxes=False`` or a caller-supplied
        ``obj_id`` when the model cannot honor it raises the unsupported-feature error.
        ``bounding_boxes`` are passed as ``boxes_xywh`` and ``text`` as ``text_str``.

        Raises:
            RuntimeError: If the session is closed, missing or propagating.
        """
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._extend_expiration_time(session)
            self._ensure_session_idle(session, action="add_prompt")
            prompt_kwargs = dict(
                inference_state=session["state"],
                frame_idx=frame_idx,
                text_str=text,
                points=points,
                point_labels=point_labels,
                clear_old_points=clear_old_points,
                boxes_xywh=bounding_boxes,
                box_labels=bounding_box_labels,
                clear_old_boxes=clear_old_boxes,
                output_prob_thresh=(
                    self.default_output_prob_thresh
                    if output_prob_thresh is None
                    else output_prob_thresh
                ),
                rel_coordinates=rel_coordinates,
            )
            if obj_id is not None:
                prompt_kwargs["obj_id"] = obj_id

            signature = inspect.signature(model.add_prompt)
            valid_params = set(signature.parameters)
            if not clear_old_points and "clear_old_points" not in valid_params:
                raise_unsupported(
                    "Sam3BasePredictor.add_prompt(clear_old_points=False)",
                    reason="video-tracker",
                    detail="The selected-frame MLX path resets point prompts per request.",
                    alternative="clear_old_points=True",
                )
            if not clear_old_boxes and "clear_old_boxes" not in valid_params:
                raise_unsupported(
                    "Sam3BasePredictor.add_prompt(clear_old_boxes=False)",
                    reason="video-tracker",
                    detail="The selected-frame MLX path resets box prompts per request.",
                    alternative="clear_old_boxes=True",
                )
            if obj_id is not None and "obj_id" not in valid_params:
                raise_unsupported(
                    "Sam3BasePredictor.add_prompt(obj_id)",
                    reason="video-tracker",
                    detail="The selected model cannot honor caller-supplied object IDs.",
                    alternative="Omit obj_id.",
                )
            filtered_kwargs = {
                key: value
                for key, value in prompt_kwargs.items()
                if key in valid_params
            }

            frame_idx, outputs = model.add_prompt(**filtered_kwargs)
            self._reject_or_invalidate_if_propagating(session, action="add_prompt")
        return {"frame_index": frame_idx, "outputs": outputs}

    def add_mask(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        mask: object,
    ) -> FrameResponse:
        """Condition a tracker object when the selected model supports masks."""
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._ensure_session_idle(session, action="add_mask")
            add_mask = getattr(model, "add_tracker_new_mask", None)
            if not callable(add_mask):
                raise_unsupported(
                    "Sam3BasePredictor.add_mask",
                    reason="video-tracker",
                    detail=(
                        f"{type(model).__name__} has no callable add_tracker_new_mask. "
                        "Mask conditioning requires an instance-interactivity tracker; "
                        "the multiplex model does not expose this upstream API."
                    ),
                    alternative="Use a point or box prompt.",
                    upstream_commit=API_TARGET_REVISION,
                )
            import mlx.core as mx

            from sam3_mlx.mlx_runtime import mlx_array, to_numpy

            mask_action = cast(Callable[..., tuple[int, object]], add_mask)
            mask_kwargs = dict(
                inference_state=session["state"],
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )
            inspect.signature(mask_action).bind(**mask_kwargs)
            mask_array = mlx_array(to_numpy(mask, dtype=bool), dtype=mx.bool_)
            if mask_array.ndim != 2:
                raise ValueError(f"mask must be HxW, got {mask_array.shape}.")
            mask_kwargs["mask"] = mask_array
            result_frame, outputs = mask_action(**mask_kwargs)
            session["state_version"] += 1
            self._extend_expiration_time(session)
        return {"frame_index": result_frame, "outputs": outputs}

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ) -> FrameResponse | dict[str, bool]:
        """Remove an object and return the resulting frame outputs.

        When the model returns nothing, empty outputs (no objects) at the original
        frame size are returned for ``frame_idx``.

        Raises:
            RuntimeError: If the session is closed, missing or propagating.
        """
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._extend_expiration_time(session)
            self._ensure_session_idle(session, action="remove_object")
            result = model.remove_object(
                session["state"],
                obj_id=obj_id,
                frame_idx=frame_idx,
                is_user_action=is_user_action,
            )
            self._reject_or_invalidate_if_propagating(session, action="remove_object")
            if result is None or (isinstance(result, tuple) and result[1] is None):
                import numpy as np

                state = cast(_FrameDimensions, session["state"])
                outputs = {
                    "out_obj_ids": np.zeros(0, dtype=np.int64),
                    "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                    "out_binary_masks": np.zeros(
                        (
                            0,
                            int(state["orig_height"]),
                            int(state["orig_width"]),
                        ),
                        dtype=bool,
                    ),
                }
            elif isinstance(result, tuple):
                outputs = cast(tuple[object, object], result)[1]
            else:
                outputs = result
        return {"frame_index": frame_idx, "outputs": outputs}

    def cancel_propagation(self, session_id: str) -> dict[str, bool]:
        """Signal the session's active propagation (if any) to stop; returns ``{"is_success": True}``."""
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._extend_expiration_time(session)
            session["cancel_event"].set()
            if hasattr(self.model, "cancel_propagation"):
                model.cancel_propagation(session["state"])
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id: str,
        propagation_direction: str = "both",
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
        force_tracker_propagation: bool = False,
        evict_cached_frame_outputs: bool = False,
        **kwargs: object,
    ) -> Generator[FrameResponse, None, None]:
        """Stream ``FrameResponse`` items while propagating prompts through the video.

        ``propagation_direction`` is ``"both"`` (forward then backward), ``"forward"``
        or ``"backward"``. With ``evict_cached_frame_outputs`` each frame's entry is
        removed from the model's cache once yielded. Extra keyword arguments must be
        declared by the model's ``propagate_in_video``. State-changing calls
        (``add_prompt``, ``add_mask``, ``remove_object``, ``reset_session``) are
        rejected with ``RuntimeError`` while the stream is active.

        Raises:
            RuntimeError: If the session already has an active stream, or its state
                version changed during iteration.
            ValueError: For an invalid ``propagation_direction``.
        """
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._extend_expiration_time(session)
            if session.get("propagation_active"):
                raise RuntimeError(
                    f"Session {session_id} already has an active propagation stream."
                )
            if propagation_direction not in {"both", "forward", "backward"}:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )
            propagate_kwargs = self._prepare_propagation_kwargs(
                model,
                session["state"],
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                output_prob_thresh=output_prob_thresh,
                force_tracker_propagation=force_tracker_propagation,
                extra_kwargs=kwargs,
            )
            if evict_cached_frame_outputs:
                self._cached_frame_output_store(session["state"])
            session["cancel_event"].clear()
            session["propagation_active"] = True
            start_version = session["state_version"]
        directions: list[bool] = []
        if propagation_direction in {"both", "forward"}:
            directions.append(False)
        if propagation_direction in {"both", "backward"}:
            directions.append(True)
        try:
            for reverse in directions:
                generator = model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=reverse,
                )
                while not session["cancel_event"].is_set():
                    try:
                        frame_idx, outputs = self._next_propagation_output(
                            session,
                            generator,
                            start_version,
                            evict_cached_frame_outputs,
                        )
                    except StopIteration:
                        break
                    yield {"frame_index": frame_idx, "outputs": outputs}
                if session["cancel_event"].is_set():
                    return
        finally:
            with session["lock"]:
                if not session["closed"]:
                    session["propagation_active"] = False

    def _prepare_propagation_kwargs(
        self,
        model: _SessionModel,
        inference_state: PredictorModelState,
        *,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        output_prob_thresh: float | None,
        force_tracker_propagation: bool,
        extra_kwargs: dict[str, object],
    ) -> dict[str, object]:
        parameters = inspect.signature(model.propagate_in_video).parameters
        kwargs: dict[str, object] = {
            "inference_state": inference_state,
            "start_frame_idx": start_frame_idx,
            "max_frame_num_to_track": max_frame_num_to_track,
        }
        if "output_prob_thresh" in parameters:
            kwargs["output_prob_thresh"] = (
                self.default_output_prob_thresh
                if output_prob_thresh is None
                else output_prob_thresh
            )
        if "force_tracker_propagation" in parameters:
            kwargs["force_tracker_propagation"] = force_tracker_propagation
        elif force_tracker_propagation:
            raise_unsupported(
                "Sam3BasePredictor.propagate_in_video(force_tracker_propagation=True)",
                reason="video-tracker",
                detail=(
                    "The selected model does not expose forced temporal tracker "
                    "propagation. A variadic signature does not establish support."
                ),
                alternative="force_tracker_propagation=False",
                upstream_commit=API_TARGET_REVISION,
            )
        for key, value in extra_kwargs.items():
            if key not in parameters:
                raise_unsupported(
                    f"Sam3BasePredictor.propagate_in_video({key})",
                    reason="port-gap",
                    detail="The selected model does not declare this propagation control.",
                    alternative=f"Omit {key}.",
                    upstream_commit=API_TARGET_REVISION,
                )
            kwargs[key] = value
        return kwargs

    @staticmethod
    def _cached_frame_output_store(
        inference_state: PredictorModelState,
    ) -> MutableMapping[int, object] | None:
        if "cached_frame_outputs" not in inference_state:
            return None
        cache = cast(object, inference_state["cached_frame_outputs"])
        if not isinstance(cache, MutableMapping):
            raise TypeError(
                "cached_frame_outputs must be a mutable mapping for eviction."
            )
        return cast(MutableMapping[int, object], cache)

    def _next_propagation_output(
        self,
        session: PredictorSession,
        generator: Generator[tuple[int, object], None, None],
        start_version: int,
        evict_cached_frame_outputs: bool,
    ) -> tuple[int, object]:
        with session["lock"]:
            self._ensure_session_open(session)
            if session["state_version"] != start_version:
                raise RuntimeError(
                    "Session state changed during propagation; "
                    "refusing to emit mixed-prompt frames. "
                    "Cancel or wait for the stream to end before mutating prompts."
                )
            frame_idx, outputs = next(generator)
            if evict_cached_frame_outputs:
                output_cache = self._cached_frame_output_store(session["state"])
                if output_cache is not None:
                    output_cache.pop(frame_idx, None)
            self._extend_expiration_time(session)
        return frame_idx, outputs

    def reset_session(self, session_id: str) -> dict[str, bool]:
        """Reset the model state of the session (prompts and results); returns ``{"is_success": True}``."""
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._extend_expiration_time(session)
            self._reject_or_invalidate_if_propagating(session, action="reset_session")
            model.reset_state(session["state"])
        return {"is_success": True}

    @staticmethod
    def _reject_or_invalidate_if_propagating(
        session: PredictorSession,
        *,
        action: str,
    ) -> None:
        """Reject state mutators while a propagation stream is active."""
        Sam3BasePredictor._ensure_session_idle(session, action=action)
        session["state_version"] = int(session.get("state_version", 0)) + 1

    @staticmethod
    def _ensure_session_idle(session: PredictorSession, *, action: str) -> None:
        if session.get("propagation_active"):
            raise RuntimeError(
                f"Cannot {action} while propagation is active on session "
                f"{session.get('session_id')!r}. Call cancel_propagation first "
                "or wait for the stream to finish."
            )

    def close_session(
        self,
        session_id: str,
        run_gc_collect: bool = True,
        clear_cache_threshold: int = _CLEAR_CACHE_THRESHOLD,
    ) -> CloseSessionResponse:
        """Close a session; optionally collect and evict a pressured MLX cache.

        Memory telemetry describes process allocator headroom against Metal's
        recommended working-set budget, rather than system-wide free memory.
        """
        if (
            type(clear_cache_threshold) is not int
            or not 0 <= clear_cache_threshold <= 100
        ):
            raise ValueError("clear_cache_threshold must be an integer from 0 to 100.")
        result: CloseSessionResponse = {"is_success": True}
        session = None
        with self._sessions_lock:
            if session_id in self._reserved_session_ids:
                self._cancelled_session_ids.add(session_id)
                return result
            session = self._all_inference_states.get(session_id)
            if session is not None:
                with session["lock"]:
                    session["closing"] = True
                    self._all_inference_states.pop(session_id, None)
        if session is not None:
            self._dispose_session(session)
            if run_gc_collect:
                gc.collect()
                snapshot = self._gpu_mem_snapshot()
                if (100.0 - snapshot["free_pct"]) >= clear_cache_threshold:
                    import mlx.core as mx

                    mx.clear_cache()
                    snapshot = self._gpu_mem_snapshot()
                result["gpu_mem"] = snapshot
        return result

    def _gpu_mem_snapshot(self) -> _MlxMemorySnapshot:
        """Snapshot of MLX allocator usage against Metal's recommended working-set size (bytes)."""
        import mlx.core as mx
        from sam3_mlx.mlx_runtime import require_mlx_runtime

        require_mlx_runtime("Sam3BasePredictor.close_session memory policy")
        budget = mx.device_info(mx.gpu).get("max_recommended_working_set_size")
        if type(budget) is not int or budget <= 0:
            raise RuntimeError(
                "MLX Metal did not expose a positive working-set budget."
            )
        allocated, cached = mx.get_active_memory(), mx.get_cache_memory()
        if allocated < 0 or cached < 0:
            raise RuntimeError("MLX allocator memory counters must be non-negative.")
        reserved = allocated + cached
        headroom = max(0, budget - reserved)
        with self._sessions_lock:
            active_count = len(self._all_inference_states)
        return {
            "backend": "mlx-metal",
            "memory_scope": "process-allocator-vs-recommended-working-set",
            "total_bytes": budget,
            "free_bytes": headroom,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "cache_bytes": cached,
            "free_pct": headroom / budget * 100.0,
            "active_session_count": active_count,
        }

    def _get_session(self, session_id: str) -> PredictorSession:
        expired_session = None
        with self._sessions_lock:
            session = self._all_inference_states.get(session_id)
            if session is not None:
                with session["lock"]:
                    if self._session_is_expired(session):
                        session["closing"] = True
                        expired_session = self._all_inference_states.pop(
                            session_id, None
                        )
                        session = None
        if expired_session is not None:
            self._dispose_session(expired_session)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _require_model(self) -> _SessionModel:
        if self.model is None:
            raise RuntimeError("Sam3BasePredictor.model must be initialized.")
        return cast(_SessionModel, self.model)

    @staticmethod
    def _ensure_session_open(session: PredictorSession) -> None:
        if session["closing"] or session["closed"]:
            raise RuntimeError(f"Session {session['session_id']} is closed")

    def _extend_expiration_time(self, session: PredictorSession) -> None:
        session["last_used_monotonic"] = time.monotonic()

    def _session_is_expired(self, session: PredictorSession) -> bool:
        expiration_sec = getattr(self, "session_expiration_sec", None)
        if expiration_sec is None or expiration_sec <= 0:
            return False
        idle_seconds = time.monotonic() - session["last_used_monotonic"]
        return idle_seconds >= expiration_sec

    def _sweep_expired_sessions(self) -> None:
        expired: list[PredictorSession] = []
        with self._sessions_lock:
            for session_id, session in list(self._all_inference_states.items()):
                with session["lock"]:
                    if self._session_is_expired(session):
                        session["closing"] = True
                        popped = self._all_inference_states.pop(session_id, None)
                        if popped is not None:
                            expired.append(popped)
        for session in expired:
            self._dispose_session(session)

    @staticmethod
    def _dispose_session(session: PredictorSession) -> None:
        close_error: OSError | RuntimeError | None = None
        with session["lock"]:
            session["closing"] = True
            session["cancel_event"].set()
            session["propagation_active"] = False
            state = session["state"]
            frames = state.get("frames")
            try:
                close = getattr(frames, "close", None)
                if callable(close):
                    close()
            except (OSError, RuntimeError) as exc:
                close_error = exc
            finally:
                cast(dict[str, object], state).clear()
                session["closed"] = True
        if close_error is not None:
            raise RuntimeError(
                f"Failed to close frame provider for session {session['session_id']}."
            ) from close_error

    def _live_session_frame_counts(self) -> list[tuple[str, int]]:
        with self._sessions_lock:
            sessions = list(self._all_inference_states.items())
        snapshot: list[tuple[str, int]] = []
        for session_id, session in sessions:
            with session["lock"]:
                if session["closing"] or session["closed"]:
                    continue
                state = session["state"]
                if "num_frames" in state:
                    snapshot.append((session_id, int(state["num_frames"])))
        return snapshot

    def shutdown(self) -> None:
        """Close every session and refuse new ones; calling it again does nothing.

        Raises:
            RuntimeError: If disposing any session failed (after all were attempted).
        """
        with self._sessions_lock:
            if self._predictor_shutdown:
                return
            self._predictor_shutdown = True
            sessions = list(self._all_inference_states.values())
            for session in sessions:
                with session["lock"]:
                    session["closing"] = True
            self._all_inference_states.clear()
            self._reserved_session_ids.clear()
            self._cancelled_session_ids.clear()

        errors: list[RuntimeError] = []
        for session in sessions:
            try:
                self._dispose_session(session)
            except RuntimeError as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"Predictor shutdown encountered {len(errors)} disposal error(s)."
            ) from errors[0]
