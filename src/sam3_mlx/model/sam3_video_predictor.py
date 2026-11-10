# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Session API for independent SAM 3 frame inference.

``Sam3VideoPredictor`` keeps sessions by id, serializes the requests of each session
and forwards them to a ``Sam3VideoInference`` model. The request and response
dictionaries follow the upstream SAM 3 video predictor API. Frames are segmented
independently; a session started with ``track=True`` links the detections of
propagated frames with ByteTrack and returns persistent object ids.
"""

from __future__ import annotations

import gc
import os
import uuid
from collections.abc import Generator, Sequence
from threading import Event, Lock, RLock
from typing import TypedDict

from PIL import Image

from sam3_mlx.model.sam3_video_inference import (
    Sam3VideoInference,
    VideoInferenceState,
    VideoOutputs,
)


class FrameResponse(TypedDict):
    """Result for one frame: its index and the model's output dictionary."""

    frame_index: int
    outputs: VideoOutputs


class _Session(TypedDict):
    """Predictor-owned lifecycle data around one model session state."""

    state: VideoInferenceState
    session_id: str
    lock: RLock
    cancel_event: Event
    propagation_active: bool
    closed: bool


class Sam3VideoPredictor:
    """Apply an image model to selected frames, optionally tracking objects across them.

    Each session has its own lock, so a long request on one session never blocks
    another. While a propagation stream is active, ``add_prompt`` and a second
    ``propagate_in_video`` on that session raise ``RuntimeError``.
    ``cancel_propagation`` stops the active stream after its current frame; the
    stream also ends when it is closed or the session is closed.
    """

    def __init__(self, model: Sam3VideoInference) -> None:
        """Wrap a selected-frame inference model with session management."""
        self.model = model
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = Lock()
        self._reserved_session_ids: set[str] = set()
        self._cancelled_session_ids: set[str] = set()

    def start_session(
        self,
        resource_path: str | os.PathLike[str] | Sequence[Image.Image],
        session_id: str | None = None,
        track: bool = False,
    ) -> dict[str, str]:
        """Open a session and return ``{"session_id": ...}``.

        A new UUID is used when ``session_id`` is None. The id is reserved while the
        video loads, so a concurrent ``close_session`` on it cancels the start. With
        ``track``, ``propagate_in_video`` returns persistent track ids, as described
        in ``Sam3VideoInference.propagate_in_video``; ``add_prompt`` still returns
        frame-local ids and starts a new tracking run.

        Raises:
            ValueError: If ``session_id`` is already in use.
            RuntimeError: If the session was closed while loading.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        with self._sessions_lock:
            if session_id in self._sessions or session_id in self._reserved_session_ids:
                raise ValueError(f"Session ID already exists: {session_id}")
            self._reserved_session_ids.add(session_id)

        loaded = False
        try:
            state = self.model.init_state(resource_path, track=track)
            loaded = True
        finally:
            if not loaded:
                with self._sessions_lock:
                    self._reserved_session_ids.discard(session_id)
                    self._cancelled_session_ids.discard(session_id)

        session: _Session = {
            "state": state,
            "session_id": session_id,
            "lock": RLock(),
            "cancel_event": Event(),
            "propagation_active": False,
            "closed": False,
        }
        with self._sessions_lock:
            self._reserved_session_ids.discard(session_id)
            cancelled = session_id in self._cancelled_session_ids
            self._cancelled_session_ids.discard(session_id)
            if not cancelled:
                self._sessions[session_id] = session
        if cancelled:
            error = RuntimeError(
                f"Session {session_id} was closed while it was loading; "
                "the loaded state was disposed."
            )
            try:
                self._dispose_session(session)
            except RuntimeError as dispose_error:
                raise error from dispose_error
            raise error
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: str | None = None,
        points: object | None = None,
        point_labels: object | None = None,
        bounding_boxes: object | None = None,
        bounding_box_labels: object | None = None,
        output_prob_thresh: float | None = None,
        rel_coordinates: bool = True,
    ) -> FrameResponse:
        """Set the session's prompt and return ``{"frame_index", "outputs"}`` for ``frame_idx``.

        Text, boxes and points may be combined. ``bounding_boxes`` are top-left
        ``xywh``; ``rel_coordinates`` selects normalized coordinates or
        original-frame pixels. ``output_prob_thresh`` defaults to the model's
        ``confidence_threshold``.

        Raises:
            RuntimeError: If the session is closed, missing or propagating.
        """
        session = self._get_session(session_id)
        with session["lock"]:
            self._ensure_idle(session, action="add_prompt")
            frame_idx, outputs = self.model.add_prompt(
                inference_state=session["state"],
                frame_idx=frame_idx,
                text_str=text,
                boxes_xywh=bounding_boxes,
                box_labels=bounding_box_labels,
                points=points,
                point_labels=point_labels,
                rel_coordinates=rel_coordinates,
                output_prob_thresh=output_prob_thresh,
            )
        return {"frame_index": frame_idx, "outputs": outputs}

    def cancel_propagation(self, session_id: str) -> dict[str, bool]:
        """Stop the session's active stream after its current frame; returns ``{"is_success": True}``.

        A stream that has not started yet is not affected.

        Raises:
            RuntimeError: If the session is missing.
        """
        self._get_session(session_id)["cancel_event"].set()
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id: str,
        propagation_direction: str = "both",
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
    ) -> Generator[FrameResponse, None, None]:
        """Stream ``FrameResponse`` items while re-running the prompt on each frame.

        ``propagation_direction`` is ``"both"`` (forward then backward), ``"forward"``
        or ``"backward"``. Each direction is a separate model stream; forward
        includes the start frame and backward excludes it, so no frame is yielded
        twice. ``start_frame_idx`` defaults to the prompt's
        frame. The stream starts, and is checked, when it is first advanced. In a
        tracked session, the backward half of ``"both"`` continues from the start
        frame's tracks, so ids agree across the two directions.

        Raises:
            RuntimeError: If the session is missing, closed or already streaming.
            ValueError: For an invalid ``propagation_direction``.
        """
        if propagation_direction not in {"both", "forward", "backward"}:
            raise ValueError(f"invalid propagation direction: {propagation_direction}")
        session = self._get_session(session_id)
        with session["lock"]:
            self._ensure_idle(session, action="propagate_in_video")
            session["cancel_event"].clear()
            session["propagation_active"] = True
        directions: list[bool] = []
        if propagation_direction in {"both", "forward"}:
            directions.append(False)
        if propagation_direction in {"both", "backward"}:
            directions.append(True)
        cancel_event = session["cancel_event"]
        try:
            for reverse in directions:
                frames = self.model.propagate_in_video(
                    inference_state=session["state"],
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    output_prob_thresh=output_prob_thresh,
                    reverse=reverse,
                )
                while not cancel_event.is_set():
                    with session["lock"]:
                        if session["closed"]:
                            return
                        item = next(frames, None)
                    if item is None:
                        break
                    frame_idx, outputs = item
                    yield {"frame_index": frame_idx, "outputs": outputs}
                if cancel_event.is_set():
                    return
        finally:
            with session["lock"]:
                session["propagation_active"] = False

    def close_session(
        self, session_id: str, run_gc_collect: bool = True
    ) -> dict[str, bool]:
        """Close a session and release its frames and cached frame encodings.

        A missing session id returns success; a session that is still loading is
        cancelled. With ``run_gc_collect``, Python garbage is collected and the
        MLX buffer cache cleared afterwards.

        Raises:
            RuntimeError: If closing the frame provider fails; the session is
                removed regardless.
        """
        with self._sessions_lock:
            if session_id in self._reserved_session_ids:
                self._cancelled_session_ids.add(session_id)
                return {"is_success": True}
            session = self._sessions.pop(session_id, None)
        if session is not None:
            self._dispose_session(session)
            if run_gc_collect:
                import mlx.core as mx

                gc.collect()
                mx.clear_cache()
        return {"is_success": True}

    def _get_session(self, session_id: str) -> _Session:
        """Return an open session by ID."""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(f"Cannot find session {session_id}")
        return session

    @staticmethod
    def _ensure_idle(session: _Session, *, action: str) -> None:
        """Reject requests on a closed session or one with an active stream."""
        if session["closed"]:
            raise RuntimeError(f"Session {session['session_id']} is closed")
        if session["propagation_active"]:
            raise RuntimeError(
                f"Cannot {action} while propagation is active on session "
                f"{session['session_id']!r}. Cancel propagation and exhaust "
                "or close the stream, or wait for it to finish."
            )

    @staticmethod
    def _dispose_session(session: _Session) -> None:
        """Stop the session's stream, drop its cached encodings and close its frames."""
        session["cancel_event"].set()
        with session["lock"]:
            session["closed"] = True
            session["propagation_active"] = False
            state = session["state"]
            state["frame_feature_cache"].clear()
            try:
                state["frames"].close()
            except (OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"Failed to close the frames of session {session['session_id']}."
                ) from error
