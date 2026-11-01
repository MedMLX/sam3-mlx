"""Session lifecycle: loading, closing, cancellation and per-session locking."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from threading import Event, Thread

import pytest
from PIL import Image

from sam3_mlx.model.sam3_video_inference import Sam3VideoInference, VideoInferenceState
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from tests.api._stubs import FRAME_SIZE, FakeImageProcessor

type Resource = str | os.PathLike[str] | Sequence[Image.Image]


def _frames(count: int = 1) -> list[Image.Image]:
    return [Image.new("RGB", FRAME_SIZE) for _ in range(count)]


def _predictor(model: Sam3VideoInference | None = None) -> Sam3VideoPredictor:
    if model is None:
        model = Sam3VideoInference(image_model=object(), processor_factory=FakeImageProcessor)
    return Sam3VideoPredictor(model)


class _BlockingInit(Sam3VideoInference):
    """Hold ``init_state`` until released, recording the states it creates."""

    def __init__(self) -> None:
        super().__init__(image_model=object(), processor_factory=FakeImageProcessor)
        self.started = Event()
        self.release = Event()
        self.states: list[VideoInferenceState] = []

    def init_state(self, resource_path: Resource, track: bool = False) -> VideoInferenceState:
        self.started.set()
        assert self.release.wait(timeout=2)
        state = super().init_state(resource_path, track=track)
        self.states.append(state)
        return state


class _FailingClose:
    """One-frame store whose ``close`` fails."""

    orig_height = FRAME_SIZE[1]
    orig_width = FRAME_SIZE[0]

    def __init__(self) -> None:
        self.closed = False

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Image.Image:
        return Image.new("RGB", FRAME_SIZE)

    def close(self) -> None:
        self.closed = True
        raise OSError("decoder close failed")


def test_close_session_prevents_inflight_session_publication(tmp_path: Path) -> None:
    """Closing a loading session disposes it while allowing the ID to be reused later."""
    Image.new("RGB", FRAME_SIZE).save(tmp_path / "0.png")
    model = _BlockingInit()
    predictor = _predictor(model)
    errors: list[str] = []

    def start() -> None:
        try:
            predictor.start_session(tmp_path, session_id="closing")
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = Thread(target=start)
    worker.start()
    assert model.started.wait(timeout=2)

    assert predictor.close_session("closing") == {"is_success": True}
    model.release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1 and "closed while it was loading" in errors[0]
    with pytest.raises(RuntimeError, match="closed"):
        model.states[0]["frames"][0]
    with pytest.raises(RuntimeError, match="Cannot find session"):
        predictor.add_prompt("closing", frame_idx=0, text="object")
    assert predictor.start_session(tmp_path, session_id="closing") == {"session_id": "closing"}
    assert predictor.close_session("closing", run_gc_collect=False) == {"is_success": True}


def test_close_provider_failure_still_removes_and_cancels_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop the stream and remove the session even if frame cleanup fails."""
    frames = _FailingClose()

    def load(resource: Resource) -> _FailingClose:
        del resource
        return frames

    monkeypatch.setattr("sam3_mlx.model.sam3_video_inference.load_video_frames", load)
    predictor = _predictor()
    predictor.start_session("ignored", session_id="bad-close")
    predictor.add_prompt("bad-close", frame_idx=0, text="object")
    stream = predictor.propagate_in_video("bad-close", propagation_direction="forward")
    assert next(stream)["frame_index"] == 0

    with pytest.raises(RuntimeError, match="Failed to close") as error:
        predictor.close_session("bad-close", run_gc_collect=False)

    assert isinstance(error.value.__cause__, OSError)
    assert frames.closed is True
    assert list(stream) == []
    with pytest.raises(RuntimeError, match="Cannot find session"):
        predictor.cancel_propagation("bad-close")


def test_busy_session_does_not_block_other_sessions() -> None:
    """Closing a session that is mid-request waits on that session only."""
    hold, entered, release = Event(), Event(), Event()

    class _BlockingProcessor(FakeImageProcessor):
        def set_image(self, image: object) -> dict[str, object]:
            if hold.is_set():
                entered.set()
                assert release.wait(timeout=5)
            return super().set_image(image)

    predictor = _predictor(
        Sam3VideoInference(image_model=object(), processor_factory=_BlockingProcessor)
    )
    predictor.start_session(_frames(), session_id="busy")
    predictor.start_session(_frames(), session_id="idle")
    hold.set()
    prompt = Thread(
        target=predictor.add_prompt, args=("busy",), kwargs={"frame_idx": 0, "text": "x"}
    )
    closer = Thread(target=predictor.close_session, args=("busy",))
    try:
        prompt.start()
        assert entered.wait(timeout=2)
        hold.clear()
        closer.start()
        closer.join(timeout=0.2)
        assert closer.is_alive()  # waits for the in-flight request on "busy"

        assert predictor.add_prompt("idle", frame_idx=0, text="object")["frame_index"] == 0
        assert predictor.start_session(_frames(), session_id="other") == {"session_id": "other"}
        assert predictor.close_session("idle", run_gc_collect=False) == {"is_success": True}
    finally:
        release.set()
        prompt.join(timeout=2)
        closer.join(timeout=2)
    assert not prompt.is_alive() and not closer.is_alive()
    with pytest.raises(RuntimeError, match="Cannot find session"):
        predictor.add_prompt("busy", frame_idx=0, text="object")


def test_cancel_stops_the_stream_after_the_current_frame() -> None:
    """A cancel issued while streaming ends the stream, including its reverse pass."""
    predictor = _predictor()
    session_id = predictor.start_session(_frames(4))["session_id"]
    predictor.add_prompt(session_id, frame_idx=1, text="object")

    stream = predictor.propagate_in_video(session_id, propagation_direction="both")
    assert next(stream)["frame_index"] == 1
    assert predictor.cancel_propagation(session_id) == {"is_success": True}
    assert list(stream) == []

    # A new stream is not affected by the earlier cancel.
    frames = [item["frame_index"] for item in predictor.propagate_in_video(session_id)]
    assert frames == [1, 2, 3, 0]


def test_active_stream_rejects_prompts_and_second_streams() -> None:
    """State changes wait for the active stream to finish or be closed."""
    predictor = _predictor()
    session_id = predictor.start_session(_frames(3))["session_id"]
    predictor.add_prompt(session_id, frame_idx=0, text="object")
    stream = predictor.propagate_in_video(session_id, propagation_direction="forward")
    next(stream)
    with pytest.raises(RuntimeError, match="propagation is active"):
        predictor.add_prompt(session_id, frame_idx=0, text="object")
    with pytest.raises(RuntimeError, match="propagation is active"):
        next(predictor.propagate_in_video(session_id))
    stream.close()
    predictor.add_prompt(session_id, frame_idx=0, text="object")
