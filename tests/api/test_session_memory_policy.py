"""``close_session`` memory cleanup: garbage collection and the MLX buffer cache."""

from __future__ import annotations

import mlx.core as mx
import pytest
from medmlx_core.runtime import require_mlx_runtime
from PIL import Image

import sam3_mlx.model.sam3_video_predictor as predictor_module
from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.model.io_utils import VideoFrames
from sam3_mlx.model.sam3_video_inference import Sam3VideoInference
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from tests.api._stubs import FRAME_SIZE, FakeImageProcessor


def _predictor() -> Sam3VideoPredictor:
    """Create two synthetic sessions so closing one leaves a live peer."""
    predictor = Sam3VideoPredictor(
        Sam3VideoInference(image_model=object(), processor_factory=FakeImageProcessor)
    )
    for session_id in ("closed", "retained"):
        predictor.start_session([Image.new("RGB", FRAME_SIZE)], session_id=session_id)
    return predictor


def _record_cleanup(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(predictor_module.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("clear"))
    return events


@pytest.mark.parametrize(("run_gc_collect", "expected"), [(True, ["collect", "clear"]), (False, [])])
def test_close_runs_cleanup_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, run_gc_collect: bool, expected: list[str]
) -> None:
    """``run_gc_collect`` collects garbage and clears the MLX cache after closing."""
    predictor = _predictor()
    events = _record_cleanup(monkeypatch)
    assert predictor.close_session("closed", run_gc_collect=run_gc_collect) == {
        "is_success": True
    }
    assert events == expected
    predictor.add_prompt("retained", frame_idx=0, text="object")


def test_missing_session_close_does_no_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing an unknown session succeeds without touching memory."""
    predictor = _predictor()
    events = _record_cleanup(monkeypatch)
    assert predictor.close_session("absent") == {"is_success": True}
    assert events == []


def test_failed_close_keeps_the_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing frame close is reported as is; cleanup does not run or replace it."""
    predictor = _predictor()
    events = _record_cleanup(monkeypatch)

    def fail(self: VideoFrames) -> None:
        raise OSError("decoder close failed")

    monkeypatch.setattr(VideoFrames, "close", fail)
    with pytest.raises(RuntimeError, match="Failed to close") as error:
        predictor.close_session("closed")
    assert isinstance(error.value.__cause__, OSError)
    assert events == []


def test_close_clears_the_real_metal_buffer_cache() -> None:
    """Check the real Metal allocator after a close with cleanup."""
    require_mlx_runtime()
    with mx.stream(mx.gpu):
        evaluated = mx.sum(mx.arange(8, dtype=mx.float32))
        evaluate_boundary(evaluated)
    assert bool(evaluated == 28.0)
    predictor = _predictor()
    predictor.close_session("closed")
    assert mx.get_cache_memory() == 0
    predictor.close_session("retained", run_gc_collect=False)
