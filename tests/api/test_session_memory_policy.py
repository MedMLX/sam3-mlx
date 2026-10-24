from __future__ import annotations

from typing import cast

import mlx.core as mx
import pytest

import sam3_mlx.model.sam3_base_predictor as predictor_module
from sam3_mlx.mlx_runtime import evaluate_boundary, require_mlx_runtime, scalar_float
from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class _SessionModel:
    def init_state(self, **_kwargs: object) -> dict[str, object]:
        return {"num_frames": 1}

    def reset_state(self, state: dict[str, object]) -> None:
        state["reset"] = True


def _predictor() -> Sam3BasePredictor:
    predictor = Sam3BasePredictor()
    predictor.model = _SessionModel()
    predictor.start_session("resource", session_id="closed")
    predictor.start_session("resource", session_id="retained")
    return predictor


@pytest.mark.parametrize(
    ("cached_bytes", "threshold", "cleared"),
    [(90, 80, False), (100, 80, True), (110, 80, True), (100, 90, False)],
)
def test_cache_policy_uses_allocator_pressure_and_reports_post_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    cached_bytes: int,
    threshold: int,
    cleared: bool,
) -> None:
    predictor = _predictor()
    events: list[str] = []
    cached = [cached_bytes]
    monkeypatch.setattr(
        predictor_module.gc, "collect", lambda: events.append("collect")
    )

    def device_info(_device: object) -> dict[str, int]:
        return {"max_recommended_working_set_size": 1000}

    monkeypatch.setattr(mx, "device_info", device_info)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 700)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: cached[0])

    def clear() -> None:
        events.append("clear")
        cached[0] = 0

    monkeypatch.setattr(mx, "clear_cache", clear)
    response = predictor.handle_request(
        {
            "type": "close_session",
            "session_id": "closed",
            "clear_cache_threshold": threshold,
        }
    )

    assert events == (["collect", "clear"] if cleared else ["collect"])
    assert response["is_success"] is True
    assert "gpu_mem" in response
    assert response["gpu_mem"] == {
        "backend": "mlx-metal",
        "memory_scope": "process-allocator-vs-recommended-working-set",
        "total_bytes": 1000,
        "free_bytes": 300 - cached[0],
        "allocated_bytes": 700,
        "reserved_bytes": 700 + cached[0],
        "cache_bytes": cached[0],
        "free_pct": (300 - cached[0]) / 1000 * 100,
        "active_session_count": 1,
    }
    predictor.close_session("retained", run_gc_collect=False)


@pytest.mark.parametrize("existing", [True, False])
def test_missing_or_no_gc_close_does_not_read_or_clear_memory(
    monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    predictor = _predictor()

    def forbidden() -> None:
        raise AssertionError("No GC or allocator call is allowed.")

    monkeypatch.setattr(predictor_module.gc, "collect", forbidden)
    monkeypatch.setattr(mx, "get_active_memory", forbidden)
    monkeypatch.setattr(mx, "clear_cache", forbidden)
    assert predictor.close_session(
        "closed" if existing else "absent", run_gc_collect=not existing
    ) == {"is_success": True}
    predictor.close_session("closed", run_gc_collect=False)
    predictor.close_session("retained", run_gc_collect=False)


@pytest.mark.parametrize("threshold", [-1, 101, True, 1.5, "80", None])
def test_invalid_threshold_leaves_session_open(threshold: object) -> None:
    predictor = _predictor()
    with pytest.raises(ValueError, match="integer from 0 to 100"):
        predictor.close_session("closed", clear_cache_threshold=cast(int, threshold))
    assert predictor.reset_session("closed") == {"is_success": True}
    assert predictor.close_session("closed", run_gc_collect=False) == {
        "is_success": True
    }
    predictor.close_session("retained", run_gc_collect=False)


def test_close_cache_policy_executes_actual_metal_counters_and_cache_clear() -> None:
    require_mlx_runtime("session cache policy test")
    with mx.stream(mx.gpu):
        evaluated = mx.sum(mx.arange(8, dtype=mx.float32))
        evaluate_boundary(evaluated)
    assert scalar_float(evaluated) == 28.0
    predictor = _predictor()
    response = predictor.close_session("closed", clear_cache_threshold=0)
    assert "gpu_mem" in response
    snapshot = response["gpu_mem"]
    assert snapshot["backend"] == "mlx-metal"
    assert snapshot["total_bytes"] > 0
    assert snapshot["allocated_bytes"] >= 0
    assert snapshot["cache_bytes"] == mx.get_cache_memory() == 0
    assert snapshot["reserved_bytes"] == snapshot["allocated_bytes"]
    assert snapshot["active_session_count"] == 1
    predictor.close_session("retained", run_gc_collect=False)
