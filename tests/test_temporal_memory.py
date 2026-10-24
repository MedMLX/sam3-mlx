from __future__ import annotations
from tests._json_contracts import require_mapping

import pytest

from sam3_mlx.temporal_memory import (
    MemorySample,
    build_scenario_plan,
    propagation_summary,
    summarize_memory,
)


def _sample(
    phase: str,
    *,
    frame_index: int | None = None,
    rss: int | None = None,
    active: int | None = None,
) -> MemorySample:
    return MemorySample(
        phase=phase,
        frame_index=frame_index,
        elapsed_seconds=float(frame_index or 0),
        process_rss_bytes=rss,
        process_peak_rss_bytes=rss,
        mlx_active_memory_bytes=active,
        mlx_peak_memory_bytes=active,
        mlx_cache_memory_bytes=0 if active is not None else None,
    )


def test_scenario_plan_isolates_single_and_multi_object_lengths() -> None:
    plan = build_scenario_plan(
        lengths=[300, 1000, 3000, 10000],
        execute_up_to=300,
    )

    assert len(plan) == 8
    assert {(spec.profile, spec.object_count) for spec in plan} == {
        ("single-object", 1),
        ("multi-object", 3),
    }
    assert [spec.length for spec in plan if spec.execute] == [300, 300]
    assert all(spec.reason is not None for spec in plan if spec.length > 300)


def test_scenario_plan_rejects_invalid_lengths() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_scenario_plan(lengths=[1, 300], execute_up_to=300)


def test_memory_summary_fits_post_warmup_slopes_and_preserves_unavailable_data() -> (
    None
):
    samples = [
        _sample("after-model-load", rss=1000, active=500),
        _sample("after-warmup", rss=2000, active=900),
        _sample("before-run", rss=2500, active=900),
        _sample("state-initialized", rss=4000, active=1500),
    ]
    samples.extend(
        _sample(
            "propagation-boundary",
            frame_index=index,
            rss=4000 + 128 * index,
            active=1500 + 64 * index,
        )
        for index in range(100)
    )
    samples.append(_sample("after-cleanup", rss=1500, active=0))

    summary = summarize_memory(samples, warmup_frames=20)
    rss_slope = require_mapping(summary["rss_slope"], "RSS slope")
    active_slope = require_mapping(summary["mlx_active_slope"], "MLX active slope")
    attribution = require_mapping(summary["attribution"], "memory attribution")
    unsupported = require_mapping(
        summary["unsupported_counters"], "unsupported counters"
    )
    wired_memory = unsupported["wired_memory"]

    assert rss_slope["slope_bytes_per_frame"] == pytest.approx(128.0)
    assert active_slope["slope_bytes_per_frame"] == pytest.approx(64.0)
    assert attribution["repeated_input_preparation_rss_delta_bytes"] == 500
    assert attribution["model_state_and_preprocessed_input_rss_delta_bytes"] == 1500
    assert isinstance(wired_memory, str)
    assert wired_memory.startswith("unavailable")


def test_propagation_summary_reports_wall_fps_and_reconciles_run_time() -> None:
    result = propagation_summary(
        [0.01, 0.02],
        propagated_frame_count=4,
        prompt_wall_seconds=0.005,
        run_wall_seconds=0.04,
    )

    assert result["state"] == "measured"
    assert result["propagation_operation_count"] == 2
    assert result["propagated_frame_count"] == 4
    assert result["propagation_wall_seconds"] == pytest.approx(0.03)
    assert result["end_to_end_propagation_fps"] == pytest.approx(4 / 0.03)
    latency = require_mapping(result["operation_latency_ms"], "operation latency")
    assert latency["p50"] == pytest.approx(15.0)
    reconciliation = require_mapping(
        result["timing_reconciliation"], "timing reconciliation"
    )
    assert reconciliation["accounted_execution_seconds"] == pytest.approx(0.035)
    assert reconciliation["unaccounted_wall_seconds"] == pytest.approx(0.005)


def test_propagation_summary_rejects_measurements_exceeding_run_wall() -> None:
    with pytest.raises(
        ValueError,
        match="exceed temporal run wall time",
    ):
        propagation_summary(
            [0.03],
            propagated_frame_count=3,
            prompt_wall_seconds=0.02,
            run_wall_seconds=0.04,
        )
