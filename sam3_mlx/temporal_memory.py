"""Memory sampling and analysis for isolated temporal stress runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import os
import platform
import resource
import subprocess
import time
from typing import Protocol, Sequence, cast

import mlx.core as mx
import numpy as np


class _MemoryInfo(Protocol):
    rss: int


class _Process(Protocol):
    def memory_info(self) -> _MemoryInfo: ...


class _Psutil(Protocol):
    def Process(self, pid: int) -> _Process: ...  # noqa: N802


@dataclass(frozen=True)
class MemorySample:
    """One memory reading in bytes (seconds for time); ``None`` when a counter is unavailable.

    ``phase`` names the run stage (for example ``"propagation-boundary"`` or
    ``"after-cleanup"``), ``frame_index`` is the propagated frame if any, and
    ``elapsed_seconds`` is measured from the caller's ``started_at``. RSS fields
    are host process memory; ``mlx_*`` fields are MLX allocator counters.
    """

    phase: str
    frame_index: int | None
    elapsed_seconds: float
    process_rss_bytes: int | None
    process_peak_rss_bytes: int | None
    mlx_active_memory_bytes: int | None
    mlx_peak_memory_bytes: int | None
    mlx_cache_memory_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSpec:
    """One planned stress scenario; ``length`` is in frames, ``reason`` explains a skipped run."""

    profile: str
    sequence_id: str
    object_count: int
    length: int
    execute: bool
    reason: str | None


def build_scenario_plan(
    *,
    lengths: Sequence[int],
    execute_up_to: int,
) -> tuple[ScenarioSpec, ...]:
    """Plan single- and three-object scenarios for each sequence length in frames.

    Lengths above ``execute_up_to`` stay in the plan with ``execute=False`` and a
    reason, so reports state what was not run on the host instead of omitting it.

    Raises:
        ValueError: ``execute_up_to`` is below 1 or any length is below 2.
    """
    if execute_up_to < 1:
        raise ValueError("execute_up_to must be at least one frame.")
    unique_lengths = sorted(set(lengths))
    if not unique_lengths or unique_lengths[0] < 2:
        raise ValueError("Long-video lengths must be unique integers of at least 2.")
    profiles = (
        ("single-object", "calibration-stable", 1),
        ("multi-object", "calibration-three-object-addition", 3),
    )
    specs: list[ScenarioSpec] = []
    for profile, sequence_id, object_count in profiles:
        for length in unique_lengths:
            execute = length <= execute_up_to
            specs.append(
                ScenarioSpec(
                    profile=profile,
                    sequence_id=sequence_id,
                    object_count=object_count,
                    length=length,
                    execute=execute,
                    reason=(
                        None
                        if execute
                        else (
                            f"not run on this host: length {length} exceeds the "
                            f"predeclared practical execution limit {execute_up_to}"
                        )
                    ),
                )
            )
    return tuple(specs)


def _process_rss_bytes() -> int | None:
    try:
        module = cast(_Psutil, importlib.import_module("psutil"))
    except ImportError:
        return None
    return int(module.Process(os.getpid()).memory_info().rss)


def _peak_rss_bytes() -> int | None:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value < 0:
        return None
    # ru_maxrss is bytes on macOS but kilobytes on Linux.
    return value if platform.system() == "Darwin" else value * 1024


def _mlx_counter(name: str) -> int | None:
    counter = getattr(mx, name, None)
    if not callable(counter):
        return None
    value = counter()
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"MLX memory counter {name} returned {type(value).__name__}.")
    return value


def capture_memory_sample(
    *,
    phase: str,
    frame_index: int | None,
    started_at: float,
) -> MemorySample:
    """Read host and MLX memory counters now; ``started_at`` is a ``time.perf_counter()`` value."""
    return MemorySample(
        phase=phase,
        frame_index=frame_index,
        elapsed_seconds=time.perf_counter() - started_at,
        process_rss_bytes=_process_rss_bytes(),
        process_peak_rss_bytes=_peak_rss_bytes(),
        mlx_active_memory_bytes=_mlx_counter("get_active_memory"),
        mlx_peak_memory_bytes=_mlx_counter("get_peak_memory"),
        mlx_cache_memory_bytes=_mlx_counter("get_cache_memory"),
    )


def _fit_slope(
    samples: Sequence[MemorySample],
    *,
    counter: str,
    warmup_frames: int,
) -> dict[str, object]:
    # Least-squares line through one ``counter`` value per frame (last
    # ``propagation-boundary`` sample of each frame at or after ``warmup_frames``),
    # in bytes per frame; fewer than three points is reported as unavailable.
    by_frame: dict[int, int] = {}
    for sample in samples:
        if sample.phase != "propagation-boundary" or sample.frame_index is None:
            continue
        value = getattr(sample, counter)
        if isinstance(value, int) and sample.frame_index >= warmup_frames:
            by_frame[sample.frame_index] = value
    if len(by_frame) < 3:
        return {
            "state": "unavailable",
            "reason": "fewer than three post-warm-up counter samples",
            "sample_count": len(by_frame),
        }
    frames = sorted(by_frame)
    x = np.asarray(frames, dtype=np.float64)
    y = np.asarray([by_frame[frame] for frame in frames], dtype=np.float64)
    coefficients: np.ndarray[tuple[int], np.dtype[np.float64]] = np.polyfit(x, y, 1)
    slope, intercept = coefficients
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - residual / total
    return {
        "state": "measured",
        "sample_count": len(by_frame),
        "first_frame": int(x[0]),
        "last_frame": int(x[-1]),
        "slope_bytes_per_frame": float(slope),
        "intercept_bytes": float(intercept),
        "r_squared": r_squared,
    }


def propagation_summary(
    operation_wall_seconds: Sequence[float],
    *,
    propagated_frame_count: int,
    prompt_wall_seconds: float,
    run_wall_seconds: float,
) -> dict[str, object]:
    """Summarize propagation timing; all inputs are wall-clock seconds.

    ``operation_wall_seconds`` holds one duration per propagation call. Prompt
    plus propagation time must not exceed ``run_wall_seconds`` (the remainder is
    reported as unaccounted). Latency percentiles are in milliseconds and
    ``end_to_end_propagation_fps`` is propagated frames per propagation second.
    With no completed propagation the result has ``state="unavailable"``.

    Raises:
        ValueError: A time is negative or non-finite, the frame count is negative,
            or the accounted time exceeds the run time.
    """
    values = np.asarray(operation_wall_seconds, dtype=np.float64)
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isfinite(prompt_wall_seconds)
        or prompt_wall_seconds < 0.0
        or not np.isfinite(run_wall_seconds)
        or run_wall_seconds < 0.0
    ):
        raise ValueError("Temporal timing values must be finite and non-negative.")
    if propagated_frame_count < 0:
        raise ValueError("propagated_frame_count must be non-negative.")
    propagation_wall_seconds = float(values.sum())
    accounted_seconds = propagation_wall_seconds + prompt_wall_seconds
    tolerance_seconds = max(1e-9, run_wall_seconds * 1e-9)
    if accounted_seconds > run_wall_seconds + tolerance_seconds:
        raise ValueError(
            "Prompt and propagation measurements exceed temporal run wall time."
        )
    reconciliation = {
        "run_wall_seconds": run_wall_seconds,
        "prompt_wall_seconds": prompt_wall_seconds,
        "propagation_wall_seconds": propagation_wall_seconds,
        "accounted_execution_seconds": accounted_seconds,
        "unaccounted_wall_seconds": max(run_wall_seconds - accounted_seconds, 0.0),
        "reconciled": True,
    }
    if values.size == 0 or propagated_frame_count == 0:
        return {
            "state": "unavailable",
            "reason": "no completed propagation operations",
            "timing_reconciliation": reconciliation,
        }
    operation_ms = values * 1000.0
    return {
        "state": "measured",
        "propagation_operation_count": int(values.size),
        "propagated_frame_count": propagated_frame_count,
        "propagation_wall_seconds": propagation_wall_seconds,
        "end_to_end_propagation_fps": (
            float(propagated_frame_count / propagation_wall_seconds)
            if propagation_wall_seconds > 0.0
            else None
        ),
        "operation_latency_ms": {
            "p50": float(np.percentile(operation_ms, 50)),
            "p95": float(np.percentile(operation_ms, 95)),
            "p99": float(np.percentile(operation_ms, 99)),
            "mean": float(operation_ms.mean()),
        },
        "timing_reconciliation": reconciliation,
    }


def _phase_counter(
    samples: Sequence[MemorySample], phase: str, counter: str
) -> int | None:
    matches = [sample for sample in samples if sample.phase == phase]
    if not matches:
        return None
    value = getattr(matches[-1], counter)
    return value if isinstance(value, int) else None


def _delta(
    samples: Sequence[MemorySample],
    start_phase: str,
    end_phase: str,
    counter: str,
) -> int | None:
    start = _phase_counter(samples, start_phase, counter)
    end = _phase_counter(samples, end_phase, counter)
    return None if start is None or end is None else end - start


def summarize_memory(
    samples: Sequence[MemorySample], *, warmup_frames: int
) -> dict[str, object]:
    """Summarize samples: post-warm-up growth slopes (bytes per frame), peaks and phase deltas.

    Slopes are descriptive; growth is not classified as a leak automatically (see
    the ``attribution`` entries in the result). Counters that were unavailable
    yield ``None`` entries.
    """
    return {
        "sample_count": len(samples),
        "post_warmup_start_frame": warmup_frames,
        "rss_slope": _fit_slope(
            samples,
            counter="process_rss_bytes",
            warmup_frames=warmup_frames,
        ),
        "mlx_active_slope": _fit_slope(
            samples,
            counter="mlx_active_memory_bytes",
            warmup_frames=warmup_frames,
        ),
        "peak_process_rss_bytes": max(
            (
                sample.process_peak_rss_bytes
                for sample in samples
                if sample.process_peak_rss_bytes is not None
            ),
            default=None,
        ),
        "peak_mlx_memory_bytes": max(
            (
                sample.mlx_peak_memory_bytes
                for sample in samples
                if sample.mlx_peak_memory_bytes is not None
            ),
            default=None,
        ),
        "final_after_cleanup": (
            next(
                (
                    sample.as_dict()
                    for sample in reversed(samples)
                    if sample.phase == "after-cleanup"
                ),
                None,
            )
        ),
        "attribution": {
            "repeated_input_preparation_rss_delta_bytes": _delta(
                samples,
                "after-warmup",
                "before-run",
                "process_rss_bytes",
            ),
            "model_state_and_preprocessed_input_rss_delta_bytes": _delta(
                samples,
                "before-run",
                "state-initialized",
                "process_rss_bytes",
            ),
            "compilation_and_warmup_rss_delta_bytes": _delta(
                samples,
                "after-model-load",
                "after-warmup",
                "process_rss_bytes",
            ),
            "decoder_buffering": (
                "source is a fully materialized repeated-input PIL sequence; its "
                "preparation delta is reported separately"
            ),
            "expected_temporal_memory": (
                "model state and preprocessed input are intentionally retained during "
                "propagation; observed slopes are reported without declaring them leaks"
            ),
            "compilation_cache_retention": (
                "warm-up occurs once in the fresh child and MLX cache memory is sampled"
            ),
            "suspected_unintended_growth": (
                "not classified automatically; inspect post-warm-up slopes and R-squared"
            ),
        },
        "unsupported_counters": {
            "wired_memory": "unavailable: no process-attributable non-privileged API",
            "compressed_memory": "unavailable: no process-attributable non-privileged API",
        },
    }


def safe_hardware_info() -> dict[str, object]:
    """Return host platform, chip and memory/CPU counts; fields are ``None`` if ``sysctl`` fails."""

    def sysctl(name: str) -> str | None:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value or None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": (
            int(value) if (value := sysctl("hw.memsize")) is not None else None
        ),
        "physical_cpu_count": (
            int(value) if (value := sysctl("hw.physicalcpu")) is not None else None
        ),
        "logical_cpu_count": (
            int(value) if (value := sysctl("hw.logicalcpu")) is not None else None
        ),
    }
