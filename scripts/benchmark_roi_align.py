#!/usr/bin/env python3
"""Benchmark synchronized adaptive ROI Align latency and active MLX memory."""

from __future__ import annotations

from sam3_mlx.mlx_runtime import mlx_array

import argparse
import json
from pathlib import Path
import sys
from typing import Protocol, cast

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.benchmarking import (  # noqa: E402
    BenchmarkOperation,
    TimingProtocol,
    summarize_samples,
    synchronized_samples,
)
from sam3_mlx.mlx_runtime import synchronize_completed, to_numpy  # noqa: E402
from sam3_mlx.model.roi_align import (  # noqa: E402
    ADAPTIVE_ROI_CHUNK_SIZE,
    ADAPTIVE_SAMPLE_CHUNK_SIZE,
    roi_align,
)
from sam3_mlx.release_contract import JsonObject  # noqa: E402
from sam3_mlx.source_binding import git_commit  # noqa: E402
from scripts._benchmark_env import environment  # noqa: E402

ROI_ALIGN_BENCHMARK_SCHEMA = "sam3_mlx.roi_align_benchmark.v2"
DEFAULT_RESOLUTIONS = (504, 672, 1008)
DEFAULT_BOX_COUNTS = (1, 8, 32)
DEFAULT_PRECISIONS = ("fp32", "fp16", "bf16", "mixed")
DEFAULT_GEOMETRIES = ("feature-map", "large-adaptive")
_PRECISION_DTYPES = {
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
    "mixed": mx.float16,
}


class _Args(Protocol):
    resolution: list[int]
    boxes: list[int]
    precision: list[str]
    geometry: list[str]
    channels: int
    pooled_size: int
    warmup_runs: int
    repetitions: int
    out: Path


def _synthetic_rois(
    count: int,
    feature_size: int,
    geometry: str = "feature-map",
) -> mx.array:
    indices = np.arange(count, dtype=np.float32)
    inset_x = (indices % 7) * 0.35
    inset_y = (indices % 5) * 0.25
    if geometry == "feature-map":
        start_x = inset_x - np.float32(0.25)
        start_y = inset_y - np.float32(0.125)
        end_x = np.float32(feature_size) - inset_x
        end_y = np.float32(feature_size) - inset_y
    elif geometry == "large-adaptive":
        start_x = np.float32(-1.5 * feature_size) + inset_x
        start_y = np.float32(-1.5 * feature_size) + inset_y
        end_x = np.float32(2.5 * feature_size) - inset_x
        end_y = np.float32(2.5 * feature_size) - inset_y
    else:
        raise ValueError(f"Unsupported ROI Align benchmark geometry: {geometry}.")
    rois = np.stack(
        [
            np.zeros_like(indices),
            start_x,
            start_y,
            end_x,
            end_y,
        ],
        axis=1,
    )
    return mlx_array(rois, dtype=mx.float32)


def _sampling_geometry(
    *,
    rois: mx.array,
    channels: int,
    feature_size: int,
    pooled_size: int,
) -> JsonObject:
    roi_values = to_numpy(rois, dtype=np.float32)
    grid_heights = np.ceil(
        np.maximum(roi_values[:, 4] - roi_values[:, 2], 1.0) / pooled_size
    ).astype(np.int64)
    grid_widths = np.ceil(
        np.maximum(roi_values[:, 3] - roi_values[:, 1], 1.0) / pooled_size
    ).astype(np.int64)
    max_grid_height = int(grid_heights.max(initial=1))
    max_grid_width = int(grid_widths.max(initial=1))
    roi_count = len(roi_values)
    legacy_dense_elements = (
        roi_count * channels * pooled_size * pooled_size * feature_size * feature_size
    )
    bounded_chunk_elements = (
        min(roi_count, ADAPTIVE_ROI_CHUNK_SIZE)
        * channels
        * pooled_size
        * pooled_size
        * min(max_grid_height, ADAPTIVE_SAMPLE_CHUNK_SIZE)
        * min(max_grid_width, ADAPTIVE_SAMPLE_CHUNK_SIZE)
    )
    return {
        "adaptive_grid_height_max": max_grid_height,
        "adaptive_grid_width_max": max_grid_width,
        "bounded_chunk_intermediate_elements": bounded_chunk_elements,
        "legacy_full_feature_intermediate_elements": legacy_dense_elements,
        "legacy_to_bounded_element_ratio": (
            legacy_dense_elements / bounded_chunk_elements
            if bounded_chunk_elements
            else 0.0
        ),
        "roi_chunk_size": ADAPTIVE_ROI_CHUNK_SIZE,
        "sample_chunk_size": ADAPTIVE_SAMPLE_CHUNK_SIZE,
    }


def _benchmark_case(
    *,
    resolution: int,
    box_count: int,
    precision: str,
    geometry: str,
    channels: int,
    pooled_size: int,
    protocol: TimingProtocol,
) -> JsonObject:
    feature_size = resolution // 14
    dtype = _PRECISION_DTYPES[precision]
    values = mx.reshape(
        mx.arange(channels * feature_size * feature_size, dtype=mx.float32),
        (1, channels, feature_size, feature_size),
    )
    values = (values / max(values.size, 1)).astype(dtype)
    rois = _synthetic_rois(box_count, feature_size, geometry)
    sampling_geometry = _sampling_geometry(
        rois=rois,
        channels=channels,
        feature_size=feature_size,
        pooled_size=pooled_size,
    )
    synchronize_completed(values, rois)
    mx.clear_cache()
    mx.reset_peak_memory()

    operation = BenchmarkOperation(
        run=lambda: roi_align(values, rois, pooled_size, pooled_size),
        synchronize=synchronize_completed,
    )
    samples = synchronized_samples(operation, protocol=protocol)
    output = operation.run()
    synchronize_completed(output)
    output_fp32 = to_numpy(output.astype(mx.float32))
    if not np.isfinite(output_fp32).all():
        raise ValueError(
            f"ROI Align produced non-finite values for precision={precision}."
        )

    return {
        **summarize_samples(samples),
        "resolution": resolution,
        "feature_shape": [1, channels, feature_size, feature_size],
        "box_count": box_count,
        "pooled_size": pooled_size,
        "precision": precision,
        "geometry": geometry,
        "input_dtype": str(dtype).rsplit(".", 1)[-1],
        "output_shape": list(output.shape),
        "finite": True,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
        "sampling_geometry": sampling_geometry,
    }


def generate_benchmark(args: _Args) -> JsonObject:
    protocol = TimingProtocol(
        warmup_runs=args.warmup_runs,
        repetitions=args.repetitions,
    )
    cases: list[JsonObject] = []
    for resolution in args.resolution:
        for box_count in args.boxes:
            for precision in args.precision:
                for geometry in args.geometry:
                    cases.append(
                        _benchmark_case(
                            resolution=resolution,
                            box_count=box_count,
                            precision=precision,
                            geometry=geometry,
                            channels=args.channels,
                            pooled_size=args.pooled_size,
                            protocol=protocol,
                        )
                    )
    return {
        "schema_version": ROI_ALIGN_BENCHMARK_SCHEMA,
        "provenance": {"git_commit": git_commit(REPO_ROOT)},
        "environment": environment(),
        "runtime": {
            "backend": "mlx-metal",
            "synchronization_boundary": "synchronize_completed output followed by mx.synchronize",
            "adaptive_sampling": True,
            "memory_measurement": "mx.reset_peak_memory/get_peak_memory",
            "legacy_geometry_is_estimated_not_measured": True,
        },
        "workload": {
            "resolutions": args.resolution,
            "box_counts": args.boxes,
            "precisions": args.precision,
            "geometries": args.geometry,
            "channels": args.channels,
            "pooled_size": args.pooled_size,
            "warmup_runs": protocol.warmup_runs,
            "repetitions": protocol.repetitions,
        },
        "cases": cases,
    }


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, action="append", default=[])
    parser.add_argument("--boxes", type=int, action="append", default=[])
    parser.add_argument(
        "--precision",
        choices=DEFAULT_PRECISIONS,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--geometry",
        choices=DEFAULT_GEOMETRIES,
        action="append",
        default=[],
    )
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--pooled-size", type=int, default=7)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/roi-align-current.json"),
    )
    args = parser.parse_args()
    args.resolution = args.resolution or list(DEFAULT_RESOLUTIONS)
    args.boxes = args.boxes or list(DEFAULT_BOX_COUNTS)
    args.precision = args.precision or list(DEFAULT_PRECISIONS)
    args.geometry = args.geometry or list(DEFAULT_GEOMETRIES)
    if any(value <= 0 or value % 14 for value in args.resolution):
        parser.error("--resolution values must be positive multiples of 14")
    if any(value <= 0 for value in args.boxes):
        parser.error("--boxes values must be positive")
    if args.channels <= 0 or args.pooled_size <= 0:
        parser.error("--channels and --pooled-size must be positive")
    return cast(_Args, args)


def main() -> None:
    args = _parse_args()
    artifact = generate_benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "benchmark": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
