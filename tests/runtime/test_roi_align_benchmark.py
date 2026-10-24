from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import tomllib
from typing import cast

import numpy as np
from numpy.typing import NDArray

from sam3_mlx.mlx_runtime import to_numpy
from tests._paths import REPO_ROOT


_SPEC = importlib.util.spec_from_file_location(
    "benchmark_roi_align",
    REPO_ROOT / "scripts" / "benchmark_roi_align.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


def test_roi_align_benchmark_defaults_cover_release_matrix() -> None:
    assert _BENCHMARK.DEFAULT_RESOLUTIONS == (504, 672, 1008)
    assert _BENCHMARK.DEFAULT_BOX_COUNTS == (1, 8, 32)
    assert _BENCHMARK.DEFAULT_PRECISIONS == ("fp32", "fp16", "bf16", "mixed")
    assert _BENCHMARK.DEFAULT_GEOMETRIES == ("feature-map", "large-adaptive")


def test_roi_align_benchmark_boxes_are_deterministic_bounded_rois() -> None:
    rois = cast(NDArray[np.float32], to_numpy(_BENCHMARK._synthetic_rois(32, 72)))

    assert rois.shape == (32, 5)
    np.testing.assert_array_equal(rois[:, 0], np.zeros(32, dtype=np.float32))
    assert np.all(rois[:, 1:3] >= -0.25)
    assert np.all(rois[:, 3:] <= 72.0)
    assert np.all(rois[:, 3] > rois[:, 1])
    assert np.all(rois[:, 4] > rois[:, 2])


def test_roi_align_benchmark_large_geometry_exercises_bounded_adaptive_grid() -> None:
    rois = _BENCHMARK._synthetic_rois(8, 72, "large-adaptive")
    geometry = _BENCHMARK._sampling_geometry(
        rois=rois,
        channels=32,
        feature_size=72,
        pooled_size=7,
    )

    assert geometry["adaptive_grid_height_max"] > 32
    assert geometry["adaptive_grid_width_max"] > 32
    assert geometry["bounded_chunk_intermediate_elements"] == 401_408
    assert geometry["legacy_full_feature_intermediate_elements"] == 65_028_096
    assert geometry["legacy_to_bounded_element_ratio"] > 160


def test_release_check_runs_pinned_full_roi_align_oracle_matrix() -> None:
    contract = json.loads(
        (REPO_ROOT / "parity/contracts/roi-align-torchvision.json").read_text()
    )
    supported_oracle = cast(dict[str, str], contract["reference"]["supported_oracle"])
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert set(project["dependency-groups"]["oracle"]) == {
        f"{name}=={version}" for name, version in supported_oracle.items()
    }
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    assert {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in supported_oracle
    } == supported_oracle

    plan = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "release-check"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    commands = [shlex.split(line) for line in plan.stdout.splitlines()]
    fuzz_commands = [
        command for command in commands if "scripts/fuzz_roi_align.py" in command
    ]
    assert sorted(
        command[command.index("--dtype") + 1] for command in fuzz_commands
    ) == ["bf16", "fp16", "fp32"]
    assert all("--require-full-matrix" in command for command in fuzz_commands)
    scripts = {argument for command in commands for argument in command}
    assert {"scripts/validate_release.py", "scripts/audit_release_evidence.py"} <= scripts
