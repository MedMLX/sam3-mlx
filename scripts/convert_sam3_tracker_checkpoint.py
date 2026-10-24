#!/usr/bin/env python3
"""Extract pinned SAM3 tracker weights for interactive image and mask sessions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from sam3_mlx.convert import save_weights  # noqa: E402
from sam3_mlx.mlx_runtime import finite_all  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
)
from sam3_mlx.sam31_checkpoint import (  # noqa: E402
    checkpoint_weights,
    to_mlx_weights,
)
from scripts.run_interactive_checkpoint_parity import TrackerManifest  # noqa: E402


def extract_tracker_weights(payload: object) -> dict[str, mx.array]:
    source = checkpoint_weights(payload)
    tracker: Mapping[str, object] = {
        key: value for key, value in source.items() if key.startswith("tracker.")
    }
    if not tracker:
        raise ValueError("SAM3 checkpoint contains no tracker.* tensors.")
    weights = to_mlx_weights(tracker)
    if set(weights) != set(tracker):
        raise ValueError("SAM3 tracker extraction must retain every tracker.* tensor.")
    converted = {key: value.astype(mx.float32) for key, value in weights.items()}
    if any(not finite_all(value) for value in converted.values()):
        raise ValueError("SAM3 tracker weights are not finite in float32.")
    return converted


def convert_tracker_checkpoint(source: Path, directory: Path) -> TrackerManifest:
    if directory.exists():
        raise FileExistsError(f"Tracker output directory already exists: {directory}.")
    if not source.is_file():
        raise FileNotFoundError(f"SAM3 source checkpoint is missing: {source}.")
    stat = source.stat()
    torch = importlib.import_module("torch")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    weights = extract_tracker_weights(payload)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("SAM3 source checkpoint changed during extraction.")
    save_weights(directory, weights)
    manifest = TrackerManifest.parse(
        {
            "schema_version": 1,
            "source_repo": OFFICIAL_CHECKPOINT_REPO,
            "source_revision": OFFICIAL_CHECKPOINT_REVISION,
            "tensor_count": len(weights),
            "dtype_policy": "float32",
            "key_policy": "all tracker.* keys retained verbatim",
            "layout_policy": "raw Torch layouts; normalization occurs in canonical variant loader",
            "scope": "full tracker, including memory encoder/attention, SAM heads, pointers and temporal state parameters",
            "mapping_api": "sam3_mlx.sam31_checkpoint.checkpoint_weights/to_mlx_weights (generic raw tensor conversion)",
        }
    )
    (directory / "conversion-manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("Paths must be relative to the repository.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=repository_path, required=True)
    parser.add_argument("--output-dir", type=repository_path, required=True)
    args = parser.parse_args()
    manifest = convert_tracker_checkpoint(args.source, args.output_dir)
    print(json.dumps(asdict(manifest), sort_keys=True))


if __name__ == "__main__":
    main()
