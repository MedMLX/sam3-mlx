#!/usr/bin/env python3
"""Qualify a pinned SAM3.1 conversion through strict MLX loading and forwards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
import importlib
import json
from pathlib import Path
import sys
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from sam3_mlx.sam31_checkpoint import (  # noqa: E402
    Sam31CheckpointBundle,
    checkpoint_weights,
    parse_sam31_bundle,
    require_stable_sam31_bundle,
    to_mlx_weights,
)

from sam3_mlx.checkpoint_loading import (  # noqa: E402
    Sam3CheckpointLoadReport,
    is_generated_checkpoint_key,
    load_multiplex_checkpoint,
)
from sam3_mlx.checkpoint_mapping import flatten_model_parameters  # noqa: E402
from sam3_mlx.mlx_runtime import (  # noqa: E402
    evaluate_boundary,
    finite_all,
    require_mlx_runtime,
    scalar_float,
    to_numpy,
)
from sam3_mlx.model.memory import SimpleMaskEncoder  # noqa: E402
from sam3_mlx.model.sam3_multiplex_detector import Sam3MultiplexDetector  # noqa: E402
from sam3_mlx.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper  # noqa: E402
from sam3_mlx.model.sam3_multiplex_tracking import (  # noqa: E402
    Sam3MultiplexTrackingWithInteractivity,
)
from sam3_mlx.model.video_tracking_multiplex import VideoTrackingMultiplex  # noqa: E402
from sam3_mlx.model_builder import (  # noqa: E402
    build_checkpoint_free_multiplex_predictor_model,
    build_sam3_predictor,
    default_bpe_path,
)
from sam3_mlx.source_binding import git_changed_paths, git_commit  # noqa: E402
from sam3_mlx.temporal_contract import (  # noqa: E402
    SAM31_CHECKPOINT_REVISION,
    SAM31_CODE_REVISION,
)


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class QualificationGate:
    name: str
    status: GateStatus
    evidence: dict[str, object]


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("Paths must be relative to the repository.")
    return path


def source_signatures() -> dict[str, list[int]]:
    paths = sorted(Path("sam3_mlx").rglob("*.py"))
    paths.extend(
        (
            Path("scripts/convert_sam31_checkpoint.py"),
            Path("scripts/qualify_sam31_checkpoint.py"),
        )
    )
    return {
        str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths
    }


def bundle_evidence(bundle: Sam31CheckpointBundle) -> dict[str, object]:
    loaded = mx.load(bundle.weights, return_metadata=True)
    if not isinstance(loaded, tuple):
        raise TypeError("SAM3.1 safetensors metadata was not returned.")
    arrays, metadata = loaded
    if metadata.get("format") != "mlx" or tuple(sorted(arrays)) != bundle.indexed_keys:
        raise ValueError("SAM3.1 safetensors format/count do not match its manifest.")
    if sum(value.nbytes for value in arrays.values()) != bundle.total_bytes:
        raise ValueError("SAM3.1 actual tensor bytes do not match the manifest.")
    return {
        "source": str(bundle.source),
        "checkpoint_revision": SAM31_CHECKPOINT_REVISION,
        "official_code_revision": SAM31_CODE_REVISION,
        "weights": str(bundle.weights),
        "source_tensor_count": bundle.source_tensor_count,
        "output_tensor_count": bundle.output_tensor_count,
        "output_total_bytes": bundle.total_bytes,
        "ignored_generated_source_keys": list(bundle.ignored_generated_source_keys),
    }


def qualify_raw_transport(bundle: Sam31CheckpointBundle) -> QualificationGate:
    torch = importlib.import_module("torch")
    source = checkpoint_weights(
        torch.load(str(bundle.source), map_location="cpu", weights_only=True, mmap=True)
    )
    retained = {key for key in source if not is_generated_checkpoint_key(key)}
    ignored = tuple(sorted(set(source) - retained))
    if len(source) != bundle.source_tensor_count or ignored != tuple(
        sorted(bundle.ignored_generated_source_keys)
    ):
        raise ValueError("SAM3.1 conversion accounting differs from the actual source.")
    raw = mx.load(bundle.weights)
    if not isinstance(raw, dict) or set(raw) != retained:
        raise ValueError(
            "SAM3.1 converted keys differ from the actual retained source."
        )
    dtype_counts: dict[str, int] = {}
    for key in sorted(retained):
        expected = to_numpy(to_mlx_weights({key: source[key]})[key])
        actual = to_numpy(raw[key])
        if expected.dtype != actual.dtype or not np.array_equal(expected, actual):
            raise ValueError(f"SAM3.1 conversion changed raw values or dtype: {key}.")
        dtype = str(actual.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return QualificationGate(
        "exact-raw-source-transport",
        GateStatus.PASSED,
        {
            "compared_tensor_count": len(retained),
            "dtype_counts": dtype_counts,
            "comparison": "exact NumPy array equality, including shape and dtype",
            "ignored_generated_source_count": len(ignored),
        },
    )


def component_forwards(
    detector: Sam3MultiplexDetector, tracker: VideoTrackingMultiplex
) -> dict[str, object]:
    text = detector.backbone.forward_text(["person"])
    evaluate_boundary(text)
    if not all(finite_all(value) for value in text.values()):
        raise ValueError(
            "SAM3.1 loaded detector text forward produced non-finite output."
        )
    memory = tracker.maskmem_backbone
    if not isinstance(memory, SimpleMaskEncoder):
        raise TypeError("Expected the SAM3.1 SimpleMaskEncoder architecture.")
    features = mx.zeros((1, 256, 72, 72), dtype=mx.float32)
    masks = mx.zeros(
        (1, memory.mask_downsampler.multiplex_count, 1008, 1008), dtype=mx.float32
    )
    encoded = memory(features, masks, skip_mask_sigmoid=True)
    evaluate_boundary(encoded)
    memory_values = [encoded["vision_features"], *encoded["vision_pos_enc"]]
    if not all(finite_all(value) for value in memory_values):
        raise ValueError(
            "SAM3.1 loaded tracker memory forward produced non-finite output."
        )
    return {
        "text_prompt": "person",
        "text_outputs": {key: list(value.shape) for key, value in text.items()},
        "memory_inputs": {"features": list(features.shape), "masks": list(masks.shape)},
        "memory_outputs": [list(value.shape) for value in memory_values],
        "all_outputs_finite": True,
        "scope": "detector text encoder and tracker memory encoder; temporal parity is not measured",
    }


def _runtime_model(
    bundle: Sam31CheckpointBundle, *, automatic_setup: bool
) -> tuple[Sam3MultiplexTrackingWithInteractivity, Sam3CheckpointLoadReport]:
    if automatic_setup:
        predictor = build_sam3_predictor(
            version="sam3.1", local_weights_dir=bundle.weights.parent
        )
        model = predictor.model
        if not isinstance(model, Sam3MultiplexTrackingWithInteractivity):
            raise TypeError("Automatic SAM3.1 setup returned an unexpected model.")
        audit = getattr(model, "checkpoint_load_report", None)
        if not isinstance(audit, Sam3CheckpointLoadReport):
            raise TypeError("Automatic SAM3.1 setup did not retain its strict audit.")
        return model, audit
    model = build_checkpoint_free_multiplex_predictor_model(
        bpe_path=default_bpe_path(),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        compile_model=False,
    )
    return model, load_multiplex_checkpoint(model, bundle.weights, strict=True)


def qualify_runtime(
    bundle: Sam31CheckpointBundle,
    gates: list[QualificationGate],
    *,
    automatic_setup: bool = False,
) -> None:
    model, audit = _runtime_model(bundle, automatic_setup=automatic_setup)
    required_missing = tuple(
        key for key in audit.missing if not is_generated_checkpoint_key(key)
    )
    if required_missing or audit.extra or audit.shape_mismatched:
        raise ValueError(
            "SAM3.1 strict checkpoint audit did not cover the whole model."
        )
    parameters = flatten_model_parameters(model)
    evaluate_boundary(parameters)
    finite = [mx.all(mx.isfinite(parameters[key])) for key in audit.loaded]
    evaluate_boundary(finite)
    if any(not bool(value.item()) for value in finite):
        raise ValueError("SAM3.1 loaded model contains non-finite parameters.")
    dtypes: dict[str, int] = {}
    for key in audit.loaded:
        dtype = str(parameters[key].dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    gates.append(
        QualificationGate(
            "strict-full-model-load",
            GateStatus.PASSED,
            {
                "loaded_count": len(audit.loaded),
                "construction": "public-automatic-sam31-builder"
                if automatic_setup
                else "explicit-checkpoint-free-graph-and-strict-loader",
                "missing_required": list(required_missing),
                "generated_missing": list(audit.missing),
                "extra": list(audit.extra),
                "shape_mismatched": list(audit.shape_mismatched),
                "all_loaded_weights_finite": True,
                "loaded_dtype_counts": dtypes,
                "model_config": {
                    "max_num_objects": model.max_num_objects,
                    "compile_model": model.compile_model,
                },
            },
        )
    )
    if not isinstance(model.detector, Sam3MultiplexDetector) or not isinstance(
        model.tracker, Sam3MultiplexPredictorWrapper
    ):
        raise TypeError("Unexpected SAM3.1 predictor architecture.")
    tracker = model.tracker.model
    if not isinstance(tracker, VideoTrackingMultiplex):
        raise TypeError("Unexpected SAM3.1 tracker architecture.")
    evidence = component_forwards(model.detector, tracker)
    gates.append(
        QualificationGate("evaluated-component-forwards", GateStatus.PASSED, evidence)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=relative_path, required=True)
    parser.add_argument("--bundle", type=relative_path, required=True)
    parser.add_argument("--output", type=relative_path, required=True)
    parser.add_argument(
        "--automatic-setup",
        action="store_true",
        help="Exercise the public SAM3.1 builder's verified automatic cache path.",
    )
    args = parser.parse_args()
    bound_sources = source_signatures()
    gates: list[QualificationGate] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "command": [sys.executable, *sys.argv],
        "source_commit": git_commit(REPO_ROOT),
        "changed_paths": list(git_changed_paths(REPO_ROOT)),
        "source_file_signatures": bound_sources,
        "qualification_scope": "checkpoint loading and component execution",
        "automatic_setup": bool(args.automatic_setup),
        "temporal_parity": "not-measured",
        "default_version_change_authorized_by_report": False,
    }
    stage = "bundle-provenance"
    try:
        bundle = parse_sam31_bundle(cast(Path, args.source), cast(Path, args.bundle))
        stage = "metal-backend"
        mx.set_default_device(mx.gpu)
        runtime = require_mlx_runtime("SAM3.1 checkpoint qualification")
        probe = mx.sum(mx.arange(8, dtype=mx.float32))
        evaluate_boundary(probe)
        if scalar_float(probe) != 28.0:
            raise RuntimeError("MLX Metal evaluated-operation probe failed.")
        gates.append(
            QualificationGate(
                stage,
                GateStatus.PASSED,
                {
                    **asdict(runtime),
                    "device_info": mx.device_info(),
                    "evaluated_probe": 28.0,
                },
            )
        )
        stage = "bundle-provenance"
        gates.append(
            QualificationGate(stage, GateStatus.PASSED, bundle_evidence(bundle))
        )
        stage = "exact-raw-source-transport"
        gates.append(qualify_raw_transport(bundle))
        stage = "strict-load-and-component-forwards"
        qualify_runtime(bundle, gates, automatic_setup=bool(args.automatic_setup))
        stage = "source-stability"
        if source_signatures() != bound_sources:
            raise RuntimeError("SAM3.1 qualification sources changed during execution.")
        require_stable_sam31_bundle(bundle)
        gates.append(QualificationGate(stage, GateStatus.PASSED, {}))
        report["status"] = GateStatus.PASSED
    except Exception as exc:
        report["status"] = GateStatus.FAILED
        gates.append(
            QualificationGate(
                stage,
                GateStatus.FAILED,
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
        )
        raise
    finally:
        report["gates"] = [asdict(gate) for gate in gates]
        output = cast(Path, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
