#!/usr/bin/env python3
"""Run the legacy official-vs-MLX SAM 3 image parity and performance harness.

New release attestations use ``run_hardened_image_parity.py``. This legacy
entrypoint shares its object assignment and case comparison implementation with
the hardened evidence path so the two runners cannot disagree numerically.

The thresholds are declared independently of observed results and apply to
every non-empty matched case:

* exact detection count;
* minimum per-object mask IoU >= 0.95 and case mean >= 0.99;
* maximum box coordinate error <= 2 pixels;
* maximum score error <= 0.025.
"""

from __future__ import annotations

from sam3_mlx.mlx_runtime import to_numpy

import argparse
from collections.abc import Callable
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.model.sam3_image_processor import (  # noqa: E402
    ProcessorState,
    Sam3Processor,
)
from sam3_mlx.model.sam3_image import Sam3Image  # noqa: E402
from sam3_mlx.convert import validate_hub_checkpoint_provenance  # noqa: E402
from sam3_mlx.precision import checkpoint_dtype_for_policy, parse_precision  # noqa: E402
from sam3_mlx.parity_evidence import CaseComparison, compare_case  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CODE_REVISION,
    RELEASE_THRESHOLDS,
    build_oracle_bindings,
)
from sam3_mlx.benchmarking import (  # noqa: E402
    BenchmarkOperation,
    TimingProtocol,
    percentile,
    synchronized_samples,
)
from scripts._oracle_runtime import (  # noqa: E402
    OracleCase,
    build_oracle_cache_identity,
    load_official_oracle_archive,
    public_oracle_metadata,
    run_upstream_oracle_subprocess,
    validate_official_checkout,
)


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


class MlxImageCaseRun:
    def __init__(
        self,
        *,
        outputs: list[dict[str, np.ndarray]],
        mask_logits: list[np.ndarray],
        mask_logit_dtypes: list[str],
        states: dict[int, ProcessorState],
        processors: dict[int, Sam3Processor],
    ) -> None:
        self.outputs = outputs
        self.mask_logits = mask_logits
        self.mask_logit_dtypes = mask_logit_dtypes
        self.states = states
        self.processors = processors


MASK_IOU_MIN = RELEASE_THRESHOLDS["mask_iou_min"]
MASK_IOU_MEAN_MIN = RELEASE_THRESHOLDS["mask_iou_mean_min"]
BOX_L_INF_MAX = RELEASE_THRESHOLDS["box_l_inf_max"]
SCORE_ABS_MAX = RELEASE_THRESHOLDS["score_abs_max"]


def case_specs(image: Image.Image, profile: str) -> list[OracleCase]:
    if profile == "holdout":
        holdout_cases: list[OracleCase] = [
            {
                "name": f"text_paper_bag_{resolution}",
                "resolution": resolution,
                "prompt": "paper bag",
                "geometric_prompts": [],
            }
            for resolution in (1008, 672, 504)
        ]
        holdout_cases.extend(
            [
                {
                    "name": "text_car_1008",
                    "resolution": 1008,
                    "prompt": "car",
                    "geometric_prompts": [],
                },
                {
                    "name": "text_nonsense_1008",
                    "resolution": 1008,
                    "prompt": "zzzz_not_a_real_class_qqq",
                    "geometric_prompts": [],
                },
            ]
        )
        return holdout_cases
    if profile != "example":
        raise ValueError(f"Unknown case profile: {profile!r}")

    width, height = image.size

    def normalized_cxcywh(x: float, y: float, w: float, h: float) -> list[float]:
        return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]

    positive = normalized_cxcywh(480, 290, 110, 360)
    negative = normalized_cxcywh(370, 280, 115, 375)
    example_cases: list[OracleCase] = [
        {
            "name": "text_shoe_1008",
            "resolution": 1008,
            "prompt": "shoe",
            "geometric_prompts": [],
        },
        {
            "name": "text_nonsense_1008",
            "resolution": 1008,
            "prompt": "zzzz_not_a_real_class_qqq",
            "geometric_prompts": [],
        },
        {
            "name": "positive_box_1008",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [{"box": positive, "label": True}],
        },
        {
            "name": "positive_negative_box_1008",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [
                {"box": positive, "label": True},
                {"box": negative, "label": False},
            ],
        },
        {
            "name": "text_shoe_672",
            "resolution": 672,
            "prompt": "shoe",
            "geometric_prompts": [],
        },
        {
            "name": "text_shoe_504",
            "resolution": 504,
            "prompt": "shoe",
            "geometric_prompts": [],
        },
    ]
    return example_cases


def evidence_path(path: Path, *, official_checkout: Path) -> str:
    try:
        relative = path.resolve().relative_to(official_checkout.resolve())
    except ValueError:
        return str(path)
    return f"official-checkout/{relative}"


def _run_prompt(
    processor: Sam3Processor, state: ProcessorState, spec: OracleCase
) -> ProcessorState:
    processor.reset_all_prompts(state)
    if spec["prompt"] is not None:
        state = processor.set_text_prompt(spec["prompt"], state)
    for geometric_prompt in spec["geometric_prompts"]:
        state = processor.add_geometric_prompt(
            geometric_prompt["box"],
            geometric_prompt["label"],
            state,
        )
    return state


def _timed_prompt(
    processor: Sam3Processor,
    image: Image.Image,
    spec: OracleCase,
) -> Callable[[], ProcessorState]:
    def run() -> ProcessorState:
        return _run_prompt(processor, processor.set_image(image), spec)

    return run


def _materialize_finite_output(state: ProcessorState) -> dict[str, np.ndarray]:
    output = {
        "masks": to_numpy(state["masks"], dtype=np.bool_),
        "boxes": to_numpy(state["boxes"], dtype=np.float32),
        "scores": to_numpy(state["scores"], dtype=np.float32),
    }
    for name in ("boxes", "scores"):
        if not np.isfinite(output[name]).all():
            raise ValueError(f"MLX parity output {name} contains NaN or Inf values.")
    return output


def build_mlx_image_model(
    *, checkpoint: Path, precision: str
) -> tuple[Sam3Image, float]:
    """Load and synchronize one MLX image model for parity workloads."""

    import mlx.core as mx
    import sam3_mlx

    mlx_eval = cast(_MlxEval, getattr(mx, "eval"))
    load_started = time.perf_counter()
    model = sam3_mlx.build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        precision=precision,
    )
    mlx_eval(cast(object, model.parameters()))
    return model, time.perf_counter() - load_started


def run_mlx_image_cases(
    *,
    model: Sam3Image,
    image: Image.Image,
    specs: list[OracleCase],
    confidence_threshold: float,
    preprocessing: str,
) -> MlxImageCaseRun:
    """Run one image and its complete prompt matrix on an already-loaded model."""

    import mlx.core as mx

    mlx_eval = cast(_MlxEval, getattr(mx, "eval"))
    outputs: list[dict[str, np.ndarray]] = []
    mask_logits: list[np.ndarray] = []
    mask_logit_dtypes: list[str] = []
    states: dict[int, ProcessorState] = {}
    processors: dict[int, Sam3Processor] = {}
    for spec in specs:
        resolution = int(spec["resolution"])
        if resolution not in states:
            processor = Sam3Processor(
                model,
                resolution=resolution,
                confidence_threshold=confidence_threshold,
                preprocessing=preprocessing,
            )
            states[resolution] = processor.set_image(image)
            processors[resolution] = processor
        processor = processors[resolution]
        state = _run_prompt(processor, states[resolution], spec)
        logits = state.get("masks_logits")
        if not isinstance(logits, mx.array):
            raise TypeError("MLX parity state masks_logits must be an MLX array.")
        mlx_eval(state["masks"], state["boxes"], state["scores"], logits)
        outputs.append(_materialize_finite_output(state))
        materialized_logits = to_numpy(logits.astype(mx.float32))
        if not np.isfinite(materialized_logits).all():
            raise ValueError("MLX parity mask logits contain NaN or Inf values.")
        mask_logits.append(materialized_logits)
        mask_logit_dtypes.append(str(logits.dtype).rsplit(".", 1)[-1])
    return MlxImageCaseRun(
        outputs=outputs,
        mask_logits=mask_logits,
        mask_logit_dtypes=mask_logit_dtypes,
        states=states,
        processors=processors,
    )


def mlx_outputs(
    *,
    checkpoint: Path,
    image: Image.Image,
    specs: list[OracleCase],
    confidence_threshold: float,
    repetitions: int,
    precision: str = "fp32",
    preprocessing: str = "parity",
) -> tuple[list[dict[str, np.ndarray]], dict[str, object]]:
    import mlx.core as mx

    mlx_eval = cast(_MlxEval, getattr(mx, "eval"))
    model, cold_load_s = build_mlx_image_model(
        checkpoint=checkpoint, precision=precision
    )
    mx.clear_cache()
    mx.reset_peak_memory()
    image_run = run_mlx_image_cases(
        model=model,
        image=image,
        specs=specs,
        confidence_threshold=confidence_threshold,
        preprocessing=preprocessing,
    )
    latencies: dict[int, list[float]] = {}

    protocol = TimingProtocol(warmup_runs=1, repetitions=repetitions)
    for resolution in (1008, 672, 504):
        spec = next(
            item
            for item in specs
            if item["resolution"] == resolution and item["prompt"] is not None
        )
        processor = image_run.processors[resolution]

        def synchronize(state: ProcessorState) -> None:
            mlx_eval(state["masks"], state["boxes"], state["scores"])
            mx.synchronize()

        latencies[resolution] = synchronized_samples(
            BenchmarkOperation(
                run=_timed_prompt(processor, image, spec),
                synchronize=synchronize,
            ),
            protocol=protocol,
        )

    return image_run.outputs, {
        "status": "passed",
        "cold_load_s": cold_load_s,
        "warmup_runs": 1,
        "repetitions": repetitions,
        "latency_by_resolution_s": {
            str(resolution): {
                "samples": samples,
                "median": statistics.median(samples),
                "p95": percentile(samples, 0.95),
            }
            for resolution, samples in latencies.items()
        },
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
        "measurement_boundary": (
            "MLX active allocator peak after model load; synchronized full "
            "set_image plus text-grounding latency"
        ),
        "precision": precision,
        "preprocessing": preprocessing,
    }


def compare_release_case(
    spec: OracleCase,
    official: dict[str, np.ndarray],
    mlx: dict[str, np.ndarray],
) -> CaseComparison:
    return compare_case(
        spec,
        official,
        mlx,
        thresholds={
            "mask_iou_min": MASK_IOU_MIN,
            "mask_iou_mean_min": MASK_IOU_MEAN_MIN,
            "box_l_inf_max": BOX_L_INF_MAX,
            "score_abs_max": SCORE_ABS_MAX,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument(
        "--profile",
        choices=("example", "holdout"),
        default="example",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "mixed"),
        default="fp32",
        help="Image-runtime precision policy (default: fp32).",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("parity", "fast", "native"),
        default="parity",
        help="Image preprocessing implementation (default: parity).",
    )
    parser.add_argument("--mlx-checkpoint-repo", default=MLX_CHECKPOINT_REPO)
    parser.add_argument(
        "--mlx-checkpoint-revision",
        default=MLX_CHECKPOINT_REVISION,
    )
    args = parser.parse_args()

    if not np.isfinite(args.confidence_threshold) or not (
        0.0 <= args.confidence_threshold <= 1.0
    ):
        raise SystemExit("--confidence-threshold must be finite and within [0, 1].")

    try:
        revision = validate_official_checkout(
            args.official_checkout,
            expected_revision=OFFICIAL_CODE_REVISION,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.repetitions < 5:
        raise SystemExit("--repetitions must be at least 5 for release evidence.")

    if not args.official_checkpoint.is_file():
        raise SystemExit(f"Official checkpoint is missing: {args.official_checkpoint}.")
    policy = parse_precision(args.precision)
    try:
        validate_hub_checkpoint_provenance(
            args.mlx_checkpoint.parent,
            expected_repo=args.mlx_checkpoint_repo,
            expected_revision=args.mlx_checkpoint_revision,
            expected_dtype_policy=(
                checkpoint_dtype_for_policy(policy) if policy != "fp32" else None
            ),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    image = Image.open(args.image).convert("RGB")
    specs = case_specs(image, args.profile)
    with tempfile.TemporaryDirectory(prefix="sam3-mlx-parity-") as temp_dir:
        temp = Path(temp_dir)
        cases_path = temp / "cases.json"
        oracle_path = args.oracle_cache or temp / "official.npz"
        cases_path.write_text(json.dumps(specs, indent=2) + "\n")
        oracle_runner = REPO_ROOT / "scripts" / "run_upstream_image_oracle_hardened.py"
        expected_bindings = build_oracle_bindings(
            image=args.image,
            case_spec=cases_path,
            confidence_threshold=args.confidence_threshold,
        )
        if not oracle_path.exists():
            run_upstream_oracle_subprocess(
                runner_path=oracle_runner,
                official_python=args.official_python,
                official_checkout=args.official_checkout,
                checkpoint_path=args.official_checkpoint,
                output_path=oracle_path,
                confidence_threshold=args.confidence_threshold,
                extra_arguments=(
                    "--image",
                    str(args.image),
                    "--cases",
                    str(cases_path),
                ),
            )
        try:
            oracle_metadata, official_outputs = load_official_oracle_archive(
                oracle_path,
                expected_bindings=expected_bindings,
                expected_cache_identity=build_oracle_cache_identity(
                    image=args.image,
                    checkpoint=args.official_checkpoint,
                    official_python=args.official_python,
                ),
                expected_specs=specs,
                image_size=image.size,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    runtime_outputs, performance = mlx_outputs(
        checkpoint=args.mlx_checkpoint,
        image=image,
        specs=specs,
        confidence_threshold=args.confidence_threshold,
        repetitions=args.repetitions,
        precision=args.precision,
        preprocessing=args.preprocessing,
    )
    cases = [
        compare_release_case(spec, official, mlx)
        for spec, official, mlx in zip(
            specs, official_outputs, runtime_outputs, strict=True
        )
    ]
    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    report = {
        "schema_version": 1,
        "status": status,
        "official_code": {
            "repo": "https://github.com/facebookresearch/sam3",
            "revision": revision,
        },
        "official_checkpoint": {
            "repo": "facebook/sam3",
            "revision": OFFICIAL_CHECKPOINT_REVISION,
        },
        "converted_checkpoint": {
            "repo": args.mlx_checkpoint_repo,
            "revision": args.mlx_checkpoint_revision,
        },
        "runtime": {
            "precision": args.precision,
            "preprocessing": args.preprocessing,
        },
        "image": {
            "path": evidence_path(
                args.image,
                official_checkout=args.official_checkout,
            ),
            "size": list(image.size),
        },
        "confidence_threshold": args.confidence_threshold,
        "precision": args.precision,
        "case_profile": args.profile,
        "thresholds": {
            "mask_iou_min": MASK_IOU_MIN,
            "mask_iou_mean_min": MASK_IOU_MEAN_MIN,
            "box_l_inf_max": BOX_L_INF_MAX,
            "score_abs_max": SCORE_ABS_MAX,
        },
        "cases": cases,
        "oracle": public_oracle_metadata(oracle_metadata),
        "performance": performance,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wrote": str(args.out), "status": status}, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
