#!/usr/bin/env python3
"""Qualify real-image interactive inference with pinned SAM3 checkpoints.

Torch CPU and MLX Metal run in separate, sequential processes. Raw outputs and
failed comparisons are retained. Frozen image-release mask and score thresholds
apply; raw-logit errors are diagnostic and do not establish logit equivalence.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import NotRequired, Protocol, TypedDict, Unpack, cast
from types import CodeType
import traceback
import warnings

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    MLX_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_REVISION,
    RELEASE_IMAGES,
    RELEASE_THRESHOLDS,
    require_json_object,
)
from scripts._oracle_runtime import (  # noqa: E402
    TorchRuntime,
    OracleTransformOwner,
    configure_official_image_resize,
    official_tokenizer_resources,
    install_cpu_oracle_adapters,
    restore_construction_adapters,
    validate_official_checkout,
    save_oracle_arrays,
)
from scripts import run_interactive_head_parity as head_parity  # noqa: E402

type Array = NDArray[np.float32]
type Prediction = tuple[Array, Array, Array]
EVIDENCE_SCHEMA_VERSION = 2


class PromptArguments(TypedDict):
    point_coords: NotRequired[Array]
    point_labels: NotRequired[NDArray[np.int32]]
    box: NotRequired[Array]
    mask_input: NotRequired[Array]
    multimask_output: bool
    return_logits: bool


class _PointArguments(TypedDict):
    point_coords: Array
    point_labels: NDArray[np.int32]


class _CleanupTransforms(Protocol):
    mask_threshold: float
    max_hole_area: float
    max_sprinkle_area: float


class _InteractivePredictor(Protocol):
    _transforms: _CleanupTransforms
    _features: object | None
    _is_image_set: bool
    _is_batch: bool


class InteractiveModel(Protocol):
    inst_interactive_predictor: _InteractivePredictor

    def predict_inst(
        self, state: object, **kwargs: Unpack[PromptArguments]
    ) -> Prediction: ...

    def predict_inst_batch(
        self, state: object, **kwargs: object
    ) -> tuple[list[Array], list[Array], list[Array]]: ...


class InteractiveProcessor(Protocol):
    def set_image(self, image: Image.Image) -> object: ...

    def set_image_batch(self, images: list[Image.Image]) -> object: ...


class OfficialInteractiveProcessor(InteractiveProcessor, OracleTransformOwner, Protocol):
    pass


class Phase(StrEnum):
    RUN = "run"
    TORCH = "torch"
    MLX = "mlx"
    COMPARE = "compare"


class PromptKind(StrEnum):
    POINT = "point"
    BOX = "box"
    MIXED = "mixed"
    MULTIPLE = "multiple"
    REFINEMENT = "refinement"
    MASK_ONLY = "mask_only"


@dataclass(frozen=True)
class RunConfig:
    official_checkout: Path
    official_revision: str
    official_checkpoint: Path
    mlx_checkpoint: Path
    tracker_checkpoint: Path
    out: Path


@dataclass(frozen=True)
class TrackerManifest:
    schema_version: int
    source_repo: str
    source_revision: str
    tensor_count: int
    dtype_policy: str
    key_policy: str
    layout_policy: str
    scope: str
    mapping_api: str

    @classmethod
    def parse(cls, value: object) -> TrackerManifest:
        if not isinstance(value, dict):
            raise TypeError("Tracker manifest must be a JSON object.")
        data = cast(dict[str, object], value)
        fields = cls.__dataclass_fields__
        if set(data) != set(fields):
            raise ValueError("Tracker manifest fields differ from schema version 1.")
        integers = {"schema_version", "tensor_count"}
        for name, item in data.items():
            expected_type = int if name in integers else str
            if type(item) is not expected_type:
                raise TypeError(
                    f"Tracker manifest {name} must be {expected_type.__name__}."
                )
        manifest = cast(Callable[..., TrackerManifest], cls)(**data)
        bindings = {
            "schema_version": 1,
            "source_repo": "facebook/sam3",
            "source_revision": OFFICIAL_CHECKPOINT_REVISION,
            "dtype_policy": "float32",
            "key_policy": "all tracker.* keys retained verbatim",
            "layout_policy": "raw Torch layouts; normalization occurs in canonical variant loader",
        }
        if any(
            getattr(manifest, name) != expected for name, expected in bindings.items()
        ):
            raise ValueError("Tracker manifest schema/source/layout binding mismatch.")
        if manifest.tensor_count < 1:
            raise ValueError("Tracker manifest tensor count is invalid.")
        return manifest


@dataclass(frozen=True)
class CaseSpec:
    name: str
    prompt: PromptKind
    multimask: bool
    refinement_source: str | None = None


@dataclass(frozen=True)
class CleanupPolicy:
    mask_threshold: float
    max_hole_area: float
    max_sprinkle_area: float


def cleanup_policy(model: InteractiveModel) -> CleanupPolicy:
    transforms = model.inst_interactive_predictor._transforms  # pyright: ignore[reportPrivateUsage]
    return CleanupPolicy(
        float(transforms.mask_threshold),
        float(transforms.max_hole_area),
        float(transforms.max_sprinkle_area),
    )


@dataclass(frozen=True)
class Comparison:
    name: str
    verdict: head_parity.Verdict
    mask_iou_min: float | None
    mask_iou_mean: float | None
    iou_score_absolute_error_max: float | None
    full_resolution_logit_absolute_error_max: float | None
    low_resolution_logit_absolute_error_max: float | None
    official_shapes: tuple[tuple[int, ...], ...]
    mlx_shapes: tuple[tuple[int, ...], ...]
    reason: str | None = None


def cases() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec("point_multimask", PromptKind.POINT, True),
        CaseSpec("box_single", PromptKind.BOX, False),
        CaseSpec("mixed_single", PromptKind.MIXED, False),
        CaseSpec("multiple_objects", PromptKind.MULTIPLE, True),
        CaseSpec("mask_refinement", PromptKind.REFINEMENT, False, "point_multimask"),
        CaseSpec("mask_only", PromptKind.MASK_ONLY, False, "point_multimask"),
    )


def _tracker_manifest(args: RunConfig) -> TrackerManifest:
    return TrackerManifest.parse(
        json.loads(
            args.tracker_checkpoint.with_name("conversion-manifest.json").read_text()
        )
    )


def _validate_resources(args: RunConfig) -> None:
    validate_official_checkout(
        args.official_checkout, expected_revision=args.official_revision
    )
    for path in (args.official_checkpoint, args.mlx_checkpoint, args.tracker_checkpoint):
        if not path.is_file():
            raise ValueError(f"Pinned checkpoint is missing: {path}")
    _tracker_manifest(args)


def _images(checkout: Path) -> tuple[list[Image.Image], list[dict[str, object]]]:
    images: list[Image.Image] = []
    records: list[dict[str, object]] = []
    for name in ("example", "holdout"):
        spec = RELEASE_IMAGES[name]
        relative = spec["path"].removeprefix("official-checkout/")
        path = checkout / relative
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
        if list(images[-1].size) != spec["size"]:
            raise ValueError(f"Pinned image size mismatch: {path}")
        records.append(
            {
                "name": name,
                "path": relative,
                "original_size": list(images[-1].size),
            }
        )
    return images, records


def prompt_arguments(
    spec: CaseSpec, size: tuple[int, int], previous: Mapping[str, Prediction]
) -> PromptArguments:
    width, height = size
    point = np.array([[0.45, 0.5]], dtype=np.float32) * [width, height]
    box = np.array([0.15, 0.15, 0.8, 0.85], dtype=np.float32) * [
        width,
        height,
        width,
        height,
    ]
    arguments: PromptArguments = {
        "multimask_output": spec.multimask,
        "return_logits": True,
    }
    if spec.prompt in {PromptKind.POINT, PromptKind.MIXED, PromptKind.REFINEMENT}:
        arguments.update(
            point_coords=point.astype(np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )
    if spec.prompt in {PromptKind.BOX, PromptKind.MIXED}:
        arguments["box"] = box.astype(np.float32)
    if spec.prompt == PromptKind.MULTIPLE:
        arguments.update(
            point_coords=(
                np.array([[[0.3, 0.4]], [[0.7, 0.6]]], dtype=np.float32)
                * [width, height]
            ).astype(np.float32),
            point_labels=np.array([[1], [1]], dtype=np.int32),
            box=(
                np.array(
                    [[0.1, 0.1, 0.55, 0.75], [0.5, 0.2, 0.95, 0.95]], dtype=np.float32
                )
                * [width, height, width, height]
            ).astype(np.float32),
        )
    if spec.refinement_source is not None:
        _, scores, low_res = previous[spec.refinement_source]
        best = int(np.argmax(scores))
        arguments["mask_input"] = low_res[best : best + 1]
    return arguments


def compare_prediction(
    name: str, actual: Prediction, expected: Prediction
) -> Comparison:
    shapes = (
        tuple(value.shape for value in expected),
        tuple(value.shape for value in actual),
    )
    if shapes[0] != shapes[1]:
        return Comparison(
            name,
            head_parity.Verdict.FAILED,
            None,
            None,
            None,
            None,
            None,
            *shapes,
            "Prediction shapes differ.",
        )
    if any(not np.isfinite(value).all() for value in (*actual, *expected)):
        return Comparison(
            name,
            head_parity.Verdict.FAILED,
            None,
            None,
            None,
            None,
            None,
            *shapes,
            "Prediction contains non-finite values.",
        )
    left = (actual[0] > 0).reshape(-1, *actual[0].shape[-2:])
    right = (expected[0] > 0).reshape(-1, *expected[0].shape[-2:])
    intersections = np.asarray(
        np.count_nonzero(left & right, axis=(1, 2)), dtype=np.int64
    )
    unions = np.asarray(np.count_nonzero(left | right, axis=(1, 2)), dtype=np.int64)
    ious = np.divide(intersections, unions, out=np.ones(unions.shape), where=unions > 0)
    minimum, mean = float(np.min(ious)), float(np.mean(ious))
    score_error = float(np.max(np.abs(actual[1] - expected[1])))
    passed = (
        minimum >= RELEASE_THRESHOLDS["mask_iou_min"]
        and mean >= RELEASE_THRESHOLDS["mask_iou_mean_min"]
        and score_error <= RELEASE_THRESHOLDS["score_abs_max"]
    )
    return Comparison(
        name,
        head_parity.Verdict.PASSED if passed else head_parity.Verdict.FAILED,
        minimum,
        mean,
        score_error,
        float(np.max(np.abs(actual[0] - expected[0]))),
        float(np.max(np.abs(actual[2] - expected[2]))),
        *shapes,
        None if passed else "Frozen image mask/score thresholds exceeded.",
    )


def _record_prediction(arrays: dict[str, Array], name: str, output: Prediction) -> None:
    for suffix, value in zip(("logits", "scores", "low_res"), output, strict=True):
        arrays[f"{name}.{suffix}"] = np.asarray(value, dtype=np.float32)


def _verify_request_cleanup(model: InteractiveModel) -> None:
    predictor = model.inst_interactive_predictor
    if (
        predictor._features is not None  # pyright: ignore[reportPrivateUsage]
        or predictor._is_image_set  # pyright: ignore[reportPrivateUsage]
        or predictor._is_batch  # pyright: ignore[reportPrivateUsage]
    ):
        raise RuntimeError(
            "Interactive request left image features/batch state attached."
        )


def _capture_prediction(
    model: InteractiveModel, state: object, arguments: PromptArguments
) -> Prediction:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = model.predict_inst(state, **arguments)
    skipped = [
        str(item.message)
        for item in caught
        if "Skipping the post-processing" in str(item.message)
    ]
    if skipped:
        raise RuntimeError("Official cleanup was skipped: " + "; ".join(skipped))
    _verify_request_cleanup(model)
    return output


def _exercise(
    model: InteractiveModel,
    processor: InteractiveProcessor,
    images: list[Image.Image],
    arrays: dict[str, Array],
) -> None:
    for image_index, image in enumerate(images):
        print(json.dumps({"stage": "encode_single", "image": image_index}), flush=True)
        state = processor.set_image(image)
        previous: dict[str, Prediction] = {}
        for spec in cases():
            arguments = prompt_arguments(spec, image.size, previous)
            previous[spec.name] = _capture_prediction(model, state, arguments)
            _record_prediction(
                arrays, f"image_{image_index}.{spec.name}", previous[spec.name]
            )
            print(
                json.dumps(
                    {"stage": "predicted", "image": image_index, "case": spec.name}
                ),
                flush=True,
            )
    print(json.dumps({"stage": "encode_batch", "batch_size": len(images)}), flush=True)
    state = processor.set_image_batch(images)
    point_arguments = [
        cast(_PointArguments, prompt_arguments(cases()[0], image.size, {}))
        for image in images
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outputs = model.predict_inst_batch(
            state,
            point_coords_batch=[item["point_coords"] for item in point_arguments],
            point_labels_batch=[item["point_labels"] for item in point_arguments],
            multimask_output=True,
            return_logits=True,
        )
    if any("Skipping the post-processing" in str(item.message) for item in caught):
        raise RuntimeError("Official batch cleanup was skipped.")
    _verify_request_cleanup(model)
    for index, values in enumerate(zip(*outputs, strict=True)):
        _record_prediction(arrays, f"batch_image_{index}", values)


def compile_cpu_fp32_fused_source(source: str, filename: str) -> CodeType:
    """Change exactly the three forced BF16 casts in the pinned fused operation."""
    tree = ast.parse(source, filename=filename)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "addmm_act":
        raise ValueError("Expected only the official addmm_act function.")
    casts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "bfloat16"
    ]
    expected = {
        ast.dump(ast.parse(f"{name} = {name}.to(torch.bfloat16)").body[0])
        for name in ("self", "mat1", "mat2")
    }
    actual = {
        ast.dump(node)
        for node in functions[0].body
        if isinstance(node, ast.Assign)
        and any(cast_node in ast.walk(node) for cast_node in casts)
    }
    if len(casts) != 3 or actual != expected:
        raise ValueError("Official fused BF16 cast contract changed.")
    for node in casts:
        node.attr = "float32"
    return compile(tree, filename, "exec")


def _install_cpu_fp32_fused_adapter(checkout: Path) -> dict[str, object]:
    fused = import_module("sam3.perflib.fused")
    vitdet = import_module("sam3.model.vitdet")
    path = Path(cast(str, fused.__file__)).resolve()
    if path != checkout.resolve() / "sam3/perflib/fused.py":
        raise RuntimeError("Fused adapter source lies outside pinned checkout.")
    if vitdet.addmm_act is not fused.addmm_act:
        raise RuntimeError("ViT fused operator binding differs from official source.")
    namespace: dict[str, object] = {"__name__": "sam3_cpu_fp32_fused_oracle_adapter"}
    exec(compile_cpu_fp32_fused_source(path.read_text(), str(path)), namespace)
    setattr(fused, "addmm_act", namespace["addmm_act"])
    setattr(vitdet, "addmm_act", namespace["addmm_act"])
    return {
        "file": "sam3/perflib/fused.py",
        "rewrite": "exactly self/mat1/mat2.to(torch.bfloat16) -> .to(torch.float32)",
        "rewritten_cast_count": 3,
        "ast_omissions": [],
        "reason": "CPU FP32 oracle; upstream forced BF16 fused result otherwise mismatches FP32 fc2",
    }


def _load_torch(
    args: RunConfig,
) -> tuple[InteractiveModel, InteractiveProcessor, dict[str, object]]:
    import torch

    torch_runtime = cast(TorchRuntime, torch)
    originals = install_cpu_oracle_adapters(torch_runtime)
    sys.path.insert(0, str(args.official_checkout.resolve()))
    try:
        with official_tokenizer_resources(args.official_checkout):
            builder = import_module("sam3.model_builder")
            processor_module = import_module("sam3.model.sam3_image_processor")
            for module in (builder, processor_module):
                source = Path(cast(str, module.__file__)).resolve()
                if not source.is_relative_to(args.official_checkout.resolve()):
                    raise RuntimeError(
                        f"Official module was imported outside pinned checkout: {source}"
                    )
            processor_factory = getattr(processor_module, "Sam3Processor")
            model = builder.build_sam3_image_model(
                checkpoint_path=str(args.official_checkpoint),
                load_from_HF=False,
                device="cpu",
                enable_inst_interactivity=True,
                compile=False,
            )
            fused_adapter = _install_cpu_fp32_fused_adapter(args.official_checkout)
    finally:
        restore_construction_adapters(torch_runtime, originals)
        torch_runtime.Tensor.pin_memory = originals.pin_memory
    payload = torch.load(
        args.official_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if "model" in payload:
        payload = payload["model"]
    expected_keys = {
        key.removeprefix("detector.") for key in payload if key.startswith("detector.")
    }
    expected_keys.update(
        "inst_interactive_predictor.model." + key.removeprefix("tracker.")
        for key in payload
        if key.startswith("tracker.")
    )
    missing = sorted(set(dict(model.named_parameters())) - expected_keys)
    if missing:
        raise ValueError(
            f"Official checkpoint misses required parameters: {missing[:10]}"
        )
    processor = cast(OfficialInteractiveProcessor, processor_factory(model, device="cpu", resolution=1008))
    configure_official_image_resize(processor, 1008)
    return (
        cast(InteractiveModel, model),
        processor,
        {
            "device": "torch.cpu",
            "precision": "float32; autocast disabled",
            "required_parameter_coverage": "complete",
            "compile": False,
            "construction_adapters": [
                "unused EDT fails fast",
                "CUDA cache allocations redirected to CPU",
                "construction pin_memory no-op; restored before inference",
                "legacy tokenizer resource via importlib.resources; exact pinned Git blob bytes",
                "explicit FP32 resize then uint8 rounding preserves TorchVision0.24 inputs",
            ],
            "ast_omissions": [],
            "cpu_fp32_fused_adapter": fused_adapter,
            "cleanup_implementation": "official host CPU connected components",
        },
    )


def _load_mlx(
    args: RunConfig,
) -> tuple[InteractiveModel, InteractiveProcessor, dict[str, object]]:
    import mlx.core as mx
    from sam3_mlx.mlx_runtime import require_mlx_runtime, evaluate_boundary
    from sam3_mlx.model_builder import build_sam3_image_model
    from sam3_mlx.model.sam3_image_processor import Sam3Processor
    from sam3_mlx.checkpoint_loading import Sam3CheckpointLoadReport

    require_mlx_runtime("interactive checkpoint parity")
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.ones((4,), dtype=mx.float32))
    evaluate_boundary(probe)
    if probe.item() != 4:
        raise RuntimeError("MLX evaluated-operation proof failed.")
    model = build_sam3_image_model(
        checkpoint_path=args.mlx_checkpoint,
        interactive_checkpoint_path=args.tracker_checkpoint,
        enable_inst_interactivity=True,
        load_from_HF=False,
        device="mlx",
        precision="fp32",
        compile=False,
        strict_checkpoint_loading=True,
    )
    report = getattr(model, "checkpoint_load_report")
    if not isinstance(report, Sam3CheckpointLoadReport):
        raise TypeError("Strict image loading did not produce a checkpoint audit.")
    processor = Sam3Processor(
        model, resolution=1008, device="mlx", preprocessing="parity"
    )
    return (
        cast(InteractiveModel, model),
        processor,
        {
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "precision": "fp32",
            "package_lane": "macOS Metal",
            "metal_available": bool(mx.metal.is_available()),
            "evaluated_operation": "sum(ones(4,float32))=4",
            "compile": False,
            "weight_coverage": asdict(report),
            "strict_checkpoint_loading": True,
            "cleanup_implementation": "MLX-compatible host CPU connected components",
            "synchronization": "evaluated host export through shared to_numpy boundary",
            "memory_policy": "sequential independent backend processes; default allocator/cache limits",
        },
    )


def _tracker_scope(args: RunConfig) -> dict[str, int]:
    import mlx.core as mx
    from sam3_mlx.checkpoint_mapping import (
        InteractiveCheckpointScope,
        classify_interactive_checkpoint_key,
    )

    weights = cast(dict[str, mx.array], mx.load(str(args.tracker_checkpoint)))
    manifest = _tracker_manifest(args)
    if len(weights) != manifest.tensor_count:
        raise ValueError("Tracker tensor count differs from conversion manifest.")
    counts = {scope.value: 0 for scope in InteractiveCheckpointScope}
    for key in weights:
        counts[classify_interactive_checkpoint_key(key).value] += 1
    if (
        counts[InteractiveCheckpointScope.UNREVIEWED]
        or counts[InteractiveCheckpointScope.OUTSIDE_TRACKER]
    ):
        raise ValueError(
            "Derived raw tracker artifact contains unreviewed source keys."
        )
    return counts


def _backend_metadata(args: RunConfig, phase: Phase) -> dict[str, object]:
    # Bind the complete package so lazy imports cannot escape the source check.
    files = list((REPO_ROOT / "sam3_mlx").rglob("*.py"))
    files += [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/run_interactive_head_parity.py",
        REPO_ROOT / "scripts/_oracle_runtime.py",
        REPO_ROOT / "scripts/_resize_reference.py",
    ]
    official_files = list((args.official_checkout.resolve() / "sam3").rglob("*.py"))
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "backend": phase.value,
        "status": "running",
        "scope": "full-backbone real-image interactive inference on the pinned two-image fixture; no training/video/performance claim",
        "local_source": asdict(head_parity.source_identity(REPO_ROOT, files)),
        "official_source": asdict(
            head_parity.source_identity(
                args.official_checkout.resolve(), official_files
            )
        ),
        "official_checkpoint": {"revision": OFFICIAL_CHECKPOINT_REVISION},
        "mlx_checkpoint": {"revision": MLX_CHECKPOINT_REVISION},
        "tracker_checkpoint": {
            "manifest": asdict(_tracker_manifest(args)),
            "scope_counts": _tracker_scope(args),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_versions": {
            name: metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "mlx",
                "numpy",
                "pillow",
                "einops",
                "timm",
                "scikit-image",
            )
        },
        "case_specs": [asdict(spec) for spec in cases()],
    }


def verify_source_snapshot(root: Path, identity: Mapping[str, object]) -> None:
    signatures = require_json_object(identity["file_signatures"], field="file_signatures")
    changed = [
        relative
        for relative, signature in signatures.items()
        if not (root / relative).is_file()
        or head_parity.file_signature(root / relative) != signature
    ]
    if changed:
        raise RuntimeError(
            "Source changed during numerical phase: " + ", ".join(changed)
        )


def run_backend(args: RunConfig, phase: Phase) -> int:
    arrays: dict[str, Array] = {}
    record: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "backend": phase.value,
        "status": "failed",
    }
    try:
        _validate_resources(args)
        images, image_records = _images(args.official_checkout)
        record = _backend_metadata(args, phase)
        record["images"] = image_records
        model, processor, runtime = (
            _load_torch(args) if phase == Phase.TORCH else _load_mlx(args)
        )
        record["runtime"] = runtime
        record["cleanup_policy"] = asdict(cleanup_policy(model))
        verify_source_snapshot(
            REPO_ROOT,
            require_json_object(record["local_source"], field="local_source"),
        )
        verify_source_snapshot(
            args.official_checkout.resolve(),
            require_json_object(record["official_source"], field="official_source"),
        )
        _exercise(model, processor, images, arrays)
        verify_source_snapshot(
            REPO_ROOT,
            require_json_object(record["local_source"], field="local_source"),
        )
        verify_source_snapshot(
            args.official_checkout.resolve(),
            require_json_object(record["official_source"], field="official_source"),
        )
        validate_official_checkout(
            args.official_checkout, expected_revision=args.official_revision
        )
        _validate_resources(args)
        record["source_stability"] = "passed"
        record["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        # The worker boundary retains partial numerical evidence on failure.
        record.update(
            status="failed",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
        )
    args.out.mkdir(parents=True, exist_ok=True)
    array_path = args.out / f"{phase.value}.npz"
    save_oracle_arrays(array_path, arrays)
    (args.out / f"{phase.value}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"backend": phase.value, "status": record["status"], "outputs": len(arrays)}
        ),
        flush=True,
    )
    return 0 if record["status"] == "passed" else 1


def evidence_errors(records: dict[str, dict[str, object]], out: Path) -> list[str]:
    errors: list[str] = []
    for backend, record in records.items():
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or record.get("source_stability") != "passed"
        ):
            errors.append(f"{backend}: source stability/schema was not qualified.")
        if not (out / f"{backend}.npz").is_file():
            errors.append(f"{backend}: raw output artifact is missing.")
    left, right = records["torch"], records["mlx"]
    for key in (
        "official_checkpoint",
        "mlx_checkpoint",
        "tracker_checkpoint",
        "images",
        "case_specs",
        "cleanup_policy",
    ):
        if key not in left or left.get(key) != right.get(key):
            errors.append(f"Backend evidence binding differs: {key}.")
    for key in ("local_source", "official_source"):
        for field in ("revision", "file_signatures"):
            if (
                key not in left
                or key not in right
                or require_json_object(left[key], field=key).get(field)
                != require_json_object(right[key], field=key).get(field)
            ):
                errors.append(f"Backend source binding differs: {key}.{field}.")
    return errors


def expected_inventory() -> set[str]:
    names = {
        f"image_{index}.{spec.name}"
        for index in range(len(RELEASE_IMAGES))
        for spec in cases()
    }
    names.update(f"batch_image_{index}" for index in range(len(RELEASE_IMAGES)))
    return {
        f"{name}.{suffix}"
        for name in names
        for suffix in ("logits", "scores", "low_res")
    }


def compare(args: RunConfig) -> int:
    records = {
        phase: json.loads((args.out / f"{phase}.json").read_text())
        for phase in ("torch", "mlx")
    }
    integrity_errors = evidence_errors(records, args.out)
    comparisons: list[Comparison] = []
    with (
        np.load(args.out / "torch.npz", allow_pickle=False) as official,
        np.load(args.out / "mlx.npz", allow_pickle=False) as mlx,
    ):
        for name in sorted(
            key.removesuffix(".logits")
            for key in official.files
            if key.endswith(".logits")
        ):
            keys = tuple(
                f"{name}.{suffix}" for suffix in ("logits", "scores", "low_res")
            )
            if any(key not in mlx.files for key in keys):
                continue
            comparisons.append(
                compare_prediction(
                    name,
                    cast(Prediction, tuple(mlx[key] for key in keys)),
                    cast(Prediction, tuple(official[key] for key in keys)),
                )
            )
        inventories_match = (
            set(official.files) == set(mlx.files) == expected_inventory()
        )
    expected_count = len(RELEASE_IMAGES) * len(cases()) + len(RELEASE_IMAGES)
    passed = (
        not integrity_errors
        and inventories_match
        and len(comparisons) == expected_count
        and all(record["status"] == "passed" for record in records.values())
        and all(result.verdict == head_parity.Verdict.PASSED for result in comparisons)
    )
    report = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "scope": "checkpoint-backed real-image full-backbone interactive outputs; official CPU FP32 fused-cast adapter versus MLX Metal FP32; raw-logit/CUDA equivalence is not qualified",
        "thresholds": {
            key: RELEASE_THRESHOLDS[key]
            for key in ("mask_iou_min", "mask_iou_mean_min", "score_abs_max")
        },
        "backends": records,
        "expected_case_count": expected_count,
        "output_inventories_match": inventories_match,
        "evidence_integrity_errors": integrity_errors,
        "cases": [asdict(result) for result in comparisons],
        "raw_evidence": ["torch.npz", "mlx.npz"],
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps({"status": report["status"], "compared_cases": len(comparisons)}),
        flush=True,
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="Repository-relative output directory"
    )
    parser.add_argument("--phase", choices=list(Phase), default=Phase.RUN)
    args = parser.parse_args()
    if args.out.is_absolute() or ".." in args.out.parts:
        parser.error("--out must be repository-relative without '..'")
    config = RunConfig(
        args.official_checkout,
        args.official_revision,
        args.official_checkpoint,
        args.mlx_checkpoint,
        args.tracker_checkpoint,
        REPO_ROOT / args.out,
    )
    phase = Phase(args.phase)
    if phase in (Phase.TORCH, Phase.MLX):
        return run_backend(config, phase)
    if phase == Phase.RUN:
        base = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        for backend in (Phase.TORCH, Phase.MLX):
            result = subprocess.run(
                [*base, "--phase", backend.value], cwd=REPO_ROOT, check=False
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"{backend} worker terminated unexpectedly: {result.returncode}"
                )
    return compare(config)


if __name__ == "__main__":
    raise SystemExit(main())
