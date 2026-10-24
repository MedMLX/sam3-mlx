#!/usr/bin/env python3
"""Compare official Torch and MLX interactive heads on shared synthetic inputs.

Run in an ephemeral oracle environment with --no-sync --group oracle --with
einops. This does not qualify the image backbone, released checkpoints, mask
cleanup, temporal tracking, or performance.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib import import_module, metadata
import json
from pathlib import Path
import subprocess
import sys
from types import CodeType, ModuleType
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.checkpoint_mapping import (  # noqa: E402
    normalize_inst_interactive_weights,
)
from sam3_mlx.mlx_runtime import to_numpy  # noqa: E402
from sam3_mlx.model.sam1_task_predictor import (  # noqa: E402
    SAM3InteractiveImageModel,
    SAM3InteractiveImagePredictor,
)
from scripts._oracle_runtime import validate_official_checkout  # noqa: E402

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


type FloatArray = NDArray[np.float32]
type LabelArray = NDArray[np.int32]


class HeadPredictionArgs(TypedDict):
    point_coords_batch: list[FloatArray]
    point_labels_batch: list[LabelArray]
    box_batch: list[FloatArray]
    mask_input_batch: list[FloatArray]
    multimask_output: bool
    return_logits: bool


class OfficialHeadModel(Protocol):
    image_size: int
    device: object
    no_mem_embed: Tensor
    sam_prompt_encoder: Module
    sam_mask_decoder: Module

    def eval(self) -> object: ...

    def state_dict(self) -> Mapping[str, Tensor]: ...


class OfficialHeadPredictor(Protocol):
    _features: dict[str, Tensor | list[Tensor]] | None
    _orig_hw: list[tuple[int, int]]
    _is_image_set: bool
    _is_batch: bool

    def predict_batch(
        self, **kwargs: Unpack[HeadPredictionArgs]
    ) -> tuple[list[FloatArray], list[FloatArray], list[FloatArray]]: ...


class _TorchNN(Protocol):
    Module: Callable[[], OfficialHeadModel]
    Parameter: Callable[[Tensor], Tensor]


class HeadTorchRuntime(Protocol):
    nn: _TorchNN

    def device(self, name: str) -> object: ...

    def zeros(self, *dims: int) -> Tensor: ...

    def manual_seed(self, seed: int) -> object: ...

    def from_numpy(self, value: NDArray[np.generic]) -> Tensor: ...


type HeadFactories = tuple[
    Callable[..., OfficialHeadPredictor],
    Callable[..., Module],
    Callable[..., Module],
    Callable[..., Module],
]


class Verdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceIdentity:
    revision: str
    git_status: str
    file_signatures: dict[str, list[int]]


@dataclass(frozen=True)
class HeadInputs:
    original_sizes: tuple[tuple[int, int], ...]
    points: list[FloatArray]
    labels: list[LabelArray]
    boxes: list[FloatArray]
    masks: list[FloatArray]


@dataclass(frozen=True)
class ArrayComparison:
    output: str
    image_index: int
    official_shape: tuple[int, ...]
    mlx_shape: tuple[int, ...]
    max_absolute_error: float | None
    verdict: Verdict
    reason: str | None = None


@dataclass(frozen=True)
class HeadCaseResult:
    multimask_output: bool
    return_logits: bool
    comparisons: list[ArrayComparison]
    verdict: Verdict


@dataclass(frozen=True)
class HeadParityReport:
    schema_version: int
    scope: str
    verdict: Verdict
    seed: int
    tolerances: dict[str, float]
    runtime_versions: dict[str, str]
    devices: dict[str, str]
    official_source: SourceIdentity
    local_source: SourceIdentity
    source_adapter: dict[str, str]
    original_sizes: tuple[tuple[int, int], ...]
    object_counts: tuple[int, ...]
    shared_weight_tensors: int
    cases: list[HeadCaseResult]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    ).stdout


def file_signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def source_identity(root: Path, files: list[Path]) -> SourceIdentity:
    return SourceIdentity(
        revision=_git(root, "rev-parse", "HEAD").strip(),
        git_status=_git(root, "status", "--porcelain", "--untracked-files=normal"),
        file_signatures={
            str(path.relative_to(root)): file_signature(path)
            for path in sorted(set(files))
        },
    )


def _annotation_names(tree: ast.Module) -> set[int]:
    annotations: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
    return {id(node) for annotation in annotations for node in ast.walk(annotation)}


def compile_predictor_source(source: str, filename: str) -> CodeType:
    """Omit exactly the annotation-only tracker import; preserve predictor code."""
    tree = ast.parse(source, filename=filename)
    tracker_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "sam3.model.sam3_tracker_base"
    ]
    if len(tracker_imports) != 1 or [
        (item.name, item.asname) for item in tracker_imports[0].names
    ] != [("Sam3TrackerBase", None)]:
        raise ValueError(
            "Expected exactly the official Sam3TrackerBase annotation import."
        )
    annotations = _annotation_names(tree)
    if any(
        isinstance(node, ast.Name)
        and node.id == "Sam3TrackerBase"
        and id(node) not in annotations
        for node in ast.walk(tree)
    ):
        raise ValueError("Sam3TrackerBase is used at runtime; cannot omit its import.")
    tree.body.remove(tracker_imports[0])
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    return compile(ast.fix_missing_locations(tree), filename, "exec")


def _load_official(checkout: Path) -> HeadFactories:
    if any(name == "sam3" or name.startswith("sam3.") for name in sys.modules):
        raise RuntimeError(
            "Run the oracle in a fresh process; sam3 is already imported."
        )
    package = ModuleType("sam3")
    package.__path__ = [str(checkout / "sam3")]
    sys.modules["sam3"] = package
    prompt = getattr(import_module("sam3.sam.prompt_encoder"), "PromptEncoder")
    decoder = getattr(import_module("sam3.sam.mask_decoder"), "MaskDecoder")
    transformer = getattr(import_module("sam3.sam.transformer"), "TwoWayTransformer")
    source = checkout / "sam3/model/sam1_task_predictor.py"
    namespace: dict[str, object] = {"__name__": "official_interactive_head_oracle"}
    exec(compile_predictor_source(source.read_text(), str(source)), namespace)
    return cast(
        HeadFactories,
        (namespace["SAM3InteractiveImagePredictor"], prompt, decoder, transformer),
    )


def _official_model(
    torch: HeadTorchRuntime, classes: HeadFactories
) -> OfficialHeadModel:
    _, prompt, decoder, transformer = classes
    model = torch.nn.Module()
    model.image_size = 16
    model.device = torch.device("cpu")
    model.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, 256))
    model.sam_prompt_encoder = prompt(256, (4, 4), (16, 16), 16)
    model.sam_mask_decoder = decoder(
        transformer_dim=256,
        transformer=transformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
    )
    model.eval()
    return model


def _inputs() -> HeadInputs:
    return HeadInputs(
        original_sizes=((12, 20), (7, 11)),
        points=[
            np.array([[[2, 3], [4, 5]], [[14, 8], [17, 9]]], dtype=np.float32),
            np.array([[5, 2]], dtype=np.float32),
        ],
        labels=[
            np.array([[1, 0], [0, 1]], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ],
        boxes=[
            np.array([[1, 2, 8, 10], [11, 4, 19, 11]], dtype=np.float32),
            np.array([1, 1, 9, 6], dtype=np.float32),
        ],
        masks=[
            np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 1, 16, 16),
            np.linspace(1, -1, 256, dtype=np.float32).reshape(1, 16, 16),
        ],
    )


def _shared_predictors(
    torch: HeadTorchRuntime, classes: HeadFactories, seed: int, inputs: HeadInputs
) -> tuple[SAM3InteractiveImagePredictor, OfficialHeadPredictor, dict[str, mx.array]]:
    torch.manual_seed(seed)
    mx.random.seed(seed)
    model = _official_model(torch, classes)
    local_model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)
    weights = normalize_inst_interactive_weights(
        {
            "tracker." + key: mx.array(value.detach().numpy())
            for key, value in model.state_dict().items()
        }
    )
    local_model.load_weights(
        [
            (key.removeprefix("inst_interactive_predictor.model."), value)
            for key, value in weights.items()
        ]
    )
    local_model.eval()
    local = SAM3InteractiveImagePredictor(local_model, max_hole_area=0.0)
    reference = classes[0](model, max_hole_area=0.0)
    rng = np.random.default_rng(seed)
    local._features = {  # pyright: ignore[reportPrivateUsage]
        "high_res_feats": [
            mx.array(rng.normal(size=(2, 32, 16, 16)).astype(np.float32)),
            mx.array(rng.normal(size=(2, 64, 8, 8)).astype(np.float32)),
        ],
        "image_embed": mx.array(rng.normal(size=(2, 256, 4, 4)).astype(np.float32)),
    }
    features = local._require_features()  # pyright: ignore[reportPrivateUsage]
    reference._features = {  # pyright: ignore[reportPrivateUsage]
        "image_embed": torch.from_numpy(to_numpy(features["image_embed"])),
        "high_res_feats": [
            torch.from_numpy(to_numpy(x)) for x in features["high_res_feats"]
        ],
    }
    local._orig_hw = list(inputs.original_sizes)  # pyright: ignore[reportPrivateUsage]
    local._is_image_set = True  # pyright: ignore[reportPrivateUsage]
    local._is_batch = True  # pyright: ignore[reportPrivateUsage]
    reference._orig_hw = list(inputs.original_sizes)  # pyright: ignore[reportPrivateUsage]
    reference._is_image_set = True  # pyright: ignore[reportPrivateUsage]
    reference._is_batch = True  # pyright: ignore[reportPrivateUsage]
    return local, reference, weights


def compare_array(
    actual: FloatArray,
    expected: FloatArray,
    *,
    output: str,
    image_index: int,
    rtol: float,
    atol: float,
) -> ArrayComparison:
    reason = None
    if actual.shape != expected.shape:
        reason = "Output shapes differ."
    elif not np.isfinite(actual).all() or not np.isfinite(expected).all():
        reason = "Output contains non-finite values."
    if reason is not None:
        return ArrayComparison(
            output,
            image_index,
            expected.shape,
            actual.shape,
            None,
            Verdict.FAILED,
            reason,
        )
    passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    return ArrayComparison(
        output,
        image_index,
        expected.shape,
        actual.shape,
        float(np.max(np.abs(actual - expected))),
        Verdict.PASSED if passed else Verdict.FAILED,
        None if passed else "Numerical tolerance exceeded.",
    )


def _case(
    local: SAM3InteractiveImagePredictor,
    reference: OfficialHeadPredictor,
    inputs: HeadInputs,
    *,
    multimask: bool,
    rtol: float,
    atol: float,
) -> HeadCaseResult:
    kwargs = HeadPredictionArgs(
        point_coords_batch=inputs.points,
        point_labels_batch=inputs.labels,
        box_batch=inputs.boxes,
        mask_input_batch=inputs.masks,
        multimask_output=multimask,
        return_logits=True,
    )
    expected = reference.predict_batch(**kwargs)
    actual = local.predict_batch(**kwargs)
    comparisons = [
        compare_array(
            mlx_image,
            torch_image,
            output=output,
            image_index=index,
            rtol=rtol,
            atol=atol,
        )
        for output, mlx_batch, torch_batch in zip(
            ("mask_logits", "iou_scores", "low_resolution_logits"),
            actual,
            expected,
            strict=True,
        )
        for index, (mlx_image, torch_image) in enumerate(
            zip(mlx_batch, torch_batch, strict=True)
        )
    ]
    verdict = (
        Verdict.PASSED
        if all(x.verdict == Verdict.PASSED for x in comparisons)
        else Verdict.FAILED
    )
    return HeadCaseResult(multimask, True, comparisons, verdict)


def loaded_source_files(root: Path, prefixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for name, module in list(sys.modules.items()):
        if not any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        ):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"{name} was loaded outside the expected source tree: {path}"
            )
        files.append(path)
    return files


def run(checkout: Path, revision: str, seed: int) -> HeadParityReport:
    checkout = checkout.resolve()
    validate_official_checkout(checkout, expected_revision=revision)
    mx.set_default_device(mx.gpu)
    torch = cast(HeadTorchRuntime, import_module("torch"))
    classes = _load_official(checkout)
    inputs = _inputs()
    local, reference, weights = _shared_predictors(torch, classes, seed, inputs)
    local_files = loaded_source_files(
        REPO_ROOT, ("sam3_mlx", "scripts._oracle_runtime")
    )
    local_files.append(Path(__file__).resolve())
    local_before = source_identity(REPO_ROOT, local_files)
    rtol, atol = 1e-4, 5e-6
    cases = [
        _case(local, reference, inputs, multimask=multi, rtol=rtol, atol=atol)
        for multi in (False, True)
    ]
    validate_official_checkout(checkout, expected_revision=revision)
    official_files = loaded_source_files(checkout, ("sam3",))
    official_files.append(checkout / "sam3/model/sam1_task_predictor.py")
    local_after = source_identity(REPO_ROOT, local_files)
    if local_before.file_signatures != local_after.file_signatures:
        raise ValueError("Imported local source changed during the numerical run.")
    return HeadParityReport(
        schema_version=1,
        scope="synthetic interactive heads only; no backbone, released checkpoint, mask cleanup, temporal tracking, or performance qualification",
        verdict=Verdict.PASSED
        if all(x.verdict == Verdict.PASSED for x in cases)
        else Verdict.FAILED,
        seed=seed,
        tolerances={"rtol": rtol, "atol": atol},
        runtime_versions={
            name: metadata.version(name)
            for name in ("torch", "torchvision", "mlx", "numpy", "einops")
        },
        devices={
            "official": "torch cpu float32",
            "mlx": f"{mx.default_device()} float32",
        },
        official_source=source_identity(checkout, official_files),
        local_source=local_after,
        source_adapter={
            "omitted_import": "from sam3.model.sam3_tracker_base import Sam3TrackerBase",
            "reason": "annotation-only tracker dependency; runtime uses are rejected",
            "added_import": "from __future__ import annotations",
            "package_loading": "sam3 namespace points to the verified checkout; package initializer is not executed",
            "predictor_code": "all predictor methods retained verbatim",
            "clean_checkout_checks": "before loading and after all numerical cases",
            "local_source_checks": "imported file sizes and modification times unchanged before and after numerical cases",
        },
        original_sizes=inputs.original_sizes,
        object_counts=(2, 1),
        shared_weight_tensors=len(weights),
        cases=cases,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument(
        "--out", type=Path, help="Optional JSON path relative to this repository"
    )
    args = parser.parse_args()
    if args.out is not None and (args.out.is_absolute() or ".." in args.out.parts):
        parser.error("--out must be a repository-relative path without '..'")
    report = run(args.official_checkout, args.official_revision, args.seed)
    rendered = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        output = REPO_ROOT / args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0 if report.verdict == Verdict.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
