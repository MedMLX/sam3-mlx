"""Typed shared boundaries for the official Torch image-oracle runners."""

from __future__ import annotations

import os
import json
import sys
import types
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files
from pathlib import Path
import time
from typing import ContextManager, Never, Protocol, Self, TypedDict, cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
import subprocess

from scripts._resize_reference import resize_reference_float32
from sam3_mlx.parity_evidence import normalize_outputs
from sam3_mlx.release_contract import (
    JsonObject,
    OracleBindings,
    require_json_list,
    require_json_nonnegative_int,
    require_json_object,
    validate_exact_mapping,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

type DynamicCall = Callable[..., object]


class OracleTensor(Protocol):
    def detach(self) -> Self: ...

    def cpu(self) -> Self: ...

    def float(self) -> Self: ...

    def numpy(self) -> NDArray[np.generic]: ...

    def __len__(self) -> int: ...


class OracleState(TypedDict):
    masks: OracleTensor
    boxes: OracleTensor
    scores: OracleTensor


class GeometricPrompt(TypedDict):
    box: list[int | float]
    label: bool


class OracleCase(TypedDict):
    name: str
    resolution: int
    prompt: str | None
    geometric_prompts: list[GeometricPrompt]


def build_oracle_cache_identity(
    *, image: Path, checkpoint: Path, official_python: Path
) -> JsonObject:
    """Identify reusable inputs without binding ephemeral case-spec files."""

    def signature(path: Path) -> JsonObject:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        return {
            "path": str(resolved),
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }

    return {
        "image": signature(image),
        "checkpoint": signature(checkpoint),
        "python": signature(official_python),
        "runtime": [
            signature(REPO_ROOT / name)
            for name in (
                "scripts/run_upstream_image_oracle_hardened.py",
                "scripts/_oracle_runtime.py",
                "scripts/_resize_reference.py",
                "sam3_mlx/release_contract.py",
            )
        ],
    }


def _parse_oracle_metadata(
    archive: NpzFile[np.generic],
    expected_bindings: OracleBindings,
    expected_cache_identity: JsonObject,
) -> JsonObject:
    if "metadata_json" not in archive.files:
        raise ValueError("Oracle cache is missing metadata_json.")
    encoded = archive["metadata_json"]
    if encoded.shape != () or encoded.dtype.kind != "U":
        raise ValueError("Oracle metadata_json must be a scalar JSON string.")
    metadata = require_json_object(json.loads(str(encoded)), field="Oracle metadata")
    schema = require_json_nonnegative_int(
        metadata.get("schema_version"), field="Oracle metadata schema_version"
    )
    if schema != expected_bindings["schema_version"]:
        raise ValueError("Oracle cache schema is stale; recompute the oracle cache.")
    bindings = require_json_object(
        metadata.get("bindings"), field="Oracle metadata bindings"
    )
    validate_exact_mapping(bindings, expected_bindings, label="oracle cache bindings")
    cache_identity = require_json_object(
        metadata.get("cache_identity"), field="Oracle cache identity"
    )
    validate_exact_mapping(
        cache_identity, expected_cache_identity, label="oracle cache input identity"
    )
    return metadata


def public_oracle_metadata(metadata: JsonObject) -> JsonObject:
    """Project semantic evidence without publishing private cache lookup paths."""
    public = dict(metadata)
    del public["cache_identity"]
    public["metadata_projection"] = {
        "purpose": "public-report",
        "omitted_transport_fields": ["cache_identity"],
    }
    return public


def load_official_oracle_archive(
    path: Path,
    *,
    expected_bindings: OracleBindings,
    expected_cache_identity: JsonObject,
    expected_specs: list[OracleCase],
    image_size: tuple[int, int],
) -> tuple[JsonObject, list[dict[str, np.ndarray]]]:
    """Parse cached oracle outputs only when their complete identity matches."""

    try:
        loaded: object = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load oracle cache {path}: {exc}") from exc
    if not isinstance(loaded, NpzFile):
        raise ValueError(f"Oracle cache must be an NPZ archive: {path}.")
    archive_context = cast("NpzFile[np.generic]", loaded)

    with archive_context as archive:
        metadata = _parse_oracle_metadata(
            archive, expected_bindings, expected_cache_identity
        )
        cases_metadata = require_json_list(
            metadata.get("cases"), field="Oracle metadata cases"
        )
        if len(cases_metadata) != len(expected_specs):
            raise ValueError("Oracle cache case metadata count is invalid.")
        expected_names = [spec["name"] for spec in expected_specs]
        observed_names = [
            require_json_object(case, field="Oracle case metadata").get("name")
            for case in cases_metadata
        ]
        if observed_names != expected_names:
            raise ValueError(
                "Oracle cache case profile mismatch: "
                f"observed={observed_names}, expected={expected_names}."
            )

        width, height = image_size
        outputs: list[dict[str, np.ndarray]] = []
        for index, (spec, raw_metadata) in enumerate(
            zip(expected_specs, cases_metadata, strict=True)
        ):
            case_metadata = require_json_object(
                raw_metadata, field=f"Oracle case {spec['name']!r} metadata"
            )
            for field in ("resolution", "prompt", "geometric_prompts"):
                if case_metadata.get(field) != spec[field]:
                    raise ValueError(
                        f"Oracle case {spec['name']!r} {field} metadata drifted."
                    )
            try:
                raw = {
                    field: np.array(archive[f"case_{index}_{field}"], copy=True)
                    for field in ("masks", "boxes", "scores")
                }
            except KeyError as exc:
                raise ValueError(
                    f"Oracle cache is missing arrays for case {spec['name']!r}."
                ) from exc
            normalized = normalize_outputs(raw, label="official")
            if normalized["masks"].shape[1:] != (height, width):
                raise ValueError(
                    f"Oracle case {spec['name']!r} masks have spatial shape "
                    f"{normalized['masks'].shape[1:]}, expected {(height, width)}."
                )
            count = require_json_nonnegative_int(
                case_metadata.get("detection_count"),
                field=f"Oracle case {spec['name']!r} detection_count",
            )
            if count != len(normalized["scores"]):
                raise ValueError(
                    f"Oracle case {spec['name']!r} detection metadata does not "
                    "match its stored arrays."
                )
            outputs.append(cast(dict[str, NDArray[np.generic]], normalized))
    return metadata, outputs


class OracleResize(Protocol):
    size: tuple[int, int]

    def __call__(self, image: object) -> object: ...


class OracleTransform(Protocol):
    transforms: list[Callable[[object], object]]


class OracleTransformOwner(Protocol):
    transform: OracleTransform


class OracleProcessor(OracleTransformOwner, Protocol):
    def reset_all_prompts(self, state: OracleState) -> object: ...

    def set_text_prompt(self, prompt: str, state: OracleState) -> OracleState: ...

    def add_geometric_prompt(
        self, box: Sequence[int | float], label: bool, state: OracleState
    ) -> OracleState: ...

    def set_image(self, image: object) -> OracleState: ...


class TensorClass(Protocol):
    pin_memory: DynamicCall


class TorchRuntime(Protocol):
    zeros: DynamicCall
    arange: DynamicCall
    Tensor: TensorClass
    autocast: Callable[..., ContextManager[None]]
    bfloat16: object
    __version__: object


class OracleModelBuilder(Protocol):
    def build_sam3_image_model(
        self,
        *,
        checkpoint_path: str,
        load_from_HF: bool,
        device: str,
        enable_inst_interactivity: bool,
    ) -> OracleModel: ...


class OracleProcessorFactory(Protocol):
    def __call__(
        self,
        model: OracleModel,
        *,
        device: str,
        resolution: int,
        confidence_threshold: float,
    ) -> OracleProcessor: ...


class _SavezCompressed(Protocol):
    def __call__(self, file: str | Path, **arrays: object) -> None: ...


@dataclass(frozen=True)
class ConstructionAdapters:
    zeros: DynamicCall
    arange: DynamicCall
    pin_memory: DynamicCall


@dataclass(frozen=True)
class OfficialImageOracleSession:
    torch: TorchRuntime
    model: OracleModel
    processor_factory: OracleProcessorFactory
    cold_load_s: float
    official_imports: Mapping[str, str]


class _RopeAttention(Protocol):
    use_rope: bool
    rope_interp: bool
    rope_pt_size: Sequence[int]
    freqs_cis: object

    def compute_cis(self, *, end_x: int, end_y: int, scale_pos: float) -> object: ...


class _RopeBlock(Protocol):
    window_size: int
    attn: _RopeAttention


class _VisionTrunk(Protocol):
    blocks: Sequence[_RopeBlock]


class _VisionBackbone(Protocol):
    trunk: _VisionTrunk


class _Backbone(Protocol):
    vision_backbone: _VisionBackbone


class OracleModel(Protocol):
    backbone: _Backbone


def validate_official_checkout(checkout: Path, *, expected_revision: str) -> str:
    """Require an exact, clean official checkout before oracle execution."""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise ValueError(
            f"Official checkout must be {expected_revision}, found {revision}."
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status.strip():
        raise ValueError(
            f"Official checkout must be clean; git status reported:\n{status.rstrip()}"
        )
    submodules = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    dirty_submodules = [line for line in submodules if line[:1] in {"+", "-", "U"}]
    if dirty_submodules:
        raise ValueError(
            "Official checkout submodules must match the pinned commit:\n"
            + "\n".join(dirty_submodules)
        )
    return revision


def run_upstream_oracle_subprocess(
    *,
    runner_path: Path,
    official_python: Path,
    official_checkout: Path,
    checkpoint_path: Path,
    output_path: Path,
    confidence_threshold: float,
    extra_arguments: Sequence[str] = (),
) -> None:
    """Spawn one pinned upstream oracle runner inside the official environment.

    The official checkout is prepended to ``PYTHONPATH`` and the subprocess runs
    from the repository root.
    """

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(official_checkout), existing_pythonpath) if value
    )
    arguments = [
        str(official_python),
        str(runner_path),
        "--official-checkout",
        str(official_checkout),
    ]
    arguments.extend(
        [
            "--checkpoint",
            str(checkpoint_path),
            *extra_arguments,
            "--out",
            str(output_path),
            "--confidence-threshold",
            str(confidence_threshold),
        ]
    )
    subprocess.run(
        arguments,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def _require_module_from_checkout(
    module: types.ModuleType, checkout: Path, *, label: str
) -> str:
    source = getattr(module, "__file__", None)
    if not source:
        raise ValueError(f"{label} has no import source path.")
    resolved = Path(source).resolve()
    try:
        relative = resolved.relative_to(checkout.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} was imported from {resolved}, outside {checkout.resolve()}."
        ) from exc
    return str(relative)


def _official_resources(official_checkout: Path) -> types.ModuleType:
    """Locate only the pinned tokenizer resource without legacy setuptools."""
    module = types.ModuleType("pkg_resources")

    def resource_filename(package: str, resource: str) -> str:
        if package != "sam3" or resource != "assets/bpe_simple_vocab_16e6.txt.gz":
            raise ValueError(
                f"Unexpected official resource request: {package}/{resource}"
            )
        path = Path(str(files(package).joinpath(resource))).resolve(strict=True)
        expected = (official_checkout / "sam3" / resource).resolve(strict=True)
        pinned = subprocess.check_output(
            ["git", "show", f"HEAD:sam3/{resource}"], cwd=official_checkout
        )
        if path != expected or path.read_bytes() != pinned:
            raise ValueError(
                "Official tokenizer resource differs from its pinned Git blob."
            )
        return str(path)

    setattr(module, "resource_filename", resource_filename)
    return module


@contextmanager
def official_tokenizer_resources(
    official_checkout: Path,
) -> Generator[None, None, None]:
    """Scope the pinned tokenizer resource locator to one reference process."""
    original = sys.modules.get("pkg_resources")
    sys.modules["pkg_resources"] = _official_resources(official_checkout)
    try:
        yield
    finally:
        if original is None:
            del sys.modules["pkg_resources"]
        else:
            sys.modules["pkg_resources"] = original


def configure_official_image_resize(
    processor: OracleTransformOwner, resolution: int
) -> None:
    """Preserve the FP32-resize/uint8-round checkpoint input under patched TorchVision."""
    resize_type = cast(
        type[OracleResize],
        getattr(import_module("torchvision.transforms.v2"), "Resize"),
    )
    transforms = processor.transform.transforms
    if len(transforms) != 4 or not isinstance(transforms[1], resize_type):
        raise ValueError("Pinned official processor resize layout changed.")

    def resize_pixels(image: object) -> object:
        return resize_reference_float32(image, (resolution, resolution))

    # Patched TorchVision otherwise selects its new ARM uint8 kernel.
    transforms[1] = resize_pixels


@contextmanager
def open_official_image_oracle(
    *,
    official_checkout: Path,
    checkpoint: Path,
) -> Generator[OfficialImageOracleSession, None, None]:
    """Build one pinned CPU oracle and restore all construction adapters."""

    torch_runtime = cast(TorchRuntime, import_module("torch"))
    originals = install_cpu_oracle_adapters(torch_runtime)
    try:
        with official_tokenizer_resources(official_checkout):
            model_builder = import_module("sam3.model_builder")
            processor_class = getattr(
                import_module("sam3.model.sam3_image_processor"), "Sam3Processor"
            )
            processor_module_name = getattr(processor_class, "__module__", None)
            if not isinstance(processor_module_name, str):
                raise ValueError("Sam3Processor has no module name.")
            builder = cast(OracleModelBuilder, model_builder)
            original_factory = cast(OracleProcessorFactory, processor_class)

            def processor_factory(
                model: OracleModel,
                *,
                device: str,
                resolution: int,
                confidence_threshold: float,
            ) -> OracleProcessor:
                processor = original_factory(
                    model,
                    device=device,
                    resolution=resolution,
                    confidence_threshold=confidence_threshold,
                )
                configure_official_image_resize(processor, resolution)
                return processor

            model_builder_source = _require_module_from_checkout(
                model_builder,
                official_checkout,
                label="sam3.model_builder",
            )
            processor_module = sys.modules[processor_module_name]
            processor_source = _require_module_from_checkout(
                processor_module,
                official_checkout,
                label=processor_module_name,
            )
            load_started = time.perf_counter()
            try:
                model = builder.build_sam3_image_model(
                    checkpoint_path=str(checkpoint),
                    load_from_HF=False,
                    device="cpu",
                    enable_inst_interactivity=False,
                )
            finally:
                restore_construction_adapters(torch_runtime, originals)
            session = OfficialImageOracleSession(
                torch=torch_runtime,
                model=model,
                processor_factory=processor_factory,
                cold_load_s=time.perf_counter() - load_started,
                official_imports={
                    "model_builder": model_builder_source,
                    "image_processor": processor_source,
                },
            )
            yield session
    finally:
        restore_construction_adapters(torch_runtime, originals)
        torch_runtime.Tensor.pin_memory = originals.pin_memory


def _cpu_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    updated = dict(kwargs)
    if str(updated.get("device", "")).startswith("cuda"):
        updated["device"] = "cpu"
    return updated


def install_cpu_oracle_adapters(torch: TorchRuntime) -> ConstructionAdapters:
    """Install only the construction-time CPU adapters required upstream."""

    edt_module = types.ModuleType("sam3.model.edt")

    def unavailable_edt(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise RuntimeError("Triton EDT is unavailable in the image-only CPU oracle.")

    setattr(edt_module, "edt_triton", unavailable_edt)
    sys.modules["sam3.model.edt"] = edt_module
    originals = ConstructionAdapters(
        zeros=torch.zeros,
        arange=torch.arange,
        pin_memory=torch.Tensor.pin_memory,
    )

    def zeros(*args: object, **kwargs: object) -> object:
        return originals.zeros(*args, **_cpu_kwargs(kwargs))

    def arange(*args: object, **kwargs: object) -> object:
        return originals.arange(*args, **_cpu_kwargs(kwargs))

    def no_pin_memory(tensor: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return tensor

    torch.zeros = zeros
    torch.arange = arange
    torch.Tensor.pin_memory = no_pin_memory
    return originals


def restore_construction_adapters(
    torch: TorchRuntime, originals: ConstructionAdapters
) -> None:
    torch.zeros = originals.zeros
    torch.arange = originals.arange


def run_prompt(
    processor: OracleProcessor, state: OracleState, spec: OracleCase
) -> OracleState:
    processor.reset_all_prompts(state)
    prompt = spec["prompt"]
    if prompt is not None:
        state = processor.set_text_prompt(prompt, state)
    for geometric_prompt in spec["geometric_prompts"]:
        state = processor.add_geometric_prompt(
            geometric_prompt["box"], geometric_prompt["label"], state
        )
    return state


def set_official_global_rope_grid(model: OracleModel, resolution: int) -> None:
    """Recompute official global-attention RoPE for a processor resolution."""

    grid_size = resolution // 14
    for block in model.backbone.vision_backbone.trunk.blocks:
        if block.window_size != 0 or not block.attn.use_rope:
            continue
        attention = block.attn
        scale_pos = 1.0
        if attention.rope_interp:
            scale_pos = attention.rope_pt_size[0] / grid_size
        attention.freqs_cis = attention.compute_cis(
            end_x=grid_size,
            end_y=grid_size,
            scale_pos=scale_pos,
        )


def _string_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} must use string keys.")
    return cast(Mapping[str, object], mapping)


def validate_case_specs(value: object) -> list[OracleCase]:
    """Validate parsed case JSON and return its precise runtime contract."""

    if not isinstance(value, list) or not value:
        raise ValueError("Oracle cases must be a non-empty JSON list.")
    raw_specs = cast(list[object], value)
    names: set[str] = set()
    validated: list[OracleCase] = []
    required = {"name", "resolution", "prompt", "geometric_prompts"}
    for index, raw_spec in enumerate(raw_specs):
        spec = _string_mapping(raw_spec, f"Oracle case {index}")
        if set(spec) != required:
            raise ValueError(
                f"Oracle case {index} fields must be exactly {sorted(required)}."
            )
        name = spec["name"]
        resolution = spec["resolution"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Oracle case name is invalid or duplicated: {name!r}.")
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise ValueError(f"Oracle case {name!r} resolution must be an integer.")
        if resolution <= 0 or resolution % 14 != 0:
            raise ValueError(
                f"Oracle case {name!r} resolution must be a positive multiple of 14."
            )
        prompt = spec["prompt"]
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"Oracle case {name!r} prompt must be a string or null.")
        prompts_value = spec["geometric_prompts"]
        if not isinstance(prompts_value, list):
            raise ValueError(f"Oracle case {name!r} geometric_prompts must be a list.")
        geometric_prompts: list[GeometricPrompt] = []
        for prompt_index, raw_prompt in enumerate(cast(list[object], prompts_value)):
            geometric_prompt = _string_mapping(
                raw_prompt, f"Oracle case {name!r} geometric prompt {prompt_index}"
            )
            if set(geometric_prompt) != {"box", "label"}:
                raise ValueError(
                    f"Oracle case {name!r} geometric prompts require box and label."
                )
            box_value = geometric_prompt["box"]
            if not isinstance(box_value, list):
                raise ValueError(f"Oracle case {name!r} contains an invalid box.")
            box_values = cast(list[object], box_value)
            if len(box_values) != 4:
                raise ValueError(f"Oracle case {name!r} contains an invalid box.")
            box: list[int | float] = []
            for coordinate in box_values:
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, int | float)
                    or not np.isfinite(coordinate)
                ):
                    raise ValueError(f"Oracle case {name!r} contains an invalid box.")
                box.append(coordinate)
            label = geometric_prompt["label"]
            if not isinstance(label, bool):
                raise ValueError(f"Oracle case {name!r} contains an invalid box label.")
            geometric_prompts.append({"box": box, "label": label})
        names.add(name)
        validated.append(
            {
                "name": name,
                "resolution": resolution,
                "prompt": prompt,
                "geometric_prompts": geometric_prompts,
            }
        )
    return validated


def save_oracle_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write a dynamic named-array mapping through NumPy's archive boundary."""

    save_arrays = cast(_SavezCompressed, np.savez_compressed)
    save_arrays(path, **dict(arrays))
