"""Raw conversion of the pinned SAM 3.1 checkpoint and verification of the bundle.

``convert_sam31_checkpoint`` copies the official PyTorch tensors into an MLX
``model.safetensors`` without renaming keys or changing layouts (that happens
later in the checkpoint loader), skipping only values that the model regenerates
(see ``is_generated_checkpoint_key``). A ``conversion-manifest.json`` records the
pinned source, tensor counts, byte totals and which source keys were skipped.
``parse_sam31_bundle`` re-checks those bindings before a bundle is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Mapping, Protocol, Sequence, cast

import mlx.core as mx
import numpy as np

from sam3_mlx.checkpoint_loading import is_generated_checkpoint_key
from sam3_mlx.convert import save_weights
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.release_contract import (
    require_json_nonnegative_int,
    require_json_object,
    require_json_string,
)
from sam3_mlx.temporal_contract import (
    SAM31_CHECKPOINT_FILENAME,
    SAM31_CHECKPOINT_REPO,
    SAM31_CHECKPOINT_REVISION,
)

DEFAULT_SAM31_SOURCE = (
    Path("checkpoints/official/facebook-sam31") / SAM31_CHECKPOINT_FILENAME
)
DEFAULT_SAM31_BUNDLE = Path("checkpoints/derived/sam31-raw-fp32")


class _TorchTensor(Protocol):
    def detach(self) -> _TorchTensor: ...
    def cpu(self) -> _TorchTensor: ...
    def numpy(self) -> np.ndarray: ...
    def float(self) -> _TorchTensor: ...


@dataclass(frozen=True)
class Sam31CheckpointBundle:
    """A verified conversion output and the bindings used to trust it.

    ``source_tensor_count`` equals ``output_tensor_count`` plus the number of
    ``ignored_generated_source_keys``. ``file_signatures`` holds
    ``(path, (size_bytes, mtime_ns))`` for later change detection; no content
    hashes are kept.
    """

    source: Path
    weights: Path
    index: Path
    manifest: Path
    source_tensor_count: int
    output_tensor_count: int
    ignored_generated_source_keys: tuple[str, ...]
    indexed_keys: tuple[str, ...]
    file_signatures: tuple[tuple[Path, tuple[int, int]], ...]
    total_bytes: int


@dataclass(frozen=True)
class _OutputBinding:
    weights: Path
    index: Path
    indexed_keys: tuple[str, ...]
    tensor_count: int
    total_bytes: int


def checkpoint_weights(payload: object) -> Mapping[str, object]:
    """Return the string-keyed tensor mapping, unwrapping a nested ``"model"`` entry."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    raw = cast(Mapping[object, object], payload)
    if "model" in raw:
        nested = raw["model"]
        if not isinstance(nested, Mapping):
            raise ValueError("Checkpoint model payload must be a mapping.")
        raw = cast(Mapping[object, object], nested)
    weights: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint keys must be strings.")
        weights[key] = value
    return weights


def to_mlx_weights(weights: Mapping[str, object]) -> dict[str, mx.array]:
    """Convert PyTorch tensors to MLX arrays, skipping model-generated keys.

    Source dtypes are preserved except ``torch.bfloat16``, which is upcast to
    float32 because NumPy has no bfloat16 for the transfer. Float64 and
    non-numeric dtypes, non-tensor values and any NaN/Inf raise.

    Raises:
        TypeError: A value is not a tensor or has an unsupported dtype.
        ValueError: A tensor contains NaN/Inf, or nothing remains to convert.
    """
    converted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if is_generated_checkpoint_key(key):
            continue
        if not all(
            callable(getattr(value, method, None))
            for method in ("detach", "cpu", "numpy")
        ):
            raise TypeError(f"Checkpoint value {key!r} is not a tensor.")
        tensor = cast(_TorchTensor, value).detach().cpu()
        if str(getattr(tensor, "dtype", None)) == "torch.bfloat16":
            tensor = tensor.float()
        array = to_numpy(tensor.numpy())
        if array.dtype.kind not in "biuf" or array.dtype == np.float64:
            raise TypeError(
                f"Checkpoint value {key!r} has unsupported dtype {array.dtype}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Checkpoint value {key!r} contains NaN or Inf.")
        converted[key] = mx.array(array)
    if not converted:
        raise ValueError("Checkpoint contains no tensors.")
    return converted


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _source_binding(source: Path, provenance: dict[str, object]) -> int:
    expected = {
        "repo": SAM31_CHECKPOINT_REPO,
        "revision": SAM31_CHECKPOINT_REVISION,
        "filename": SAM31_CHECKPOINT_FILENAME,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"SAM3.1 conversion source {field} does not match the pin."
            )
    if not source.is_file():
        raise FileNotFoundError(f"SAM3.1 original checkpoint is missing: {source}.")
    count = require_json_nonnegative_int(
        provenance.get("tensor_count"), field="source tensor_count"
    )
    if not count:
        raise ValueError("SAM3.1 conversion tensor counts must be positive.")
    return count


def _output_binding(directory: Path, output: dict[str, object]) -> _OutputBinding:
    if output.get("path") != "model.safetensors":
        raise ValueError("SAM3.1 conversion must name model.safetensors.")
    weights = directory / "model.safetensors"
    index = directory / "model.safetensors.index.json"
    if not weights.is_file():
        raise FileNotFoundError(f"SAM3.1 conversion is missing {weights}.")
    index_bytes = index.read_bytes()
    index_data = require_json_object(json.loads(index_bytes), field="index")
    weight_map = require_json_object(index_data.get("weight_map"), field="weight_map")
    if not weight_map or any(value != weights.name for value in weight_map.values()):
        raise ValueError(
            "SAM3.1 conversion index must bind every tensor to model.safetensors."
        )
    count = require_json_nonnegative_int(
        output.get("tensor_count"), field="output tensor_count"
    )
    if not count:
        raise ValueError("SAM3.1 conversion tensor counts must be positive.")
    if count != len(weight_map):
        raise ValueError("SAM3.1 conversion source accounting does not balance.")
    total_bytes = require_json_nonnegative_int(
        output.get("total_bytes"), field="output total_bytes"
    )
    metadata = require_json_object(index_data.get("metadata"), field="index metadata")
    index_size = require_json_nonnegative_int(
        metadata.get("total_size"), field="index total_size"
    )
    if not total_bytes or index_size != total_bytes:
        raise ValueError("SAM3.1 conversion byte accounting does not match its index.")
    return _OutputBinding(
        weights, index, tuple(sorted(weight_map)), count, total_bytes
    )


def _generated_source_keys(conversion: dict[str, object]) -> tuple[str, ...]:
    ignored = conversion.get("ignored_generated_source_keys")
    if not isinstance(ignored, list):
        raise ValueError("ignored_generated_source_keys must be a list.")
    generated = tuple(
        require_json_string(key, field="ignored key")
        for key in cast(list[object], ignored)
    )
    if any(not is_generated_checkpoint_key(key) for key in generated):
        raise ValueError("SAM3.1 conversion ignored a required source weight.")
    if len(set(generated)) != len(generated):
        raise ValueError("SAM3.1 conversion repeated an ignored source key.")
    if conversion.get("all_retained_tensors_finite") is not True:
        raise ValueError("SAM3.1 conversion has no finite-value proof.")
    return generated


def parse_sam31_bundle(source: Path, directory: Path) -> Sam31CheckpointBundle:
    """Verify a converted bundle against the pinned source and return its bindings.

    Checks the manifest schema, the pinned repo/revision/filename, that the index
    maps every tensor to ``model.safetensors`` with matching counts and byte
    totals, and that source tensors equal output tensors plus skipped generated
    keys. Does not read the tensor data itself.

    Raises:
        ValueError: Any binding or count disagrees.
        FileNotFoundError: The source or output weights are missing.
    """
    manifest = directory / "conversion-manifest.json"
    raw = require_json_object(json.loads(manifest.read_bytes()), field="manifest")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported SAM3.1 conversion manifest schema_version.")
    source_count = _source_binding(
        source, require_json_object(raw.get("source"), field="source")
    )
    output = _output_binding(
        directory, require_json_object(raw.get("output"), field="output")
    )
    generated = _generated_source_keys(
        require_json_object(raw.get("conversion"), field="conversion")
    )
    if source_count != output.tensor_count + len(generated):
        raise ValueError("SAM3.1 conversion source accounting does not balance.")
    signatures = tuple(
        (path, _file_signature(path))
        for path in (source, output.weights, output.index, manifest)
    )
    return Sam31CheckpointBundle(
        source,
        output.weights,
        output.index,
        manifest,
        source_count,
        output.tensor_count,
        generated,
        output.indexed_keys,
        signatures,
        output.total_bytes,
    )


def require_stable_sam31_bundle(bundle: Sam31CheckpointBundle) -> None:
    """Raise ``RuntimeError`` if any bundle file's size or mtime changed since parsing."""
    for path, expected in bundle.file_signatures:
        if _file_signature(path) != expected:
            raise RuntimeError(
                f"SAM3.1 qualification input changed during execution: {path}."
            )


def _conversion_manifest(
    source: Path,
    output: Path,
    source_weights: Mapping[str, object],
    weights: dict[str, mx.array],
    command: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": {
            "repo": SAM31_CHECKPOINT_REPO,
            "revision": SAM31_CHECKPOINT_REVISION,
            "filename": SAM31_CHECKPOINT_FILENAME,
            "tensor_count": len(source_weights),
        },
        "output": {
            "path": output.name,
            "tensor_count": len(weights),
            "total_bytes": sum(value.nbytes for value in weights.values()),
        },
        "conversion": {
            "policy": "preserve retained raw tensors; regenerate derived buffers; layout normalization occurs in audited MLX checkpoint loader",
            "dtype_policy": "preserve source dtype; upcast torch.bfloat16 to float32 for NumPy transport",
            "ignored_generated_source_keys": sorted(set(source_weights) - set(weights)),
            "all_retained_tensors_finite": True,
            "command": list(command),
            "python": platform.python_version(),
            "torch": str(
                getattr(importlib.import_module("torch"), "__version__", "unknown")
            ),
            "mlx": str(
                getattr(importlib.import_module("mlx"), "__version__", "unknown")
            ),
        },
    }


def _load_sam31_source(source: Path) -> Mapping[str, object]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError(
            "SAM3.1 conversion requires PyTorch. Install the oracle dependency "
            "group with 'uv sync --group oracle', or pass an existing MLX "
            "checkpoint_path to skip automatic conversion."
        ) from exc
    return checkpoint_weights(
        torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
    )


def convert_sam31_checkpoint(
    source: Path, directory: Path, *, command: Sequence[str]
) -> Sam31CheckpointBundle:
    """Convert ``source`` into a new bundle directory and return the verified bundle.

    Requires PyTorch. ``command`` is recorded in the manifest. The output is
    staged in a sibling temporary directory, verified, then renamed into place;
    an existing ``directory`` is never overwritten.

    Raises:
        FileExistsError: ``directory`` already exists.
        FileNotFoundError: ``source`` is missing.
        RuntimeError: PyTorch is not installed, or ``source`` changed mid-run.
    """
    if directory.exists():
        raise FileExistsError(f"SAM3.1 output directory already exists: {directory}.")
    if not source.is_file():
        raise FileNotFoundError(f"SAM3.1 source checkpoint is missing: {source}.")
    signature = _file_signature(source)
    source_weights = _load_sam31_source(source)
    weights = to_mlx_weights(source_weights)
    if _file_signature(source) != signature:
        raise RuntimeError("SAM3.1 source checkpoint changed during conversion.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{directory.name}.tmp-", dir=directory.parent
    ) as temporary:
        staged = Path(temporary) / directory.name
        output = save_weights(staged, weights, metadata={"format": "mlx"})
        manifest = _conversion_manifest(
            source, output, source_weights, weights, command
        )
        (staged / "conversion-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        parse_sam31_bundle(source, staged)
        if directory.exists():
            raise FileExistsError(
                f"SAM3.1 output directory already exists: {directory}."
            )
        staged.rename(directory)
    return parse_sam31_bundle(source, directory)


def prepare_sam31_checkpoint(
    source: Path, directory: Path, *, command: Sequence[str]
) -> Sam31CheckpointBundle:
    """Reuse a validated bundle or convert once; never replace an invalid cache.

    An existing ``directory`` is verified with ``parse_sam31_bundle`` and raises
    if invalid instead of being reconverted.
    """
    if directory.exists():
        return parse_sam31_bundle(source, directory)
    return convert_sam31_checkpoint(source, directory, command=command)
