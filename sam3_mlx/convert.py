"""Convert and load SAM 3 image checkpoints for MLX.

Two routes produce an MLX ``model.safetensors`` bundle:

- ``load_from_hub`` downloads the pinned preconverted
  ``mlx-community/sam3-image`` revision (``DEFAULT_MLX_CHECKPOINT``).
- ``download_and_convert`` converts the official PyTorch ``sam3.pt`` (requires
  ``torch``, imported lazily) at an immutable source revision. It keeps the
  detector weights and records ``tracker.*`` tensors as ignored keys.

Conversion renames the fused attention ``in_proj_*`` tensors to separate
query/key/value projections and rewrites convolution kernels from PyTorch's
channels-first layout to MLX's channels-last layout (see
``normalize_sam3_image_weight_layout``). Output directories carry a
``conversion-manifest.json`` and a ``model.safetensors.index.json`` and are
published under a file lock so an interrupted run leaves a recoverable state.

Run as a script for a command-line wrapper around the same functions.
"""

import argparse
from contextlib import contextmanager
import errno
import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import NotRequired, Protocol, Required, TypedDict, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from sam3_mlx.mlx_runtime import mlx_array


MLX_COMMUNITY_REPO = "mlx-community/sam3-image"
PYTORCH_REPO = "facebook/sam3"
CONVERSION_MANIFEST = "conversion-manifest.json"
_COMMIT_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class _SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        allow_patterns: Sequence[str],
        revision: str,
        local_dir: str | None = None,
    ) -> str: ...


class _TorchTensor(Protocol):
    def numpy(self) -> npt.NDArray[np.generic]: ...


class _SaveSafetensors(Protocol):
    def __call__(
        self,
        path: str,
        arrays: dict[str, mx.array],
        metadata: dict[str, str] | None = None,
    ) -> object: ...


class _TorchModule(Protocol):
    def load(
        self,
        path: str,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object: ...


class _CliArgs(Protocol):
    mlx_repo: str
    source_revision: str | None
    pytorch_repo: str
    mlx_path: str | None
    convert: bool
    dtype: str


class ConversionProvenance(TypedDict):
    status: Required[str]
    repo: Required[str]
    revision: Required[str]
    architecture: Required[str]
    manifest_path: Required[str | None]
    manifest: NotRequired[dict[str, JsonValue]]


class WeightIndex(TypedDict):
    metadata: dict[str, int]
    weight_map: dict[str, str]


snapshot_download = cast(
    _SnapshotDownload,
    getattr(import_module("huggingface_hub"), "snapshot_download"),
)


def torch_weights(value: object) -> dict[str, _TorchTensor]:
    if not isinstance(value, Mapping):
        raise ValueError("SAM3 PyTorch checkpoint payload must be a mapping.")
    result: dict[str, _TorchTensor] = {}
    for key, tensor in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError("checkpoint weight keys must be strings")
        if not callable(getattr(tensor, "numpy", None)):
            raise TypeError(f"checkpoint weight {key!r} must expose numpy()")
        result[key] = cast(_TorchTensor, tensor)
    return result


def json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = json_value(item)
        return result
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def _json_object(text: str, source: str) -> dict[str, JsonValue]:
    value = json_value(json.loads(text))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class DefaultMlxCheckpoint:
    """Pinned default preconverted MLX image checkpoint.

    The community snapshot does not currently ship a conversion-manifest.json, so
    the immutable revision is the release provenance source of truth. When a
    conversion-manifest.json is present it is also validated against this pin.
    """

    repo: str
    revision: str
    architecture: str = "sam3-image"


# Pinned to the audited mlx-community/sam3-image revision.
DEFAULT_MLX_CHECKPOINT = DefaultMlxCheckpoint(
    repo=MLX_COMMUNITY_REPO,
    revision="b72a14d8127e17e6f2a3d2e075bbbf4307ba146e",
)

SAM3_IMAGE_CONV_TRANSPOSE2D_WEIGHTS = frozenset(
    {
        "backbone.vision_backbone.convs.0.dconv_2x2_0.weight",
        "backbone.vision_backbone.convs.0.dconv_2x2_1.weight",
        "backbone.vision_backbone.convs.1.dconv_2x2.weight",
        "backbone.vision_backbone.sam2_convs.0.dconv_2x2_0.weight",
        "backbone.vision_backbone.sam2_convs.0.dconv_2x2_1.weight",
        "backbone.vision_backbone.sam2_convs.1.dconv_2x2.weight",
    }
)

SAM3_IMAGE_CONV2D_WEIGHTS = frozenset(
    {
        "backbone.vision_backbone.trunk.patch_embed.proj.weight",
        "backbone.vision_backbone.convs.0.conv_1x1.weight",
        "backbone.vision_backbone.convs.0.conv_3x3.weight",
        "backbone.vision_backbone.convs.1.conv_1x1.weight",
        "backbone.vision_backbone.convs.1.conv_3x3.weight",
        "backbone.vision_backbone.convs.2.conv_1x1.weight",
        "backbone.vision_backbone.convs.2.conv_3x3.weight",
        "backbone.vision_backbone.convs.3.conv_1x1.weight",
        "backbone.vision_backbone.convs.3.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.0.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.0.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.1.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.1.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.2.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.2.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.3.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.3.conv_3x3.weight",
        "geometry_encoder.boxes_pool_project.weight",
        "segmentation_head.pixel_decoder.conv_layers.0.weight",
        "segmentation_head.pixel_decoder.conv_layers.1.weight",
        "segmentation_head.pixel_decoder.conv_layers.2.weight",
        "segmentation_head.semantic_seg_head.weight",
        "segmentation_head.instance_seg_head.weight",
    }
)


def normalize_sam3_image_weight_layout(key: str, value: mx.array) -> mx.array:
    """Map known PyTorch conv kernels into MLX's channels-last kernel layout.

    ``Conv2d`` weights go from PyTorch ``(out, in, kH, kW)`` to MLX
    ``(out, kH, kW, in)``; the 2x2 ``ConvTranspose2d`` upsampling kernels go from
    ``(in, out, 2, 2)`` to ``(out, 2, 2, in)``. Only keys listed in the two
    ``SAM3_IMAGE_CONV*`` sets are touched, and only when the shape still has the
    PyTorch pattern, so tensors that are already in MLX layout pass through
    unchanged. All other tensors are returned as is.
    """

    if (
        key in SAM3_IMAGE_CONV_TRANSPOSE2D_WEIGHTS
        and len(value.shape) == 4
        and value.shape[2] == 2
        and value.shape[3] == 2
    ):
        return mx.transpose(value, (1, 2, 3, 0))

    if (
        key in SAM3_IMAGE_CONV2D_WEIGHTS
        and len(value.shape) == 4
        and value.shape[2] == value.shape[3]
        and value.shape[1] != value.shape[2]
    ):
        return mx.transpose(value, (0, 2, 3, 1))

    return value


def load_from_hub(
    hf_repo: str = DEFAULT_MLX_CHECKPOINT.repo,
    local_dir: str | None = None,
    revision: str | None = None,
    *,
    expected_architecture: str = "sam3-image",
    expected_dtype_policy: str | None = None,
    verify_provenance: bool = True,
) -> Path:
    """Download an MLX checkpoint snapshot and return the path to its ``model.safetensors``.

    ``revision`` must be a full 40-character commit SHA (mandatory; there is no
    unpinned default). With ``verify_provenance`` the snapshot is checked by
    ``validate_hub_checkpoint_provenance`` before the path is returned.

    Raises:
        ValueError: ``revision`` is missing or not a full commit SHA, or the
            provenance check fails.
        FileNotFoundError: The snapshot has no ``model.safetensors``.
    """
    if revision is None:
        raise ValueError(
            "load_from_hub requires an immutable Hugging Face revision "
            "(full 40-character commit SHA). Pass revision= explicitly or use "
            "DEFAULT_MLX_CHECKPOINT.revision."
        )
    revision = validate_source_revision(revision)

    if local_dir:
        snapshot_path = snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
            revision=revision,
            local_dir=local_dir,
        )
    else:
        snapshot_path = snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
            revision=revision,
        )
    model_path = Path(snapshot_path)
    weights_file = model_path / "model.safetensors"

    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {hf_repo}.")

    if verify_provenance:
        validate_hub_checkpoint_provenance(
            weights_file.parent,
            expected_repo=hf_repo,
            expected_revision=revision,
            expected_architecture=expected_architecture,
            expected_dtype_policy=expected_dtype_policy,
        )

    return weights_file


def validate_hub_checkpoint_provenance(
    checkpoint_dir: str | Path,
    *,
    expected_repo: str,
    expected_revision: str,
    expected_architecture: str = "sam3-image",
    expected_dtype_policy: str | None = None,
) -> ConversionProvenance:
    """Validate a downloaded MLX checkpoint before model mutation.

    When a conversion-manifest.json is present, its architecture must match the
    pin. Precision-specific artifacts require a manifest with an exact
    floating-point dtype inventory.
    """
    expected_repo = _nonempty_string(expected_repo, "expected_repo")
    expected_architecture = _nonempty_string(
        expected_architecture, "expected_architecture"
    )
    expected_revision = validate_source_revision(expected_revision)

    checkpoint_dir = Path(checkpoint_dir)
    weights_file = checkpoint_dir / "model.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_dir}.")


    manifest_file = checkpoint_dir / CONVERSION_MANIFEST
    provenance: ConversionProvenance = {
        "status": "package-pinned",
        "repo": expected_repo,
        "revision": expected_revision,
        "architecture": expected_architecture,
        "manifest_path": None,
    }
    if not manifest_file.exists() and expected_dtype_policy is not None:
        raise ValueError(
            "Precision-specific MLX checkpoints require a conversion-manifest.json."
        )
    if not manifest_file.exists():
        return provenance

    manifest = _json_object(manifest_file.read_text(), "conversion-manifest.json")
    _nonempty_string(manifest.get("source_repo"), "manifest.source_repo")
    validate_source_revision(manifest.get("source_revision"))
    expected: dict[str, JsonValue] = {"architecture": expected_architecture}
    if expected_dtype_policy is not None:
        from sam3_mlx.precision import parse_checkpoint_dtype

        expected["dtype_policy"] = parse_checkpoint_dtype(expected_dtype_policy)
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={cached!r}, expected={requested!r}"
            for key, (cached, requested) in sorted(mismatches.items())
        )
        raise ValueError(
            "conversion-manifest.json does not match the pinned default "
            f"checkpoint provenance. {details}"
        )
    if expected_dtype_policy is not None:
        dtype_counts = manifest.get("dtype_counts")
        if not isinstance(dtype_counts, dict) or not dtype_counts:
            raise ValueError(
                "Precision-specific conversion manifest must contain dtype_counts."
            )
        floating_dtypes = {
            name
            for name in ("float32", "float16", "bfloat16")
            if dtype_counts.get(name, 0) != 0
        }
        if floating_dtypes != {expected_dtype_policy}:
            raise ValueError(
                "Precision-specific conversion manifest floating dtype inventory "
                f"must be exactly {expected_dtype_policy!r}; got "
                f"{sorted(floating_dtypes)!r}."
            )
        index_file = checkpoint_dir / "model.safetensors.index.json"
        if not index_file.exists():
            raise ValueError(
                "Precision-specific MLX checkpoints require "
                "model.safetensors.index.json."
            )
    provenance["status"] = "manifest-verified"
    provenance["manifest_path"] = str(manifest_file)
    provenance["manifest"] = manifest
    return provenance


def save_weights(
    save_path: str | Path,
    weights: dict[str, mx.array],
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write ``model.safetensors`` and ``model.safetensors.index.json`` into ``save_path``.

    The directory is created if needed. The pair is published under a file lock
    with a backup, so a failed write restores the previous files. The index
    records the total tensor byte size and maps every tensor to
    ``model.safetensors``. Returns the path of ``model.safetensors``.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    total_size = sum(v.nbytes for v in weights.values())
    index_data: WeightIndex = {
        "metadata": {"total_size": total_size},
        "weight_map": {},
    }

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    model_path = save_path / "model.safetensors"
    index_path = save_path / "model.safetensors.index.json"
    with _conversion_destination_lock(save_path):
        _recover_weight_pair(model_path=model_path, index_path=index_path)
        _sweep_stale_weight_temp_files(save_path)
        model_temp = _unique_temp_path(
            save_path, prefix=".model.safetensors.tmp-", suffix=".safetensors"
        )
        index_temp = _unique_temp_path(
            save_path,
            prefix=".model.safetensors.index.tmp-",
            suffix=".json",
        )
        try:
            save_safetensors = cast(_SaveSafetensors, getattr(mx, "save_safetensors"))
            if metadata is None:
                save_safetensors(str(model_temp), weights)
            else:
                save_safetensors(str(model_temp), weights, metadata=metadata)
            _fsync_file(model_temp)
            write_weight_index(index_temp, index_data)
            _fsync_file(index_temp)
            _publish_weight_pair(
                model_temp=model_temp,
                model_path=model_path,
                index_temp=index_temp,
                index_path=index_path,
            )
        finally:
            model_temp.unlink(missing_ok=True)
            index_temp.unlink(missing_ok=True)
    return model_path


def _unique_temp_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)


def _sweep_stale_weight_temp_files(save_path: Path) -> None:
    for pattern in (
        ".model.safetensors.tmp-*",
        ".model.safetensors.index.tmp-*",
    ):
        for candidate in save_path.glob(pattern):
            if candidate.is_file():
                candidate.unlink()


def _recover_weight_pair(*, model_path: Path, index_path: Path) -> None:
    model_backup = model_path.parent / ".model.safetensors.backup"
    index_backup = index_path.parent / ".model.safetensors.index.backup"
    if not model_backup.exists() and not index_backup.exists():
        return
    if model_path.exists() and index_path.exists():
        model_backup.unlink(missing_ok=True)
        index_backup.unlink(missing_ok=True)
        _fsync_directory(model_path.parent)
        return
    if model_backup.exists():
        model_path.unlink(missing_ok=True)
        os.replace(model_backup, model_path)
    if index_backup.exists():
        index_path.unlink(missing_ok=True)
        os.replace(index_backup, index_path)
    _fsync_directory(model_path.parent)


def _publish_weight_pair(
    *,
    model_temp: Path,
    model_path: Path,
    index_temp: Path,
    index_path: Path,
) -> None:
    """Publish a complete weight/index pair under the destination file lock."""

    model_backup = model_path.parent / ".model.safetensors.backup"
    index_backup = index_path.parent / ".model.safetensors.index.backup"
    if model_backup.exists() or index_backup.exists():
        raise RuntimeError("Weight transaction backup exists after recovery.")
    model_had_value = model_path.exists()
    index_had_value = index_path.exists()
    model_published = False
    index_published = False
    published = False
    try:
        if model_had_value:
            os.replace(model_path, model_backup)
        if index_had_value:
            os.replace(index_path, index_backup)
        os.replace(index_temp, index_path)
        index_published = True
        os.replace(model_temp, model_path)
        model_published = True
        _fsync_directory(model_path.parent)
        published = True
    finally:
        if not published:
            if model_published:
                model_path.unlink(missing_ok=True)
            if index_published:
                index_path.unlink(missing_ok=True)
            if model_had_value and model_backup.exists():
                os.replace(model_backup, model_path)
            if index_had_value and index_backup.exists():
                os.replace(index_backup, index_path)
            _fsync_directory(model_path.parent)
        model_backup.unlink(missing_ok=True)
        index_backup.unlink(missing_ok=True)


def _fsync_conversion_directory(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file():
            _fsync_file(path)
    _fsync_directory(directory)


def write_weight_index(path: Path, index: WeightIndex) -> None:
    path.write_text(json.dumps(index, indent=4) + "\n", encoding="utf-8")


def download(hf_repo: str, *, revision: str) -> Path:
    """Download the ``*.pt`` and ``*.json`` files of ``hf_repo`` at ``revision``; return the snapshot directory."""
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            revision=revision,
            allow_patterns=["*.pt", "*.json"],
        )
    )


def update_attn_keys(key: str, mlx_weights: MutableMapping[str, mx.array]) -> None:
    """Replace one fused ``in_proj_weight``/``in_proj_bias`` entry by separate q/k/v entries.

    PyTorch's ``nn.MultiheadAttention`` stacks the query, key and value
    projections along axis 0 in that order; they are split into equal thirds and
    stored as ``query_proj``/``key_proj``/``value_proj`` ``.weight`` or ``.bias``
    under the same prefix. Mutates ``mlx_weights``.
    """
    value = mlx_weights[key]
    del mlx_weights[key]

    if "in_proj_weight" in key:
        qkv, _ = value.shape[0], value.shape[1]
        qkv_dim = qkv // 3
        key_prefix = key.rsplit(".", 1)[0]
        new_dict = {
            f"{key_prefix}.query_proj.weight": value[0:qkv_dim, :],
            f"{key_prefix}.key_proj.weight": value[qkv_dim : 2 * qkv_dim, :],
            f"{key_prefix}.value_proj.weight": value[2 * qkv_dim :, :],
        }
        mlx_weights.update(new_dict)

    if "in_proj_bias" in key:
        qkv = value.shape[0]
        qkv_dim = qkv // 3
        key_prefix = key.rsplit(".", 1)[0]
        new_dict = {
            f"{key_prefix}.query_proj.bias": value[0:qkv_dim],
            f"{key_prefix}.key_proj.bias": value[qkv_dim : 2 * qkv_dim],
            f"{key_prefix}.value_proj.bias": value[2 * qkv_dim :],
        }
        mlx_weights.update(new_dict)


def unwrap_checkpoint_payload(payload: object) -> object:
    if isinstance(payload, Mapping):
        mapping = cast(Mapping[object, object], payload)
        model = mapping.get("model")
        if isinstance(model, Mapping):
            return cast(Mapping[object, object], model)
    return cast(object, payload)


def remap_official_prefix_keys[T](weights: Mapping[str, T]) -> Mapping[str, T]:
    if not any(
        k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in weights
    ):
        return weights

    remapped: dict[str, T] = {}
    for key, value in weights.items():
        if key.startswith("sam3_model."):
            key = "detector." + key[len("sam3_model.") :]
        elif key.startswith("sam2_predictor."):
            key = "tracker." + key[len("sam2_predictor.") :]
        if key in remapped:
            raise ValueError(f"SAM3 source prefixes collide at {key!r}.")
        remapped[key] = value
    return remapped


def _convert_checkpoint_weights(
    weights: Mapping[str, _TorchTensor], *, source_label: str
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    mlx_weights: dict[str, mx.array] = {}
    ignored_keys: list[str] = []
    unmapped_keys: list[str] = []
    for k, v in weights.items():
        source_key = k
        if k.startswith("tracker."):
            ignored_keys.append(source_key)
            continue
        if "detector" in k:
            k = k.replace("detector.", "")
            if k.startswith("backbone."):
                v = mlx_array(v.numpy())
                v = normalize_sam3_image_weight_layout(k, v)
                mlx_weights[k] = v

            elif k.startswith("transformer."):
                v = mlx_array(v.numpy())
                mlx_weights[k] = v

            elif k.startswith("dot_prod_scoring."):
                v = mlx_array(v.numpy())
                mlx_weights[k] = v

            elif k.startswith("segmentation_head."):
                v = mlx_array(v.numpy())
                v = normalize_sam3_image_weight_layout(k, v)
                mlx_weights[k] = v

            elif k.startswith("geometry_encoder."):
                v = mlx_array(v.numpy())
                v = normalize_sam3_image_weight_layout(k, v)
                mlx_weights[k] = v
            else:
                unmapped_keys.append(source_key)
                continue

            if k.endswith("in_proj_weight") or k.endswith("in_proj_bias"):
                update_attn_keys(k, mlx_weights)
        else:
            unmapped_keys.append(source_key)

    if not mlx_weights:
        raise ValueError(
            f"No detector weights were converted from {source_label}. Expected "
            "official SAM3 keys with a detector. prefix."
        )
    if unmapped_keys:
        examples = ", ".join(sorted(unmapped_keys)[:5])
        raise ValueError(
            "SAM3 conversion encountered source keys without a reviewed mapping: "
            f"count={len(unmapped_keys)}; examples={examples}"
        )

    return mlx_weights, tuple(sorted(ignored_keys))


def convert(
    model_path: Path,
    *,
    dtype: str = "float32",
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    """Convert ``<model_path>/sam3.pt`` to MLX weights.

    Loads the PyTorch checkpoint on CPU with ``weights_only=True`` (requires
    ``torch``). Returns ``(weights, ignored_keys)``: MLX arrays keyed by the
    runtime parameter names, with floating tensors cast to ``dtype``, and the
    sorted ``tracker.*`` source keys that were skipped.

    Raises:
        ValueError: No detector weights were found, or a source key has no
            reviewed mapping.
    """
    from sam3_mlx.precision import cast_checkpoint_weights, parse_checkpoint_dtype

    torch = cast(_TorchModule, import_module("torch"))
    weight_file = str(model_path / "sam3.pt")
    weights = torch.load(weight_file, map_location="cpu", weights_only=True)
    weights = unwrap_checkpoint_payload(weights)
    weights = remap_official_prefix_keys(torch_weights(weights))
    mlx_weights, ignored = _convert_checkpoint_weights(
        weights, source_label=weight_file
    )
    return cast_checkpoint_weights(mlx_weights, parse_checkpoint_dtype(dtype)), ignored


def write_conversion_manifest(
    output_dir: Path,
    *,
    source_repo: str,
    source_revision: str,
    source_checkpoint: Path,
    output_checkpoint: Path,
    weights: Mapping[str, mx.array],
    ignored_keys: tuple[str, ...],
    dtype_policy: str = "float32",
) -> Path:
    """Write ``conversion-manifest.json`` describing a conversion and return its path.

    Records the source repo/revision/file name, converter version, output file
    name, tensor counts, ignored source keys and a per-dtype tensor inventory.
    ``mapped_count`` and ``tensor_count`` both count output tensors (after the
    q/k/v split), not source keys.
    """
    from sam3_mlx.precision import dtype_name, parse_checkpoint_dtype

    dtype_policy = parse_checkpoint_dtype(dtype_policy)
    dtype_counts: dict[str, int] = {}
    for value in weights.values():
        name = dtype_name(value.dtype)
        dtype_counts[name] = dtype_counts.get(name, 0) + 1
    manifest = {
        "architecture": "sam3-image",
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_checkpoint": source_checkpoint.name,
        "converter_version": version("sam3-mlx"),
        "output_checkpoint": output_checkpoint.name,
        "mapped_count": len(weights),
        "tensor_count": len(weights),
        "unmapped_keys": [],
        "ignored_keys": list(ignored_keys),
        "dtype_policy": dtype_policy,
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }
    manifest_path = output_dir / CONVERSION_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _validate_cached_conversion(
    weights_file: Path,
    manifest_file: Path,
    *,
    source_repo: str,
    source_revision: str,
    dtype_policy: str = "float32",
) -> None:
    from sam3_mlx.precision import parse_checkpoint_dtype

    index_file = weights_file.with_name("model.safetensors.index.json")
    if not index_file.exists():
        raise ValueError(
            "Cached MLX conversion is missing model.safetensors.index.json."
        )
    manifest = _json_object(manifest_file.read_text(), "conversion-manifest.json")
    index = _json_object(index_file.read_text(), "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Cached MLX conversion index must contain a weight_map.")
    if set(weight_map.values()) != {"model.safetensors"}:
        raise ValueError(
            "Cached MLX conversion index must map every tensor to model.safetensors."
        )
    cached_dtype = manifest.get("dtype_policy", "float32")
    expected = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "dtype_policy": parse_checkpoint_dtype(dtype_policy),
    }
    observed = {
        "source_repo": manifest.get("source_repo"),
        "source_revision": manifest.get("source_revision"),
        "dtype_policy": parse_checkpoint_dtype(cached_dtype),
    }
    mismatches = {
        key: (observed[key], value)
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cached={cached!r}, requested={requested!r}"
            for key, (cached, requested) in sorted(mismatches.items())
        )
        raise ValueError(
            "Cached MLX conversion provenance does not match the requested "
            f"source; use force=True to regenerate. {details}"
        )


def validate_source_revision(source_revision: object) -> str:
    """Return a full 40-character hexadecimal commit SHA, lower-cased.

    Raises:
        TypeError: The value is not a string.
        ValueError: It is not a full commit SHA (branch names and short SHAs are rejected).
    """
    if not isinstance(source_revision, str):
        raise TypeError("source_revision must be a string")
    if not _COMMIT_REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError(
            "source_revision must be a full 40-character hexadecimal commit SHA."
        )
    return source_revision.lower()


@contextmanager
def _conversion_destination_lock(destination: Path) -> Generator[None, None, None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _conversion_backup_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.backup"


def _sweep_stale_conversion_temp_dirs(destination: Path) -> None:
    """Remove staged temp dirs abandoned by crashed conversions.

    Callers must hold the conversion destination lock, so any surviving
    ``.{name}.tmp-*`` directory belongs to a process that no longer exists.
    """

    for candidate in destination.parent.glob(f".{destination.name}.tmp-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


def _recover_conversion_transaction(
    destination: Path,
    *,
    source_repo: str,
    source_revision: str,
    dtype_policy: str,
) -> None:
    """Recover the last complete directory after an interrupted swap."""

    _sweep_stale_conversion_temp_dirs(destination)
    backup = _conversion_backup_path(destination)
    if not backup.exists():
        return
    if not destination.exists():
        os.replace(backup, destination)
        _fsync_directory(destination.parent)
        return

    try:
        _validate_cached_conversion(
            destination / "model.safetensors",
            destination / CONVERSION_MANIFEST,
            source_repo=source_repo,
            source_revision=source_revision,
            dtype_policy=dtype_policy,
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        shutil.rmtree(destination)
        os.replace(backup, destination)
        _fsync_directory(destination.parent)
    else:
        shutil.rmtree(backup)
        _fsync_directory(destination.parent)


def _publish_conversion_directory(staged: Path, destination: Path) -> None:
    """Swap one fully validated directory while retaining a recovery point."""

    backup = _conversion_backup_path(destination)
    if backup.exists():
        raise RuntimeError(f"Conversion recovery backup still exists: {backup}")
    if not destination.exists():
        os.replace(staged, destination)
        _fsync_directory(destination.parent)
        return

    os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except OSError:
        os.replace(backup, destination)
        _fsync_directory(destination.parent)
        raise
    shutil.rmtree(backup)
    _fsync_directory(destination.parent)


def download_and_convert(
    hf_repo: str = PYTORCH_REPO,
    mlx_path: str | Path = "sam3-mod-weights",
    force: bool = False,
    *,
    source_revision: str,
    dtype: str = "float32",
) -> Path:
    """Convert the official PyTorch checkpoint into ``mlx_path`` and return ``model.safetensors``.

    ``source_revision`` must be a full commit SHA. A complete existing conversion
    is reused only if its manifest matches the requested repo, revision and
    ``dtype``; a mismatch or an incomplete directory raises unless ``force`` is
    set. New output is staged in a sibling temporary directory, validated, then
    swapped in under a file lock so readers never see a partial bundle.

    Raises:
        ValueError: Invalid revision or dtype, or a stale/incomplete cache
            without ``force``.
    """
    from sam3_mlx.precision import parse_checkpoint_dtype

    source_revision = validate_source_revision(source_revision)
    dtype = parse_checkpoint_dtype(dtype)
    mlx_path = Path(mlx_path)
    weights_file = mlx_path / "model.safetensors"
    index_file = mlx_path / "model.safetensors.index.json"
    manifest_file = mlx_path / CONVERSION_MANIFEST

    with _conversion_destination_lock(mlx_path):
        _recover_conversion_transaction(
            mlx_path,
            source_repo=hf_repo,
            source_revision=source_revision,
            dtype_policy=dtype,
        )
        if (
            weights_file.exists()
            and index_file.exists()
            and manifest_file.exists()
            and not force
        ):
            _validate_cached_conversion(
                weights_file,
                manifest_file,
                source_repo=hf_repo,
                source_revision=source_revision,
                dtype_policy=dtype,
            )
            return weights_file
        if not force and any(
            path.exists() for path in (weights_file, index_file, manifest_file)
        ):
            raise ValueError(
                "Cached MLX conversion is incomplete and cannot be verified; use "
                "force=True to regenerate it."
            )

        print(f"Downloading and converting weights from {hf_repo}...")
        model_path = download(hf_repo, revision=source_revision)
        source_checkpoint = model_path / "sam3.pt"
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{mlx_path.name}.tmp-",
                dir=mlx_path.parent,
            )
        )
        try:
            mlx_weights, ignored_keys = convert(model_path, dtype=dtype)
            output_checkpoint = save_weights(temp_dir, mlx_weights)
            temp_manifest = write_conversion_manifest(
                temp_dir,
                source_repo=hf_repo,
                source_revision=source_revision,
                source_checkpoint=source_checkpoint,
                output_checkpoint=output_checkpoint,
                weights=mlx_weights,
                ignored_keys=ignored_keys,
                dtype_policy=dtype,
            )
            _validate_cached_conversion(
                output_checkpoint,
                temp_manifest,
                source_repo=hf_repo,
                source_revision=source_revision,
                dtype_policy=dtype,
            )

            _fsync_conversion_directory(temp_dir)
            _publish_conversion_directory(temp_dir, mlx_path)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    return weights_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SAM-3 MLX weights or convert from PyTorch"
    )
    parser.add_argument(
        "--mlx-repo",
        default=MLX_COMMUNITY_REPO,
        type=str,
        help=f"MLX Community repo to download pre-converted weights (default: {MLX_COMMUNITY_REPO})",
    )
    parser.add_argument(
        "--source-revision",
        type=str,
        default=None,
        help="Immutable Hugging Face commit revision required for conversion.",
    )
    parser.add_argument(
        "--pytorch-repo",
        default=PYTORCH_REPO,
        type=str,
        help=f"PyTorch repo to download and convert weights (default: {PYTORCH_REPO})",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default=None,
        help="Local path to save/cache the MLX Model weights.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert from PyTorch weights instead of loading pre-converted MLX weights",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Storage dtype for converted floating-point tensors (default: float32).",
    )
    args = cast(_CliArgs, parser.parse_args())

    if args.convert:
        if not args.source_revision:
            parser.error("--source-revision is required with --convert")
        try:
            args.source_revision = validate_source_revision(args.source_revision)
        except ValueError as exc:
            parser.error(str(exc))
        mlx_path = args.mlx_path or (
            "sam3-mod-weights"
            if args.dtype == "float32"
            else f"sam3-mod-weights-{args.dtype}"
        )
        output_checkpoint = download_and_convert(
            args.pytorch_repo,
            mlx_path,
            source_revision=args.source_revision,
            dtype=args.dtype,
        )
        print(f"Converted weights saved to {output_checkpoint.parent}")
    else:
        print(f"Downloading MLX weights from {args.mlx_repo}...")
        if args.mlx_repo == DEFAULT_MLX_CHECKPOINT.repo and not args.source_revision:
            revision = DEFAULT_MLX_CHECKPOINT.revision
        elif args.source_revision:
            revision = validate_source_revision(args.source_revision)
        else:
            parser.error(
                "--source-revision is required when downloading a non-default "
                "MLX repository (use a full 40-character commit SHA)."
            )
        weights_path = load_from_hub(
            args.mlx_repo,
            args.mlx_path,
            revision=revision,
        )
        print(f"MLX weights available at: {weights_path}")

