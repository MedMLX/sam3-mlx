from sam3_mlx.mlx_runtime import to_numpy
import json
import multiprocessing
import os
from typing import Callable, cast
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import pytest
from hypothesis import given, strategies as st

import sam3_mlx.convert as convert_module
from sam3_mlx.convert import (
    DEFAULT_MLX_CHECKPOINT,
    _convert_checkpoint_weights,  # pyright: ignore[reportPrivateUsage]
    _validate_cached_conversion,  # pyright: ignore[reportPrivateUsage]
    write_conversion_manifest,
    download_and_convert,
    load_from_hub,
    remap_official_prefix_keys,
    save_weights,
    validate_hub_checkpoint_provenance,
    validate_source_revision,
)


class _TorchLikeTensor:
    def __init__(self, value: object) -> None:
        self._value = to_numpy(value, dtype=np.float32)

    def numpy(self) -> NDArray[np.float32]:
        return self._value


def _unexpected_download(*args: object, **kwargs: object) -> Path:
    raise AssertionError("validated or recovered cache must avoid a second download")


def _concurrent_save_worker(directory: str, value: float) -> None:
    save_weights(
        directory,
        {
            "head.weight": mx.full((256, 256), value, dtype=mx.float32),
            "head.bias": mx.full((256,), value, dtype=mx.float32),
        },
    )


def _crash_before_weight_publication(directory: str) -> None:
    original = cast(
        Callable[[str, dict[str, mx.array]], object], getattr(mx, "save_safetensors")
    )

    def terminate_after_serialization(path: str, arrays: dict[str, mx.array]) -> None:
        original(path, arrays)
        os._exit(17)

    mx.save_safetensors = terminate_after_serialization
    save_weights(
        directory,
        {"head.weight": mx.ones((16, 16), dtype=mx.float32)},
    )


def _write_fixture_index(directory: Path) -> None:
    index = directory / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 9},
                "weight_map": {"fixture.weight": "model.safetensors"},
            }
        )
    )


def test_conversion_normalizes_conv_layout_once() -> None:
    source = np.arange(2 * 3 * 1 * 1, dtype=np.float32).reshape(2, 3, 1, 1)
    weights, ignored = _convert_checkpoint_weights(
        {
            "detector.geometry_encoder.boxes_pool_project.weight": (
                _TorchLikeTensor(source)
            ),
            "tracker.unused.weight": _TorchLikeTensor([1.0]),
        },
        source_label="fixture",
    )

    converted = to_numpy(weights["geometry_encoder.boxes_pool_project.weight"])
    np.testing.assert_array_equal(converted, source.transpose(0, 2, 3, 1))
    assert converted.shape == (2, 1, 1, 3)
    assert ignored == ("tracker.unused.weight",)


@pytest.mark.parametrize(("source_prefix", "target_prefix"), [
    ("sam3_model.", "detector."), ("sam2_predictor.", "tracker."),
])
@given(st.dictionaries(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789._", min_size=1, max_size=20),
    st.integers(),
    max_size=10,
))
def test_official_prefix_remapping_roundtrip_and_idempotence(
    source_prefix: str, target_prefix: str, values: dict[str, int],
) -> None:
    source = {source_prefix + key: value for key, value in values.items()}
    remapped = remap_official_prefix_keys(source)
    restored = {
        source_prefix + key.removeprefix(target_prefix): value
        for key, value in remapped.items()
    }
    assert restored == source
    assert remap_official_prefix_keys(remapped) == remapped


@pytest.mark.parametrize(("source_prefix", "target_prefix"), [
    ("sam3_model.", "detector."), ("sam2_predictor.", "tracker."),
])
@given(
    suffix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789._", min_size=1, max_size=20),
    reverse=st.booleans(),
)
def test_official_prefix_remapping_rejects_alias_collisions(
    source_prefix: str, target_prefix: str, suffix: str, reverse: bool,
) -> None:
    sources = [(source_prefix + suffix, 1), (target_prefix + suffix, 2)]
    if reverse:
        sources.reverse()
    with pytest.raises(ValueError, match="source prefixes collide"):
        remap_official_prefix_keys(dict(sources))


def test_conversion_manifest_records_pinned_source_and_inventory(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "sam3.pt"
    source_checkpoint.write_bytes(b"official-checkpoint")
    weights = {
        "head.weight": mx.ones((2, 3), dtype=mx.float32),
        "head.bias": mx.zeros((2,), dtype=mx.float16),
    }
    output_checkpoint = save_weights(tmp_path, weights)

    manifest_path = write_conversion_manifest(
        tmp_path,
        source_repo="facebook/sam3",
        source_revision="a" * 40,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        weights=weights,
        ignored_keys=("tracker.memory.weight",),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["architecture"] == "sam3-image"
    assert manifest["source_repo"] == "facebook/sam3"
    assert manifest["source_revision"] == "a" * 40
    assert manifest["source_checkpoint"] == "sam3.pt"
    assert manifest["output_checkpoint"] == output_checkpoint.name
    assert manifest["mapped_count"] == 2
    assert manifest["tensor_count"] == 2
    assert manifest["unmapped_keys"] == []
    assert manifest["ignored_keys"] == ["tracker.memory.weight"]
    assert manifest["dtype_policy"] == "float32"
    assert manifest["dtype_counts"] == {"float16": 1, "float32": 1}
    assert manifest["converter_version"]


def test_conversion_manifest_records_requested_reduced_precision(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "sam3.pt"
    source_checkpoint.write_bytes(b"official-checkpoint")
    weights = {
        "head.weight": mx.ones((2, 3), dtype=mx.float16),
        "rope.freqs": mx.ones((2,), dtype=mx.complex64),
    }
    output_checkpoint = save_weights(tmp_path, weights)

    manifest_path = write_conversion_manifest(
        tmp_path,
        source_repo="facebook/sam3",
        source_revision="a" * 40,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        weights=weights,
        ignored_keys=(),
        dtype_policy="float16",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dtype_policy"] == "float16"
    assert manifest["tensor_count"] == 2
    assert manifest["dtype_counts"] == {"complex64": 1, "float16": 1}


def test_save_weights_fsyncs_and_atomically_replaces_existing_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "weights with spaces Ω"
    save_weights(destination, {"head.weight": mx.zeros((2, 3))})
    calls: list[int] = []
    original_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(convert_module.os, "fsync", record_fsync)

    output = save_weights(destination, {"head.weight": mx.ones((2, 3))})

    loaded = mx.load(str(output))
    assert isinstance(loaded, dict)
    np.testing.assert_array_equal(to_numpy(loaded["head.weight"]), np.ones((2, 3)))
    assert len(calls) >= 3
    assert not list(destination.glob(".model.safetensors.*tmp-*"))
    assert not list(destination.glob("*.backup"))


def test_save_weights_serialization_failure_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "existing"
    model_path = save_weights(destination, {"head.weight": mx.full((2, 2), 7.0)})
    index_path = destination / "model.safetensors.index.json"
    before_model = model_path.read_bytes()
    before_index = index_path.read_bytes()
    original = cast(
        Callable[[str, dict[str, mx.array]], object], getattr(mx, "save_safetensors")
    )

    def fail_after_serialization(path: str, arrays: dict[str, mx.array]) -> None:
        original(path, arrays)
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(mx, "save_safetensors", fail_after_serialization)

    with pytest.raises(RuntimeError, match="injected serialization failure"):
        save_weights(destination, {"head.weight": mx.zeros((2, 2))})

    assert model_path.read_bytes() == before_model
    assert index_path.read_bytes() == before_index
    assert not list(destination.glob(".model.safetensors.*tmp-*"))


def test_save_weights_publish_failure_rolls_back_complete_existing_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "rollback"
    model_path = save_weights(destination, {"head.weight": mx.full((2, 2), 5.0)})
    index_path = destination / "model.safetensors.index.json"
    before_model = model_path.read_bytes()
    before_index = index_path.read_bytes()
    original_replace = convert_module.os.replace

    def fail_model_publish(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        if Path(target) == model_path and ".tmp-" in Path(source).name:
            raise OSError("injected model publication failure")
        original_replace(source, target)

    monkeypatch.setattr(convert_module.os, "replace", fail_model_publish)

    with pytest.raises(OSError, match="injected model publication failure"):
        save_weights(destination, {"head.weight": mx.zeros((2, 2))})

    assert model_path.read_bytes() == before_model
    assert index_path.read_bytes() == before_index
    assert not list(destination.glob("*.backup"))


def test_save_weights_serializes_two_processes_without_interleaving(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "concurrent"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_concurrent_save_worker,
            args=(str(destination), value),
        )
        for value in (1.0, 2.0)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    assert [process.exitcode for process in processes] == [0, 0]
    loaded = mx.load(str(destination / "model.safetensors"))
    assert isinstance(loaded, dict)
    observed = to_numpy(loaded["head.weight"])
    assert np.all(observed == 1.0) or np.all(observed == 2.0)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"head.bias", "head.weight"}


def test_save_weights_crash_before_publication_is_cleaned_on_next_writer(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "crash recovery"
    model_path = save_weights(destination, {"head.weight": mx.full((2, 2), 3.0)})
    before = model_path.read_bytes()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_before_weight_publication,
        args=(str(destination),),
    )
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 17
    assert model_path.read_bytes() == before
    assert list(destination.glob(".model.safetensors.tmp-*"))

    save_weights(destination, {"head.weight": mx.full((2, 2), 9.0)})

    loaded = mx.load(str(model_path))
    assert isinstance(loaded, dict)
    np.testing.assert_array_equal(to_numpy(loaded["head.weight"]), np.full((2, 2), 9.0))
    assert not list(destination.glob(".model.safetensors.tmp-*"))


def test_cached_conversion_rejects_revision_drift(tmp_path: Path) -> None:
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"converted")
    _write_fixture_index(tmp_path)
    manifest_file = tmp_path / "conversion-manifest.json"
    manifest_file.write_text(
        json.dumps({"source_repo": "facebook/sam3", "source_revision": "a" * 40})
    )

    _validate_cached_conversion(
        weights_file,
        manifest_file,
        source_repo="facebook/sam3",
        source_revision="a" * 40,
    )

    with pytest.raises(ValueError, match="source_revision"):
        _validate_cached_conversion(
            weights_file,
            manifest_file,
            source_repo="facebook/sam3",
            source_revision="b" * 40,
        )


def test_cached_conversion_rejects_dtype_policy_drift(tmp_path: Path) -> None:
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"converted")
    _write_fixture_index(tmp_path)
    manifest_file = tmp_path / "conversion-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "source_repo": "facebook/sam3",
                "source_revision": "a" * 40,
                "dtype_policy": "float32",
            }
        )
    )

    _validate_cached_conversion(
        weights_file,
        manifest_file,
        source_repo="facebook/sam3",
        source_revision="a" * 40,
        dtype_policy="float32",
    )
    with pytest.raises(ValueError, match="dtype_policy"):
        _validate_cached_conversion(
            weights_file,
            manifest_file,
            source_repo="facebook/sam3",
            source_revision="a" * 40,
            dtype_policy="float16",
        )


def _stub_conversion_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sam3.pt").write_bytes(b"official")

    def download_fixture(*args: object, **kwargs: object) -> Path:
        return source

    monkeypatch.setattr(convert_module, "download", download_fixture)

    def convert_fixture(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, mx.array], tuple[str, ...]]:
        return {"head.weight": mx.ones((2, 3), dtype=mx.float16)}, ()

    monkeypatch.setattr(convert_module, "convert", convert_fixture)
    return source


def test_download_and_convert_publishes_validated_artifacts_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_conversion_inputs(tmp_path, monkeypatch)
    destination = tmp_path / "converted-fp16"

    weights = download_and_convert(
        mlx_path=destination,
        source_revision="a" * 40,
        dtype="float16",
    )

    index = destination / "model.safetensors.index.json"
    manifest_path = destination / "conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert weights == destination / "model.safetensors"
    assert manifest["dtype_policy"] == "float16"
    assert index.is_file()
    assert not list(tmp_path.glob(".converted-fp16.tmp-*"))

    monkeypatch.setattr(
        convert_module,
        "download",
        _unexpected_download,
    )
    assert (
        download_and_convert(
            mlx_path=destination,
            source_revision="a" * 40,
            dtype="float16",
        )
        == weights
    )


def test_conversion_directory_swap_rolls_back_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_conversion_inputs(tmp_path, monkeypatch)
    destination = tmp_path / "converted-swap"
    revision = "a" * 40
    weights = download_and_convert(
        mlx_path=destination,
        source_revision=revision,
        dtype="float16",
    )
    before = {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    }
    original_replace = convert_module.os.replace

    def fail_staged_swap(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name.startswith(".converted-swap.tmp-") and (
            target_path == destination
        ):
            raise OSError("injected directory swap failure")
        original_replace(source, target)

    monkeypatch.setattr(convert_module.os, "replace", fail_staged_swap)

    with pytest.raises(OSError, match="injected directory swap failure"):
        download_and_convert(
            mlx_path=destination,
            force=True,
            source_revision=revision,
            dtype="float16",
        )

    assert weights.exists()
    assert {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    } == before
    assert not (tmp_path / ".converted-swap.backup").exists()
    assert not list(tmp_path.glob(".converted-swap.tmp-*"))


def test_conversion_recovers_a_complete_backup_before_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_conversion_inputs(tmp_path, monkeypatch)
    destination = tmp_path / "converted-recovery"
    revision = "a" * 40
    weights = download_and_convert(
        mlx_path=destination,
        source_revision=revision,
        dtype="float16",
    )
    backup = tmp_path / ".converted-recovery.backup"
    convert_module.os.replace(destination, backup)
    monkeypatch.setattr(
        convert_module,
        "download",
        _unexpected_download,
    )

    recovered = download_and_convert(
        mlx_path=destination,
        source_revision=revision,
        dtype="float16",
    )

    assert recovered == weights
    assert recovered.exists()
    assert not backup.exists()


@pytest.mark.parametrize("fault_after", ("weights", "index", "manifest"))
def test_conversion_faults_never_replace_an_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_after: str,
) -> None:
    _stub_conversion_inputs(tmp_path, monkeypatch)
    destination = tmp_path / f"converted-{fault_after}"
    destination.mkdir()
    sentinels = {
        "model.safetensors": b"old-weights",
        "model.safetensors.index.json": b"old-index",
        "conversion-manifest.json": b"old-manifest",
    }
    for name, value in sentinels.items():
        (destination / name).write_bytes(value)

    if fault_after == "weights":
        original_save = cast(
            Callable[[str, dict[str, mx.array]], object],
            getattr(mx, "save_safetensors"),
        )

        def fail_after_weights(path: str, arrays: dict[str, mx.array]) -> None:
            original_save(path, arrays)
            raise RuntimeError("fault after weights")

        monkeypatch.setattr(mx, "save_safetensors", fail_after_weights)
    elif fault_after == "index":
        original_index = convert_module.write_weight_index

        def fail_after_index(path: Path, index: convert_module.WeightIndex) -> None:
            original_index(path, index)
            raise RuntimeError("fault after index")

        monkeypatch.setattr(convert_module, "write_weight_index", fail_after_index)
    else:
        original_manifest = convert_module.write_conversion_manifest

        def fail_after_manifest(
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
            original_manifest(
                output_dir,
                source_repo=source_repo,
                source_revision=source_revision,
                source_checkpoint=source_checkpoint,
                output_checkpoint=output_checkpoint,
                weights=weights,
                ignored_keys=ignored_keys,
                dtype_policy=dtype_policy,
            )
            raise RuntimeError("fault after manifest")

        monkeypatch.setattr(
            convert_module,
            "write_conversion_manifest",
            fail_after_manifest,
        )

    with pytest.raises(RuntimeError, match=f"fault after {fault_after}"):
        download_and_convert(
            mlx_path=destination,
            force=True,
            source_revision="a" * 40,
            dtype="float16",
        )

    assert {name: (destination / name).read_bytes() for name in sentinels} == sentinels
    assert not list(tmp_path.glob(f".{destination.name}.tmp-*"))


@pytest.mark.parametrize("revision", ["main", "a" * 39, "g" * 40])
def test_conversion_requires_full_immutable_commit_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="40-character hexadecimal commit SHA"):
        validate_source_revision(revision)


def test_default_mlx_checkpoint_pin_is_immutable() -> None:
    assert DEFAULT_MLX_CHECKPOINT.repo == "mlx-community/sam3-image"
    assert validate_source_revision(DEFAULT_MLX_CHECKPOINT.revision) == (
        DEFAULT_MLX_CHECKPOINT.revision
    )
    assert DEFAULT_MLX_CHECKPOINT.architecture == "sam3-image"


def test_validate_hub_checkpoint_provenance_rejects_manifest_drift(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"pinned-weights")
    revision = "a" * 40

    report = validate_hub_checkpoint_provenance(
        tmp_path,
        expected_repo="mlx-community/sam3-image",
        expected_revision=revision,
    )
    assert report["status"] == "package-pinned"

    (tmp_path / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "architecture": "not-sam3-image",
                "source_repo": "facebook/sam3",
                "source_revision": "b" * 40,
            }
        )
    )
    with pytest.raises(ValueError, match="architecture"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="mlx-community/sam3-image",
            expected_revision=revision,
        )

    (tmp_path / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "architecture": "sam3-image",
                "source_repo": "facebook/sam3",
                "source_revision": "main",
            }
        )
    )
    with pytest.raises(ValueError, match="source_revision"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="mlx-community/sam3-image",
            expected_revision=revision,
        )


def test_precision_specific_hub_checkpoint_requires_exact_dtype_manifest(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"pinned-fp16")
    expected = {
        "expected_repo": "example/sam3-image-fp16",
        "expected_revision": "a" * 40,
        "expected_dtype_policy": "float16",
    }

    with pytest.raises(ValueError, match="require a conversion-manifest"):
        validate_hub_checkpoint_provenance(tmp_path, **expected)

    manifest = {
        "architecture": "sam3-image",
        "source_repo": "facebook/sam3",
        "source_revision": "b" * 40,
        "dtype_policy": "float16",
        "dtype_counts": {"float32": 1, "float16": 1399},
    }
    _write_fixture_index(tmp_path)
    manifest_path = tmp_path / "conversion-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="floating dtype inventory"):
        validate_hub_checkpoint_provenance(tmp_path, **expected)

    manifest["dtype_counts"] = {"complex64": 32, "float16": 1368}
    manifest_path.write_text(json.dumps(manifest))
    provenance = validate_hub_checkpoint_provenance(tmp_path, **expected)

    assert provenance["status"] == "manifest-verified"
    assert "manifest" in provenance
    assert provenance["manifest"] == manifest


def test_default_hf_checkpoint_is_pinned_and_manifest_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"default-pinned")
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        "sam3_mlx.convert.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        "sam3_mlx.convert.DEFAULT_MLX_CHECKPOINT",
        type(DEFAULT_MLX_CHECKPOINT)(
            repo="mlx-community/sam3-image",
            revision="c" * 40,
            architecture="sam3-image",
        ),
    )
    # DEFAULT_MLX_CHECKPOINT is imported by name in load_from_hub default args;
    # patch the module attribute used for compare inside the function.
    import sam3_mlx.convert as convert_mod

    convert_mod.DEFAULT_MLX_CHECKPOINT = type(DEFAULT_MLX_CHECKPOINT)(
        repo="mlx-community/sam3-image",
        revision="c" * 40,
        architecture="sam3-image",
    )

    path = load_from_hub(
        hf_repo="mlx-community/sam3-image",
        revision="c" * 40,
    )
    assert path == weights
    assert captured["revision"] == "c" * 40
    captured_revision = captured["revision"]
    assert isinstance(captured_revision, str)
    assert len(captured_revision) == 40

    with pytest.raises(ValueError, match="requires an immutable"):
        load_from_hub(hf_repo="mlx-community/sam3-image")

    # Bad revision format is rejected before download.
    with pytest.raises(ValueError, match="40-character"):
        load_from_hub(
            hf_repo="mlx-community/sam3-image",
            revision="main",
        )
