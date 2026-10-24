from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from sam3_mlx.convert import (
    json_value,
    torch_weights,
    convert,
    validate_hub_checkpoint_provenance,
    validate_source_revision,
)


def test_provenance_rejects_non_object_manifest(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    (tmp_path / "conversion-manifest.json").write_text("[]")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="example/repo",
            expected_revision="a" * 40,
        )


@pytest.mark.parametrize("payload", [None, [], (), "weights", 1, True])
def test_checkpoint_payload_must_be_a_mapping(payload: object) -> None:
    with pytest.raises(ValueError, match="checkpoint payload must be a mapping"):
        torch_weights(payload)


def test_checkpoint_weights_require_string_keys_and_numpy_method() -> None:
    with pytest.raises(TypeError, match="weight keys must be strings"):
        torch_weights({1: object()})
    with pytest.raises(TypeError, match="weight 'detector.weight' must expose numpy"):
        torch_weights({"detector.weight": object()})


def test_json_value_rejects_non_string_object_keys() -> None:
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        json_value({1: "value"})


@pytest.mark.parametrize("revision", [None, 1, True, object()])
def test_source_revision_rejects_non_string_values(revision: object) -> None:
    with pytest.raises(TypeError, match="source_revision must be a string"):
        validate_source_revision(revision)


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("expected_repo", 1, TypeError, "expected_repo must be a non-empty string"),
        (
            "expected_revision",
            True,
            TypeError,
            "source_revision must be a string",
        ),
        (
            "expected_architecture",
            "",
            TypeError,
            "expected_architecture must be a non-empty string",
        ),
    ],
)
def test_provenance_rejects_malformed_expected_fields(
    tmp_path: Path,
    field: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    arguments: dict[str, object] = {
        "expected_repo": "example/repo",
        "expected_revision": "a" * 40,
        "expected_architecture": "sam3-image",
    }
    arguments[field] = value

    with pytest.raises(exception, match=message):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo=cast(str, arguments["expected_repo"]),
            expected_revision=cast(str, arguments["expected_revision"]),
            expected_architecture=cast(str, arguments["expected_architecture"]),
        )


@pytest.mark.parametrize(
    ("manifest_field", "manifest_value", "exception"),
    [
        ("source_repo", 1, TypeError),
        ("source_revision", False, TypeError),
        ("architecture", ["sam3-image"], ValueError),
    ],
)
def test_provenance_rejects_malformed_manifest_fields(
    tmp_path: Path,
    manifest_field: str,
    manifest_value: object,
    exception: type[Exception],
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    manifest: dict[str, object] = {
        "source_repo": "facebook/sam3",
        "source_revision": "b" * 40,
        "architecture": "sam3-image",
    }
    manifest[manifest_field] = manifest_value
    (tmp_path / "conversion-manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(exception, match=manifest_field):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="example/repo",
            expected_revision="a" * 40,
        )


def test_convert_keeps_safe_cpu_torch_load_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, object] = {}

    class _FakeTorch:
        def load(self, path: str, *, map_location: str, weights_only: bool) -> object:
            captured.update(
                path=path,
                map_location=map_location,
                weights_only=weights_only,
            )
            return []

    def fake_import_module(name: str) -> object:
        assert name == "torch"
        return _FakeTorch()

    monkeypatch.setattr("sam3_mlx.convert.import_module", fake_import_module)

    with pytest.raises(ValueError, match="checkpoint payload must be a mapping"):
        convert(tmp_path)
    assert captured == {
        "path": str(tmp_path / "sam3.pt"),
        "map_location": "cpu",
        "weights_only": True,
    }


def test_save_weights_leaves_detectable_state_when_index_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlx.core as mx
    import os

    from sam3_mlx.convert import save_weights

    save_weights(tmp_path, {"a": mx.zeros((2, 2))})
    index_path = tmp_path / "model.safetensors.index.json"
    assert index_path.is_file()

    real_replace = os.replace

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if str(dst) == str(index_path):
            raise OSError("simulated crash between model and index publish")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated crash"):
        save_weights(tmp_path, {"b": mx.zeros((3,))})

    assert not index_path.exists()
    monkeypatch.undo()
    from sam3_mlx.convert import write_weight_index

    write_weight_index(index_path, {"metadata": {}, "weight_map": {}})
