from __future__ import annotations

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import huggingface_hub
from hypothesis import given, strategies as st
import mlx.core as mx
from mlx import nn
import numpy as np
import pytest

from sam3_mlx.checkpoint_loading import CheckpointLoadable

import sam3_mlx.model_builder as model_builder
import sam3_mlx.sam31_checkpoint as sam31_checkpoint
from sam3_mlx.checkpoint_mapping import flatten_model_parameters
from sam3_mlx.mlx_runtime import to_numpy


class _TinyMultiplexModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = nn.Module()
        self.detector.backbone = nn.Module()
        self.detector.backbone.language_backbone = nn.Module()
        self.detector.backbone.language_backbone.resizer = nn.Linear(1, 1)
        self.tracker = nn.Module()
        self.tracker.model = nn.Module()
        self.tracker.model.no_mem_embed = mx.zeros((1, 1, 1))


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    torch = importlib.import_module("torch")
    source = tmp_path / "sam3.1_multiplex.pt"
    torch.save(
        {
            "model": {
                "detector_model.text_projection.weight": torch.tensor([[3.0]]),
                "detector_model.text_projection.bias": torch.tensor([2.0]),
                "tracker_model.no_memory_embedding": torch.tensor([[[7.0]]]),
                "tracker_model.memory_attention.freqs_cis": torch.tensor([1j]),
            }
        },
        source,
    )
    monkeypatch.setattr(sam31_checkpoint, "DEFAULT_SAM31_SOURCE", source)

    def tiny_model(**_kwargs: object) -> _TinyMultiplexModel:
        return _TinyMultiplexModel()

    monkeypatch.setattr(
        model_builder, "build_checkpoint_free_multiplex_predictor_model", tiny_model
    )
    return source


def _assert_loaded(predictor: object) -> None:
    model = getattr(predictor, "model")
    assert isinstance(model, _TinyMultiplexModel)
    parameters = flatten_model_parameters(model)
    np.testing.assert_array_equal(
        to_numpy(parameters["detector.backbone.language_backbone.resizer.weight"]),
        [[3.0]],
    )
    np.testing.assert_array_equal(
        to_numpy(parameters["detector.backbone.language_backbone.resizer.bias"]), [2.0]
    )
    np.testing.assert_array_equal(
        to_numpy(parameters["tracker.model.no_mem_embed"]), [[[7.0]]]
    )


def test_public_sam31_builder_converts_local_source_and_reuses_verified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    directory = tmp_path / "converted"

    def forbidden_download(**_kwargs: object) -> str:
        pytest.fail("Pinned local source should avoid downloading.")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden_download)
    _assert_loaded(
        model_builder.build_sam3_predictor(
            version="sam3.1", local_weights_dir=directory
        )
    )
    bundle = sam31_checkpoint.parse_sam31_bundle(source, directory)
    assert (bundle.source_tensor_count, bundle.output_tensor_count) == (4, 3)
    original_files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path, _ in bundle.file_signatures
    }
    torch = importlib.import_module("torch")

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        pytest.fail("Verified cache reuse should not deserialize the Torch source.")

    monkeypatch.setattr(torch, "load", forbidden_load)
    _assert_loaded(
        model_builder.build_sam3_predictor(
            version="sam3.1", local_weights_dir=directory
        )
    )
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in original_files
    } == original_files


def test_public_sam31_builder_downloads_only_the_canonical_pin_when_source_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sam31_checkpoint, "DEFAULT_SAM31_SOURCE", tmp_path / "missing.pt"
    )
    calls: list[tuple[str, str, str]] = []

    def download(*, repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return str(
            source
            if filename == sam31_checkpoint.SAM31_CHECKPOINT_FILENAME
            else tmp_path / "config.json"
        )

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    _assert_loaded(
        model_builder.build_sam3_predictor(
            version="sam3.1", local_weights_dir=tmp_path / "converted"
        )
    )
    assert calls == [
        (
            sam31_checkpoint.SAM31_CHECKPOINT_REPO,
            filename,
            sam31_checkpoint.SAM31_CHECKPOINT_REVISION,
        )
        for filename in ("config.json", sam31_checkpoint.SAM31_CHECKPOINT_FILENAME)
    ]


def test_corrupt_sam31_cache_fails_without_overwriting_or_building_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    directory = tmp_path / "converted"
    bundle = sam31_checkpoint.prepare_sam31_checkpoint(
        source, directory, command=["test"]
    )
    bundle.weights.unlink()

    def forbidden_model(**_kwargs: object) -> object:
        pytest.fail("Invalid cache must fail before model construction.")

    monkeypatch.setattr(
        model_builder,
        "build_checkpoint_free_multiplex_predictor_model",
        forbidden_model,
    )
    with pytest.raises(FileNotFoundError, match="conversion is missing"):
        model_builder.build_sam3_predictor(
            version="sam3.1", local_weights_dir=directory
        )
    assert not bundle.weights.exists()


@pytest.mark.parametrize("write_stage", ["weights", "manifest"])
def test_failed_bundle_write_does_not_publish_cache_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_stage: str
) -> None:
    source = _source(tmp_path, monkeypatch)
    directory = tmp_path / "converted"
    save_weights = sam31_checkpoint.save_weights
    write_text = Path.write_text

    def failed_weights(
        save_path: str | Path,
        weights: dict[str, mx.array],
        *,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        save_weights(save_path, weights, metadata=metadata)
        assert not directory.exists()
        raise OSError("Injected bundle write failure.")

    def failed_manifest(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path.name == "conversion-manifest.json":
            assert not directory.exists()
            raise OSError("Injected bundle write failure.")
        return write_text(path, data, encoding, errors, newline)

    with pytest.MonkeyPatch.context() as patch:
        if write_stage == "weights":
            patch.setattr(sam31_checkpoint, "save_weights", failed_weights)
        else:
            patch.setattr(Path, "write_text", failed_manifest)
        with pytest.raises(OSError, match="Injected bundle write failure"):
            sam31_checkpoint.prepare_sam31_checkpoint(
                source, directory, command=["test"]
            )
    assert not directory.exists()
    assert not tuple(tmp_path.glob(f".{directory.name}.tmp-*"))
    bundle = sam31_checkpoint.prepare_sam31_checkpoint(
        source, directory, command=["retry"]
    )
    assert bundle.source == source
    assert {bundle.weights.parent, bundle.index.parent, bundle.manifest.parent} == {
        directory
    }
    assert {path for path, _ in bundle.file_signatures} == {
        source, bundle.weights, bundle.index, bundle.manifest
    }
    sam31_checkpoint.require_stable_sam31_bundle(bundle)
    _assert_loaded(
        model_builder.build_sam3_predictor(
            version="sam3.1", local_weights_dir=directory
        )
    )


def test_sam31_explicit_checkpoint_takes_precedence_over_automatic_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    bundle = sam31_checkpoint.prepare_sam31_checkpoint(
        source, tmp_path / "converted", command=["test"]
    )
    source.unlink()

    def forbidden_download(**_kwargs: object) -> str:
        pytest.fail("Explicit checkpoint should avoid automatic acquisition.")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden_download)
    _assert_loaded(
        model_builder.build_sam3_predictor(
            version="sam3.1", checkpoint_path=bundle.weights
        )
    )



def test_automatic_sam31_builder_rechecks_bundle_after_strict_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    directory = tmp_path / "converted"
    bundle = sam31_checkpoint.prepare_sam31_checkpoint(
        source, directory, command=["test"]
    )
    strict_loader = model_builder.load_multiplex_checkpoint

    def load_and_change(
        model: CheckpointLoadable,
        checkpoint_path: model_builder.PathLikeStr,
        *,
        strict: bool = True,
    ) -> object:
        report = strict_loader(model, checkpoint_path, strict=strict)
        bundle.manifest.write_text("changed-after-strict-load")
        return report

    monkeypatch.setattr(model_builder, "load_multiplex_checkpoint", load_and_change)
    with pytest.raises(RuntimeError, match="input changed during execution"):
        model_builder.build_sam3_predictor(
            version="sam3.1", local_weights_dir=directory
        )


def test_missing_torch_has_an_actionable_error_before_bundle_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    import_module = sam31_checkpoint.importlib.import_module

    def without_torch(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return import_module(name, package)

    monkeypatch.setattr(sam31_checkpoint.importlib, "import_module", without_torch)
    directory = tmp_path / "converted"
    with pytest.raises(RuntimeError, match="uv sync --group oracle"):
        sam31_checkpoint.prepare_sam31_checkpoint(source, directory, command=["test"])
    assert not directory.exists()


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, width=32), min_size=1, max_size=32
    )
)
def test_sam31_package_conversion_round_trips_actual_torch_tensors(
    values: list[float],
) -> None:
    torch = importlib.import_module("torch")
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.pt"
        torch.save(
            {"tracker.weight": torch.tensor(values, dtype=torch.float32)}, source
        )
        with pytest.MonkeyPatch.context():
            bundle = sam31_checkpoint.prepare_sam31_checkpoint(
                source, root / "bundle", command=["hypothesis-round-trip"]
            )
            loaded = mx.load(bundle.weights, return_metadata=True)
            assert isinstance(loaded, tuple)
            arrays, metadata = loaded
            assert metadata == {"format": "mlx"}
            np.testing.assert_array_equal(
                to_numpy(arrays["tracker.weight"]), np.asarray(values, dtype=np.float32)
            )
            assert bundle.total_bytes == len(values) * 4
            assert bundle.source_tensor_count == bundle.output_tensor_count == 1
            sam31_checkpoint.require_stable_sam31_bundle(bundle)


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=12,
)


@given(_JSON)
def test_sam31_manifest_parser_rejects_arbitrary_json_without_crashing(
    raw: object,
) -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "conversion-manifest.json").write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            sam31_checkpoint.parse_sam31_bundle(root / "source.pt", root)
