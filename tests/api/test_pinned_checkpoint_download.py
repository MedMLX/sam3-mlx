"""The builder's weight sources: the shared MedMLX cache and an optional interactive head file."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

import sam3_mlx.model_builder as model_builder
from sam3_mlx.checkpoint_mapping import flatten_model_parameters
from sam3_mlx.mlx_runtime import to_numpy
from tests._mlx_runtime import save_safetensors


class _StopAfterLoad(Exception):
    pass


def test_default_checkpoint_is_the_medmlx_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without checkpoint_path the builder loads model.safetensors from weights_dir('sam3')."""
    requested: list[str] = []
    loaded: list[object] = []

    def cached(model_id: str) -> Path:
        requested.append(model_id)
        return tmp_path

    def load(_model: object, checkpoint_path: object, **_kwargs: object) -> None:
        loaded.append(checkpoint_path)
        raise _StopAfterLoad

    monkeypatch.setattr(model_builder, "weights_dir", cached)
    monkeypatch.setattr(model_builder, "load_image_checkpoint", load)

    with pytest.raises(_StopAfterLoad):
        model_builder.build_sam3_image_model()

    assert requested == ["sam3"]
    assert loaded == [tmp_path / "model.safetensors"]


def test_interactive_checkpoint_combines_with_the_cached_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interactive head file is loaded together with the cached base weights."""
    loaded: list[tuple[object, object]] = []

    def load(_model: object, checkpoint_path: object, **kwargs: object) -> None:
        loaded.append((checkpoint_path, kwargs["interactive_checkpoint_path"]))
        raise _StopAfterLoad

    def cached(_model_id: str) -> Path:
        return tmp_path

    monkeypatch.setattr(model_builder, "weights_dir", cached)
    monkeypatch.setattr(model_builder, "load_image_checkpoint", load)

    with pytest.raises(_StopAfterLoad):
        model_builder.build_sam3_image_model(
            enable_inst_interactivity=True,
            interactive_checkpoint_path="interactive.safetensors",
        )

    assert loaded == [(tmp_path / "model.safetensors", "interactive.safetensors")]


def test_interactive_head_defaults_to_its_medmlx_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabling interactivity without a head file downloads MedMLX/sam3-interactive."""
    requested: list[str] = []
    loaded: list[tuple[object, object]] = []

    def cached(model_id: str) -> Path:
        requested.append(model_id)
        return tmp_path / model_id

    def load(_model: object, checkpoint_path: object, **kwargs: object) -> None:
        loaded.append((checkpoint_path, kwargs["interactive_checkpoint_path"]))
        raise _StopAfterLoad

    monkeypatch.setattr(model_builder, "weights_dir", cached)
    monkeypatch.setattr(model_builder, "load_image_checkpoint", load)

    with pytest.raises(_StopAfterLoad):
        model_builder.build_sam3_image_model(enable_inst_interactivity=True)

    assert requested == ["sam3", "sam3-interactive"]
    assert loaded == [(
        tmp_path / "sam3" / "model.safetensors",
        tmp_path / "sam3-interactive" / "model.safetensors",
    )]


def test_interactive_checkpoint_without_interactivity_is_rejected_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("The builder must reject the request before building or downloading.")

    monkeypatch.setattr(model_builder, "weights_dir", unexpected)
    monkeypatch.setattr(model_builder, "_assemble_sam3_image_components", unexpected)

    with pytest.raises(ValueError, match="enable_inst_interactivity=True"):
        model_builder.build_sam3_image_model(
            interactive_checkpoint_path="interactive.safetensors"
        )


class _TinyInteractiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(1, 1)
        self.inst_interactive_predictor = {"model": {"no_mem_embed": mx.zeros((1, 1, 2))}}


def test_interactive_checkpoint_loads_without_a_base_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A randomly initialized base still takes the interactive head file, with a warning."""
    head = tmp_path / "head.safetensors"
    key = "inst_interactive_predictor.model.no_mem_embed"
    save_safetensors(head, {key: mx.array([[[3.0, 4.0]]])})

    def assemble(**_kwargs: object) -> _TinyInteractiveModel:
        return _TinyInteractiveModel()

    monkeypatch.setattr(model_builder, "_assemble_sam3_image_components", assemble)

    with pytest.warns(UserWarning, match="randomly initialized image model"):
        model = model_builder.build_sam3_image_model(
            load_from_HF=False,
            enable_inst_interactivity=True,
            interactive_checkpoint_path=head,
        )

    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)[key]), [[[3.0, 4.0]]]
    )
