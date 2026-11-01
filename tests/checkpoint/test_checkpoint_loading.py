"""Checkpoint coverage and atomic loading with synthetic, tiny MLX models."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import mlx.core as mx
from mlx import nn
from numpy.typing import NDArray

from sam3_mlx.checkpoint_loading import (
    Sam3CheckpointLoadReport,
    audit_sam3_checkpoint_load,
    is_generated_checkpoint_key,
    load_image_checkpoint,
    read_interactive_checkpoint,
)
from sam3_mlx.checkpoint_mapping import flatten_model_parameters
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import UniversalSegmentationHead
from sam3_mlx.model.model_misc import DotProductScoring, TransformerWrapper
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.mlx_runtime import to_numpy
from tests._mlx_runtime import save_safetensors


class WeightLeaf(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = mx.zeros(shape)


class _TinyCheckpointModel(nn.Module):
    checkpoint_load_report: Sam3CheckpointLoadReport | None

    def __init__(self) -> None:
        super().__init__()
        self.inst_interactive_predictor = None
        self.head = nn.Linear(3, 2)
        self.scale = mx.ones((1,))
        self.checkpoint_load_report = None


def _tiny_image_checkpoint_model() -> Sam3Image:
    return Sam3Image(
        backbone=cast(SAM3VLBackbone, WeightLeaf((2,))),
        transformer=cast(TransformerWrapper, nn.Module()),
        input_geometry_encoder=cast(SequenceGeometryEncoder, nn.Module()),
        segmentation_head=cast(UniversalSegmentationHead, WeightLeaf((1,))),
        dot_prod_scoring=cast(DotProductScoring, nn.Linear(2, 1)),
    )


@pytest.fixture
def image_checkpoint_values() -> dict[str, NDArray[np.float32]]:
    """Supply distinct values for every tiny image-model tensor."""
    return {
        "backbone.weight": np.array([2.0, 3.0], dtype=np.float32),
        "dot_prod_scoring.weight": np.array([[4.0, 5.0]], dtype=np.float32),
        "dot_prod_scoring.bias": np.array([6.0], dtype=np.float32),
        "segmentation_head.weight": np.array([7.0], dtype=np.float32),
    }


class _TinyConvLayoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inst_interactive_predictor = None
        self.up = nn.ConvTranspose2d(4, 3, 2)


class _TinyInteractivePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = {"no_mem_embed": mx.zeros((1, 1, 2))}


class _TinyInteractiveCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = WeightLeaf((1,))
        self.inst_interactive_predictor = _TinyInteractivePredictor()


def test_checkpoint_audit_reports_loaded_missing_extra_and_shape_mismatch() -> None:
    """Keep shape mismatches distinct from missing and unrecognized weights."""
    model = _TinyCheckpointModel()
    weights = {
        "head.weight": mx.ones((2, 3)),
        "head.bias": mx.ones((3,)),
        "extra.weight": mx.ones((1,)),
    }

    report = audit_sam3_checkpoint_load(model, weights)

    assert report.loaded == ("head.weight",)
    assert report.missing == ("scale",)
    assert report.extra == ("extra.weight",)
    assert len(report.shape_mismatched) == 1
    mismatch = report.shape_mismatched[0]
    assert mismatch.key == "head.bias"
    assert mismatch.model_shape == (2,)
    assert mismatch.checkpoint_shape == (3,)


def test_generated_checkpoint_allowlist_is_exact() -> None:
    """Admit known generated language masks without admitting arbitrary attention masks."""
    assert is_generated_checkpoint_key("backbone.language_backbone.encoder.attn_mask")
    assert is_generated_checkpoint_key("backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis")
    assert not is_generated_checkpoint_key("transformer.attn_mask")


@pytest.mark.parametrize(
    "extra",
    [
        {},  # partial checkpoint: required keys missing
        {"head.bias": mx.ones((3,))},  # shape mismatch
        {"head.bias": mx.ones((2,)), "scale": mx.ones((1,)), "unexpected.weight": mx.ones((1,))},
    ],
    ids=("missing", "shape-mismatch", "unreviewed-extra"),
)
def test_strict_load_rejects_bad_checkpoint_without_mutating_model(
    tmp_path: Path, extra: dict[str, mx.array]
) -> None:
    """Strict coverage and shape checks run before any valid tensor is applied."""
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "bad.safetensors"
    original = to_numpy(flatten_model_parameters(model)["head.weight"]).copy()
    replacement = mx.reshape(mx.arange(6), (2, 3)).astype(mx.float32)
    save_safetensors(checkpoint_path, {"head.weight": replacement, **extra})

    with pytest.raises(ValueError):
        load_image_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["head.weight"]), original
    )


def test_load_checkpoint_allows_explicit_development_partial_load(
    tmp_path: Path,
) -> None:
    """An explicit non-strict load applies available tensors and reports omissions."""
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "partial-development.safetensors"
    replacement = mx.reshape(mx.arange(6), (2, 3)).astype(mx.float32)
    save_safetensors(checkpoint_path, {"head.weight": replacement})

    report = load_image_checkpoint(model, checkpoint_path, strict=False)

    assert report.loaded == ("head.weight",)
    assert report.missing == ("head.bias", "scale")
    assert model.checkpoint_load_report is report
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["head.weight"]),
        to_numpy(replacement),
    )


def test_load_checkpoint_loads_every_image_model_tensor(
    tmp_path: Path,
    image_checkpoint_values: dict[str, NDArray[np.float32]],
) -> None:
    """A complete checkpoint loads the backbone, scorer and segmentation head."""
    model = _tiny_image_checkpoint_model()
    checkpoint_path = tmp_path / "image.safetensors"
    save_safetensors(
        checkpoint_path,
        {key: mx.array(value) for key, value in image_checkpoint_values.items()},
    )

    report = load_image_checkpoint(model, checkpoint_path)

    assert report.extra == ()
    assert report.missing == ()
    assert report.shape_mismatched == ()
    parameters = flatten_model_parameters(model)
    for key, expected in image_checkpoint_values.items():
        np.testing.assert_array_equal(to_numpy(parameters[key]), expected)


@pytest.mark.parametrize(
    ("extra_key", "missing_key"),
    [
        ("segmentation_head.unreviewed", None),
        (None, "backbone.weight"),
        (None, "segmentation_head.weight"),
    ],
)
def test_image_checkpoint_coverage_stays_strict(
    tmp_path: Path,
    image_checkpoint_values: dict[str, NDArray[np.float32]],
    extra_key: str | None,
    missing_key: str | None,
) -> None:
    """Missing (including mask-head) or unexpected weights are rejected without partial loads."""
    model = _tiny_image_checkpoint_model()
    original = {
        key: to_numpy(value).copy()
        for key, value in flatten_model_parameters(model).items()
    }
    if missing_key is not None:
        del image_checkpoint_values[missing_key]
    if extra_key is not None:
        image_checkpoint_values[extra_key] = np.array([8.0], dtype=np.float32)
    checkpoint_path = tmp_path / "invalid-image.safetensors"
    save_safetensors(
        checkpoint_path,
        {key: mx.array(value) for key, value in image_checkpoint_values.items()},
    )

    with pytest.raises(ValueError):
        load_image_checkpoint(model, checkpoint_path)

    parameters = flatten_model_parameters(model)
    for key, expected in original.items():
        np.testing.assert_array_equal(to_numpy(parameters[key]), expected)


def test_load_checkpoint_normalizes_known_conv_layout_before_audit(
    tmp_path: Path,
) -> None:
    """Convert a PyTorch-layout transposed-convolution kernel before checking shapes."""
    model = _TinyConvLayoutModel()
    checkpoint_path = tmp_path / "conv-layout.safetensors"
    key = "up.weight"
    torch_layout = mx.reshape(mx.arange(4 * 3 * 2 * 2), (4, 3, 2, 2)).astype(
        mx.float32
    )
    save_safetensors(checkpoint_path, {key: torch_layout})

    report = load_image_checkpoint(model, checkpoint_path, strict=False)

    assert report.loaded == (key,)
    assert report.shape_mismatched == ()
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)[key]),
        to_numpy(torch_layout).transpose(1, 2, 3, 0),
    )


_HEAD_KEY = "inst_interactive_predictor.model.no_mem_embed"


def test_load_checkpoint_merges_explicit_interactive_checkpoint(tmp_path: Path) -> None:
    """Load the base tensor and the flat interactive head from separate files."""
    model = _TinyInteractiveCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive.safetensors"
    save_safetensors(checkpoint_path, {"base.weight": mx.array([5.0])})
    save_safetensors(interactive_checkpoint_path, {_HEAD_KEY: mx.array([[[7.0, 8.0]]])})

    report = load_image_checkpoint(
        model,
        checkpoint_path,
        interactive_checkpoint_path=interactive_checkpoint_path,
    )

    assert report.loaded == ("base.weight", _HEAD_KEY)
    assert report.missing == report.extra == ()
    params = flatten_model_parameters(model)
    np.testing.assert_array_equal(to_numpy(params["base.weight"]), [5.0])
    np.testing.assert_array_equal(to_numpy(params[_HEAD_KEY]), [[[7.0, 8.0]]])


def test_strict_load_without_interactive_weights_names_interactive_checkpoint(
    tmp_path: Path,
) -> None:
    """A base-only checkpoint for an interactive model asks for the interactive file."""
    model = _TinyInteractiveCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    save_safetensors(checkpoint_path, {"base.weight": mx.array([5.0])})

    with pytest.raises(ValueError, match="interactive_checkpoint_path"):
        load_image_checkpoint(model, checkpoint_path)


def test_interactive_checkpoint_requires_an_interactive_model(tmp_path: Path) -> None:
    """Do not accept head weights when the image model has no interactive predictor."""
    interactive_checkpoint_path = tmp_path / "interactive.safetensors"
    save_safetensors(interactive_checkpoint_path, {_HEAD_KEY: mx.ones((1, 1, 2))})

    with pytest.raises(ValueError, match="enable_inst_interactivity=True"):
        read_interactive_checkpoint(_TinyCheckpointModel(), interactive_checkpoint_path)


@pytest.mark.parametrize(
    ("head_weights", "problem"),
    [
        ({_HEAD_KEY: mx.ones((1, 1, 3))}, "has shape"),
        (
            {_HEAD_KEY: mx.ones((1, 1, 2)), "inst_interactive_predictor.extra": mx.ones((1,))},
            "unexpected",
        ),
        ({"base.weight": mx.ones((1,))}, "missing"),
        # Meta's Transformers layout must be converted first.
        ({"tracker_model.no_memory_embedding": mx.ones((1, 1, 2))}, "missing"),
    ],
    ids=("shape-mismatch", "unexpected-key", "base-file", "unconverted-tracker"),
)
def test_wrong_interactive_checkpoint_is_rejected_without_mutating_model(
    tmp_path: Path, head_weights: dict[str, mx.array], problem: str
) -> None:
    """A wrong interactive file prevents both base and interactive updates."""
    model = _TinyInteractiveCheckpointModel()
    original = {
        key: to_numpy(value).copy() for key, value in flatten_model_parameters(model).items()
    }
    base = tmp_path / "base.safetensors"
    head = tmp_path / "head.safetensors"
    save_safetensors(base, {"base.weight": mx.array([5.0])})
    save_safetensors(head, head_weights)

    with pytest.raises(ValueError, match="not the SAM 3 interactive head") as error:
        load_image_checkpoint(model, base, interactive_checkpoint_path=head)
    assert problem in str(error.value)

    for key, value in flatten_model_parameters(model).items():
        np.testing.assert_array_equal(to_numpy(value), original[key])
