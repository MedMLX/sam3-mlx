from sam3_mlx.mlx_runtime import to_numpy
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx
from mlx import nn
from hypothesis import given, strategies as st

from sam3_mlx.checkpoint_loading import (
    Sam3CheckpointLoadReport,
    audit_sam3_checkpoint_load as audit_sam3_image_checkpoint_load,
    is_generated_checkpoint_key as _is_generated_checkpoint_key,
    load_image_checkpoint as _load_checkpoint,
    load_multiplex_checkpoint as load_multiplex_checkpoint,
    load_multiplex_tracker_checkpoint as _load_multiplex_tracker_checkpoint,
    load_tracker_checkpoint as _load_tracker_checkpoint,
)
from tests._mlx_runtime import save_safetensors
from sam3_mlx.checkpoint_mapping import (
    flatten_model_parameters,
    normalize_sam31_multiplex_tracker_weights,
    normalize_sam31_multiplex_weights,
    normalize_tracker_checkpoint_weights,
)
from tests.checkpoint._factories import WeightLeaf


class _TinyCheckpointModel(nn.Module):
    checkpoint_load_report: Sam3CheckpointLoadReport | None

    def __init__(self) -> None:
        super().__init__()
        self.inst_interactive_predictor = None
        self.head = nn.Linear(3, 2)
        self.scale = mx.ones((1,))
        self.checkpoint_load_report = None


class _Sam2ConvStage(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.dconv_2x2 = WeightLeaf(shape)


class _TinyConvLayoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inst_interactive_predictor = None
        self.backbone = {
            "vision_backbone": {
                "sam2_convs": [
                    _Sam2ConvStage((1, 1, 1, 1)),
                    _Sam2ConvStage((3, 2, 2, 4)),
                ]
            }
        }


class _TinyInteractivePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = {"no_mem_embed": mx.zeros((1, 1, 2))}


class _TinyInteractiveCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = WeightLeaf((1,))
        self.inst_interactive_predictor = _TinyInteractivePredictor()


class _TinyTrackerCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.no_mem_embed = mx.zeros((1, 1, 2))


class _TinyMultiplexCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = {
            "backbone": {
                "language_backbone": {
                    "encoder": {
                        "text_projection": mx.zeros((2, 1)),
                    },
                    "resizer": {
                        "weight": mx.zeros((2, 1)),
                        "bias": mx.zeros((2,)),
                    },
                }
            }
        }
        self.tracker = {
            "model": {
                "output_valid_embed": mx.zeros((2, 2)),
            }
        }


class _TinyMultiplexTrackerCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_valid_embed = mx.zeros((2, 2))


def test_checkpoint_audit_reports_loaded_missing_extra_and_shape_mismatch() -> None:
    model = _TinyCheckpointModel()
    weights = {
        "head.weight": mx.ones((2, 3)),
        "head.bias": mx.ones((3,)),
        "extra.weight": mx.ones((1,)),
    }

    report = audit_sam3_image_checkpoint_load(model, weights)

    assert report.loaded == ("head.weight",)
    assert report.missing == ("scale",)
    assert report.extra == ("extra.weight",)
    assert len(report.shape_mismatched) == 1
    mismatch = report.shape_mismatched[0]
    assert mismatch.key == "head.bias"
    assert mismatch.model_shape == (2,)
    assert mismatch.checkpoint_shape == (3,)


def test_generated_checkpoint_allowlist_is_exact() -> None:
    assert _is_generated_checkpoint_key("backbone.language_backbone.encoder.attn_mask")
    assert _is_generated_checkpoint_key(
        "detector.backbone.language_backbone.encoder.attn_mask"
    )
    assert not _is_generated_checkpoint_key("detector.attn_mask")


def test_load_checkpoint_rejects_partial_checkpoint_without_mutating_model(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "partial.safetensors"
    original = to_numpy(flatten_model_parameters(model)["head.weight"]).copy()
    replacement = mx.reshape(mx.arange(6), (2, 3)).astype(mx.float32)
    save_safetensors(
        checkpoint_path,
        {"head.weight": replacement},
    )

    with pytest.raises(ValueError, match="missing_required=2"):
        _load_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["head.weight"]),
        original,
    )


def test_load_checkpoint_allows_explicit_development_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "partial-development.safetensors"
    replacement = mx.reshape(mx.arange(6), (2, 3)).astype(mx.float32)
    save_safetensors(checkpoint_path, {"head.weight": replacement})

    report = _load_checkpoint(model, checkpoint_path, strict=False)

    assert report.loaded == ("head.weight",)
    assert report.missing == ("head.bias", "scale")
    assert model.checkpoint_load_report is report
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["head.weight"]),
        to_numpy(replacement),
    )


def test_load_checkpoint_rejects_unreviewed_source_weight(tmp_path: Path) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "unexpected.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "head.weight": mx.ones((2, 3)),
            "head.bias": mx.ones((2,)),
            "scale": mx.ones((1,)),
            "unexpected.weight": mx.ones((1,)),
        },
    )

    with pytest.raises(ValueError, match="without a reviewed model mapping"):
        _load_checkpoint(model, checkpoint_path)


def test_load_checkpoint_rejects_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "bad-shape.safetensors"
    original_weight = to_numpy(flatten_model_parameters(model)["head.weight"]).copy()
    replacement = mx.reshape(mx.arange(6), (2, 3)).astype(mx.float32)
    save_safetensors(
        checkpoint_path,
        {
            "head.weight": replacement,
            "head.bias": mx.ones((3,)),
        },
    )

    with pytest.raises(ValueError, match="shape-mismatched weights"):
        _load_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["head.weight"]),
        original_weight,
    )


def test_load_checkpoint_normalizes_known_conv_layout_before_audit(
    tmp_path: Path,
) -> None:
    model = _TinyConvLayoutModel()
    checkpoint_path = tmp_path / "conv-layout.safetensors"
    key = "backbone.vision_backbone.sam2_convs.1.dconv_2x2.weight"
    torch_layout = mx.reshape(mx.arange(4 * 3 * 2 * 2), (4, 3, 2, 2)).astype(
        mx.float32
    )
    save_safetensors(checkpoint_path, {key: torch_layout})

    report = _load_checkpoint(model, checkpoint_path, strict=False)

    assert report.loaded == (key,)
    assert report.shape_mismatched == ()
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)[key]),
        to_numpy(torch_layout).transpose(1, 2, 3, 0),
    )


def test_load_checkpoint_merges_explicit_interactive_checkpoint(tmp_path: Path) -> None:
    model = _TinyInteractiveCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive.safetensors"
    save_safetensors(
        checkpoint_path,
        {"base.weight": mx.array([5.0], dtype=mx.float32)},
    )
    save_safetensors(
        interactive_checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0]]],
                dtype=mx.float32,
            )
        },
    )

    report = _load_checkpoint(
        model,
        checkpoint_path,
        interactive_checkpoint_path=interactive_checkpoint_path,
    )

    assert report.loaded == (
        "base.weight",
        "inst_interactive_predictor.model.no_mem_embed",
    )
    params = flatten_model_parameters(model)
    np.testing.assert_array_equal(
        to_numpy(params["base.weight"]),
        np.array([5.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(params["inst_interactive_predictor.model.no_mem_embed"]),
        np.array([[[7.0, 8.0]]], dtype=np.float32),
    )


def test_load_checkpoint_rejects_interactive_checkpoint_without_interactivity(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive.safetensors"
    save_safetensors(
        checkpoint_path,
        {"head.weight": mx.ones((2, 3))},
    )
    save_safetensors(
        interactive_checkpoint_path,
        {"tracker_model.no_memory_embedding": mx.ones((1, 1, 2))},
    )

    with pytest.raises(
        ValueError,
        match="interactive_checkpoint_path requires",
    ):
        _load_checkpoint(
            model,
            checkpoint_path,
            interactive_checkpoint_path=interactive_checkpoint_path,
        )


def test_load_checkpoint_rejects_interactive_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyInteractiveCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive-bad-shape.safetensors"
    original_base = to_numpy(flatten_model_parameters(model)["base.weight"]).copy()
    original_interactive = to_numpy(
        flatten_model_parameters(model)["inst_interactive_predictor.model.no_mem_embed"]
    ).copy()
    save_safetensors(
        checkpoint_path,
        {"base.weight": mx.array([5.0], dtype=mx.float32)},
    )
    save_safetensors(
        interactive_checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0, 9.0]]],
                dtype=mx.float32,
            )
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "inst_interactive_predictor.model.no_mem_embed: "
            r"model \(1, 1, 2\), checkpoint \(1, 1, 3\)"
        ),
    ):
        _load_checkpoint(
            model,
            checkpoint_path,
            interactive_checkpoint_path=interactive_checkpoint_path,
        )

    params = flatten_model_parameters(model)
    np.testing.assert_array_equal(
        to_numpy(params["base.weight"]),
        original_base,
    )
    np.testing.assert_array_equal(
        to_numpy(params["inst_interactive_predictor.model.no_mem_embed"]),
        original_interactive,
    )


def test_load_tracker_checkpoint_maps_official_alias_and_loads(tmp_path: Path) -> None:
    model = _TinyTrackerCheckpointModel()
    checkpoint_path = tmp_path / "tracker.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0]]],
                dtype=mx.float32,
            )
        },
    )

    report = _load_tracker_checkpoint(model, checkpoint_path)

    assert report.loaded == ("no_mem_embed",)
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["no_mem_embed"]),
        np.array([[[7.0, 8.0]]], dtype=np.float32),
    )


def test_load_tracker_checkpoint_rejects_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyTrackerCheckpointModel()
    checkpoint_path = tmp_path / "tracker-bad.safetensors"
    original = to_numpy(flatten_model_parameters(model)["no_mem_embed"]).copy()
    save_safetensors(
        checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0, 9.0]]],
                dtype=mx.float32,
            )
        },
    )

    with pytest.raises(ValueError, match="shape-mismatched weights"):
        _load_tracker_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["no_mem_embed"]),
        original,
    )


@pytest.mark.parametrize("prefix", ["", "tracker.", "tracker_model.", "sam2_predictor."])
def test_tracker_checkpoint_rejects_unreviewed_source_before_mutation(
    tmp_path: Path, prefix: str
) -> None:
    model = _TinyTrackerCheckpointModel()
    original = to_numpy(model.no_mem_embed).copy()
    checkpoint = tmp_path / "unreviewed-tracker.safetensors"
    save_safetensors(checkpoint, {
        "tracker.no_mem_embed": mx.ones((1, 1, 2)),
        prefix + "unreviewed.weight": mx.ones((1,)),
    })
    with pytest.raises(ValueError, match="source weights without a reviewed model mapping"):
        _load_tracker_checkpoint(model, checkpoint)
    np.testing.assert_array_equal(to_numpy(model.no_mem_embed), original)


def test_tracker_checkpoint_rejects_colliding_source_aliases(tmp_path: Path) -> None:
    model = _TinyTrackerCheckpointModel()
    checkpoint = tmp_path / "colliding-tracker.safetensors"
    save_safetensors(checkpoint, {
        "tracker.no_mem_embed": mx.ones((1, 1, 2)),
        "tracker_model.no_memory_embedding": mx.zeros((1, 1, 2)),
    })
    with pytest.raises(ValueError, match="source weights collide"):
        _load_tracker_checkpoint(model, checkpoint)


@pytest.mark.parametrize("key", [
    "detector.weight", "detector_model.weight", "sam3_model.weight",
    "tracker_neck.fpn_layers.0.proj1.bias",
])
def test_tracker_checkpoint_ignores_source_outside_tracker_scope(
    tmp_path: Path, key: str,
) -> None:
    model = _TinyTrackerCheckpointModel()
    checkpoint = tmp_path / "full-source.safetensors"
    save_safetensors(checkpoint, {
        "tracker.no_mem_embed": mx.ones((1, 1, 2)),
        key: mx.ones((1,)),
    })
    report = _load_tracker_checkpoint(model, checkpoint)
    assert report.loaded == ("no_mem_embed",)
    assert report.extra == ()


@given(st.dictionaries(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12),
    st.integers(-100, 100),
    min_size=1,
    max_size=5,
))
def test_tracker_normalization_preserves_unknown_native_weights(
    unknown: dict[str, int],
) -> None:
    model = _TinyTrackerCheckpointModel()
    payload = {"no_mem_embed": mx.ones((1, 1, 2))}
    unknown_weights = {
        "unreviewed." + key: mx.array([value]) for key, value in unknown.items()
    }
    payload.update(unknown_weights)

    normalized = normalize_tracker_checkpoint_weights(payload, model)
    report = audit_sam3_image_checkpoint_load(model, normalized)

    assert report.loaded == ("no_mem_embed",)
    assert set(report.extra) == set(unknown_weights)
    assert all(normalized[key] is value for key, value in unknown_weights.items())


def test_load_multiplex_checkpoint_loads_detector_and_tracker_weights(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex.safetensors"
    text_projection = mx.array([[5.0, 6.0]], dtype=mx.float32)
    text_resizer = mx.array([[7.0], [8.0]], dtype=mx.float32)
    text_resizer_bias = mx.array([9.0, 10.0], dtype=mx.float32)
    output_valid = mx.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=mx.float32,
    )
    save_safetensors(
        checkpoint_path,
        {
            "detector_model.text_encoder.text_projection.weight": text_projection,
            "detector_model.text_projection.weight": text_resizer,
            "detector_model.text_projection.bias": text_resizer_bias,
            "tracker_model.output_valid_embed": output_valid,
        },
    )

    report = load_multiplex_checkpoint(model, checkpoint_path)

    assert report.loaded == (
        "detector.backbone.language_backbone.encoder.text_projection",
        "detector.backbone.language_backbone.resizer.bias",
        "detector.backbone.language_backbone.resizer.weight",
        "tracker.model.output_valid_embed",
    )
    params = flatten_model_parameters(model)
    np.testing.assert_array_equal(
        to_numpy(params["detector.backbone.language_backbone.encoder.text_projection"]),
        np.array([[5.0], [6.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(params["detector.backbone.language_backbone.resizer.weight"]),
        to_numpy(text_resizer),
    )
    np.testing.assert_array_equal(
        to_numpy(params["detector.backbone.language_backbone.resizer.bias"]),
        to_numpy(text_resizer_bias),
    )
    np.testing.assert_array_equal(
        to_numpy(params["tracker.model.output_valid_embed"]),
        to_numpy(output_valid),
    )


def test_load_multiplex_checkpoint_accepts_pinned_merged_namespace(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-merged.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "detector.backbone.language_backbone.encoder.text_projection": mx.array(
                [[5.0, 6.0]], dtype=mx.float32
            ),
            "detector.backbone.language_backbone.resizer.weight": mx.array(
                [[7.0], [8.0]], dtype=mx.float32
            ),
            "detector.backbone.language_backbone.resizer.bias": mx.array(
                [9.0, 10.0], dtype=mx.float32
            ),
            "tracker.model.output_valid_embed": mx.array(
                [[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32
            ),
        },
    )

    report = load_multiplex_checkpoint(model, checkpoint_path)

    assert report.missing == ()
    params = flatten_model_parameters(model)
    np.testing.assert_array_equal(
        to_numpy(params["detector.backbone.language_backbone.encoder.text_projection"]),
        np.array([[5.0], [6.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(params["tracker.model.output_valid_embed"]),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


class _TinyMergedProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = {
            "backbone": {
                "language_backbone": {
                    "resizer": {
                        "weight": mx.zeros((2, 1)),
                        "bias": mx.zeros((2,)),
                    }
                }
            },
            "attention": {
                "query_proj": nn.Linear(2, 2),
                "key_proj": nn.Linear(2, 2),
                "value_proj": nn.Linear(2, 2),
            },
        }
        self.tracker = {
            "model": {
                "output_valid_embed": mx.zeros((2, 2)),
                "projection": {"conv": nn.Conv2d(1, 2, 3)},
            }
        }


def test_load_merged_namespace_splits_in_projection_and_inserts_conv_wrapper(
    tmp_path: Path,
) -> None:
    model = _TinyMergedProjectionModel()
    checkpoint_path = tmp_path / "multiplex-merged-layout.safetensors"
    in_weight = mx.reshape(mx.arange(12, dtype=mx.float32), (6, 2))
    in_bias = mx.arange(6, dtype=mx.float32)
    conv_weight = mx.reshape(mx.arange(18, dtype=mx.float32), (2, 1, 3, 3))
    save_safetensors(
        checkpoint_path,
        {
            "detector.backbone.language_backbone.resizer.weight": mx.array(
                [[7.0], [8.0]], dtype=mx.float32
            ),
            "detector.backbone.language_backbone.resizer.bias": mx.array(
                [9.0, 10.0], dtype=mx.float32
            ),
            "detector.attention.in_proj_weight": in_weight,
            "detector.attention.in_proj_bias": in_bias,
            "tracker.model.output_valid_embed": mx.ones((2, 2)),
            "tracker.model.projection.weight": conv_weight,
            "tracker.model.projection.bias": mx.array([1.0, 2.0]),
        },
    )

    report = load_multiplex_checkpoint(model, checkpoint_path)

    assert report.missing == ()
    params = flatten_model_parameters(model)
    np.testing.assert_array_equal(
        to_numpy(params["detector.attention.query_proj.weight"]),
        to_numpy(in_weight[:2]),
    )
    np.testing.assert_array_equal(
        to_numpy(params["detector.attention.value_proj.bias"]),
        to_numpy(in_bias[4:]),
    )
    np.testing.assert_array_equal(
        to_numpy(params["tracker.model.projection.conv.weight"]),
        to_numpy(conv_weight).transpose(0, 2, 3, 1),
    )


def test_load_multiplex_checkpoint_requires_text_resizer_weights(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-missing-resizer.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "detector_model.text_encoder.text_projection.weight": mx.array(
                [[5.0, 6.0]],
                dtype=mx.float32,
            ),
            "tracker_model.output_valid_embed": mx.ones((2, 2), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match="required VE text resizer weights"):
        load_multiplex_checkpoint(model, checkpoint_path)


def test_load_multiplex_checkpoint_rejects_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-bad.safetensors"
    original_text = to_numpy(
        flatten_model_parameters(model)[
            "detector.backbone.language_backbone.encoder.text_projection"
        ]
    ).copy()
    save_safetensors(
        checkpoint_path,
        {
            "detector_model.text_encoder.text_projection.weight": mx.array(
                [[5.0, 6.0]],
                dtype=mx.float32,
            ),
            "tracker_model.output_valid_embed": mx.ones((3, 2), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match="shape-mismatched weights"):
        load_multiplex_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        to_numpy(
            flatten_model_parameters(model)[
                "detector.backbone.language_backbone.encoder.text_projection"
            ]
        ),
        original_text,
    )


def test_load_multiplex_checkpoint_rejects_pytorch_checkpoint_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="PyTorch SAM 3.1 multiplex"):
        load_multiplex_checkpoint(
            _TinyMultiplexCheckpointModel(),
            tmp_path / "sam3.1_multiplex.pt",
        )


def test_load_multiplex_tracker_checkpoint_loads_direct_tracker_model(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexTrackerCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-tracker.safetensors"
    output_valid = mx.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=mx.float32,
    )
    save_safetensors(
        checkpoint_path,
        {"tracker_model.output_valid_embed": output_valid},
    )

    report = _load_multiplex_tracker_checkpoint(model, checkpoint_path)

    assert report.loaded == ("output_valid_embed",)
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["output_valid_embed"]),
        to_numpy(output_valid),
    )


def test_load_multiplex_tracker_checkpoint_accepts_merged_tracker_namespace(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexTrackerCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-tracker-merged.safetensors"
    output_valid = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    save_safetensors(
        checkpoint_path,
        {"tracker.model.output_valid_embed": output_valid},
    )

    report = _load_multiplex_tracker_checkpoint(model, checkpoint_path)

    assert report.loaded == ("output_valid_embed",)
    np.testing.assert_array_equal(
        to_numpy(flatten_model_parameters(model)["output_valid_embed"]),
        to_numpy(output_valid),
    )


class _TinyTrackerWithBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.no_mem_embed = mx.zeros((1, 1, 2))
        self.backbone = {"vision_backbone": {"head": nn.Linear(2, 2)}}

    def __call__(self, values: mx.array) -> mx.array:
        return self.backbone["vision_backbone"]["head"](values) + self.no_mem_embed[0]


@pytest.mark.parametrize("separate_backbone", [False, True])
def test_tracker_with_backbone_loads_and_evaluates_all_native_weights(
    tmp_path: Path, separate_backbone: bool,
) -> None:
    model = _TinyTrackerWithBackbone()
    checkpoint = tmp_path / "tracker.safetensors"
    image_checkpoint = tmp_path / "image.safetensors"
    tracker_weights = {"tracker.no_mem_embed": mx.array([[[1.0, -1.0]]])}
    image_weights = {
        "detector.backbone.vision_backbone.head.weight": mx.array([[2.0, 3.0], [4.0, 5.0]]),
        "detector.backbone.vision_backbone.head.bias": mx.array([6.0, 7.0]),
    }
    if separate_backbone:
        save_safetensors(checkpoint, tracker_weights)
        save_safetensors(image_checkpoint, image_weights)
    else:
        save_safetensors(checkpoint, tracker_weights | image_weights)
    report = _load_tracker_checkpoint(
        model, checkpoint,
        backbone_checkpoint_path=image_checkpoint if separate_backbone else None,
    )
    assert report.missing == report.extra == report.shape_mismatched == ()
    np.testing.assert_array_equal(to_numpy(model(mx.array([[1.0, 2.0]]))), [[15.0, 20.0]])


def test_tracker_checkpoint_missing_backbone_fails_before_mutation(tmp_path: Path) -> None:
    model = _TinyTrackerWithBackbone()
    checkpoint = tmp_path / "tracker.safetensors"
    save_safetensors(checkpoint, {"tracker.no_mem_embed": mx.ones((1, 1, 2))})
    with pytest.raises(ValueError, match="no detector weights"):
        _load_tracker_checkpoint(model, checkpoint)
    np.testing.assert_array_equal(to_numpy(model.no_mem_embed), np.zeros((1, 1, 2)))



@pytest.mark.parametrize("unknown_key", ["tracker.model.unreviewed.weight", "unreviewed.weight"])
def test_multiplex_strict_loading_rejects_unmapped_source_before_mutation(
    tmp_path: Path, unknown_key: str,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    original = to_numpy(flatten_model_parameters(model)["tracker.model.output_valid_embed"]).copy()
    checkpoint = tmp_path / "unreviewed.safetensors"
    weights = {key: mx.ones(value.shape) for key, value in flatten_model_parameters(model).items()}
    weights[unknown_key] = mx.ones((1,))
    save_safetensors(checkpoint, weights)

    with pytest.raises(ValueError, match="source weights without a reviewed model mapping"):
        load_multiplex_checkpoint(model, checkpoint)
    np.testing.assert_array_equal(to_numpy(flatten_model_parameters(model)["tracker.model.output_valid_embed"]), original)


@pytest.mark.parametrize("full_tree", [False, True])
def test_multiplex_tracker_prefix_uses_declared_target_namespace(full_tree: bool) -> None:
    model = _TinyMultiplexCheckpointModel() if full_tree else _TinyMultiplexTrackerCheckpointModel()
    value = mx.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalize_sam31_multiplex_tracker_weights(
        {"tracker.model.output_valid_embed": value}, model, prefix="tracker.model.",
    )
    assert tuple(normalized) == ("tracker.model.output_valid_embed",)
    np.testing.assert_array_equal(to_numpy(normalized["tracker.model.output_valid_embed"]), [[1.0, 2.0], [3.0, 4.0]])


def test_multiplex_normalization_rejects_colliding_source_targets() -> None:
    model = _TinyMergedProjectionModel()
    with pytest.raises(ValueError, match="source weights collide at target"):
        normalize_sam31_multiplex_weights({
            "detector.attention.in_proj_weight": mx.ones((6, 2)),
            "detector.attention.query_proj.weight": mx.ones((2, 2)),
        }, model)


def test_multiplex_partial_projection_target_does_not_consume_packed_source() -> None:
    class PartialTarget(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = {"attention": {"query_proj": nn.Linear(2, 2)}}

    source = mx.reshape(mx.arange(12, dtype=mx.float32), (6, 2))
    key = "detector.attention.in_proj_weight"
    normalized = normalize_sam31_multiplex_weights({key: source}, PartialTarget())
    assert tuple(normalized) == (key,)
    np.testing.assert_array_equal(to_numpy(normalized[key]), [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])


def test_interactive_checkpoint_extracts_image_heads_from_full_tracker(
    tmp_path: Path,
) -> None:
    model = _TinyInteractiveCheckpointModel()
    base = tmp_path / "base.safetensors"
    tracker = tmp_path / "full-tracker.safetensors"
    save_safetensors(base, {"base.weight": mx.array([5.0])})
    save_safetensors(tracker, {
        "tracker.no_mem_embed": mx.array([[[7.0, 8.0]]]),
        "tracker.transformer.encoder.weight": mx.ones((2, 2)),
        "tracker.maskmem_backbone.memory.weight": mx.ones((2, 2)),
    })
    report = _load_checkpoint(model, base, interactive_checkpoint_path=tracker)
    assert report.extra == ()
    assert report.loaded == ("base.weight", "inst_interactive_predictor.model.no_mem_embed")
    np.testing.assert_array_equal(to_numpy(flatten_model_parameters(model)["inst_interactive_predictor.model.no_mem_embed"]), [[[7.0, 8.0]]])


def test_interactive_checkpoint_rejects_unknown_head_without_mutating_model(
    tmp_path: Path,
) -> None:
    model = _TinyInteractiveCheckpointModel()
    base = tmp_path / "base.safetensors"
    tracker = tmp_path / "unknown-head.safetensors"
    original = to_numpy(flatten_model_parameters(model)["base.weight"]).copy()
    save_safetensors(base, {"base.weight": mx.array([5.0])})
    save_safetensors(tracker, {
        "tracker.no_mem_embed": mx.array([[[7.0, 8.0]]]),
        "tracker.sam_mask_decoder.unreviewed.weight": mx.ones((1,)),
    })
    with pytest.raises(ValueError, match="source weights without a reviewed model mapping"):
        _load_checkpoint(model, base, interactive_checkpoint_path=tracker)
    np.testing.assert_array_equal(to_numpy(flatten_model_parameters(model)["base.weight"]), original)


@pytest.mark.parametrize("reverse", [False, True])
def test_interactive_checkpoint_rejects_colliding_aliases_without_mutating_model(
    tmp_path: Path, reverse: bool,
) -> None:
    model = _TinyInteractiveCheckpointModel()
    original = {
        key: to_numpy(value).copy() for key, value in flatten_model_parameters(model).items()
    }
    base = tmp_path / "base.safetensors"
    tracker = tmp_path / "colliding-head.safetensors"
    save_safetensors(base, {"base.weight": mx.array([5.0])})
    sources = [
        ("tracker.no_mem_embed", mx.ones((1, 1, 2))),
        ("sam2_predictor.no_mem_embed", mx.zeros((1, 1, 2))),
    ]
    if reverse:
        sources.reverse()
    save_safetensors(tracker, dict(sources))

    with pytest.raises(ValueError, match="source weights collide"):
        _load_checkpoint(model, base, interactive_checkpoint_path=tracker)

    for key, value in flatten_model_parameters(model).items():
        np.testing.assert_array_equal(to_numpy(value), original[key])
