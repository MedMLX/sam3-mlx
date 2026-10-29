from sam3_mlx.mlx_runtime import to_numpy
import numpy as np
import mlx.core as mx
from mlx import nn
import pytest
from hypothesis import given, strategies as st

from sam3_mlx.convert import normalize_sam3_image_weight_layout
from sam3_mlx.checkpoint_mapping import (
    normalize_inst_interactive_weights,
    normalize_sam31_multiplex_tracker_weights,
    normalize_sam31_multiplex_weights,
    normalize_sam3_image_weights,
    normalize_tracker_checkpoint_weights,
)
from sam3_mlx.model_builder import build_tracker
from tests.checkpoint._factories import WeightLeaf


class _TinySam31MultiplexTree(nn.Module):
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
                },
                "vision_backbone": {
                    "trunk": {
                        "blocks": [
                            {"attn": {"qkv": WeightLeaf((6, 2))}},
                        ],
                        "pos_embed": mx.zeros((1, 4, 2)),
                    },
                    "convs": [
                        {"conv_1x1": WeightLeaf((2, 1, 1, 2))},
                    ],
                },
            },
            "transformer": {
                "decoder": {
                    "layers": [
                        {
                            "norm1": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                            "norm2": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                            "catext_norm": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                            "norm3": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                        }
                    ]
                }
            },
            "geometry_encoder": {
                "norm": {"weight": mx.zeros((2,)), "bias": mx.zeros((2,))},
                "img_pre_norm": {
                    "weight": mx.zeros((2,)),
                    "bias": mx.zeros((2,)),
                },
                "encode_norm": {
                    "weight": mx.zeros((2,)),
                    "bias": mx.zeros((2,)),
                },
            },
        }
        self.tracker = {
            "model": {
                "interactive_sam_prompt_encoder": {
                    "point_embeddings": [
                        WeightLeaf((1, 2)),
                        WeightLeaf((1, 2)),
                    ],
                },
                "sam_mask_decoder": {
                    "conv_s0": {"conv": WeightLeaf((2, 1, 1, 2))},
                    "output_hypernetworks_mlps": [
                        {
                            "layers": [
                                WeightLeaf((2, 2)),
                                WeightLeaf((2, 2)),
                                WeightLeaf((1, 2)),
                            ]
                        }
                    ],
                },
            }
        }


class _TinySam31TrackerTree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_valid_embed = mx.zeros((2, 2))


def test_normalize_sam3_image_weight_layout_converts_sam2_neck_conv_transpose_to_hwio() -> None:
    # dconv branch contract: PyTorch ConvTranspose2d weight (in, out, H, W) -> MLX (out, H, W, in).
    # Expected literal is hand-computed, NOT a transpose call mirroring production.
    key = "backbone.vision_backbone.sam2_convs.0.dconv_2x2_0.weight"
    torch_layout = mx.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ],
        dtype=mx.float32,
    )
    assert torch_layout.shape == (2, 1, 2, 2)

    mlx_layout = normalize_sam3_image_weight_layout(key, torch_layout)

    np.testing.assert_array_equal(
        to_numpy(mlx_layout),
        np.array(
            [
                [
                    [[1.0, 5.0], [2.0, 6.0]],
                    [[3.0, 7.0], [4.0, 8.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_sam3_image_weight_layout_converts_sam2_neck_conv2d_to_ohwi() -> None:
    # conv2d branch contract: PyTorch Conv2d weight (O, I, H, W) -> MLX (O, H, W, I).
    # Expected literal is hand-computed, NOT a transpose call mirroring production.
    key = "backbone.vision_backbone.sam2_convs.1.conv_3x3.weight"
    torch_layout = mx.array(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ],
        dtype=mx.float32,
    )
    assert torch_layout.shape == (1, 2, 3, 3)

    mlx_layout = normalize_sam3_image_weight_layout(key, torch_layout)

    np.testing.assert_array_equal(
        to_numpy(mlx_layout),
        np.array(
            [
                [
                    [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
                    [[4.0, 13.0], [5.0, 14.0], [6.0, 15.0]],
                    [[7.0, 16.0], [8.0, 17.0], [9.0, 18.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_sam3_image_weight_layout_leaves_non_conv_keys_untouched() -> None:
    # Boundary: keys outside both conv sets must pass through unchanged.
    key = "geometry_encoder.unrelated.weight"
    payload = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)

    out = normalize_sam3_image_weight_layout(key, payload)

    assert out is payload, "Pass-through path must not allocate a new array."


def test_normalize_sam3_image_weights_unwraps_only_model_container() -> None:
    value = mx.array([1.0, 2.0], dtype=mx.float16)

    normalized = normalize_sam3_image_weights(
        {"model": {"head.weight": value}},
        include_tracker=False,
    )

    assert tuple(normalized) == ("head.weight",)
    assert normalized["head.weight"].dtype == mx.float16
    np.testing.assert_array_equal(
        to_numpy(normalized["head.weight"]),
        np.array([1.0, 2.0], dtype=np.float16),
    )


def test_normalize_sam3_image_weights_rejects_unsupported_state_dict_container() -> (
    None
):
    with pytest.raises(
        TypeError,
        match="Expected checkpoint value for 'state_dict' to be an MLX array",
    ):
        normalize_sam3_image_weights(
            {"state_dict": {"head.weight": mx.ones((2,))}},
            include_tracker=False,
        )


def test_normalize_sam3_image_weights_repairs_legacy_unprefixed_checkpoint() -> None:
    # Distinct channel values distinguish the expected transpose from any
    # transpose because wrong axes place different scalars in different positions.
    dconv_key = "backbone.vision_backbone.sam2_convs.1.dconv_2x2.weight"
    conv_key = "backbone.vision_backbone.sam2_convs.2.conv_1x1.weight"
    dconv_input = mx.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ],
        dtype=mx.float32,
    )
    conv_input = mx.array(
        [[[[10.0]], [[20.0]], [[30.0]]]],
        dtype=mx.float32,
    )
    payload = {dconv_key: dconv_input, conv_key: conv_input}

    normalized = normalize_sam3_image_weights(payload, include_tracker=False)

    np.testing.assert_array_equal(
        to_numpy(normalized[dconv_key]),
        np.array(
            [
                [
                    [[1.0, 5.0], [2.0, 6.0]],
                    [[3.0, 7.0], [4.0, 8.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized[conv_key]),
        np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32),
    )


def test_normalize_sam3_image_weights_strips_detector_prefix_and_keeps_layout() -> None:
    # Verify prefix stripping, tracker-key filtering, and layout conversion with a
    # hand-computed result, not a second production transpose.
    conv_key = "backbone.vision_backbone.sam2_convs.3.conv_3x3.weight"
    torch_layout = mx.array(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ],
        dtype=mx.float32,
    )
    payload = {
        "detector." + conv_key: torch_layout,
        "tracker.prompt_encoder.weight": mx.ones((2, 2)),
    }

    normalized = normalize_sam3_image_weights(payload, include_tracker=False)

    assert sorted(normalized) == [conv_key]
    np.testing.assert_array_equal(
        to_numpy(normalized[conv_key]),
        np.array(
            [
                [
                    [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
                    [[4.0, 13.0], [5.0, 14.0], [6.0, 15.0]],
                    [[7.0, 16.0], [8.0, 17.0], [9.0, 18.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_sam3_image_weights_maps_tracker_when_requested() -> None:
    # 1-D inputs do not trigger any layout transpose; this pins the *key renaming*
    # contract specifically (detector prefix stripped, tracker keys remapped).
    payload = {
        "sam3_model.some_detector_weight": mx.array([1.0], dtype=mx.float32),
        "sam2_predictor.sam_prompt_encoder.no_mask_embed.weight": mx.array(
            [2.0],
            dtype=mx.float32,
        ),
    }

    normalized = normalize_sam3_image_weights(payload, include_tracker=True)

    assert sorted(normalized) == [
        "inst_interactive_predictor.model.sam_prompt_encoder.no_mask_embed.weight",
        "some_detector_weight",
    ]
    assert normalized["some_detector_weight"].tolist() == [1.0]
    assert normalized[
        "inst_interactive_predictor.model.sam_prompt_encoder.no_mask_embed.weight"
    ].tolist() == [2.0]


def test_normalize_sam3_image_weights_rejects_tracker_only_official_payload() -> None:
    payload = {"tracker.prompt_encoder.weight": mx.ones((2, 2))}

    with pytest.raises(
        ValueError,
        match="official prefixes but no detector weights",
    ):
        normalize_sam3_image_weights(payload, include_tracker=True)


def test_normalize_inst_interactive_weights_splits_transformers_point_embeddings() -> None:
    payload = {
        "tracker_model.prompt_encoder.point_embed.weight": mx.reshape(
            mx.arange(12), (4, 3)
        ).astype(mx.float32)
    }

    normalized = normalize_inst_interactive_weights(payload)

    assert sorted(normalized) == [
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.0.weight",
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.1.weight",
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.2.weight",
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.3.weight",
    ]
    np.testing.assert_array_equal(
        to_numpy(
            normalized[
                "inst_interactive_predictor.model.sam_prompt_encoder."
                "point_embeddings.2.weight"
            ]
        ),
        np.array([[6.0, 7.0, 8.0]], dtype=np.float32),
    )


@given(st.lists(
    st.lists(st.integers(-100, 100), min_size=3, max_size=3),
    min_size=4,
    max_size=4,
))
def test_interactive_point_embedding_split_roundtrip(rows: list[list[int]]) -> None:
    packed = mx.array(rows, dtype=mx.float32)
    normalized = normalize_inst_interactive_weights({
        "tracker_model.prompt_encoder.point_embed.weight": packed,
    })
    restored = mx.concatenate([
        normalized[
            "inst_interactive_predictor.model.sam_prompt_encoder."
            f"point_embeddings.{index}.weight"
        ]
        for index in range(4)
    ])
    np.testing.assert_array_equal(to_numpy(restored), to_numpy(packed))


@pytest.mark.parametrize("aliases", [
    ("tracker.no_mem_embed", "sam2_predictor.no_mem_embed"),
    ("tracker_model.no_memory_embedding", "inst_interactive_predictor.model.no_mem_embed"),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_interactive_normalization_rejects_colliding_aliases(
    aliases: tuple[str, str], reverse: bool,
) -> None:
    sources = [(aliases[0], mx.ones((1, 1, 2))), (aliases[1], mx.zeros((1, 1, 2)))]
    if reverse:
        sources.reverse()
    with pytest.raises(ValueError, match="source weights collide"):
        normalize_inst_interactive_weights(dict(sources))


@pytest.mark.parametrize("prefix", [
    "tracker.", "sam2_predictor.", "inst_interactive_predictor.model.",
])
@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("reverse", [False, True])
def test_interactive_normalization_rejects_colliding_split_point_embeddings(
    prefix: str, index: int, reverse: bool,
) -> None:
    sources = [
        ("tracker_model.prompt_encoder.point_embed.weight", mx.ones((4, 3))),
        (prefix + f"sam_prompt_encoder.point_embeddings.{index}.weight", mx.zeros((1, 3))),
    ]
    if reverse:
        sources.reverse()
    with pytest.raises(ValueError, match="source weights collide"):
        normalize_inst_interactive_weights(dict(sources))


def test_normalize_inst_interactive_weights_maps_transformers_aliases() -> None:
    payload = {
        "tracker_model.no_memory_embedding": mx.ones((1, 1, 2)),
        "tracker_model.prompt_encoder.shared_embedding.positional_embedding": mx.ones(
            (2, 1)
        ),
        "tracker_model.prompt_encoder.mask_embed.layer_norm1.weight": mx.ones((4,)),
        "tracker_model.mask_decoder.conv_s0.bias": mx.ones((32,)),
        "tracker_model.mask_decoder.iou_prediction_head.proj_in.weight": mx.ones(
            (2, 2)
        ),
        "tracker_model.mask_decoder.iou_prediction_head.layers.0.bias": mx.ones((2,)),
        "tracker_model.mask_decoder.iou_prediction_head.proj_out.bias": mx.ones((1,)),
        "tracker_model.mask_decoder.output_hypernetworks_mlps.3.proj_out.weight": mx.ones(
            (1, 2)
        ),
        "tracker_model.mask_decoder.transformer.layers.0.self_attn.o_proj.bias": mx.ones(
            (2,)
        ),
        "tracker_model.mask_decoder.transformer.layers.0.layer_norm3.weight": mx.ones(
            (2,)
        ),
        "tracker_model.mask_decoder.transformer.layers.0.mlp.proj_in.bias": mx.ones(
            (4,)
        ),
    }

    normalized = normalize_inst_interactive_weights(payload)

    assert "inst_interactive_predictor.model.no_mem_embed" in normalized
    assert (
        "inst_interactive_predictor.model.sam_prompt_encoder.pe_layer."
        "positional_encoding_gaussian_matrix" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_prompt_encoder."
        "mask_downscaling.1.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.conv_s0.conv.bias"
        in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "iou_prediction_head.layers.0.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "iou_prediction_head.layers.1.bias" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "iou_prediction_head.layers.2.bias" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "output_hypernetworks_mlps.3.layers.2.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.transformer."
        "layers.0.self_attn.out_proj.bias" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.transformer."
        "layers.0.norm3.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.transformer."
        "layers.0.mlp.lin1.bias" in normalized
    )


def test_normalize_inst_interactive_weights_converts_official_conv2d_layout() -> None:
    payload = {
        "tracker_model.prompt_encoder.mask_embed.conv1.weight": mx.reshape(
            mx.arange(16), (4, 1, 2, 2)
        ).astype(mx.float32)
    }

    normalized = normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_prompt_encoder."
        "mask_downscaling.0.conv.weight"
    )
    assert tuple(normalized[key].shape) == (4, 2, 2, 1)
    np.testing.assert_array_equal(
        to_numpy(normalized[key]),
        np.array(
            [
                [[[0.0], [1.0]], [[2.0], [3.0]]],
                [[[4.0], [5.0]], [[6.0], [7.0]]],
                [[[8.0], [9.0]], [[10.0], [11.0]]],
                [[[12.0], [13.0]], [[14.0], [15.0]]],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_inst_interactive_weights_maps_sam2_predictor_prompt_conv_alias() -> None:
    payload = {
        "sam2_predictor.sam_prompt_encoder.mask_downscaling.0.weight": mx.reshape(
            mx.arange(16), (4, 1, 2, 2)
        ).astype(mx.float32)
    }

    normalized = normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_prompt_encoder."
        "mask_downscaling.0.conv.weight"
    )
    assert sorted(normalized) == [key]
    np.testing.assert_array_equal(
        to_numpy(normalized[key]),
        np.array(
            [
                [[[0.0], [1.0]], [[2.0], [3.0]]],
                [[[4.0], [5.0]], [[6.0], [7.0]]],
                [[[8.0], [9.0]], [[10.0], [11.0]]],
                [[[12.0], [13.0]], [[14.0], [15.0]]],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_inst_interactive_weights_converts_official_convtranspose_layout() -> None:
    payload = {
        "tracker_model.mask_decoder.upscale_conv1.weight": mx.reshape(
            mx.arange(256 * 64 * 2 * 2), (256, 64, 2, 2)
        ).astype(mx.float32)
    }

    normalized = normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "output_upscaling.0.conv.weight"
    )
    converted = normalized[key]
    assert tuple(converted.shape) == (64, 2, 2, 256)
    assert (
        converted[3, 1, 0, 5].item()
        == payload["tracker_model.mask_decoder.upscale_conv1.weight"][5, 3, 1, 0].item()
    )


def test_normalize_inst_interactive_weights_maps_tracker_decoder_upscale_alias() -> None:
    payload = {
        "tracker.sam_mask_decoder.output_upscaling.0.weight": mx.reshape(
            mx.arange(256 * 64 * 2 * 2), (256, 64, 2, 2)
        ).astype(mx.float32)
    }

    normalized = normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "output_upscaling.0.conv.weight"
    )
    assert sorted(normalized) == [key]
    converted = normalized[key]
    assert tuple(converted.shape) == (64, 2, 2, 256)
    assert (
        converted[3, 1, 0, 5].item()
        == payload["tracker.sam_mask_decoder.output_upscaling.0.weight"][
            5, 3, 1, 0
        ].item()
    )


def test_normalize_tracker_checkpoint_weights_maps_core_tracker_aliases() -> None:
    model = build_tracker(apply_temporal_disambiguation=False, with_backbone=False)
    payload = {
        "tracker_model.no_memory_embedding": mx.ones((1, 1, 256), dtype=mx.float32),
        "tracker_model.memory_temporal_positional_encoding": mx.ones(
            (7, 1, 1, 64),
            dtype=mx.float32,
        ),
        "tracker_model.mask_downsample.bias": mx.array([3.0], dtype=mx.float32),
        "tracker_model.mask_downsample.weight": mx.reshape(
            mx.arange(16), (1, 1, 4, 4)
        ).astype(mx.float32),
        "tracker_model.prompt_encoder.point_embed.weight": mx.reshape(
            mx.arange(4 * 256), (4, 256)
        ).astype(mx.float32),
        "tracker_model.memory_encoder.projection.weight": mx.reshape(
            mx.arange(64 * 256), (64, 256, 1, 1)
        ).astype(mx.float32),
        "tracker_model.memory_attention.layers.0.self_attn.o_proj.bias": mx.ones(
            (256,),
            dtype=mx.float32,
        ),
        "tracker_model.mask_decoder.upscale_conv1.weight": mx.reshape(
            mx.arange(256 * 64 * 2 * 2), (256, 64, 2, 2)
        ).astype(mx.float32),
    }

    normalized = normalize_tracker_checkpoint_weights(payload, model)

    assert normalized["no_mem_embed"].shape == (1, 1, 256)
    assert normalized["maskmem_tpos_enc"].shape == (7, 1, 1, 64)
    assert normalized["mask_downsample.bias"].shape == (1,)
    mask_downsample = normalized["mask_downsample.weight"]
    assert tuple(mask_downsample.shape) == (1, 4, 4, 1)
    assert (
        mask_downsample[0, 2, 3, 0].item()
        == payload["tracker_model.mask_downsample.weight"][0, 0, 2, 3].item()
    )
    assert "sam_prompt_encoder.point_embeddings.3.weight" in normalized
    assert "transformer.encoder.layers.0.self_attn.out_proj.bias" in normalized
    projection = normalized["maskmem_backbone.out_proj.conv.weight"]
    assert tuple(projection.shape) == (64, 1, 1, 256)
    assert (
        projection[3, 0, 0, 5].item()
        == payload["tracker_model.memory_encoder.projection.weight"][3, 5, 0, 0].item()
    )
    upscaled = normalized["sam_mask_decoder.output_upscaling.0.conv.weight"]
    assert tuple(upscaled.shape) == (64, 2, 2, 256)
    assert (
        upscaled[3, 1, 0, 5].item()
        == payload["tracker_model.mask_decoder.upscale_conv1.weight"][5, 3, 1, 0].item()
    )


@pytest.mark.parametrize("prefix", ["tracker.", "sam2_predictor."])
def test_tracker_raw_wrappers_preserve_conv_and_deconv_values(prefix: str) -> None:
    model = build_tracker(apply_temporal_disambiguation=False, with_backbone=False)
    depthwise = mx.reshape(mx.arange(256 * 7 * 7), (256, 1, 7, 7)).astype(mx.float32)
    deconv = mx.reshape(mx.arange(256 * 64 * 2 * 2), (256, 64, 2, 2)).astype(mx.float32)
    payload = {
        prefix + "maskmem_backbone.fuser.layers.0.dwconv.weight": depthwise,
        prefix + "maskmem_backbone.fuser.layers.0.dwconv.bias": mx.full((256,), 7.0),
        prefix + "sam_mask_decoder.output_upscaling.0.weight": deconv,
    }
    normalized = normalize_tracker_checkpoint_weights(payload, model)
    conv_key = "maskmem_backbone.fuser.layers.0.dwconv.conv"
    assert normalized[conv_key + ".weight"].shape == (256, 7, 7, 1)
    assert normalized[conv_key + ".weight"][3, 2, 1, 0].item() == depthwise[3, 0, 2, 1].item()
    assert normalized[conv_key + ".bias"][3].item() == 7.0
    upscaled = normalized["sam_mask_decoder.output_upscaling.0.conv.weight"]
    assert upscaled.shape == (64, 2, 2, 256)
    assert upscaled[3, 1, 0, 5].item() == deconv[5, 3, 1, 0].item()


def test_normalize_sam31_multiplex_weights_maps_detector_and_tracker_aliases() -> None:
    model = _TinySam31MultiplexTree()
    q = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    k = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
    v = mx.array([[9.0, 10.0], [11.0, 12.0]], dtype=mx.float32)
    pos = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=mx.float32)
    text_projection = mx.array([[13.0, 14.0]], dtype=mx.float32)
    text_resizer = mx.array([[15.0], [16.0]], dtype=mx.float32)
    text_resizer_bias = mx.array([17.0, 18.0], dtype=mx.float32)
    neck = mx.reshape(mx.arange(4), (2, 1, 1, 2)).astype(mx.float32)
    decoder_self_norm = mx.array([31.0, 32.0], dtype=mx.float32)
    decoder_vision_norm = mx.array([33.0, 34.0], dtype=mx.float32)
    geometry_prompt_norm = mx.array([35.0, 36.0], dtype=mx.float32)
    geometry_vision_norm = mx.array([37.0, 38.0], dtype=mx.float32)
    point_embed = mx.reshape(mx.arange(8), (4, 2)).astype(mx.float32)
    conv_s0 = mx.ones((2, 1, 1, 2), dtype=mx.float32)
    proj_out = mx.array([[21.0, 22.0]], dtype=mx.float32)
    payload = {
        "detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight": q,
        "detector_model.vision_encoder.backbone.layers.0.attention.k_proj.weight": k,
        "detector_model.vision_encoder.backbone.layers.0.attention.v_proj.weight": v,
        "detector_model.vision_encoder.backbone.embeddings.position_embeddings": pos,
        "detector_model.text_encoder.text_projection.weight": text_projection,
        "detector_model.text_projection.weight": text_resizer,
        "detector_model.text_projection.bias": text_resizer_bias,
        "detector_model.vision_encoder.neck.convs.0.proj1.weight": neck,
        "detector_model.detr_decoder.layers.0.self_attn_layer_norm.weight": (
            decoder_self_norm
        ),
        "detector_model.detr_decoder.layers.0.vision_cross_attn_layer_norm.weight": (
            decoder_vision_norm
        ),
        "detector_model.geometry_encoder.prompt_layer_norm.weight": (
            geometry_prompt_norm
        ),
        "detector_model.geometry_encoder.vision_layer_norm.weight": (
            geometry_vision_norm
        ),
        "tracker_model.interactive_sam_prompt_encoder.point_embed.weight": point_embed,
        "tracker_model.sam_mask_decoder.conv_s0.weight": conv_s0,
        "tracker_model.sam_mask_decoder.output_hypernetworks_mlps.0.proj_out.weight": proj_out,
    }

    normalized = normalize_sam31_multiplex_weights(payload, model)

    np.testing.assert_array_equal(
        to_numpy(
            normalized[
                "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"
            ]
        ),
        to_numpy(mx.concat([q, k, v], axis=0)),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.backbone.vision_backbone.trunk.pos_embed"]),
        np.array(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        to_numpy(
            normalized["detector.backbone.language_backbone.encoder.text_projection"]
        ),
        np.array([[13.0], [14.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.backbone.language_backbone.resizer.weight"]),
        to_numpy(text_resizer),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.backbone.language_backbone.resizer.bias"]),
        to_numpy(text_resizer_bias),
    )
    np.testing.assert_array_equal(
        to_numpy(
            normalized["detector.backbone.vision_backbone.convs.0.conv_1x1.weight"]
        ),
        to_numpy(neck),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.transformer.decoder.layers.0.norm1.weight"]),
        to_numpy(decoder_self_norm),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.transformer.decoder.layers.0.norm2.weight"]),
        to_numpy(decoder_vision_norm),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.geometry_encoder.img_pre_norm.weight"]),
        to_numpy(geometry_prompt_norm),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["detector.geometry_encoder.norm.weight"]),
        to_numpy(geometry_vision_norm),
    )
    np.testing.assert_array_equal(
        to_numpy(
            normalized[
                "tracker.model.interactive_sam_prompt_encoder.point_embeddings.1.weight"
            ]
        ),
        np.array([[2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(normalized["tracker.model.sam_mask_decoder.conv_s0.conv.weight"]),
        to_numpy(conv_s0),
    )
    np.testing.assert_array_equal(
        to_numpy(
            normalized[
                "tracker.model.sam_mask_decoder.output_hypernetworks_mlps.0."
                "layers.2.weight"
            ]
        ),
        to_numpy(proj_out),
    )


def test_normalize_sam31_multiplex_tracker_weights_maps_direct_model_prefix() -> None:
    model = _TinySam31TrackerTree()
    payload = {
        "tracker_model.output_valid_embed": mx.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=mx.float32,
        )
    }

    normalized = normalize_sam31_multiplex_tracker_weights(payload, model)

    assert sorted(normalized) == ["output_valid_embed"]
    np.testing.assert_array_equal(
        to_numpy(normalized["output_valid_embed"]),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_sam31_incomplete_split_projection_preserves_unmapped_source() -> None:
    model = _TinySam31MultiplexTree()
    key = "detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight"
    value = mx.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalize_sam31_multiplex_weights({key: value}, model)
    assert tuple(normalized) == (key,)
    np.testing.assert_array_equal(to_numpy(normalized[key]), [[1.0, 2.0], [3.0, 4.0]])
