from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest
from mlx import nn

from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.data_misc import FindStage
from sam3_mlx.model.decoder import TransformerDecoder, TransformerDecoderLayer
from sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3_mlx.model.geometry_encoders import Prompt, SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import (
    PixelDecoder,
    SegmentationHead,
    UniversalSegmentationHead,
)
from sam3_mlx.model.model_misc import (
    DotProductScoring,
    MultiheadAttention,
    TransformerWrapper,
)
from sam3_mlx.model.sam3_image import (
    Output,
    Sam3Image,
    grounding_compile_key,
    raw_prediction_from_output,
)
from sam3_mlx.precision import apply_precision_policy


def test_raw_prediction_from_output_requires_fixed_shape_arrays() -> None:
    out = Output(
        {
            "pred_logits": mx.ones((1, 2, 1), dtype=mx.float32),
            "pred_boxes": mx.ones((1, 2, 4), dtype=mx.float32),
            "pred_masks": mx.ones((1, 2, 2, 2), dtype=mx.float32),
            "presence_logit_dec": mx.ones((1, 1), dtype=mx.float32),
            "pred_boxes_xyxy": mx.ones((1, 2, 4), dtype=mx.float32),
        }
    )

    raw = raw_prediction_from_output(out)

    assert raw["pred_logits"].shape == (1, 2, 1)
    assert raw["pred_boxes"].shape == (1, 2, 4)
    assert raw["pred_masks"].shape == (1, 2, 2, 2)
    assert raw["presence_logit_dec"].shape == (1, 1)
    assert "pred_boxes_xyxy" in raw
    assert raw["pred_boxes_xyxy"].shape == (1, 2, 4)


def test_raw_prediction_from_output_rejects_missing_masks() -> None:
    with pytest.raises(KeyError, match="pred_masks"):
        raw_prediction_from_output(
            {
                "pred_logits": mx.ones((1, 1, 1)),
                "pred_boxes": mx.ones((1, 1, 4)),
                "presence_logit_dec": mx.ones((1, 1)),
            }
        )


def test_grounding_compile_key_includes_dtype_shape_and_prompt_geometry() -> None:
    fp32_key = grounding_compile_key(
        dtype=mx.float32,
        feat_shape=(72, 1, 256),
        prompt_shape=(32, 1, 256),
        prompt_mask_shape=(1, 32),
        vis_feat_sizes=((72, 72),),
        fpn_shapes=((1, 256, 72, 72),),
        batch_size=1,
    )
    fp16_key = grounding_compile_key(
        dtype=mx.float16,
        feat_shape=(72, 1, 256),
        prompt_shape=(32, 1, 256),
        prompt_mask_shape=(1, 32),
        vis_feat_sizes=((72, 72),),
        fpn_shapes=((1, 256, 72, 72),),
        batch_size=1,
    )
    prompt_key = grounding_compile_key(
        dtype=mx.float32,
        feat_shape=(72, 1, 256),
        prompt_shape=(40, 1, 256),
        prompt_mask_shape=(1, 40),
        vis_feat_sizes=((72, 72),),
        fpn_shapes=((1, 256, 72, 72),),
        batch_size=1,
    )

    assert fp32_key != fp16_key
    assert fp32_key != prompt_key
    assert fp32_key[0] == "float32"
    assert fp16_key[0] == "float16"
    assert prompt_key[2] == (40, 1, 256)


class _Clearable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.cleared: list[str] = []

    def clear_compiled_visual(self) -> None:
        self.cleared.append("visual")

    def clear_compiled_grounding(self) -> None:
        self.cleared.append("grounding")


def test_apply_precision_policy_clears_compiled_visual_and_grounding() -> None:
    model = _Clearable()
    apply_precision_policy(model, "fp16")
    assert model.cleared == ["visual", "grounding"]


def _tiny_grounding_model(*, semantic_output: bool) -> Sam3Image:
    mx.random.seed(19)

    def encoder_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            activation="relu",
            d_model=8,
            dim_feedforward=16,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention=MultiheadAttention(num_heads=2, dims=8),
            cross_attention=MultiheadAttention(num_heads=2, dims=8),
        )

    def decoder_layer() -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            activation="relu",
            d_model=8,
            dim_feedforward=16,
            cross_attention=MultiheadAttention(num_heads=2, dims=8),
            n_heads=2,
            use_text_cross_attention=True,
        )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=1,
        d_model=8,
        num_feature_levels=1,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=1,
        num_queries=2,
        return_intermediate=True,
        box_refine=True,
        boxRPB="log",
        d_model=8,
        interaction_layer=None,
        presence_token=True,
    )
    geometry = SequenceGeometryEncoder(
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=False,
        points_pos_enc=False,
        boxes_direct_project=True,
        boxes_pool=False,
        boxes_pos_enc=False,
        d_model=8,
        pos_enc=None,
        num_layers=0,
        layer=None,
        add_post_encode_proj=False,
    )
    pixel_decoder = PixelDecoder(hidden_dim=8, num_upsampling_stages=1)
    head = (
        UniversalSegmentationHead(
            hidden_dim=8, upsampling_stages=1, pixel_decoder=pixel_decoder
        )
        if semantic_output
        else SegmentationHead(
            hidden_dim=8,
            upsampling_stages=1,
            pixel_decoder=pixel_decoder,
            use_encoder_inputs=True,
        )
    )
    model = Sam3Image(
        # predict_raw accepts precomputed image/text features; no backbone runs.
        backbone=nn.Identity(),
        transformer=TransformerWrapper(encoder, decoder, d_model=8),
        input_geometry_encoder=geometry,
        segmentation_head=head,
        dot_prod_scoring=DotProductScoring(d_model=8, d_proj=8),
    )
    model.eval()
    evaluate_boundary(cast(object, model.parameters()))
    return model


@pytest.mark.parametrize("semantic_output", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_compiled_predict_raw_preserves_eager_outputs(
    semantic_output: bool, batch_size: int
) -> None:
    model = _tiny_grounding_model(semantic_output=semantic_output)
    stage = FindStage(
        img_ids=mx.arange(batch_size, dtype=mx.int64),
        text_ids=mx.arange(batch_size, dtype=mx.int64),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    prompt = Prompt(box_embeddings=mx.zeros((0, batch_size, 4)))
    features = [
        mx.reshape(
            mx.sin(mx.arange(batch_size * 128, dtype=mx.float32)),
            (batch_size, 8, 4, 4),
        ),
        mx.reshape(
            mx.cos(mx.arange(batch_size * 32, dtype=mx.float32)),
            (batch_size, 8, 2, 2),
        ),
    ]
    previous_semantic: npt.NDArray[np.generic] | None = None
    for sign in (1.0, -1.0):
        backbone_out = {
            "backbone_fpn": [sign * feature for feature in features],
            "vision_pos_enc": [mx.zeros_like(feature) for feature in features],
            "language_features": mx.reshape(
                mx.linspace(-0.5, 0.5, batch_size * 24), (3, batch_size, 8)
            ),
            "language_mask": mx.tile(
                mx.array([[False, False, True]]), (batch_size, 1)
            ),
        }
        model.compile_grounding = False
        expected = model.predict_raw(backbone_out, stage, None, prompt)
        model.compile_grounding = True
        actual = model.predict_raw(backbone_out, stage, None, prompt)

        assert ("semantic_seg" in expected) is semantic_output
        assert set(actual) == set(expected)
        actual_arrays = cast(Mapping[str, mx.array], actual)
        for key, expected_value in cast(Mapping[str, mx.array], expected).items():
            np.testing.assert_allclose(
                to_numpy(actual_arrays[key]),
                to_numpy(expected_value),
                rtol=1e-5,
                atol=1e-6,
                err_msg=key,
            )
        if semantic_output:
            assert "semantic_seg" in actual
            semantic = to_numpy(actual["semantic_seg"])
            assert semantic.shape == (batch_size, 1, 4, 4)
            if previous_semantic is not None:
                assert not np.array_equal(semantic, previous_semantic)
            previous_semantic = semantic
