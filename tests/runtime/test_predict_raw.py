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
    UniversalSegmentationHead,
)
from sam3_mlx.model.model_misc import (
    DotProductScoring,
    MultiheadAttention,
    TransformerWrapper,
)
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.precision import apply_precision_policy


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
    apply_precision_policy(model, "bf16")
    assert model.cleared == ["visual", "grounding"]


def _tiny_grounding_model() -> Sam3Image:
    mx.random.seed(19)

    def encoder_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            d_model=8,
            dim_feedforward=16,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            self_attention=MultiheadAttention(8, 2),
            cross_attention=MultiheadAttention(8, 2),
        )

    def geometry_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            d_model=8,
            dim_feedforward=16,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            self_attention=MultiheadAttention(8, 2),
            cross_attention=MultiheadAttention(8, 2),
        )

    def decoder_layer() -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            d_model=8,
            dim_feedforward=16,
            cross_attention=MultiheadAttention(8, 2),
            n_heads=2,
        )

    encoder = TransformerEncoderFusion(layer=encoder_layer, num_layers=1)
    decoder = TransformerDecoder(
        layer=decoder_layer, num_layers=1, num_queries=2, d_model=8
    )
    geometry = SequenceGeometryEncoder(
        d_model=8,
        pos_enc=PositionEmbeddingSine(num_pos_feats=8),
        num_layers=1,
        layer=geometry_layer,
        roi_size=2,
    )
    head = UniversalSegmentationHead(
        hidden_dim=8,
        pixel_decoder=PixelDecoder(hidden_dim=8, num_upsampling_stages=1),
        cross_attend_prompt=MultiheadAttention(8, 2),
    )
    model = Sam3Image(
        # predict_raw accepts precomputed image/text features; no backbone runs.
        backbone=cast(SAM3VLBackbone, nn.Identity()),
        transformer=TransformerWrapper(encoder, decoder, d_model=8),
        input_geometry_encoder=geometry,
        segmentation_head=head,
        dot_prod_scoring=DotProductScoring(d_model=8, d_proj=8),
    )
    model.eval()
    evaluate_boundary(cast(object, model.parameters()))
    return model


@pytest.mark.parametrize("batch_size", [1, 2])
def test_compiled_predict_raw_preserves_eager_outputs(batch_size: int) -> None:
    model = _tiny_grounding_model()
    stage = FindStage(
        img_ids=mx.arange(batch_size, dtype=mx.int64),
        text_ids=mx.arange(batch_size, dtype=mx.int64),
    )
    prompt = Prompt.empty(batch_size)
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
        keys = set(backbone_out)
        model.compile_grounding = False
        expected = model.predict_raw(backbone_out, stage, prompt)
        model.compile_grounding = True
        actual = model.predict_raw(backbone_out, stage, prompt)

        assert set(backbone_out) == keys
        assert expected["pred_masks"].shape == (batch_size, 2, 4, 4)
        assert "semantic_seg" in expected
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
        assert "semantic_seg" in actual
        semantic = to_numpy(actual["semantic_seg"])
        assert semantic.shape == (batch_size, 1, 4, 4)
        if previous_semantic is not None:
            assert not np.array_equal(semantic, previous_semantic)
        previous_semantic = semantic
