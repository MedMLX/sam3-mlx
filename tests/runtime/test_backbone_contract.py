from __future__ import annotations

from typing import cast

import mlx.core as mx
import pytest
from tests._images import chest_input

from sam3_mlx.checkpoint_mapping import flatten_model_parameters
from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.model.necks import (
    Scale0_5FN,
    Scale1FN,
    Scale2FN,
    Scale4FN,
    ScaleHead,
    Sam3DualViTDetNeck,
)
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.vitdet import ViT
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.precision import apply_precision_policy


def _dual_neck() -> Sam3DualViTDetNeck:
    # A 16x16 image gives a 4x4 grid of 8-channel trunk tokens.
    trunk = ViT(
        img_size=16,
        patch_size=4,
        embed_dim=8,
        depth=1,
        num_heads=2,
        global_att_blocks=(0,),
        pretrain_img_size=16,
    )
    return Sam3DualViTDetNeck(
        trunk=trunk,
        position_encoding=PositionEmbeddingSine(num_pos_feats=4, normalize=True),
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=True,
    )


@pytest.mark.parametrize("input_dtype", [mx.float32, mx.bfloat16])
def test_bf16_scale2_matches_native_autocast_transposed_convolution(
    input_dtype: mx.Dtype,
) -> None:
    mx.random.seed(1729)
    head = Scale2FN(8, 4)
    apply_precision_policy(head, "bf16")
    x = mx.random.normal((1, 3, 4, 8)).astype(input_dtype)

    # Compare the actual head with the original native deconvolution path,
    # retaining the same weights and subsequent projections.
    expected = head.conv_3x3(head.conv_1x1(head.dconv_2x2(x)))
    actual = head(x)
    evaluate_boundary(actual, expected)
    assert actual.dtype == expected.dtype == mx.bfloat16
    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
    assert bool(error <= 0.0005)


def test_scale_heads_produce_expected_spatial_shapes() -> None:
    x = mx.ones((1, 4, 4, 8), dtype=mx.float32)
    cases: list[tuple[ScaleHead, tuple[int, int, int, int]]] = [
        (Scale4FN(8, 4), (1, 16, 16, 4)),
        (Scale2FN(8, 4), (1, 8, 8, 4)),
        (Scale1FN(8, 4), (1, 4, 4, 4)),
        (Scale0_5FN(8, 4), (1, 2, 2, 4)),
    ]
    for head, expected in cases:
        out = head(x)
        evaluate_boundary(out)
        assert out.shape == expected


class _CountingScale0_5FN(Scale0_5FN):
    def __init__(self) -> None:
        super().__init__(8, 4)
        self.calls = 0

    def __call__(self, x: mx.array) -> mx.array:
        self.calls += 1
        return super().__call__(x)


def _assert_array_lists_equal(left: list[mx.array], right: list[mx.array]) -> None:
    assert len(left) == len(right)
    for left_item, right_item in zip(left, right, strict=True):
        evaluate_boundary(left_item, right_item)
        assert bool(mx.array_equal(left_item, right_item).item())
        assert left_item.dtype == right_item.dtype


def test_vl_backbone_scalp_does_not_execute_discarded_head() -> None:
    """Scalping drops the lowest-resolution level without changing retained levels."""
    neck = _dual_neck()
    discarded = _CountingScale0_5FN()
    assert neck.sam2_convs is not None
    discarded_sam2 = _CountingScale0_5FN()
    neck.convs[3] = discarded
    neck.sam2_convs[3] = discarded_sam2

    full_neck = neck(mx.array(chest_input((16, 16))))
    backbone = SAM3VLBackbone(visual=neck, text=None, scalp=1)
    output = backbone.forward_image(mx.array(chest_input((16, 16))))
    features = cast(list[mx.array], output["backbone_fpn"])
    positions = cast(list[mx.array], output["vision_pos_enc"])
    sam2_out = cast(dict[str, object], output["sam2_backbone_out"])

    assert discarded.calls == 1
    assert discarded_sam2.calls == 1
    assert [item.shape for item in features] == [
        (1, 4, 16, 16),
        (1, 4, 8, 8),
        (1, 4, 4, 4),
    ]
    _assert_array_lists_equal(features, full_neck[0][:3])
    _assert_array_lists_equal(positions, full_neck[1][:3])
    assert full_neck[2] is not None
    _assert_array_lists_equal(
        cast(list[mx.array], sam2_out["backbone_fpn"]), full_neck[2][:3]
    )


def test_dual_neck_keeps_discarded_head_checkpoint_keys() -> None:
    dual = _dual_neck()
    evaluate_boundary(*dual(mx.array(chest_input((16, 16))), scalp=1)[0])
    keys = set(flatten_model_parameters(dual))
    assert any(key.startswith("convs.3.") for key in keys)
    assert any(key.startswith("sam2_convs.3.") for key in keys)
