from __future__ import annotations

from typing import cast
from sam3_mlx.checkpoint_mapping import flatten_model_parameters

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.vitdet import (
    Attention,
    ViT,
    get_abs_pos,
    window_partition,
    window_unpartition,
)
from tests._mlx_runtime import as_nested
from sam3_mlx.mlx_runtime import evaluate_boundary


def _attention_numpy_reference(
    values: np.ndarray,
    *,
    num_heads: int,
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray,
    proj_weight: np.ndarray,
    proj_bias: np.ndarray,
) -> np.ndarray:
    original_shape = values.shape
    batch = original_shape[0]
    embedding_dim = original_shape[-1]
    tokens = values.reshape(batch, -1, embedding_dim)
    qkv = tokens @ qkv_weight.T + qkv_bias
    head_dim = embedding_dim // num_heads
    qkv = qkv.reshape(batch, tokens.shape[1], 3, num_heads, head_dim)
    q, k, v = qkv.transpose(2, 0, 3, 1, 4)
    logits = (q @ k.transpose(0, 1, 3, 2)) * (head_dim**-0.5)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    attended = probabilities @ v
    merged = attended.transpose(0, 2, 1, 3).reshape(tokens.shape)
    projected = merged @ proj_weight.T + proj_bias
    return projected.reshape(original_shape)


def _deterministic_attention(
    *, cls_token: bool
) -> tuple[Attention, dict[str, np.ndarray]]:
    dim = 8
    qkv_weight = (
        np.arange(3 * dim * dim, dtype=np.float32).reshape(3 * dim, dim) - 64.0
    ) / 512.0
    qkv_bias = (np.arange(3 * dim, dtype=np.float32) - 8.0) / 128.0
    proj_weight = (
        np.arange(dim * dim, dtype=np.float32).reshape(dim, dim) - 24.0
    ) / 256.0
    proj_bias = (np.arange(dim, dtype=np.float32) - 2.0) / 64.0
    attention = Attention(
        dim=dim,
        num_heads=2,
        qkv_bias=True,
        cls_token=cls_token,
        use_rel_pos=False,
        use_rope=False,
    )
    attention.qkv.weight = mx.array(qkv_weight)
    attention.qkv.bias = mx.array(qkv_bias)
    attention.proj.weight = mx.array(proj_weight)
    attention.proj.bias = mx.array(proj_bias)
    return attention, {
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "proj_weight": proj_weight,
        "proj_bias": proj_bias,
    }


def test_vit_attention_recomputes_rope_for_smaller_global_grid() -> None:
    attention = Attention(
        dim=8,
        num_heads=2,
        input_size=(4, 4),
        use_rope=True,
        rope_pt_size=(2, 2),
        rope_interp=True,
        cls_token=False,
    )
    x = mx.ones((1, 2, 2, 8), dtype=mx.float32)

    out = attention(x)
    evaluate_boundary(out)

    assert out.shape == (1, 2, 2, 8)
    cache = cast(
        dict[tuple[int, int], mx.array], getattr(attention, "_freqs_cis_cache")
    )
    assert (2, 2) in cache
    assert cache[(2, 2)].shape == (4, 2)


def test_attention_token_and_spatial_layouts_match_independent_numpy_reference() -> (
    None
):
    token_values = (
        np.arange(2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8) - 20.0
    ) / 40.0
    token_attention, token_weights = _deterministic_attention(cls_token=True)
    token_output = token_attention(mx.array(token_values))

    spatial_values = token_values[:, :4].reshape(2, 2, 2, 8)
    spatial_attention, spatial_weights = _deterministic_attention(cls_token=False)
    spatial_output = spatial_attention(mx.array(spatial_values))
    evaluate_boundary(token_output, spatial_output)

    np.testing.assert_allclose(
        to_numpy(token_output),
        _attention_numpy_reference(token_values, num_heads=2, **token_weights),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(spatial_output),
        _attention_numpy_reference(spatial_values, num_heads=2, **spatial_weights),
        rtol=1e-5,
        atol=1e-6,
    )


def test_rel_pos_blocks_false_disables_all_blocks() -> None:
    vit = ViT(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=3,
        num_heads=2,
        rel_pos_blocks=False,
        global_att_blocks=(2,),
        window_size=0,
        retain_cls_token=False,
        pretrain_use_cls_token=False,
        use_abs_pos=True,
        tile_abs_pos=False,
        use_rope=False,
    )

    assert vit.rel_pos_blocks == [False, False, False]
    assert all(not block.attn.use_rel_pos for block in vit.blocks)

    out = vit(mx.ones((1, 3, 32, 32), dtype=mx.float32))
    feat = out[0]
    assert isinstance(feat, mx.array)
    evaluate_boundary(feat)
    assert len(out) == 1
    assert feat.shape == (1, 8, 2, 2)


def test_exact_window_persist_matches_per_block_partition() -> None:
    def model(persist_exact_windows: bool) -> ViT:
        return ViT(
            img_size=8,
            patch_size=2,
            embed_dim=8,
            depth=4,
            num_heads=2,
            mlp_ratio=1.0,
            rel_pos_blocks=(),
            global_att_blocks=(1, 3),
            window_size=2,
            retain_cls_token=False,
            pretrain_use_cls_token=False,
            use_abs_pos=False,
            tile_abs_pos=False,
            use_rope=False,
            persist_exact_windows=persist_exact_windows,
        )

    per_block = model(False)
    persisted = model(True)
    persisted.load_weights(list(flatten_model_parameters(per_block).items()))
    samples = mx.reshape(mx.arange(1 * 3 * 8 * 8, dtype=mx.float32), (1, 3, 8, 8))

    per_block_out = per_block(samples)[0]
    persisted_out = persisted(samples)[0]
    assert isinstance(per_block_out, mx.array)
    assert isinstance(persisted_out, mx.array)
    evaluate_boundary(per_block_out, persisted_out)

    assert per_block_out.shape == (1, 8, 4, 4)
    assert bool(mx.allclose(per_block_out, persisted_out).item())


def test_window_partition_unpartition_roundtrip_with_padding() -> None:
    x = mx.reshape(mx.arange(2 * 5 * 5 * 3, dtype=mx.float32), (2, 5, 5, 3))
    windows, pad_hw = window_partition(x, window_size=2)

    assert pad_hw == (6, 6)
    assert windows.shape == (2 * 3 * 3, 2, 2, 3)

    restored = window_unpartition(windows, window_size=2, pad_hw=pad_hw, hw=(5, 5))
    evaluate_boundary(restored)
    assert restored.shape == x.shape
    assert bool(mx.allclose(restored, x).item())


def test_get_abs_pos_with_and_without_class_token() -> None:
    abs_pos = mx.reshape(mx.arange(1 * 4 * 2, dtype=mx.float32), (1, 4, 2))
    no_cls = get_abs_pos(abs_pos, has_cls_token=False, hw=(3, 3), tiling=True)
    assert no_cls.shape == (1, 3, 3, 2)

    abs_pos_cls = mx.reshape(mx.arange(1 * 5 * 2, dtype=mx.float32), (1, 5, 2))
    with_cls = get_abs_pos(
        abs_pos_cls,
        has_cls_token=True,
        hw=(3, 3),
        retain_cls_token=True,
        tiling=True,
    )
    assert with_cls.shape == (1, 1 + 3 * 3, 2)
    assert bool(mx.allclose(with_cls[:, :1], abs_pos_cls[:, :1]).item())


def test_attention_rope_theta_attribute_is_spelled_correctly_and_plumbed_through() -> (
    None
):
    """Regression test for a typo (``rope_theata``) in the stored attribute name.

    The constructor keyword is ``rope_theta`` and upstream SAM3 stores it as
    ``self.rope_theta``; a stray ``self.rope_theata`` would silently diverge
    from that name. Guard both the attribute name and that a non-default
    value actually reaches ``compute_axial_cis``.
    """
    # head_dim = dim // num_heads must exceed 4 so compute_axial_cis derives
    # more than the always-theta-independent zero-exponent frequency term.
    default_attention = Attention(
        dim=32,
        num_heads=2,
        input_size=(2, 2),
        use_rope=True,
        cls_token=False,
    )
    custom_attention = Attention(
        dim=32,
        num_heads=2,
        input_size=(2, 2),
        use_rope=True,
        rope_theta=123.0,
        cls_token=False,
    )

    assert not hasattr(default_attention, "rope_theata")
    assert default_attention.rope_theta == 10000.0
    assert custom_attention.rope_theta == 123.0

    x = mx.ones((1, 2, 2, 32), dtype=mx.float32)
    default_attention(x)
    custom_attention(x)
    evaluate_boundary(default_attention.freqs_cis, custom_attention.freqs_cis)

    assert default_attention.freqs_cis is not None
    assert custom_attention.freqs_cis is not None
    assert not bool(
        mx.allclose(default_attention.freqs_cis, custom_attention.freqs_cis).item()
    )


def test_relative_position_attention_executes() -> None:
    attention = Attention(
        dim=8,
        num_heads=2,
        input_size=(2, 2),
        use_rel_pos=True,
        rel_pos_zero_init=True,
        cls_token=False,
        use_rope=False,
    )
    assert attention.rel_pos_h is not None
    assert attention.rel_pos_w is not None

    x = mx.ones((1, 2, 2, 8), dtype=mx.float32)
    out = attention(x)
    evaluate_boundary(out)
    assert out.shape == (1, 2, 2, 8)


def test_vit_accepts_plain_array_and_nested_tensor() -> None:
    vit = ViT(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=2,
        num_heads=2,
        rel_pos_blocks=(),
        global_att_blocks=(1,),
        window_size=0,
        retain_cls_token=False,
        pretrain_use_cls_token=False,
        use_abs_pos=True,
        tile_abs_pos=False,
        use_rope=False,
    )
    plain = mx.ones((1, 3, 32, 32), dtype=mx.float32)
    plain_out = vit(plain)
    plain_feat = plain_out[0]
    assert isinstance(plain_feat, mx.array)
    evaluate_boundary(plain_feat)
    assert plain_feat.shape == (1, 8, 2, 2)

    mask = mx.zeros((1, 32, 32), dtype=mx.bool_)
    nested = NestedTensor(plain, mask)
    nested_out = vit(nested)
    nested_feat = nested_out[0]
    assert isinstance(nested_feat, NestedTensor)
    nested_view = as_nested(nested_feat)
    evaluate_boundary(nested_view.tensors)
    assert nested_view.tensors.shape == (1, 8, 2, 2)
    assert nested_view.mask is not None
    assert nested_view.mask.shape[-2:] == (2, 2)
