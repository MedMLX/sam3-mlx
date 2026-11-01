from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.vitdet import (
    Attention,
    Mlp,
    ViT,
    apply_rotary_enc,
    compute_axial_cis,
    get_abs_pos,
    window_partition,
    window_unpartition,
)
from sam3_mlx.precision import apply_precision_policy


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", (mx.float32, mx.bfloat16))
@pytest.mark.parametrize("strided_channels", (False, True))
def test_rotary_metal_matches_complex_reference(
    dtype: mx.Dtype, strided_channels: bool,
) -> None:
    from sam3_mlx.model.rotary_metal import rotary_qk

    # A packed QKV projection supplies non-contiguous batch/head/token strides.
    width = 16 if strided_channels else 8
    packed = mx.reshape(mx.sin(mx.arange(2 * 6 * 3 * 3 * width)), (2, 6, 3, 3, width))
    qkv = mx.transpose(packed.astype(dtype), (2, 0, 3, 1, 4))
    q, k = qkv[0], qkv[1]
    if strided_channels:
        q, k = q[..., ::2], k[..., ::2]
    freqs = compute_axial_cis(8, 3, 2)
    expected = apply_rotary_enc(q, k, freqs)
    actual = rotary_qk(q, k, freqs)
    for reference, result in zip(expected, actual, strict=True):
        assert result.shape == reference.shape and result.dtype == dtype
        np.testing.assert_array_equal(
            to_numpy(result.astype(mx.float32)), to_numpy(reference.astype(mx.float32))
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_attention_eval_rotary_preserves_training_path_and_compilation() -> None:
    model = Attention(dim=8, num_heads=2, input_size=(2, 2), use_rope=True)
    inputs = mx.reshape(mx.sin(mx.arange(64)), (2, 2, 2, 8))
    model.train()
    reference = model(inputs)
    model.eval()
    eager = model(inputs)
    compiled = mx.compile(model)(inputs)
    for result in (eager, compiled):
        np.testing.assert_allclose(to_numpy(result), to_numpy(reference), atol=1e-6, rtol=1e-5)


def test_bf16_vit_mlp_preserves_fused_gelu_rounding() -> None:
    model = Mlp(4, hidden_features=4)
    values = np.array([[-0.75, 0.25, 1.5, -0.5]], dtype=np.float32)
    weight = np.arange(16, dtype=np.float32).reshape(4, 4) / 16 - 0.5
    bias = np.array([0.125, -0.25, 0.5, -0.125], dtype=np.float32)
    model.fc1.weight = mx.array(weight)
    model.fc1.bias = mx.array(bias)
    model.fc2.weight = mx.eye(4)
    model.fc2.bias = mx.zeros((4,))
    apply_precision_policy(model, "bf16")

    projected = values @ weight.T + bias
    expected = 0.5 * projected * (
        1 + np.tanh(np.sqrt(2 / np.pi) * (projected + 0.044715 * projected ** 3))
    )
    rounded = mx.array(expected, dtype=mx.float32).astype(mx.bfloat16).astype(mx.float32)
    actual = model(mx.array(values))
    np.testing.assert_array_equal(to_numpy(actual.astype(mx.float32)), to_numpy(rounded))


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


def test_attention_matches_independent_numpy_reference() -> None:
    dim = 8
    weights = {
        "qkv_weight": (
            np.arange(3 * dim * dim, dtype=np.float32).reshape(3 * dim, dim) - 64.0
        ) / 512.0,
        "qkv_bias": (np.arange(3 * dim, dtype=np.float32) - 8.0) / 128.0,
        "proj_weight": (
            np.arange(dim * dim, dtype=np.float32).reshape(dim, dim) - 24.0
        ) / 256.0,
        "proj_bias": (np.arange(dim, dtype=np.float32) - 2.0) / 64.0,
    }
    attention = Attention(dim=dim, num_heads=2, qkv_bias=True, use_rope=False)
    attention.qkv.weight = mx.array(weights["qkv_weight"])
    attention.qkv.bias = mx.array(weights["qkv_bias"])
    attention.proj.weight = mx.array(weights["proj_weight"])
    attention.proj.bias = mx.array(weights["proj_bias"])
    values = (np.arange(2 * 2 * 3 * dim, dtype=np.float32).reshape(2, 2, 3, dim) - 20.0) / 40.0

    output = attention(mx.array(values))
    evaluate_boundary(output)

    np.testing.assert_allclose(
        to_numpy(output),
        _attention_numpy_reference(values, num_heads=2, **weights),
        rtol=1e-5,
        atol=1e-6,
    )


def test_vit_attention_recomputes_rope_for_equal_area_grid() -> None:
    """A new grid with the same token count needs its own spatial frequencies."""
    mx.random.seed(418)
    resized = Attention(dim=16, num_heads=2, input_size=(4, 6), use_rope=True)
    native = Attention(dim=16, num_heads=2, input_size=(3, 8), use_rope=True)
    native.qkv.weight, native.qkv.bias = resized.qkv.weight, resized.qkv.bias
    native.proj.weight, native.proj.bias = resized.proj.weight, resized.proj.bias
    values = mx.random.normal((1, 3, 8, 16))

    np.testing.assert_allclose(
        to_numpy(resized(values)), to_numpy(native(values)), rtol=1e-6, atol=1e-6
    )


def test_attention_rope_rotates_x_by_column_on_non_square_grid() -> None:
    """Row-major tokens of an (H, W) grid rotate by column (x) and row (y)."""
    height, width, head_dim = 2, 3, 8
    attention = Attention(
        dim=2 * head_dim, num_heads=2, input_size=(height, width), use_rope=True
    )
    assert attention.freqs_cis is not None

    base = 1.0 / 10000.0 ** (np.arange(0, head_dim, 4)[: head_dim // 4] / head_dim)
    rows, cols = np.divmod(np.arange(height * width), width)
    angles = np.concatenate([np.outer(cols, base), np.outer(rows, base)], axis=-1)
    np.testing.assert_allclose(
        to_numpy(mx.real(attention.freqs_cis)), np.cos(angles), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        to_numpy(mx.imag(attention.freqs_cis)), np.sin(angles), rtol=0, atol=1e-6
    )

    # Frequencies built on demand for a new grid follow the same convention.
    transposed = Attention(
        dim=2 * head_dim, num_heads=2, input_size=(width, height), use_rope=True
    )
    transposed.qkv.weight, transposed.qkv.bias = attention.qkv.weight, attention.qkv.bias
    transposed.proj.weight, transposed.proj.bias = attention.proj.weight, attention.proj.bias
    values = mx.reshape(mx.sin(mx.arange(height * width * 2 * head_dim)), (1, height, width, -1))
    np.testing.assert_allclose(
        to_numpy(transposed(values)), to_numpy(attention(values)), rtol=1e-6, atol=1e-6
    )


def test_vit_persistent_windows_match_per_block_partition() -> None:
    model = ViT(
        img_size=8,
        patch_size=2,
        embed_dim=8,
        depth=4,
        num_heads=2,
        mlp_ratio=1.0,
        global_att_blocks=(1, 3),
        window_size=2,
        pretrain_use_cls_token=False,
        use_abs_pos=False,
        use_rope=False,
    )
    samples = mx.reshape(mx.arange(1 * 3 * 8 * 8, dtype=mx.float32), (1, 3, 8, 8))

    # The 4x4 token grid tiles exactly into 2x2 windows, so the trunk keeps
    # them partitioned across consecutive windowed blocks.
    per_block = model.ln_pre(model.patch_embed(samples))
    for block in model.blocks:
        per_block = block(per_block)
    persisted = model(samples)
    evaluate_boundary(per_block, persisted)

    assert persisted.shape == (1, 4, 4, 8)
    np.testing.assert_allclose(to_numpy(persisted), to_numpy(per_block), rtol=1e-6, atol=1e-6)


def test_window_partition_unpartition_roundtrip_with_padding() -> None:
    x = mx.reshape(mx.arange(2 * 5 * 5 * 3, dtype=mx.float32), (2, 5, 5, 3))
    windows, pad_hw = window_partition(x, window_size=2)

    assert pad_hw == (6, 6)
    assert windows.shape == (2 * 3 * 3, 2, 2, 3)

    restored = window_unpartition(windows, window_size=2, pad_hw=pad_hw, hw=(5, 5))
    evaluate_boundary(restored)
    assert restored.shape == x.shape
    assert bool(mx.allclose(restored, x).item())


def test_get_abs_pos_tiles_grid_and_drops_class_token() -> None:
    spatial = np.arange(1 * 4 * 2, dtype=np.float32).reshape(1, 4, 2)
    expected = np.tile(spatial.reshape(1, 2, 2, 2), (1, 2, 3, 1))[:, :3, :5]

    no_cls = get_abs_pos(mx.array(spatial), has_cls_token=False, hw=(3, 5))
    with_cls = get_abs_pos(
        mx.array(np.concatenate([np.full((1, 1, 2), 100.0, np.float32), spatial], axis=1)),
        has_cls_token=True,
        hw=(3, 5),
    )

    np.testing.assert_array_equal(to_numpy(no_cls), expected)
    np.testing.assert_array_equal(to_numpy(with_cls), expected)
