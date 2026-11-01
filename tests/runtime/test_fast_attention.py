"""Attention numerics against an independent NumPy reference of PyTorch MHA."""

import math

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.model_misc import MultiheadAttention
from sam3_mlx.precision_layers import attention

type FloatArray = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]

_HEADS = 2


def _softmax(value: FloatArray, axis: int = -1) -> FloatArray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return to_numpy(exp / np.sum(exp, axis=axis, keepdims=True), dtype=np.float32)


def _split_heads(batch_first: FloatArray) -> FloatArray:
    batch_size, seq_len, embed_dim = batch_first.shape
    return batch_first.reshape(
        batch_size, seq_len, _HEADS, embed_dim // _HEADS
    ).transpose(0, 2, 1, 3)


def _mha_reference(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    key_padding_mask: BoolArray | None = None,
    attn_mask: FloatArray | None = None,
) -> FloatArray:
    """Batch-first MHA with identity projections; ``attn_mask`` is ``(B, H, T, S)``."""
    q, k, v = _split_heads(query), _split_heads(key), _split_heads(value)
    scores = q @ np.swapaxes(k, -1, -2) * (q.shape[-1] ** -0.5)
    if key_padding_mask is not None:
        scores = np.where(key_padding_mask[:, None, None, :], -np.inf, scores)
    if attn_mask is not None:
        scores = scores + attn_mask
    out = _softmax(scores, axis=-1) @ v
    batch_size, seq_len, embed_dim = query.shape
    return out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)


def _identity_attention(
    biases: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None,
) -> MultiheadAttention:
    module = MultiheadAttention(4, _HEADS)
    projections = (module.query_proj, module.key_proj, module.value_proj, module.out_proj)
    for index, projection in enumerate(projections):
        projection.weight = mx.eye(4, dtype=mx.float32)
        projection.bias = (
            mx.zeros((4,), dtype=mx.float32) if biases is None else mx.array(biases[index])
        )
    return module


def _inputs() -> tuple[FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed=0)
    return (
        rng.standard_normal((2, 2, 4)).astype(np.float32),
        rng.standard_normal((2, 3, 4)).astype(np.float32),
        rng.standard_normal((2, 3, 4)).astype(np.float32),
    )


def test_key_padding_mask_is_applied_per_batch_row() -> None:
    query, key, value = _inputs()
    key_padding_mask = np.array([[False, False, True], [True, False, False]])
    expected = _mha_reference(query, key, value, key_padding_mask=key_padding_mask)

    out = _identity_attention()(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        key_padding_mask=mx.array(key_padding_mask),
    )

    out_np = to_numpy(out)
    np.testing.assert_allclose(out_np, expected, atol=1e-6, rtol=1e-6)
    # Rows with different masks must differ, or collapsed broadcasting would pass.
    assert not np.allclose(out_np[0], out_np[1], atol=1e-3)


@pytest.mark.parametrize("flat_heads", (True, False), ids=("BH-T-S", "B-H-T-S"))
def test_additive_attn_mask_maps_to_batch_and_head(flat_heads: bool) -> None:
    query, key, value = _inputs()
    rng = np.random.default_rng(seed=1)
    bias = rng.standard_normal((2, _HEADS, 2, 3)).astype(np.float32)
    bias[1, 0, :, 0] = -np.inf
    expected = _mha_reference(query, key, value, attn_mask=bias)

    attn_mask = bias.reshape(2 * _HEADS, 2, 3) if flat_heads else bias
    out = _identity_attention()(
        mx.array(query), mx.array(key), mx.array(value), attn_mask=mx.array(attn_mask)
    )

    np.testing.assert_allclose(to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_projection_biases_are_applied_separately() -> None:
    query, key, value = _inputs()
    q_bias = np.array([0.05, -0.02, 0.03, 0.01], dtype=np.float32)
    k_bias = np.array([-0.01, 0.04, -0.03, 0.02], dtype=np.float32)
    v_bias = np.array([0.07, -0.05, 0.02, -0.04], dtype=np.float32)
    out_bias = np.array([0.11, -0.09, 0.03, -0.02], dtype=np.float32)
    expected = _mha_reference(query + q_bias, key + k_bias, value + v_bias) + out_bias

    out = _identity_attention((q_bias, k_bias, v_bias, out_bias))(
        mx.array(query), mx.array(key), mx.array(value)
    )

    np.testing.assert_allclose(to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_rejects_both_masks() -> None:
    query, key, value = (mx.array(x) for x in _inputs())
    with pytest.raises(ValueError, match="not both"):
        _identity_attention()(
            query,
            key,
            value,
            key_padding_mask=mx.zeros((2, 3), dtype=mx.bool_),
            attn_mask=mx.zeros((2 * _HEADS, 2, 3)),
        )


def test_bf16_attention_accumulates_scaled_scores_and_bias_in_float32() -> None:
    query = (np.arange(64, dtype=np.float32).reshape(1, 1, 2, 32) % 17 - 8) / 8
    key = (np.arange(160, dtype=np.float32).reshape(1, 1, 5, 32) % 23 - 11) / 8
    value = (np.arange(160, dtype=np.float32).reshape(1, 1, 5, 32) % 19 - 9) / 8
    bias = np.array([0, -0.5, -2, -4, -8], dtype=np.float32).reshape(1, 1, 1, 5)
    expected_scores = np.asarray(
        query @ np.swapaxes(key, -1, -2) / np.float32(math.sqrt(32)) + bias,
        dtype=np.float32,
    )
    expected = _softmax(expected_scores) @ value
    expected = mx.array(expected).astype(mx.bfloat16).astype(mx.float32)

    output = attention(
        mx.array(query, dtype=mx.bfloat16),
        mx.array(key, dtype=mx.bfloat16),
        mx.array(value, dtype=mx.bfloat16),
        scale=32**-0.5,
        mask=mx.array(bias, dtype=mx.bfloat16),
    )

    assert output.dtype == mx.bfloat16
    np.testing.assert_array_equal(to_numpy(output.astype(mx.float32)), to_numpy(expected))
