import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.decoder import TransformerDecoderLayer
from sam3_mlx.model.model_misc import MultiheadAttention


def _layer_norm(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5)


def test_decoder_layer_cross_attention_mask_is_batch_first_with_presence_row() -> None:
    """Each batch row uses its own box-bias rows; the presence row attends everywhere.

    Self-attention, text cross-attention and the feed-forward branch are zeroed
    and image cross-attention uses identity projections, so with zero queries
    each output is the layer-normed mean of the memory rows its mask allows.
    """
    d_model, src_len, batch = 4, 3, 2
    layer = TransformerDecoderLayer(
        d_model=d_model,
        dim_feedforward=8,
        cross_attention=MultiheadAttention(d_model, 1),
        n_heads=1,
    )
    for projection in (layer.self_attn.out_proj, layer.ca_text.out_proj, layer.linear2):
        projection.weight = mx.zeros_like(projection.weight)
        projection.bias = mx.zeros_like(projection.bias)
    zero_bias = mx.zeros((d_model,))
    cross = layer.cross_attn
    for projection in (cross.query_proj, cross.key_proj, cross.value_proj, cross.out_proj):
        projection.weight = mx.eye(d_model)
        projection.bias = zero_bias

    memory_np = np.arange(batch * src_len * d_model, dtype=np.float32).reshape(
        batch, src_len, d_model
    ) % 5
    memory = mx.array(memory_np)
    # Batch 0's query may only attend to source 1; batch 1's only to source 2.
    bias = np.full((batch, 1, 2, src_len), -np.inf, dtype=np.float32)
    bias[:, :, 0] = 0.0
    bias[0, 0, 1, 1] = 0.0
    bias[1, 0, 1, 2] = 0.0
    tokens = mx.zeros((batch, 1, d_model))

    out, presence = layer(
        tgt=tokens,
        tgt_query_pos=tokens,
        presence_token=tokens,
        memory=memory,
        memory_with_pos=memory,
        memory_text=mx.ones((batch, 2, d_model)),
        text_attention_mask=mx.zeros((batch, 2), dtype=mx.bool_),
        cross_attn_mask=mx.array(bias),
    )

    expected = np.stack([memory_np[0, 1], memory_np[1, 2]])[:, None]
    np.testing.assert_allclose(
        to_numpy(out), _layer_norm(_layer_norm(expected)), atol=1e-5, rtol=1e-5
    )
    expected_presence = memory_np.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
        to_numpy(presence),
        _layer_norm(_layer_norm(expected_presence)),
        atol=1e-5,
        rtol=1e-5,
    )
