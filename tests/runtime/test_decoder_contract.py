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


def test_final_decoder_output_preserves_refinement_references() -> None:
    """Keep final queries, entering boxes and presence equal to the diagnostic layer stack."""
    from sam3_mlx.model.decoder import TransformerDecoder

    mx.random.seed(975)
    decoder = TransformerDecoder(
        layer=lambda: TransformerDecoderLayer(
            d_model=8, dim_feedforward=16,
            cross_attention=MultiheadAttention(8, 2), n_heads=2,
        ),
        num_layers=3, num_queries=5, d_model=8,
    )
    arguments = (
        mx.random.normal((2, 5, 8)),
        mx.random.normal((2, 6, 8)),
        mx.random.normal((2, 6, 8)),
        mx.random.normal((4, 2, 8)),
        mx.zeros((2, 4), dtype=mx.bool_),
    )
    full = decoder(*arguments, feat_size=(2, 3))
    final = decoder(*arguments, feat_size=(2, 3), return_intermediate=False)
    for index in range(3):
        np.testing.assert_array_equal(to_numpy(final[index][0]), to_numpy(full[index][-1]))
    np.testing.assert_array_equal(to_numpy(final[3]), to_numpy(full[3]))
