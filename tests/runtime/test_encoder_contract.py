import mlx.core as mx

from sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3_mlx.model.model_misc import MultiheadAttention


def _encoder() -> TransformerEncoderFusion:
    mx.random.seed(3)

    def layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            d_model=8,
            dim_feedforward=16,
            self_attention=MultiheadAttention(8, 2),
            cross_attention=MultiheadAttention(8, 2),
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
        )

    return TransformerEncoderFusion(layer=layer, num_layers=2)


def test_encoder_returns_batch_first_memory_and_ignores_prompt_padding() -> None:
    encoder = _encoder()
    hw, batch, length = 4, 2, 3
    src = mx.reshape(mx.sin(mx.arange(hw * batch * 8, dtype=mx.float32)), (hw, batch, 8))
    src_pos = mx.reshape(mx.cos(mx.arange(hw * batch * 8, dtype=mx.float32)), (hw, batch, 8))
    prompt = mx.reshape(mx.linspace(-1, 1, length * batch * 8), (length, batch, 8))
    mask = mx.array([[False, False, True], [False, True, True]])

    out = encoder(src=src, src_pos=src_pos, prompt=prompt, prompt_key_padding_mask=mask)
    # Padded tokens get -inf scores, so their content must not change the result.
    padded = mx.where(mx.transpose(mask)[..., None], 100.0, prompt)
    out_padded = encoder(
        src=src, src_pos=src_pos, prompt=padded, prompt_key_padding_mask=mask
    )

    assert out["memory"].shape == (batch, hw, 8)
    assert bool(mx.array_equal(out["pos_embed"], mx.transpose(src_pos, (1, 0, 2))))
    assert bool(mx.array_equal(out["memory"], out_padded["memory"]))
    assert not bool(mx.array_equal(out["memory"][0], out["memory"][1]))
