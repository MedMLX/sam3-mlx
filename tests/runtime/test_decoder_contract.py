from typing import cast

import pytest
import mlx.core as mx

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv1,
    TransformerDecoderLayerv2,
    nn,
)
from sam3_mlx.model.model_misc import MultiheadAttentionWrapper, TransformerWrapper


class _FailingMultiheadAttention(MultiheadAttentionWrapper):
    def __init__(self) -> None:
        super().__init__(4, 1)
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> mx.array:
        del args, kwargs
        self.calls += 1
        raise TypeError("attention kernel failure")


def test_decoder_does_not_retry_internal_attention_type_error() -> None:
    self_attention = _FailingMultiheadAttention()
    layer = TransformerDecoderLayerv1(
        activation="relu",
        cross_attention=MultiheadAttentionWrapper(4, 1),
        d_model=4,
        dim_feedforward=8,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=False,
        self_attention=self_attention,
    )
    values = mx.zeros((1, 1, 4))

    with pytest.raises(TypeError, match="^attention kernel failure$"):
        layer(values, values)

    assert self_attention.calls == 1


def test_transformer_wrapper_requires_decoder_num_queries() -> None:
    with pytest.raises(AttributeError, match="num_queries"):
        TransformerWrapper(encoder=None, decoder=object(), d_model=4)


class _DummyCrossAttention:
    num_heads = 2


class _DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_attn = _DummyCrossAttention()
        self.cross_attn_image = _DummyCrossAttention()
        self.proj = nn.Linear(4, 4)

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("decoder constructor test must not execute a layer")


class _IdentityRope(nn.Module):
    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        del k, v, num_k_exclude_rope
        return q


class _ZeroArray(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.zeros_like(x)


def test_transformer_decoder_rejects_unknown_boxrpb_without_instance_query() -> None:
    with pytest.raises(AssertionError):
        TransformerDecoder(
            d_model=4,
            interaction_layer=None,
            layer=_DummyLayer(),
            num_layers=1,
            num_queries=1,
            return_intermediate=True,
            box_refine=True,
            boxRPB="cubic",
            instance_query=False,
        )


def test_decoupled_decoder_layer_returns_image_and_target() -> None:
    layer = DecoupledTransformerDecoderLayerv2(
        activation="relu",
        d_model=4,
        num_heads=1,
        dim_feedforward=8,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention_rope=_IdentityRope(),
        cross_attention_rope=_IdentityRope(),
    )
    layer.self_attn_out_proj = _ZeroArray()
    layer.cross_attn_out_proj = _ZeroArray()
    layer.linear1 = _ZeroArray()
    layer.linear2 = _ZeroArray()
    image = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    target = mx.array([[[9.0, 8.0, 7.0, 6.0]]])

    image_out, target_out = layer(
        image=image,
        tgt=target,
        memory_image=image,
        memory=target,
    )

    assert bool(mx.array_equal(image_out, image))
    assert not bool(mx.array_equal(image_out, target))
    assert bool(mx.array_equal(target_out, target))
    assert not bool(mx.array_equal(target_out, image))


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


def _transformer_decoder_layer_v2(
    d_model: int = 4, n_heads: int = 1
) -> TransformerDecoderLayerv2:
    return TransformerDecoderLayerv2(
        activation="relu",
        cross_attention=MultiheadAttentionWrapper(d_model, n_heads),
        d_model=d_model,
        dim_feedforward=8,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttentionWrapper(d_model, n_heads),
    )


def test_decoder_layer_v2_rejects_dac_true() -> None:
    """Regression test for a dead ``assert False is False``.

    Upstream's ``TransformerDecoderLayerv2.forward_pre`` asserts ``dac is
    False``; this port never ported the ``dac`` (decoupled self-attention)
    parameter, so the assert had degenerated into an always-true tautology.
    A stray ``dac=True`` passed through the layer's compatibility ``**kwargs``
    should still fail loudly instead of being silently ignored.
    """
    layer = _transformer_decoder_layer_v2()
    tgt = mx.zeros((1, 1, 4))
    memory = mx.zeros((1, 1, 4))

    with pytest.raises(Sam3MlxUnsupportedError, match="dac=True"):
        layer.forward_pre(tgt, memory, dac=True)


def test_decoder_layer_v2_forward_pre_runs_without_dac_kwarg() -> None:
    layer = _transformer_decoder_layer_v2()
    tgt = mx.zeros((1, 1, 4))
    memory = mx.zeros((1, 1, 4))

    out_without_kwarg = layer.forward_pre(tgt, memory)
    out_with_false = layer.forward_pre(tgt, memory, dac=False)

    assert out_without_kwarg.shape == (1, 1, 4)
    assert out_with_false.shape == (1, 1, 4)


def test_decoder_layer_cross_attention_mask_is_laid_out_batch_first() -> None:
    """Regression test for a ``mx.transpose(mask, (0, 1))`` no-op.

    The encoder hands the decoder a ``memory_key_padding_mask`` shaped
    ``(src_len, B)``; cross-attention needs ``(B, src_len)``. Transposing with
    axes ``(0, 1)`` is a no-op on a 2-D array (unlike PyTorch's
    ``tensor.transpose(0, 1)``, which always swaps the two axes), so the
    mismatched shape used to reach ``MultiheadAttentionWrapper`` unchanged. With
    ``src_len != B`` that raises immediately; this also checks that, once fixed,
    each batch row gets its own mask row rather than a scrambled one.
    """
    d_model = 4
    src_len = 3
    batch = 2

    layer = TransformerDecoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=8,
        cross_attention=MultiheadAttentionWrapper(d_model, 1),
        n_heads=1,
    )
    # Neutralize everything except cross-attention so the layer's output is
    # exactly the cross-attention result: zero self-attention's contribution,
    # make cross-attention's projections the identity, and make the FFN and
    # norms pass-through. layer.self_attn/cross_attn/linear2 are statically
    # typed as narrower protocols than the concrete modules built internally,
    # so cast back to the concrete types to set their parameters.
    self_attn = cast(MultiheadAttentionWrapper, layer.self_attn)
    self_attn.out_proj.weight = mx.zeros((d_model, d_model))
    self_attn.out_proj.bias = mx.zeros((d_model,))
    identity = mx.eye(d_model)
    zero_bias = mx.zeros((d_model,))
    cross_attn = cast(MultiheadAttentionWrapper, layer.cross_attn)
    cross_attn.query_proj.weight = identity
    cross_attn.query_proj.bias = zero_bias
    cross_attn.key_proj.weight = identity
    cross_attn.key_proj.bias = zero_bias
    cross_attn.value_proj.weight = identity
    cross_attn.value_proj.bias = zero_bias
    cross_attn.out_proj.weight = identity
    cross_attn.out_proj.bias = zero_bias
    # Both replaced (not just linear2) so the shape-preserving zero function
    # never has to cross the dim_feedforward-widened intermediate shape.
    layer.linear1 = _ZeroArray()
    layer.linear2 = _ZeroArray()
    layer.norm1 = _Identity()
    layer.norm2 = _Identity()
    layer.norm3 = _Identity()

    tgt = mx.zeros((1, batch, d_model))
    memory = mx.reshape(
        mx.arange(src_len * batch * d_model, dtype=mx.float32),
        (src_len, batch, d_model),
    )
    # Batch 0 may only attend to source position 1; batch 1 only to position 2.
    memory_key_padding_mask = mx.array(
        [
            [True, True],
            [False, True],
            [True, False],
        ]
    )

    out, presence = layer(
        tgt,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    assert presence is None
    assert out.shape == (1, batch, d_model)
    expected = mx.stack([memory[1, 0], memory[2, 1]])[None]
    assert bool(mx.array_equal(out, expected))
