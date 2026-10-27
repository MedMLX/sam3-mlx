# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""CLIP-style text transformer used as SAM 3's text encoder (VE text encoder).

Text is tokenized to ``(B, L)`` int ids with 0 as padding. Token features are
batch-first inside the transformer and returned sequence-first (``(L, B, C)``)
for the decoder, matching the upstream PyTorch convention.
"""

from collections.abc import Callable
from typing import Protocol, TypedDict, cast

import mlx.core as mx
from mlx import nn
from sam3_mlx.precision_layers import MultiHeadAttention

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.model_misc import LayerScale
from sam3_mlx.sam.tensor_protocols import ActivationFactory, ArrayModule


class _AttentionModule(Protocol):
    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        mask: mx.array | None = None,
    ) -> mx.array: ...


class _Tokenizer(Protocol):
    def __call__(
        self,
        texts: str | list[str],
        context_length: int | None = None,
    ) -> mx.array: ...


class _PreencodedTokens(TypedDict):
    inputs_embeds: mx.array


# Output of a previous VETextEncoder call, as accepted by VETextEncoder.__call__:
# (padding mask (B, L), resized text memory (L, B, d_model),
# {"inputs_embeds": token embeddings (B, L, width)}).
PreencodedText = tuple[mx.array, mx.array, _PreencodedTokens]
NormFactory = Callable[[int], ArrayModule]


class MLP(nn.Module):
    """Two-layer feed-forward block (``c_fc``, activation, ``c_proj``) over the last axis."""

    def __init__(
        self,
        d_model: int,
        mlp_width: int,
        act_layer: ActivationFactory = nn.GELU,
    ) -> None:
        super().__init__()
        self.c_fc = cast(ArrayModule, nn.Linear(d_model, mlp_width))
        self.gelu = act_layer()
        self.c_proj = cast(ArrayModule, nn.Linear(mlp_width, d_model))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    """Pre-norm transformer block: self-attention then MLP, each with a residual.

    Operates on batch-first ``(B, L, d_model)`` input. Optional LayerScale is
    applied to both residual branches when ``ls_init_value`` is set.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: ActivationFactory = nn.GELU,
        norm_layer: NormFactory = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.attn = cast(
            _AttentionModule,
            MultiHeadAttention(d_model, n_head, bias=True),
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = MLP(d_model, mlp_width, act_layer=act_layer)

    def attention(
        self,
        q_x: mx.array,
        k_x: mx.array | None = None,
        v_x: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        """Attend from ``q_x`` to ``k_x``/``v_x`` (default: ``q_x``).

        ``attn_mask`` is either boolean or additive (``-inf`` blocks); non-boolean
        masks are cast to the query dtype.
        """
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None:
            # Preserve boolean masks without conversion.
            if not attn_mask.dtype == mx.bool_:
                attn_mask = attn_mask.astype(q_x.dtype)

        return self.attn(q_x, k_x, v_x, mask=attn_mask)

    def __call__(
        self,
        q_x: mx.array,
        k_x: mx.array | None = None,
        v_x: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    """Stack of ``layers`` residual attention blocks over batch-first ``(B, L, width)``."""

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: ActivationFactory = nn.GELU,
        norm_layer: NormFactory = nn.LayerNorm,
        compile_mode: str | None = None,
    ) -> None:
        super().__init__()
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.text_encoder_ve.Transformer(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.width = width
        self.layers = layers
        self.resblocks = [
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for _ in range(layers)
        ]

    def __call__(
        self,
        x: mx.array,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        for _, r in enumerate(self.resblocks):
            x = r(
                x,
                attn_mask=attn_mask,
            )
        return x


def text_global_pool(
    x: mx.array,
    text: mx.array | None = None,
    pool_type: str = "argmax",
) -> tuple[mx.array, mx.array]:
    """Select a pooled vector and the remaining token features from ``x`` ``(B, L, C)``.

    ``"first"``/``"last"`` take that position as the pooled vector and drop it from
    the tokens; ``"argmax"`` takes the position of the largest token id in ``text``
    (the end-of-text token) and keeps all tokens; any other value returns ``x`` for
    both. ``text`` is the ``(B, L)`` id array and is required for ``"argmax"``.
    """
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # Use the highest token ID as the end-of-text position.
        assert text is not None
        pooled, tokens = x[mx.arange(x.shape[0]), mx.argmax(text, axis=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    """CLIP-style causal text transformer.

    ``positional_embedding`` (and ``text_projection`` when ``proj_bias`` is False)
    start as zeros and are expected to be filled by checkpoint loading.
    """

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",  # no pooling
        proj_bias: bool = False,
        act_layer: ActivationFactory = nn.GELU,
        norm_layer: NormFactory = nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        compile_mode: str | None = None,
    ) -> None:
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.token_embedding = cast(
            ArrayModule,
            nn.Embedding(self.vocab_size, width),
        )
        self.positional_embedding = mx.zeros((self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            compile_mode=compile_mode,
        )
        self.ln_final = (
            norm_layer(width) if use_ln_post else cast(ArrayModule, nn.Identity())
        )
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.attn_mask = self.build_causal_mask()

        if proj_bias:
            self.text_projection: ArrayModule | mx.array = cast(
                ArrayModule,
                nn.Linear(width, output_dim),
            )
        else:
            self.text_projection = mx.zeros((width, output_dim))

    def build_causal_mask(self) -> mx.array:
        """Return the ``(context_length, context_length)`` additive causal mask."""
        # Build an additive causal mask with full attention on and below the diagonal.
        mask = mx.full((self.num_pos, self.num_pos), -float("inf"))
        mask = mx.triu(mask, k=1)
        return mask

    def __call__(self, text: mx.array) -> mx.array | tuple[mx.array, mx.array]:
        """Encode ``(B, L)`` token ids with ``L <= context_length``.

        Returns the projected pooled output, or ``(pooled, tokens)`` when
        ``output_tokens`` is set. Note that ``tokens`` are the final-layer-norm
        features before ``text_projection``.
        """
        seq_len = text.shape[1]
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if isinstance(self.text_projection, mx.array):
            pooled = pooled @ self.text_projection
        else:
            pooled = self.text_projection(pooled)
        if self.output_tokens:
            return pooled, tokens
        return pooled


class VETextEncoder(nn.Module):
    """Text encoder for SAM 3 prompts: tokenizer, ``TextTransformer`` and width resizer.

    Args:
        d_model: Width of the returned text memory (the decoder width).
        tokenizer: Callable ``(texts, context_length=...)`` returning ``(B, L)``
            int token ids, with 0 for padding.
        width: Transformer width.
        heads: Attention heads.
        layers: Transformer depth.
        context_length: Token length (prompts are padded or truncated by the
            tokenizer to this).
        vocab_size: Token embedding rows.
        use_ln_post: Apply a final LayerNorm.
        compile_mode: Must be ``None``; ``torch.compile`` is not supported.
    """

    def __init__(
        self,
        d_model: int,
        tokenizer: _Tokenizer,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: str | None = None,
    ) -> None:
        super().__init__()
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.text_encoder_ve.VETextEncoder(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            # Return token features, not only the pooled output.
            output_tokens=True,
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
        )
        self.resizer = cast(ArrayModule, nn.Linear(self.encoder.width, d_model))

    def __call__(
        self,
        text: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Encode prompts, or reuse a previous encoding.

        Args:
            text: A non-empty list of prompt strings, or a ``PreencodedText``
                tuple from an earlier call.
            input_boxes: Unsupported; must be ``None`` or empty.

        Returns:
            ``(attention_mask, memory, embeds)``. ``attention_mask`` is a
            ``(B, L)`` bool array that is True at padding positions (the PyTorch
            ``key_padding_mask`` convention). ``memory`` is ``(L, B, d_model)``
            and ``embeds`` is the ``(L, B, width)`` token embedding, both
            sequence-first.

        Raises:
            ValueError: If ``text`` is an empty list.
        """
        if isinstance(text, list):
            if not text:
                raise ValueError("text must contain at least one prompt")
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            tokenized = self.tokenizer(
                text, context_length=self.context_length
            )  # [b, seq_len]
            text_attention_mask = (tokenized != 0).astype(mx.bool_)

            # Keep token embeddings for the public return value.
            inputs_embeds = self.encoder.token_embedding(
                tokenized
            )  # [b, seq_len, d=1024]
            encoded = self.encoder(tokenized)
            if not isinstance(encoded, tuple):
                raise RuntimeError("text encoder must return token features")
            _, text_memory = encoded  # [b, seq_len, d=1024]

            assert text_memory.shape[1] == inputs_embeds.shape[1]
            # Invert the mask to match the PyTorch transformer convention.
            text_attention_mask = mx.not_equal(text_attention_mask, 1)
            # Return text memory in sequence-first order for the decoder.
            text_memory = mx.transpose(text_memory, (1, 0, 2))
            # Match the decoder embedding width.
            text_memory_resized = self.resizer(text_memory)
        else:
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0, (
                "Can't replace boxes in text if it's already encoded"
            )

        # Return embeddings in PyTorch sequence-first order.
        return (
            text_attention_mask,
            text_memory_resized,
            mx.transpose(inputs_embeds, (1, 0, 2)),
        )
