# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""CLIP-style text transformer used as SAM 3's text encoder (VE text encoder).

Text is tokenized to ``(B, L)`` int ids with 0 as padding. Token features are
batch-first inside the transformer and returned sequence-first (``(L, B, C)``)
for the decoder, matching the upstream PyTorch convention.
"""

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.precision_layers import MultiHeadAttention


class MLP(nn.Module):
    """Two-layer GELU feed-forward block (``c_fc``, ``gelu``, ``c_proj``) over the last axis."""

    def __init__(self, d_model: int, mlp_width: int) -> None:
        """Build the two projections and intermediate text-encoder activation."""
        super().__init__()
        self.c_fc = nn.Linear(d_model, mlp_width)
        # nn.GELU leaves ``__call__`` unannotated.
        self.gelu: Callable[[mx.array], mx.array] = nn.GELU()
        self.c_proj = nn.Linear(mlp_width, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the text feed-forward block."""
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    """Pre-norm transformer block: self-attention then MLP, each with a residual.

    Operates on batch-first ``(B, L, d_model)`` input.
    """

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0) -> None:
        """Configure attention, normalization and MLP residual layers."""
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, bias=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio))

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        """Apply attention and feed-forward residual updates to text tokens.

        ``attn_mask`` is additive (``-inf`` blocks) and is cast to the
        normalized-token dtype.
        """
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, mask=attn_mask.astype(h.dtype))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    """Stack of ``layers`` residual attention blocks over batch-first ``(B, L, width)``."""

    def __init__(
        self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0
    ) -> None:
        """Build the configured stack of text attention blocks."""
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = [
            ResidualAttentionBlock(width, heads, mlp_ratio) for _ in range(layers)
        ]

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        """Apply each text attention block with the attention mask."""
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class TextTransformer(nn.Module):
    """CLIP-style causal text transformer returning final-layer-norm token features.

    ``positional_embedding`` starts as zeros and is expected to be filled by
    checkpoint loading. ``text_projection`` is the checkpoint's pooled-output
    projection; SAM 3 uses token features only, so it is loaded but not applied.
    """

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        use_ln_post: bool = True,
    ) -> None:
        """Configure token, position, transformer and text-projection layers."""
        super().__init__()
        self.num_pos = self.context_length = context_length
        self.width = width

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = mx.zeros((self.num_pos, width))
        self.transformer = Transformer(
            width=width, layers=layers, heads=heads, mlp_ratio=mlp_ratio
        )
        self.ln_final = nn.LayerNorm(width) if use_ln_post else nn.Identity()
        # Additive causal mask: attend on and below the diagonal.
        self.attn_mask = mx.triu(
            mx.full((self.num_pos, self.num_pos), -float("inf")), k=1
        )
        self.text_projection = mx.zeros((width, output_dim))

    def __call__(self, text: mx.array) -> mx.array:
        """Encode ``(B, L)`` token ids with ``L <= context_length`` to ``(B, L, width)``."""
        seq_len = text.shape[1]
        x = self.token_embedding(text) + self.positional_embedding[:seq_len]
        x = self.transformer(x, self.attn_mask[:seq_len, :seq_len])
        return self.ln_final(x)


class VETextEncoder(nn.Module):
    """Text encoder for SAM 3 prompts: tokenizer, ``TextTransformer`` and width resizer.

    Args:
        d_model: Width of the returned text memory (the decoder width).
        tokenizer: CLIP BPE tokenizer returning ``(B, L)`` int token ids, with 0
            for padding.
        width: Transformer width.
        heads: Attention heads.
        layers: Transformer depth.
        context_length: Token length (prompts are padded or truncated by the
            tokenizer to this).
        vocab_size: Token embedding rows.
        use_ln_post: Apply a final LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        tokenizer: SimpleTokenizer,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
    ) -> None:
        """Configure tokenization and the vision-language text encoder."""
        super().__init__()
        self.context_length = context_length
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            use_ln_post=use_ln_post,
        )
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def __call__(self, text: list[str]) -> tuple[mx.array, mx.array]:
        """Encode a non-empty list of prompts.

        Returns:
            ``(attention_mask, memory)``. ``attention_mask`` is a ``(B, L)`` bool
            array that is True at padding positions (the PyTorch
            ``key_padding_mask`` convention). ``memory`` is the sequence-first
            ``(L, B, d_model)`` text memory.

        Raises:
            ValueError: If ``text`` is empty.
        """
        if not text:
            raise ValueError("text must contain at least one prompt")
        tokenized = self.tokenizer(text, context_length=self.context_length)
        text_memory = self.encoder(tokenized)  # (B, L, width)
        # Sequence-first for the decoder, resized to the decoder width.
        text_memory = self.resizer(mx.transpose(text_memory, (1, 0, 2)))
        return tokenized == 0, text_memory
