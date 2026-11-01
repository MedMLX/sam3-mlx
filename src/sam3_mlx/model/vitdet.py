# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""ViTDet-style vision transformer trunk (SAM 3 image encoder) in MLX.

Layout: the trunk takes NCHW images and returns one channels-last ``(B, H, W, C)``
feature map; tokens stay channels-last throughout, because MLX convolutions and
linear layers are channels-last.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import mlx.core as mx
from mlx import nn

from sam3_mlx.resolutions import window_layout_is_exact
from sam3_mlx.precision_layers import attention, gelu_tanh
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.model_misc import Mlp as _Mlp
from sam3_mlx.model.rotary_metal import rotary_qk

_DEFAULT_ROPE_CACHE_SIZE = 8


def _source_truncated_normal(shape: tuple[int, ...]) -> mx.array:
    """Match the source's std-0.02 normal truncated at absolute bounds -2 and 2."""
    # timm's bounds are absolute values; MLX samples standardized values first.
    return mx.random.truncated_normal(lower=-100, upper=100, shape=shape) * 0.02


class Mlp(_Mlp):
    """ViT GELU MLP with the CUDA BF16 fused linear/GELU rounding boundary."""

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the MLP, preserving the fused BF16 linear and GELU rounding boundary."""
        if self.fc1.weight.dtype != mx.bfloat16:
            return super().__call__(x)
        # cuBLASLt applies tanh GELU to the FP32 accumulator before BF16 storage.
        x = x.astype(mx.bfloat16).astype(mx.float32) @ self.fc1.weight.astype(mx.float32).T
        bias = self.fc1.bias
        assert bias is not None  # ViT MLPs always have biases.
        x = x + bias.astype(mx.float32)
        x = gelu_tanh(x).astype(mx.bfloat16)
        return self.fc2(x)


def polar(a: mx.array, b: mx.array) -> mx.array:
    """Return ``a * exp(1j * b)`` as complex64 (magnitude ``a``, angle ``b``)."""
    return (a * mx.exp(1j * b)).astype(mx.complex64)


def real(x: mx.array) -> mx.array:
    """Flatten complex64 ``(..., D)`` to float32 ``(..., 2 * D)`` as interleaved (re, im) pairs."""
    parts = mx.reshape(mx.view(x, mx.float32), (*x.shape, 2))
    return mx.reshape(parts, (*x.shape[:-1], -1))


def view_as_complex(x: mx.array) -> mx.array:
    """Pair adjacent channels of ``(..., D)`` as (re, im) into complex64 ``(..., D // 2)``."""
    parts = mx.reshape(x, (*x.shape[:-1], -1, 2))
    return (parts[..., 0] + 1j * parts[..., 1]).astype(mx.complex64)


def init_t_xy(
    end_x: int, end_y: int, scale: float = 1.0, offset: int = 0
) -> tuple[mx.array, mx.array]:
    """Return flattened (x, y) grid coordinates, each ``(end_x * end_y,)`` float32.

    Tokens are in row-major order with x varying fastest, then scaled and offset.
    """
    t = mx.arange(end_x * end_y, dtype=mx.float32)
    t_x = (t % end_x).astype(mx.float32)
    t_y = mx.floor(mx.divide(t, end_x)).astype(mx.float32)
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
) -> mx.array:
    """Axial 2-D rotary frequencies as complex64 ``(end_x * end_y, dim // 2)``.

    ``dim`` is the per-head width. The first ``dim // 4`` columns rotate by the
    x coordinate and the last ``dim // 4`` by the y coordinate (same base
    frequencies for both); rows follow ``init_t_xy`` order.
    """
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 4)[: (dim // 4)].astype(mx.float32) / dim))
    t_x, t_y = init_t_xy(end_x, end_y, scale=scale_pos, offset=offset)
    freqs_x = mx.outer(t_x, freqs)
    freqs_y = mx.outer(t_y, freqs)
    freqs_cis_x = polar(mx.ones_like(freqs_x), freqs_x)
    freqs_cis_y = polar(mx.ones_like(freqs_y), freqs_y)
    return mx.concat([freqs_cis_x, freqs_cis_y], axis=-1)


def apply_rotary_enc(
    xq: mx.array, xk: mx.array, freqs_cis: mx.array
) -> tuple[mx.array, mx.array]:
    """Rotate query and key channel pairs by complex ``freqs_cis``.

    ``xq`` and ``xk`` are ``(B, num_heads, L, head_dim)`` and ``freqs_cis`` is
    ``(L, head_dim // 2)``; adjacent channel pairs form the complex components.
    Outputs keep the input dtypes.
    """
    xq_ = view_as_complex(xq)
    xk_ = view_as_complex(xk)
    xq_out = mx.flatten(real(xq_ * freqs_cis), start_axis=3)
    xk_out = mx.flatten(real(xk_ * freqs_cis), start_axis=3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def window_partition(x: mx.array, window_size: int) -> tuple[mx.array, tuple[int, int]]:
    """Split ``x`` ``(B, H, W, C)`` into non-overlapping square windows.

    The bottom/right edges are zero-padded up to a multiple of ``window_size``.
    Returns the windows ``(B * num_windows, window_size, window_size, C)`` and the
    padded size ``(Hp, Wp)``.
    """
    batch, height, width, channels = x.shape

    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
    hp, wp = height + pad_h, width + pad_w

    x = mx.reshape(
        x,
        (
            batch,
            hp // window_size,
            window_size,
            wp // window_size,
            window_size,
            channels,
        ),
    )
    windows = mx.reshape(
        mx.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
        (-1, window_size, window_size, channels),
    )
    return windows, (hp, wp)


def window_unpartition(
    windows: mx.array,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> mx.array:
    """Invert ``window_partition``: ``(B * nW, ws, ws, C)`` to ``(B, H, W, C)``.

    ``pad_hw`` is the padded size returned by the partition and ``hw`` the
    original ``(H, W)`` to crop back to.
    """
    hp, wp = pad_hw
    height, width = hw
    batch = windows.shape[0] // (hp * wp // window_size // window_size)
    x = mx.reshape(
        windows,
        (batch, hp // window_size, wp // window_size, window_size, window_size, -1),
    )
    x = mx.reshape(mx.transpose(x, axes=(0, 1, 3, 2, 4, 5)), (batch, hp, wp, -1))

    if hp > height or wp > width:
        x = x[:, :height, :width, :]

    return x


def get_abs_pos(
    abs_pos: mx.array, has_cls_token: bool, hw: tuple[int, int]
) -> mx.array:
    """Tile the absolute position embedding over an ``(H, W)`` token grid.

    ``abs_pos`` is ``(1, N, C)`` (with a leading class-token row, which is
    dropped, when ``has_cls_token``) for a square pretraining grid. The grid is
    repeated and cropped to ``(1, H, W, C)`` channels-last.
    """
    height, width = hw
    spatial_pos = abs_pos[:, 1:] if has_cls_token else abs_pos
    size = math.isqrt(spatial_pos.shape[1])
    grid = mx.reshape(spatial_pos, (1, size, size, -1))
    tiled = mx.tile(grid, (1, height // size + 1, width // size + 1, 1))
    return tiled[:, :height, :width]


class PatchEmbed(nn.Module):
    """Convolutional patch embedding: NCHW RGB image to channels-last tokens."""

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        """Configure the convolution that projects image patches to token features."""
        super().__init__()
        self.proj = nn.Conv2d(
            3, embed_dim, kernel_size=kernel_size, stride=stride, bias=bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Embed ``(B, 3, H, W)`` into ``(B, H / stride, W / stride, embed_dim)``."""
        return self.proj(mx.transpose(x, axes=(0, 2, 3, 1)))


class Attention(nn.Module):
    """Multi-head self-attention over ``(B, H, W, C)`` tokens with optional 2-D RoPE.

    ``input_size`` is the ``(H, W)`` token grid the RoPE frequencies are built
    for. RoPE frequencies for other grid sizes are computed on demand and cached.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        input_size: tuple[int, int] | None = None,
        use_rope: bool = False,
        rope_pt_size: tuple[int, int] | None = None,
        rope_interp: bool = False,
    ) -> None:
        """Configure multihead projections and optional spatial rotary positions."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.input_size = input_size
        self.use_rope = use_rope
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        self._freqs_cis_cache: BoundedLRUCache[tuple[int, int], mx.array] = (
            BoundedLRUCache(maxsize=_DEFAULT_ROPE_CACHE_SIZE)
        )
        self.freqs_cis: mx.array | None = None
        if use_rope:
            assert input_size is not None
            if self.rope_pt_size is None:
                self.rope_pt_size = input_size
            self.freqs_cis = self._build_rope_freqs(input_size)

    def _build_rope_freqs(self, spatial_size: tuple[int, int]) -> mx.array:
        """Build rotary frequencies for an ``(H, W)`` token grid."""
        height, width = spatial_size
        scale_pos = 1.0
        if self.rope_interp:
            assert self.rope_pt_size is not None
            scale_pos = self.rope_pt_size[0] / width
        return compute_axial_cis(
            self.head_dim, end_x=width, end_y=height, scale_pos=scale_pos
        )

    def _rope_freqs(self, spatial_size: tuple[int, int]) -> mx.array:
        """Select frequencies by grid shape; equal token counts can have different positions."""
        if spatial_size == self.input_size:
            assert self.freqs_cis is not None
            return self.freqs_cis
        cached = self._freqs_cis_cache.get(spatial_size)
        if cached is None:
            cached = self._build_rope_freqs(spatial_size)
            self._freqs_cis_cache[spatial_size] = cached
        return cached

    def _apply_rope(
        self, q: mx.array, k: mx.array, spatial_size: tuple[int, int]
    ) -> tuple[mx.array, mx.array]:
        """Rotate queries and keys when spatial rotary encoding is enabled."""
        if not self.use_rope:
            return q, k
        freqs_cis = self._rope_freqs(spatial_size)
        if not self.training and mx.metal.is_available() and mx.default_device().type == mx.gpu:
            return rotary_qk(q, k, freqs_cis)
        return apply_rotary_enc(q, k, freqs_cis=freqs_cis)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply self-attention to ``(B, H, W, C)``; the output has the shape of ``x``."""
        batch, height, width, _ = x.shape
        # qkv with shape (3, B, nHead, L, C)
        qkv = mx.reshape(self.qkv(x), (batch, height * width, 3, self.num_heads, -1))
        # q, k, v with shape (B, nHead, L, C)
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self._apply_rope(q, k, (height, width))
        x = attention(q, k, v, scale=self.head_dim**-0.5)
        x = mx.reshape(
            mx.transpose(
                mx.reshape(x, (batch, self.num_heads, height, width, -1)),
                axes=(0, 2, 3, 1, 4),
            ),
            (batch, height, width, -1),
        )
        return self.proj(x)


class Block(nn.Module):
    """Pre-norm transformer block with global or windowed attention.

    ``window_size > 0`` makes attention local to ``window_size x window_size``
    token windows; ``0`` makes it global. The attention module is built for a
    window-sized grid in the windowed case, so RoPE is relative to the window.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
        use_rope: bool = False,
        rope_pt_size: tuple[int, int] | None = None,
        rope_interp: bool = False,
    ) -> None:
        """Configure a pre-normalized attention and MLP block with optional local windows."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.window_size = window_size

    def __call__(self, x: mx.array, *, windowed: bool = False) -> mx.array:
        """Apply the block to ``x`` ``(B, H, W, C)``.

        With ``windowed=True`` the caller has already partitioned ``x`` into
        windows (``(B * nW, ws, ws, C)``), so the block neither partitions nor
        merges them.
        """
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0 and not windowed:
            height, width = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, pad_hw, (height, width))
        else:
            x = self.attn(x)

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """ViTDet trunk: patch embedding, tiled absolute position embedding and blocks.

    Blocks listed in ``global_att_blocks`` use global attention; all others use
    ``window_size`` windows. Calling the trunk on an NCHW image returns the last
    block's output as an NHWC map ``(B, H / patch_size, W / patch_size, embed_dim)``.
    When the token grid tiles exactly into windows, consecutive windowed blocks
    stay partitioned instead of merging and re-splitting around each one.

    Args:
        img_size: Image side in pixels that the RoPE tables are built for.
        patch_size: Patch edge in pixels.
        pretrain_img_size: Side of the pretraining image used to size
            ``pos_embed``; it is tiled to the actual token grid at run time.
        pretrain_use_cls_token: ``pos_embed`` has a leading class-token row
            (unused at run time).

    Weights start from a random initialization and are expected to be replaced by
    checkpoint loading.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_abs_pos: bool = True,
        window_size: int = 14,
        global_att_blocks: Sequence[int] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: int | None = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        ln_pre: bool = False,
        bias_patch_embed: bool = True,
    ) -> None:
        """Configure patch tokens, positional embeddings and transformer blocks."""
        super().__init__()
        self.embed_dim = embed_dim
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        self.pos_embed: mx.array | None = None
        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) ** 2
            num_positions = num_patches + 1 if pretrain_use_cls_token else num_patches
            self.pos_embed = mx.zeros((1, num_positions, embed_dim))

        global_blocks = set(global_att_blocks)
        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                window_size=0 if i in global_blocks else window_size,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=(window_size, window_size)
                if rope_pt_size is None
                else (rope_pt_size, rope_pt_size),
                rope_interp=use_interp_rope,
            )
            for i in range(depth)
        ]

        if self.pos_embed is not None:
            self.pos_embed = _source_truncated_normal(self.pos_embed.shape)

        self.ln_pre = nn.LayerNorm(embed_dim) if ln_pre else nn.Identity()

        self.apply_to_modules(lambda _path, module: self._init_weights(module))

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize one linear or layer-normalization module as the source does."""
        if isinstance(m, nn.Linear):
            m.weight = _source_truncated_normal(m.weight.shape)
            if "bias" in m:
                m.bias = mx.zeros_like(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.weight = mx.ones_like(m.weight)
            m.bias = mx.zeros_like(m.bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Return the NHWC feature map of an NCHW image batch."""
        tokens = self.patch_embed(x)
        if self.pos_embed is not None:
            tokens = tokens + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (tokens.shape[1], tokens.shape[2]),
            )
        tokens = self.ln_pre(tokens)

        index = 0
        block_count = len(self.blocks)
        while index < block_count:
            block = self.blocks[index]
            window_size = block.window_size
            height, width = tokens.shape[1], tokens.shape[2]
            if window_size == 0 or not window_layout_is_exact(height, width, window_size):
                tokens = block(tokens)
                index += 1
                continue
            stop = index
            while stop < block_count and self.blocks[stop].window_size == window_size:
                stop += 1
            windowed, pad_hw = window_partition(tokens, window_size)
            for group_index in range(index, stop):
                windowed = self.blocks[group_index](windowed, windowed=True)
            tokens = window_unpartition(windowed, window_size, pad_hw, (height, width))
            index = stop
        return tokens
