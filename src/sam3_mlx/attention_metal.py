"""Fused FP32 attention for 32-wide heads on Metal.

MLX 0.32's fused attention kernel starts at head dim 64, so 32-wide heads would
otherwise be zero-padded to 64, doubling the score and value products. This
kernel runs the same online-softmax (flash) attention at the real width with
simdgroup 8x8 FP32 matrix products and FP32 accumulation throughout.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Protocol, cast

import mlx.core as mx


class _AttentionKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: list[mx.array],
        template: list[tuple[str, int]],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[mx.Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
    ) -> list[mx.array]:
        """Launch the attention Metal kernel."""
        ...


# Each threadgroup owns SIMDGROUPS * 8 query rows of one (batch, head) and walks
# the keys in blocks of BLOCK_KEYS staged in threadgroup memory. A simdgroup
# holds its 8 rows of Q (pre-scaled by scale * log2(e)) and of the running
# output as four 8x8 fragments. In Metal's 8x8 fragment layout each lane holds
# elements (fm, fn) and (fm, fn + 1); the four lanes sharing row fm differ in
# lane bits 0 and 3, so row maxima and sums reduce with shuffles 1 and 8.
_SOURCE = r"""
    constexpr int D = 32;
    constexpr int LD = D + 4;
    constexpr int NF = BLOCK_KEYS / 8;
    threadgroup float Ks[BLOCK_KEYS * LD];
    threadgroup float Vs[BLOCK_KEYS * LD];

    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const int bh = threadgroup_position_in_grid.y;
    const int heads = q_shape[1];
    const int b = bh / heads;
    const int h = bh % heads;
    const int Lq = q_shape[2];
    const int Lk = k_shape[2];

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const int row = (threadgroup_position_in_grid.x * SIMDGROUPS + sg) * 8 + fm;
    const float sc = scale[0];
    const device float* qb = q + b * q_strides[0] + h * q_strides[1];
    const device float* kb = k + b * k_strides[0] + h * k_strides[1];
    const device float* vb = v + b * v_strides[0] + h * v_strides[1];

    simdgroup_float8x8 Qf[4];
    simdgroup_float8x8 Of[4];
    _Pragma("clang loop unroll(full)")
    for (int d = 0; d < 4; d++) {
        float2 value = float2(0.0f);
        if (row < Lq) {
            const device float* p = qb + row * q_strides[2] + (d * 8 + fn) * q_strides[3];
            value = float2(p[0] * sc, p[q_strides[3]] * sc);
        }
        Qf[d].thread_elements()[0] = value.x;
        Qf[d].thread_elements()[1] = value.y;
        Of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -INFINITY;
    float l = 0.0f;
    const bool unit_stride = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = 0; k0 < Lk; k0 += BLOCK_KEYS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < BLOCK_KEYS * D / 4; i += SIMDGROUPS * 32) {
            const int r = i / (D / 4);
            const int c = (i % (D / 4)) * 4;
            const int key = k0 + r;
            float4 kv = float4(0.0f);
            float4 vv = float4(0.0f);
            if (key < Lk && unit_stride) {
                // packed_float4 needs only float alignment, so any view loads safely.
                kv = float4(*(const device packed_float4*)(kb + key * k_strides[2] + c));
                vv = float4(*(const device packed_float4*)(vb + key * v_strides[2] + c));
            } else if (key < Lk) {
                for (int e = 0; e < 4; e++) {
                    kv[e] = kb[key * k_strides[2] + (c + e) * k_strides[3]];
                    vv[e] = vb[key * v_strides[2] + (c + e) * v_strides[3]];
                }
            }
            *(threadgroup float4*)(&Ks[r * LD + c]) = kv;
            *(threadgroup float4*)(&Vs[r * LD + c]) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 Sf[NF];
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < NF; j++) {
            Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            _Pragma("clang loop unroll(full)")
            for (int d = 0; d < 4; d++) {
                simdgroup_float8x8 Kt;
                simdgroup_load(Kt, &Ks[j * 8 * LD + d * 8], LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sf[j], Qf[d], Kt, Sf[j]);
            }
        }
        const bool tail = k0 + BLOCK_KEYS > Lk;
        float m_new = m;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < NF; j++) {
            if (tail) {
                const int key = k0 + j * 8 + fn;
                if (key >= Lk) Sf[j].thread_elements()[0] = -INFINITY;
                if (key + 1 >= Lk) Sf[j].thread_elements()[1] = -INFINITY;
            }
            m_new = max(m_new, max(Sf[j].thread_elements()[0], Sf[j].thread_elements()[1]));
        }
        m_new = max(m_new, simd_shuffle_xor(m_new, 1));
        m_new = max(m_new, simd_shuffle_xor(m_new, 8));
        const float factor = fast::exp2(m - m_new);
        float row_sum = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < NF; j++) {
            const float e0 = fast::exp2(Sf[j].thread_elements()[0] - m_new);
            const float e1 = fast::exp2(Sf[j].thread_elements()[1] - m_new);
            Sf[j].thread_elements()[0] = e0;
            Sf[j].thread_elements()[1] = e1;
            row_sum += e0 + e1;
        }
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 8);
        l = l * factor + row_sum;
        m = m_new;
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < 4; d++) {
            Of[d].thread_elements()[0] *= factor;
            Of[d].thread_elements()[1] *= factor;
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < NF; j++) {
                simdgroup_float8x8 Vf;
                simdgroup_load(Vf, &Vs[j * 8 * LD + d * 8], LD);
                simdgroup_multiply_accumulate(Of[d], Sf[j], Vf, Of[d]);
            }
        }
    }
    if (row < Lq) {
        const float inv = 1.0f / l;
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < 4; d++) {
            device float* p = out + ((size_t(b) * Lq + row) * heads + h) * D + d * 8 + fn;
            p[0] = Of[d].thread_elements()[0] * inv;
            p[1] = Of[d].thread_elements()[1] * inv;
        }
    }
"""

HEAD_DIM = 32
# 4 simdgroups x 8 rows and 32-key blocks measured fastest on an M5 Max for
# 8 heads over 5184 tokens; 16-64 key blocks and 2-16 simdgroups were within
# 5% or slower.
_SIMDGROUPS = 4
_BLOCK_KEYS = 32


@lru_cache(maxsize=1)
def _kernel() -> _AttentionKernel:
    """Construct or retrieve the cached attention Metal kernel."""
    return cast(
        _AttentionKernel,
        mx.fast.metal_kernel(
            name="sam3_attention_head_dim_32",
            input_names=["q", "k", "v", "scale"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        ),
    )


def attention_head_dim_32(
    query: mx.array, key: mx.array, value: mx.array, *, scale: float
) -> mx.array:
    """Unmasked FP32 softmax attention for ``(B, H, L, 32)`` heads.

    ``query`` is ``(B, H, Lq, 32)`` and ``key``/``value`` ``(B, H, Lk, 32)``,
    float32. Strides are read as given, so transposed projection views need no
    copy; unit stride along the head dimension takes vector loads. Returns
    ``(B, H, Lq, 32)`` as a transposed view of a ``(B, Lq, H, 32)`` buffer, the
    layout callers merge heads into. The kernel has no differentiation rule.
    """
    batch, heads, length, head_dim = query.shape
    if (
        head_dim != HEAD_DIM
        or key.shape[:2] != (batch, heads)
        or key.shape != value.shape
        or key.shape[-1] != HEAD_DIM
        or not all(x.dtype == mx.float32 for x in (query, key, value))
    ):
        raise ValueError(
            "attention_head_dim_32 needs float32 (B, H, L, 32) queries and "
            "matching keys and values"
        )
    rows = _SIMDGROUPS * 8
    output = _kernel()(
        inputs=[
            query,
            key,
            value,
            mx.array([scale * math.log2(math.e)], dtype=mx.float32),
        ],
        template=[("SIMDGROUPS", _SIMDGROUPS), ("BLOCK_KEYS", _BLOCK_KEYS)],
        output_shapes=[(batch, length, heads, HEAD_DIM)],
        output_dtypes=[mx.float32],
        grid=(((length + rows - 1) // rows) * _SIMDGROUPS * 32, batch * heads, 1),
        threadgroup=(_SIMDGROUPS * 32, 1, 1),
    )[0]
    return mx.transpose(output, (0, 2, 1, 3))
