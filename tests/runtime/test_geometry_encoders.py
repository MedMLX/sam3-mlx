import mlx.core as mx

from sam3_mlx.model.encoder import TransformerEncoderLayer
from sam3_mlx.model.geometry_encoders import (
    Prompt,
    SequenceGeometryEncoder,
    concat_padded_sequences,
)
from sam3_mlx.model.model_misc import MultiheadAttention
from sam3_mlx.model.position_encoding import PositionEmbeddingSine


def test_concat_padded_sequences_matches_scatter_order() -> None:
    seq1 = mx.array([[[10], [100]], [[11], [101]], [[12], [102]]], dtype=mx.int64)
    mask1 = mx.array([[False, False, True], [False, True, True]])
    seq2 = mx.array([[[20], [200]], [[21], [201]]], dtype=mx.int64)
    mask2 = mx.array([[False, True], [False, False]])

    sequence, mask = concat_padded_sequences(seq1, mask1, seq2, mask2)

    assert sequence.squeeze(-1).tolist() == [
        [10, 100],
        [11, 200],
        [20, 201],
        [21, 0],
        [0, 0],
    ]
    assert mask.tolist() == [
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]


def test_prompt_appends_repeatedly_from_empty() -> None:
    prompt = Prompt.empty(1)

    prompt.append_points(
        mx.array([[[1, 1]], [[9, 9]]], dtype=mx.float32),
        mx.array([[1], [1]], dtype=mx.int64),
        mx.array([[False, True]]),
    )
    prompt.append_points(
        mx.array([[[2, 2]], [[3, 3]]], dtype=mx.float32),
        mx.array([[0], [1]], dtype=mx.int64),
    )
    prompt.append_boxes(mx.full((1, 1, 4), 0.5), mx.array([[True]]))
    prompt.append_boxes(mx.full((1, 1, 4), 0.25), mx.array([[False]]))

    assert prompt.point_embeddings.tolist() == [
        [[1.0, 1.0]],
        [[2.0, 2.0]],
        [[3.0, 3.0]],
        [[0.0, 0.0]],
    ]
    assert prompt.point_labels.tolist() == [[1], [0], [1], [0]]
    assert prompt.point_mask.tolist() == [[False, False, False, True]]
    assert prompt.box_embeddings.tolist() == [[[0.5] * 4], [[0.25] * 4]]
    assert prompt.box_labels.tolist() == [[True], [False]]
    assert prompt.box_mask.tolist() == [[False, False]]


def test_geometry_encoder_returns_points_boxes_then_cls_token() -> None:
    mx.random.seed(0)

    def layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            d_model=8,
            dim_feedforward=16,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            self_attention=MultiheadAttention(8, 2),
            cross_attention=MultiheadAttention(8, 2),
        )

    encoder = SequenceGeometryEncoder(
        d_model=8,
        pos_enc=PositionEmbeddingSine(num_pos_feats=8),
        num_layers=1,
        layer=layer,
        roi_size=2,
    )
    prompt = Prompt.empty(2)
    prompt.append_points(mx.array([[[0.3, 0.6], [0.5, 0.5]]]), mx.array([[1, 0]]))
    prompt.append_boxes(
        mx.full((2, 2, 4), 0.4),
        mx.ones((2, 2), dtype=mx.int64),
        mx.array([[False, False], [False, True]]),
    )
    feats = mx.random.normal((4 * 4, 2, 8))

    embeds, mask = encoder(prompt, [feats], [(4, 4)], [mx.zeros_like(feats)])

    assert embeds.shape == (4, 2, 8)
    assert mask.tolist() == [[False, False, False, False], [False, False, False, True]]
    assert bool(mx.all(mx.isfinite(embeds)))
