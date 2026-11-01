import mlx.core as mx
from mlx import nn

from sam3_mlx.checkpoint_mapping import flatten_model_parameters as _parameters
from sam3_mlx.model.model_misc import MultiheadAttention


def test_parameters_and_seeded_init_match_mlx_multi_head_attention() -> None:
    """Checkpoint keys, shapes and the seeded draws match ``nn.MultiHeadAttention``."""
    mx.random.seed(7)
    base = _parameters(nn.MultiHeadAttention(8, 2, bias=True))
    mx.random.seed(7)
    ours = _parameters(MultiheadAttention(8, 2))

    assert list(ours) == list(base)
    for name, value in base.items():
        assert ours[name].shape == value.shape, name
        assert bool(mx.array_equal(ours[name], value)), name
