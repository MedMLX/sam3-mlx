import gc

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.position_encoding import PositionEmbeddingSine


def test_bounded_lru_cache_evicts_least_recently_used() -> None:
    cache = BoundedLRUCache[int, str](maxsize=2)
    cache[1] = "a"
    cache[2] = "b"
    assert cache.get(1) == "a"  # refresh 1
    cache[3] = "c"
    assert cache.get(2) is None
    assert cache.get(1) == "a"
    assert cache.get(3) == "c"
    cache.clear()
    assert cache.get(1) is None


def test_position_encoding_cache_is_bounded() -> None:
    pos = PositionEmbeddingSine(num_pos_feats=8, cache_size=2)
    for height, width in ((4, 4), (8, 8), (16, 16)):
        out = pos(mx.zeros((1, 1, height, width), dtype=mx.float32))
        evaluate_boundary(out)
        assert out.shape == (1, 8, height, width)
    assert pos.cache.get((4, 4)) is None
    assert pos.cache.get((8, 8)) is not None
    assert pos.cache.get((16, 16)) is not None


def test_position_encoding_cache_keeps_one_grid_for_a_batched_miss() -> None:
    """A batch-8 miss caches one unbatched grid, not a view into the batch buffer."""
    channels, side = 64, 32
    grid_bytes = channels * side * side * 4
    pos = PositionEmbeddingSine(num_pos_feats=channels)
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()

    batched = pos((8, channels, side, side))
    batched_np = to_numpy(batched)
    del batched
    gc.collect()
    mx.clear_cache()
    # The old cache kept the whole batch-8 graph (32 grids here) alive.
    assert mx.get_active_memory() - before < 2 * grid_bytes

    single = to_numpy(pos((1, channels, side, side)))
    np.testing.assert_array_equal(batched_np, np.repeat(single, 8, axis=0))
