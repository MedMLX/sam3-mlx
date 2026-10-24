from typing import cast

import mlx.core as mx
import pytest

from sam3_mlx.model.model_misc import (
    CloneableModule,
    LayerScale,
    get_clones,
)


def test_get_clones_module_and_factory_paths() -> None:
    template = LayerScale(dim=2, init_values=2.0)
    clones = get_clones(template, 3)
    assert len(clones) == 3
    assert all(isinstance(clone, LayerScale) for clone in clones)
    assert clones[0] is not template
    assert mx.allclose(clones[0].gamma, mx.full((2,), 2.0)).item()

    made = get_clones(lambda: LayerScale(dim=2, init_values=3.0), 2)
    assert len(made) == 2
    assert all(mx.allclose(clone.gamma, mx.full((2,), 3.0)).item() for clone in made)


def test_get_clones_rejects_non_callable_non_module() -> None:
    with pytest.raises(TypeError, match="zero-argument factory"):
        get_clones(cast(CloneableModule[LayerScale], object()), 1)
