from __future__ import annotations
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.mlx_runtime import evaluate_boundary

import warnings

import mlx.core as mx
from mlx import nn
from sam3_mlx.checkpoint_mapping import flatten_model_parameters
import numpy as np

from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model_builder import (
    build_sam3_image_model,
)


def _parameter_contract(model: nn.Module) -> tuple[int, int]:
    parameters = flatten_model_parameters(model)
    return (
        len(parameters),
        sum(int(value.size) for value in parameters.values()),
    )


def _build_characterized_image_model() -> Sam3Image:
    mx.random.seed(20260829)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_sam3_image_model(
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )


def test_image_builder_parameter_contract_is_frozen() -> None:
    model = _build_characterized_image_model()

    assert model.inst_interactive_predictor is None
    assert _parameter_contract(model) == (1379, 841_690_422)


def test_image_builder_seeded_component_output_stays_within_fp32_envelope() -> None:
    model = _build_characterized_image_model()
    hidden_states = mx.reshape(mx.arange(512, dtype=mx.float32) / 512.0, (1, 1, 2, 256))
    prompt = mx.reshape(mx.arange(768, dtype=mx.float32) / 768.0, (3, 1, 256))
    prompt_mask = mx.array([[False, False, True]], dtype=mx.bool_)

    assert model.dot_prod_scoring is not None
    output = model.dot_prod_scoring(hidden_states, prompt, prompt_mask)
    evaluate_boundary(output)

    actual = to_numpy(output)
    expected = np.array(
        [[[[0.009574081748723984], [0.07803979516029358]]]], dtype=np.float32
    )
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    # The checked-in golden predates this MLX runtime; current Metal and CPU
    # results differ by several FP32 ULPs while the parameter graph stays fixed.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)
