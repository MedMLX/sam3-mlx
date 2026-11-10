"""Public builders enforce the shared full-float32 MLX process policy."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["image", "predictor"])
def test_builder_rejects_preinitialized_reduced_precision(entry: str) -> None:
    """A process that already fixed reduced precision cannot silently claim FP32."""
    code = r'''
import sys
import mlx.core as mx
x = mx.full((2, 2), 1 + 2**-15, dtype=mx.float32)
y = (x @ mx.ones((2, 2), dtype=mx.float32))[0, 0].item()
if y == 2 + 2**-14:
    sys.exit(77)  # This hardware did not select reduced products.
from medmlx_core.errors import MissingDependencyError
from sam3_mlx.model_builder import build_sam3_image_model, build_sam3_predictor, build_sam3_video_model
try:
    if sys.argv[1] == 'image':
        build_sam3_image_model(load_from_HF=False)
    elif sys.argv[1] == 'video':
        build_sam3_video_model(image_model=object())
    else:
        build_sam3_predictor(image_model=object())
except MissingDependencyError as exc:
    assert 'reduced precision' in str(exc)
    assert exc.hint is not None and 'MLX_ENABLE_TF32=0' in exc.hint
else:
    raise AssertionError('Builder accepted MLX initialized in reduced precision')
'''
    environment = os.environ.copy()
    environment.pop("MLX_ENABLE_TF32", None)
    completed = subprocess.run(
        [sys.executable, "-c", code, entry], env=environment, capture_output=True, text=True,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip("MLX did not select reduced-precision products on this hardware")
    assert completed.returncode == 0, completed.stderr


def test_builder_preserves_explicit_reduced_precision_opt_in() -> None:
    """The shared process policy must retain the caller's explicit TF32 choice."""
    code = r'''
import os
from sam3_mlx.model_builder import build_sam3_predictor
predictor = build_sam3_predictor(image_model=object())
assert predictor.model is not None
assert os.environ['MLX_ENABLE_TF32'] == '1'
'''
    environment = os.environ.copy()
    environment["MLX_ENABLE_TF32"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_builder_defaults_to_full_precision_in_a_fresh_process() -> None:
    """A direct builder applies the shared default before its first MLX product."""
    code = r'''
import os
from sam3_mlx.model_builder import build_sam3_predictor
import mlx.core as mx
predictor = build_sam3_predictor(image_model=object())
assert predictor.model is not None
assert os.environ['MLX_ENABLE_TF32'] == '0'
x = mx.full((2, 2), 1 + 2**-15, dtype=mx.float32)
assert (x @ mx.ones((2, 2), dtype=mx.float32))[0, 0].item() == 2 + 2**-14
'''
    environment = os.environ.copy()
    environment.pop("MLX_ENABLE_TF32", None)
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_video_predictor_selects_metal_from_cpu_default() -> None:
    """Public predictor construction selects the GPU before model execution."""
    code = r'''
import sys
import mlx.core as mx
if not mx.metal.is_available():
    sys.exit(77)
mx.set_default_device(mx.cpu)
from sam3_mlx.model_builder import build_sam3_predictor
build_sam3_predictor(image_model=object())
assert mx.default_device() == mx.gpu, mx.default_device()
'''
    environment = os.environ.copy()
    environment["MLX_ENABLE_TF32"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip("MLX Metal is unavailable")
    assert completed.returncode == 0, completed.stderr


def test_tf32_precision_selects_reduced_products_or_refuses() -> None:
    """``precision='tf32'`` runs TF32 products on capable GPUs and errors elsewhere."""
    code = r"""
import mlx.core as mx
from sam3_mlx.model_builder import build_sam3_image_model
try:
    model = build_sam3_image_model(load_from_HF=False, precision="tf32")
except ValueError as exc:
    assert "not active" in str(exc), exc
else:
    assert model.precision == "tf32"
    x = mx.full((2, 2), 1 + 2**-15, dtype=mx.float32)
    assert (x @ mx.ones((2, 2), dtype=mx.float32))[0, 0].item() == 2
"""
    environment = os.environ.copy()
    environment.pop("MLX_ENABLE_TF32", None)
    completed = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_fp32_precision_rejects_a_tf32_process() -> None:
    """An FP32 model is refused when the process already runs TF32 products."""
    code = r"""
import sys
import mlx.core as mx
x = mx.full((2, 2), 1 + 2**-15, dtype=mx.float32)
if (x @ mx.ones((2, 2), dtype=mx.float32))[0, 0].item() != 2:
    sys.exit(77)  # This hardware did not select reduced products.
from sam3_mlx.model_builder import build_sam3_image_model
try:
    build_sam3_image_model(load_from_HF=False)
except ValueError as exc:
    assert "precision='tf32'" in str(exc), exc
else:
    raise AssertionError("FP32 build accepted a TF32 process")
"""
    environment = os.environ.copy()
    environment["MLX_ENABLE_TF32"] = "1"
    completed = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], env=environment,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode == 77:
        pytest.skip("MLX did not select reduced-precision products on this hardware")
    assert completed.returncode == 0, completed.stderr
