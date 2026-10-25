from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
import sys
import types
from typing import ContextManager

import pytest

from sam3_mlx.release_contract import FusedAdapterBinding, ORACLE_FP32_FUSED_ADAPTER
from scripts._oracle_runtime import (
    install_cpu_oracle_adapters,
    open_official_image_oracle,
    restore_construction_adapters,
    TensorClass,
    validate_case_specs,
)


class _TensorApi:
    def __init__(self) -> None:
        self.pin_memory: Callable[..., object] = self._pin_memory

    @staticmethod
    def _pin_memory(tensor: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return ("pinned", tensor)


class _TorchRuntime:
    bfloat16: object = object()
    __version__: object = "test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.zeros: Callable[..., object] = self._zeros
        self.arange: Callable[..., object] = self._arange
        self.Tensor: TensorClass = _TensorApi()
        self.autocast: Callable[..., ContextManager[None]] = self._autocast

    def _zeros(self, *args: object, **kwargs: object) -> object:
        del args
        self.calls.append(("zeros", kwargs))
        return kwargs

    def _arange(self, *args: object, **kwargs: object) -> object:
        del args
        self.calls.append(("arange", kwargs))
        return kwargs

    @staticmethod
    def _autocast(*args: object, **kwargs: object) -> ContextManager[None]:
        del args, kwargs
        return nullcontext()

    @staticmethod
    def is_autocast_enabled(device_type: str) -> bool:
        del device_type
        return False


def test_cpu_oracle_adapters_redirect_only_cuda_construction_devices() -> None:
    torch = _TorchRuntime()
    original_zeros = torch.zeros
    original_arange = torch.arange
    original_pin_memory = torch.Tensor.pin_memory

    originals = install_cpu_oracle_adapters(torch)
    torch.zeros(2, device="cuda:0")
    torch.arange(2, device="mps")

    assert torch.calls == [
        ("zeros", {"device": "cpu"}),
        ("arange", {"device": "mps"}),
    ]
    marker = object()
    assert torch.Tensor.pin_memory(marker) is marker

    restore_construction_adapters(torch, originals)
    assert torch.zeros is original_zeros
    assert torch.arange is original_arange
    assert originals.pin_memory is original_pin_memory


def test_oracle_case_validation_returns_precise_shape_and_rejects_bool_aliases() -> (
    None
):
    valid = [
        {
            "name": "positive_box",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [{"box": [1, 2.5, 3, 4], "label": True}],
        }
    ]

    assert validate_case_specs(valid) == valid

    invalid_label = [
        {
            **valid[0],
            "geometric_prompts": [{"box": [1, 2, 3, 4], "label": 1}],
        }
    ]
    with pytest.raises(ValueError, match="invalid box label"):
        validate_case_specs(invalid_label)

    with pytest.raises(ValueError, match="resolution must be an integer"):
        validate_case_specs([{**valid[0], "resolution": True}])


@pytest.mark.parametrize(
    "failure",
    [
        "builder_import",
        "processor_import",
        "builder_source",
        "processor_source",
        "build",
        "fused_adapter",
        "parameter_dtype",
        "body",
    ],
)
def test_oracle_restores_torch_factories_on_every_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    torch = _TorchRuntime()
    originals = (torch.zeros, torch.arange, torch.Tensor.pin_memory)
    builder = types.ModuleType("sam3.model_builder")
    processor = types.ModuleType("sam3.model.sam3_image_processor")
    builder.__file__ = str(tmp_path / "sam3/model_builder.py")
    processor.__file__ = str(tmp_path / "sam3/model/sam3_image_processor.py")
    if failure == "builder_source":
        builder.__file__ = str(tmp_path.parent / "wrong_builder.py")
    if failure == "processor_source":
        processor.__file__ = str(tmp_path.parent / "wrong_processor.py")

    class Parameter:
        dtype = "torch.bfloat16" if failure == "parameter_dtype" else "torch.float32"

    class Model:
        def parameters(self) -> tuple[Parameter, ...]:
            return (Parameter(),)

        def buffers(self) -> tuple[Parameter, ...]:
            return ()

    def build_model(**kwargs: object) -> object:
        del kwargs
        assert torch.zeros is not originals[0]
        assert torch.arange is not originals[1]
        if failure == "build":
            raise ValueError("build failed")
        return Model()

    def adapt_fused(checkout: Path) -> FusedAdapterBinding:
        assert checkout == tmp_path
        if failure == "fused_adapter":
            raise ValueError("fused adapter failed")
        return ORACLE_FP32_FUSED_ADAPTER.copy()

    class Processor:
        pass

    Processor.__module__ = processor.__name__
    setattr(builder, "build_sam3_image_model", build_model)
    setattr(processor, "Sam3Processor", Processor)
    monkeypatch.setitem(sys.modules, processor.__name__, processor)
    # The adapter replaces this import; register it with pytest for cleanup.
    monkeypatch.setitem(
        sys.modules, "sam3.model.edt", types.ModuleType("sam3.model.edt")
    )

    def import_oracle_module(name: str) -> object:
        if name == "torch":
            return torch
        if name == "sam3.model_builder":
            if failure == "builder_import":
                raise ImportError("builder import failed")
            return builder
        if failure == "processor_import":
            raise ImportError("processor import failed")
        return processor

    monkeypatch.setattr("scripts._oracle_runtime.import_module", import_oracle_module)
    monkeypatch.setattr("scripts._oracle_runtime.install_cpu_fp32_fused_adapter", adapt_fused)
    with pytest.raises((ImportError, ValueError)):
        with open_official_image_oracle(
            official_checkout=tmp_path, checkpoint=tmp_path / "checkpoint.pt"
        ):
            assert torch.zeros is originals[0]
            assert torch.arange is originals[1]
            raise ValueError("body failed")
    assert torch.zeros is originals[0]
    assert torch.arange is originals[1]
    assert torch.Tensor.pin_memory is originals[2]
