"""Explicit image-runtime precision policies and dtype boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Protocol, TypeAlias, cast

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from sam3_mlx.precision_layers import (
    AutocastConv1d, AutocastConv2d, AutocastConvTranspose2d, AutocastLinear,
    Float32GELU, Float32GroupNorm, Float32LayerNorm, copy_layer,
)

PrecisionPolicy: TypeAlias = Literal["fp32", "fp16", "bf16", "mixed"]
CheckpointDType: TypeAlias = Literal["float32", "float16", "bfloat16"]

PRECISION_POLICIES: tuple[PrecisionPolicy, ...] = ("fp32", "fp16", "bf16", "mixed")
CHECKPOINT_DTYPES: tuple[CheckpointDType, ...] = ("float32", "float16", "bfloat16")
_POLICY_ALIASES = {
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}
_POLICY_COMPUTE_DTYPE: dict[PrecisionPolicy, mx.Dtype] = {
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
    "mixed": mx.float16,
}
_POLICY_CHECKPOINT_DTYPE: dict[PrecisionPolicy, CheckpointDType] = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "float32",
    "mixed": "float16",
}
_CHECKPOINT_MLX_DTYPE: dict[CheckpointDType, mx.Dtype] = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}
_CASTABLE_FLOAT_DTYPES = frozenset({mx.float32, mx.float16, mx.bfloat16})


def parse_precision(value: object) -> PrecisionPolicy:
    """Normalize a precision name (``"float32"`` style aliases accepted) to a policy.

    Raises:
        TypeError: ``value`` is not a string.
        ValueError: The name is not one of ``fp32``, ``fp16``, ``bf16``, ``mixed``.
    """
    if not isinstance(value, str):
        raise TypeError("precision must be a string")
    normalized = _POLICY_ALIASES.get(value, value)
    if normalized not in PRECISION_POLICIES:
        raise ValueError(
            f"precision must be one of fp32, fp16, bf16, mixed; got {value!r}"
        )
    return normalized


def parse_checkpoint_dtype(value: object) -> CheckpointDType:
    """Validate a floating-point checkpoint dtype name (no ``fp*`` aliases)."""
    if not isinstance(value, str):
        raise TypeError("checkpoint dtype must be a string")
    if value not in CHECKPOINT_DTYPES:
        raise ValueError(
            f"checkpoint dtype must be one of float32, float16, bfloat16; got {value!r}"
        )
    return value


def compute_dtype_for_policy(policy: PrecisionPolicy) -> mx.Dtype:
    """Return the matrix/convolution compute dtype after applying the policy.

    ``"mixed"`` computes in float16; ``apply_precision_policy`` keeps LayerNorm in
    float32. BF16 follows CUDA autocast for the backbone and detector, retaining
    FP32 residual parameters, normalization, and decoder feed-forward layers.
    The segmentation head stays FP32 to stabilize mask-logit zero crossings.
    """
    return _POLICY_COMPUTE_DTYPE[parse_precision(policy)]


def checkpoint_dtype_for_policy(policy: PrecisionPolicy) -> CheckpointDType:
    """Return the floating dtype a checkpoint must store to load under ``policy``."""
    return _POLICY_CHECKPOINT_DTYPE[parse_precision(policy)]


def mlx_dtype_for_checkpoint(dtype: CheckpointDType) -> mx.Dtype:
    return _CHECKPOINT_MLX_DTYPE[parse_checkpoint_dtype(dtype)]


def dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def is_castable_float_dtype(dtype: mx.Dtype) -> bool:
    """Return whether ``dtype`` is a float32/float16/bfloat16 tensor the policy may cast.

    Integer, boolean and complex tensors keep their dtype.
    """
    return dtype in _CASTABLE_FLOAT_DTYPES


def model_precision(model: object) -> PrecisionPolicy:
    """Return the policy recorded on ``model`` (``"fp32"`` if none was applied)."""
    return parse_precision(getattr(model, "precision", "fp32"))


def cast_visual_input(image: mx.array, policy: PrecisionPolicy) -> mx.array:
    """Cast a normalized FP32 image exactly once at the visual-model boundary."""

    target = compute_dtype_for_policy(policy)
    if image.dtype == target:
        return image
    return image.astype(target)


def cast_checkpoint_weights(
    weights: dict[str, mx.array],
    dtype: CheckpointDType,
) -> dict[str, mx.array]:
    """Return a new weight dict with castable floating tensors converted to ``dtype``.

    Non-float tensors are passed through unchanged; ``weights`` is not mutated.
    """
    target = mlx_dtype_for_checkpoint(dtype)
    converted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if is_castable_float_dtype(value.dtype) and value.dtype != target:
            converted[key] = value.astype(target)
        else:
            converted[key] = value
    return converted


def apply_precision_policy(
    model: nn.Module, policy: PrecisionPolicy
) -> PrecisionPolicy:
    """Cast ``model`` in place to ``policy`` after weights load and before compilation.

    Records ``precision`` and ``compute_dtype`` on the model and drops any compiled
    visual/grounding graphs so they are rebuilt at the new dtype. Under
    ``"mixed"``, LayerNorm modules stay float32. BF16 uses FP32 source weights
    and casts at matrix/convolution boundaries. Returns the normalized policy.
    """

    policy = parse_precision(policy)
    compute_dtype = compute_dtype_for_policy(policy)
    model.set_dtype(
        mx.float32 if policy == "bf16" else compute_dtype,
        predicate=is_castable_float_dtype,
    )
    _configure_autocast_layers(model, bf16=policy == "bf16")
    if policy == "mixed":
        modules = cast(Iterable[object], model.modules())
        for module in modules:
            if isinstance(module, nn.LayerNorm):
                module.set_dtype(mx.float32)
    setattr(model, "precision", policy)
    setattr(model, "compute_dtype", compute_dtype)
    for owner in (model, getattr(model, "backbone", None)):
        for name in ("clear_compiled_visual", "clear_compiled_grounding"):
            clearer = getattr(owner, name, None)
            if callable(clearer):
                clearer()
    return policy


class _TreeUnflatten(Protocol):
    def __call__(self, tree: list[tuple[str, nn.Module]]) -> dict[str, object]: ...


class _UpdateModules(Protocol):
    def __call__(self, modules: dict[str, object]) -> nn.Module: ...


def _configure_autocast_layers(model: nn.Module, *, bf16: bool) -> None:
    from sam3_mlx.model.decoder import TransformerDecoderLayer
    from sam3_mlx.model.maskformer_segmentation import SegmentationHead

    modules = list(cast(Iterable[tuple[str, nn.Module]], model.named_modules()))
    fp32_projections: set[int] = set()
    for _, module in modules:
        if bf16 and isinstance(module, TransformerDecoderLayer):
            fp32_projections.update((id(module.linear1), id(module.linear2)))
        if bf16 and isinstance(module, SegmentationHead):
            fp32_projections.update(
                id(child) for child in cast(Iterable[object], module.modules())
            )
    implementations: dict[type[nn.Module], type[nn.Module]] = {
        nn.Linear: AutocastLinear,
        nn.Conv1d: AutocastConv1d,
        nn.Conv2d: AutocastConv2d,
        nn.ConvTranspose2d: AutocastConvTranspose2d,
        nn.LayerNorm: Float32LayerNorm,
        nn.GroupNorm: Float32GroupNorm,
        nn.GELU: Float32GELU,
    }
    if not bf16:
        implementations = {value: key for key, value in implementations.items()}
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in modules:
        implementation = implementations.get(type(module))
        replacement = copy_layer(module, implementation) if implementation else module
        if (
            bf16
            and id(module) not in fp32_projections
            and isinstance(replacement, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d))
        ):
            replacement.set_dtype(mx.bfloat16)
        if replacement is not module:
            replacements.append((name, replacement))
    update_modules = cast(_UpdateModules, getattr(model, "update_modules"))
    update_modules(cast(_TreeUnflatten, tree_unflatten)(replacements))
