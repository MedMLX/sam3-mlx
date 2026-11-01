"""Explicit image-runtime precision policies and dtype boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias, cast

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from sam3_mlx.precision_layers import (
    AutocastConv2d, AutocastConvTranspose2d, AutocastLinear,
    Float32GELU, Float32GroupNorm, Float32LayerNorm, copy_layer,
)

PrecisionPolicy: TypeAlias = Literal["fp32", "bf16"]

_POLICY_COMPUTE_DTYPE: dict[PrecisionPolicy, mx.Dtype] = {
    "fp32": mx.float32,
    "bf16": mx.bfloat16,
}


def parse_precision(value: object) -> PrecisionPolicy:
    """Validate a precision name.

    Raises:
        ValueError: The value is not ``"fp32"`` or ``"bf16"``.
    """
    if value != "fp32" and value != "bf16":
        raise ValueError(f"precision must be 'fp32' or 'bf16'; got {value!r}")
    return cast(PrecisionPolicy, value)


def compute_dtype_for_policy(policy: PrecisionPolicy) -> mx.Dtype:
    """Return the matrix/convolution compute dtype after applying the policy.

    BF16 follows CUDA autocast for the backbone and detector, retaining FP32
    residual parameters, normalization, and decoder feed-forward layers. The
    segmentation head stays FP32 to stabilize mask-logit zero crossings.
    """
    return _POLICY_COMPUTE_DTYPE[policy]


def dtype_name(dtype: mx.Dtype) -> str:
    """Return the unqualified name of an MLX dtype."""
    return str(dtype).rsplit(".", 1)[-1]


def model_precision(model: object) -> PrecisionPolicy:
    """Return the policy recorded on ``model`` (``"fp32"`` if none was applied)."""
    return cast(PrecisionPolicy, getattr(model, "precision", "fp32"))


def cast_visual_input(image: mx.array, policy: PrecisionPolicy) -> mx.array:
    """Cast a normalized FP32 image exactly once at the visual-model boundary."""

    target = compute_dtype_for_policy(policy)
    if image.dtype == target:
        return image
    return image.astype(target)


def apply_precision_policy(model: nn.Module, policy: PrecisionPolicy) -> None:
    """Cast ``model`` in place to ``policy`` after weights load and before compilation.

    Records ``precision`` and ``compute_dtype`` on the model and drops any compiled
    visual/grounding graphs so they are rebuilt at the new dtype. BF16 keeps FP32
    source weights for everything except the autocast projections, which cast
    their inputs to the BF16 weight dtype.
    """

    model.set_dtype(mx.float32)
    _configure_autocast_layers(model, bf16=policy == "bf16")
    setattr(model, "precision", policy)
    setattr(model, "compute_dtype", compute_dtype_for_policy(policy))
    for owner in (model, getattr(model, "backbone", None)):
        for name in ("clear_compiled_visual", "clear_compiled_grounding"):
            clearer = getattr(owner, name, None)
            if callable(clearer):
                clearer()


def _configure_autocast_layers(model: nn.Module, *, bf16: bool) -> None:
    """Select projection and normalization implementations for the precision policy."""
    from sam3_mlx.model.decoder import TransformerDecoderLayer
    from sam3_mlx.model.maskformer_segmentation import UniversalSegmentationHead

    modules = list(cast(Iterable[tuple[str, nn.Module]], model.named_modules()))
    fp32_projections: set[int] = set()
    for _, module in modules:
        if bf16 and isinstance(module, TransformerDecoderLayer):
            fp32_projections.update((id(module.linear1), id(module.linear2)))
        if bf16 and isinstance(module, UniversalSegmentationHead):
            fp32_projections.update(
                id(child) for child in cast(Iterable[object], module.modules())
            )
    implementations: dict[type[nn.Module], type[nn.Module]] = {
        nn.Linear: AutocastLinear,
        nn.Conv2d: AutocastConv2d,
        nn.ConvTranspose2d: AutocastConvTranspose2d,
        nn.LayerNorm: Float32LayerNorm,
        nn.GroupNorm: Float32GroupNorm,
        nn.GELU: Float32GELU,
    }
    if not bf16:
        # FP32 restores the plain layers a previous BF16 policy replaced.
        implementations = {value: key for key, value in implementations.items()}
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in modules:
        implementation = implementations.get(type(module))
        replacement = copy_layer(module, implementation) if implementation else module
        if (
            bf16
            and id(module) not in fp32_projections
            and isinstance(replacement, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))
        ):
            replacement.set_dtype(mx.bfloat16)
        if replacement is not module:
            replacements.append((name, replacement))
    getattr(model, "update_modules")(tree_unflatten(replacements))
