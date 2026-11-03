# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Build SAM 3 image models and independent-frame video predictors on MLX."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, TypedDict, Unpack

# Importing the shared runtime sets its full-FP32 default before model modules load.
from medmlx_core.runtime import import_mlx
from medmlx_core.weights import weights_dir

from sam3_mlx.checkpoint_loading import (
    PathLikeStr, load_image_checkpoint, read_interactive_checkpoint,
)
from sam3_mlx.checkpoint_mapping import flatten_model_parameters
from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_video_inference import ProcessorFactory, Sam3VideoInference
from sam3_mlx.model_components import (
    create_dot_product_scoring, create_geometry_encoder, create_sam3_model,
    create_sam3_transformer, create_segmentation_head, create_text_encoder,
    create_vision_backbone, create_vl_backbone,
)
from sam3_mlx.precision import apply_precision_policy, parse_precision

if TYPE_CHECKING:
    import mlx.core as mx

    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

# MedMLX model ID of the interactive head; medmlx-core resolves it to MedMLX/sam3-interactive.
INTERACTIVE_WEIGHTS_ID = "sam3-interactive"

# FP32 checkpoint tensors are materialized in batches of about this many bytes.
_LOAD_BATCH_BYTES = 64 << 20

class Sam3PredictorOptions(TypedDict, total=False):
    """Keyword options that ``build_sam3_predictor`` forwards to ``build_sam3_video_predictor``."""

    strict_state_dict_loading: bool
    image_model: object | None
    resolution: int
    confidence_threshold: float
    processor_factory: ProcessorFactory | None
    frame_feature_cache_size: int


def default_bpe_path() -> str:
    """Return the path of the packaged BPE vocabulary used by the text tokenizer."""
    return os.path.join(
        os.path.dirname(__file__),
        "assets",
        "bpe_simple_vocab_16e6.txt.gz",
    )


def _assemble_sam3_image_components(
    *,
    bpe_path: str,
    enable_inst_interactivity: bool,
    compile: bool,
) -> Sam3Image:
    """Construct the image module graph without loading weights or setting policy."""

    vision_encoder = create_vision_backbone(
        enable_inst_interactivity=enable_inst_interactivity,
    )
    text_encoder = create_text_encoder(bpe_path)
    backbone = create_vl_backbone(
        vision_encoder,
        text_encoder,
        compile_visual=compile,
    )
    transformer = create_sam3_transformer()
    dot_product_scoring = create_dot_product_scoring()
    segmentation_head = create_segmentation_head()
    input_geometry_encoder = create_geometry_encoder()
    inst_interactive_predictor = (
        SAM3InteractiveImagePredictor() if enable_inst_interactivity else None
    )
    return create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring=dot_product_scoring,
        inst_interactive_predictor=inst_interactive_predictor,
        compile_grounding=compile,
    )


def build_sam3_image_model(
    bpe_path: PathLikeStr | None = None,
    eval_mode: bool = True,
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    precision: str = "fp32",
    interactive_checkpoint_path: PathLikeStr | None = None,
    strict_checkpoint_loading: bool = True,
) -> Sam3Image:
    """Build the SAM 3 image model and load its weights.

    Weights come from ``checkpoint_path`` (an MLX ``.safetensors`` file) when it
    is given. Otherwise, with ``load_from_HF``, they come from ``model.safetensors``
    in the MedMLX cache, ``medmlx_core.weights.weights_dir("sam3")``
    (``~/.cache/medmlx/sam3``), which downloads the pinned
    ``mlx-community/sam3-image`` revision on first use. With neither source the
    image model is randomly initialized and a warning is emitted. With
    ``enable_inst_interactivity`` and ``load_from_HF``, the interactive head
    comes from ``interactive_checkpoint_path`` or, when that is omitted, from
    ``model.safetensors`` in ``weights_dir("sam3-interactive")``, which
    downloads ``MedMLX/sam3-interactive`` on first use.

    Args:
        bpe_path: Tokenizer vocabulary; defaults to the packaged file.
        eval_mode: Put the model in evaluation mode.
        checkpoint_path: Local MLX ``.safetensors`` file, not a directory.
        load_from_HF: Without ``checkpoint_path``, load the cached default
            checkpoint, downloading it if needed; likewise the interactive head
            when it is enabled without ``interactive_checkpoint_path``. False
            leaves random initialization and downloads nothing.
        enable_inst_interactivity: Also build the SAM1-style interactive
            point/box/mask predictor (``Sam3Image.predict_inst``) and the SAM2
            feature pyramid. Without ``load_from_HF``, strict loading of a base
            checkpoint then needs ``interactive_checkpoint_path``.
        compile: Compile the visual backbone and grounding graphs with MLX
            (opt-in; image model only).
        precision: ``"fp32"`` (default) or ``"bf16"``. BF16 loads the FP32
            checkpoint, runs autocast projections in BF16 and keeps the
            segmentation head in FP32.
        interactive_checkpoint_path: The converted interactive head, a flat MLX
            ``.safetensors`` file of ``inst_interactive_predictor.*`` tensors;
            requires ``enable_inst_interactivity=True``. It is checked strictly
            and combines with any base checkpoint, including the cached default.
        strict_checkpoint_loading: Require every non-generated parameter to be
            covered. ``False`` allows partial loading for development only; the
            audit is stored on ``model.checkpoint_load_report``.

    Raises:
        ValueError: An unknown precision, ``interactive_checkpoint_path``
            without ``enable_inst_interactivity``, a wrong interactive file, or
            checkpoint coverage/shape errors.
    """
    import_mlx()
    policy = parse_precision(precision)
    if interactive_checkpoint_path is not None and not enable_inst_interactivity:
        raise ValueError(
            "interactive_checkpoint_path requires enable_inst_interactivity=True."
        )
    if checkpoint_path is None and load_from_HF:
        checkpoint_path = weights_dir("sam3") / "model.safetensors"
    if enable_inst_interactivity and interactive_checkpoint_path is None and load_from_HF:
        # mlx-community/sam3-image holds only the detector; the head converted
        # from sam3.pt's tracker (scripts/convert_interactive_head.py) is
        # published separately as MedMLX/sam3-interactive.
        interactive_checkpoint_path = (
            weights_dir(INTERACTIVE_WEIGHTS_ID) / "model.safetensors"
        )
    model = _assemble_sam3_image_components(
        bpe_path=os.fspath(default_bpe_path() if bpe_path is None else bpe_path),
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
    )
    if checkpoint_path is not None:
        load_image_checkpoint(
            model,
            checkpoint_path,
            interactive_checkpoint_path=interactive_checkpoint_path,
            strict=strict_checkpoint_loading,
        )
    else:
        warnings.warn(
            "build_sam3_image_model returned a randomly initialized image model: "
            "no base checkpoint was loaded (checkpoint_path=None, "
            "load_from_HF=False). Pass checkpoint_path= or keep load_from_HF=True "
            "to load real weights; keep suppressing this warning only for "
            "deliberately checkpoint-free construction.",
            stacklevel=2,
        )
        if interactive_checkpoint_path is not None:
            interactive = read_interactive_checkpoint(model, interactive_checkpoint_path)
            model.load_weights(list(interactive.items()), strict=False)
    apply_precision_policy(model, policy)
    if checkpoint_path is not None:
        # Materialize after the cast. A BF16 load evaluates one tensor at a time, so
        # it holds at most one FP32 source tensor beyond the cast weights. FP32 has no
        # cast, so it evaluates ~64 MB batches: one evaluation per tensor spent more
        # time synchronizing than reading.
        limit = _LOAD_BATCH_BYTES if policy == "fp32" else 0
        batch: list[mx.array] = []
        size = 0
        for value in flatten_model_parameters(model).values():
            batch.append(value)
            size += value.nbytes
            if size >= limit:
                evaluate_boundary(batch)
                batch, size = [], 0
        evaluate_boundary(batch)
    if eval_mode:
        model.eval()
    return model


def build_sam3_video_predictor(
    checkpoint_path: PathLikeStr | None = None,
    bpe_path: PathLikeStr | None = None,
    strict_state_dict_loading: bool = True,
    *,
    image_model: object | None = None,
    resolution: int = 1008,
    confidence_threshold: float = 0.5,
    load_from_HF: bool = True,
    processor_factory: ProcessorFactory | None = None,
    frame_feature_cache_size: int = 4,
) -> Sam3VideoPredictor:
    """Build the selected-frame SAM 3 video predictor.

    The arguments are those of ``build_sam3_video_model``. The predictor runs
    image inference independently on each frame; sessions started with
    ``track=True`` link the detections across propagated frames.
    """
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

    return Sam3VideoPredictor(
        build_sam3_video_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            bpe_path=bpe_path,
            strict_state_dict_loading=strict_state_dict_loading,
            image_model=image_model,
            resolution=resolution,
            confidence_threshold=confidence_threshold,
            processor_factory=processor_factory,
            frame_feature_cache_size=frame_feature_cache_size,
        )
    )


def build_sam3_video_model(
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    bpe_path: PathLikeStr | None = None,
    strict_state_dict_loading: bool = True,
    image_model: object | None = None,
    resolution: int = 1008,
    confidence_threshold: float = 0.5,
    processor_factory: ProcessorFactory | None = None,
    frame_feature_cache_size: int = 4,
) -> Sam3VideoInference:
    """Wrap an image model for independent, bounded-memory frame inference.

    Without ``image_model``, the image model is built by ``build_sam3_image_model``
    from ``checkpoint_path``, ``load_from_HF``, ``bpe_path`` and
    ``strict_state_dict_loading`` (its ``strict_checkpoint_loading``). A supplied
    ``image_model`` is used as given. ``resolution`` is the square model input
    size, ``confidence_threshold`` the default score threshold,
    ``processor_factory`` builds each session's image processor (default
    ``Sam3Processor``) and ``frame_feature_cache_size`` bounds the frame encodings
    kept per session.

    Raises:
        ValueError: If both ``checkpoint_path`` and ``image_model`` are given.
    """
    if image_model is None:
        image_model = build_sam3_image_model(
            bpe_path=bpe_path, eval_mode=True,
            checkpoint_path=checkpoint_path, load_from_HF=load_from_HF,
            strict_checkpoint_loading=strict_state_dict_loading,
        )
    elif checkpoint_path is not None:
        raise ValueError("Pass checkpoint_path or image_model, not both.")
    else:
        import_mlx()
    return Sam3VideoInference(
        image_model,
        resolution=resolution,
        confidence_threshold=confidence_threshold,
        processor_factory=processor_factory,
        frame_feature_cache_size=frame_feature_cache_size,
    )


def build_sam3_predictor(
    checkpoint_path: PathLikeStr | None = None,
    bpe_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    **kwargs: Unpack[Sam3PredictorOptions],
) -> Sam3VideoPredictor:
    """Build the SAM 3 independent-frame predictor (see ``build_sam3_video_predictor``)."""
    return build_sam3_video_predictor(
        checkpoint_path=checkpoint_path, bpe_path=bpe_path,
        load_from_HF=load_from_HF, **kwargs,
    )
