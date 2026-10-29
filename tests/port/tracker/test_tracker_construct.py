"""Structural parity for the ported Sam3TrackerBase construction layer.

This pins the tracker construction layer against the official module shape. It pins the
constructed MLX parameter tree against the SAM2-canonical tracker structure so
later checkpoint-mapping and forward-parity increments build on a stable base.
"""

from collections.abc import Callable
from typing import cast

import mlx.core as mx
import pytest

from sam3_mlx.checkpoint_mapping import flatten_model_parameters
from sam3_mlx.model.sam3_tracker_base import Sam3TrackerBase, concat_points
from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3_mlx.model_builder import build_tracker
from sam3_mlx.model_components import (
    create_tracker_maskmem_backbone,
    create_tracker_transformer,
)
from tests.port.tracker._tracker_kwargs import TRACKER_KWARGS


@pytest.fixture(scope="module")
def tracker_base() -> Sam3TrackerBase:
    maskmem_backbone = create_tracker_maskmem_backbone()
    transformer = create_tracker_transformer()
    return Sam3TrackerBase(
        backbone=None,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        **TRACKER_KWARGS,
    )


def test_component_builders_match_official_shape() -> None:
    transformer = create_tracker_transformer()
    # the tracker memory-attention transformer is encoder-only
    assert transformer.decoder is None
    assert transformer.d_model == 256

    maskmem_backbone = create_tracker_maskmem_backbone()
    # out_dim=64 introduces channel compression -> mem_dim must read back as 64
    maskmem_params = flatten_model_parameters(maskmem_backbone)
    assert maskmem_params["out_proj.conv.weight"].shape[0] == 64


def test_tracker_base_constructs_with_expected_dims(
    tracker_base: Sam3TrackerBase,
) -> None:
    assert tracker_base.hidden_dim == 256
    assert tracker_base.mem_dim == 64
    assert tracker_base.num_maskmem == 7
    assert tracker_base.device == "mlx"
    # derived sizes from the SAM2 mask geometry
    assert tracker_base.sam_image_embedding_size == 72
    assert tracker_base.low_res_mask_size == 1008 // 14 * 4


def test_build_tracker_returns_official_shaped_predictor_without_checkpoint() -> None:
    tracker = build_tracker(apply_temporal_disambiguation=True, with_backbone=False)

    assert isinstance(tracker, Sam3TrackerPredictor)
    assert tracker.backbone is None
    assert tracker.use_memory_selection is True
    assert tracker.clear_non_cond_mem_around_input is True
    assert tracker.non_overlap_masks_for_output is False
    assert tracker.max_cond_frames_in_attn == 4


def test_param_tree_top_level_groups(tracker_base: Sam3TrackerBase) -> None:
    groups = {key.split(".")[0] for key in flatten_model_parameters(tracker_base)}
    expected = {
        "mask_downsample",
        "maskmem_backbone",
        "maskmem_tpos_enc",
        "no_mem_embed",
        "no_mem_pos_enc",
        "no_obj_embed_spatial",
        "no_obj_ptr",
        "obj_ptr_proj",
        "obj_ptr_tpos_proj",
        "sam_mask_decoder",
        "sam_prompt_encoder",
        "transformer",
    }
    assert groups == expected


def test_bare_parameter_shapes_match_sam2_canonical(
    tracker_base: Sam3TrackerBase,
) -> None:
    params = flatten_model_parameters(tracker_base)
    assert tuple(params["maskmem_tpos_enc"].shape) == (7, 1, 1, 64)
    assert tuple(params["no_mem_embed"].shape) == (1, 1, 256)
    assert tuple(params["no_mem_pos_enc"].shape) == (1, 1, 256)
    assert tuple(params["no_obj_ptr"].shape) == (1, 256)
    assert tuple(params["no_obj_embed_spatial"].shape) == (1, 64)
    # obj_ptr_proj is a 3-layer MLP (Linear weight+bias per layer)
    assert tuple(params["obj_ptr_tpos_proj.weight"].shape) == (64, 256)


def test_backbone_required_and_direct_mask_validates_shape(
    tracker_base: Sam3TrackerBase,
) -> None:
    with pytest.raises(RuntimeError, match="requires a backbone"):
        tracker_base.forward_image(mx.zeros((1, 3, 1008, 1008)))
    use_mask_value: object = getattr(tracker_base, "_use_mask_as_output", None)
    if not callable(use_mask_value):
        raise AssertionError("tracker _use_mask_as_output must be callable")
    use_mask_as_output = cast(
        Callable[[mx.array, list[mx.array] | None, mx.array], object],
        use_mask_value,
    )
    with pytest.raises(ValueError, match="mask_inputs"):
        use_mask_as_output(
            mx.zeros((1, 256, 72, 72)),
            None,
            mx.zeros((1, 72, 72)),
        )


def test_concat_points_mlx_roundtrip() -> None:
    new = concat_points(
        None,
        mx.zeros((1, 1, 2)),
        mx.ones((1, 1), dtype=mx.int32),
    )
    merged = concat_points(
        new,
        mx.ones((1, 1, 2)),
        mx.zeros((1, 1), dtype=mx.int32),
    )
    assert tuple(merged["point_coords"].shape) == (1, 2, 2)
    assert tuple(merged["point_labels"].shape) == (1, 2)
