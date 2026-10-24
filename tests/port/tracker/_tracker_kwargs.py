"""Canonical tracker-construction kwargs shared by the tracker port tests."""

from typing import TypedDict

from sam3_mlx.model.sam3_tracker_base import SamMaskDecoderExtraArgs


class _TrackerKwargs(TypedDict):
    image_size: int
    num_maskmem: int
    backbone_stride: int
    multimask_output_in_sam: bool
    forward_backbone_per_frame_for_eval: bool
    multimask_output_for_tracking: bool
    multimask_min_pt_num: int
    multimask_max_pt_num: int
    non_overlap_masks_for_mem_enc: bool
    max_cond_frames_in_attn: int
    sam_mask_decoder_extra_args: SamMaskDecoderExtraArgs


# kwargs mirror the official build_tracker(...) call at upstream commit
# 2814fa619404a722d03e9a012e083e4f293a4e53.
TRACKER_KWARGS = _TrackerKwargs(
    image_size=1008,
    num_maskmem=7,
    backbone_stride=14,
    multimask_output_in_sam=True,
    forward_backbone_per_frame_for_eval=True,
    multimask_output_for_tracking=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    non_overlap_masks_for_mem_enc=False,
    max_cond_frames_in_attn=4,
    sam_mask_decoder_extra_args={
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    },
)
