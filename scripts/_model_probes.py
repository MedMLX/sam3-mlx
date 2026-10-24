"""Typed access to protected model operations measured by script benchmarks."""

from __future__ import annotations

import mlx.core as mx
from collections.abc import Mapping
from sam3_mlx.model.data_misc import FindStage
from sam3_mlx.model import sam3_image as img
from sam3_mlx.model import sam3_image_processor as proc
from sam3_mlx.model import vitdet as vit


class Sam3ImageProbe(img.Sam3Image):
    @staticmethod
    def encode_prompt(
        model: img.Sam3Image,
        backbone_out: img.BackboneOutput,
        find_input: FindStage,
        geometric_prompt: img.Prompt,
        visual_prompt_embed: mx.array | None = None,
        visual_prompt_mask: mx.array | None = None,
        encode_text: bool = True,
        prev_mask_pred: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, img.BackboneOutput]:
        return model._encode_prompt(
            backbone_out,
            img.image_prompt_indices(find_input),
            geometric_prompt,
            visual_prompt_embed,
            visual_prompt_mask,
            encode_text,
            prev_mask_pred,
        )

    @staticmethod
    def run_encoder(
        model: img.Sam3Image,
        backbone_out: img.BackboneOutput,
        find_input: FindStage,
        prompt: mx.array,
        prompt_mask: mx.array,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[img.BackboneOutput, img.EncoderOutput, img.ImageFeatures]:
        return model._run_encoder(
            backbone_out,
            img.image_prompt_indices(find_input),
            prompt,
            prompt_mask,
            encoder_extra_kwargs,
        )

    @staticmethod
    def run_decoder(
        model: img.Sam3Image,
        pos_embed: mx.array,
        memory: mx.array,
        src_mask: mx.array | None,
        out: img.Output,
        prompt: mx.array,
        prompt_mask: mx.array,
        encoder_out: img.EncoderOutput,
    ) -> tuple[img.Output, mx.array]:
        return model._run_decoder(
            pos_embed, memory, src_mask, out, prompt, prompt_mask, encoder_out
        )

    @staticmethod
    def run_segmentation_heads(
        model: img.Sam3Image,
        out: img.Output,
        backbone_out: img.BackboneOutput,
        img_ids: mx.array,
        vis_feat_sizes: list[tuple[int, int]],
        encoder_hidden_states: mx.array,
        prompt: mx.array,
        prompt_mask: mx.array,
        hs: mx.array,
    ) -> None:
        return model._run_segmentation_heads(
            out,
            backbone_out,
            img_ids,
            vis_feat_sizes,
            encoder_hidden_states,
            prompt,
            prompt_mask,
            hs,
        )

    @staticmethod
    def get_dummy_prompt(model: img.Sam3Image, num_prompts: int = 1) -> img.Prompt:
        return model._get_dummy_prompt(num_prompts)


class Sam3ProcessorProbe(proc.Sam3Processor):
    @staticmethod
    def patch_interactive_backbone_features(
        model: proc.Sam3Processor, backbone_out: Mapping[str, object]
    ) -> None:
        return model._patch_interactive_backbone_features(backbone_out)

    @staticmethod
    def forward_grounding(
        model: proc.Sam3Processor, state: proc.ProcessorState
    ) -> proc.ProcessorState:
        return model._forward_grounding(state)


class ViTProbe(vit.ViT):
    @staticmethod
    def prepare_tokens(
        model: vit.ViT, x: mx.array | vit.NestedTensor
    ) -> tuple[mx.array, int, bool, mx.array | None]:
        return model._prepare_tokens(x)
