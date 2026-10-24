"""Frozen provenance and metric gates for SAM 3.1 temporal parity."""

from __future__ import annotations

from typing import Mapping

SAM31_CODE_REPO = "https://github.com/facebookresearch/sam3.git"
SAM31_CODE_REVISION = "2814fa619404a722d03e9a012e083e4f293a4e53"
SAM31_CHECKPOINT_REPO = "facebook/sam3.1"
SAM31_CHECKPOINT_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_CHECKPOINT_FILENAME = "sam3.1_multiplex.pt"

TEMPORAL_PARITY_THRESHOLDS = {
    "iou_min": 0.95,
    "dice_min": 0.97,
    "boundary_f1_min": 0.90,
    "identity_errors_max": 0.0,
}


def official_temporal_bindings(
    sequence: Mapping[str, object],
    *,
    source_commit: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "official-pytorch-cuda",
        "sequence_id": sequence["sequence_id"],
        "adapter_source": {"commit": source_commit},
        "official_code": {
            "repo": SAM31_CODE_REPO,
            "revision": SAM31_CODE_REVISION,
        },
        "checkpoint": {
            "repo": SAM31_CHECKPOINT_REPO,
            "revision": SAM31_CHECKPOINT_REVISION,
        },
    }


def mlx_temporal_bindings(
    sequence: Mapping[str, object],
    *,
    source_commit: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "sam3-mlx",
        "sequence_id": sequence["sequence_id"],
        "adapter_source": {"commit": source_commit},
        "official_code": {
            "repo": SAM31_CODE_REPO,
            "revision": SAM31_CODE_REVISION,
        },
        "source_checkpoint": {
            "repo": SAM31_CHECKPOINT_REPO,
            "revision": SAM31_CHECKPOINT_REVISION,
        },
    }
