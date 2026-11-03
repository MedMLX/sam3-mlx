"""The published checkpoint reproduces upstream SAM 3 text-prompt detections.

``fixtures/upstream_text_prompts.json`` holds upstream PyTorch SAM 3's
detections for each medical fixture, recorded on the CPU in FP32 by
``scripts/compare_upstream_image.py --record``. The test loads the real 3.4 GB
weights, so it runs only when they are already on disk: in the MedMLX cache
(``medmlx setup sam3``) or the Hugging Face cache. It never downloads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache
from PIL import Image

from sam3_mlx import build_sam3_image_model
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_image_processor import Sam3Processor
from tests._json_contracts import require_list, require_mapping
from tests._paths import FIXTURE_ROOT, MEDICAL_FIXTURE_ROOT

_REPO = "mlx-community/sam3-image"
_REVISION = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
# Observed MLX-vs-upstream differences are at most 4.3e-5 in score, 0.003 px in
# box and 2 pixels per mask; the bounds leave room for Metal kernel changes
# while catching a changed checkpoint or a broken graph.
_SCORE_TOLERANCE = 1e-3
_BOX_TOLERANCE_PX = 0.5
_MASK_AREA_TOLERANCE = 1e-3


def _cached_checkpoint() -> Path | None:
    """Return the pinned checkpoint if either cache already holds it."""
    cache = Path(os.environ.get("MEDMLX_ASSET_CACHE", Path.home() / ".cache/medmlx"))
    medmlx = cache.expanduser() / "sam3" / _REPO.replace("/", "--") / "model.safetensors"
    if medmlx.is_file():
        return medmlx
    hub = try_to_load_from_cache(_REPO, "model.safetensors", revision=_REVISION)
    return Path(hub) if isinstance(hub, str) else None


CHECKPOINT = _cached_checkpoint()
REFERENCE = require_mapping(
    json.loads((FIXTURE_ROOT / "upstream_text_prompts.json").read_text("utf-8")),
    "upstream_text_prompts.json",
)
CASES = [
    require_mapping(item, "reference image")
    for item in require_list(REFERENCE["images"], "reference images")
]


@pytest.fixture(scope="module")
def model() -> Sam3Image:
    assert CHECKPOINT is not None
    return build_sam3_image_model(checkpoint_path=CHECKPOINT)


@pytest.mark.skipif(CHECKPOINT is None, reason="SAM 3 weights are not cached locally")
@pytest.mark.parametrize("case", CASES, ids=[str(case["file"]) for case in CASES])
def test_text_prompt_detections_match_upstream(
    model: Sam3Image, case: dict[str, object]
) -> None:
    processor = Sam3Processor(model)
    with Image.open(MEDICAL_FIXTURE_ROOT / str(case["file"])) as image:
        state = processor.set_text_prompt(str(case["text_prompt"]), processor.set_image(image))
    scores = to_numpy(state["scores"], dtype=np.float64)
    boxes = to_numpy(state["boxes"], dtype=np.float64).reshape(-1, 4)
    areas = to_numpy(state["masks"]).sum(axis=(1, 2, 3))

    expected_scores = np.asarray(require_list(case["scores"], "scores"), dtype=np.float64)
    assert scores.shape == expected_scores.shape
    np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=_SCORE_TOLERANCE)
    np.testing.assert_allclose(
        boxes,
        np.asarray(require_list(case["boxes_xyxy"], "boxes"), dtype=np.float64).reshape(-1, 4),
        rtol=0,
        atol=_BOX_TOLERANCE_PX,
    )
    np.testing.assert_allclose(
        areas,
        np.asarray(require_list(case["mask_areas"], "mask areas"), dtype=np.float64),
        rtol=_MASK_AREA_TOLERANCE,
        atol=0,
    )
