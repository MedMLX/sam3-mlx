"""Shared fake image-processor doubles for the API surface tests.

Both variants share the official ``Sam3Processor`` factory surface
(``__init__(image_model, resolution=..., confidence_threshold=...)`` plus the
prompt/grounding method set). Behavior is parameterized by class attributes so
subclasses keep the same constructor signature:

- ``set_image_payload``: ``"empty"`` returns zero detections, ``"single"``
  returns one synthetic object (mask/box/score).
- ``record_prompts``: when ``False`` the prompt methods return the state
  untouched; when ``True`` they copy the state and store the prompt, matching
  the payloads the public-API tests assert on.
"""

import numpy as np


class FakeImageProcessor:
    set_image_payload: str = "empty"
    record_prompts: bool = False

    def __init__(
        self,
        image_model: object,
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.image_model = image_model
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold

    def set_image(self, image: object) -> dict[str, object]:
        del image
        if self.set_image_payload == "single":
            mask = np.zeros((1, 480, 640), dtype=bool)
            mask[:, 10:20, 30:45] = True
            return {
                "masks": mask,
                "boxes": np.array([[30.0, 10.0, 45.0, 20.0]], dtype=np.float32),
                "scores": np.array([0.91], dtype=np.float32),
            }
        return {
            "masks": np.zeros((0, 480, 640), dtype=bool),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    def set_text_prompt(
        self,
        prompt: str,
        state: dict[str, object],
        *,
        run_grounding: bool = True,
        text_outputs: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del run_grounding, text_outputs
        if not self.record_prompts:
            del prompt
            return state
        state = dict(state)
        state["text_prompt"] = prompt
        return state

    def add_geometric_prompt(
        self,
        box: object,
        label: bool,
        state: dict[str, object],
        *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        if not self.record_prompts:
            del box, label
            return state
        state = dict(state)
        state["box_prompt"] = (box, label)
        return state

    def add_point_prompt(
        self,
        point: object,
        label: bool,
        state: dict[str, object],
        *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        if not self.record_prompts:
            del point, label
            return state
        state = dict(state)
        state["point_prompt"] = (point, label)
        return state

    def run_grounding(self, state: dict[str, object]) -> dict[str, object]:
        return state


class RecordingFakeImageProcessor(FakeImageProcessor):
    set_image_payload = "single"
    record_prompts = True
