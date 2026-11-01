"""Fake image processor with the ``Sam3Processor`` factory surface.

``set_image`` returns one synthetic object (mask, box and score) for a 640 x 480
frame; the prompt methods record the prompt in a copy of the state.
"""

import numpy as np

FRAME_SIZE = (640, 480)


class FakeImageProcessor:
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
        mask = np.zeros((1, FRAME_SIZE[1], FRAME_SIZE[0]), dtype=bool)
        mask[:, 10:20, 30:45] = True
        return {
            "masks": mask,
            "boxes": np.array([[30.0, 10.0, 45.0, 20.0]], dtype=np.float32),
            "scores": np.array([0.91], dtype=np.float32),
        }

    def set_text_prompt(
        self,
        prompt: str,
        state: dict[str, object],
        *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        return {**state, "text_prompt": prompt}

    def add_geometric_prompt(
        self,
        box: list[float],
        label: bool,
        state: dict[str, object],
        *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        return {**state, "box_prompt": (box, label)}

    def add_point_prompt(
        self,
        point: list[float],
        label: bool,
        state: dict[str, object],
        *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        return {**state, "point_prompt": (point, label)}

    def predict(self, state: dict[str, object]) -> dict[str, object]:
        return state
