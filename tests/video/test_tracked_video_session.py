"""Tracked video sessions: persistent ids, aligned outputs, directions and restarts."""

from __future__ import annotations

from tests._images import chest_frame

import warnings
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from medmlx_core.runtime import require_mlx_runtime
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx.model.sam3_video_inference import Sam3VideoInference, VideoOutputs
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from sam3_mlx.tracking import LOW_SCORE_THRESH
from tests.api._stubs import FRAME_SIZE
from tests._paths import MEDICAL_FIXTURE_ROOT

WIDTH, HEIGHT = FRAME_SIZE
type SceneObject = tuple[str, tuple[float, float, float, float], float]
type Scene = Callable[[int], list[SceneObject]]


def _crossing(frame_idx: int) -> list[SceneObject]:
    """Two boxes that meet at frame 7; listed in reverse order on odd frames, so frame-local ids flip."""
    objects: list[SceneObject] = [
        ("left", (150.0 + 10 * frame_idx, 100.0, 210.0 + 10 * frame_idx, 160.0), 0.9),
        ("right", (290.0 - 10 * frame_idx, 120.0, 350.0 - 10 * frame_idx, 180.0), 0.8),
    ]
    return objects[::-1] if frame_idx % 2 else objects


class _SceneProcessor:
    """Image processor stub whose detections depend on the frame index stored in pixel (0, 0)."""

    def __init__(self, scene: Scene, confidence_threshold: float) -> None:
        self.scene = scene
        self.confidence_threshold = confidence_threshold
        self.thresholds: list[float] = []

    def set_image(self, image: Image.Image) -> dict[str, object]:
        pixel = image.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        return {"frame_idx": int(pixel[0])}

    def set_text_prompt(
        self, prompt: str, state: dict[str, object], *, run_grounding: bool = True,
    ) -> dict[str, object]:
        del prompt, run_grounding
        return state

    def add_geometric_prompt(
        self, box: list[float], label: bool, state: dict[str, object], *, run_grounding: bool = True,
    ) -> dict[str, object]:
        del box, label, run_grounding
        return state

    def add_point_prompt(
        self, point: list[float], label: bool, state: dict[str, object], *, run_grounding: bool = True,
    ) -> dict[str, object]:
        del point, label, run_grounding
        return state

    def predict(self, state: dict[str, object]) -> dict[str, object]:
        self.thresholds.append(self.confidence_threshold)
        frame_idx = state["frame_idx"]
        assert isinstance(frame_idx, int)
        objects = [item for item in self.scene(frame_idx) if item[2] > self.confidence_threshold]
        masks = np.zeros((len(objects), 1, HEIGHT, WIDTH), dtype=bool)
        for index, (_, (x1, y1, x2, y2), _) in enumerate(objects):
            masks[index, 0, int(y1) : int(y2), int(x1) : int(x2)] = True
        return {
            **state,
            "masks": masks,
            "boxes": np.array([box for _, box, _ in objects], dtype=np.float32).reshape(-1, 4),
            "scores": np.array([score for _, _, score in objects], dtype=np.float32),
        }


def _model(scene: Scene) -> tuple[Sam3VideoInference, list[_SceneProcessor]]:
    processors: list[_SceneProcessor] = []

    def factory(
        image_model: object, *, resolution: int, confidence_threshold: float,
    ) -> _SceneProcessor:
        del image_model, resolution
        processors.append(_SceneProcessor(scene, confidence_threshold))
        return processors[-1]

    return Sam3VideoInference(image_model=object(), processor_factory=factory), processors


def _frames(count: int) -> list[Image.Image]:
    """Use radiograph frames with one marked pixel carrying the stub's frame index."""
    return [chest_frame(FRAME_SIZE, marker=index) for index in range(count)]


def _labeled(scene: Scene, frame_idx: int, outputs: VideoOutputs) -> dict[str, int]:
    """Map each returned object to its scene label, checking mask, box and score alignment."""
    labels: dict[str, int] = {}
    scale = np.array([WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32)
    for k, obj_id in enumerate(outputs["out_obj_ids"].tolist()):
        x, y, w, h = (outputs["out_boxes_xywh"][k] * scale).tolist()
        matches = [
            (label, score) for label, box, score in scene(frame_idx)
            if np.allclose(box, (x, y, x + w, y + h), atol=1e-3)
        ]
        assert len(matches) == 1
        label, score = matches[0]
        assert outputs["out_probs"][k] == np.float32(score)
        rows, cols = np.nonzero(outputs["out_binary_masks"][k])
        assert (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1) == (
            round(x), round(y), round(x + w), round(y + h)
        )
        labels[label] = int(obj_id)
    return labels


def _propagate(
    model: Sam3VideoInference, scene: Scene, *, track: bool,
) -> list[tuple[int, dict[str, int]]]:
    """Prompt frame 0 of an eight-frame clip and propagate forward."""
    state = model.init_state(_frames(8), track=track)
    model.add_prompt(state, frame_idx=0, text_str="object")
    return [
        (frame_idx, _labeled(scene, frame_idx, outputs))
        for frame_idx, outputs in model.propagate_in_video(state)
    ]


def test_tracked_session_keeps_ids_and_aligns_masks_boxes_and_scores() -> None:
    model, _ = _model(_crossing)
    untracked = _propagate(model, _crossing, track=False)
    assert {labels["left"] for _, labels in untracked} == {0, 1}  # frame-local ids flip

    tracked = _propagate(model, _crossing, track=True)
    assert [frame_idx for frame_idx, _ in tracked] == list(range(8))
    assert all(labels == {"left": 0, "right": 1} for _, labels in tracked)


def test_tracking_uses_low_score_detections_and_holds_back_new_objects() -> None:
    """A dimmed object stays tracked below the threshold; a late one waits three frames."""

    def scene(frame_idx: int) -> list[SceneObject]:
        objects = _crossing(frame_idx)
        if frame_idx in (3, 4):
            objects = [(label, box, 0.3 if label == "left" else score) for label, box, score in objects]
        if frame_idx >= 2:
            objects.append(("late", (300.0, 300.0, 360.0, 360.0), 0.7))
        objects.append(("faint", (500.0, 400.0, 560.0, 460.0), 0.3))
        return objects

    model, processors = _model(scene)
    untracked = _propagate(model, scene, track=False)
    assert ["left" in labels for _, labels in untracked] == [f not in (3, 4) for f in range(8)]
    assert processors[-1].thresholds == [0.5] * 9

    tracked = _propagate(model, scene, track=True)
    assert all(labels["left"] == 0 and labels["right"] == 1 for _, labels in tracked)
    assert ["late" in labels for _, labels in tracked] == [f >= 5 for f in range(8)]
    assert {labels["late"] for _, labels in tracked if "late" in labels} == {2}
    assert all("faint" not in labels for _, labels in tracked)
    # The prompt frame uses the output threshold; propagation also fetches low scores.
    assert processors[-1].thresholds == [0.5] + [LOW_SCORE_THRESH] * 8


def test_both_directions_share_ids_through_the_predictor() -> None:
    model, _ = _model(_crossing)
    predictor = Sam3VideoPredictor(model)
    session_id = predictor.start_session(_frames(8), track=True)["session_id"]
    predictor.add_prompt(session_id, frame_idx=4, text="object")
    responses = list(predictor.propagate_in_video(session_id, propagation_direction="both"))
    assert [item["frame_index"] for item in responses] == [4, 5, 6, 7, 3, 2, 1, 0]
    labels = [_labeled(_crossing, item["frame_index"], item["outputs"]) for item in responses]
    assert all(frame == labels[0] for frame in labels)
    assert sorted(labels[0].values()) == [0, 1]
    predictor.close_session(session_id, run_gc_collect=False)


def test_chunks_continue_while_new_prompts_and_jumps_restart_tracking() -> None:
    def scene(frame_idx: int) -> list[SceneObject]:
        objects = _crossing(frame_idx)
        if frame_idx >= 2:
            objects.append(("late", (300.0, 300.0, 360.0, 360.0), 0.7))
        return objects

    model, _ = _model(scene)
    whole = _propagate(model, scene, track=True)
    state = model.init_state(_frames(8), track=True)
    model.add_prompt(state, frame_idx=0, text_str="object")
    chunked = [
        (frame_idx, _labeled(scene, frame_idx, outputs))
        for start, limit in ((0, 3), (4, None))
        for frame_idx, outputs in model.propagate_in_video(
            state, start_frame_idx=start, max_frame_num_to_track=limit
        )
    ]
    assert chunked == whole

    # A new prompt restarts the run, so the same propagation repeats exactly.
    model.add_prompt(state, frame_idx=0, text_str="object")
    assert [
        (frame_idx, _labeled(scene, frame_idx, outputs))
        for frame_idx, outputs in model.propagate_in_video(state)
    ] == whole

    # A jump starts a new run: its first frame's objects get ids 0, 1, 2 at once.
    restarted = [
        _labeled(scene, frame_idx, outputs)
        for frame_idx, outputs in model.propagate_in_video(state, start_frame_idx=6)
    ]
    assert restarted == [restarted[0]] * 2 and sorted(restarted[0].values()) == [0, 1, 2]


def _seeded_model() -> object:
    from sam3_mlx.model_builder import build_sam3_image_model

    mx.random.seed(20261012)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_sam3_image_model(load_from_HF=False)


def _rows(outputs: VideoOutputs) -> list[tuple[bytes, float, bytes]]:
    return [
        (outputs["out_boxes_xywh"][k].tobytes(), float(outputs["out_probs"][k]),
         outputs["out_binary_masks"][k].tobytes())
        for k in range(outputs["out_obj_ids"].shape[0])
    ]


def test_seeded_model_tracks_a_static_clip() -> None:
    """Real (checkpoint-free, seeded) forward passes: tracked outputs are untracked detections."""
    require_mlx_runtime()
    model = Sam3VideoInference(_seeded_model(), resolution=336, confidence_threshold=0.0)
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as image:
        frame = image.convert("RGB").crop((128, 128, 164, 172))
    frames = [frame] * 4

    plain = model.init_state(frames)
    model.add_prompt(plain, frame_idx=0, text_str="lesion")
    untracked = [outputs for _, outputs in model.propagate_in_video(plain)]
    detections = set(_rows(untracked[0]))
    scores: NDArray[np.float32] = untracked[0]["out_probs"]
    assert len(detections) > 1 and all(set(_rows(outputs)) == detections for outputs in untracked)
    assert LOW_SCORE_THRESH < scores.min() and scores.max() < 0.2

    state = model.init_state(frames, track=True)
    model.add_prompt(state, frame_idx=0, text_str="lesion")
    first = [outputs for _, outputs in model.propagate_in_video(state, max_frame_num_to_track=1)]
    ids = first[0]["out_obj_ids"].tolist()
    assert len(ids) > 1 and ids == list(range(len(ids)))
    for outputs in first:
        assert outputs["out_obj_ids"].tolist() == ids
        assert set(_rows(outputs)) <= detections
        assert _rows(outputs) == _rows(first[0])

    # At 0.2 every detection is low-score: none is returned without tracking, yet the
    # continuing run keeps all its tracks through them.
    model.add_prompt(plain, frame_idx=0, text_str="lesion", output_prob_thresh=0.2)
    assert all(
        outputs["out_obj_ids"].size == 0
        for _, outputs in model.propagate_in_video(plain, start_frame_idx=2, output_prob_thresh=0.2)
    )
    rest = list(model.propagate_in_video(state, start_frame_idx=2, output_prob_thresh=0.2))
    assert [frame_idx for frame_idx, _ in rest] == [2, 3]
    for _, outputs in rest:
        assert outputs["out_obj_ids"].tolist() == ids
        assert set(_rows(outputs)) <= detections


@pytest.mark.parametrize("track", [False, True])
def test_session_tracking_flag_reaches_the_state(track: bool) -> None:
    model, _ = _model(_crossing)
    predictor = Sam3VideoPredictor(model)
    session_id = predictor.start_session(_frames(1), track=track)["session_id"]
    response = predictor.add_prompt(session_id, frame_idx=0, text="object")
    np.testing.assert_array_equal(response["outputs"]["out_obj_ids"], [0, 1])
    stream = list(predictor.propagate_in_video(session_id))
    assert [item["outputs"]["out_obj_ids"].tolist() for item in stream] == [[0, 1]]
    predictor.close_session(session_id, run_gc_collect=False)
