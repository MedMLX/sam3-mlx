from pathlib import Path

import numpy as np
import pytest

from sam3_mlx.parity_evidence import CaseComparison
from scripts.generate_image_parity_corpus import (
    NONSENSE_PROMPT,
    CocoAnnotations,
    aspect_bucket,
    build_cases,
    fetch_image,
    load_coco_annotations,
    select_images,
    sha256_file,
)
from scripts.run_image_corpus_parity import summarize


def _annotations(image_count: int = 30) -> CocoAnnotations:
    sizes = [(640, 400), (400, 640), (500, 480)]
    images: list[object] = []
    annotations: list[object] = []
    for image_id in range(1, image_count + 1):
        width, height = sizes[image_id % 3]
        images.append(
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": width,
             "height": height, "license": 1}
        )
        annotations.extend(
            [
                {"image_id": image_id, "category_id": 1, "iscrowd": 0,
                 "area": 0.2 * width * height, "bbox": [10, 20, 100, 200]},
                {"image_id": image_id, "category_id": 2, "iscrowd": 0,
                 "area": 0.05 * width * height, "bbox": [5, 5, 30, 40]},
                {"image_id": image_id, "category_id": 3, "iscrowd": 1,
                 "area": 0.9 * width * height, "bbox": [0, 0, width, height]},
            ]
        )
    return load_coco_annotations(
        {
            "categories": [
                {"id": 1, "name": "dining table"},
                {"id": 2, "name": "cup"},
                {"id": 3, "name": "person"},
            ],
            "licenses": [{"id": 1, "name": "Attribution License", "url": "http://x"}],
            "images": images,
            "annotations": annotations,
        }
    )


def test_selection_is_seeded_and_balanced_across_aspect_buckets() -> None:
    annotations = _annotations()

    first = select_images(annotations, count=6, seed=7)
    second = select_images(annotations, count=6, seed=7)
    other = select_images(annotations, count=6, seed=8)

    assert [image.id for image in first] == [image.id for image in second]
    assert [image.id for image in first] != [image.id for image in other]
    assert sorted(aspect_bucket(image) for image in first) == [
        "landscape", "landscape", "near_square", "near_square", "portrait", "portrait",
    ]
    with pytest.raises(ValueError, match="multiple of 3"):
        select_images(annotations, count=4, seed=7)


def test_cases_come_from_the_image_labels_and_skip_crowd_regions() -> None:
    annotations = _annotations()
    image = annotations.images[3]
    rng = np.random.default_rng(0)

    cases = build_cases(image, annotations.instances[3], position=0, rng=rng)

    assert [case["name"] for case in cases] == [
        "text_dining_table_1008",
        "text_cup_1008",
        "positive_box_1008",
        "text_dining_table_672",
        "text_nonsense_1008",
    ]
    assert cases[-1]["prompt"] == NONSENSE_PROMPT
    box = cases[2]["geometric_prompts"][0]
    assert box["label"] is True
    assert box["box"] == pytest.approx(
        [60 / image.width, 120 / image.height, 100 / image.width, 200 / image.height]
    )
    later = build_cases(image, annotations.instances[3], position=1, rng=rng)
    assert [case["name"] for case in later][-1] == "text_dining_table_504"


def test_fetch_image_rejects_a_file_that_does_not_match_its_pin(tmp_path: Path) -> None:
    image = tmp_path / "000000000001.jpg"
    image.write_bytes(b"pixels")

    assert fetch_image("https://unused", image, expected_sha256=sha256_file(image))
    with pytest.raises(ValueError, match="manifest pins"):
        fetch_image("https://unused", image, expected_sha256="0" * 64)


def _case(status: str, *, iou: float | None, score: float | None) -> CaseComparison:
    return {
        "name": "case", "resolution": 1008, "prompt": "cup", "geometric_prompts": [],
        "status": "passed" if status == "passed" else "failed",
        "official_detection_count": 1, "mlx_detection_count": 1,
        "detection_count_match": True, "mask_iou_min": iou, "mask_iou_mean": iou,
        "box_l_inf_max": 0.1, "score_abs_max": score, "matches": [],
    }


def test_summary_reports_worst_metrics_and_ignores_empty_cases() -> None:
    summary = summarize(
        [
            _case("passed", iou=0.999, score=0.001),
            _case("failed", iou=0.96, score=0.03),
            _case("passed", iou=None, score=None),
        ]
    )

    assert summary["cases"] == 3
    assert summary["failed"] == 1
    assert summary["worst_mask_iou_min"] == 0.96
    assert summary["worst_score_abs_max"] == 0.03
