#!/usr/bin/env python3
"""Select a seeded COCO val2017 image corpus and write its parity manifest.

Selection is a pure function of the annotation file, the seed, and the image
count, so the corpus is fixed before any MLX or official output is observed.
Images are stratified by aspect ratio (landscape, portrait, near-square). Each
image receives prompts derived from its own COCO labels:

* a text prompt for the category with the largest labeled area, at 1008;
* a text prompt for one other present category, when one exists, at 1008;
* a positive box prompt from the largest labeled instance, at 1008;
* the first text prompt again at 672 or 504 (alternating by image);
* a nonsense text prompt at 1008 on every fourth image.

The manifest pins each image by URL and SHA-256; images are downloaded on
demand and are not stored in the repository.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    JsonObject,
    require_json_finite_number,
    require_json_list,
    require_json_nonnegative_int,
    require_json_object,
    require_json_string,
)
from scripts._oracle_runtime import OracleCase  # noqa: E402

MANIFEST_SCHEMA_VERSION = 1
# images.cocodataset.org serves an S3 certificate for another host; the
# path-style bucket URL serves the same objects over valid HTTPS.
COCO_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}"
NONSENSE_PROMPT = "zzzz_not_a_real_class_qqq"
MIN_PROMPT_AREA_FRACTION = 0.01
LANDSCAPE_MIN_ASPECT = 1.2
PORTRAIT_MAX_ASPECT = 0.85


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    license_id: int


@dataclass(frozen=True)
class CocoInstance:
    category: str
    area: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class CocoAnnotations:
    images: dict[int, CocoImage]
    instances: dict[int, list[CocoInstance]]
    licenses: dict[int, JsonObject]


def load_coco_annotations(raw: object) -> CocoAnnotations:
    """Parse the fields of a COCO instances file that corpus selection uses."""

    document = require_json_object(raw, field="COCO annotations")
    categories: dict[int, str] = {}
    for item in require_json_list(document.get("categories"), field="categories"):
        category = require_json_object(item, field="category")
        categories[require_json_nonnegative_int(category.get("id"), field="category id")] = (
            require_json_string(category.get("name"), field="category name")
        )
    licenses: dict[int, JsonObject] = {}
    for item in require_json_list(document.get("licenses"), field="licenses"):
        license_entry = require_json_object(item, field="license")
        licenses[require_json_nonnegative_int(license_entry.get("id"), field="license id")] = {
            "name": require_json_string(license_entry.get("name"), field="license name"),
            "url": require_json_string(license_entry.get("url"), field="license url"),
        }
    images: dict[int, CocoImage] = {}
    for item in require_json_list(document.get("images"), field="images"):
        image = require_json_object(item, field="image")
        image_id = require_json_nonnegative_int(image.get("id"), field="image id")
        images[image_id] = CocoImage(
            id=image_id,
            file_name=require_json_string(image.get("file_name"), field="file_name"),
            width=require_json_nonnegative_int(image.get("width"), field="width"),
            height=require_json_nonnegative_int(image.get("height"), field="height"),
            license_id=require_json_nonnegative_int(image.get("license"), field="license"),
        )
    instances: dict[int, list[CocoInstance]] = defaultdict(list)
    for item in require_json_list(document.get("annotations"), field="annotations"):
        annotation = require_json_object(item, field="annotation")
        if annotation.get("iscrowd") == 1:
            continue
        bbox_values = require_json_list(annotation.get("bbox"), field="bbox")
        if len(bbox_values) != 4:
            raise ValueError("COCO bbox must have four values.")
        x, y, w, h = (require_json_finite_number(v, field="bbox value") for v in bbox_values)
        image_id = require_json_nonnegative_int(annotation.get("image_id"), field="image_id")
        category_id = require_json_nonnegative_int(
            annotation.get("category_id"), field="category_id"
        )
        instances[image_id].append(
            CocoInstance(
                category=categories[category_id],
                area=require_json_finite_number(annotation.get("area"), field="area"),
                bbox=(x, y, w, h),
            )
        )
    return CocoAnnotations(images=images, instances=dict(instances), licenses=licenses)


def aspect_bucket(image: CocoImage) -> str:
    aspect = image.width / image.height
    if aspect >= LANDSCAPE_MIN_ASPECT:
        return "landscape"
    if aspect <= PORTRAIT_MAX_ASPECT:
        return "portrait"
    return "near_square"


def _eligible(image: CocoImage, instances: list[CocoInstance]) -> bool:
    min_area = MIN_PROMPT_AREA_FRACTION * image.width * image.height
    return image.width > 0 and image.height > 0 and any(
        instance.area >= min_area for instance in instances
    )


def select_images(annotations: CocoAnnotations, *, count: int, seed: int) -> list[CocoImage]:
    """Pick ``count`` images, an equal share from each aspect bucket, by seed."""

    buckets: dict[str, list[CocoImage]] = defaultdict(list)
    for image_id in sorted(annotations.images):
        image = annotations.images[image_id]
        if _eligible(image, annotations.instances.get(image_id, [])):
            buckets[aspect_bucket(image)].append(image)
    names = ("landscape", "portrait", "near_square")
    if count % len(names):
        raise ValueError(f"count must be a multiple of {len(names)}.")
    per_bucket = count // len(names)
    rng = np.random.default_rng(seed)
    selected: list[CocoImage] = []
    for name in names:
        candidates = buckets[name]
        if len(candidates) < per_bucket:
            raise ValueError(f"Only {len(candidates)} eligible {name} images.")
        picks = rng.choice(len(candidates), size=per_bucket, replace=False)
        selected.extend(candidates[int(index)] for index in sorted(picks))
    return selected


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_cases(
    image: CocoImage,
    instances: list[CocoInstance],
    *,
    position: int,
    rng: np.random.Generator,
) -> list[OracleCase]:
    """Derive one image's prompt matrix from its own COCO labels."""

    area_by_category: dict[str, float] = defaultdict(float)
    for instance in instances:
        area_by_category[instance.category] += instance.area
    primary = max(sorted(area_by_category), key=lambda name: area_by_category[name])
    others = sorted(name for name in area_by_category if name != primary)
    cases: list[OracleCase] = [
        {
            "name": f"text_{_slug(primary)}_1008",
            "resolution": 1008,
            "prompt": primary,
            "geometric_prompts": [],
        }
    ]
    if others:
        secondary = others[int(rng.integers(len(others)))]
        cases.append(
            {
                "name": f"text_{_slug(secondary)}_1008",
                "resolution": 1008,
                "prompt": secondary,
                "geometric_prompts": [],
            }
        )
    largest = max(instances, key=lambda instance: instance.area)
    x, y, w, h = largest.bbox
    box = [
        (x + w / 2) / image.width,
        (y + h / 2) / image.height,
        w / image.width,
        h / image.height,
    ]
    cases.append(
        {
            "name": "positive_box_1008",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [{"box": box, "label": True}],
        }
    )
    lower = 672 if position % 2 == 0 else 504
    cases.append(
        {
            "name": f"text_{_slug(primary)}_{lower}",
            "resolution": lower,
            "prompt": primary,
            "geometric_prompts": [],
        }
    )
    if position % 4 == 0:
        cases.append(
            {
                "name": "text_nonsense_1008",
                "resolution": 1008,
                "prompt": NONSENSE_PROMPT,
                "geometric_prompts": [],
            }
        )
    return cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_image(url: str, destination: Path, *, expected_sha256: str | None) -> str:
    """Download ``url`` unless already present, verify it, and return its SHA-256."""

    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response:
            partial.write_bytes(response.read())
        partial.replace(destination)
    digest = sha256_file(destination)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{destination} has SHA-256 {digest}; the manifest pins {expected_sha256}."
        )
    return digest


def build_manifest(
    annotations: CocoAnnotations,
    *,
    annotations_sha256: str,
    count: int,
    seed: int,
    image_dir: Path,
) -> JsonObject:
    rng = np.random.default_rng(seed + 1)
    entries: list[object] = []
    for position, image in enumerate(select_images(annotations, count=count, seed=seed)):
        url = COCO_IMAGE_URL.format(file_name=image.file_name)
        digest = fetch_image(url, image_dir / image.file_name, expected_sha256=None)
        entries.append(
            {
                "id": image.id,
                "file_name": image.file_name,
                "url": url,
                "sha256": digest,
                "size": [image.width, image.height],
                "aspect_bucket": aspect_bucket(image),
                "license": annotations.licenses[image.license_id],
                "cases": build_cases(
                    image,
                    annotations.instances[image.id],
                    position=position,
                    rng=rng,
                ),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "coco-val2017-v1",
        "source": {
            "dataset": "COCO 2017 validation",
            "annotations": "annotations_trainval2017.zip:annotations/instances_val2017.json",
            "annotations_sha256": annotations_sha256,
        },
        "selection": {
            "seed": seed,
            "count": count,
            "aspect_buckets": {
                "landscape": f"width/height >= {LANDSCAPE_MIN_ASPECT}",
                "portrait": f"width/height <= {PORTRAIT_MAX_ASPECT}",
                "near_square": "otherwise",
            },
            "eligibility": (
                "at least one non-crowd instance covering "
                f">= {MIN_PROMPT_AREA_FRACTION:.0%} of the image"
            ),
        },
        "images": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_coco_annotations(
        json.loads(args.annotations.read_text(encoding="utf-8"))
    )
    manifest = build_manifest(
        annotations,
        annotations_sha256=sha256_file(args.annotations),
        count=args.count,
        seed=args.seed,
        image_dir=args.image_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    case_count = sum(
        len(require_json_list(require_json_object(entry, field="image").get("cases"), field="cases"))
        for entry in require_json_list(manifest["images"], field="images")
    )
    print(json.dumps({"wrote": str(args.out), "images": args.count, "cases": case_count}))


if __name__ == "__main__":
    main()
