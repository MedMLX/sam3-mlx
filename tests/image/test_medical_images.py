"""Open-license medical images: provenance, and preprocessing parity with TorchVision."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_image_processor import transform
from tests._json_contracts import require_list, require_mapping
from tests._paths import MEDICAL_FIXTURE_ROOT
from tests._resize_reference import official_resize

OPEN_LICENSES = frozenset({"CC0-1.0", "public-domain"})


def _entries() -> list[dict[str, str]]:
    """Read sources.json, keeping the string fields every image must record."""
    record = require_mapping(
        json.loads((MEDICAL_FIXTURE_ROOT / "sources.json").read_text(encoding="utf-8")),
        "sources.json",
    )
    entries: list[dict[str, str]] = []
    for item in require_list(record["images"], "sources.json images"):
        fields = require_mapping(item, "sources.json image")
        entries.append(
            {
                key: str(fields[key])
                for key in ("file", "sha256", "license", "license_url", "attribution", "source_url")
            }
        )
    return entries


ENTRIES = _entries()


def test_every_medical_image_is_recorded_unchanged_with_an_open_license() -> None:
    """Each image matches its recorded checksum and carries an open license."""
    on_disk = {path.name for path in MEDICAL_FIXTURE_ROOT.iterdir()} - {"sources.json"}
    assert on_disk == {entry["file"] for entry in ENTRIES}
    for entry in ENTRIES:
        data = (MEDICAL_FIXTURE_ROOT / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["file"]
        assert entry["license"] in OPEN_LICENSES, entry["file"]
        assert entry["license_url"].startswith("https://"), entry["file"]
        assert entry["source_url"].startswith("https://"), entry["file"]
        assert entry["attribution"], entry["file"]


@pytest.mark.parametrize("name", [entry["file"] for entry in ENTRIES])
def test_default_preprocessing_matches_torchvision_on_medical_images(name: str) -> None:
    """The default resize equals the FP32 TorchVision reference at every pixel.

    The images cover RGB and 8-bit grayscale inputs in JPEG and PNG.
    """
    with Image.open(MEDICAL_FIXTURE_ROOT / name) as image:
        image.load()
    normalized = to_numpy(transform(image, resolution=1008), dtype=np.float32)
    local = np.rint((normalized.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    reference = official_resize(np.asarray(image.convert("RGB"), dtype=np.uint8), (1008, 1008))
    np.testing.assert_array_equal(local, reference)
