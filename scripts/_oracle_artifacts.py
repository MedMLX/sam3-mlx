"""Canonical NPZ metadata and array boundaries for CUDA oracle artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.generic]

OFFICIAL_CUDA_ARTIFACT_TYPE = "sam3_mlx.official_cuda_oracle"
OFFICIAL_CUDA_ARTIFACT_SCHEMA_VERSION = 1
OFFICIAL_CUDA_ARTIFACT_METADATA_KEY = "__metadata_json__"


def array_ref(key: str, value: Array) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "key": key,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }


def write_official_cuda_artifact(
    path: Path,
    *,
    metadata: Mapping[str, object],
    arrays: dict[str, Array],
) -> None:
    if metadata.get("artifact_type") != OFFICIAL_CUDA_ARTIFACT_TYPE:
        raise ValueError("Official CUDA artifact metadata has the wrong artifact_type.")
    if (
        type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != OFFICIAL_CUDA_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Official CUDA artifact metadata has the wrong schema_version."
        )
    payload = {
        OFFICIAL_CUDA_ARTIFACT_METADATA_KEY: np.array(
            json.dumps(metadata, sort_keys=True),
        ),
        **arrays,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, allow_pickle=False, **payload)


def load_official_cuda_artifact(path: Path) -> tuple[dict[str, object], dict[str, Array]]:
    with np.load(path, allow_pickle=False) as data:
        if OFFICIAL_CUDA_ARTIFACT_METADATA_KEY not in data.files:
            raise ValueError(
                f"{path} is missing {OFFICIAL_CUDA_ARTIFACT_METADATA_KEY!r}."
            )
        metadata_array = data[OFFICIAL_CUDA_ARTIFACT_METADATA_KEY]
        if metadata_array.shape != ():
            raise ValueError(f"{path} metadata must be a scalar JSON string.")
        metadata_text: object = metadata_array.item()
        if not isinstance(metadata_text, str):
            raise ValueError(f"{path} metadata must be a scalar JSON string.")
        try:
            raw_metadata: object = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} metadata must contain valid JSON.") from exc
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"{path} metadata must be a JSON object.")
        metadata = cast(dict[str, object], raw_metadata)
        if metadata.get("artifact_type") != OFFICIAL_CUDA_ARTIFACT_TYPE:
            raise ValueError(f"{path} is not a {OFFICIAL_CUDA_ARTIFACT_TYPE} artifact.")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != OFFICIAL_CUDA_ARTIFACT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"{path} uses unsupported official CUDA artifact schema "
                f"{metadata.get('schema_version')!r}."
            )
        arrays = {
            name: np.array(data[name])
            for name in data.files
            if name != OFFICIAL_CUDA_ARTIFACT_METADATA_KEY
        }
    return metadata, arrays
