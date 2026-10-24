from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

from numpy.typing import NDArray

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from scripts._oracle_artifacts import (
    OFFICIAL_CUDA_ARTIFACT_METADATA_KEY,
    OFFICIAL_CUDA_ARTIFACT_SCHEMA_VERSION,
    OFFICIAL_CUDA_ARTIFACT_TYPE,
    array_ref,
    load_official_cuda_artifact,
    write_official_cuda_artifact,
)

JsonScalar: TypeAlias = None | bool | int | float | str


@settings(max_examples=50, deadline=None, derandomize=True)
@given(st.lists(st.integers(-100, 100), max_size=30))
def test_official_artifact_numeric_arrays_round_trip(values: list[int]) -> None:
    metadata = {
        "artifact_type": OFFICIAL_CUDA_ARTIFACT_TYPE,
        "schema_version": OFFICIAL_CUDA_ARTIFACT_SCHEMA_VERSION,
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "oracle.npz"
        for dtype in (np.float32, np.int64, np.bool_):
            array = np.asarray(values, dtype=dtype)
            write_official_cuda_artifact(
                path, metadata=metadata, arrays={"values": array}
            )
            restored_metadata, restored_arrays = load_official_cuda_artifact(path)
            assert restored_metadata == metadata
            np.testing.assert_array_equal(restored_arrays["values"], array)
            assert restored_arrays["values"].dtype == array.dtype
            assert array_ref("values", restored_arrays["values"]) == array_ref(
                "values", array
            )


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text(),
        st.lists(st.integers(), max_size=10),
    )
)
def test_official_artifact_rejects_non_object_metadata(
    metadata: JsonScalar | list[int],
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.npz"
        np.savez_compressed(
            path,
            allow_pickle=False,
            **{OFFICIAL_CUDA_ARTIFACT_METADATA_KEY: np.array(json.dumps(metadata))},
        )
        with pytest.raises(ValueError, match="metadata must be a JSON object"):
            load_official_cuda_artifact(path)


@pytest.mark.parametrize("metadata", (np.array(1), np.array(["{}"]), np.array("{")))
def test_official_artifact_rejects_invalid_metadata_encoding(
    tmp_path: Path,
    metadata: NDArray[np.generic],
) -> None:
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path, allow_pickle=False, **{OFFICIAL_CUDA_ARTIFACT_METADATA_KEY: metadata}
    )
    with pytest.raises(ValueError, match="metadata must"):
        load_official_cuda_artifact(path)


@pytest.mark.parametrize("version", (None, True, False, 0, "1"))
def test_official_artifact_rejects_invalid_schema_version(
    tmp_path: Path, version: JsonScalar
) -> None:
    path = tmp_path / "invalid.npz"
    metadata = {"artifact_type": OFFICIAL_CUDA_ARTIFACT_TYPE, "schema_version": version}
    np.savez_compressed(
        path,
        allow_pickle=False,
        **{OFFICIAL_CUDA_ARTIFACT_METADATA_KEY: np.array(json.dumps(metadata))},
    )
    with pytest.raises(ValueError, match="unsupported official CUDA artifact schema"):
        load_official_cuda_artifact(path)
    with pytest.raises(ValueError, match="wrong schema_version"):
        write_official_cuda_artifact(path, metadata=metadata, arrays={})
