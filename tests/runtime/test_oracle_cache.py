from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from sam3_mlx.release_contract import (
    JsonObject,
    OracleBindings,
    ORACLE_OUTPUT_DTYPES,
    build_oracle_bindings,
)
from scripts._oracle_runtime import (
    OracleCase,
    build_oracle_cache_identity,
    load_official_oracle_archive,
    save_oracle_arrays,
)
from scripts import _oracle_runtime


SPEC: OracleCase = {
    "name": "synthetic",
    "resolution": 14,
    "prompt": "object",
    "geometric_prompts": [],
}
CACHE_IDENTITY: JsonObject = {
    "image": {"path": "/synthetic/image.jpg", "size": 1, "mtime_ns": 1}
}


def _bindings() -> OracleBindings:
    return build_oracle_bindings(
        image="image.jpg",
        case_spec="cases.json",
        confidence_threshold=0.5,
    )


def _metadata(bindings: OracleBindings, count: int = 1) -> JsonObject:
    return {
        "schema_version": bindings["schema_version"],
        "bindings": bindings,
        "cache_identity": CACHE_IDENTITY,
        "cases": [{**SPEC, "detection_count": count, "tensor_dtypes": ORACLE_OUTPUT_DTYPES}],
    }


def _write_cache(path: Path, metadata: object, masks: np.ndarray | None = None) -> None:
    if masks is None:
        masks = np.ones((1, 2, 2), dtype=np.bool_)
    count = len(masks)
    save_oracle_arrays(
        path,
        {
            "metadata_json": np.array(json.dumps(metadata)),
            "case_0_masks": masks,
            "case_0_boxes": np.zeros((count, 4), dtype=np.float32),
            "case_0_scores": np.full(count, 0.75, dtype=np.float32),
        },
    )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    st.lists(
        st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans()),
        max_size=6,
    ),
    st.booleans(),
)
def test_oracle_cache_round_trips_binary_masks(
    values: list[tuple[bool, bool, bool, bool]], channel_dimension: bool
) -> None:
    masks = np.array(values, dtype=np.uint8).reshape(len(values), 2, 2)
    bindings = _bindings()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "oracle.npz"
        _write_cache(
            path,
            _metadata(bindings, len(values)),
            masks[:, None] if channel_dimension else masks,
        )
        metadata, outputs = load_official_oracle_archive(
            path, expected_bindings=bindings, expected_cache_identity=CACHE_IDENTITY, expected_specs=[SPEC], image_size=(2, 2)
        )
    assert metadata["bindings"] == bindings
    np.testing.assert_array_equal(outputs[0]["masks"], masks.astype(np.bool_))
    assert outputs[0]["masks"].dtype == np.bool_


@pytest.mark.parametrize("field", tuple(OracleBindings.__annotations__))
def test_oracle_cache_rejects_each_stale_binding(
    tmp_path: Path, field: str
) -> None:
    expected = _bindings()
    metadata = _metadata(deepcopy(expected))
    tampered: JsonObject = dict(expected)
    tampered[field] = None
    metadata["bindings"] = tampered
    path = tmp_path / "stale.npz"
    _write_cache(path, metadata)
    with pytest.raises(ValueError, match="oracle cache bindings"):
        load_official_oracle_archive(
            path, expected_bindings=expected, expected_cache_identity=CACHE_IDENTITY, expected_specs=[SPEC], image_size=(2, 2)
        )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text(),
        st.lists(st.integers(), max_size=5),
    )
)
def test_oracle_cache_rejects_non_object_metadata_without_crashing(
    metadata: object,
) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "malformed.npz"
        _write_cache(path, metadata)
        with pytest.raises(ValueError, match="Oracle metadata must be a JSON object"):
            load_official_oracle_archive(
                path,
                expected_bindings=_bindings(),
                expected_cache_identity=CACHE_IDENTITY,
                expected_specs=[SPEC],
                image_size=(2, 2),
            )


@pytest.mark.parametrize("schema", [2, True, None])
def test_oracle_cache_rejects_old_or_invalid_schema(
    tmp_path: Path, schema: object
) -> None:
    bindings = _bindings()
    metadata = _metadata(bindings)
    metadata["schema_version"] = schema
    path = tmp_path / "old.npz"
    _write_cache(path, metadata)
    with pytest.raises(ValueError, match="(schema|recompute)"):
        load_official_oracle_archive(
            path, expected_bindings=bindings, expected_cache_identity=CACHE_IDENTITY, expected_specs=[SPEC], image_size=(2, 2)
        )


def test_oracle_cache_rejects_npy_arrays(tmp_path: Path) -> None:
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="NPZ archive"):
        load_official_oracle_archive(
            path,
            expected_bindings=_bindings(),
                expected_cache_identity=CACHE_IDENTITY,
            expected_specs=[SPEC],
            image_size=(2, 2),
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 0.25, 2.0])
def test_oracle_cache_rejects_malformed_masks(tmp_path: Path, invalid: float) -> None:
    bindings = _bindings()
    path = tmp_path / "malformed-masks.npz"
    _write_cache(path, _metadata(bindings), np.full((1, 2, 2), invalid))
    with pytest.raises(ValueError, match="masks.*(non-finite|binary)"):
        load_official_oracle_archive(
            path, expected_bindings=bindings, expected_cache_identity=CACHE_IDENTITY, expected_specs=[SPEC], image_size=(2, 2)
        )


@pytest.mark.parametrize(
    "changed",
    [
        "image_content", "image_location", "mtime", "checkpoint",
        "python", "runtime", "resize_runtime", "prompt", "geometric_prompts",
    ],
)
def test_oracle_cache_rejects_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    image = tmp_path / "image.jpg"
    checkpoint = tmp_path / "weights.pt"
    image.write_bytes(b"first image")
    checkpoint.write_bytes(b"first checkpoint")
    python = tmp_path / "python"
    python.write_bytes(b"synthetic interpreter")
    runtime = tmp_path / "runtime"
    for name in (
        "scripts/run_upstream_image_oracle_hardened.py",
        "scripts/_oracle_runtime.py",
        "scripts/_resize_reference.py",
        "sam3_mlx/release_contract.py",
    ):
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic runtime")
    monkeypatch.setattr(_oracle_runtime, "REPO_ROOT", runtime)
    expected = build_oracle_cache_identity(
        image=image, checkpoint=checkpoint, official_python=python
    )
    metadata = _metadata(_bindings())
    metadata["cache_identity"] = expected
    cache = tmp_path / "oracle.npz"
    _write_cache(cache, metadata)
    load_official_oracle_archive(
        cache,
        expected_bindings=_bindings(),
        expected_cache_identity=expected,
        expected_specs=[SPEC],
        image_size=(2, 2),
    )
    requested = deepcopy(SPEC)
    if changed == "image_content":
        image.write_bytes(b"changed image with another size")
    elif changed == "image_location":
        image = tmp_path / "other" / image.name
        image.parent.mkdir()
        image.write_bytes(b"first image")
    elif changed == "checkpoint":
        checkpoint.write_bytes(b"changed checkpoint")
    elif changed == "mtime":
        original = image.stat()
        os.utime(image, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    elif changed == "python":
        python.write_bytes(b"changed interpreter")
    elif changed == "runtime":
        (runtime / "scripts/_oracle_runtime.py").write_text("changed runtime")
    elif changed == "resize_runtime":
        (runtime / "scripts/_resize_reference.py").write_text("changed resize contract")
    elif changed == "prompt":
        requested["prompt"] = "another object"
    else:
        requested["geometric_prompts"] = [{"box": [0, 0, 1, 1], "label": False}]
    current = build_oracle_cache_identity(
        image=image, checkpoint=checkpoint, official_python=python
    )
    with pytest.raises(ValueError, match="(input identity|metadata drifted)"):
        load_official_oracle_archive(
            cache,
            expected_bindings=_bindings(),
            expected_cache_identity=current,
            expected_specs=[requested],
            image_size=(2, 2),
        )
