"""Upstream features that are not ported fail fast with the shared error."""

import pytest

import sam3_mlx
from sam3_mlx._unsupported import UPSTREAM_COMMIT, Sam3MlxUnsupportedError, raise_unsupported


def test_canonical_unsupported_error_preserves_notimplemented_contract() -> None:
    assert sam3_mlx.Sam3MlxUnsupportedError is Sam3MlxUnsupportedError

    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        raise_unsupported(
            "sam3_mlx.example",
            reason="optional-dependency",
            alternative="sam3_mlx.native",
            detail="This upstream path is not ported.",
        )

    error = exc_info.value
    assert isinstance(error, NotImplementedError)
    assert error.feature == "sam3_mlx.example"
    assert error.reason == "optional-dependency"
    assert error.alternative == "sam3_mlx.native"
    assert error.detail == "This upstream path is not ported."
    assert error.upstream_commit == UPSTREAM_COMMIT
    assert str(error).endswith(f"Upstream source commit: {UPSTREAM_COMMIT}.")
    with pytest.raises(ValueError):
        raise_unsupported("sam3_mlx.bad", reason="torch-ish")
