"""Upstream download calls retain the port's immutable checkpoint provenance."""

from __future__ import annotations

import huggingface_hub
import pytest

from sam3_mlx.model_builder import download_ckpt_from_hf
from sam3_mlx.release_contract import (
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
)
from sam3_mlx.temporal_contract import (
    SAM31_CHECKPOINT_FILENAME,
    SAM31_CHECKPOINT_REPO,
    SAM31_CHECKPOINT_REVISION,
)


@pytest.mark.parametrize(
    ("version", "repo", "filename", "revision"),
    [
        (None, OFFICIAL_CHECKPOINT_REPO, "sam3.pt", OFFICIAL_CHECKPOINT_REVISION),
        ("sam3", OFFICIAL_CHECKPOINT_REPO, "sam3.pt", OFFICIAL_CHECKPOINT_REVISION),
        (
            "sam3.1",
            SAM31_CHECKPOINT_REPO,
            SAM31_CHECKPOINT_FILENAME,
            SAM31_CHECKPOINT_REVISION,
        ),
    ],
)
def test_upstream_download_without_revision_pins_config_and_weights(
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
    repo: str,
    filename: str,
    revision: str,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def download(*, repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return f"/cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    result = (
        download_ckpt_from_hf() if version is None else download_ckpt_from_hf(version)
    )

    assert result == f"/cache/{filename}"
    assert calls == [(repo, "config.json", revision), (repo, filename, revision)]


def test_unknown_checkpoint_version_fails_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_download(**_kwargs: object) -> str:
        pytest.fail("Unknown checkpoint versions must fail before network access.")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)

    with pytest.raises(ValueError, match="Unknown version"):
        download_ckpt_from_hf("sam3.2")
