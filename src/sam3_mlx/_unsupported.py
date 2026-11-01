"""Canonical fail-fast contract for unsupported SAM3 MLX features."""

from __future__ import annotations

from typing import NoReturn

# Upstream facebookresearch/sam3 commit the port was checked against.
UPSTREAM_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"

# Closed vocabulary of failure categories; unknown reasons raise ValueError.
REASONS = frozenset({"optional-dependency"})


class Sam3MlxUnsupportedError(NotImplementedError):
    """Raised when an upstream SAM3 feature has no MLX implementation."""

    def __init__(
        self,
        feature: str,
        *,
        reason: str,
        alternative: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record a recognized unsupported-feature reason and optional guidance."""
        if reason not in REASONS:
            known = ", ".join(sorted(REASONS))
            raise ValueError(
                f"Unknown unsupported reason {reason!r}; use one of: {known}."
            )

        self.feature = feature
        self.reason = reason
        self.alternative = alternative
        self.detail = detail
        self.upstream_commit = UPSTREAM_COMMIT

        message = f"{feature} is not supported in sam3_mlx ({reason})."
        if detail:
            message = f"{message} {detail}"
        if alternative:
            message = f"{message} Use {alternative} instead."
        super().__init__(f"{message} Upstream source commit: {UPSTREAM_COMMIT}.")


def raise_unsupported(
    feature: str,
    *,
    reason: str,
    alternative: str | None = None,
    detail: str | None = None,
) -> NoReturn:
    """Raise ``Sam3MlxUnsupportedError`` for ``feature``."""
    raise Sam3MlxUnsupportedError(
        feature, reason=reason, alternative=alternative, detail=detail
    )
