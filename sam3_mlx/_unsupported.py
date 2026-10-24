"""Canonical fail-fast contract for unsupported SAM3 MLX features."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from os import PathLike
from types import ModuleType
from typing import NamedTuple, NoReturn

UnsupportedReason = str

# Upstream facebookresearch/sam3 commit audited for the unsupported-feature
# inventory; equals release_contract.OFFICIAL_CODE_REVISION.
# Pass upstream_commit explicitly for entries recorded against another commit.
UPSTREAM_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"

# Closed vocabulary of failure categories; unknown reasons raise ValueError.
REASONS = frozenset(
    {
        "agent-llm",
        "eval-stack",
        "flash-attn-3",
        "image-interactivity",
        "optional-dependency",
        "port-gap",
        "torch-autograd",
        "torch-compile",
        "torch-distributed",
        "torchcodec",
        "triton-kernel",
        "unsupported-device",
        "video-multi-gpu",
        "video-multiplex",
        "video-tracker",
        "xformers",
    }
)

UNSUPPORTED_METADATA_ATTR = "__sam3_mlx_unsupported__"


class UnsupportedFeatureInfo(NamedTuple):
    """Registry record attached to an ``@unsupported_function`` stub."""

    feature: str
    reason: UnsupportedReason
    alternative: str | None
    detail: str | None
    upstream_commit: str | None


class Sam3MlxUnsupportedError(NotImplementedError):
    """Raised when an upstream SAM3 feature has no MLX implementation."""

    def __init__(
        self,
        feature: str,
        *,
        reason: UnsupportedReason,
        alternative: str | None = None,
        detail: str | None = None,
        upstream_commit: str | None = UPSTREAM_COMMIT,
    ) -> None:
        if reason not in REASONS:
            known = ", ".join(sorted(REASONS))
            raise ValueError(
                f"Unknown unsupported reason {reason!r}; use one of: {known}."
            )

        self.feature = feature
        self.reason = reason
        self.alternative = alternative
        self.detail = detail
        self.upstream_commit = upstream_commit

        message = f"{feature} is not supported in sam3_mlx ({reason})."
        if detail:
            message = f"{message} {detail}"
        if alternative:
            message = f"{message} Use {alternative} instead."
        if upstream_commit:
            message = f"{message} Upstream source commit: {upstream_commit}."
        super().__init__(message)


def unsupported(
    feature: str,
    *,
    reason: UnsupportedReason,
    alternative: str | None = None,
    detail: str | None = None,
    upstream_commit: str | None = UPSTREAM_COMMIT,
) -> Sam3MlxUnsupportedError:
    """Build (without raising) the canonical unsupported-feature error."""
    return Sam3MlxUnsupportedError(
        feature,
        reason=reason,
        alternative=alternative,
        detail=detail,
        upstream_commit=upstream_commit,
    )


def raise_unsupported(
    feature: str,
    *,
    reason: UnsupportedReason,
    alternative: str | None = None,
    detail: str | None = None,
    upstream_commit: str | None = UPSTREAM_COMMIT,
) -> NoReturn:
    """Raise ``Sam3MlxUnsupportedError`` for ``feature``."""
    raise unsupported(
        feature,
        reason=reason,
        alternative=alternative,
        detail=detail,
        upstream_commit=upstream_commit,
    )


def unsupported_function[**P, R](
    feature: str,
    *,
    reason: UnsupportedReason,
    alternative: str | None = None,
    detail: str | None = None,
    upstream_commit: str | None = UPSTREAM_COMMIT,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate an import-compatible function that always fails fast."""

    info = UnsupportedFeatureInfo(
        feature=feature,
        reason=reason,
        alternative=alternative,
        detail=detail,
        upstream_commit=upstream_commit,
    )

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> NoReturn:
            raise_unsupported(
                feature,
                reason=reason,
                alternative=alternative,
                detail=detail,
                upstream_commit=upstream_commit,
            )

        setattr(wrapper, UNSUPPORTED_METADATA_ATTR, info)
        return wrapper

    return decorator


def reject_kwargs(
    feature: str,
    *,
    reason: UnsupportedReason,
    alternative: str | None = None,
    detail: str | None = None,
    upstream_commit: str | None = UPSTREAM_COMMIT,
    **kwargs: object,
) -> None:
    """Reject truthy unsupported keyword flags with the canonical error type."""

    for name, value in kwargs.items():
        if value:
            raise_unsupported(
                f"{feature}({name}={value!r})",
                reason=reason,
                alternative=alternative,
                detail=detail,
                upstream_commit=upstream_commit,
            )


def get_unsupported_info(obj: object) -> UnsupportedFeatureInfo | None:
    """Return the registered UnsupportedFeatureInfo for an object, if any."""
    info = getattr(obj, UNSUPPORTED_METADATA_ATTR, None)
    if isinstance(info, UnsupportedFeatureInfo):
        return info
    return None


def unsupported_features(
    package: str = "sam3_mlx",
    *,
    on_import_error: Callable[[str, Exception], None] | None = None,
) -> list[UnsupportedFeatureInfo]:
    """Walk `package` and collect every @unsupported_function entry.

    Import failures propagate by default. Callers that explicitly want a
    partial registry may provide ``on_import_error`` to receive and skip an
    import failure.
    """
    import importlib
    import pkgutil

    seen_ids: set[int] = set()
    collected: list[UnsupportedFeatureInfo] = []

    def _import_module(name: str) -> ModuleType | None:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            if on_import_error is None:
                raise
            on_import_error(name, exc)
            return None

    def _scan(module: ModuleType) -> None:
        for name in dir(module):
            if name.startswith("__"):
                continue
            obj = getattr(module, name, None)
            info = get_unsupported_info(obj)
            if info is None:
                continue
            ident = id(obj)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            collected.append(info)

    root = _import_module(package)
    if root is None:
        return collected
    _scan(root)
    if hasattr(root, "__path__"):
        for module_info in pkgutil.walk_packages(
            root.__path__, prefix=root.__name__ + "."
        ):
            submodule = _import_module(module_info.name)
            if submodule is None:
                continue
            _scan(submodule)

    collected.sort(key=lambda info: (info.reason, info.feature))
    return collected


_UNSUPPORTED_DOCS_MARKER_BEGIN = "<!-- BEGIN AUTOGENERATED:UNSUPPORTED -->"
_UNSUPPORTED_DOCS_MARKER_END = "<!-- END AUTOGENERATED:UNSUPPORTED -->"


def render_unsupported_markdown(
    features: list[UnsupportedFeatureInfo] | None = None,
) -> str:
    """Render the registered unsupported features as a Markdown section.

    Output is deterministic so the table can be checked into source and asserted
    in CI. Pass ``features`` to override the registry walk; ``None`` uses
    ``unsupported_features()``.
    """
    if features is None:
        features = unsupported_features()

    lines: list[str] = [
        "## Unsupported upstream features",
        "",
        (
            "Every entry below corresponds to an `@unsupported_function` stub. "
            "Invoking one raises `sam3_mlx.Sam3MlxUnsupportedError` "
            "(a subclass of `NotImplementedError`)."
        ),
        "",
        f"Total registered features: **{len(features)}**.",
        "",
        "| Reason | Feature | Alternative | Upstream commit |",
        "| --- | --- | --- | --- |",
    ]
    for info in features:
        feature = info.feature.replace("|", r"\|")
        alternative = (info.alternative or "").replace("|", r"\|")
        commit = (info.upstream_commit or "")[:7]
        lines.append(f"| `{info.reason}` | `{feature}` | {alternative} | {commit} |")
    return "\n".join(lines) + "\n"


def write_unsupported_section(path: str | PathLike[str]) -> bool:
    """Rewrite the auto-generated section of the Markdown file at ``path``.

    The file must contain the BEGIN/END HTML-comment markers. Returns True if
    the file content changed.
    """
    import pathlib

    p = pathlib.Path(path)
    original = p.read_text()
    if (
        _UNSUPPORTED_DOCS_MARKER_BEGIN not in original
        or _UNSUPPORTED_DOCS_MARKER_END not in original
    ):
        raise ValueError(
            f"{p} is missing the autogenerated unsupported-features markers."
        )
    before, _, rest = original.partition(_UNSUPPORTED_DOCS_MARKER_BEGIN)
    _, _, after = rest.partition(_UNSUPPORTED_DOCS_MARKER_END)
    block = (
        f"{_UNSUPPORTED_DOCS_MARKER_BEGIN}\n"
        f"{render_unsupported_markdown()}"
        f"{_UNSUPPORTED_DOCS_MARKER_END}"
    )
    updated = f"{before}{block}{after}"
    if updated == original:
        return False
    p.write_text(updated)
    return True


def _cli() -> int:
    """``python -m sam3_mlx._unsupported [--write PATH] [--check PATH]``."""
    import argparse
    import pathlib
    import sys

    parser = argparse.ArgumentParser(prog="sam3_mlx._unsupported")
    parser.add_argument(
        "--write", help="Path to update in place (e.g. docs/technical.md)."
    )
    parser.add_argument(
        "--check",
        help="Path to verify; exit 1 if the autogenerated section is stale.",
    )
    args = parser.parse_args()

    if args.write:
        changed = write_unsupported_section(args.write)
        sys.stdout.write(f"{'updated' if changed else 'unchanged'}: {args.write}\n")
        return 0

    if args.check:
        p = pathlib.Path(args.check)
        original = p.read_text()
        if _UNSUPPORTED_DOCS_MARKER_BEGIN not in original:
            sys.stderr.write(f"{p}: missing autogenerated markers.\n")
            return 1
        before, _, rest = original.partition(_UNSUPPORTED_DOCS_MARKER_BEGIN)
        _, _, after = rest.partition(_UNSUPPORTED_DOCS_MARKER_END)
        block = (
            f"{_UNSUPPORTED_DOCS_MARKER_BEGIN}\n"
            f"{render_unsupported_markdown()}"
            f"{_UNSUPPORTED_DOCS_MARKER_END}"
        )
        expected = f"{before}{block}{after}"
        if expected != original:
            sys.stderr.write(
                f"{p}: autogenerated section is stale. "
                f"Run `python -m sam3_mlx._unsupported --write {p}`.\n"
            )
            return 1
        sys.stdout.write(f"{p}: up to date.\n")
        return 0

    sys.stdout.write(render_unsupported_markdown())
    return 0


def main() -> int:
    """Run the unsupported-feature documentation CLI."""

    return _cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    # Re-import the package so decorators and the registry share
    # UnsupportedFeatureInfo when this module runs with python -m.
    import sam3_mlx._unsupported as _pkg

    raise SystemExit(_pkg.main())
