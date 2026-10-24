"""Source-based inference API inventory; signature agreement is not model parity."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


API_TARGET_REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
API_REPORT_SCHEMA_VERSION = 1
BACKEND_DEFAULT_ADAPTATIONS = frozenset(
    {
        ("device", "'cuda'", "'mlx'"),
        ("device", "'cpu'", "'mlx'"),
        ("device", "'cuda' if torch.cuda.is_available() else 'cpu'", "'mlx'"),
        ("use_fa3", "True", "False"),
        ("use_rope_real", "True", "False"),
    }
)


class ApiContractError(ValueError):
    """An API source cannot be interpreted under this contract."""


class ApiGroup(StrEnum):
    BUILDERS = "builders"
    IMAGE = "image"
    INTERACTIVE = "interactive-image"
    VIDEO = "video-requests"
    MULTIPLEX = "multiplex"


class ParameterKind(StrEnum):
    POSITIONAL_ONLY = "positional-only"
    POSITIONAL = "positional-or-keyword"
    KEYWORD_ONLY = "keyword-only"
    VAR_POSITIONAL = "var-positional"
    VAR_KEYWORD = "var-keyword"


class ApiVerdict(StrEnum):
    """Declaration-level result: exact match, only backend-default adaptations, other gaps, or absent."""

    MATCH = "signature-match"
    BACKEND_DIFFERENCE = "backend-difference"
    GAP = "api-gap"
    MISSING = "missing-surface"


@dataclass(frozen=True, slots=True)
class ApiDeclaration:
    """A function or method location: ``symbol`` (dotted) inside a package-relative ``source_file``."""

    source_file: Path
    symbol: str

    def __post_init__(self) -> None:
        if self.source_file.is_absolute() or ".." in self.source_file.parts:
            raise ApiContractError("API source paths must be package-relative.")
        if not all(part.isidentifier() for part in self.symbol.split(".")):
            raise ApiContractError(f"Invalid API symbol: {self.symbol!r}.")


@dataclass(frozen=True, slots=True)
class ApiSurface:
    """One inventoried upstream API; ``local_declaration`` overrides where the port declares it."""

    group: ApiGroup
    source_file: Path
    symbol: str
    local_declaration: ApiDeclaration | None = None

    def __post_init__(self) -> None:
        ApiDeclaration(self.source_file, self.symbol)

    @property
    def official(self) -> ApiDeclaration:
        return ApiDeclaration(self.source_file, self.symbol)

    @property
    def local(self) -> ApiDeclaration:
        return self.local_declaration or self.official


@dataclass(frozen=True, slots=True)
class ApiParameter:
    """One parameter; ``default`` is the default's source expression (``None`` if it has none)."""

    name: str
    kind: ParameterKind
    default: str | None


@dataclass(frozen=True, slots=True)
class ApiSignature:
    parameters: tuple[ApiParameter, ...]


@dataclass(frozen=True, slots=True)
class ApiDifference:
    """A mismatch on ``parameter``; ``backend_specific`` marks a listed device/kernel default change."""

    parameter: str
    detail: str
    backend_specific: bool = False


@dataclass(frozen=True, slots=True)
class ApiDecision:
    """The verdict and supporting signatures for one surface (``local`` is ``None`` if missing)."""

    surface: ApiSurface
    verdict: ApiVerdict
    official: ApiSignature
    local: ApiSignature | None
    differences: tuple[ApiDifference, ...]


def _surfaces(
    group: ApiGroup, file: str, owner: str, methods: tuple[str, ...]
) -> tuple[ApiSurface, ...]:
    return tuple(
        ApiSurface(group, Path(file), f"{owner}.{name}" if owner else name)
        for name in methods
    )


# This inventory owns the inference API target; numerical release pins stay separate.
INFERENCE_API_SURFACES = (
    *_surfaces(
        ApiGroup.BUILDERS,
        "model_builder.py",
        "",
        (
            "build_tracker",
            "build_sam3_image_model",
            "build_sam3_video_model",
            "build_sam3_video_predictor",
            "build_sam3_multiplex_video_model",
            "build_sam3_multiplex_video_predictor",
            "build_sam3_predictor",
            "download_ckpt_from_hf",
        ),
    ),
    *_surfaces(
        ApiGroup.IMAGE,
        "model/sam3_image_processor.py",
        "Sam3Processor",
        (
            "__init__",
            "set_image",
            "set_image_batch",
            "set_text_prompt",
            "add_geometric_prompt",
            "reset_all_prompts",
            "set_confidence_threshold",
        ),
    ),
    *_surfaces(
        ApiGroup.INTERACTIVE,
        "model/sam1_task_predictor.py",
        "SAM3InteractiveImagePredictor",
        (
            "__init__",
            "set_image",
            "set_image_batch",
            "predict",
            "predict_batch",
            "get_image_embedding",
            "reset_predictor",
        ),
    ),
    *_surfaces(
        ApiGroup.INTERACTIVE,
        "model/sam3_image.py",
        "Sam3Image",
        (
            "predict_inst",
            "predict_inst_batch",
        ),
    ),
    *_surfaces(
        ApiGroup.VIDEO,
        "model/sam3_base_predictor.py",
        "Sam3BasePredictor",
        (
            "handle_request",
            "handle_stream_request",
            "start_session",
            "add_prompt",
            "add_mask",
            "remove_object",
            "propagate_in_video",
            "cancel_propagation",
            "reset_session",
            "close_session",
            "shutdown",
        ),
    ),
    *_surfaces(
        ApiGroup.VIDEO,
        "model/sam3_video_predictor.py",
        "Sam3VideoPredictor",
        ("__init__", "remove_object"),
    ),
    *_surfaces(
        ApiGroup.MULTIPLEX,
        "model/sam3_multiplex_video_predictor.py",
        "Sam3MultiplexVideoPredictor",
        ("__init__",),
    ),
    *_surfaces(
        ApiGroup.MULTIPLEX,
        "model/video_tracking_multiplex_demo.py",
        "Sam3VideoTrackingMultiplexDemo",
        ("init_state",),
    ),
    *_surfaces(
        ApiGroup.MULTIPLEX,
        "model/video_tracking_multiplex_demo.py",
        "VideoTrackingMultiplexDemo",
        ("add_new_masks",),
    ),
)


def _find_function(
    tree: ast.Module, symbol: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    body: list[ast.stmt] = tree.body
    node: ast.stmt | None = None
    for part in symbol.split("."):
        matches = [
            item
            for item in body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == part
        ]
        if not matches:
            return None
        node = matches[-1]
        body = node.body
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def _name_binding(
    name: str, tree: ast.Module, before_line: int | None
) -> ast.stmt | None:
    binding: ast.stmt | None = None
    for item in tree.body:
        if before_line is not None and item.lineno >= before_line:
            continue
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in item.targets
        ):
            binding = item
        elif (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == name
        ):
            binding = item
        elif isinstance(item, ast.ImportFrom) and any(
            (entry.asname or entry.name) == name for entry in item.names
        ):
            binding = item
    return binding


def _default_expression(
    node: ast.expr,
    tree: ast.Module,
    package_root: Path,
    *,
    before_line: int | None = None,
    module_file: Path = Path("<api-source>"),
    seen: frozenset[tuple[Path, str, int | None]] = frozenset(),
) -> str:
    if not isinstance(node, ast.Name):
        return ast.unparse(node)
    key = (module_file, node.id, before_line)
    if key in seen:
        raise ApiContractError(
            f"Cyclic default binding for {node.id} in {module_file}."
        )
    visited = seen | {key}
    binding = _name_binding(node.id, tree, before_line)
    if isinstance(binding, (ast.Assign, ast.AnnAssign)) and binding.value is not None:
        return _default_expression(
            binding.value,
            tree,
            package_root,
            before_line=binding.lineno,
            module_file=module_file,
            seen=visited,
        )
    if isinstance(binding, ast.ImportFrom) and binding.module and binding.level == 0:
        alias = next(
            entry for entry in binding.names if (entry.asname or entry.name) == node.id
        )
        parts = binding.module.split(".")
        if parts[0] == package_root.name:
            imported_path = package_root.joinpath(*parts[1:]).with_suffix(".py")
            imported = ast.parse(imported_path.read_text(encoding="utf-8"))
            return _default_expression(
                ast.Name(id=alias.name),
                imported,
                package_root,
                module_file=imported_path,
                seen=visited,
            )
    return ast.unparse(node)


def parse_api_signature(
    source: str, symbol: str, *, package_root: Path | None = None
) -> ApiSignature | None:
    """Read effective declarations without importing Torch, MLX, or executing defaults."""
    try:
        tree = ast.parse(source)
        compile(tree, "<api-source>", "exec")
    except (SyntaxError, ValueError) as exc:
        raise ApiContractError(
            f"Could not parse API source for {symbol}: {exc}"
        ) from exc
    function = _find_function(tree, symbol)
    if function is None:
        return None
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    parameters: list[ApiParameter] = []
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = (
            ParameterKind.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else ParameterKind.POSITIONAL
        )
        parameters.append(
            _parameter(argument.arg, kind, default, tree, package_root, function.lineno)
        )
    if args.vararg:
        parameters.append(
            ApiParameter(args.vararg.arg, ParameterKind.VAR_POSITIONAL, None)
        )
    parameters.extend(
        _parameter(
            argument.arg,
            ParameterKind.KEYWORD_ONLY,
            default,
            tree,
            package_root,
            function.lineno,
        )
        for argument, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    if args.kwarg:
        parameters.append(ApiParameter(args.kwarg.arg, ParameterKind.VAR_KEYWORD, None))
    return ApiSignature(tuple(parameters))


def _parameter(
    name: str,
    kind: ParameterKind,
    default: ast.expr | None,
    tree: ast.Module,
    package_root: Path | None,
    before_line: int,
) -> ApiParameter:
    expression = None if default is None else ast.unparse(default)
    if default is not None and package_root is not None:
        expression = _default_expression(
            default, tree, package_root, before_line=before_line
        )
    return ApiParameter(name, kind, expression)


def _parameter_differences(
    official: ApiSignature, local: ApiSignature
) -> tuple[ApiDifference, ...]:
    differences: list[ApiDifference] = []
    local_by_name = {parameter.name: parameter for parameter in local.parameters}
    for parameter in official.parameters:
        observed = local_by_name.get(parameter.name)
        if observed is None:
            differences.append(
                ApiDifference(parameter.name, "Upstream parameter is missing.")
            )
            continue
        if observed.kind != parameter.kind:
            differences.append(
                ApiDifference(
                    parameter.name,
                    f"Parameter kind: {parameter.kind} -> {observed.kind}.",
                )
            )
        if observed.default != parameter.default:
            differences.append(
                ApiDifference(
                    parameter.name,
                    f"Default: {parameter.default!r} -> {observed.default!r}.",
                    (parameter.name, parameter.default, observed.default)
                    in BACKEND_DEFAULT_ADAPTATIONS,
                )
            )
    positional_kinds = {ParameterKind.POSITIONAL, ParameterKind.POSITIONAL_ONLY}
    official_order = tuple(
        parameter.name
        for parameter in official.parameters
        if parameter.kind in positional_kinds
    )
    local_order = tuple(
        parameter.name
        for parameter in local.parameters
        if parameter.kind in positional_kinds
    )
    if local_order[: len(official_order)] != official_order:
        differences.append(
            ApiDifference(
                "<positional-order>",
                f"Positional arguments: {official_order} -> {local_order}.",
            )
        )
    if len(local_order) > len(official_order) and any(
        parameter.kind is ParameterKind.VAR_POSITIONAL
        for parameter in official.parameters
    ):
        differences.append(
            ApiDifference(
                "<variadic-positional>",
                "Added positional parameter consumes an upstream variadic argument.",
            )
        )
    official_names = {parameter.name for parameter in official.parameters}
    for parameter in local.parameters:
        if (
            parameter.name not in official_names
            and parameter.kind in {*positional_kinds, ParameterKind.KEYWORD_ONLY}
            and parameter.default is None
        ):
            differences.append(
                ApiDifference(parameter.name, "Port adds a required argument.")
            )
    return tuple(differences)


def compare_api_surface(
    surface: ApiSurface, official: ApiSignature, local: ApiSignature | None
) -> ApiDecision:
    """Adjudicate API declarations; never infer runtime or numerical qualification."""
    if local is None:
        return ApiDecision(surface, ApiVerdict.MISSING, official, None, ())
    differences = _parameter_differences(official, local)
    verdict = ApiVerdict.MATCH
    if differences:
        verdict = (
            ApiVerdict.BACKEND_DIFFERENCE
            if all(item.backend_specific for item in differences)
            else ApiVerdict.GAP
        )
    return ApiDecision(surface, verdict, official, local, differences)


def audit_inference_api(
    official_root: Path, local_root: Path
) -> tuple[ApiDecision, ...]:
    """Load each backing source once and decide every inference API surface."""
    sources: dict[tuple[Path, Path], str] = {}
    decisions: list[ApiDecision] = []
    for surface in INFERENCE_API_SURFACES:
        signatures: list[ApiSignature | None] = []
        for root, declaration in (
            (official_root, surface.official),
            (local_root, surface.local),
        ):
            key = (root, declaration.source_file)
            if key not in sources:
                sources[key] = (root / declaration.source_file).read_text(
                    encoding="utf-8"
                )
            signatures.append(
                parse_api_signature(sources[key], declaration.symbol, package_root=root)
            )
        official, local = signatures
        if official is None:
            raise ApiContractError(f"Pinned upstream API is missing {surface.symbol}.")
        decisions.append(compare_api_surface(surface, official, local))
    return tuple(decisions)
