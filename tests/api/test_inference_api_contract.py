from __future__ import annotations

import ast
from pathlib import Path

from hypothesis import given, strategies as st
import pytest

from sam3_mlx.api_contract import (
    ApiContractError,
    ApiGroup,
    ApiSurface,
    ApiVerdict,
    compare_api_surface,
    parse_api_signature,
)


SURFACE = ApiSurface(ApiGroup.BUILDERS, Path("model_builder.py"), "predict")


def test_contract_detects_changed_defaults_missing_args_and_reordered_positionals() -> (
    None
):
    official = parse_api_signature(
        "def predict(image, version='sam3.1', *, threshold=0.5): pass", "predict"
    )
    local = parse_api_signature(
        "def predict(version='sam3', image=None): pass", "predict"
    )
    assert official is not None and local is not None
    decision = compare_api_surface(SURFACE, official, local)
    assert decision.verdict is ApiVerdict.GAP
    assert {item.parameter for item in decision.differences} == {
        "image",
        "version",
        "threshold",
        "<positional-order>",
    }
    assert compare_api_surface(SURFACE, official, None).verdict is ApiVerdict.MISSING


def test_optional_port_extensions_preserve_signature_but_required_extensions_fail() -> (
    None
):
    official = parse_api_signature(
        "def predict(image, *, threshold=0.5): pass", "predict"
    )
    extended = parse_api_signature(
        "def predict(image, *, threshold=0.5, precision='fp32'): pass", "predict"
    )
    required = parse_api_signature(
        "def predict(image, *, threshold=0.5, precision): pass", "predict"
    )
    assert official is not None and extended is not None and required is not None
    assert compare_api_surface(SURFACE, official, extended).verdict is ApiVerdict.MATCH
    decision = compare_api_surface(SURFACE, official, required)
    assert decision.verdict is ApiVerdict.GAP
    assert [(item.parameter, item.detail) for item in decision.differences] == [
        ("precision", "Port adds a required argument.")
    ]


def test_backend_default_difference_is_visible_and_does_not_hide_version_gap() -> None:
    official = parse_api_signature(
        "def predict(device='cuda', version='sam3.1'): pass", "predict"
    )
    adapted = parse_api_signature(
        "def predict(device='mlx', version='sam3.1'): pass", "predict"
    )
    changed = parse_api_signature(
        "def predict(device='mlx', version='sam3'): pass", "predict"
    )
    assert official is not None and adapted is not None and changed is not None
    assert (
        compare_api_surface(SURFACE, official, adapted).verdict
        is ApiVerdict.BACKEND_DIFFERENCE
    )
    assert compare_api_surface(SURFACE, official, changed).verdict is ApiVerdict.GAP
    invalid_backend = parse_api_signature(
        "def predict(device='tpu', version='sam3.1'): pass", "predict"
    )
    assert invalid_backend is not None
    assert (
        compare_api_surface(SURFACE, official, invalid_backend).verdict
        is ApiVerdict.GAP
    )


def test_optional_positional_argument_cannot_consume_upstream_varargs() -> None:
    official = parse_api_signature("def predict(image, *args): pass", "predict")
    local = parse_api_signature(
        "def predict(image, precision='fp32', *args): pass", "predict"
    )
    assert official is not None and local is not None
    decision = compare_api_surface(SURFACE, official, local)
    assert decision.verdict is ApiVerdict.GAP
    assert [item.parameter for item in decision.differences] == [
        "<variadic-positional>"
    ]


def test_parser_uses_effective_method_and_resolves_package_constant(
    tmp_path: Path,
) -> None:
    package = tmp_path / "example"
    package.mkdir()
    (package / "settings.py").write_text("RESOLUTION = 1008\n", encoding="utf-8")
    source = "from example.settings import RESOLUTION\nclass Processor:\n def predict(self, resolution): ...\n def predict(self, resolution=RESOLUTION): return resolution\n"
    signature = parse_api_signature(source, "Processor.predict", package_root=package)
    assert signature is not None
    assert [(item.name, item.default) for item in signature.parameters] == [
        ("self", None),
        ("resolution", "1008"),
    ]


def test_default_binding_uses_value_at_definition_and_rejects_import_cycles(
    tmp_path: Path,
) -> None:
    package = tmp_path / "example"
    package.mkdir()
    source = "VALUE=1\nVALUE=2\ndef predict(value=VALUE): pass\nVALUE=3\n"
    signature = parse_api_signature(source, "predict", package_root=package)
    assert signature is not None
    assert signature.parameters[0].default == "2"
    (package / "first.py").write_text(
        "from example.second import LIMIT\n", encoding="utf-8"
    )
    (package / "second.py").write_text(
        "from example.first import LIMIT\n", encoding="utf-8"
    )
    cyclic = "from example.first import LIMIT\ndef predict(value=LIMIT): pass\n"
    with pytest.raises(ApiContractError, match="Cyclic default binding"):
        parse_api_signature(cyclic, "predict", package_root=package)


@pytest.mark.parametrize(
    "source",
    [
        "def predict(image, image): pass",
        "def predict(value=1, image): pass",
        "def predict(*args, *more): pass",
    ],
)
def test_malformed_python_signatures_fail_before_adjudication(source: str) -> None:
    with pytest.raises(ApiContractError, match="Could not parse API source"):
        parse_api_signature(source, "predict")


@given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=12))
def test_signature_ast_round_trip_preserves_parameter_contract(
    defaults: list[int],
) -> None:
    args = ", ".join(f"value_{index}={value}" for index, value in enumerate(defaults))
    source = f"def predict({args}):\n pass\n"
    initial = parse_api_signature(source, "predict")
    rewritten = parse_api_signature(ast.unparse(ast.parse(source)), "predict")
    assert initial is not None and rewritten is not None
    assert initial.parameters == rewritten.parameters
    assert tuple(item.default for item in initial.parameters) == tuple(
        str(value) for value in defaults
    )


@given(st.text(max_size=300))
def test_adversarial_source_yields_a_signature_or_specific_parse_error(
    source: str,
) -> None:
    try:
        signature = parse_api_signature(source, "predict")
    except ApiContractError as exc:
        assert "Could not parse API source" in str(exc)
        return
    if signature is not None:
        names = tuple(item.name for item in signature.parameters)
        assert len(names) == len(set(names))


def test_contract_rejects_source_path_escape() -> None:
    with pytest.raises(ApiContractError, match="package-relative"):
        ApiSurface(ApiGroup.IMAGE, Path("../external.py"), "predict")
