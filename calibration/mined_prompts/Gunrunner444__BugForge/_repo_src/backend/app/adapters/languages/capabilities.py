"""Runtime language capability matrix.

Values are YES, LIMITED, UNSUPPORTED, or UNAVAILABLE_AT_RUNTIME.
A Tree-sitter grammar that BugForge does not wire stays detection-only.
"""

from __future__ import annotations

from app.discovery.process import tool_path
from app.domain.language import ParserTier
from app.parsing.engine import parser_backend_for, parser_tier_for

PROMOTION_STAGES = (
    "DETECTION_ONLY",
    "PARSER_AVAILABLE",
    "FULL_AST",
    "SYMBOLS_IMPORTS_SCOPES",
    "SEMANTIC_GRAPH",
    "DATA_FLOW",
    "SECURITY_VOCABULARY",
    "CODE_QUALITY_CATALOG",
    "FIXTURE_MATRIX",
    "FULL_ANALYSIS",
)

_LEVELS = frozenset({"YES", "LIMITED", "UNSUPPORTED", "UNAVAILABLE_AT_RUNTIME"})


def capability_matrix(language_id: str) -> dict[str, str]:
    key = language_id.strip().lower()
    tier = parser_tier_for(key)
    if key == "solidity":
        report = _solidity_matrix(tier)
    elif key == "vyper":
        report = _vyper_matrix()
    elif key in {"go", "rust", "c", "cpp"}:
        report = _runtime_matrix(key, tier)
    elif tier is ParserTier.DETECTION_ONLY:
        report = _detection_matrix()
    elif tier is ParserTier.SPECIALIZED or key in {"html", "css", "scss", "sql"}:
        report = _base(tier, data_flow="UNSUPPORTED", security="LIMITED", quality="UNSUPPORTED")
    else:
        report = _base(tier, data_flow="YES", security="YES", quality="YES")
    _check(report)
    return report


def promotion_stage(language_id: str) -> str:
    key = language_id.strip().lower()
    tier = parser_tier_for(key)
    if key == "solidity":
        return "FULL_ANALYSIS"
    if tier is ParserTier.FULL_AST:
        return "FULL_ANALYSIS"
    if tier is ParserTier.SPECIALIZED:
        return "SECURITY_VOCABULARY"
    if tier is ParserTier.PROFILE_FALLBACK:
        return "PARSER_AVAILABLE"
    return "DETECTION_ONLY"


def _solidity_matrix(tier: ParserTier) -> dict[str, str]:
    fuzzers = [name for name in ("forge", "echidna", "medusa") if tool_path(name)]
    symbolic = tool_path("halmos")
    slither = tool_path("slither")
    return {
        "parser_backend": parser_backend_for("solidity"),
        "parser_tier": str(tier),
        "syntax_graph": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "entities": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "imports": "YES",
        "scopes": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "calls": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "data_flow": "LIMITED",
        "interprocedural": "LIMITED",
        "security_rules": "YES",
        "quality_rules": "YES",
        "runtime_testing": "LIMITED" if tool_path("forge") else "UNAVAILABLE_AT_RUNTIME",
        "fuzzing": "LIMITED" if fuzzers else "UNAVAILABLE_AT_RUNTIME",
        "symbolic": "LIMITED" if symbolic else "UNAVAILABLE_AT_RUNTIME",
        "framework": "YES",
        "external_static": "LIMITED" if slither else "UNAVAILABLE_AT_RUNTIME",
        "compiler_ast": "LIMITED"
        if tool_path("solc") or tool_path("forge")
        else "UNAVAILABLE_AT_RUNTIME",
        "detect": "YES",
        "parse": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "ast": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "symbols": "YES" if tier is ParserTier.FULL_AST else "LIMITED",
        "cfg": "LIMITED" if tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "property_testing": "LIMITED"
        if tool_path("echidna") or tool_path("medusa")
        else "UNAVAILABLE_AT_RUNTIME",
        "sanitizers": "UNSUPPORTED",
    }


def _vyper_matrix() -> dict[str, str]:
    report = _detection_matrix()
    report["external_static"] = "LIMITED" if tool_path("slither") else "UNAVAILABLE_AT_RUNTIME"
    report["security_rules"] = "UNSUPPORTED"
    report["cfg"] = "UNSUPPORTED"
    report["property_testing"] = "UNSUPPORTED"
    report["sanitizers"] = "UNSUPPORTED"
    return report


def _runtime_matrix(language_id: str, tier: ParserTier) -> dict[str, str]:
    report = _base(tier, data_flow="YES", security="YES", quality="YES")
    if language_id == "go":
        present = tool_path("go") is not None
        report["runtime_testing"] = "LIMITED" if present else "UNAVAILABLE_AT_RUNTIME"
        report["fuzzing"] = "LIMITED" if present else "UNAVAILABLE_AT_RUNTIME"
    elif language_id == "rust":
        present = tool_path("cargo") is not None
        report["runtime_testing"] = "LIMITED" if present else "UNAVAILABLE_AT_RUNTIME"
        report["fuzzing"] = "LIMITED" if tool_path("cargo-fuzz") else "UNAVAILABLE_AT_RUNTIME"
        report["sanitizers"] = "LIMITED" if present else "UNAVAILABLE_AT_RUNTIME"
    else:
        present = tool_path("clang") is not None or tool_path("afl-fuzz") is not None
        report["runtime_testing"] = "UNAVAILABLE_AT_RUNTIME"
        report["fuzzing"] = "LIMITED" if present else "UNAVAILABLE_AT_RUNTIME"
        report["sanitizers"] = "LIMITED" if tool_path("clang") else "UNAVAILABLE_AT_RUNTIME"
    report["cfg"] = "UNSUPPORTED"
    report["property_testing"] = "UNSUPPORTED"
    report["symbolic"] = "UNSUPPORTED"
    return report


def _detection_matrix() -> dict[str, str]:
    return _base(
        ParserTier.DETECTION_ONLY,
        data_flow="UNSUPPORTED",
        security="UNSUPPORTED",
        quality="UNSUPPORTED",
    )


def _base(tier: ParserTier, *, data_flow: str, security: str, quality: str) -> dict[str, str]:
    parsed = tier in {ParserTier.FULL_AST, ParserTier.SPECIALIZED}
    level = "YES" if parsed else "UNSUPPORTED"
    return {
        "parser_backend": "none" if tier is ParserTier.DETECTION_ONLY else str(tier),
        "parser_tier": str(tier),
        "syntax_graph": level,
        "entities": level,
        "imports": level,
        "scopes": "YES" if tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "calls": "YES" if tier is ParserTier.FULL_AST else "LIMITED" if parsed else "UNSUPPORTED",
        "data_flow": data_flow if tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "interprocedural": "LIMITED" if tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "security_rules": security if parsed or tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "quality_rules": quality if tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "runtime_testing": "UNSUPPORTED",
        "fuzzing": "UNSUPPORTED",
        "symbolic": "UNSUPPORTED",
        "framework": "LIMITED" if tier is ParserTier.FULL_AST else "UNSUPPORTED",
        "external_static": "UNSUPPORTED",
        "compiler_ast": "UNSUPPORTED",
        "detect": "YES",
        "parse": "YES" if parsed else "UNSUPPORTED",
        "ast": level,
        "symbols": level,
        "cfg": "UNSUPPORTED",
        "property_testing": "UNSUPPORTED",
        "sanitizers": "UNSUPPORTED",
    }


def _check(report: dict[str, str]) -> None:
    for key, value in report.items():
        if key in {"parser_backend", "parser_tier"}:
            continue
        if value not in _LEVELS:
            raise ValueError(f"{key} has ambiguous capability {value!r}")
