"""Language-neutral syntax parser registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.domain.language import ParserTier
from app.parsing.extract import parse_with_profile
from app.parsing.model import LanguageProfile, ParserStatus, SyntaxGraph
from app.parsing.profiles import PROFILES, profile_for
from app.parsing.python_graph import parse_python_graph
from app.parsing.treesitter import native_parser_status, treesitter_available
from app.parsing.treesitter_graph import parse_treesitter_graph

ParserFn = Callable[[Path, str], SyntaxGraph]


class SyntaxParserRegistry:
    """Maps language ids to graph builders. Core code looks up by id.

    Registry metadata describes *installed* capability. Each SyntaxGraph reports
    the parser that actually processed that file.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, ParserFn] = {}
        self._backends: dict[str, str] = {}
        self._tiers: dict[str, ParserTier] = {}

    def register(
        self,
        language_id: str,
        parser: ParserFn,
        *,
        replace: bool = False,
        backend: str = "custom",
        tier: ParserTier = ParserTier.FULL_AST,
    ) -> None:
        key = language_id.strip().lower()
        if key in self._parsers and not replace:
            raise ValueError(f"syntax parser {key!r} is already registered")
        self._parsers[key] = parser
        self._backends[key] = backend
        self._tiers[key] = tier

    def has(self, language_id: str) -> bool:
        return language_id.strip().lower() in self._parsers

    def parse(self, language_id: str, file_path: Path, source: str) -> SyntaxGraph:
        key = language_id.strip().lower()
        parser = self._parsers.get(key)
        if parser is None:
            raise KeyError(f"No syntax parser registered for {key!r}")
        return parser(file_path, source)

    def backend(self, language_id: str) -> str:
        return self._backends.get(language_id.strip().lower(), "unknown")

    def tier(self, language_id: str) -> ParserTier:
        return self._tiers.get(language_id.strip().lower(), ParserTier.DETECTION_ONLY)

    def available_ids(self) -> list[str]:
        return sorted(self._parsers)


def _profile_parser(
    profile: LanguageProfile, *, reason: str = "native parser unavailable"
) -> ParserFn:
    def parse(file_path: Path, source: str) -> SyntaxGraph:
        graph = parse_with_profile(profile, file_path, source)
        graph.parser_backend = "profile"
        graph.parser_tier = ParserTier.PROFILE_FALLBACK
        graph.diagnostics = graph.diagnostics.__class__(
            has_errors=graph.diagnostics.has_errors,
            error_count=graph.diagnostics.error_count,
            error_spans=graph.diagnostics.error_spans,
            recoverable=graph.diagnostics.recoverable,
            truncated=graph.diagnostics.truncated,
            message=graph.diagnostics.message or reason,
            native_available=False,
            status=ParserStatus.PROFILE_FALLBACK,
            fallback_reason=reason,
        )
        return graph

    return parse


def _treesitter_or_profile(language_id: str, profile: LanguageProfile) -> ParserFn:
    fallback = _profile_parser(profile, reason="profile fallback used")

    def parse(file_path: Path, source: str) -> SyntaxGraph:
        native = treesitter_available(language_id) or (
            language_id == "typescript"
            and str(file_path).endswith(".tsx")
            and treesitter_available("tsx")
        )
        if native:
            graph = parse_treesitter_graph(language_id, file_path, source)
            if graph is not None:
                return graph
            return fallback(file_path, source)
        return _profile_parser(profile, reason="native parser unavailable")(file_path, source)

    return parse


def default_syntax_registry() -> SyntaxParserRegistry:
    registry = SyntaxParserRegistry()
    registry.register(
        "python",
        parse_python_graph,
        backend="cpython_ast",
        tier=ParserTier.FULL_AST,
    )
    for language_id, profile in PROFILES.items():
        if language_id == "python":
            continue
        if language_id == "solidity":
            from app.parsing.solidity_graph import parse_solidity_source

            available = treesitter_available(language_id)
            registry.register(
                language_id,
                parse_solidity_source,
                backend="tree_sitter" if available else "profile",
                tier=ParserTier.FULL_AST if available else ParserTier.PROFILE_FALLBACK,
            )
            continue
        if treesitter_available(language_id):
            registry.register(
                language_id,
                _treesitter_or_profile(language_id, profile),
                backend="tree_sitter",
                tier=ParserTier.FULL_AST,
            )
        else:
            registry.register(
                language_id,
                _profile_parser(profile, reason="native parser unavailable"),
                backend="profile",
                tier=ParserTier.PROFILE_FALLBACK,
            )
    return registry


_REGISTRY: SyntaxParserRegistry | None = None


def get_syntax_registry() -> SyntaxParserRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = default_syntax_registry()
    return _REGISTRY


def reset_syntax_registry() -> None:
    global _REGISTRY
    _REGISTRY = None


def parse_source(language_id: str, file_path: Path, source: str) -> SyntaxGraph:
    return get_syntax_registry().parse(language_id, file_path, source)


def can_parse(language_id: str) -> bool:
    return get_syntax_registry().has(language_id) or profile_for(language_id) is not None


def parser_backend_for(language_id: str) -> str:
    """Installed backend. Per-file truth lives on SyntaxGraph.parser_backend."""
    key = language_id.strip().lower()
    if key == "python":
        return "cpython_ast"
    if treesitter_available(key):
        return "tree_sitter"
    if profile_for(key) is not None:
        return "profile"
    if get_syntax_registry().has(key):
        return get_syntax_registry().backend(key)
    return "none"


def parser_tier_for(language_id: str) -> ParserTier:
    """Installed capability. Per-file truth lives on SyntaxGraph.parser_tier."""
    key = language_id.strip().lower()
    if key == "python":
        return ParserTier.FULL_AST
    if treesitter_available(key):
        return ParserTier.FULL_AST
    if profile_for(key) is not None:
        return ParserTier.PROFILE_FALLBACK
    if get_syntax_registry().has(key):
        return get_syntax_registry().tier(key)
    return ParserTier.DETECTION_ONLY


def installed_parser_report(language_id: str) -> dict[str, str | bool]:
    key = language_id.strip().lower()
    status = native_parser_status(key) if key != "python" else None
    native = True if key == "python" else bool(status and status.available)
    if key == "python":
        reason = "native parser available"
    elif status is not None:
        reason = status.reason
    else:
        reason = "native parser unavailable"
    tier = parser_tier_for(key)
    backend = parser_backend_for(key)
    return {
        "language_id": key,
        "native_available": native,
        "parser_backend": backend,
        "parser_tier": str(tier),
        "status": (ParserStatus.NATIVE_AVAILABLE if native else ParserStatus.NATIVE_UNAVAILABLE),
        "reason": reason,
    }
