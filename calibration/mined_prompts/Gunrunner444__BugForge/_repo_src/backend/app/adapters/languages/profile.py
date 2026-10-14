"""Profile-driven language adapters sharing the SyntaxGraph substrate.

These adapters implement parse, entity/import extraction, and security
analysis. Optional quality rules opt a language into CODE_QUALITY /
STATIC_ANALYSIS. Tree-sitter is the primary parser; the regex profile
parser is an explicitly labeled fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from app.adapters.languages.base import LanguageAdapter
from app.analysis.base import AnalyzerRule
from app.analysis.finding import Finding
from app.analysis.quality import analyzer_error_finding
from app.domain.language import FULL_ANALYSIS_CAPS, LanguageCapability, ParserTier
from app.domain.source import LanguageParseResult
from app.parsing.engine import parse_source, parser_backend_for, parser_tier_for
from app.parsing.model import SyntaxGraph

logger = logging.getLogger(__name__)


class ProfileLanguageAdapter(LanguageAdapter):
    """Language adapter backed by a registered syntax-graph parser."""

    def __init__(
        self,
        language_id: str,
        display_name: str,
        extensions: frozenset[str],
        *,
        is_source: bool = True,
        rules: Sequence[AnalyzerRule] | None = None,
        specialized: bool = False,
    ) -> None:
        self._language_id = language_id
        self._display_name = display_name
        self._extensions = extensions
        self._rules: list[AnalyzerRule] = list(rules or ())
        self._specialized = specialized
        caps = set(FULL_ANALYSIS_CAPS)
        if not is_source:
            caps.discard(LanguageCapability.SOURCE)
        if self._rules:
            caps.add(LanguageCapability.CODE_QUALITY)
            caps.add(LanguageCapability.STATIC_ANALYSIS)
        if specialized:
            caps.discard(LanguageCapability.DATA_FLOW)
            caps.add(LanguageCapability.PARSE)
            caps.add(LanguageCapability.AST)
            caps.add(LanguageCapability.SECURITY_ANALYSIS)
        self._capabilities = frozenset(caps)

    @property
    def language_id(self) -> str:
        return self._language_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def file_extensions(self) -> frozenset[str]:
        return self._extensions

    @property
    def capabilities(self) -> frozenset[LanguageCapability]:
        tier = self.parser_tier()
        if tier is ParserTier.PROFILE_FALLBACK:
            reduced = set(self._capabilities)
            reduced.discard(LanguageCapability.AST)
            reduced.discard(LanguageCapability.SCOPE_ANALYSIS)
            reduced.discard(LanguageCapability.DATA_FLOW)
            return frozenset(reduced)
        return self._capabilities

    def parser_tier(self) -> ParserTier:
        if self._specialized:
            return (
                ParserTier.SPECIALIZED
                if parser_tier_for(self.language_id) is ParserTier.FULL_AST
                else ParserTier.PROFILE_FALLBACK
            )
        return parser_tier_for(self.language_id)

    def parser_backend(self) -> str:
        return parser_backend_for(self.language_id)

    def parse_file(self, file_path: Path, *, context: object | None = None) -> LanguageParseResult:
        del context
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return LanguageParseResult(
                file_path=str(file_path),
                language=self.language_id,
                errors=[str(exc)],
                has_errors=True,
                error_count=1,
                parser_backend=self.parser_backend(),
                parser_tier=str(self.parser_tier()),
            )
        return self.syntax_graph(file_path, source).to_parse_result()

    def syntax_graph(self, file_path: Path, source: str) -> SyntaxGraph:
        return parse_source(self.language_id, file_path, source)

    def analyze_file(self, file_path: Path, source: str) -> list[Finding]:
        if not self._rules:
            return super().analyze_file(file_path, source)
        graph = self.syntax_graph(file_path, source)
        findings: list[Finding] = []
        for rule in self._rules:
            try:
                if hasattr(rule, "check_graph"):
                    findings.extend(rule.check_graph(graph))
                else:
                    findings.extend(rule.check_file(file_path, source))
            except Exception as exc:
                findings.append(
                    analyzer_error_finding(
                        file_path, self.language_id, graph.parser_backend, rule.RULE_ID, exc
                    )
                )
        for finding in findings:
            if not finding.language:
                finding.language = self.language_id
            if not finding.parser_backend:
                finding.parser_backend = graph.parser_backend
        return findings

    def static_rules(self) -> Sequence[AnalyzerRule]:
        return tuple(self._rules)

    def with_static_rules(self, rules: Sequence[object]) -> ProfileLanguageAdapter:
        typed: list[AnalyzerRule] = []
        for rule in rules:
            if not isinstance(rule, AnalyzerRule):
                raise TypeError(
                    "Static-analysis rule overrides must be AnalyzerRule "
                    f"instances, got {type(rule)!r}"
                )
            typed.append(rule)
        return ProfileLanguageAdapter(
            self._language_id,
            self._display_name,
            self._extensions,
            is_source=LanguageCapability.SOURCE in self._capabilities,
            rules=typed,
            specialized=self._specialized,
        )
