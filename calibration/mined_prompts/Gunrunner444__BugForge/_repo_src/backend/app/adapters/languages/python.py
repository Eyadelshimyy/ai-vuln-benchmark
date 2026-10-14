"""Python language adapter — wraps the existing parser and static-analysis rules."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from app.adapters.languages.base import LanguageAdapter
from app.analysis.base import AnalyzerRule
from app.analysis.finding import Finding
from app.analysis.python_analyzer import ALL_PYTHON_RULES
from app.analysis.quality import analyzer_error_finding
from app.analyzers.python.language_analyzer import PythonLanguageAnalyzer
from app.domain.language import FULL_ANALYSIS_CAPS, LanguageCapability, ParserTier
from app.domain.source import LanguageParseResult

logger = logging.getLogger(__name__)


class PythonAdapter(LanguageAdapter):
    """Fully functional Python adapter used by repository and static analysis."""

    def __init__(self, rules: Sequence[AnalyzerRule] | None = None) -> None:
        self._analyzer = PythonLanguageAnalyzer()
        self._rules: list[AnalyzerRule] = (
            list(rules) if rules is not None else list(ALL_PYTHON_RULES)
        )

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def display_name(self) -> str:
        return "Python"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".py", ".pyi"})

    @property
    def capabilities(self) -> frozenset[LanguageCapability]:
        return frozenset(
            FULL_ANALYSIS_CAPS
            | {LanguageCapability.CODE_QUALITY, LanguageCapability.STATIC_ANALYSIS}
        )

    def parser_tier(self) -> ParserTier:
        return ParserTier.FULL_AST

    def parser_backend(self) -> str:
        return "cpython_ast"

    def prepare_repository(self, repo_root: Path) -> frozenset[str]:
        return PythonLanguageAnalyzer.discover_local_packages(repo_root)

    def parse_file(self, file_path: Path, *, context: object | None = None) -> LanguageParseResult:
        local_packages: frozenset[str] | None = None
        if isinstance(context, frozenset):
            local_packages = context
        return self._analyzer.analyze_file(file_path, local_packages=local_packages)

    def analyze_file(self, file_path: Path, source: str) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self._rules:
            try:
                findings.extend(rule.check_file(file_path, source))
            except Exception as exc:
                findings.append(
                    analyzer_error_finding(
                        file_path, self.language_id, self.parser_backend(), rule.RULE_ID, exc
                    )
                )
        return findings

    def static_rules(self) -> Sequence[AnalyzerRule]:
        return tuple(self._rules)

    def with_static_rules(self, rules: Sequence[object]) -> PythonAdapter:
        typed: list[AnalyzerRule] = []
        for rule in rules:
            if not isinstance(rule, AnalyzerRule):
                raise TypeError(
                    "Python static-analysis rule overrides must be AnalyzerRule "
                    f"instances, got {type(rule)!r}"
                )
            typed.append(rule)
        return PythonAdapter(rules=typed)
