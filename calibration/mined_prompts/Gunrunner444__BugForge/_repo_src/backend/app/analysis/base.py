"""Base rule interfaces for quality and (legacy) file-based analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.analysis.catalog import FindingCatalog
from app.analysis.finding import Finding
from app.parsing.model import SyntaxGraph


class AnalyzerRule(ABC):
    """A single self-contained analysis rule.

    Subclass this and implement `check_file()`.
    Every rule is responsible for exactly one concern.
    """

    RULE_ID: str
    SEVERITY: str
    CONFIDENCE: str
    CATALOG: str = FindingCatalog.CODE_QUALITY

    @abstractmethod
    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        """Analyse a single source file and return findings (empty list = clean)."""
        ...

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        """Optional syntax-graph implementation. Default delegates to check_file."""
        return self.check_file(Path(graph.file_path), graph.source)


class CodeQualityRule(AnalyzerRule):
    """Quality findings are never security observations."""

    CATALOG = FindingCatalog.CODE_QUALITY


class SecurityQualityBridge(AnalyzerRule):
    """Legacy file-based checks that belong in the security catalog."""

    CATALOG = FindingCatalog.SECURITY
