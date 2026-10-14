"""Language adapter contract.

Core analysis looks up a :class:`LanguageAdapter` from the language registry
instead of branching on language names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from app.analysis.finding import Finding
from app.domain.language import LanguageCapability, ParserTier
from app.domain.source import LanguageParseResult
from app.parsing.model import SyntaxGraph
from app.plugins.errors import UnsupportedCapabilityError


class LanguageAdapter(ABC):
    """Language-specific parse and static-analysis operations.

    Adapters that only support detection must not implement fake parsers.
    Calling :meth:`parse_file` or :meth:`analyze_file` without the matching
    capability raises :class:`UnsupportedCapabilityError`.
    """

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Stable lowercase id, e.g. ``python``."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name, e.g. ``Python``."""

    @property
    @abstractmethod
    def file_extensions(self) -> frozenset[str]:
        """Lowercase extensions including the leading dot."""

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[LanguageCapability]:
        """Capabilities this adapter actually implements."""

    def supports(self, capability: LanguageCapability) -> bool:
        return capability in self.capabilities

    def matches_path(self, path: Path) -> bool:
        return path.suffix.lower() in self.file_extensions

    def prepare_repository(self, repo_root: Path) -> object | None:
        """Optional per-repository context (local packages, module roots, ...)."""
        return None

    def parse_file(self, file_path: Path, *, context: object | None = None) -> LanguageParseResult:
        raise UnsupportedCapabilityError(
            self.language_id,
            LanguageCapability.PARSE,
            detail=f"{self.display_name} parsing is not implemented in this phase.",
        )

    def syntax_graph(self, file_path: Path, source: str) -> SyntaxGraph:
        """Return a language-neutral syntax graph used by security analysis.

        Default implementation uses the shared parser registry. Detection-only
        adapters raise :class:`UnsupportedCapabilityError`.
        """
        from app.parsing.engine import can_parse, parse_source

        if not can_parse(self.language_id):
            raise UnsupportedCapabilityError(
                self.language_id,
                LanguageCapability.PARSE,
                detail=f"{self.display_name} has no syntax parser.",
            )
        return parse_source(self.language_id, file_path, source)

    def analyze_file(self, file_path: Path, source: str) -> list[Finding]:
        raise UnsupportedCapabilityError(
            self.language_id,
            LanguageCapability.STATIC_ANALYSIS,
            detail=f"{self.display_name} static analysis is not implemented in this phase.",
        )

    def static_rules(self) -> Sequence[object]:
        return ()

    def with_static_rules(self, rules: Sequence[object]) -> LanguageAdapter:
        """Return an adapter that runs ``rules`` instead of the built-in set.

        The engine uses this instead of language-specific branching. Adapters
        that cannot apply rule overrides raise :class:`UnsupportedCapabilityError`.
        """
        raise UnsupportedCapabilityError(
            self.language_id,
            "static_rule_override",
            detail=f"{self.display_name} does not support static-analysis rule overrides.",
        )

    def parser_tier(self) -> ParserTier:
        return ParserTier.DETECTION_ONLY

    def parser_backend(self) -> str:
        return "none"

    def analysis_diagnostics(self) -> dict[str, object]:
        return {
            "language_id": self.language_id,
            "display_name": self.display_name,
            "parser_tier": str(self.parser_tier()),
            "parser_backend": self.parser_backend(),
            "capabilities": sorted(cap.value for cap in self.capabilities),
        }
