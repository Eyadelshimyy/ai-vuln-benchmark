"""Static analysis engine — orchestrates language adapters across files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.analysis.base import AnalyzerRule
from app.analysis.finding import Finding
from app.core.config import settings
from app.core.paths import to_relative_path
from app.domain.language import LanguageCapability
from app.plugins.errors import AdapterNotFoundError, UnsupportedCapabilityError

if TYPE_CHECKING:
    from app.adapters.languages.base import LanguageAdapter
    from app.adapters.languages.registry import LanguageRegistry

logger = logging.getLogger(__name__)


class StaticAnalysisEngine:
    """Runs registered language analyzers against source files in a repository."""

    def __init__(self, rules: list[AnalyzerRule] | None = None) -> None:
        # Optional override is applied through LanguageAdapter.with_static_rules().
        self._rule_override: list[AnalyzerRule] | None = rules

    def analyze_repository(self, repo_path: Path, file_paths: list[Path]) -> list[Finding]:
        """Run analyzers over the supplied file paths. Returns findings with RELATIVE paths."""
        from app.plugins import get_plugin_catalog

        languages = get_plugin_catalog().languages
        enabled_ids = self._resolve_enabled_analyzers(languages)

        findings: list[Finding] = []
        analyzed_files = 0

        for file_path in file_paths:
            adapter = languages.for_path(file_path)
            if adapter is None or not adapter.supports(LanguageCapability.STATIC_ANALYSIS):
                continue
            if adapter.language_id not in enabled_ids:
                continue

            file_findings = self._analyze_file(file_path, adapter)
            analyzed_files += 1
            for finding in file_findings:
                try:
                    finding.file_path = to_relative_path(Path(finding.file_path), repo_path)
                except ValueError:
                    pass
                findings.append(finding)

        logger.info(
            "Static analysis complete: %d findings across %d analyzed files",
            len(findings),
            analyzed_files,
        )
        return findings

    def _resolve_enabled_analyzers(self, languages: LanguageRegistry) -> set[str]:
        requested = settings.get_enabled_language_analyzers()
        capable = {
            adapter.language_id
            for adapter in languages.all_adapters()
            if adapter.supports(LanguageCapability.STATIC_ANALYSIS)
        }
        if not requested:
            return capable

        enabled: set[str] = set()
        for name in requested:
            if not languages.has(name):
                raise AdapterNotFoundError("language", name, languages.available_ids())
            adapter = languages.get(name)
            if not adapter.supports(LanguageCapability.STATIC_ANALYSIS):
                raise UnsupportedCapabilityError(
                    adapter.language_id,
                    LanguageCapability.STATIC_ANALYSIS,
                    detail=(
                        f"{adapter.display_name} is detected but has no analyzer yet. "
                        "Remove it from LANGUAGE_ANALYZERS or implement the adapter."
                    ),
                )
            enabled.add(adapter.language_id)
        return enabled

    def _analyze_file(self, file_path: Path, adapter: LanguageAdapter) -> list[Finding]:
        try:
            size = file_path.stat().st_size
        except OSError:
            return []

        if size > settings.max_file_size_bytes:
            logger.debug("Skipping oversized file for static analysis: %s", file_path)
            return []

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as exc:
            logger.warning("Cannot read %s for static analysis: %s", file_path, exc)
            return []

        if self._rule_override is not None:
            try:
                adapter = adapter.with_static_rules(self._rule_override)
            except UnsupportedCapabilityError:
                # Rule overrides are adapter-specific. Languages that cannot
                # apply them keep their built-in rules instead of special-casing
                # Python in this engine.
                pass
        return adapter.analyze_file(file_path, source)
