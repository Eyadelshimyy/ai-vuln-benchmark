"""
Context builder — selects relevant evidence for the AI debugger.

Enforces a character budget so the AI provider receives the minimum
context necessary, not the entire repository.

Security: All file paths are validated to be within the repository root
before any content is read.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from uuid import UUID

from app.ai.provider import (
    DebuggingRequest,
    SourceFileEvidence,
    StaticFindingEvidence,
    TestFailureEvidence,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# Regex to extract file paths from pytest tracebacks (e.g. "  File "path.py", line 42")
_TRACEBACK_FILE_RE = re.compile(r'File "([^"]+\.py)"')

_MAX_SOURCE_FILE_CHARS = 8_000  # per file limit


class ContextBuilder:
    """Loads and trims evidence for a debugging session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def build(
        self,
        project_name: str,
        repository_path: str,
        test_run_id: UUID | None,
        analysis_id: UUID | None,
    ) -> DebuggingRequest:
        repo_root = Path(repository_path).resolve()

        failing_tests = await self._load_failing_tests(test_run_id)
        mentioned_paths = self._extract_paths_from_tracebacks(failing_tests, repo_root)
        static_findings = await self._load_static_findings(analysis_id, mentioned_paths)
        source_files = self._load_source_files(mentioned_paths, repo_root)

        from app.adapters.evidence import default_debugging_collectors
        from app.domain.evidence import EvidenceSource

        evidence_bundle = default_debugging_collectors().collect_bundle(
            EvidenceSource(
                static_findings=static_findings,
                test_failures=failing_tests,
                source_files=source_files,
            )
        )

        return DebuggingRequest(
            project_name=project_name,
            repository_path=repository_path,
            failing_tests=failing_tests,
            static_findings=static_findings,
            source_files=source_files,
            max_hypotheses=settings.ai_max_hypotheses,
            evidence_bundle=evidence_bundle,
        )

    async def _load_failing_tests(self, test_run_id: UUID | None) -> list[TestFailureEvidence]:
        if test_run_id is None:
            return []
        from sqlalchemy import select

        from app.models.test_run import TestResult

        result = await self._session.execute(
            select(TestResult)
            .where(TestResult.test_run_id == test_run_id)
            .where(TestResult.status.in_(["failed", "error"]))
            .order_by(TestResult.node_id)
            .limit(10)
        )
        rows = result.scalars().all()
        return [
            TestFailureEvidence(
                node_id=r.node_id,
                test_file=r.test_file,
                test_name=r.test_name,
                traceback=r.traceback,
                stdout=r.stdout,
                stderr=r.stderr,
                duration_seconds=r.duration_seconds,
            )
            for r in rows
        ]

    def _extract_paths_from_tracebacks(
        self, tests: list[TestFailureEvidence], repo_root: Path
    ) -> list[Path]:
        seen: set[Path] = set()
        paths: list[Path] = []

        for test in tests:
            if test.test_file:
                candidate = repo_root / test.test_file
                self._add_if_safe(candidate, repo_root, seen, paths)

            if test.traceback:
                for match in _TRACEBACK_FILE_RE.finditer(test.traceback):
                    raw = match.group(1)
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = repo_root / candidate
                    self._add_if_safe(candidate, repo_root, seen, paths)

        return paths[:8]  # limit

    @staticmethod
    def _add_if_safe(
        path: Path,
        repo_root: Path,
        seen: set[Path],
        out: list[Path],
    ) -> None:
        try:
            resolved = path.resolve()
            if resolved.is_relative_to(repo_root) and resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
        except (OSError, ValueError):
            pass

    async def _load_static_findings(
        self, analysis_id: UUID | None, paths: list[Path]
    ) -> list[StaticFindingEvidence]:
        if analysis_id is None:
            return []
        from sqlalchemy import select

        from app.models.finding import DBFinding

        path_strs = [str(p) for p in paths]
        result = await self._session.execute(
            select(DBFinding)
            .where(DBFinding.analysis_id == analysis_id)
            .where(
                DBFinding.file_path.in_(path_strs)
                if path_strs
                else DBFinding.analysis_id == analysis_id
            )
            .order_by(DBFinding.severity.desc(), DBFinding.line)
            .limit(20)
        )
        rows = result.scalars().all()
        return [
            StaticFindingEvidence(
                category=r.category,
                severity=r.severity,
                confidence=r.confidence,
                file_path=r.file_path,
                line=r.line,
                message=r.message,
                explanation=r.explanation,
                evidence=r.evidence,
            )
            for r in rows
        ]

    def _load_source_files(self, paths: list[Path], repo_root: Path) -> list[SourceFileEvidence]:
        sources: list[SourceFileEvidence] = []
        char_budget = settings.ai_max_context_chars // 2  # half budget for sources

        for p in paths:
            if char_budget <= 0:
                break
            try:
                if p.stat().st_size > settings.max_file_size_bytes:
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")[:_MAX_SOURCE_FILE_CHARS]
                char_budget -= len(content)
                rel = p.relative_to(repo_root)
                from app.plugins import get_plugin_catalog

                language = get_plugin_catalog().languages.language_id_for_path(p) or "python"
                sources.append(
                    SourceFileEvidence(file_path=str(rel), content=content, language=language)
                )
            except (OSError, PermissionError, ValueError):
                continue

        return sources
