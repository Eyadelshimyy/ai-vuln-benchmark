"""Deterministic repository analysis for a Cursor-controlled session.

No language-model provider is constructed or called.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from app.domain.language import ParserTier
from app.security.engine import SecurityAnalysisEngine
from app.security_agent.agent import SecurityResearchAgent
from app.security_agent.states import EvidenceGraphKind

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "_build",
    "__pycache__",
}
_MAX_FILES = 500


def analyze_session_repository(
    agent: SecurityResearchAgent, *, max_files: int = 200
) -> dict[str, Any]:
    """Parse and rule-check the session repository. Findings stay unverified."""

    limit = max(1, min(int(max_files), _MAX_FILES))
    root = Path(agent.session.repo_root)
    files = _source_files(root, limit=limit)
    result = SecurityAnalysisEngine().analyze_repository(root, files)
    parsers: list[dict[str, str]] = []
    for path, graph in sorted(result.graphs.items()):
        tier = graph.parser_tier
        parsers.append(
            {
                "path": path,
                "parser_backend": str(graph.parser_backend or ""),
                "parser_tier": tier.value if isinstance(tier, ParserTier) else str(tier),
                "parser_tier_label": _tier_label(tier),
            }
        )
    findings: list[dict[str, Any]] = []
    for item in result.findings:
        status = item.status.value
        if status == "verified":
            status = "potential"
        location = item.source_location
        findings.append(
            {
                "id": str(item.id),
                "title": item.title,
                "status": status,
                "verified": False,
                "vulnerability_class": item.vulnerability_class,
                "confidence": item.confidence,
                "rule_ids": list(item.rule_ids),
                "file": None if location is None else location.file_path,
                "line": None if location is None else location.line,
                "description": item.description[:500],
            }
        )
    counts = Counter(item["parser_tier_label"] for item in parsers)
    node = agent.session.graph.add(
        kind=EvidenceGraphKind.OBSERVATION.value,
        provenance="deterministic_analysis",
        summary=(
            f"repository analysis files={result.files_analyzed} "
            f"findings={len(findings)} full_ast={counts.get('FULL_AST', 0)} "
            f"profile_fallback={counts.get('PROFILE_FALLBACK', 0)}"
        ),
        source="cursor_external",
        extra={
            "llm_invoked": False,
            "files_analyzed": result.files_analyzed,
            "finding_count": len(findings),
            "parser_tiers": dict(counts),
        },
    )
    agent.session.graph.session_id = agent.session.id
    agent._timeline(
        "analysis",
        decision="deterministic_repository_analysis",
        evidence_id=node.id,
        result="llm_not_invoked",
    )
    return {
        "llm_invoked": False,
        "files_considered": len(files),
        "files_analyzed": result.files_analyzed,
        "findings": findings,
        "parsers": parsers,
        "parser_tier_counts": dict(counts),
        "diagnostics": [
            {
                "kind": item.kind,
                "message": item.message,
                "file": item.file_path,
                "parser_backend": item.parser_backend,
                "parser_tier": item.parser_tier,
            }
            for item in result.diagnostics[:40]
        ],
        "evidence_id": node.id,
    }


def _tier_label(tier: object) -> str:
    value = tier.value if isinstance(tier, ParserTier) else str(tier)
    if value in {ParserTier.FULL_AST.value, "FULL_AST"}:
        return "FULL_AST"
    if value in {ParserTier.PROFILE_FALLBACK.value, "PROFILE_FALLBACK"}:
        return "PROFILE_FALLBACK"
    return value or "UNKNOWN"


def _source_files(root: Path, *, limit: int) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            found.append(Path(dirpath) / name)
            if len(found) >= limit:
                return found
    return found
