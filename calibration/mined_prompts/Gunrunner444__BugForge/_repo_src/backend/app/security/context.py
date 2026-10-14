"""Select a bounded, relevant context window for local security models.

Never dumps an entire repository into the model. Practical budgets are
enforced independently of advertised context-window sizes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.analyzers.framework_detector import FrameworkInfo
from app.core.config import settings
from app.domain.source import ParsedEntity
from app.parsing.model import SyntaxGraph
from app.security.correlation import ObservationCluster
from app.security.rules.base import SecurityObservation

_CONFIG_NAMES = frozenset(
    {
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "Gemfile",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
        "composer.json",
        "settings.py",
        "config.py",
        ".env.example",
        "next.config.js",
        "tsconfig.json",
    }
)


@dataclass(frozen=True)
class ContextChunk:
    path: str
    kind: str
    score: float
    content: str
    language: str = ""
    start_line: int = 1


@dataclass
class SelectedContext:
    chunks: list[ContextChunk] = field(default_factory=list)
    observations: list[SecurityObservation] = field(default_factory=list)
    frameworks: list[FrameworkInfo] = field(default_factory=list)
    char_count: int = 0
    omitted: int = 0
    cache_key: str = ""


class SecurityContextBuilder:
    """Score, chunk, and budget repository material for one cluster."""

    def __init__(
        self,
        *,
        budget_chars: int | None = None,
        max_file_chars: int = 4_000,
        max_files: int = 12,
    ) -> None:
        self._budget = budget_chars or min(settings.ai_max_context_chars, 12_000)
        self._max_file_chars = max_file_chars
        self._max_files = max_files
        self._cache: dict[str, SelectedContext] = {}

    def build(
        self,
        *,
        repo_root: Path,
        cluster: ObservationCluster,
        graphs: dict[str, SyntaxGraph],
        frameworks: list[FrameworkInfo],
    ) -> SelectedContext:
        key = self._cache_key(repo_root, cluster)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        candidates = self._score(repo_root, cluster, graphs, frameworks)
        selected: list[ContextChunk] = []
        used = 0
        omitted = 0
        seen_content: set[str] = set()
        for chunk in candidates:
            digest = hashlib.sha256(chunk.content.encode("utf-8", errors="replace")).hexdigest()
            if digest in seen_content:
                omitted += 1
                continue
            if len(selected) >= self._max_files or used + len(chunk.content) > self._budget:
                omitted += 1
                continue
            seen_content.add(digest)
            selected.append(chunk)
            used += len(chunk.content)

        result = SelectedContext(
            chunks=selected,
            observations=list(cluster.observations),
            frameworks=frameworks,
            char_count=used,
            omitted=omitted,
            cache_key=key,
        )
        self._cache[key] = result
        return result

    def _score(
        self,
        repo_root: Path,
        cluster: ObservationCluster,
        graphs: dict[str, SyntaxGraph],
        frameworks: list[FrameworkInfo],
    ) -> list[ContextChunk]:
        scored: list[ContextChunk] = []
        primary = cluster.file_path
        graph = graphs.get(primary)
        if graph is not None:
            snippet = _window(graph.source, cluster.line, radius=25)
            scored.append(
                ContextChunk(
                    path=_rel(primary, repo_root),
                    kind="affected_source",
                    score=100.0,
                    content=snippet,
                    language=graph.language,
                    start_line=max(1, cluster.line - 25),
                )
            )
            enclosing = _enclosing_entity(graph, cluster.line)
            if enclosing is not None:
                scored.append(
                    ContextChunk(
                        path=_rel(primary, repo_root),
                        kind="function",
                        score=95.0,
                        content=_entity_window(graph, enclosing),
                        language=graph.language,
                        start_line=enclosing.start_line,
                    )
                )
            calls = ", ".join(
                sorted({c.qualified for c in graph.calls if abs(c.line - cluster.line) <= 8})[:20]
            )
            if calls:
                scored.append(
                    ContextChunk(
                        path=_rel(primary, repo_root),
                        kind="calls",
                        score=80.0,
                        content=calls,
                        language=graph.language,
                    )
                )
            sources = ", ".join(
                f"{b.name}={b.rhs[:80]}"
                for b in graph.bindings
                if not b.rhs_is_literal and abs(b.line - cluster.line) <= 30
            )[:1500]
            if sources:
                scored.append(
                    ContextChunk(
                        path=_rel(primary, repo_root),
                        kind="data_flow",
                        score=78.0,
                        content=sources,
                        language=graph.language,
                    )
                )
            imports = ", ".join(imp.module for imp in graph.imports[:20])
            if imports:
                scored.append(
                    ContextChunk(
                        path=_rel(primary, repo_root),
                        kind="imports",
                        score=70.0,
                        content=imports,
                        language=graph.language,
                    )
                )
            symbols = ", ".join(ent.qualified_name for ent in graph.entities[:20])
            if symbols:
                scored.append(
                    ContextChunk(
                        path=_rel(primary, repo_root),
                        kind="symbols",
                        score=65.0,
                        content=symbols,
                        language=graph.language,
                    )
                )
            scored.append(
                ContextChunk(
                    path=_rel(primary, repo_root),
                    kind="parser",
                    score=50.0,
                    content=(
                        f"language={graph.language} backend={graph.parser_backend} "
                        f"tier={graph.parser_tier} framework={graph.framework or 'none'} "
                        f"context={graph.file_context} "
                        f"taint_precision=flow-sensitive path_sensitive=false"
                    ),
                    language=graph.language,
                )
            )

        for obs in cluster.observations:
            scored.append(
                ContextChunk(
                    path=_rel(obs.file_path, repo_root),
                    kind="static_finding",
                    score=90.0,
                    content=(
                        f"{obs.rule_id} L{obs.line}: {obs.summary}\n"
                        f"node={obs.node_id} scope={obs.scope_id} "
                        f"sink={obs.sink_id} arg={obs.argument_index} "
                        f"taint={obs.taint_path}\n{obs.evidence_text}"
                    ),
                    language=obs.language,
                    start_line=obs.line,
                )
            )

        if frameworks:
            names = ", ".join(f"{fw.name} ({fw.language})" for fw in frameworks)
            evidence = ", ".join(ev for fw in frameworks for ev in fw.evidence[:3])
            scored.append(
                ContextChunk(
                    path="[frameworks]",
                    kind="framework",
                    score=55.0,
                    content=f"{names}. evidence: {evidence}",
                )
            )

        for graph in graphs.values():
            rel = _rel(graph.file_path, repo_root)
            name = Path(rel).name
            if name in _CONFIG_NAMES or Path(rel).suffix in {".toml", ".ini", ".yml"}:
                scored.append(
                    ContextChunk(
                        path=rel,
                        kind="config",
                        score=40.0,
                        content=graph.source[: self._max_file_chars],
                        language=graph.language,
                    )
                )

        scored.sort(key=lambda c: (-c.score, c.path, c.kind))
        return scored

    def _cache_key(self, repo_root: Path, cluster: ObservationCluster) -> str:
        refs = "|".join(f"{obs.rule_id}:{obs.file_path}:{obs.line}" for obs in cluster.observations)
        return hashlib.sha256(f"{repo_root}:{refs}:{self._budget}".encode()).hexdigest()


def _window(source: str, line: int, *, radius: int) -> str:
    lines = source.splitlines()
    start = max(0, line - 1 - radius)
    end = min(len(lines), line + radius)
    return "\n".join(lines[start:end])[:4_000]


def _rel(path: str, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return path


def _enclosing_entity(graph: SyntaxGraph, line: int) -> ParsedEntity | None:
    enclosing = None
    for ent in graph.entities:
        if ent.start_line <= line <= (ent.end_line or ent.start_line):
            enclosing = ent
    return enclosing


def _entity_window(graph: SyntaxGraph, entity: object) -> str:
    start = max(1, int(getattr(entity, "start_line", 1)))
    end = min(len(graph.lines), int(getattr(entity, "end_line", start)))
    return "\n".join(graph.lines[start - 1 : end])[:4_000]
