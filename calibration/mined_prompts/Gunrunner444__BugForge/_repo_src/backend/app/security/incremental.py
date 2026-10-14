"""In-memory incremental parse cache.

Unchanged files keep their syntax graph. The semantic graph and security
rules still run on the current set of graphs, so the result matches a clean
scan. The cache is not written to disk. Secret files and credential settings
are not cache keys.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.adapters.languages.base import LanguageAdapter
from app.core.config import settings
from app.core.paths import to_relative_path
from app.domain.language import LanguageCapability
from app.parsing.model import SyntaxGraph
from app.plugins import get_plugin_catalog
from app.security.engine import AnalysisDiagnostic, SecurityAnalysisEngine, SecurityScanResult

ANALYSIS_CACHE_GENERATION = "phase17-v1"
_SECRET_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "id_rsa",
        "id_dsa",
        "credentials.json",
        "secrets.json",
    }
)


@dataclass(frozen=True)
class CachedGraph:
    path: str
    content_hash: str
    parser_id: str
    config_id: str
    graph: SyntaxGraph


@dataclass
class AnalysisIndex:
    """Process-local parse index. Keys are paths, hashes, parser id, and config id."""

    records: dict[str, CachedGraph] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    skipped_sensitive: int = 0


def config_identity() -> str:
    """Hash of analysis limits. Secrets and tokens are not included."""
    payload = {
        "generation": ANALYSIS_CACHE_GENERATION,
        "max_file_size_bytes": settings.max_file_size_bytes,
        "taint_max_files": settings.taint_max_files,
        "taint_max_import_depth": settings.taint_max_import_depth,
        "taint_max_cross_file_rounds": settings.taint_max_cross_file_rounds,
        "taint_max_cross_file_edges": settings.taint_max_cross_file_edges,
        "taint_cross_file_budget_ms": settings.taint_cross_file_budget_ms,
        "taint_max_field_depth": settings.taint_max_field_depth,
        "taint_max_field_bindings": settings.taint_max_field_bindings,
        "taint_max_alias_edges": settings.taint_max_alias_edges,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_sensitive_path(path: str) -> bool:
    name = Path(path.replace("\\", "/")).name.lower()
    if name in _SECRET_NAMES or name.startswith(".env"):
        return True
    return name.endswith((".pem", ".key", ".p12", ".pfx"))


def analyze_incremental(
    engine: SecurityAnalysisEngine,
    repo_path: Path,
    file_paths: list[Path],
    index: AnalysisIndex | None = None,
) -> tuple[SecurityScanResult, AnalysisIndex]:
    """Scan ``file_paths``, reusing parsed graphs whose keys still match."""
    index = index or AnalysisIndex()
    index.hits = 0
    index.misses = 0
    index.skipped_sensitive = 0
    languages = get_plugin_catalog().languages
    frameworks = engine._frameworks.detect(repo_path, file_paths)
    graphs: dict[str, SyntaxGraph] = {}
    used: set[str] = set()
    analyzed = 0
    diagnostics: list[AnalysisDiagnostic] = []
    config_id = config_identity()
    for file_path in sorted(file_paths, key=lambda path: path.as_posix()):
        try:
            relative = to_relative_path(file_path, repo_path)
        except ValueError:
            relative = file_path.as_posix()
        sensitive = is_sensitive_path(relative)
        if sensitive:
            index.skipped_sensitive += 1
            index.records.pop(relative, None)
        adapter = languages.for_path(file_path)
        if adapter is None or not adapter.supports(LanguageCapability.SECURITY_ANALYSIS):
            continue
        if sensitive:
            graph, parse_diag = engine._parse(adapter, file_path)
        else:
            graph, parse_diag = _cached_parse(
                engine, adapter, file_path, relative, config_id, index
            )
        if parse_diag is not None:
            diagnostics.append(parse_diag)
        if graph is None:
            continue
        graph.file_path = relative
        analyzed += 1
        used.add(adapter.language_id)
        graphs[relative] = graph
        if graph.diagnostics.truncated or graph.diagnostics.has_errors:
            diagnostics.append(
                AnalysisDiagnostic(
                    kind="partial_analysis" if graph.diagnostics.truncated else "parser_error",
                    message=graph.diagnostics.message or "parse reported errors",
                    file_path=relative,
                    language=graph.language,
                    parser_backend=graph.parser_backend,
                    parser_tier=str(graph.parser_tier),
                )
            )
    result = engine._rules_over_graphs(repo_path, graphs, diagnostics, frameworks, used, analyzed)
    return result, index


def _cached_parse(
    engine: SecurityAnalysisEngine,
    adapter: LanguageAdapter,
    file_path: Path,
    relative: str,
    config_id: str,
    index: AnalysisIndex,
) -> tuple[SyntaxGraph | None, AnalysisDiagnostic | None]:
    parser_id = _parser_id(adapter)
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        index.misses += 1
        return engine._parse(adapter, file_path)
    digest = content_hash(text)
    cached = index.records.get(relative)
    if (
        cached is not None
        and cached.content_hash == digest
        and cached.parser_id == parser_id
        and cached.config_id == config_id
    ):
        index.hits += 1
        return cached.graph, None
    index.misses += 1
    graph, diag = engine._parse(adapter, file_path)
    if graph is not None:
        actual = (
            f"{graph.language}:{graph.parser_backend}:{graph.parser_tier}:"
            f"{ANALYSIS_CACHE_GENERATION}"
        )
        # Store the parser that actually ran. A later lookup uses the adapter
        # claim, so a mismatch is never reused as if it were that claim.
        index.records[relative] = CachedGraph(
            path=relative,
            content_hash=digest,
            parser_id=actual,
            config_id=config_id,
            graph=graph,
        )
    return graph, diag


def _parser_id(adapter: LanguageAdapter) -> str:
    return (
        f"{adapter.language_id}:{adapter.parser_backend()}:{adapter.parser_tier()}:"
        f"{ANALYSIS_CACHE_GENERATION}"
    )
