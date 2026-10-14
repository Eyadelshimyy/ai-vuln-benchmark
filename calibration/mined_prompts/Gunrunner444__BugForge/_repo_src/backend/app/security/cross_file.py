"""Bounded cross-file taint for uniquely resolved local imports.

Python, JavaScript/TypeScript, and a few other full-syntax languages
participate. Profile fallback is not dataflow. Ambiguous or unresolved names
produce no edge. A partial or truncated callee is not a summary. Limits stop
the walk and leave an explicit diagnostic; a stopped walk is not a clean result.

A limit of zero disables that step. It never indexes an empty list and never
performs one accidental propagation hop.

High-confidence extensions, still unique-identity only:

    * Python and JS/TS re-exports, including package ``__init__.py`` and
      ``export { name } from ...``
    * local default imports whose default export is a unique function or class
    * methods whose class is imported and constructed, or called as ``Class.method``
    * module-level callable aliases that are assigned once
    * TypeScript ``paths`` mappings read from ``tsconfig.json`` as data
    * Go ``import "path"`` when one local ``path.go`` matches, including an
      explicit package alias recorded on the import spec
    * Java and Kotlin type imports when the dotted path matches one local
      file and that file has one class or function of the imported name
    * no inheritance guessing and no dynamic dispatch
    * Ruby, C#, and Swift imports are not resolved to source files
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from app.core.config import settings
from app.domain.language import ParserTier
from app.domain.security import VulnerabilityClass
from app.domain.source import ParsedEntity, ParsedImport
from app.parsing.model import Binding, CallSite, SymbolKind, SyntaxGraph
from app.security.definitions import SINK_SANITIZER_KINDS, SanitizerDefinition, SinkDefinition
from app.security.language_vocab import vocab_for
from app.security.taint import (
    AliasMirrors,
    ExternalCallee,
    _callee_is_source,
    _empty_rhs,
    _external_for_call,
    _killing_assignment,
    _ordered_bindings,
    _peel_sanitizer,
    _preserves_taint,
    _scope_for_entity,
    _use_byte,
    call_matches_sink,
    calls_contained_in,
    field_flow_allowed,
    field_limit_facts,
    field_root,
    fragment_matches_source,
    is_field_path,
    proven_alias_mirrors,
    receiver_argument_offset,
    sanitizer_intervened,
)

_CROSS_FILE_LANGUAGES = frozenset({"python", "javascript", "typescript", "go", "java", "kotlin"})
_LOCAL_MODULE_SUFFIX = {"go": ".go", "java": ".java", "kotlin": ".kt"}
_SKIPPED_IMPORT_KINDS = frozenset({"go_unresolved", "java_static"})
_FUNCTION_TYPES = frozenset({"function", "async_function"})
_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CrossFileLimits:
    max_files: int = 400
    max_import_depth: int = 3
    max_rounds: int = 4
    max_edges: int = 2_000
    budget_ms: int = 1_500

    @staticmethod
    def from_settings() -> CrossFileLimits:
        return CrossFileLimits(
            max_files=settings.taint_max_files,
            max_import_depth=settings.taint_max_import_depth,
            max_rounds=settings.taint_max_cross_file_rounds,
            max_edges=settings.taint_max_cross_file_edges,
            budget_ms=settings.taint_cross_file_budget_ms,
        )


@dataclass(frozen=True)
class FlowDiagnostic:
    kind: str
    message: str
    file_path: str


@dataclass(frozen=True)
class FlowEdge:
    """Stable repository edge.

    Identifiers are ``file::module::class::symbol``. They never contain
    addresses or process-specific values.
    """

    kind: str
    source_id: str
    target_id: str


@dataclass(frozen=True)
class _Summary:
    file_path: str
    function_name: str
    scope_id: str
    parameters: tuple[str, ...]
    return_effects: tuple[str, ...]
    param_sinks: tuple[tuple[int, str, str, int], ...]
    hops: int = 0
    partial: bool = False
    decorators: tuple[str, ...] = ()


@dataclass
class ProjectFlow:
    """Import-resolved callees for one repository scan."""

    externals: dict[str, dict[str, ExternalCallee]] = field(default_factory=dict)
    diagnostics: list[FlowDiagnostic] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)
    incomplete: bool = False

    def externals_for(self, file_path: str) -> dict[str, ExternalCallee]:
        return self.externals.get(file_path, {})

    def lookup(self, file_path: str, call: CallSite) -> ExternalCallee | None:
        return _external_for_call(call, self.externals_for(file_path))


def build_project(
    graphs: dict[str, SyntaxGraph],
    *,
    limits: CrossFileLimits | None = None,
    repo_root: Path | None = None,
) -> ProjectFlow:
    limits = _clamp_limits(limits or CrossFileLimits.from_settings())
    project = ProjectFlow()
    started = time.monotonic()

    def expired() -> bool:
        if limits.budget_ms <= 0:
            return True
        return (time.monotonic() - started) * 1000 >= limits.budget_ms

    if expired():
        return _stop(
            project,
            "cross-file taint budget exhausted before propagation",
        )
    if limits.max_files <= 0:
        return _stop(project, "cross-file taint stopped after 0 files")
    if limits.max_edges <= 0:
        return _stop(project, "cross-file taint stopped at configured edge limit")
    if limits.max_import_depth <= 0:
        return _stop(
            project,
            "cross-file taint stopped at max import depth",
            kind="cross_file_depth_limited",
        )
    if limits.max_rounds <= 0:
        return _stop(project, "cross-file taint stopped before any propagation round")

    eligible = {path: graph for path, graph in graphs.items() if _eligible(graph)}
    ordered = sorted(eligible)
    if limits.max_files < len(ordered):
        project.incomplete = True
        project.diagnostics.append(
            FlowDiagnostic(
                kind="cross_file_incomplete",
                message=f"cross-file taint stopped after {limits.max_files} files",
                file_path=ordered[limits.max_files],
            )
        )
        ordered = ordered[: limits.max_files]
        eligible = {path: eligible[path] for path in ordered}

    targets, classes, link_kinds, target_diag = _import_targets(
        eligible,
        max_reexport_depth=limits.max_import_depth,
        ts_mappings=_ts_path_mappings(repo_root),
    )
    project.diagnostics.extend(target_diag)
    summaries: dict[tuple[str, str], _Summary] = {}
    charged: set[tuple[str, str, str, str]] = set()
    depth_limited = False
    changed = False

    for _round in range(limits.max_rounds):
        if expired():
            project.incomplete = True
            project.diagnostics.append(
                FlowDiagnostic(
                    kind="cross_file_incomplete",
                    message="cross-file taint budget exhausted",
                    file_path="",
                )
            )
            break
        changed = False
        for path in ordered:
            graph = eligible[path]
            partial = _graph_partial(graph)
            for entity in _summarizable(graph):
                if expired():
                    project.incomplete = True
                    project.diagnostics.append(
                        FlowDiagnostic(
                            kind="cross_file_incomplete",
                            message="cross-file taint budget exhausted",
                            file_path=path,
                        )
                    )
                    break
                scope_id = _scope_for_entity(graph, entity)
                if scope_id is None:
                    continue
                summary_name = entity.qualified_name or entity.name
                summary, _used_edges, limited, capped = _summarize(
                    graph,
                    summary_name,
                    scope_id,
                    tuple(p.name for p in entity.parameters),
                    summaries,
                    targets,
                    partial=partial,
                    max_depth=limits.max_import_depth,
                    decorators=tuple(entity.decorators),
                    edge_budget=limits.max_edges,
                    charged=charged,
                )
                if limited:
                    depth_limited = True
                if capped:
                    project.incomplete = True
                    project.diagnostics.append(
                        FlowDiagnostic(
                            kind="cross_file_incomplete",
                            message="cross-file taint stopped at configured edge limit",
                            file_path=path,
                        )
                    )
                key = (path, summary_name)
                if summaries.get(key) != summary:
                    summaries[key] = summary
                    changed = True
                if project.incomplete:
                    break
            if project.incomplete:
                break
        if project.incomplete or not changed:
            break
    else:
        if changed:
            project.incomplete = True
            project.diagnostics.append(
                FlowDiagnostic(
                    kind="cross_file_incomplete",
                    message="cross-file taint stopped at max propagation rounds",
                    file_path="",
                )
            )

    if depth_limited:
        project.incomplete = True
        project.diagnostics.append(
            FlowDiagnostic(
                kind="cross_file_depth_limited",
                message="cross-file taint stopped at max import depth",
                file_path="",
            )
        )
    project.externals, dropped, export_diag = _externals_from(
        eligible,
        summaries,
        targets,
        classes,
        link_kinds,
        max_depth=limits.max_import_depth,
    )
    project.diagnostics.extend(export_diag)
    if dropped and not any(item.kind == "cross_file_depth_limited" for item in project.diagnostics):
        project.incomplete = True
        project.diagnostics.append(
            FlowDiagnostic(
                kind="cross_file_depth_limited",
                message="cross-file taint stopped at max import depth",
                file_path="",
            )
        )
    budgeted = _flow_edges(targets, classes, link_kinds, project.externals)
    owners = _method_owner_edges(eligible)
    kept, truncated = _take_edges(budgeted, limits.max_edges)
    project.edges = _dedupe_edges([*owners, *kept])
    if truncated:
        project.incomplete = True
        project.externals = _externals_for_kept_calls(project.externals, kept)
        project.diagnostics.append(
            FlowDiagnostic(
                kind="cross_file_incomplete",
                message="cross-file taint stopped at configured edge limit",
                file_path="",
            )
        )
    if any(
        item.kind in {"cross_file_partial", "cross_file_depth_limited", "cross_file_incomplete"}
        for item in project.diagnostics
    ):
        project.incomplete = True
    project.diagnostics = _stable_diagnostics(project.diagnostics)
    return project


def _clamp_limits(limits: CrossFileLimits) -> CrossFileLimits:
    """Negative limits disable the corresponding step. Zero stays zero."""
    return CrossFileLimits(
        max_files=max(0, limits.max_files),
        max_import_depth=max(0, limits.max_import_depth),
        max_rounds=max(0, limits.max_rounds),
        max_edges=max(0, limits.max_edges),
        budget_ms=max(0, limits.budget_ms),
    )


def _stop(
    project: ProjectFlow,
    message: str,
    *,
    kind: str = "cross_file_incomplete",
) -> ProjectFlow:
    project.incomplete = True
    project.externals = {}
    project.edges = []
    project.diagnostics.append(FlowDiagnostic(kind=kind, message=message, file_path=""))
    project.diagnostics = _stable_diagnostics(project.diagnostics)
    return project


def _eligible(graph: SyntaxGraph) -> bool:
    if graph.language not in _CROSS_FILE_LANGUAGES:
        return False
    return graph.parser_tier is ParserTier.FULL_AST


def _module_functions(graph: SyntaxGraph) -> list[ParsedEntity]:
    found: list[ParsedEntity] = []
    for entity in graph.entities:
        if entity.parent:
            continue
        if entity.entity_type not in _FUNCTION_TYPES:
            continue
        found.append(entity)
    return found


def _unique_function_names(graph: SyntaxGraph) -> dict[str, ParsedEntity]:
    counts: dict[str, list[ParsedEntity]] = {}
    for entity in _module_functions(graph):
        counts.setdefault(entity.name, []).append(entity)
    return {name: ents[0] for name, ents in counts.items() if len(ents) == 1}


def _graph_partial(graph: SyntaxGraph) -> bool:
    return bool(graph.diagnostics.truncated or graph.diagnostics.has_errors)


def _summarizable(graph: SyntaxGraph) -> list[ParsedEntity]:
    found: list[ParsedEntity] = []
    seen: set[str] = set()
    for entity in graph.entities:
        if entity.entity_type not in _FUNCTION_TYPES | {"method", "async_method"}:
            continue
        if entity.entity_type in _FUNCTION_TYPES and entity.parent:
            continue
        key = entity.qualified_name or entity.name
        if key in seen:
            continue
        seen.add(key)
        found.append(entity)
    return found


def _unique_classes(graph: SyntaxGraph) -> dict[str, ParsedEntity]:
    counts: dict[str, list[ParsedEntity]] = {}
    for entity in graph.entities:
        if entity.entity_type != "class" or entity.parent:
            continue
        counts.setdefault(entity.name, []).append(entity)
    return {name: ents[0] for name, ents in counts.items() if len(ents) == 1}


def _methods_by_class(graph: SyntaxGraph) -> dict[str, dict[str, ParsedEntity]]:
    grouped: dict[str, dict[str, list[ParsedEntity]]] = {}
    for entity in graph.entities:
        if entity.entity_type not in {"method", "async_method"} or not entity.parent:
            continue
        grouped.setdefault(entity.parent, {}).setdefault(entity.name, []).append(entity)
    unique: dict[str, dict[str, ParsedEntity]] = {}
    for class_name, methods in grouped.items():
        unique[class_name] = {name: ents[0] for name, ents in methods.items() if len(ents) == 1}
    return unique


def _module_name(file_path: str) -> str:
    path = PurePosixPath(file_path.replace("\\", "/"))
    if path.name == "__init__.py":
        parent = path.parent.as_posix()
        return "" if parent == "." else parent.replace("/", ".")
    stem = path.with_suffix("").as_posix()
    return "" if stem == "." else stem.replace("/", ".")


def _symbol_id(file_path: str, name: str, *, owner: str = "") -> str:
    """``file::module::class::symbol``. Empty slots stay empty; nothing is hashed."""
    file_norm = file_path.replace("\\", "/")
    return f"{file_norm}::{_module_name(file_norm)}::{owner}::{name}"


def _summary_symbol_id(file_path: str, function_name: str) -> str:
    owner = ""
    symbol = function_name
    if function_name.count(".") == 1 and "/" not in function_name:
        owner, symbol = function_name.split(".", 1)
    return _symbol_id(file_path, symbol, owner=owner)


def _stable_diagnostics(items: list[FlowDiagnostic]) -> list[FlowDiagnostic]:
    seen: set[tuple[str, str, str]] = set()
    ordered: list[FlowDiagnostic] = []
    for item in sorted(items, key=lambda diag: (diag.file_path, diag.kind, diag.message)):
        key = (item.file_path, item.kind, item.message)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _flow_edges(
    targets: dict[tuple[str, str], tuple[str, str] | str],
    classes: dict[tuple[str, str], tuple[str, str]],
    link_kinds: dict[tuple[str, str], str],
    externals: dict[str, dict[str, ExternalCallee]],
) -> list[FlowEdge]:
    edges: list[FlowEdge] = []
    for (importer, local), target in sorted(targets.items()):
        if isinstance(target, tuple):
            kind = link_kinds.get((importer, local), "import")
            edges.append(
                FlowEdge(
                    kind=kind,
                    source_id=_symbol_id(importer, local),
                    target_id=_summary_symbol_id(target[0], target[1]),
                )
            )
    for (importer, local), (file_path, class_name) in sorted(classes.items()):
        edges.append(
            FlowEdge(
                kind="class_import",
                source_id=_symbol_id(importer, local),
                target_id=_symbol_id(file_path, "", owner=class_name),
            )
        )
    for file_path, table in sorted(externals.items()):
        for local, ext in sorted(table.items()):
            edges.append(
                FlowEdge(
                    kind="call",
                    source_id=_symbol_id(file_path, local),
                    target_id=ext.semantic_id
                    or _summary_symbol_id(ext.callee_file, ext.callee_function),
                )
            )
    return _dedupe_edges(edges)


def _method_owner_edges(graphs: dict[str, SyntaxGraph]) -> list[FlowEdge]:
    edges: list[FlowEdge] = []
    for path, graph in sorted(graphs.items()):
        for class_name, methods in sorted(_methods_by_class(graph).items()):
            class_id = _symbol_id(path, "", owner=class_name)
            for method_name in sorted(methods):
                edges.append(
                    FlowEdge(
                        kind="method_owner",
                        source_id=_symbol_id(path, method_name, owner=class_name),
                        target_id=class_id,
                    )
                )
    return edges


_EDGE_PRIORITY = {
    "call": 0,
    "alias": 1,
    "default": 2,
    "import": 3,
    "re_export": 4,
    "class_import": 5,
}


def _dedupe_edges(edges: list[FlowEdge]) -> list[FlowEdge]:
    seen: set[tuple[str, str, str]] = set()
    ordered: list[FlowEdge] = []
    for edge in edges:
        key = (edge.kind, edge.source_id, edge.target_id)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(edge)
    return ordered


def _take_edges(edges: list[FlowEdge], max_edges: int) -> tuple[list[FlowEdge], bool]:
    """Keep at most ``max_edges`` edges, checking the limit before each append."""
    ordered = sorted(
        edges,
        key=lambda edge: (
            _EDGE_PRIORITY.get(edge.kind, 9),
            edge.source_id,
            edge.target_id,
            edge.kind,
        ),
    )
    kept: list[FlowEdge] = []
    truncated = False
    for edge in ordered:
        if len(kept) >= max_edges:
            truncated = True
            break
        kept.append(edge)
    return kept, truncated


def _externals_for_kept_calls(
    externals: dict[str, dict[str, ExternalCallee]],
    kept: list[FlowEdge],
) -> dict[str, dict[str, ExternalCallee]]:
    calls = {(edge.source_id, edge.target_id) for edge in kept if edge.kind == "call"}
    filtered: dict[str, dict[str, ExternalCallee]] = {}
    for path, table in sorted(externals.items()):
        surviving: dict[str, ExternalCallee] = {}
        for local, ext in table.items():
            source = _symbol_id(path, local)
            target = ext.semantic_id or _summary_symbol_id(ext.callee_file, ext.callee_function)
            if (source, target) in calls:
                surviving[local] = ext
        if surviving:
            filtered[path] = surviving
    return filtered


def _import_targets(
    graphs: dict[str, SyntaxGraph],
    *,
    max_reexport_depth: int,
    ts_mappings: tuple[tuple[str, str], ...] = (),
) -> tuple[
    dict[tuple[str, str], tuple[str, str] | str],
    dict[tuple[str, str], tuple[str, str]],
    dict[tuple[str, str], str],
    list[FlowDiagnostic],
]:
    """Map ``(importer, local call name)`` to ``(file, function)`` or ambiguous.

    ``classes`` maps ``(importer, local class name)`` to ``(file, class)``.
    ``link_kinds`` records ``import``, ``re_export``, ``alias``, or ``default``.
    Re-exports and aliases are followed only while the next hop is unique.
    """
    targets: dict[tuple[str, str], tuple[str, str] | str] = {}
    classes: dict[tuple[str, str], tuple[str, str]] = {}
    link_kinds: dict[tuple[str, str], str] = {}
    diagnostics: list[FlowDiagnostic] = []
    pending: list[tuple[str, str, str, str, str]] = []

    def bind(
        importer: str,
        local: str,
        value: tuple[str, str] | str,
        line_file: str,
        *,
        kind: str = "import",
    ) -> None:
        if not local:
            return
        key = (importer, local)
        current = targets.get(key)
        if current is None:
            targets[key] = value
            if isinstance(value, tuple):
                link_kinds[key] = kind
            return
        if current != value:
            targets[key] = _AMBIGUOUS
            link_kinds.pop(key, None)
            diagnostics.append(
                FlowDiagnostic(
                    kind="ambiguous_import",
                    message=f"local name {local} resolves to more than one function",
                    file_path=line_file,
                )
            )

    def bind_class(importer: str, local: str, value: tuple[str, str], line_file: str) -> None:
        key = (importer, local)
        current = classes.get(key)
        if current is None:
            classes[key] = value
            return
        if current != value:
            classes.pop(key, None)
            diagnostics.append(
                FlowDiagnostic(
                    kind="ambiguous_import",
                    message=f"local class {local} resolves to more than one type",
                    file_path=line_file,
                )
            )

    for path, graph in sorted(graphs.items()):
        names_here = _unique_function_names(graph)
        classes_here = _unique_classes(graph)
        for imp in graph.imports:
            if imp.syntax_kind == "export_local":
                source_name = imp.name or ""
                public = imp.alias or source_name
                if source_name and public:
                    if source_name in names_here:
                        bind(path, public, (path, source_name), path, kind="re_export")
                    elif source_name in classes_here:
                        bind_class(path, public, (path, source_name), path)
                    else:
                        pending.append((path, public, path, source_name, "re_export"))
                continue
            if imp.name == "*" or imp.syntax_kind == "export_star":
                diagnostics.append(
                    FlowDiagnostic(
                        kind="ambiguous_import",
                        message="star import is not a dataflow edge",
                        file_path=path,
                    )
                )
                continue
            if _is_dynamic_or_side_effect_import(graph, imp):
                continue
            if imp.syntax_kind in _SKIPPED_IMPORT_KINDS:
                continue
            resolved = _resolve_module(graph, imp, graphs, ts_mappings)
            if resolved is None:
                if (
                    imp.import_type == "relative"
                    or imp.relative_level > 0
                    or _looks_relative(imp.module)
                    or _matches_ts_pattern(imp.module, ts_mappings)
                ):
                    diagnostics.append(
                        FlowDiagnostic(
                            kind="unresolved_import",
                            message=f"relative import {imp.module or imp.name or ''} was not resolved",
                            file_path=path,
                        )
                    )
                continue
            if resolved == _AMBIGUOUS:
                diagnostics.append(
                    FlowDiagnostic(
                        kind="ambiguous_import",
                        message=f"import {imp.module or imp.name or ''} matches more than one file",
                        file_path=path,
                    )
                )
                local = _local_name(imp)
                if local:
                    bind(path, local, _AMBIGUOUS, path)
                continue
            target = graphs.get(resolved)
            if target is None:
                continue
            if _graph_partial(target):
                diagnostics.append(
                    FlowDiagnostic(
                        kind="cross_file_partial",
                        message="partial callee graph is not a cross-file summary",
                        file_path=resolved,
                    )
                )
                continue
            names = _unique_function_names(target)
            class_names = _unique_classes(target)
            relationship = "re_export" if imp.syntax_kind.startswith("export_") else "import"
            if graph.language in {"java", "kotlin"} and imp.syntax_kind == "import":
                _bind_type_import(path, imp, resolved, names, class_names, bind, bind_class)
                continue
            if _is_namespace_import(imp) or _is_python_module_import(graph, imp):
                _bind_module_surface(
                    path, imp, resolved, names, class_names, bind, bind_class, relationship
                )
                continue
            if _is_module_alias_import(imp, names):
                _bind_module_surface(
                    path, imp, resolved, names, class_names, bind, bind_class, relationship
                )
                continue
            if imp.syntax_kind == "import_default":
                _bind_default_import(
                    path,
                    imp,
                    resolved,
                    target,
                    names,
                    class_names,
                    bind,
                    bind_class,
                    pending,
                    diagnostics,
                )
                continue
            imported = imp.name
            if not imported:
                continue
            local = imp.alias or imported
            if imported in names:
                bind(path, local, (resolved, imported), path, kind=relationship)
                continue
            if imported in class_names:
                bind_class(path, local, (resolved, imported), path)
                continue
            pending.append((path, local, resolved, imported, relationship))

    for _depth in range(max(0, max_reexport_depth)):
        progressed = _follow_pending(pending, targets, classes, bind, bind_class)
        pending = progressed[1]
        alias_progress = _bind_callable_aliases(graphs, targets, classes, bind, bind_class)
        if not progressed[0] and not alias_progress:
            break
    else:
        if pending:
            diagnostics.append(
                FlowDiagnostic(
                    kind="cross_file_depth_limited",
                    message="cross-file taint stopped at max import depth",
                    file_path=pending[0][0],
                )
            )
    return targets, classes, link_kinds, diagnostics


def _is_dynamic_or_side_effect_import(graph: SyntaxGraph, imp: ParsedImport) -> bool:
    """``import()`` and side-effect ``import "./x"`` are not namespace bindings."""
    if graph.language not in {"javascript", "typescript"}:
        return False
    return imp.syntax_kind == "import" and not imp.name and not imp.alias


def _is_python_module_import(graph: SyntaxGraph, imp: ParsedImport) -> bool:
    return graph.language == "python" and imp.syntax_kind == "import" and not imp.is_from_import


def _bind_module_surface(
    path: str,
    imp: ParsedImport,
    resolved: str,
    names: dict[str, ParsedEntity],
    class_names: dict[str, ParsedEntity],
    bind: Callable[..., None],
    bind_class: Callable[..., None],
    relationship: str,
) -> None:
    alias = imp.alias or _module_alias(imp.module, imp.name) or imp.name or ""
    if not alias:
        return
    for func_name in sorted(names):
        bind(path, f"{alias}.{func_name}", (resolved, func_name), path, kind=relationship)
    for class_name in sorted(class_names):
        bind_class(path, f"{alias}.{class_name}", (resolved, class_name), path)


def _bind_default_import(
    path: str,
    imp: ParsedImport,
    resolved: str,
    target: SyntaxGraph,
    names: dict[str, ParsedEntity],
    class_names: dict[str, ParsedEntity],
    bind: Callable[..., None],
    bind_class: Callable[..., None],
    pending: list[tuple[str, str, str, str, str]],
    diagnostics: list[FlowDiagnostic],
) -> None:
    local = imp.alias or imp.name or ""
    exported = target.default_export
    if exported and not _default_export_stable(target, exported):
        diagnostics.append(
            FlowDiagnostic(
                kind="unresolved_export",
                message="default export is reassigned or conditional",
                file_path=path,
            )
        )
        return
    if not local or not exported:
        diagnostics.append(
            FlowDiagnostic(
                kind="unresolved_export",
                message="default import has no unique default export",
                file_path=path,
            )
        )
        return
    if exported in names:
        bind(path, local, (resolved, exported), path, kind="default")
        return
    if exported in class_names:
        bind_class(path, local, (resolved, exported), path)
        return
    pending.append((path, local, resolved, exported, "default"))


def _default_export_stable(graph: SyntaxGraph, name: str) -> bool:
    """A default export stays bound only when that name is not reassigned."""
    for binding in graph.bindings:
        if binding.name != name or binding.kind is SymbolKind.PARAMETER:
            continue
        if binding.is_conditional or not binding.is_declaration:
            return False
    return True


def _follow_pending(
    pending: list[tuple[str, str, str, str, str]],
    targets: dict[tuple[str, str], tuple[str, str] | str],
    classes: dict[tuple[str, str], tuple[str, str]],
    bind: Callable[..., None],
    bind_class: Callable[..., None],
) -> tuple[bool, list[tuple[str, str, str, str, str]]]:
    if not pending:
        return False, pending
    progressed = False
    still: list[tuple[str, str, str, str, str]] = []
    for importer, local, src_file, src_name, kind in pending:
        dest = targets.get((src_file, src_name))
        if isinstance(dest, tuple):
            followed = "re_export" if dest[0] != src_file or kind == "re_export" else kind
            bind(importer, local, dest, importer, kind=followed)
            progressed = True
            continue
        class_dest = classes.get((src_file, src_name))
        if class_dest is not None:
            bind_class(importer, local, class_dest, importer)
            progressed = True
            continue
        if dest == _AMBIGUOUS:
            bind(importer, local, _AMBIGUOUS, importer)
            continue
        still.append((importer, local, src_file, src_name, kind))
    return progressed, still


def _bind_callable_aliases(
    graphs: dict[str, SyntaxGraph],
    targets: dict[tuple[str, str], tuple[str, str] | str],
    classes: dict[tuple[str, str], tuple[str, str]],
    bind: Callable[..., None],
    bind_class: Callable[..., None],
) -> bool:
    """One hop of ``alias = known_callable`` when the name is assigned exactly once."""
    snapshot_targets = dict(targets)
    snapshot_classes = dict(classes)
    progressed = False
    for path, graph in sorted(graphs.items()):
        grouped: dict[str, list[Binding]] = {}
        for binding in graph.bindings:
            if binding.kind is SymbolKind.PARAMETER:
                continue
            grouped.setdefault(binding.name, []).append(binding)
        names = _unique_function_names(graph)
        class_names = _unique_classes(graph)
        for name, binds in sorted(grouped.items()):
            if len(binds) != 1 or (path, name) in targets or (path, name) in classes:
                continue
            binding = binds[0]
            if binding.is_conditional:
                continue
            if binding.rhs_is_literal or binding.rhs_callees or binding.rhs_accesses:
                continue
            if len(binding.rhs_idents) != 1:
                continue
            if any(token in binding.rhs for token in (".", "(", "[", "{")):
                continue
            ident = binding.rhs_idents[0]
            if not ident or ident == name:
                continue
            calls = [call for call in graph.calls if call.name == name or call.qualified == name]
            if any(call.scope_id != binding.scope_id for call in calls):
                continue
            dest = snapshot_targets.get((path, ident))
            if isinstance(dest, tuple):
                bind(path, name, dest, path, kind="alias")
                progressed = True
                continue
            if dest == _AMBIGUOUS:
                continue
            class_dest = snapshot_classes.get((path, ident))
            if class_dest is not None:
                bind_class(path, name, class_dest, path)
                progressed = True
                continue
            if ident in names:
                bind(path, name, (path, ident), path, kind="alias")
                progressed = True
                continue
            if ident in class_names:
                bind_class(path, name, (path, ident), path)
                progressed = True
    return progressed


def _is_module_alias_import(imp: ParsedImport, function_names: dict[str, ParsedEntity]) -> bool:
    """``from . import helpers`` binds the module, not a function of that name."""
    if not imp.is_from_import or imp.module:
        return False
    imported = imp.name or ""
    return bool(imported) and imported not in function_names


def _bind_type_import(
    path: str,
    imp: ParsedImport,
    resolved: str,
    names: dict[str, ParsedEntity],
    class_names: dict[str, ParsedEntity],
    bind: Callable[..., None],
    bind_class: Callable[..., None],
) -> None:
    """Bind ``import pkg.Name`` to that class or function, not to every name."""
    simple = (imp.module or "").rsplit(".", 1)[-1].rsplit("/", 1)[-1]
    local = imp.alias or simple
    if not simple.isidentifier() or not local.isidentifier():
        return
    if simple in class_names and simple in names:
        bind(path, local, _AMBIGUOUS, path)
        return
    if simple in class_names:
        bind_class(path, local, (resolved, simple), path)
        return
    if simple in names:
        bind(path, local, (resolved, simple), path, kind="import")


def _is_namespace_import(imp: ParsedImport) -> bool:
    if imp.syntax_kind == "import_namespace":
        return True
    if imp.syntax_kind == "import":
        return True
    return not imp.is_from_import and not imp.name


def _local_name(imp: ParsedImport) -> str:
    if imp.alias:
        return imp.alias
    if imp.name:
        return imp.name
    return _module_alias(imp.module, None)


def _module_alias(module: str, name: str | None) -> str:
    text = name or module
    text = text.replace("\\", "/").rstrip("/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if text.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".py")):
        text = text.rsplit(".", 1)[0]
    if "." in text and "/" not in module:
        return text.rsplit(".", 1)[-1]
    return text


def _looks_relative(module: str) -> bool:
    return module.startswith("./") or module.startswith("../") or module.startswith(".")


def _resolve_module(
    graph: SyntaxGraph,
    imp: ParsedImport,
    graphs: dict[str, SyntaxGraph],
    ts_mappings: tuple[tuple[str, str], ...] = (),
) -> str | None:
    if graph.language == "python":
        return _resolve_python(graph.file_path, imp, graphs)
    if graph.language in {"javascript", "typescript"}:
        return _resolve_javascript(graph.file_path, imp.module, graphs, ts_mappings)
    if graph.language in _LOCAL_MODULE_SUFFIX:
        return _resolve_local_module(graph.language, imp.module, graphs)
    return None


def _resolve_python(importer: str, imp: ParsedImport, graphs: dict[str, SyntaxGraph]) -> str | None:
    level = imp.relative_level
    module = imp.module
    if level <= 0 and imp.import_type == "relative":
        level = 1
    if not module and imp.is_from_import and imp.name and level > 0:
        module = imp.name
        # `from . import helpers` binds the submodule, not a function.
    if level > 0:
        return _python_relative_file(importer, module, level, graphs)
    if not module:
        return None
    return _unique_suffix(module, graphs, "python")


def _python_relative_file(
    importer: str, module: str, level: int, graphs: dict[str, SyntaxGraph]
) -> str | None:
    directory = PurePosixPath(importer).parent
    for _ in range(level - 1):
        if directory == PurePosixPath("."):
            return None
        directory = directory.parent
    parts = [part for part in module.split(".") if part]
    base = directory.joinpath(*parts) if parts else directory
    norm = PurePosixPath(os.path.normpath(base.as_posix()))
    if ".." in norm.parts:
        return None
    candidates = []
    for suffix in (".py", "/__init__.py"):
        cand = f"{norm.as_posix()}{suffix}"
        if cand in graphs and graphs[cand].language == "python":
            candidates.append(cand)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return _AMBIGUOUS
    return None


def _resolve_local_module(language: str, module: str, graphs: dict[str, SyntaxGraph]) -> str | None:
    """Match a dotted or slash path to one local file. Do not guess packages."""
    suffix = _LOCAL_MODULE_SUFFIX.get(language)
    if suffix is None:
        return None
    text = module.replace("\\", "/").strip()
    if not text or any(ch in text for ch in ' *?"<>|'):
        return None
    if language == "go":
        parts = [part for part in text.split("/") if part]
        if any("." in part for part in parts):
            return None
    else:
        if "/" in text:
            return None
        parts = [part for part in text.split(".") if part]
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    relative = "/".join(parts) + suffix
    hits = [
        path
        for path, graph in graphs.items()
        if graph.language == language
        and (
            path.replace("\\", "/") == relative or path.replace("\\", "/").endswith("/" + relative)
        )
    ]
    unique = sorted(set(hits))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return _AMBIGUOUS
    return None


def _unique_suffix(module: str, graphs: dict[str, SyntaxGraph], language: str) -> str | None:
    parts = [part for part in module.replace("\\", "/").split(".") if part and part != "/"]
    if not parts or any(part in {"..", "."} for part in parts):
        return None
    suffix_py = "/".join(parts) + ".py"
    suffix_init = "/".join(parts) + "/__init__.py"
    hits: list[str] = []
    for path, graph in graphs.items():
        if graph.language != language:
            continue
        norm = path.replace("\\", "/")
        if norm == suffix_py or norm.endswith("/" + suffix_py):
            hits.append(path)
        elif norm == suffix_init or norm.endswith("/" + suffix_init):
            hits.append(path)
    unique = sorted(set(hits))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return _AMBIGUOUS
    return None


_JS_SOURCE_SUFFIXES = (".js", ".jsx", ".mjs", ".ts", ".tsx")
_JS_INDEX_SUFFIXES = (
    "",
    ".js",
    ".jsx",
    ".mjs",
    ".ts",
    ".tsx",
    "/index.js",
    "/index.jsx",
    "/index.mjs",
    "/index.ts",
    "/index.tsx",
)


def _resolve_javascript(
    importer: str,
    module: str,
    graphs: dict[str, SyntaxGraph],
    ts_mappings: tuple[tuple[str, str], ...] = (),
) -> str | None:
    if _looks_relative(module):
        raw = os.path.normpath(str(PurePosixPath(importer).parent / module))
        return _js_file_candidates(raw, graphs)
    return _resolve_ts_alias(module, graphs, ts_mappings)


def _js_file_candidates(raw: str, graphs: dict[str, SyntaxGraph]) -> str | None:
    norm = PurePosixPath(raw)
    if ".." in norm.parts or "node_modules" in norm.parts:
        return None
    text = norm.as_posix()
    suffixes: tuple[str, ...] = ("",)
    if PurePosixPath(text).suffix not in _JS_SOURCE_SUFFIXES:
        suffixes = _JS_INDEX_SUFFIXES
    hits: list[str] = []
    for suffix in suffixes:
        cand = text + suffix
        graph = graphs.get(cand)
        if graph is not None and graph.language in {"javascript", "typescript"}:
            hits.append(cand)
    unique = sorted(set(hits))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return _AMBIGUOUS
    return None


def _ts_path_mappings(repo_root: Path | None) -> tuple[tuple[str, str], ...]:
    """Read ``compilerOptions.paths`` as JSON. Configuration is never executed."""
    if repo_root is None:
        return ()
    path = repo_root / "tsconfig.json"
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    if not isinstance(data, dict):
        return ()
    options = data.get("compilerOptions")
    if not isinstance(options, dict):
        return ()
    base = options.get("baseUrl", ".")
    paths = options.get("paths")
    if not isinstance(base, str) or not isinstance(paths, dict):
        return ()
    mappings: list[tuple[str, str]] = []
    for pattern, targets in paths.items():
        if not isinstance(pattern, str) or not isinstance(targets, list):
            continue
        if pattern.count("*") > 1:
            continue
        concrete = [item for item in targets if isinstance(item, str)]
        if len(concrete) != 1 or len(targets) != 1:
            continue
        target = concrete[0]
        if target.count("*") > 1:
            continue
        rewritten = PurePosixPath(base) / target
        mappings.append((pattern, rewritten.as_posix()))
    return tuple(mappings)


def _matches_ts_pattern(module: str, mappings: tuple[tuple[str, str], ...]) -> bool:
    return any(_apply_ts_pattern(pattern, module) is not None for pattern, _target in mappings)


def _apply_ts_pattern(pattern: str, module: str) -> str | None:
    if "*" not in pattern:
        return "" if pattern == module else None
    prefix, suffix = pattern.split("*", 1)
    if not module.startswith(prefix) or not module.endswith(suffix):
        return None
    end = len(module) - len(suffix) if suffix else len(module)
    if end < len(prefix):
        return None
    middle = module[len(prefix) : end]
    if not middle or "*" in middle:
        return None
    return middle


def _resolve_ts_alias(
    module: str,
    graphs: dict[str, SyntaxGraph],
    mappings: tuple[tuple[str, str], ...],
) -> str | None:
    if not mappings or _looks_relative(module):
        return None
    hits: list[str] = []
    for pattern, target in mappings:
        middle = _apply_ts_pattern(pattern, module)
        if middle is None:
            continue
        if "*" in target:
            rewritten = target.replace("*", middle, 1)
        elif middle == "":
            rewritten = target
        else:
            continue
        resolved = _js_file_candidates(rewritten, graphs)
        if resolved is None:
            continue
        if resolved == _AMBIGUOUS:
            return _AMBIGUOUS
        hits.append(resolved)
    unique = sorted(set(hits))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return _AMBIGUOUS
    return None


def _summarize(
    graph: SyntaxGraph,
    function_name: str,
    scope_id: str,
    parameters: tuple[str, ...],
    summaries: dict[tuple[str, str], _Summary],
    targets: dict[tuple[str, str], tuple[str, str] | str],
    *,
    partial: bool,
    max_depth: int,
    decorators: tuple[str, ...] = (),
    edge_budget: int = 10**9,
    charged: set[tuple[str, str, str, str]] | None = None,
) -> tuple[_Summary, int, bool, bool]:
    vocab = vocab_for(graph.language)
    source_pats = tuple(p for src in vocab.sources for p in src.patterns) if vocab else ()
    sinks: tuple[SinkDefinition, ...] = vocab.sinks if vocab else ()
    sanitizers: tuple[SanitizerDefinition, ...] = vocab.sanitizers if vocab else ()
    param_index = {name: index for index, name in enumerate(parameters)}
    timeline: dict[str, list[tuple[int, frozenset[str]]]] = {}
    latest: dict[str, frozenset[str]] = {}
    edges = 0
    limited = False
    capped = False
    hops = 0
    charged_edges = charged if charged is not None else set()
    blocked_fields, _, _ = field_limit_facts(graph)
    alias_info = proven_alias_mirrors(graph)

    def at(name: str, byte: int) -> frozenset[str]:
        hit: frozenset[str] = frozenset()
        best = -1
        for start, tokens in timeline.get(f"{scope_id}::{name}", []):
            if start <= byte and start >= best:
                best = start
                hit = tokens
        return hit

    def usable(summary: _Summary | None) -> _Summary | None:
        if summary is None or summary.partial:
            return None
        if summary.file_path == graph.file_path and summary.function_name == function_name:
            return None
        return summary

    def resolve(call: CallSite) -> _Summary | None:
        if "." in function_name:
            parent = function_name.split(".", 1)[0]
            qualified_self = call.qualified
            for prefix in ("self.", "cls."):
                if qualified_self.startswith(prefix):
                    method = qualified_self[len(prefix) :]
                    if method and "." not in method:
                        return usable(summaries.get((graph.file_path, f"{parent}.{method}")))
        qualified = call.qualified if call.qualified and call.qualified != call.name else ""
        if qualified and ("." in qualified or "/" in qualified):
            target = targets.get((graph.file_path, qualified))
            if isinstance(target, tuple):
                return usable(summaries.get(target))
            if target == _AMBIGUOUS:
                return None
        locals_ = [entity for entity in _module_functions(graph) if entity.name == call.name]
        if len(locals_) > 1:
            return None
        if len(locals_) == 1 and "." not in call.qualified:
            return usable(summaries.get((graph.file_path, call.name)))
        for key in (call.qualified, call.name):
            target = targets.get((graph.file_path, key))
            if isinstance(target, tuple):
                return usable(summaries.get(target))
            if target == _AMBIGUOUS:
                return None
        return None

    for binding in _ordered_bindings(graph):
        if binding.scope_id != scope_id:
            continue
        byte = _use_byte(binding.span, binding.line)
        field_key = (binding.symbol_id, binding.definition_index)
        if field_key in blocked_fields:
            continue
        tokens, added, hop, hit_limit, hit_cap = _binding_tokens(
            binding,
            graph,
            scope_id,
            param_index,
            source_pats,
            sanitizers,
            at,
            resolve,
            max_depth,
            edge_budget,
            charged_edges,
            graph.file_path,
            function_name,
        )
        edges += added
        hops = max(hops, hop)
        limited = limited or hit_limit
        capped = capped or hit_cap
        if binding.is_conditional:
            tokens = tokens | latest.get(binding.symbol_id, frozenset())
        latest[binding.symbol_id] = tokens
        timeline.setdefault(binding.symbol_id, []).append((byte, tokens))
        if field_key not in blocked_fields and is_field_path(binding.name):
            _mirror_summary_field(binding, tokens, byte, scope_id, timeline, alias_info)

    returned: set[str] = set()
    for ret in graph.returns:
        if ret.scope_id != scope_id:
            continue
        byte = _use_byte(ret.span, ret.line)
        source_tokens: set[str] = set()
        for fragment in (*ret.accesses, *ret.idents):
            hit = fragment_matches_source(fragment, source_pats)
            if hit:
                source_tokens.add(f"source:{hit}")
        ident_tokens: set[str] = set()
        for ident in ret.idents:
            ident_tokens |= set(at(ident, byte))
        ident_tokens |= _field_tokens(ret.accesses, at, byte)
        opaque = False
        from_calls: set[str] = set()
        for call in calls_contained_in(
            graph.calls,
            scope_id=scope_id,
            line=ret.line,
            span=ret.span,
            outermost=True,
        ):
            if _call_is_transparent(call, source_pats, sanitizers):
                continue
            summary = resolve(call)
            if summary is None:
                opaque = True
                continue
            if summary.file_path != graph.file_path:
                if summary.hops + 1 > max_depth:
                    limited = True
                    opaque = True
                    continue
                status = _charge_cross_file(
                    charged_edges, edge_budget, graph.file_path, function_name, summary
                )
                if status == "capped":
                    capped = True
                    opaque = True
                    continue
                if status == "new":
                    edges += 1
                hops = max(hops, summary.hops + 1)
            if summary.return_effects:
                from_calls |= _instantiate(summary.return_effects, call, at, byte, source_pats)
            else:
                opaque = True
        if opaque:
            returned |= source_tokens | from_calls
        else:
            returned |= source_tokens | ident_tokens | from_calls
        sanitizer = _sanitizer_on(ret.accesses, sanitizers)
        if sanitizer is not None and sanitizer.effective and returned:
            returned = {f"sanitized:{sanitizer.kind}:{token}" for token in returned}

    param_sinks: set[tuple[int, str, str, int]] = set()
    for call in graph.calls:
        if call.scope_id != scope_id:
            continue
        summary = resolve(call)
        if (
            summary is not None
            and summary.file_path == graph.file_path
            and summary.function_name == function_name
        ):
            summary = None
        if summary is not None:
            cross = summary.file_path != graph.file_path
            if cross and summary.hops + 1 > max_depth:
                limited = True
                continue
            if cross:
                status = _charge_cross_file(
                    charged_edges, edge_budget, graph.file_path, function_name, summary
                )
                if status == "capped":
                    capped = True
                    continue
                if status == "new":
                    edges += 1
                hops = max(hops, summary.hops + 1)
            for index, vuln, sink_name, line in summary.param_sinks:
                for token in _call_arg_tokens(
                    call, index, source_pats, at, _use_byte(call.span, call.line)
                ):
                    kind, core = _peel_sanitizer(token)
                    if not core.startswith("param:"):
                        continue
                    if kind and _kind_blocks(kind, vuln):
                        continue
                    try:
                        param_i = int(core.split(":", 1)[1])
                    except ValueError:
                        continue
                    param_sinks.add((param_i, vuln, sink_name, line))
            continue
        for sink in sinks:
            if sink.required_context:
                continue
            if not call_matches_sink(call, sink):
                continue
            indexes = (
                list(sink.argument_indexes) if sink.argument_indexes else _all_arg_indexes(call)
            )
            kinds = sink.sanitizer_kinds or tuple(
                SINK_SANITIZER_KINDS.get(sink.vulnerability_class, ())
            )
            if sanitizer_intervened(
                call, sanitizers, allowed_kinds=kinds, argument_indexes=indexes
            ):
                continue
            for index in indexes:
                for token in _call_arg_tokens(
                    call, index, source_pats, at, _use_byte(call.span, call.line)
                ):
                    kind, core = _peel_sanitizer(token)
                    if not core.startswith("param:"):
                        continue
                    if kind and kind in kinds:
                        continue
                    try:
                        param_i = int(core.split(":", 1)[1])
                    except ValueError:
                        continue
                    param_sinks.add(
                        (
                            param_i,
                            sink.vulnerability_class.value,
                            call.qualified or call.name,
                            call.line,
                        )
                    )

    summary = _Summary(
        file_path=graph.file_path,
        function_name=function_name,
        scope_id=scope_id,
        parameters=parameters,
        return_effects=tuple(sorted(returned)),
        param_sinks=tuple(sorted(param_sinks)),
        hops=hops,
        partial=partial,
        decorators=decorators,
    )
    return summary, edges, limited, capped


def _call_is_transparent(
    call: CallSite,
    source_pats: tuple[str, ...],
    sanitizers: tuple[SanitizerDefinition, ...],
) -> bool:
    names = tuple(name for name in (call.qualified, call.name) if name)
    if any(_callee_is_source(name, source_pats) for name in names):
        return True
    if any(_preserves_taint(name) for name in names):
        return True
    return _sanitizer_on(names, sanitizers) is not None


def _binding_tokens(
    binding: Binding,
    graph: SyntaxGraph,
    scope_id: str,
    param_index: dict[str, int],
    source_pats: tuple[str, ...],
    sanitizers: tuple[SanitizerDefinition, ...],
    at: Callable[[str, int], frozenset[str]],
    resolve: Callable[[CallSite], _Summary | None],
    max_depth: int,
    edge_budget: int,
    charged: set[tuple[str, str, str, str]],
    caller_file: str,
    caller_name: str,
) -> tuple[frozenset[str], int, int, bool, bool]:
    if (
        _empty_rhs(binding)
        and binding.name in param_index
        and (
            binding.kind is SymbolKind.PARAMETER
            or binding.declarator == "param"
            or binding.is_declaration
        )
    ):
        return frozenset({f"param:{param_index[binding.name]}"}), 0, 0, False, False
    if _killing_assignment(binding):
        return frozenset(), 0, 0, False, False
    source_tokens: set[str] = set()
    ident_tokens: set[str] = set()
    edges = 0
    hops = 0
    limited = False
    capped = False
    byte = _use_byte(binding.span, binding.line)
    for fragment in (*binding.rhs_accesses, *binding.rhs_callees, *binding.rhs_idents):
        hit = fragment_matches_source(fragment, source_pats)
        if hit:
            source_tokens.add(f"source:{hit}")
    for ident in binding.rhs_idents:
        ident_tokens |= set(at(ident, byte))
    ident_tokens |= _field_tokens(binding.rhs_accesses, at, byte)
    opaque = False
    from_calls: set[str] = set()
    names = set(binding.rhs_callees)
    contained = calls_contained_in(
        graph.calls,
        scope_id=scope_id,
        line=binding.line,
        span=binding.span,
        names=names,
    )
    for call in calls_contained_in(
        graph.calls,
        scope_id=scope_id,
        line=binding.line,
        span=binding.span,
        names=names,
        outermost=True,
    ):
        if _call_is_transparent(call, source_pats, sanitizers):
            continue
        summary = resolve(call)
        if summary is None:
            opaque = True
            continue
        if summary.file_path != graph.file_path and summary.hops + 1 > max_depth:
            limited = True
            opaque = True
            continue
        if summary.file_path != graph.file_path:
            status = _charge_cross_file(charged, edge_budget, caller_file, caller_name, summary)
            if status == "capped":
                capped = True
                opaque = True
                continue
            if status == "new":
                edges += 1
            hops = max(hops, summary.hops + 1)
        if summary.return_effects:
            from_calls |= _instantiate(summary.return_effects, call, at, byte, source_pats)
        else:
            opaque = True
    if binding.rhs_callees and not opaque:
        seen = {call.qualified for call in contained}
        seen.update(call.name for call in contained)
        for callee in binding.rhs_callees:
            if _callee_is_source(callee, source_pats) or _preserves_taint(callee):
                continue
            if _sanitizer_on((callee,), sanitizers) is not None:
                continue
            if callee not in seen and callee.rsplit(".", 1)[-1] not in seen:
                opaque = True
    tokens = source_tokens | from_calls
    if not opaque:
        tokens |= ident_tokens
    sanitizer = _sanitizer_on(binding.rhs_callees, sanitizers)
    if sanitizer is not None and sanitizer.effective and tokens:
        tokens = {f"sanitized:{sanitizer.kind}:{token}" for token in tokens}
    return frozenset(tokens), edges, hops, limited, capped


def _instantiate(
    effects: tuple[str, ...],
    call: CallSite,
    at: Callable[[str, int], frozenset[str]],
    byte: int,
    source_pats: tuple[str, ...],
) -> set[str]:
    out: set[str] = set()
    for effect in effects:
        kind, core = _peel_sanitizer(effect)
        produced: set[str] = set()
        if core.startswith("source:"):
            produced.add(core)
        elif core.startswith("param:"):
            try:
                index = int(core.split(":", 1)[1])
            except ValueError:
                continue
            produced |= _call_arg_tokens(call, index, source_pats, at, byte)
        for token in produced:
            out.add(f"sanitized:{kind}:{token}" if kind else token)
    return out


def _call_arg_tokens(
    call: CallSite,
    index: int,
    source_pats: tuple[str, ...],
    at: Callable[[str, int], frozenset[str]],
    byte: int,
) -> set[str]:
    arg = call.argument_at(index) if call.arguments else None
    if arg is not None:
        if (
            arg.is_literal
            and not arg.idents
            and not arg.accesses
            and not arg.callees
            and not arg.dynamic
        ):
            return set()
        fragments = (*arg.accesses, *arg.callees, *arg.idents)
        idents = arg.idents
        accesses = arg.accesses
    elif index == 0:
        fragments = (*call.argument_accesses, *call.argument_idents)
        idents = call.argument_idents
        accesses = call.argument_accesses
    else:
        return set()
    tokens: set[str] = set()
    for fragment in fragments:
        hit = fragment_matches_source(fragment, source_pats)
        if hit:
            tokens.add(f"source:{hit}")
    for ident in idents:
        tokens |= set(at(ident, byte))
    tokens |= _field_tokens(accesses, at, byte)
    return tokens


def _field_tokens(
    accesses: tuple[str, ...],
    at: Callable[[str, int], frozenset[str]],
    byte: int,
) -> set[str]:
    tokens: set[str] = set()
    for access in accesses:
        if is_field_path(access) and field_flow_allowed(access):
            tokens |= set(at(access, byte))
    return tokens


def _mirror_summary_field(
    binding: Binding,
    tokens: frozenset[str],
    byte: int,
    scope_id: str,
    timeline: dict[str, list[tuple[int, frozenset[str]]]],
    alias_info: AliasMirrors,
) -> None:
    parsed = field_root(binding.name)
    if parsed is None:
        return
    root, suffix = parsed
    mirrors = alias_info.targets.get((binding.scope_id, root), ())
    budget = settings.taint_max_alias_edges
    for offset, (other, hold) in enumerate(mirrors):
        if budget <= 0 or offset >= budget:
            break
        other_name = f"{other}{suffix}"
        if other_name == binding.name or not field_flow_allowed(other_name):
            continue
        sid = f"{scope_id}::{other_name}"
        timeline.setdefault(sid, []).append((max(byte, hold), tokens))


def _all_arg_indexes(call: CallSite) -> list[int]:
    if call.arguments:
        return [arg.index for arg in call.arguments]
    return [0]


def _sanitizer_on(
    fragments: tuple[str, ...], sanitizers: tuple[SanitizerDefinition, ...]
) -> SanitizerDefinition | None:
    for sanitizer in sanitizers:
        for name in sanitizer.api_names:
            for fragment in fragments:
                if fragment == name or fragment.endswith("." + name):
                    return sanitizer
    return None


def _kind_blocks(kind: str, vuln_value: str) -> bool:
    try:
        vuln = VulnerabilityClass(vuln_value)
    except ValueError:
        return False
    return kind in SINK_SANITIZER_KINDS.get(vuln, frozenset())


def _externals_from(
    graphs: dict[str, SyntaxGraph],
    summaries: dict[tuple[str, str], _Summary],
    targets: dict[tuple[str, str], tuple[str, str] | str],
    classes: dict[tuple[str, str], tuple[str, str]],
    link_kinds: dict[tuple[str, str], str],
    *,
    max_depth: int,
) -> tuple[dict[str, dict[str, ExternalCallee]], bool, list[FlowDiagnostic]]:
    externals: dict[str, dict[str, ExternalCallee]] = {}
    diagnostics: list[FlowDiagnostic] = []
    dropped = False
    partial_files: set[str] = set()
    for path, graph in sorted(graphs.items()):
        local_functions = {entity.name for entity in _module_functions(graph)}
        caller_partial = _graph_partial(graph)
        table: dict[str, ExternalCallee] = {}
        for (importer, local), target in sorted(targets.items()):
            if importer != path or not isinstance(target, tuple):
                continue
            bare = local.rsplit(".", 1)[-1]
            if local in local_functions or (bare in local_functions and "." not in local):
                continue
            placed = _place_external(
                table,
                local,
                summaries.get(target),
                path,
                offset=0,
                caller_partial=caller_partial,
                max_depth=max_depth,
                relationship=link_kinds.get((path, local), "import"),
            )
            if placed == "partial":
                partial_files.add(target[0])
            elif placed == "depth":
                dropped = True
        file_classes = {
            local: dest for (importer, local), dest in classes.items() if importer == path
        }
        for local, (class_file, class_name) in sorted(file_classes.items()):
            methods = (
                _methods_by_class(graphs[class_file]).get(class_name, {})
                if class_file in graphs
                else {}
            )
            for method_name, _entity in sorted(methods.items()):
                summary = summaries.get((class_file, f"{class_name}.{method_name}"))
                explicit = _place_external(
                    table,
                    f"{local}.{method_name}",
                    summary,
                    path,
                    offset=_receiver_offset(summary, instance=False),
                    caller_partial=caller_partial,
                    max_depth=max_depth,
                    relationship="method",
                )
                constructed = _place_external(
                    table,
                    f"{local}().{method_name}",
                    summary,
                    path,
                    offset=_receiver_offset(summary, instance=True),
                    caller_partial=caller_partial,
                    max_depth=max_depth,
                    relationship="method",
                )
                if explicit == "partial" or constructed == "partial":
                    partial_files.add(class_file)
                if explicit == "depth" or constructed == "depth":
                    dropped = True
        for binding in _ordered_bindings(graph):
            for callee in binding.rhs_callees:
                simple = callee.rsplit(".", 1)[-1]
                if simple.startswith("new."):
                    simple = simple[4:]
                dest = file_classes.get(callee) or file_classes.get(simple)
                if dest is None:
                    continue
                class_file, class_name = dest
                methods = (
                    _methods_by_class(graphs[class_file]).get(class_name, {})
                    if class_file in graphs
                    else {}
                )
                for method_name in sorted(methods):
                    summary = summaries.get((class_file, f"{class_name}.{method_name}"))
                    placed = _place_external(
                        table,
                        f"{binding.name}.{method_name}",
                        summary,
                        path,
                        offset=_receiver_offset(summary, instance=True),
                        caller_partial=caller_partial,
                        max_depth=max_depth,
                        relationship="method",
                    )
                    if placed == "partial":
                        partial_files.add(class_file)
                    elif placed == "depth":
                        dropped = True
        if table:
            externals[path] = dict(sorted(table.items()))
            if caller_partial:
                diagnostics.append(
                    FlowDiagnostic(
                        kind="cross_file_partial",
                        message="partial importer graph is not a complete cross-file summary",
                        file_path=path,
                    )
                )
    for file_path in sorted(partial_files):
        diagnostics.append(
            FlowDiagnostic(
                kind="cross_file_partial",
                message="partial callee graph is not a cross-file summary",
                file_path=file_path,
            )
        )
    return externals, dropped, diagnostics


def _receiver_offset(summary: _Summary | None, *, instance: bool) -> int:
    if summary is None:
        return 0
    return receiver_argument_offset(summary.parameters, summary.decorators, instance=instance)


def _charge_cross_file(
    charged: set[tuple[str, str, str, str]],
    max_edges: int,
    caller_file: str,
    caller_name: str,
    summary: _Summary,
) -> str:
    """Charge one caller/callee relationship at most once across rounds."""
    if summary.file_path == caller_file:
        return "local"
    key = (caller_file, caller_name, summary.file_path, summary.function_name)
    if key in charged:
        return "paid"
    if len(charged) >= max_edges:
        return "capped"
    charged.add(key)
    return "new"


def _shift_effect(effect: str, offset: int) -> str | None:
    kind, core = _peel_sanitizer(effect)
    if core.startswith("param:"):
        try:
            index = int(core.split(":", 1)[1])
        except ValueError:
            return None
        shifted = index - offset
        if shifted < 0:
            return None
        core = f"param:{shifted}"
    rendered = f"sanitized:{kind}:{core}" if kind else core
    return rendered


def _place_external(
    table: dict[str, ExternalCallee],
    local: str,
    summary: _Summary | None,
    importer: str,
    *,
    offset: int,
    caller_partial: bool,
    max_depth: int,
    relationship: str = "import",
) -> str:
    if summary is None:
        return ""
    if summary.file_path == importer and relationship != "alias":
        return ""
    if summary.partial:
        return "partial"
    if summary.hops >= max_depth and (summary.return_effects or summary.param_sinks):
        return "depth"
    effects_list: list[str] = []
    for effect in summary.return_effects:
        shifted = _shift_effect(effect, offset)
        if shifted is not None:
            effects_list.append(shifted)
    effects = tuple(effects_list)
    sinks: list[tuple[int, str, str, int]] = []
    for index, vuln, name, line in summary.param_sinks:
        shifted_index = index - offset
        if shifted_index < 0:
            continue
        sinks.append((shifted_index, vuln, name, line))
    semantic_id = _summary_symbol_id(summary.file_path, summary.function_name)
    previous = table.get(local)
    if previous is not None and previous.semantic_id and previous.semantic_id != semantic_id:
        table.pop(local, None)
        return ""
    table[local] = ExternalCallee(
        local_name=local,
        callee_file=summary.file_path,
        callee_function=summary.function_name,
        return_effects=tuple(sorted(effects)),
        param_sinks=tuple(sorted(sinks)),
        partial=caller_partial,
        hops=summary.hops,
        semantic_id=semantic_id,
        relationship=relationship,
    )
    return "ok"
