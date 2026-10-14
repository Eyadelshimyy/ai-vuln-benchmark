from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from rox_dox.components import (
    Endpoint,
    ExternalCall,
    FileFacts,
    Worker,
)
from rox_dox.features import (
    Feature,
    FeatureEvidence,
    FeatureFile,
    FeatureMap,
    TableLink,
    _is_deployment_path,
)
from rox_dox.model import (
    BlockDiagram,
    Claim,
    CodeSource,
    DataModel,
    MembershipGroup,
    MembershipRow,
    Page,
    Related,
    Relation,
    SchemaDomain,
    SummaryTable,
    SummaryTableLink,
    Tldr,
)
from rox_dox.projection import MAX_SOURCES_PER_EDGE, project_component_diagram
from rox_dox.schema import Table
from rox_dox.sequences import block_sequences

Layer = Literal["web", "routes", "workers", "models", "logic", "skills", "deploy"]

MAX_CALLER_IMPORT_HOPS = 3

LAYER_ORDER: tuple[Layer, ...] = (
    "web",
    "routes",
    "workers",
    "logic",
    "models",
    "skills",
    "deploy",
)
LAYER_LABELS: dict[Layer, str] = {
    "web": "Web screens",
    "routes": "HTTP routes",
    "workers": "Background workers",
    "models": "Models",
    "logic": "Business logic",
    "skills": "Agent skills",
    "deploy": "Deployment",
}
ROUTE_FILENAMES = {"routes.py", "router.py", "views.py", "endpoints.py"}
WORKER_SEGMENTS = {"tasks", "temporal", "workers", "executors", "cron", "jobs"}
GENERIC_SEGMENTS = {
    "backend",
    "src",
    "tasks",
    "task",
    "executors",
    "executor",
    "workers",
    "worker",
    "temporal",
    "activities",
    "activity",
    "workflows",
    "workflow",
    "jobs",
    "job",
    "cron",
}


def layer_of(path: str) -> Layer:
    parts = PurePosixPath(path).parts
    filename = PurePosixPath(path).name
    if path.startswith("web/"):
        return "web"
    if path.startswith(".agents/skills/"):
        return "skills"
    if _is_deployment_path(path):
        return "deploy"
    if {"routes", "router"} & set(parts) or filename in ROUTE_FILENAMES:
        return "routes"
    if WORKER_SEGMENTS & set(parts):
        return "workers"
    if "models" in parts or filename == "models.py":
        return "models"
    return "logic"


def _humanize(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("-", "_").split("_"))


def _feature_name(feature: Feature, names: Mapping[str, str]) -> str:
    name = names.get(feature.id)
    if name is not None:
        return name
    return _humanize(feature.id)


@dataclass(frozen=True)
class _FeatureBlock:
    diagram: BlockDiagram
    tables_by_node: Mapping[str, tuple[str, ...]]
    services_by_node: Mapping[str, tuple[str, ...]]
    kinds_by_node: Mapping[str, str]
    callers: tuple[_Caller, ...]
    library_rows: tuple[_LibraryRow, ...]


@dataclass(frozen=True)
class _Caller:
    path: str
    targets: tuple[str, ...]
    hops: int
    import_sources: tuple[CodeSource, ...]
    owner_name: str | None
    owner_page: str | None


@dataclass(frozen=True)
class _LibraryRow:
    label: str
    tables: tuple[str, ...]
    services: tuple[str, ...]
    sources: tuple[CodeSource, ...]


def _feature_facts(
    feature: Feature,
    components: Mapping[str, FileFacts],
) -> dict[str, FileFacts]:
    return {
        file.path: components.get(
            file.path,
            FileFacts(
                path=file.path,
                api_group=None,
                endpoints=[],
                workers=[],
                externals=[],
            ),
        )
        for file in feature.files
    }


def _feature_callers(
    feature: Feature,
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    primary_features: Mapping[str, tuple[str, str]],
) -> tuple[_Caller, ...]:
    member_paths = {file.path for file in feature.files}
    callers = []
    for path in sorted(component_imports):
        if path in member_paths:
            continue
        facts = component_facts.get(path)
        if facts is None or not (facts.endpoints or facts.workers):
            continue

        targets: set[str] = set()
        target_hops: dict[str, int] = {}
        import_lines: set[int] = set()
        pending: deque[tuple[str, int, int | None]] = deque([(path, 0, None)])
        visited: set[tuple[str, int | None]] = {(path, None)}
        while pending:
            current_path, hops, first_import_line = pending.popleft()
            if hops >= MAX_CALLER_IMPORT_HOPS:
                continue
            for imported_path, import_line in sorted(
                component_imports.get(current_path, {}).items()
            ):
                next_first_import_line = (
                    import_line if first_import_line is None else first_import_line
                )
                if imported_path in member_paths:
                    targets.add(imported_path)
                    target_hops[imported_path] = min(
                        target_hops.get(imported_path, hops + 1),
                        hops + 1,
                    )
                    import_lines.add(next_first_import_line)
                    continue
                imported_facts = component_facts.get(imported_path)
                if (
                    imported_facts is None
                    or imported_facts.endpoints
                    or imported_facts.workers
                ):
                    continue
                state = (imported_path, next_first_import_line)
                if state in visited:
                    continue
                visited.add(state)
                pending.append((imported_path, hops + 1, next_first_import_line))

        if not targets:
            continue
        owner = primary_features.get(path)
        callers.append(
            _Caller(
                path=path,
                targets=tuple(sorted(targets)),
                hops=min(target_hops.values()),
                import_sources=tuple(
                    _source(path, line)
                    for line in sorted(import_lines)[:MAX_SOURCES_PER_EDGE]
                ),
                owner_name=owner[0] if owner else None,
                owner_page=owner[1] if owner else None,
            )
        )
    return tuple(callers)


def _runtime_records(
    facts: Mapping[str, FileFacts],
) -> tuple[
    list[tuple[str, Endpoint]],
    list[tuple[str, Worker]],
    list[tuple[str, ExternalCall]],
]:
    endpoints = sorted(
        (
            (path, endpoint)
            for path, file_facts in facts.items()
            for endpoint in file_facts.endpoints
        ),
        key=lambda item: (
            item[1].path,
            item[1].method,
            item[0],
            item[1].handler,
            item[1].line,
        ),
    )
    workers = sorted(
        (
            (path, worker)
            for path, file_facts in facts.items()
            for worker in file_facts.workers
        ),
        key=lambda item: (item[1].kind, item[1].name, item[0], item[1].line),
    )
    externals = sorted(
        (
            (path, external)
            for path, file_facts in facts.items()
            for external in file_facts.externals
        ),
        key=lambda item: (
            item[1].service,
            item[1].module,
            item[0],
            item[1].line,
        ),
    )
    return endpoints, workers, externals


def _worker_label(
    directory: str,
    records: list[tuple[str, Worker]],
    family: str,
) -> str:
    label_family = "workers" if family == "task_executor" else "workflows"
    for segment in reversed(PurePosixPath(directory).parts):
        normalized = segment.lower().replace("-", "_")
        if normalized not in GENERIC_SEGMENTS:
            return f"{_humanize(normalized)} {label_family}"
    first_name = min(records, key=lambda item: (item[1].name, item[0], item[1].line))[
        1
    ].name
    base_name = re.sub(r"(TaskExecutor|Executor|Workflow|Activity)$", "", first_name)
    label = _humanize(base_name or first_name)
    more = len(records) - 1
    if more:
        label = f"{label} + {more} more"
    return f"{label} {label_family}"


def _caller_records(
    callers: Collection[_Caller],
    component_facts: Mapping[str, FileFacts],
) -> tuple[list[tuple[str, Endpoint]], list[tuple[str, Worker]]]:
    facts = {caller.path: component_facts[caller.path] for caller in callers}
    endpoints, workers, _ = _runtime_records(facts)
    return endpoints, workers


def _caller_groups(
    endpoints: list[tuple[str, Endpoint]],
    workers: list[tuple[str, Worker]],
    component_facts: Mapping[str, FileFacts],
) -> list[str]:
    groups = {
        _humanize(component_facts[path].api_group or PurePosixPath(path).stem)
        for path, _ in endpoints
    }
    worker_groups: defaultdict[tuple[str, str], list[tuple[str, Worker]]] = defaultdict(
        list
    )
    for path, worker in workers:
        family = "task_executor" if worker.kind == "task_executor" else "temporal"
        worker_groups[(PurePosixPath(path).parent.as_posix(), family)].append(
            (path, worker)
        )
    groups.update(
        _worker_label(directory, records, family)
        for (directory, family), records in sorted(worker_groups.items())
    )
    return sorted(groups)


def _web_directory_detail(directory: str, file_count: int) -> str:
    suffix = f" — {file_count} files"
    max_path_length = 90 - len(suffix)
    parts = PurePosixPath(directory).parts
    retained = list(parts)
    while len("/".join(retained)) > max_path_length - 2 and len(retained) > 1:
        retained.pop(0)
    summary = "/".join(retained)
    if len(retained) < len(parts):
        summary = f"…/{summary}"
    if len(summary) > max_path_length:
        summary = f"…{summary[-(max_path_length - 1) :]}"
    return f"{summary}{suffix}"


def _source(path: str, line: int) -> CodeSource:
    return CodeSource(path=path, lines=(line, line))


def _access_label(
    write_sources: Collection[CodeSource],
    read_sources: Collection[CodeSource],
) -> str:
    if write_sources and read_sources:
        return "reads & writes"
    if write_sources:
        return "writes"
    if read_sources:
        return "reads"
    return "uses"


def _access_source(
    write_sources: Collection[CodeSource],
    read_sources: Collection[CodeSource],
    evidence_sources: Collection[CodeSource],
) -> CodeSource:
    candidates = write_sources or read_sources or evidence_sources
    return min(candidates, key=lambda source: (source.path, source.lines[0]))


def _unique_sources(sources: list[CodeSource]) -> list[CodeSource]:
    found: dict[tuple[str, int, int], CodeSource] = {}
    for source in sources:
        found.setdefault((source.path, *source.lines), source)
    return list(found.values())


def _table_sources(feature: Feature, tables: Mapping[str, Table]) -> list[CodeSource]:
    return _unique_sources([tables[name].source for name in feature.tables])


def _file_sources(file: FeatureFile, fallback: CodeSource) -> list[CodeSource]:
    return _unique_sources(
        [
            _source(evidence.path, evidence.line)
            for evidence in sorted(
                file.evidence,
                key=lambda item: (item.path, item.line, item.kind),
            )
        ]
    ) or [fallback]


def _evidence_text(evidence: FeatureEvidence) -> str:
    if evidence.kind == "table":
        return f"uses {evidence.table}"
    if evidence.kind == "call":
        return f"imports {evidence.to}"
    if evidence.kind == "route":
        return f"route tag {evidence.tag}"
    if evidence.kind == "reference":
        return f"references {evidence.target}"
    return evidence.kind


def _file_evidence_source(file: FeatureFile, fallback: CodeSource) -> CodeSource:
    evidence = sorted(
        (item for item in file.evidence if item.path == file.path),
        key=lambda item: (item.path, item.line, item.kind),
    )
    if evidence:
        item = evidence[0]
        return _source(item.path, item.line)
    if file.evidence:
        item = min(file.evidence, key=lambda evidence: (evidence.path, evidence.line))
        return _source(item.path, item.line)
    return fallback


def _top_directories(
    files: list[FeatureFile], layer: Layer
) -> list[tuple[str, list[FeatureFile]]]:
    by_directory: defaultdict[str, list[FeatureFile]] = defaultdict(list)
    for file in files:
        if layer_of(file.path) == layer:
            directory = PurePosixPath(file.path).parent.as_posix()
            by_directory[directory].append(file)
    return sorted(by_directory.items(), key=lambda item: (-len(item[1]), item[0]))[:3]


def _directory_detail_source(
    files: list[FeatureFile], fallback: CodeSource
) -> CodeSource:
    file = min(files, key=lambda item: item.path)
    return _file_evidence_source(file, fallback)


def _feature_tldr(
    feature: Feature,
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    block: _FeatureBlock,
) -> Tldr:
    table_sources = _table_sources(feature, tables)
    files = feature.files
    facts = _feature_facts(feature, component_facts)
    endpoints, workers, externals = _runtime_records(facts)
    other_feature_callers = [
        caller for caller in block.callers if caller.owner_name is not None
    ]
    caller_endpoints, caller_workers = _caller_records(
        other_feature_callers,
        component_facts,
    )
    caller_endpoints = list(
        {
            (path, endpoint.method, endpoint.path, endpoint.handler, endpoint.line): (
                path,
                endpoint,
            )
            for path, endpoint in caller_endpoints
        }.values()
    )
    caller_workers = list(
        {
            (path, worker.kind, worker.name, worker.line): (path, worker)
            for path, worker in caller_workers
        }.values()
    )
    summary = []
    endpoints_by_group: defaultdict[str, list[tuple[str, Endpoint]]] = defaultdict(list)
    for path, endpoint in endpoints:
        group = facts[path].api_group or PurePosixPath(path).stem
        endpoints_by_group[group].append((path, endpoint))
    if endpoints:
        endpoint_sources = [
            _source(path, endpoint.line)
            for group in sorted(endpoints_by_group)
            for path, endpoint in [
                min(
                    endpoints_by_group[group],
                    key=lambda item: (
                        item[1].path,
                        item[1].method,
                        item[0],
                        item[1].line,
                    ),
                )
            ]
        ]
        summary.append(
            Claim(
                text=(
                    f"Owns {len(endpoints)} HTTP endpoints in "
                    f"{len(endpoints_by_group)} API groups: "
                    f"{', '.join(_humanize(group) for group in sorted(endpoints_by_group))}."
                ),
                sources=_unique_sources(endpoint_sources),
            )
        )
    if caller_endpoints or caller_workers:
        caller_sources = [
            *[_source(path, endpoint.line) for path, endpoint in caller_endpoints],
            *[_source(path, worker.line) for path, worker in caller_workers],
            *[
                source
                for caller in other_feature_callers
                for source in caller.import_sources
            ],
        ]
        summary.append(
            Claim(
                text=(
                    f"Reached from {len(caller_endpoints)} HTTP endpoints and "
                    f"{len(caller_workers)} workers in other features: "
                    f"{', '.join(_caller_groups(caller_endpoints, caller_workers, component_facts))}."
                ),
                sources=_unique_sources(caller_sources),
            )
        )
    if workers:
        named_workers = sorted(
            workers,
            key=lambda item: (item[1].name, item[0], item[1].line),
        )
        worker_names = [worker.name for _, worker in named_workers[:5]]
        more = len(named_workers) - len(worker_names)
        summary.append(
            Claim(
                text=(
                    f"Runs {len(workers)} background workers and workflows: "
                    f"{', '.join(worker_names)}"
                    f"{f' and {more} more' if more else ''}."
                ),
                sources=_unique_sources(
                    [_source(path, worker.line) for path, worker in named_workers[:5]]
                ),
            )
        )
    services = sorted({external.service for _, external in externals})
    if services:
        summary.append(
            Claim(
                text=f"Calls outside services: {', '.join(services)}.",
                sources=_unique_sources(
                    [_source(path, external.line) for path, external in externals]
                ),
            )
        )
    if feature.tables:
        summary.append(
            Claim(
                text=f"Stores data in: {', '.join(feature.tables)}.",
                sources=table_sources,
            )
        )
    if not endpoints and not workers:
        summary.append(
            Claim(
                text=(
                    "No HTTP endpoints or background workers; its code is library "
                    "code used by other features."
                ),
                sources=[
                    _file_evidence_source(
                        min(files, key=lambda file: file.path),
                        table_sources[0],
                    )
                    if files
                    else table_sources[0]
                ],
            )
        )
    key_points = []
    for layer in LAYER_ORDER:
        layer_files = [file for file in files if layer_of(file.path) == layer]
        if not layer_files:
            continue
        directories = _top_directories(files, layer)
        directory_list = ", ".join(
            f"{directory} ({len(directory_files)})"
            for directory, directory_files in directories
        )
        key_points.append(
            Claim(
                text=f"{LAYER_LABELS[layer]}: {directory_list}.",
                sources=_unique_sources(
                    [
                        _directory_detail_source(directory_files, table_sources[0])
                        for _, directory_files in directories
                    ]
                ),
            )
        )
    rows = []
    row_sources = list(table_sources)
    for node in block.diagram.nodes:
        tables_used = block.tables_by_node.get(node.id, ())
        outside_services = block.services_by_node.get(node.id, ())
        if node.kind == "store" and not tables_used:
            tables_used = (node.label,)
        if node.kind == "external" and not outside_services:
            outside_services = (node.label,)
        rows.append(
            [
                node.label,
                block.kinds_by_node.get(node.id, node.kind),
                ", ".join(tables_used) or "—",
                ", ".join(outside_services) or "—",
            ]
        )
        row_sources.extend(
            [
                node.source,
                *(source for detail in node.details for source in detail.sources),
            ]
        )
        row_sources.extend(
            edge.source
            for edge in block.diagram.edges
            if edge.src == node.id or edge.dst == node.id
        )
    for library in block.library_rows:
        rows.append(
            [
                library.label,
                "Library code",
                ", ".join(library.tables) or "—",
                ", ".join(library.services) or "—",
            ]
        )
        row_sources.extend(library.sources)
    return Tldr(
        summary=summary,
        key_points=key_points,
        table=SummaryTable(
            columns=["Component", "Kind", "Tables used", "Outside services"],
            rows=rows,
            sources=_unique_sources(row_sources),
        ),
        notes=[],
    )


def _relationship_source(
    feature: Feature,
    other: Feature,
    shared_paths: set[str],
    fallback: CodeSource,
) -> CodeSource:
    evidence = [
        item
        for candidate in (feature, other)
        for file in candidate.files
        if file.path in shared_paths
        for item in file.evidence
    ]
    if not evidence:
        return fallback
    item = min(evidence, key=lambda item: (item.path, item.line))
    return _source(item.path, item.line)


def _library_rows_for_feature(
    feature: Feature,
    unreached_paths: Collection[str],
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]],
) -> tuple[_LibraryRow, ...]:
    fallback = tables[feature.id].source
    files = {file.path: file for file in feature.files}
    paths_by_directory: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(unreached_paths):
        paths_by_directory[PurePosixPath(path).parent.as_posix()].append(path)
    rows = []
    for directory, paths in sorted(paths_by_directory.items()):
        row_tables = set()
        row_services = set()
        sources = []
        for path in paths:
            file = files[path]
            for evidence in file.evidence:
                if evidence.kind == "table" and evidence.table in feature.tables:
                    row_tables.add(evidence.table)
                    sources.append(_source(evidence.path, evidence.line))
            for table, (write_line, read_line) in table_accesses.get(path, {}).items():
                if table not in feature.tables:
                    continue
                row_tables.add(table)
                if write_line is not None:
                    sources.append(_source(path, write_line))
                if read_line is not None:
                    sources.append(_source(path, read_line))
            facts = component_facts.get(path)
            if facts is not None:
                for external in facts.externals:
                    row_services.add(external.service)
                    sources.append(_source(path, external.line))
        if not row_tables and not row_services:
            continue
        label = f"{_humanize(PurePosixPath(directory).name)} (library)"
        rows.append(
            _LibraryRow(
                label=label,
                tables=tuple(sorted(row_tables)),
                services=tuple(sorted(row_services)),
                sources=tuple(_unique_sources(sources or [fallback])),
            )
        )
    return tuple(rows)


def _feature_block(
    feature: Feature,
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]],
    primary_features: Mapping[str, tuple[str, str]],
    component_catalog: Sequence[Mapping[str, Any]],
) -> _FeatureBlock:
    callers = _feature_callers(
        feature,
        component_facts,
        component_imports,
        primary_features,
    )
    projection = project_component_diagram(
        feature.files,
        component_catalog=component_catalog,
        component_facts=component_facts,
        component_imports=component_imports,
        table_accesses=table_accesses,
        scope_tables=feature.tables,
        external_callers={caller.path: caller.targets for caller in callers},
        external_caller_hops={caller.path: caller.hops for caller in callers},
    )
    return _FeatureBlock(
        diagram=projection.diagram,
        tables_by_node=projection.tables_by_node,
        services_by_node=projection.services_by_node,
        kinds_by_node=projection.kinds_by_node,
        callers=callers,
        library_rows=_library_rows_for_feature(
            feature,
            projection.diagram.unreached_files,
            tables,
            component_facts,
            table_accesses,
        ),
    )


_RELATION_PRIORITY = {
    "fk": 0,
    "primaryjoin": 1,
    "comparison": 2,
    "id_column": 3,
    "commented_fk": 4,
}
_RELATION_LABELS = {
    "fk": ("model ForeignKey", "symbolic"),
    "primaryjoin": ("ORM join", "symbolic"),
    "comparison": ("query join", "symbolic"),
    "id_column": ("ID column name", "symbolic"),
    "commented_fk": ("commented-out FK", "symbolic"),
}


def _table_relations(links: Collection[TableLink]) -> list[Relation]:
    strongest: dict[tuple[str, str], TableLink] = {}
    for link in links:
        if link.signal == "same_file" or link.signal.endswith(":ambiguous"):
            continue
        pair = tuple(sorted((link.a, link.b)))
        current = strongest.get(pair)
        candidate_key = (
            _RELATION_PRIORITY.get(link.signal, 5),
            link.path,
            link.line,
            link.signal,
        )
        current_key = (
            (
                _RELATION_PRIORITY.get(current.signal, 5),
                current.path,
                current.line,
                current.signal,
            )
            if current is not None
            else None
        )
        if current is None or candidate_key < current_key:
            strongest[pair] = link
    relations = []
    for (src, dst), link in sorted(strongest.items()):
        label, kind = _RELATION_LABELS.get(
            link.signal,
            (link.signal, "symbolic"),
        )
        relations.append(
            Relation(
                src=src,
                dst=dst,
                label=label,
                kind=kind,
                source=_source(link.path, link.line),
            )
        )
    return relations


def _feature_relations(feature: Feature) -> list[Relation]:
    return _table_relations(feature.table_links)


def _feature_shared_notes(
    feature: Feature,
    feature_map: FeatureMap,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
) -> list[Claim]:
    paths = {file.path for file in feature.files}
    notes = []
    for other in feature_map.features:
        if other.id == feature.id:
            continue
        shared = paths & {file.path for file in other.files}
        if not shared:
            continue
        notes.append(
            Claim(
                text=f"Shares {len(shared)} files with {_feature_name(other, names)}.",
                sources=[
                    _relationship_source(
                        feature,
                        other,
                        shared,
                        tables[feature.id].source,
                    )
                ],
            )
        )
    if not notes:
        notes.append(
            Claim(
                text="Shares no files with other features.",
                sources=[tables[feature.id].source],
            )
        )
    return notes


def _feature_tldr_for_map(
    feature: Feature,
    feature_map: FeatureMap,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    block: _FeatureBlock,
) -> Tldr:
    result = _feature_tldr(feature, tables, component_facts, block)
    result.notes = _feature_shared_notes(feature, feature_map, names, tables)
    return result


def _feature_membership(
    feature: Feature,
    tables: Mapping[str, Table],
) -> list[MembershipGroup]:
    fallback = tables[feature.id].source
    groups = []
    for layer in LAYER_ORDER:
        files = sorted(
            (file for file in feature.files if layer_of(file.path) == layer),
            key=lambda file: (not file.primary, file.path),
        )
        if not files:
            continue
        rows = []
        for file in files:
            evidence = sorted(
                file.evidence,
                key=lambda item: (
                    item.kind,
                    item.path,
                    item.line,
                    item.to or "",
                    item.table or "",
                    item.tag or "",
                    item.target or "",
                ),
            )
            evidence_text = "; ".join(
                dict.fromkeys(_evidence_text(item) for item in evidence)
            )
            rows.append(
                MembershipRow(
                    path=file.path,
                    primary=file.primary,
                    evidence=evidence_text or "feature-map placement",
                    sources=_file_sources(file, fallback),
                )
            )
        groups.append(MembershipGroup(layer=LAYER_LABELS[layer], rows=rows))
    return groups


def _feature_page(
    feature: Feature,
    feature_map: FeatureMap,
    *,
    domain_title: str,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]],
    primary_features: Mapping[str, tuple[str, str]],
    component_catalog: Sequence[Mapping[str, Any]],
) -> Page:
    title = _feature_name(feature, names)
    domain_id = f"domain-{feature_map.domain}"
    page_id = f"feature-{feature_map.domain}-{feature.id.replace('_', '-')}"
    paths = sorted(file.path for file in feature.files if file.primary)
    if not paths:
        paths = [tables[feature.id].source.path]
    parent_source = tables[feature.id].source
    shared_ids = {
        candidate.id
        for candidate in feature_map.features
        if candidate.id != feature.id
        and {file.path for file in candidate.files}
        & {file.path for file in feature.files}
    }
    related = [
        Related(
            label=domain_title,
            page=domain_id,
            source=parent_source,
        )
    ]
    for other in feature_map.features:
        if other.id not in shared_ids:
            continue
        related.append(
            Related(
                label=_feature_name(other, names),
                page=f"feature-{feature_map.domain}-{other.id.replace('_', '-')}",
                source=_relationship_source(
                    feature,
                    other,
                    {file.path for file in feature.files}
                    & {file.path for file in other.files},
                    parent_source,
                ),
            )
        )
    block = _feature_block(
        feature,
        tables,
        component_facts,
        component_imports,
        table_accesses,
        primary_features,
        component_catalog,
    )
    return Page(
        id=page_id,
        title=title,
        kind="feature",
        commit=feature_map.commit,
        parent=domain_id,
        paths=paths,
        membership=_feature_membership(feature, tables),
        tldr=_feature_tldr_for_map(
            feature,
            feature_map,
            names,
            tables,
            component_facts,
            block,
        ),
        block=block.diagram,
        data=DataModel(
            sql_tables=feature.tables,
            relations=_feature_relations(feature),
        ),
        sequences=block_sequences(block.diagram),
        states=[],
        related=related,
    )


def _domain_calls(
    feature_map: FeatureMap,
    primary_owner: Mapping[str, str],
) -> list[tuple[str, str, int, str, str]]:
    calls = {}
    for feature in feature_map.features:
        for file in feature.files:
            source_feature = primary_owner.get(file.path)
            if source_feature is None:
                continue
            for evidence in file.evidence:
                if evidence.kind != "call" or evidence.to is None:
                    continue
                target_feature = primary_owner.get(evidence.to)
                if target_feature is None or target_feature == source_feature:
                    continue
                calls[
                    (
                        source_feature,
                        target_feature,
                        evidence.line,
                        evidence.path,
                        evidence.to,
                    )
                ] = None
    return sorted(calls)


def _domain_tldr(
    feature_map: FeatureMap,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    primary_features: Mapping[str, tuple[str, str]],
    domain_titles: Mapping[str, str],
    table_domains: Mapping[str, str],
) -> Tldr:
    table_names = sorted(
        {table for feature in feature_map.features for table in feature.tables}
    )
    table_sources = _unique_sources([tables[name].source for name in table_names])
    endpoint_records = {}
    worker_records = {}
    web_sources = {}
    external_records = {}
    feature_metrics = {}
    reached_metrics = {}
    reached_sources = []
    for feature in feature_map.features:
        facts = _feature_facts(feature, component_facts)
        endpoints, workers, externals = _runtime_records(facts)
        feature_metrics[feature.id] = (endpoints, workers, externals)
        callers = _feature_callers(
            feature,
            component_facts,
            component_imports,
            primary_features,
        )
        caller_endpoints, caller_workers = _caller_records(callers, component_facts)
        unique_caller_endpoints = {
            (path, endpoint.method, endpoint.path, endpoint.handler, endpoint.line)
            for path, endpoint in caller_endpoints
        }
        unique_caller_workers = {
            (path, worker.kind, worker.name, worker.line)
            for path, worker in caller_workers
        }
        reached_metrics[feature.id] = len(unique_caller_endpoints) + len(
            unique_caller_workers
        )
        for path, endpoint in caller_endpoints:
            reached_sources.append(_source(path, endpoint.line))
        for path, worker in caller_workers:
            reached_sources.append(_source(path, worker.line))
        for caller in callers:
            reached_sources.extend(caller.import_sources)
        for path, endpoint in endpoints:
            endpoint_records[
                (path, endpoint.method, endpoint.path, endpoint.handler, endpoint.line)
            ] = _source(path, endpoint.line)
        for path, worker in workers:
            worker_records[(path, worker.kind, worker.name, worker.line)] = _source(
                path, worker.line
            )
        for path in facts:
            if path.startswith("web/"):
                web_sources[path] = _file_evidence_source(
                    next(file for file in feature.files if file.path == path),
                    tables[feature.id].source,
                )
        for path, external in externals:
            external_records[
                (path, external.service, external.module, external.line)
            ] = _source(path, external.line)
    runtime_sources = _unique_sources(
        [
            *endpoint_records.values(),
            *worker_records.values(),
            *web_sources.values(),
            *external_records.values(),
            *table_sources,
        ]
    )
    summary = [
        Claim(
            text=(
                f"{len(feature_map.features)} features with "
                f"{len(endpoint_records)} HTTP endpoints, "
                f"{len(worker_records)} workers and workflows, "
                f"{len(web_sources)} web screens."
            ),
            sources=runtime_sources,
        )
    ]
    all_services = sorted({service for _, service, _, _ in external_records})
    if all_services:
        summary.append(
            Claim(
                text=f"Outside services: {', '.join(all_services)}.",
                sources=_unique_sources(list(external_records.values())),
            )
        )
    largest = sorted(
        feature_map.features,
        key=lambda feature: (-len(feature.files), feature.id),
    )[:3]
    key_points = [
        Claim(
            text=(f"{_feature_name(feature, names)}: {len(feature.files)} files."),
            sources=[
                _file_evidence_source(
                    min(
                        (file for file in feature.files if file.primary),
                        key=lambda file: file.path,
                    ),
                    tables[feature.id].source,
                )
                if any(file.primary for file in feature.files)
                else tables[feature.id].source
            ],
        )
        for feature in largest
    ]
    rows = []
    links = []
    row_sources = list(reached_sources)
    for row_number, feature in enumerate(feature_map.features):
        endpoints, workers, externals = feature_metrics[feature.id]
        services = sorted({external.service for _, external in externals})
        rows.append(
            [
                _feature_name(feature, names),
                str(len(endpoints)),
                str(len(workers)),
                str(reached_metrics[feature.id]),
                str(len(feature.tables)),
                ", ".join(services) or "—",
            ]
        )
        links.append(
            SummaryTableLink(
                row=row_number,
                column=0,
                page=f"feature-{feature_map.domain}-{feature.id.replace('_', '-')}",
            )
        )
        row_sources.extend(_source(path, endpoint.line) for path, endpoint in endpoints)
        row_sources.extend(_source(path, worker.line) for path, worker in workers)
        row_sources.extend(_source(path, external.line) for path, external in externals)
        row_sources.extend(tables[table_name].source for table_name in feature.tables)
        if not feature.files:
            row_sources.append(tables[feature.id].source)
    notes = [
        Claim(
            text=f"{len(feature_map.uncovered)} in-scope files are uncovered.",
            sources=table_sources,
        )
    ]
    primary_owner = {
        file.path: feature.id
        for feature in feature_map.features
        for file in feature.files
        if file.primary
    }
    imports: defaultdict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for source_feature, target_feature, line, path, _target in _domain_calls(
        feature_map,
        primary_owner,
    ):
        imports[(source_feature, target_feature)].append((path, line))
    import_notes = sorted(
        imports.items(),
        key=lambda item: (
            -len(item[1]),
            _feature_name(
                next(
                    feature
                    for feature in feature_map.features
                    if feature.id == item[0][0]
                ),
                names,
            ),
            _feature_name(
                next(
                    feature
                    for feature in feature_map.features
                    if feature.id == item[0][1]
                ),
                names,
            ),
        ),
    )
    feature_by_id = {feature.id: feature for feature in feature_map.features}
    notes.extend(
        Claim(
            text=(
                f"{_feature_name(feature_by_id[source_feature], names)} imports "
                f"code from {_feature_name(feature_by_id[target_feature], names)} "
                f"({len(sources)} imports)."
            ),
            sources=[_source(*min(sources))],
        )
        for (source_feature, target_feature), sources in import_notes
    )
    external_rows = []
    external_links = []
    external_sources = []
    for link in feature_map.external_links:
        a_domain = table_domains.get(link.a)
        b_domain = table_domains.get(link.b)
        if (
            a_domain is None
            or b_domain is None
            or a_domain == b_domain
            or feature_map.domain not in {a_domain, b_domain}
        ):
            continue
        own_table, other_table = (
            (link.a, link.b) if a_domain == feature_map.domain else (link.b, link.a)
        )
        other_domain = b_domain if a_domain == feature_map.domain else a_domain
        external_rows.append(
            [
                own_table,
                other_table,
                domain_titles.get(other_domain, _humanize(other_domain)),
                _RELATION_LABELS.get(link.signal, (link.signal, "symbolic"))[0],
            ]
        )
        external_links.append(
            SummaryTableLink(
                row=len(external_rows) - 1,
                column=2,
                page=f"domain-{other_domain}",
            )
        )
        external_sources.append(_source(link.path, link.line))
    additional_tables = (
        [
            SummaryTable(
                title="Links to other domains",
                columns=["This table", "Other table", "Other domain", "Signal"],
                rows=external_rows,
                links=external_links,
                sources=_unique_sources(external_sources),
            )
        ]
        if external_rows
        else None
    )
    return Tldr(
        summary=summary,
        key_points=key_points,
        table=SummaryTable(
            columns=[
                "Feature",
                "Endpoints",
                "Workers",
                "Reached from",
                "Tables",
                "Outside services",
            ],
            rows=rows,
            links=links,
            sources=_unique_sources([*row_sources, *table_sources]),
        ),
        notes=notes,
        additional_tables=additional_tables,
    )


def _domain_data(
    feature_map: FeatureMap,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
) -> DataModel:
    relations = _table_relations(feature_map.cross_links)
    feature_by_table = {
        table: feature.id
        for feature in feature_map.features
        for table in feature.tables
    }
    relation_tables: defaultdict[str, set[str]] = defaultdict(set)
    for relation in relations:
        for table in (relation.src, relation.dst):
            feature_id = feature_by_table.get(table)
            if feature_id is not None:
                relation_tables[feature_id].add(table)
    domains = []
    for feature in feature_map.features:
        users: Counter[str] = Counter()
        for file in feature.files:
            for evidence in file.evidence:
                if evidence.kind == "table" and evidence.table is not None:
                    users[evidence.table] += 1
        key_tables = set(
            sorted(feature.tables, key=lambda table: (-users[table], table))[:3]
        )
        key_tables.update(relation_tables[feature.id])
        domains.append(
            SchemaDomain(
                id=feature.id,
                title=_feature_name(feature, names),
                tables=feature.tables,
                key_tables=sorted(
                    key_tables,
                    key=lambda table: (-users[table], table),
                ),
                page=f"feature-{feature_map.domain}-{feature.id.replace('_', '-')}",
            )
        )
    return DataModel(domains=domains, relations=relations)


def _domain_component_diagram(
    feature_map: FeatureMap,
    *,
    component_catalog: Sequence[Mapping[str, Any]],
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]],
    primary_features: Mapping[str, tuple[str, str]],
) -> BlockDiagram:
    files_by_path = {}
    for file in sorted(
        (file for feature in feature_map.features for file in feature.files),
        key=lambda item: (not item.primary, item.path),
    ):
        files_by_path.setdefault(file.path, file)
    scope_paths = set(files_by_path)
    external_callers: defaultdict[str, set[str]] = defaultdict(set)
    external_caller_hops = {}
    for feature in feature_map.features:
        for caller in _feature_callers(
            feature,
            component_facts,
            component_imports,
            primary_features,
        ):
            if caller.path not in scope_paths:
                external_callers[caller.path].update(caller.targets)
                external_caller_hops[caller.path] = min(
                    external_caller_hops.get(caller.path, caller.hops),
                    caller.hops,
                )
    return project_component_diagram(
        tuple(files_by_path[path] for path in sorted(files_by_path)),
        component_catalog=component_catalog,
        component_facts=component_facts,
        component_imports=component_imports,
        table_accesses=table_accesses,
        scope_tables=feature_map.tables,
        external_callers=external_callers,
        external_caller_hops=external_caller_hops,
    ).diagram


def _domain_page(
    feature_map: FeatureMap,
    *,
    root_id: str,
    domain_title: str,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]],
    primary_features: Mapping[str, tuple[str, str]],
    domain_titles: Mapping[str, str],
    table_domains: Mapping[str, str],
    component_catalog: Sequence[Mapping[str, Any]],
) -> Page:
    domain_page_id = f"domain-{feature_map.domain}"
    primary_paths = sorted(
        {
            file.path
            for feature in feature_map.features
            for file in feature.files
            if file.primary
        }
    )
    if not primary_paths:
        primary_paths = [tables[feature_map.features[0].id].source.path]
    block = _domain_component_diagram(
        feature_map,
        component_catalog=component_catalog,
        component_facts=component_facts,
        component_imports=component_imports,
        table_accesses=table_accesses,
        primary_features=primary_features,
    )
    table_sources = _unique_sources(
        [tables[name].source for name in feature_map.tables]
    )
    root_source = table_sources[0]
    return Page(
        id=domain_page_id,
        title=domain_title,
        kind="domain",
        commit=feature_map.commit,
        parent=root_id,
        paths=primary_paths,
        tldr=_domain_tldr(
            feature_map,
            names,
            tables,
            component_facts,
            component_imports,
            primary_features,
            domain_titles,
            table_domains,
        ),
        block=block,
        block_figures=[],
        data=_domain_data(feature_map, names, tables),
        sequences=block_sequences(block),
        states=[],
        related=[
            Related(
                label="rox-core",
                page=root_id,
                source=root_source,
            )
        ],
    )


def feature_pages(
    feature_map: FeatureMap,
    *,
    root_id: str,
    domain_title: str,
    names: Mapping[str, str],
    tables: Mapping[str, Table],
    component_facts: Mapping[str, FileFacts] | None = None,
    component_imports: Mapping[str, Mapping[str, int]] | None = None,
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]]
    | None = None,
    primary_features: Mapping[str, tuple[str, str]] | None = None,
    domain_titles: Mapping[str, str] | None = None,
    table_domains: Mapping[str, str] | None = None,
    component_catalog: Sequence[Mapping[str, Any]] = (),
) -> list[Page]:
    runtime_facts = component_facts or {}
    runtime_imports = component_imports or {}
    runtime_table_accesses = table_accesses or {}
    primary_feature_owners = primary_features or {}
    pages = [
        _domain_page(
            feature_map,
            root_id=root_id,
            domain_title=domain_title,
            names=names,
            tables=tables,
            component_facts=runtime_facts,
            component_imports=runtime_imports,
            table_accesses=runtime_table_accesses,
            primary_features=primary_feature_owners,
            domain_titles=domain_titles or {},
            table_domains=table_domains or {},
            component_catalog=component_catalog,
        )
    ]
    pages.extend(
        _feature_page(
            feature,
            feature_map,
            domain_title=domain_title,
            names=names,
            tables=tables,
            component_facts=runtime_facts,
            component_imports=runtime_imports,
            table_accesses=runtime_table_accesses,
            primary_features=primary_feature_owners,
            component_catalog=component_catalog,
        )
        for feature in feature_map.features
    )
    return pages
