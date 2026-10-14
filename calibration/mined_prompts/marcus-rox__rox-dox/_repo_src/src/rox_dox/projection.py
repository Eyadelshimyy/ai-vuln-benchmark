from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import PurePosixPath
from typing import Any

from rox_dox.components import (
    Endpoint,
    FileFacts,
    TaskConsumer,
    TaskType,
    Worker,
    collect_task_facts,
)
from rox_dox.features import FeatureEvidence, FeatureFile
from rox_dox.model import BlockDiagram, Claim, CodeSource, Edge, Group, Node

MAX_FLOW_NOTES = 6
MAX_SOURCES_PER_EDGE = 3
TEMPORAL_WORKFLOW_DETAIL_LIMIT = 2
EARLY_COLUMNS = {"Callers", "HTTP services", "Workflows"}


@dataclass(frozen=True)
class ComponentProjection:
    diagram: BlockDiagram
    tables_by_node: Mapping[str, tuple[str, ...]]
    services_by_node: Mapping[str, tuple[str, ...]]
    kinds_by_node: Mapping[str, str]


@dataclass(frozen=True)
class _FlowFact:
    action: str
    backward: bool
    sources: tuple[CodeSource, ...]


def _source(path: str, line: int) -> CodeSource:
    return CodeSource(path=path, lines=(line, line))


def _source_key(source: CodeSource) -> tuple[str, int, int]:
    return source.path, source.lines[0], source.lines[1]


def _first_source(sources: Collection[CodeSource]) -> CodeSource:
    return min(sources, key=_source_key)


def _unique_sources(sources: Iterable[CodeSource]) -> list[CodeSource]:
    unique = {_source_key(source): source for source in sources}
    return [unique[key] for key in sorted(unique)]


def _flow_action(
    label: str,
    target_label: str,
    target_id: str,
) -> str | None:
    if label == "REST calls":
        return f"make REST calls to {target_label}"
    if label == "provider events":
        return f"send provider events to {target_label}"
    if label == "enqueue tasks":
        if target_id == "sqs":
            return "enqueue SQS tasks"
        return f"enqueue tasks to {target_label}"
    if label == "long-poll":
        return f"long-poll {target_label.removesuffix(' queues')}"
    if label == "start workflow":
        if "Temporal" in target_label:
            return "start Temporal workflows"
        return f"start workflows on {target_label}"
    if label == "API calls":
        return f"make API calls to {target_label}"
    if label.startswith("calls "):
        return f"{label} through {target_label}"
    if label == "reads + writes":
        return f"read + write {target_label}"
    if label == "reads":
        return f"read {target_label}"
    if label == "writes":
        return f"write {target_label}"
    return None


def _singular_flow_action(action: str, *, plural: bool) -> str:
    if plural:
        return action
    if action.startswith("read + write "):
        return action.replace("read + write ", "reads + writes ", 1)
    verb, separator, remainder = action.partition(" ")
    singular_verbs = {
        "call": "calls",
        "enqueue": "enqueues",
        "long-poll": "long-polls",
        "make": "makes",
        "read": "reads",
        "send": "sends",
        "start": "starts",
        "write": "writes",
    }
    return f"{singular_verbs.get(verb, verb)}{separator}{remainder}"


def _join_flow_phrases(phrases: Sequence[str]) -> str:
    if len(phrases) < 2:
        return phrases[0] if phrases else ""
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def _flow_action_order(action: str) -> tuple[int, str]:
    if action.startswith("long-poll "):
        return 0, action
    if action.startswith(("read ", "write ")):
        return 1, action
    if action.startswith("enqueue "):
        return 2, action
    if action.startswith("start "):
        return 3, action
    if action.startswith(("make ", "send ", "call ")):
        return 4, action
    return 5, action


def _flow_action_text(actions: Collection[str], *, plural: bool) -> str:
    return _join_flow_phrases(
        [
            _singular_flow_action(action, plural=plural)
            for action in sorted(actions, key=_flow_action_order)
        ]
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _catalog_source(entry: Mapping[str, Any]) -> CodeSource | None:
    source = entry.get("source")
    if not isinstance(source, Mapping):
        return None
    path = source.get("path")
    lines = source.get("lines")
    if isinstance(path, str) and isinstance(lines, list) and len(lines) == 2:
        return CodeSource(path=path, lines=(int(lines[0]), int(lines[1])))
    return None


def _catalog_ids_by_match(
    catalog: Sequence[Mapping[str, Any]],
    match_key: str,
) -> dict[str, str]:
    return {
        str(value): str(entry["id"])
        for entry in catalog
        for value in entry.get("match", {}).get(match_key, [])
    }


def _provider_api_ids(
    catalog_by_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    return {
        component_id
        for component_id, entry in catalog_by_id.items()
        if entry.get("match", {}).get("external_services")
    }


def _task_component_id(
    task: TaskType,
    queue_class_components: Mapping[str, str],
    target_components: Mapping[str, str],
) -> str | None:
    if task.queue_class in queue_class_components:
        return queue_class_components[task.queue_class]
    if task.deploy_target is not None:
        return target_components.get(task.deploy_target)
    return None


def _reachable_import_paths(
    path: str,
    imports: Mapping[str, Mapping[str, int]],
) -> set[str]:
    reachable = {path}
    pending = deque([path])
    while pending:
        current = pending.popleft()
        for target in sorted(imports.get(current, {})):
            if target in reachable:
                continue
            reachable.add(target)
            pending.append(target)
    return reachable


def _consumer_worker_paths(
    consumers: Collection[TaskConsumer],
    workers_by_name: Mapping[str, Collection[tuple[str, Worker]]],
    imports: Mapping[str, Mapping[str, int]],
) -> dict[tuple[str, str], str]:
    worker_paths_by_name = {
        name: {path for path, _ in records} for name, records in workers_by_name.items()
    }
    reachable_by_consumer_path = {}
    worker_path_by_consumer = {}
    for consumer in consumers:
        if consumer.path not in reachable_by_consumer_path:
            reachable_by_consumer_path[consumer.path] = _reachable_import_paths(
                consumer.path,
                imports,
            )
        worker_paths = worker_paths_by_name.get(consumer.executor, set())
        matches = worker_paths & reachable_by_consumer_path[consumer.path]
        if len(matches) == 1:
            worker_path_by_consumer[(consumer.path, consumer.executor)] = next(
                iter(matches)
            )
    return worker_path_by_consumer


def _endpoint_component_id(
    path: str,
    endpoint: Endpoint,
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    target_components: Mapping[str, str],
) -> str | None:
    component_id = (
        target_components.get(endpoint.deploy_target)
        if endpoint.deploy_target is not None
        else None
    )
    if (
        component_id is None
        or catalog_by_id.get(component_id, {}).get("kind") != "service"
    ):
        component_id = next(
            (
                str(entry["id"])
                for entry in catalog_by_id.values()
                if entry.get("kind") == "service"
                and any(
                    path.startswith(prefix)
                    for prefix in entry.get("match", {}).get("path_prefixes", [])
                )
            ),
            None,
        )
    return component_id


def _endpoint_component_ids(
    path: str,
    facts: FileFacts,
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    target_components: Mapping[str, str],
) -> set[str]:
    return {
        component_id
        for endpoint in facts.endpoints
        if (
            component_id := _endpoint_component_id(
                path,
                endpoint,
                catalog_by_id,
                target_components,
            )
        )
        is not None
    }


def _worker_component_ids(
    path: str,
    facts: FileFacts,
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    target_components: Mapping[str, str],
    queue_class_components: Mapping[str, str],
    consumers_by_executor: Mapping[str, Collection[TaskConsumer]],
    consumer_worker_paths: Mapping[tuple[str, str], str],
    task_types: Mapping[str, TaskType],
) -> set[str]:
    matches = set()
    for worker in facts.workers:
        for entry in catalog_by_id.values():
            if worker.kind in entry.get("match", {}).get("worker_kinds", []):
                matches.add(str(entry["id"]))
        if worker.kind != "task_executor":
            continue
        for consumer in consumers_by_executor.get(worker.name, []):
            if consumer_worker_paths.get((consumer.path, consumer.executor)) != path:
                continue
            task = task_types.get(consumer.task_type)
            if task is None:
                continue
            component_id = _task_component_id(
                task,
                queue_class_components,
                target_components,
            )
            if component_id is not None:
                matches.add(component_id)
    return matches


def _entry_component_ids(
    path: str,
    facts: FileFacts,
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    target_components: Mapping[str, str],
    queue_class_components: Mapping[str, str],
    consumers_by_executor: Mapping[str, Collection[TaskConsumer]],
    consumer_worker_paths: Mapping[tuple[str, str], str],
    task_types: Mapping[str, TaskType],
) -> set[str]:
    matches = set()
    for entry in catalog_by_id.values():
        if entry.get("match", {}).get("web_files") and path.startswith("web/"):
            matches.add(str(entry["id"]))
    matches.update(
        _endpoint_component_ids(path, facts, catalog_by_id, target_components)
    )
    matches.update(
        _worker_component_ids(
            path,
            facts,
            catalog_by_id,
            target_components,
            queue_class_components,
            consumers_by_executor,
            consumer_worker_paths,
            task_types,
        )
    )
    return matches


def _reachable_component_candidates(
    seeds: Collection[tuple[str, str, str, int]],
    scope_paths: set[str],
    imports: Mapping[str, Mapping[str, int]],
) -> dict[str, set[tuple[int, str, str]]]:
    reached_by_entry: dict[tuple[str, str], int] = {}
    candidates: defaultdict[str, set[tuple[int, str, str]]] = defaultdict(set)
    pending = deque(sorted(seeds))
    while pending:
        current, entry_path, component_id, distance = pending.popleft()
        key = current, entry_path
        if reached_by_entry.get(key, distance + 1) <= distance:
            continue
        reached_by_entry[key] = distance
        candidates[current].add((distance, component_id, entry_path))
        for target in sorted(set(imports.get(current, {})) & scope_paths):
            pending.append((target, entry_path, component_id, distance + 1))
    return candidates


def _nearest_component_owners(
    candidates: Mapping[str, Collection[tuple[int, str, str]]],
    catalog_order: Mapping[str, int],
) -> dict[str, str]:
    owners = {}
    for path, records in candidates.items():
        nearest_distance = min(distance for distance, _, _ in records)
        entry_counts: defaultdict[str, set[str]] = defaultdict(set)
        for distance, component_id, entry_path in records:
            if distance == nearest_distance:
                entry_counts[component_id].add(entry_path)
        owners[path] = min(
            entry_counts,
            key=lambda component_id: (
                -len(entry_counts[component_id]),
                catalog_order.get(component_id, len(catalog_order)),
                component_id,
            ),
        )
    return owners


def _nearest_component_owner_entries(
    candidates: Mapping[str, Collection[tuple[int, str, str]]],
    owners: Mapping[str, str],
) -> set[str]:
    entries = set()
    for path, records in candidates.items():
        owner_id = owners.get(path)
        if owner_id is None:
            continue
        nearest_distance = min(distance for distance, _, _ in records)
        entries.update(
            entry_path
            for distance, component_id, entry_path in records
            if distance == nearest_distance and component_id == owner_id
        )
    return entries


def _requires_scope_entry(entry: Mapping[str, Any]) -> bool:
    return entry.get("kind") in {"service", "service(many)"} or bool(
        entry.get("match", {}).get("worker_kinds")
    )


def _eligible_owner_candidates(
    candidates: Mapping[str, Collection[tuple[int, str, str]]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    entry_components: Collection[str],
    external_entry_components: Collection[str] = (),
) -> dict[str, set[tuple[int, str, str]]]:
    eligible_components = set(entry_components)
    eligible_components.update(external_entry_components)
    eligible_components.update(
        component_id
        for component_id, entry in catalog_by_id.items()
        if not _requires_scope_entry(entry)
    )
    return {
        path: {record for record in records if record[1] in eligible_components}
        for path, records in candidates.items()
        if any(record[1] in eligible_components for record in records)
    }


def _add_detail(
    details: defaultdict[str, defaultdict[str, list[CodeSource]]],
    node_id: str,
    text: str,
    source: CodeSource,
) -> None:
    details[node_id][text].append(source)


def _detail_claims(
    details: Mapping[str, Collection[CodeSource]],
    limit: int = 3,
) -> list[Claim]:
    items = [
        Claim(
            text=text,
            sources=_unique_sources(sources),
        )
        for text, sources in sorted(details.items())
        if sources
    ]
    if len(items) <= limit:
        return items
    overflow = items[limit:]
    overflow_sources = _unique_sources(
        source for item in overflow for source in item.sources
    )
    return [
        *items[:limit],
        Claim(text=f"+{len(overflow)} more", sources=overflow_sources),
    ]


def _deduplicate_edges(edges: Collection[Edge]) -> list[Edge]:
    grouped: defaultdict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in edges:
        grouped[(edge.src, edge.dst)].append(edge)
    result = []
    for (src, dst), records in sorted(grouped.items()):
        labels = {edge.label for edge in records}
        if labels <= {"reads", "writes", "reads + writes"}:
            modes = {
                mode
                for label in labels
                for mode in ("reads", "writes")
                if mode in label
            }
            label = "reads + writes" if len(modes) == 2 else next(iter(modes))
        elif all(label.startswith("calls ") for label in labels):
            services = sorted(
                {
                    service.strip()
                    for label in labels
                    for service in label.removeprefix("calls ").split(",")
                }
            )
            label = f"calls {', '.join(services)}"
        else:
            label = (
                next(iter(labels)) if len(labels) == 1 else " / ".join(sorted(labels))
            )
        result.append(
            Edge(
                src=src,
                dst=dst,
                label=label,
                source=_first_source(_unique_sources(edge.source for edge in records)),
            )
        )
    return result


def _prune_early_edges(
    edges: Collection[Edge],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    source_counts: Mapping[tuple[str, str], int],
) -> tuple[list[Edge], list[Edge]]:
    provider_api_ids = _provider_api_ids(catalog_by_id)
    catalog_order = {
        component_id: index for index, component_id in enumerate(catalog_by_id)
    }
    column_order = {}
    for entry in catalog_by_id.values():
        column = str(entry.get("column", ""))
        column_order.setdefault(column, len(column_order))

    kept = list(edges)
    pruned: list[Edge] = []

    worker_providers = {
        edge.dst
        for edge in kept
        if catalog_by_id.get(edge.src, {}).get("column") == "Background workers"
    }
    for edge in kept[:]:
        if (
            catalog_by_id.get(edge.src, {}).get("column") in EARLY_COLUMNS
            and edge.dst in provider_api_ids
            and edge.dst in worker_providers
        ):
            kept.remove(edge)
            pruned.append(edge)

    early_store_edges: defaultdict[str, list[Edge]] = defaultdict(list)
    for edge in kept:
        if (
            catalog_by_id.get(edge.src, {}).get("column") in EARLY_COLUMNS
            and catalog_by_id.get(edge.dst, {}).get("kind") == "store"
        ):
            early_store_edges[edge.dst].append(edge)
    for store_edges in early_store_edges.values():
        if len(store_edges) < 2:
            continue
        winner = min(
            store_edges,
            key=lambda edge: (
                -source_counts.get((edge.src, edge.dst), 1),
                catalog_order.get(edge.src, len(catalog_order)),
                edge.src,
                edge.label,
            ),
        )
        for edge in store_edges:
            if edge != winner:
                kept.remove(edge)
                pruned.append(edge)

    while True:
        active_nodes = {edge.src for edge in kept} | {edge.dst for edge in kept}
        if len(kept) <= ceil(1.5 * len(active_nodes)):
            break
        active_columns = {
            str(catalog_by_id.get(node_id, {}).get("column", ""))
            for node_id in active_nodes
        }
        skip_edges = []
        for edge in kept:
            source_column = str(catalog_by_id.get(edge.src, {}).get("column", ""))
            target_column = str(catalog_by_id.get(edge.dst, {}).get("column", ""))
            source_index = column_order.get(source_column, 0)
            target_index = column_order.get(target_column, 0)
            if (
                source_column not in EARLY_COLUMNS
                or target_index <= source_index + 1
                or not any(
                    column in active_columns and source_index < index < target_index
                    for column, index in column_order.items()
                )
            ):
                continue
            skip_edges.append(edge)
        if not skip_edges:
            break
        dropped = min(
            skip_edges,
            key=lambda edge: (
                source_counts.get((edge.src, edge.dst), 1),
                catalog_order.get(edge.src, len(catalog_order)),
                catalog_order.get(edge.dst, len(catalog_order)),
                edge.src,
                edge.dst,
                edge.label,
            ),
        )
        kept.remove(dropped)
        pruned.append(dropped)

    return kept, pruned


def _merge_node_group(
    nodes: Collection[Node],
    *,
    node_id: str,
    label: str,
    column: str,
    kind: str,
) -> Node:
    detail_sources: defaultdict[str, list[CodeSource]] = defaultdict(list)
    node_sources = []
    for node in nodes:
        node_sources.append(node.source)
        detail_sources[node.label].append(node.source)
        for detail in node.details:
            detail_sources[detail.text].extend(
                source for source in detail.sources if isinstance(source, CodeSource)
            )
    details = _detail_claims(detail_sources)
    return Node(
        id=node_id,
        label=label,
        source=_first_source(node_sources),
        kind=kind,
        group=f"column-{_slug(column)}",
        many=True,
        details=details,
    )


def _fit_node_budget(
    nodes: list[Node],
    edges: list[Edge],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Node], list[Edge], dict[str, str], dict[str, str]]:
    node_map = {}
    semantic_kinds = {
        node.id: str(catalog_by_id[node.id].get("kind", node.kind))
        for node in nodes
        if node.id in catalog_by_id
    }
    if len(nodes) > 15:
        provider_api_ids = _provider_api_ids(catalog_by_id)
        provider_nodes = [node for node in nodes if node.id in provider_api_ids]
        if provider_nodes:
            provider_ids = {node.id for node in provider_nodes}
            nodes = [node for node in nodes if node.id not in provider_ids]
            provider_column = str(catalog_by_id[provider_nodes[0].id].get("column", ""))
            nodes.append(
                _merge_node_group(
                    provider_nodes,
                    node_id="provider-apis",
                    label="Provider APIs",
                    column=provider_column,
                    kind="external",
                )
            )
            node_map.update({node_id: "provider-apis" for node_id in provider_ids})
            semantic_kinds["provider-apis"] = "external"
            edges = _deduplicate_edges(
                [
                    Edge(
                        src=node_map.get(edge.src, edge.src),
                        dst=node_map.get(edge.dst, edge.dst),
                        label=edge.label,
                        source=edge.source,
                    )
                    for edge in edges
                ]
            )
    if len(nodes) > 15:
        worker_nodes = [
            node
            for node in nodes
            if catalog_by_id.get(node.id, {}).get("column") == "Background workers"
        ]
        if worker_nodes:
            worker_ids = {node.id for node in worker_nodes}
            nodes = [node for node in nodes if node.id not in worker_ids]
            nodes.append(
                _merge_node_group(
                    worker_nodes,
                    node_id="background-workers",
                    label="Background workers",
                    column="Background workers",
                    kind="component",
                )
            )
            node_map.update({node_id: "background-workers" for node_id in worker_ids})
            semantic_kinds["background-workers"] = "service(many)"
            edges = _deduplicate_edges(
                [
                    Edge(
                        src=node_map.get(edge.src, edge.src),
                        dst=node_map.get(edge.dst, edge.dst),
                        label=edge.label,
                        source=edge.source,
                    )
                    for edge in edges
                ]
            )
    if len(nodes) > 15:
        raise ValueError(f"component diagram has {len(nodes)} boxes; maximum is 15")
    return nodes, edges, node_map, semantic_kinds


def _flow_notes(
    nodes: Collection[Node],
    edges: Collection[Edge],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    backward_edges: Collection[Edge] = (),
    pruned_edges: Collection[Edge] = (),
    sources_by_pair: Mapping[tuple[str, str], Collection[CodeSource]] | None = None,
) -> list[Claim]:
    nodes_by_id = {node.id: node for node in nodes}
    sources_by_pair = sources_by_pair or {}
    columns = {}
    for entry in catalog_by_id.values():
        column = str(entry.get("column", ""))
        columns.setdefault(column, len(columns))

    facts_by_component: defaultdict[str, list[_FlowFact]] = defaultdict(list)
    flow_edges = [(edge, False) for edge in edges]
    flow_edges.extend((edge, False) for edge in pruned_edges)
    flow_edges.extend((edge, True) for edge in backward_edges)
    for edge, backward in flow_edges:
        edge_sources = tuple(
            _unique_sources(
                sources_by_pair.get((edge.src, edge.dst), [edge.source])
                or [edge.source]
            )[:MAX_SOURCES_PER_EDGE]
        )
        if edge.label == "long-poll" and edge.src == "sqs":
            owner_id = edge.dst
            queue_node = nodes_by_id.get(edge.src)
            queue_entry = catalog_by_id.get(edge.src, {})
            queue_label = (
                queue_node.label
                if queue_node is not None
                else str(queue_entry.get("label", edge.src))
            ).removesuffix(" queues")
            facts_by_component[owner_id].append(
                _FlowFact(
                    action=f"long-poll {queue_label}",
                    backward=False,
                    sources=edge_sources,
                )
            )
            continue
        target_node = nodes_by_id.get(edge.dst)
        target_entry = catalog_by_id.get(edge.dst, {})
        target_label = (
            target_node.label
            if target_node is not None
            else str(target_entry.get("label", edge.dst))
        )
        for label in edge.label.split(" / "):
            if backward and label == "enqueue tasks":
                action = "enqueue follow-up SQS tasks"
            elif backward and label == "start workflow":
                action = "start Temporal workflows"
            else:
                action = _flow_action(label, target_label, edge.dst)
            if action:
                facts_by_component[edge.src].append(
                    _FlowFact(
                        action=action,
                        backward=backward,
                        sources=edge_sources,
                    )
                )

    node_order = {node.id: index for index, node in enumerate(nodes)}
    catalog_order = {
        component_id: index for index, component_id in enumerate(catalog_by_id)
    }
    component_ids = sorted(
        facts_by_component,
        key=lambda component_id: (
            columns.get(
                str(catalog_by_id.get(component_id, {}).get("column", "")),
                len(columns),
            ),
            node_order.get(
                component_id,
                len(node_order) + catalog_order.get(component_id, len(catalog_order)),
            ),
        ),
    )
    labels_by_component = {}
    plural_by_component = {}
    components_by_column: defaultdict[str, list[str]] = defaultdict(list)
    for component_id in component_ids:
        entry = catalog_by_id.get(component_id, {})
        node = nodes_by_id.get(component_id)
        label = (
            node.label if node is not None else str(entry.get("label", component_id))
        )
        if entry.get("kind") == "service(many)" and not label.endswith("workers"):
            label = f"{label} workers"
        labels_by_component[component_id] = label
        plural_by_component[component_id] = entry.get(
            "kind"
        ) == "service(many)" or label.casefold().endswith(
            ("callers", "clients", "services", "workers", "workflows")
        )
        components_by_column[str(entry.get("column", ""))].append(component_id)

    notes = []
    for column, column_components in sorted(
        components_by_column.items(),
        key=lambda item: columns.get(item[0], len(columns)),
    ):
        forward_by_component = {
            component_id: {
                fact.action
                for fact in facts_by_component[component_id]
                if not fact.backward
            }
            for component_id in column_components
        }
        backward_by_component = {
            component_id: {
                fact.action
                for fact in facts_by_component[component_id]
                if fact.backward
            }
            for component_id in column_components
        }
        action_sets = [
            frozenset(
                (fact.backward, fact.action)
                for fact in facts_by_component[component_id]
            )
            for component_id in column_components
        ]
        same_actions = all(action_set == action_sets[0] for action_set in action_sets)
        if same_actions:
            plural = (
                len(column_components) > 1 or plural_by_component[column_components[0]]
            )
            subject = (
                labels_by_component[column_components[0]]
                if len(column_components) == 1
                else column
            )
            forward_actions = forward_by_component[column_components[0]]
            backward_actions = backward_by_component[column_components[0]]
            if forward_actions:
                text = f"{subject} {_flow_action_text(forward_actions, plural=plural)}"
                if backward_actions:
                    text += (
                        ", and also "
                        f"{_flow_action_text(backward_actions, plural=plural)}"
                    )
                text += "."
            else:
                text = (
                    f"{subject} also "
                    f"{_flow_action_text(backward_actions, plural=plural)}."
                )
        else:
            common_forward = set.intersection(
                *(set(actions) for actions in forward_by_component.values())
            )
            common_backward = set.intersection(
                *(set(actions) for actions in backward_by_component.values())
            )
            forward_clauses = []
            if common_forward:
                common_subject = _join_flow_phrases(
                    [
                        labels_by_component[component_id]
                        for component_id in column_components
                    ]
                )
                forward_clauses.append(
                    f"{common_subject} "
                    f"{_flow_action_text(common_forward, plural=len(column_components) > 1)}"
                )
            else:
                for component_id in column_components:
                    actions = forward_by_component[component_id]
                    if actions:
                        forward_clauses.append(
                            f"{labels_by_component[component_id]} "
                            f"{_flow_action_text(actions, plural=plural_by_component[component_id])}"
                        )
            if common_forward:
                for component_id in column_components:
                    actions = forward_by_component[component_id] - common_forward
                    if actions:
                        forward_clauses.append(
                            f"{labels_by_component[component_id]} also "
                            f"{_flow_action_text(actions, plural=plural_by_component[component_id])}"
                        )
            text = "; ".join(forward_clauses)
            if common_backward:
                if common_forward and len(forward_clauses) == 1:
                    text += (
                        ", and also "
                        f"{_flow_action_text(common_backward, plural=len(column_components) > 1)}"
                    )
                elif text:
                    subject = _join_flow_phrases(
                        [
                            labels_by_component[component_id]
                            for component_id in column_components
                        ]
                    )
                    text += (
                        f"; and also {subject} "
                        f"{_flow_action_text(common_backward, plural=len(column_components) > 1)}"
                    )
                else:
                    subject = _join_flow_phrases(
                        [
                            labels_by_component[component_id]
                            for component_id in column_components
                        ]
                    )
                    text = (
                        f"{subject} also "
                        f"{_flow_action_text(common_backward, plural=len(column_components) > 1)}"
                    )
            for component_id in column_components:
                actions = backward_by_component[component_id] - common_backward
                if actions:
                    subject = labels_by_component[component_id]
                    phrase = _flow_action_text(
                        actions,
                        plural=plural_by_component[component_id],
                    )
                    text += (
                        f"; and also {subject} {phrase}"
                        if text
                        else f"{subject} also {phrase}"
                    )
            text += "."

        notes.append(
            Claim(
                text=text,
                sources=_unique_sources(
                    source
                    for component_id in column_components
                    for fact in facts_by_component[component_id]
                    for source in fact.sources
                ),
            )
        )

    assert len(notes) <= MAX_FLOW_NOTES
    return notes


def project_component_diagram(
    scope_files: Collection[FeatureFile],
    *,
    component_catalog: Sequence[Mapping[str, Any]],
    component_facts: Mapping[str, FileFacts],
    component_imports: Mapping[str, Mapping[str, int]],
    table_accesses: Mapping[str, Mapping[str, tuple[int | None, int | None]]],
    scope_tables: Collection[str],
    external_callers: Mapping[str, Collection[str]],
    external_caller_hops: Mapping[str, int] | None = None,
) -> ComponentProjection:
    catalog_by_id = {str(entry["id"]): entry for entry in component_catalog}
    target_components = _catalog_ids_by_match(component_catalog, "deploy_targets")
    queue_class_components = _catalog_ids_by_match(
        component_catalog,
        "queue_type_classes",
    )
    external_components = _catalog_ids_by_match(
        component_catalog,
        "external_services",
    )
    task_facts = collect_task_facts(component_facts)
    task_types = {task.name: task for task in task_facts.task_types}
    consumers_by_executor: defaultdict[str, list[TaskConsumer]] = defaultdict(list)
    workers_by_name: defaultdict[str, list[tuple[str, Worker]]] = defaultdict(list)
    for consumer in task_facts.consumers:
        consumers_by_executor[consumer.executor].append(consumer)
    for path, facts in component_facts.items():
        for worker in facts.workers:
            if worker.kind == "task_executor":
                workers_by_name[worker.name].append((path, worker))
    consumer_worker_paths = _consumer_worker_paths(
        task_facts.consumers,
        workers_by_name,
        component_imports,
    )
    files_by_path = {
        file.path: file for file in sorted(scope_files, key=lambda item: item.path)
    }
    scope_paths = set(files_by_path)
    scope_table_names = set(scope_tables)
    facts_by_path = {
        path: component_facts.get(
            path,
            FileFacts(
                path=path,
                api_group=None,
                endpoints=[],
                workers=[],
                externals=[],
            ),
        )
        for path in scope_paths
    }

    catalog_order = {
        str(entry["id"]): index for index, entry in enumerate(component_catalog)
    }
    entry_components_by_path = {
        path: _entry_component_ids(
            path,
            facts,
            catalog_by_id,
            target_components,
            queue_class_components,
            consumers_by_executor,
            consumer_worker_paths,
            task_types,
        )
        for path, facts in facts_by_path.items()
    }
    entry_components = {
        path: min(
            component_ids,
            key=lambda component_id: (
                catalog_order.get(component_id, len(catalog_order)),
                component_id,
            ),
        )
        for path, component_ids in entry_components_by_path.items()
        if component_ids
    }
    entry_candidates = _reachable_component_candidates(
        [
            (path, path, component_id, 0)
            for path, component_id in entry_components.items()
        ],
        scope_paths,
        component_imports,
    )
    entry_candidates = _eligible_owner_candidates(
        entry_candidates,
        catalog_by_id,
        entry_components.values(),
    )
    owners_by_path = _nearest_component_owners(entry_candidates, catalog_order)
    endpoint_seeds = []
    worker_seeds = []
    endpoint_components = set()
    worker_components = set()
    endpoint_fallback_candidates = []
    scope_directories = {PurePosixPath(path).parent for path in scope_paths}
    for caller_path, targets in sorted(external_callers.items()):
        if caller_path in scope_paths:
            continue
        caller_facts = component_facts.get(caller_path)
        if caller_facts is None:
            continue
        target_paths = sorted(set(targets) & scope_paths)
        if not target_paths:
            continue
        caller_hops = (external_caller_hops or {}).get(caller_path, 1)
        if caller_facts.endpoints:
            caller_components = _endpoint_component_ids(
                caller_path,
                caller_facts,
                catalog_by_id,
                target_components,
            )
            seedable_components = [
                component_id
                for component_id in caller_components
                if catalog_by_id.get(component_id, {}).get("kind")
                in {"service", "service(many)"}
            ]
            if seedable_components:
                component_id = min(
                    seedable_components,
                    key=lambda candidate: (
                        catalog_order.get(candidate, len(catalog_order)),
                        candidate,
                    ),
                )
                endpoint_fallback_candidates.append(
                    (
                        caller_hops,
                        caller_path,
                        target_paths,
                        component_id,
                    )
                )
                if PurePosixPath(caller_path).parent in scope_directories:
                    endpoint_components.add(component_id)
                    endpoint_seeds.extend(
                        (target, caller_path, component_id, caller_hops)
                        for target in target_paths
                    )

        caller_worker_components = _worker_component_ids(
            caller_path,
            caller_facts,
            catalog_by_id,
            target_components,
            queue_class_components,
            consumers_by_executor,
            consumer_worker_paths,
            task_types,
        )
        seedable_worker_components = [
            component_id
            for component_id in caller_worker_components
            if _requires_scope_entry(catalog_by_id.get(component_id, {}))
        ]
        if seedable_worker_components:
            component_id = min(
                seedable_worker_components,
                key=lambda candidate: (
                    catalog_order.get(candidate, len(catalog_order)),
                    candidate,
                ),
            )
            worker_components.add(component_id)
            worker_seeds.extend(
                (target, caller_path, component_id, caller_hops)
                for target in target_paths
            )

    endpoint_candidates = _reachable_component_candidates(
        endpoint_seeds,
        scope_paths,
        component_imports,
    )
    endpoint_candidates = _eligible_owner_candidates(
        endpoint_candidates,
        catalog_by_id,
        entry_components.values(),
        endpoint_components,
    )
    endpoint_candidates = {
        path: records
        for path, records in endpoint_candidates.items()
        if path not in owners_by_path
    }
    endpoint_owners = _nearest_component_owners(endpoint_candidates, catalog_order)
    endpoint_owner_paths = _nearest_component_owner_entries(
        endpoint_candidates,
        endpoint_owners,
    )
    owners_by_path.update(endpoint_owners)

    worker_candidates = _reachable_component_candidates(
        worker_seeds,
        scope_paths,
        component_imports,
    )
    worker_candidates = _eligible_owner_candidates(
        worker_candidates,
        catalog_by_id,
        entry_components.values(),
        worker_components,
    )
    worker_candidates = {
        path: records
        for path, records in worker_candidates.items()
        if path not in owners_by_path
    }
    worker_owners = _nearest_component_owners(worker_candidates, catalog_order)
    owners_by_path.update(worker_owners)
    external_worker_paths = _nearest_component_owner_entries(
        worker_candidates,
        worker_owners,
    )

    fallback_endpoint_paths = set()
    if not owners_by_path and endpoint_fallback_candidates:
        minimum_hops = min(hops for hops, _, _, _ in endpoint_fallback_candidates)
        fallback_seeds = []
        fallback_components = set()
        for hops, caller_path, targets, component_id in endpoint_fallback_candidates:
            if hops != minimum_hops:
                continue
            fallback_components.add(component_id)
            fallback_seeds.extend(
                (target, caller_path, component_id, hops) for target in targets
            )
        fallback_candidates = _reachable_component_candidates(
            fallback_seeds,
            scope_paths,
            component_imports,
        )
        fallback_candidates = _eligible_owner_candidates(
            fallback_candidates,
            catalog_by_id,
            entry_components.values(),
            fallback_components,
        )
        fallback_owners = _nearest_component_owners(fallback_candidates, catalog_order)
        fallback_endpoint_paths = _nearest_component_owner_entries(
            fallback_candidates,
            fallback_owners,
        )
        owners_by_path = fallback_owners
    unreached = sorted(scope_paths - owners_by_path.keys())

    edge_sources: defaultdict[tuple[str, str, str], list[CodeSource]] = defaultdict(
        list
    )
    component_table_sources: defaultdict[tuple[str, str, str], list[CodeSource]] = (
        defaultdict(list)
    )
    component_tables: defaultdict[str, set[str]] = defaultdict(set)
    component_services: defaultdict[str, set[str]] = defaultdict(set)
    provider_services: defaultdict[str, set[str]] = defaultdict(set)
    details: defaultdict[str, defaultdict[str, list[CodeSource]]] = defaultdict(
        lambda: defaultdict(list)
    )
    endpoint_groups: defaultdict[
        tuple[str, str], dict[tuple[str, str, str, str, int], CodeSource]
    ] = defaultdict(dict)
    task_names_by_component: defaultdict[str, set[str]] = defaultdict(set)
    queue_task_names: set[str] = set()
    service_endpoint_sources: defaultdict[str, list[CodeSource]] = defaultdict(list)
    webhook_id = target_components.get("WEBHOOK")
    http_clients_id = "http_clients" if "http_clients" in catalog_by_id else None
    provider_push_id = "provider_push" if "provider_push" in catalog_by_id else None
    temporal_id = "temporal" if "temporal" in catalog_by_id else None
    sqs_id = "sqs" if "sqs" in catalog_by_id else None
    postgres_id = "postgres" if "postgres" in catalog_by_id else None

    endpoint_facts_by_path = dict(facts_by_path)
    endpoint_facts_by_path.update(
        {
            path: component_facts[path]
            for path, targets in external_callers.items()
            if path in component_facts
            and path
            in endpoint_owner_paths | external_worker_paths | fallback_endpoint_paths
            and set(targets) & scope_paths
        }
    )
    for path, facts in sorted(endpoint_facts_by_path.items()):
        owner_id = owners_by_path.get(path)
        for endpoint in facts.endpoints:
            endpoint_component_id = _endpoint_component_id(
                path,
                endpoint,
                catalog_by_id,
                target_components,
            )
            if (
                endpoint_component_id is not None
                and catalog_by_id.get(endpoint_component_id, {}).get("kind")
                == "service"
            ):
                prefix = "/" + endpoint.path.strip("/").split("/", 1)[0]
                if prefix == "/":
                    prefix = "/"
                endpoint_groups[(endpoint_component_id, prefix)][
                    (
                        path,
                        endpoint.method,
                        endpoint.path,
                        endpoint.handler,
                        endpoint.line,
                    )
                ] = _source(path, endpoint.line)
                if (
                    catalog_by_id.get(endpoint_component_id, {}).get("kind")
                    == "service"
                ):
                    service_endpoint_sources[endpoint_component_id].append(
                        _source(path, endpoint.line)
                    )
            if (
                provider_push_id is not None
                and webhook_id is not None
                and endpoint.deploy_target == "WEBHOOK"
            ):
                edge_sources[(provider_push_id, webhook_id, "provider events")].append(
                    _source(path, endpoint.line)
                )

        if owner_id is None:
            continue
        for external in facts.externals:
            provider_id = external_components.get(external.service)
            if provider_id is None:
                continue
            source = _source(path, external.line)
            edge_sources[(owner_id, provider_id, "API calls")].append(source)
            component_services[owner_id].add(external.service)
            provider_services[provider_id].add(external.service)
        for line in facts.workflow_starts:
            if temporal_id is not None and owner_id != temporal_id:
                edge_sources[(owner_id, temporal_id, "start workflow")].append(
                    _source(path, line)
                )
        if owner_id == temporal_id:
            for worker in facts.workers:
                if worker.kind in {"temporal_workflow", "temporal_activity"}:
                    _add_detail(
                        details,
                        temporal_id,
                        worker.name,
                        _source(path, worker.line),
                    )
        for producer in facts.task_producers:
            if sqs_id is None:
                continue
            source = _source(producer.path, producer.line)
            edge_sources[(owner_id, sqs_id, "enqueue tasks")].append(source)
        for table, (write_line, read_line) in table_accesses.get(path, {}).items():
            if table not in scope_table_names:
                continue
            if write_line is not None:
                source = _source(path, write_line)
                component_table_sources[(owner_id, table, "writes")].append(source)
                component_tables[owner_id].add(table)
            if read_line is not None:
                source = _source(path, read_line)
                component_table_sources[(owner_id, table, "reads")].append(source)
                component_tables[owner_id].add(table)

    scope_worker_paths = {
        path
        for path, component_ids in entry_components_by_path.items()
        if component_ids
        and any(
            worker.kind == "task_executor" for worker in facts_by_path[path].workers
        )
    }
    eligible_worker_paths = scope_worker_paths | external_worker_paths
    for consumer in task_facts.consumers:
        task = task_types.get(consumer.task_type)
        if task is None:
            continue
        component_id = _task_component_id(
            task,
            queue_class_components,
            target_components,
        )
        if component_id is None or sqs_id is None:
            continue
        worker_path = consumer_worker_paths.get((consumer.path, consumer.executor))
        if worker_path is None or worker_path not in eligible_worker_paths:
            continue
        worker_records = [
            (path, worker)
            for path, worker in workers_by_name.get(consumer.executor, [])
            if path == worker_path
        ]
        if not worker_records:
            continue
        source = _source(consumer.path, consumer.line)
        queue_task_names.add(task.name)
        task_names_by_component[component_id].add(task.name)
        edge_sources[(sqs_id, component_id, "long-poll")].append(source)

    if sqs_id is not None:
        sqs_details: defaultdict[str, list[CodeSource]] = defaultdict(list)
        queue_names_by_label: defaultdict[str, set[str]] = defaultdict(set)
        for task_name in sorted(queue_task_names):
            task = task_types.get(task_name)
            if task is None:
                continue
            label = task.queue_class or "Unmapped"
            sqs_details[label].append(_source(task.path, task.line))
            queue_names_by_label[label].add(task_name)
        for label, names in sorted(queue_names_by_label.items()):
            _add_detail(
                details,
                sqs_id,
                f"{label} · {len(names)} task types",
                _first_source(sqs_details[label]),
            )

    component_table_keys = sorted(
        {(component_id, table) for component_id, table, _ in component_table_sources}
    )
    for component_id, table in component_table_keys:
        sources = [
            *component_table_sources.get((component_id, table, "reads"), []),
            *component_table_sources.get((component_id, table, "writes"), []),
        ]
        if postgres_id is not None and sources:
            component_tables[postgres_id].add(table)
            _add_detail(details, postgres_id, table, _first_source(sources))
    for component_id, task_names in task_names_by_component.items():
        tasks_by_target: defaultdict[str, dict[str, TaskType]] = defaultdict(dict)
        for task in task_facts.task_types:
            if task.name not in task_names:
                continue
            target = task.deploy_target or task.queue_class or "Unmapped"
            tasks_by_target[target][task.name] = task
        for target, tasks_by_name in sorted(tasks_by_target.items()):
            target_tasks = sorted(
                tasks_by_name.values(),
                key=lambda task: (task.path, task.line, task.name),
            )
            if not target_tasks:
                continue
            first_task = target_tasks[0]
            _add_detail(
                details,
                component_id,
                f"{target} · {len(tasks_by_name)} task types",
                _source(first_task.path, first_task.line),
            )

    for provider_id, services in provider_services.items():
        for service in services:
            _add_detail(
                details,
                provider_id,
                service,
                _first_source(
                    [
                        source
                        for (src, dst, label), sources in edge_sources.items()
                        if dst == provider_id and label == "API calls"
                        for source in sources
                    ]
                ),
            )
    route_evidence: defaultdict[str, dict[str, list[FeatureEvidence]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for file in scope_files:
        for evidence in file.evidence:
            if evidence.kind == "route" and evidence.tag is not None:
                route_evidence[evidence.tag][
                    "web" if evidence.path.startswith("web/") else "backend"
                ].append(evidence)
    route_service_ids = set()
    for groups in route_evidence.values():
        for web in groups.get("web", []):
            for backend in groups.get("backend", []):
                web_component_id = owners_by_path.get(web.path)
                service_id = owners_by_path.get(backend.path)
                if web_component_id is None or service_id is None:
                    continue
                if (
                    catalog_by_id.get(web_component_id, {}).get("kind") != "client"
                    or catalog_by_id.get(service_id, {}).get("kind") != "service"
                ):
                    continue
                route_service_ids.add(service_id)
                edge_sources[(web_component_id, service_id, "REST calls")].append(
                    _source(web.path, web.line)
                )
    if http_clients_id is not None:
        for service_id, sources in sorted(service_endpoint_sources.items()):
            if service_id == webhook_id or service_id in route_service_ids:
                continue
            edge_sources[(http_clients_id, service_id, "REST calls")].append(
                _first_source(sources)
            )

    for (component_id, table), modes in sorted(
        {
            (component_id, table): {
                mode
                for owner, name, mode in component_table_sources
                if owner == component_id and name == table
            }
            for component_id, table, _ in component_table_sources
        }.items()
    ):
        if postgres_id is None or not modes:
            continue
        label = "reads + writes" if len(modes) == 2 else next(iter(modes))
        edge_sources[(component_id, postgres_id, label)].extend(
            source
            for mode in modes
            for source in component_table_sources[(component_id, table, mode)]
        )

    sources_by_pair: defaultdict[tuple[str, str], list[CodeSource]] = defaultdict(list)
    for (src, dst, _), sources in edge_sources.items():
        sources_by_pair[(src, dst)].extend(sources)
    sources_by_pair = defaultdict(
        list,
        {pair: _unique_sources(sources) for pair, sources in sources_by_pair.items()},
    )
    source_counts = {pair: len(sources) for pair, sources in sources_by_pair.items()}
    edges = _deduplicate_edges(
        [
            Edge(src=src, dst=dst, label=label, source=_first_source(sources))
            for (src, dst, label), sources in edge_sources.items()
            if sources and src != dst
        ]
    )
    column_order = {}
    for entry in component_catalog:
        column = str(entry.get("column", ""))
        column_order.setdefault(column, len(column_order))
    backward_edges = []
    figure_edges = []
    for edge in edges:
        source_column = str(catalog_by_id.get(edge.src, {}).get("column", ""))
        target_column = str(catalog_by_id.get(edge.dst, {}).get("column", ""))
        if column_order.get(target_column, 0) < column_order.get(source_column, 0):
            backward_edges.append(edge)
        else:
            figure_edges.append(edge)
    edges, pruned_edges = _prune_early_edges(
        figure_edges,
        catalog_by_id,
        source_counts,
    )
    participating = {edge.src for edge in edges} | {edge.dst for edge in edges}

    endpoint_details: defaultdict[str, defaultdict[str, list[CodeSource]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for (component_id, prefix), records in endpoint_groups.items():
        endpoint_details[component_id][f"{prefix}: {len(records)} endpoints"].extend(
            records.values()
        )
    for component_id, detail_map in endpoint_details.items():
        for text, sources in detail_map.items():
            details[component_id][text].extend(sources)

    nodes = []
    semantic_kinds = {}
    for node_id in sorted(participating):
        entry = catalog_by_id.get(node_id, {})
        column = str(entry.get("column", ""))
        group_id = f"column-{_slug(column)}"
        node_details = _detail_claims(
            details.get(node_id, {}),
            limit=(
                TEMPORAL_WORKFLOW_DETAIL_LIMIT
                if node_id == temporal_id
                else 4
                if node_id == postgres_id
                else 3
            ),
        )
        incident_sources = [
            edge.source for edge in edges if edge.src == node_id or edge.dst == node_id
        ]
        source = _catalog_source(entry) or _first_source(incident_sources)
        nodes.append(
            Node(
                id=node_id,
                label=str(entry.get("label", node_id)),
                source=source,
                kind=str(entry.get("shape", "component")),
                group=group_id,
                many=bool(entry.get("many", False)),
                details=node_details,
            )
        )
        semantic_kinds[node_id] = str(entry.get("kind", "component"))

    notes = _flow_notes(
        nodes,
        edges,
        catalog_by_id,
        backward_edges=backward_edges,
        pruned_edges=pruned_edges,
        sources_by_pair=sources_by_pair,
    )
    nodes, edges, node_map, semantic_kinds = _fit_node_budget(
        nodes,
        edges,
        catalog_by_id,
    )
    table_values: defaultdict[str, set[str]] = defaultdict(set)
    service_values: defaultdict[str, set[str]] = defaultdict(set)
    for node_id, values in component_tables.items():
        table_values[node_map.get(node_id, node_id)].update(values)
    for node_id, values in component_services.items():
        service_values[node_map.get(node_id, node_id)].update(values)
    for node_id, values in provider_services.items():
        service_values[node_map.get(node_id, node_id)].update(values)
    table_map = {
        node_id: tuple(sorted(values)) for node_id, values in table_values.items()
    }
    service_map = {
        node_id: tuple(sorted(values)) for node_id, values in service_values.items()
    }
    column_indexes = {
        f"column-{_slug(column)}": index
        for index, column in enumerate(
            dict.fromkeys(str(entry.get("column", "")) for entry in component_catalog)
        )
    }
    nodes.sort(key=lambda node: (column_indexes.get(node.group or "", 0), node.id))
    edges.sort(key=lambda edge: (edge.src, edge.dst, edge.label))
    active_node_ids = {node.id for node in nodes}
    semantic_kinds = {
        node_id: kind
        for node_id, kind in semantic_kinds.items()
        if node_id in active_node_ids
    }
    groups = []
    for column in dict.fromkeys(
        str(entry.get("column", "")) for entry in component_catalog
    ):
        members = [node for node in nodes if node.group == f"column-{_slug(column)}"]
        if members:
            groups.append(
                Group(
                    id=f"column-{_slug(column)}",
                    label=column,
                    source=members[0].source,
                )
            )
    diagram = BlockDiagram(
        notes=notes,
        groups=groups,
        nodes=nodes,
        edges=edges,
        unreached_files=unreached,
    )
    return ComponentProjection(
        diagram=diagram,
        tables_by_node=table_map,
        services_by_node=service_map,
        kinds_by_node=semantic_kinds,
    )
