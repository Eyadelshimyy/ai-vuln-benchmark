"""Bounded protocol interaction graph.

Edges exist only when a call, storage link, or implementation link is
established. Similar names do not connect contracts. A path is a candidate,
not a verification, a reproduction, or proof of safety.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.discovery.sequences import (
    MAX_EXECUTIONS,
    ExplorationBounds,
    PlannedCall,
    PlannedSequence,
    clamp_bounds,
    describe_call,
)
from app.domain.evidence import Evidence, EvidenceKind
from app.parsing.solidity_arguments import ArgumentCandidate, combine, parameter_types
from app.parsing.solidity_economics import AssetDelta, EconomicResult, OracleStatus
from app.parsing.solidity_identity import canonical_function_key
from app.parsing.solidity_ir import SemanticCall, SemanticFunction, SemanticProgram
from app.parsing.solidity_spec import function_signature

MAX_PROTOCOL_CONTRACTS = 8
MAX_PROTOCOL_EDGES = 24
MAX_PROTOCOL_EXPANSION = 16
MAX_CALLBACK_DEPTH = 2

_FORBIDDEN = frozenset({"verified", "proved", "reproduced", "confirmed", "safe", "exploited"})
_RESOLVED = frozenset({"resolved", "interface"})
_CALLBACKS = frozenset(
    {
        "ontokenreceived",
        "tokensreceived",
        "onerc721received",
        "onerc1155received",
        "onerc1155batchreceived",
    }
)
_PRIORITY = {
    "storage_dependency": 0,
    "asset_movement": 1,
    "token_transfer": 1,
    "transfer_from": 1,
    "oracle_read": 2,
    "price_dependency": 2,
    "callback": 3,
    "authorization_dependency": 4,
    "delegatecall": 5,
    "proxy_implementation": 5,
    "approval": 6,
    "permit": 6,
    "staticcall": 7,
    "external_call": 8,
    "internal_call": 9,
    "event_state": 9,
}


@dataclass(frozen=True)
class ProtocolBounds:
    max_contracts: int = MAX_PROTOCOL_CONTRACTS
    max_edges: int = MAX_PROTOCOL_EDGES
    max_expansion: int = MAX_PROTOCOL_EXPANSION
    max_callback_depth: int = MAX_CALLBACK_DEPTH


@dataclass(frozen=True)
class StorageLink:
    declaration_id: str
    writer_function: str
    reader_function: str
    provenance: str
    source_file: str


@dataclass(frozen=True)
class ImplementationLink:
    proxy: str
    implementation: str
    provenance: str
    proxy_file: str
    implementation_file: str
    compiler_configuration: str = ""


@dataclass(frozen=True)
class EventLink:
    event: str
    function_identity: str
    declaration_id: str
    provenance: str


@dataclass(frozen=True)
class PriceLink:
    """Oracle value flowing through a computation into an asset operation.

    Two calls in one function are not this link.
    """

    oracle_operation: str
    computation_id: str
    valuation_id: str
    asset_operation: str
    provenance: str
    span: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrustBoundary:
    """Evidence that a privileged operation can be triggered across a trust boundary.

    A call into an authorized function is not enough.
    """

    caller_function: str
    callee_function: str
    call_id: str
    caller_controlled: str
    missing_authorization: str
    privileged_callee: str
    propagation: str
    relationship: str
    provenance: str


@dataclass(frozen=True)
class ContractNode:
    node_id: str
    project: str
    source_snapshot: str
    compiler_configuration: str
    source_file: str
    contract: str
    kind: str
    span: tuple[int, ...]
    address: str
    provenance: str
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionNode:
    function_id: str
    node_id: str
    contract: str
    name: str
    signature: str
    source_file: str
    line: int
    span: tuple[int, ...]
    authorization: str = "unknown"


@dataclass(frozen=True)
class InteractionEdge:
    edge_id: str
    source: str
    target: str
    kind: str
    provenance: str
    function_id: str
    call_id: str


@dataclass(frozen=True)
class UnresolvedCall:
    function_id: str
    call_id: str
    callee: str
    reason: str


@dataclass(frozen=True)
class ProtocolHypothesis:
    kind: str
    status: str
    summary: str
    contracts: tuple[str, ...]
    functions: tuple[str, ...]
    edges: tuple[str, ...]
    assumptions: tuple[str, ...]
    uncertainty: str
    project: str
    source_snapshot: str
    compiler_configuration: str


@dataclass(frozen=True)
class ProtocolGraph:
    nodes: tuple[ContractNode, ...]
    functions: tuple[FunctionNode, ...]
    edges: tuple[InteractionEdge, ...]
    unresolved: tuple[UnresolvedCall, ...]
    hypotheses: tuple[ProtocolHypothesis, ...]
    incomplete: bool
    limit_reason: str
    project: str
    source_snapshot: str
    compiler_configuration: str


@dataclass(frozen=True)
class ProtocolPath:
    path_id: str
    nodes: tuple[str, ...]
    edges: tuple[str, ...]
    kinds: tuple[str, ...]
    status: str
    stable_id: str = ""


@dataclass(frozen=True)
class PathExpansion:
    paths: tuple[ProtocolPath, ...]
    truncated: bool


@dataclass(frozen=True)
class FlowStep:
    kind: str
    identity: str
    contract: str


@dataclass(frozen=True)
class CrossFlow:
    steps: tuple[FlowStep, ...]
    status: str
    provenance: str


@dataclass(frozen=True)
class ModalityCorrelation:
    static_id: str
    runtime_event: str
    transition_id: str
    economic_status: str
    executable_input: str
    status: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolEvidence:
    project: str
    source_snapshot: str
    compiler_configuration: str
    contracts: tuple[str, ...]
    functions: tuple[str, ...]
    edges: tuple[str, ...]
    path: str
    sequence_id: str
    actors: tuple[str, ...]
    locations: tuple[str, ...]
    engines: tuple[str, ...]
    environment: str
    assumptions: tuple[str, ...]
    uncertainty: str
    status: str
    paths: tuple[ProtocolPath, ...] = ()
    paths_truncated: bool = False

    @property
    def path_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.stable_id or item.path_id for item in self.paths))


def clamp_protocol_bounds(requested: ProtocolBounds | None) -> ProtocolBounds:
    """A planner may only tighten these caps."""
    if requested is None:
        return ProtocolBounds()
    return ProtocolBounds(
        max_contracts=_cap(requested.max_contracts, MAX_PROTOCOL_CONTRACTS),
        max_edges=_cap(requested.max_edges, MAX_PROTOCOL_EDGES),
        max_expansion=_cap(requested.max_expansion, MAX_PROTOCOL_EXPANSION),
        max_callback_depth=_cap(requested.max_callback_depth, MAX_CALLBACK_DEPTH),
    )


def same_storage(
    left_declaration: str, right_declaration: str, left_symbol: str, right_symbol: str
) -> bool:
    """Shared variable names are not aliasing."""
    del left_symbol, right_symbol
    return bool(left_declaration) and left_declaration == right_declaration


def actor_privilege(actor: str, authorization: str) -> str:
    """A label is not a privilege. Only established semantic evidence counts."""
    del actor
    if authorization == "established":
        return "established"
    return "unknown"


def candidate_status(status: str) -> str:
    if status in _FORBIDDEN:
        return "candidate"
    if status in {"candidate", "unknown", "unsupported", "incomplete", "reachable"}:
        return status
    return "unknown"


def build_protocol_graph(
    programs: tuple[SemanticProgram, ...],
    *,
    project: str,
    source_snapshot: str,
    compiler_configuration: str,
    sources: dict[str, str] | None = None,
    bounds: ProtocolBounds | None = None,
    addresses: dict[tuple[str, str], str] | None = None,
    storage_links: tuple[StorageLink, ...] = (),
    implementation_links: tuple[ImplementationLink, ...] = (),
    event_links: tuple[EventLink, ...] = (),
    price_links: tuple[PriceLink, ...] = (),
    trust_boundaries: tuple[TrustBoundary, ...] = (),
) -> ProtocolGraph:
    active = clamp_protocol_bounds(bounds)
    texts = sources or {}
    nodes, functions = _nodes(
        programs,
        project=project,
        source_snapshot=source_snapshot,
        compiler_configuration=compiler_configuration,
        sources=texts,
        addresses=addresses or {},
        limit=active.max_contracts,
    )
    incomplete = len(_contract_rows(programs)) > len(nodes)
    reason = "contract cap" if incomplete else ""
    edges, unresolved = _edges(programs, nodes, functions, texts)
    edges.extend(_storage_edges(storage_links, functions))
    edges.extend(_proxy_edges(implementation_links, nodes, compiler_configuration))
    edges.extend(_event_edges(event_links, functions))
    edges.extend(_authorization_edges(trust_boundaries, edges, functions))
    edges.extend(_price_edges(edges, price_links))
    edges.sort(key=lambda item: (_PRIORITY.get(item.kind, 10), item.edge_id))
    if len(edges) > active.max_edges:
        edges = edges[: active.max_edges]
        incomplete = True
        reason = reason or "edge cap"
    roles = _roles(nodes, edges)
    hypotheses = _hypotheses(
        edges,
        functions,
        project=project,
        source_snapshot=source_snapshot,
        compiler_configuration=compiler_configuration,
    )
    return ProtocolGraph(
        tuple(roles),
        tuple(functions),
        tuple(edges),
        tuple(unresolved),
        hypotheses,
        incomplete,
        reason,
        project,
        source_snapshot,
        compiler_configuration,
    )


def expand_paths(
    graph: ProtocolGraph, bounds: ProtocolBounds | None = None
) -> tuple[ProtocolPath, ...]:
    """Deterministic expansion that keeps the exact edge sequence.

    Parallel edges between the same nodes stay distinct. Path identity includes
    those edge identities and is not rebuilt by searching for some A-to-B edge.
    """
    return expand_paths_report(graph, bounds).paths


def expand_paths_report(
    graph: ProtocolGraph, bounds: ProtocolBounds | None = None
) -> PathExpansion:
    """Expand within the hard caps and say whether the cap hid further paths."""
    active = clamp_protocol_bounds(bounds)
    limit = active.max_expansion
    by_source: dict[str, list[InteractionEdge]] = {}
    by_id: dict[str, InteractionEdge] = {}
    for edge in graph.edges:
        by_source.setdefault(edge.source, []).append(edge)
        by_id[edge.edge_id] = edge
    for items in by_source.values():
        items.sort(key=lambda item: (_PRIORITY.get(item.kind, 10), item.edge_id))
    found: list[ProtocolPath] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for origin in sorted(graph.nodes, key=lambda item: item.node_id):
        queue: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((origin.node_id,), ())]
        seen.add(((origin.node_id,), ()))
        while queue and len(found) <= limit:
            nodes, edge_ids = queue.pop(0)
            if len(edge_ids) >= active.max_edges:
                continue
            depth = sum(1 for item in edge_ids if by_id[item].kind == "callback")
            for edge in by_source.get(nodes[-1], ()):
                if edge.kind == "callback" and depth + 1 > active.max_callback_depth:
                    continue
                nxt_nodes = (*nodes, edge.target)
                nxt_edges = (*edge_ids, edge.edge_id)
                if (nxt_nodes, nxt_edges) in seen or len(set(nxt_nodes)) > active.max_contracts:
                    continue
                if len(nxt_edges) > active.max_edges:
                    continue
                seen.add((nxt_nodes, nxt_edges))
                kinds = tuple(by_id[item].kind for item in nxt_edges)
                path_id = "nodes:" + "->".join(nxt_nodes) + "|edges:" + ",".join(nxt_edges)
                found.append(
                    ProtocolPath(
                        path_id,
                        nxt_nodes,
                        nxt_edges,
                        kinds,
                        "candidate",
                        path_stable_id(graph, path_id),
                    )
                )
                queue.append((nxt_nodes, nxt_edges))
                if len(found) > limit:
                    break
        if len(found) > limit:
            break
    truncated = len(found) > limit
    kept = found[:limit]
    kept.sort(key=lambda item: item.path_id)
    return PathExpansion(tuple(kept), truncated)


def path_stable_id(graph: ProtocolGraph, path_id: str) -> str:
    """Identity from project, snapshot, compiler, and the ordered node and edge ids."""
    blob = json.dumps(
        [graph.project, graph.source_snapshot, graph.compiler_configuration, path_id],
        separators=(",", ":"),
    )
    return "path-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def cross_flows(
    graph: ProtocolGraph, bounds: ProtocolBounds | None = None
) -> tuple[CrossFlow, ...]:
    active = clamp_protocol_bounds(bounds)
    flows: list[CrossFlow] = []
    for edge in graph.edges:
        if len(flows) >= active.max_edges:
            break
        if edge.kind in {"unresolved", "internal_call"}:
            continue
        flows.append(
            CrossFlow(
                (
                    FlowStep(edge.kind, edge.function_id or edge.edge_id, edge.source),
                    FlowStep(edge.kind, edge.call_id or edge.edge_id, edge.target),
                ),
                "candidate",
                edge.provenance,
            )
        )
    return tuple(flows)


def plan_protocol_calls(
    steps: tuple[tuple[str, ...], ...],
    program: SemanticProgram | None,
    source: str,
    *,
    exploration: ExplorationBounds | None = None,
    protocol: ProtocolBounds | None = None,
    programs: tuple[SemanticProgram, ...] = (),
    sources: dict[str, str] | None = None,
) -> tuple[PlannedCall, ...] | None:
    """Build calls only from established signatures. Missing arguments reject the plan.

    A step is ``(contract, name)`` in one program, or ``(file, contract, selector)``
    across programs. An unresolved or overloaded selector rejects the sequence.
    """
    active = clamp_bounds(exploration)
    caps = clamp_protocol_bounds(protocol)
    if not steps or len(steps) > active.max_sequence_length:
        return None
    if len({_step_contract(step) for step in steps}) > caps.max_contracts:
        return None
    if active.max_executions > MAX_EXECUTIONS:
        return None
    catalog = programs or ((program,) if program is not None else ())
    if not catalog:
        return None
    texts = dict(sources or {})
    if program is not None and source and program.file not in texts:
        texts[program.file] = source
    calls: list[PlannedCall] = []
    for step in steps:
        call = _resolve_protocol_step(step, catalog, texts, source, program)
        if call is None:
            return None
        calls.append(call)
    return tuple(calls)


def protocol_sequences(
    steps: tuple[tuple[str, ...], ...],
    program: SemanticProgram | None,
    source: str,
    *,
    exploration: ExplorationBounds | None = None,
    protocol: ProtocolBounds | None = None,
    project: str = "",
    target: str = "",
    programs: tuple[SemanticProgram, ...] = (),
    sources: dict[str, str] | None = None,
) -> tuple[PlannedSequence, ...]:
    calls = plan_protocol_calls(
        steps,
        program,
        source,
        exploration=exploration,
        protocol=protocol,
        programs=programs,
        sources=sources,
    )
    if calls is None:
        return ()
    active = clamp_bounds(exploration)
    sequence = PlannedSequence(
        "protocol:"
        + "|".join(f"{item.contract}.{item.function_identity or item.function}" for item in calls),
        "protocol",
        calls,
        "planned",
        ("a protocol sequence is not a finding", "an actor is not authorization"),
        "protocol",
        project,
        target,
    )
    return (sequence,)[: active.max_executions]


def separate_asset_deltas(deltas: tuple[AssetDelta, ...]) -> EconomicResult:
    """Keep actors, tokens, and states apart. A balance change is not profit."""
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for delta in deltas:
        index = "" if delta.transaction_index is None else str(delta.transaction_index)
        key = (
            delta.actor,
            delta.token,
            delta.kind,
            index,
            delta.snapshot,
            delta.contract,
            delta.source,
        )
        if key in seen:
            return EconomicResult(
                OracleStatus.UNKNOWN.value,
                "two observations of the same actor, token, and state were not merged",
            )
        seen.add(key)
    tokens = {item.token for item in deltas}
    if len(tokens) > 1:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "cross-asset comparison requires an explicit conversion",
            ("no price was invented",),
        )
    return EconomicResult(
        OracleStatus.INCOMPLETE.value,
        "actor-separated balances are observations",
        ("a positive delta is not profit", "a loss is not a vulnerability"),
    )


def correlate_modalities(
    *,
    static_id: str,
    runtime_event: str,
    transition_id: str,
    economic_status: str,
    trace: str,
    static_operation: str = "",
    runtime_operation: str = "",
    transition_operation: str = "",
    economic_operation: str = "",
    overload_ambiguous: bool = False,
) -> ModalityCorrelation:
    """Correlate modalities only when their operation identities are the same.

    An event name, a bare function name, or an ambiguous overload does not match.
    A match is still a candidate observation, not proof of the static property.
    """
    assumptions = (
        "an event name is not proof",
        "a static relationship is not a runtime observation",
        "a runtime event is not source-code proof",
        "a caller-supplied trace is not executable input",
        "an economic observation stays an observation",
    )
    observed = candidate_status(economic_status) if economic_status else "unknown"
    operations = (static_operation, runtime_operation, transition_operation)
    matched = _operations_match(operations, economic_operation, overload_ambiguous)
    if matched:
        status = "candidate"
    elif not static_id and not runtime_event and not transition_id and not any(operations):
        status = "unknown"
    else:
        status = "incomplete"
    return ModalityCorrelation(
        static_id,
        runtime_event,
        transition_id,
        observed,
        "",
        status,
        assumptions
        if trace or runtime_event or static_id or any(operations)
        else ("modalities stay separate",),
    )


def protocol_evidence(
    graph: ProtocolGraph,
    *,
    path: str = "",
    sequence_id: str = "",
    actors: tuple[str, ...] = (),
    engines: tuple[str, ...] = (),
    environment: str = "local",
    uncertainty: str = "candidate",
    paths: tuple[ProtocolPath, ...] = (),
    paths_truncated: bool = False,
) -> ProtocolEvidence:
    if not path and len(paths) == 1:
        path = paths[0].path_id
    return ProtocolEvidence(
        graph.project,
        graph.source_snapshot,
        graph.compiler_configuration,
        tuple(item.contract for item in graph.nodes),
        tuple(item.function_id for item in graph.functions),
        tuple(item.edge_id for item in graph.edges),
        path,
        sequence_id,
        actors,
        tuple(item.source_file for item in graph.nodes),
        engines,
        environment,
        ("a cross-contract candidate is not verification",),
        uncertainty,
        "candidate",
        paths,
        paths_truncated,
    )


def protocol_domain_evidence(evidence: ProtocolEvidence) -> Evidence:
    """Place a protocol candidate on the normal evidence lifecycle.

    The record does not contribute to verification.
    """
    return Evidence(
        kind=EvidenceKind.PROTOCOL_OBSERVATION,
        source="bugforge-protocol",
        summary="protocol candidate; not verification",
        details=evidence.uncertainty,
        metadata={
            "verified": "false",
            "status": "candidate",
            "project": evidence.project,
            "source_snapshot": evidence.source_snapshot,
            "compiler_configuration": evidence.compiler_configuration,
            "path": evidence.path,
            "path_count": str(len(evidence.paths)),
            "path_ids": json.dumps(list(evidence.path_ids)),
            "paths_truncated": str(evidence.paths_truncated).lower(),
            "sequence_id": evidence.sequence_id,
            "environment": evidence.environment,
            "evidence_class": "protocol",
        },
    )


def _cap(requested: int, limit: int) -> int:
    return max(1, min(int(requested), limit))


def _contract_rows(programs: tuple[SemanticProgram, ...]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for program in programs:
        if program.contracts:
            for item in program.contracts:
                rows.append((program.file, item.name, item.kind or "contract"))
        else:
            for name in sorted({item.contract for item in program.functions if item.contract}):
                rows.append((program.file, name, "contract"))
    return rows


def _nodes(
    programs: tuple[SemanticProgram, ...],
    *,
    project: str,
    source_snapshot: str,
    compiler_configuration: str,
    sources: dict[str, str],
    addresses: dict[tuple[str, str], str],
    limit: int,
) -> tuple[list[ContractNode], list[FunctionNode]]:
    nodes: list[ContractNode] = []
    functions: list[FunctionNode] = []
    for source_file, contract, kind in _contract_rows(programs):
        if len(nodes) >= limit:
            break
        if kind not in {"contract", "interface", "library"}:
            kind = "contract"
        span = _contract_span(programs, source_file, contract)
        node_id = f"{source_file}::{contract}::{','.join(str(part) for part in span)}"
        nodes.append(
            ContractNode(
                node_id,
                project,
                source_snapshot,
                compiler_configuration,
                source_file,
                contract,
                kind,
                span,
                addresses.get((source_file, contract), ""),
                "parser-contract",
            )
        )
        program = next(item for item in programs if item.file == source_file)
        text = sources.get(source_file, "")
        for function in program.functions:
            if function.contract != contract:
                continue
            types = _types(text, contract, function, program)
            signature = "unresolved" if types is None else f"{function.name}({','.join(types)})"
            functions.append(
                FunctionNode(
                    canonical_function_key(
                        function,
                        source_file=source_file,
                        parameter_types=() if types is None else types,
                        project=project,
                        source_snapshot=source_snapshot,
                        compiler_configuration=compiler_configuration,
                    ),
                    node_id,
                    contract,
                    function.name,
                    signature,
                    source_file,
                    function.line,
                    function.span,
                    function.authorization,
                )
            )
    return nodes, functions


def _types(
    source: str, contract: str, function: SemanticFunction, program: SemanticProgram
) -> tuple[str, ...] | None:
    """Types from this function's own header. Unknown stays unresolved."""
    header = (function.source or "").split("{", 1)[0]
    if "(" in header:
        parsed = parameter_types(header)
        if parsed is not None:
            return parsed
        return None
    unique = (
        sum(
            1
            for item in program.functions
            if item.contract == contract and item.name == function.name
        )
        == 1
    )
    if not unique or not source:
        return None
    fact = function_signature(source, contract, function.name, program)
    types = fact.get("parameter_types") if fact else None
    if isinstance(types, tuple):
        return tuple(str(item) for item in types)
    return None


def _contract_span(
    programs: tuple[SemanticProgram, ...], source_file: str, contract: str
) -> tuple[int, ...]:
    """The contract declaration span. A function span is not a substitute."""
    for program in programs:
        if program.file != source_file:
            continue
        for item in program.contracts:
            if item.name == contract and item.span:
                return item.span
    return (0, 0, 0, 0)


def _edges(
    programs: tuple[SemanticProgram, ...],
    nodes: list[ContractNode],
    functions: list[FunctionNode],
    sources: dict[str, str],
) -> tuple[list[InteractionEdge], list[UnresolvedCall]]:
    del sources
    edges: list[InteractionEdge] = []
    unresolved: list[UnresolvedCall] = []
    by_contract = _unique_nodes(nodes)
    for program in programs:
        for function in program.functions:
            caller = _function_node(functions, program.file, function)
            if caller is None:
                continue
            for site in function.call_sites:
                kind = _edge_kind(site, function, by_contract)
                target = _target_node(site, by_contract, function, functions)
                if kind is None or target is None:
                    unresolved.append(
                        UnresolvedCall(
                            caller.function_id,
                            site.call_id,
                            site.callee,
                            "unresolved" if site.resolution not in _RESOLVED else "callee-unknown",
                        )
                    )
                    continue
                edges.append(
                    InteractionEdge(
                        f"{caller.function_id}:{site.call_id}:{kind}",
                        caller.node_id,
                        target,
                        kind,
                        site.call_id or site.resolution,
                        caller.function_id,
                        site.call_id,
                    )
                )
    return edges, unresolved


def _edge_kind(
    site: SemanticCall,
    function: SemanticFunction,
    nodes: dict[str, ContractNode],
) -> str | None:
    del function
    if site.resolution not in _RESOLVED and site.call_type != "delegatecall":
        return None
    if site.call_type == "delegatecall":
        return "delegatecall" if site.resolution == "resolved" and site.target in nodes else None
    if site.call_type == "staticcall":
        return "staticcall" if site.target in nodes else None
    if site.call_type == "internal":
        return "internal_call"
    if site.callee.lower() in _CALLBACKS:
        return "callback" if site.target in nodes else None
    if site.call_type == "token" and site.callee == "transferFrom":
        return "transfer_from" if site.target in nodes else None
    if site.call_type == "token" and site.callee in {"approve"}:
        return "approval" if site.target in nodes else None
    if site.call_type == "token" and site.callee == "permit":
        return "permit" if site.target in nodes else None
    if site.call_type == "token":
        return "token_transfer" if site.target in nodes else None
    if site.call_type == "oracle":
        return "oracle_read" if site.target in nodes else None
    if site.call_type in {"contract-typed", "external"} and site.target in nodes:
        return "external_call"
    return None


def _target_node(
    site: SemanticCall,
    nodes: dict[str, ContractNode],
    function: SemanticFunction,
    functions: list[FunctionNode],
) -> str | None:
    if site.call_type == "internal":
        matches = [
            item
            for item in functions
            if item.contract == function.contract and item.name == site.callee
        ]
        if len(matches) != 1:
            return None
        return matches[0].node_id
    node = nodes.get(site.target)
    if node is None:
        return None
    if site.call_type == "delegatecall":
        return node.node_id
    if site.call_type == "staticcall" and _named_function(functions, node.contract, site.callee):
        return node.node_id
    if site.callee.lower() in _CALLBACKS and _named_function(functions, node.contract, site.callee):
        return node.node_id
    if site.call_type in {"token", "oracle", "contract-typed", "external"}:
        if _named_function(functions, node.contract, site.callee) is None:
            return None
        return node.node_id
    return None


def _named_function(functions: list[FunctionNode], contract: str, name: str) -> FunctionNode | None:
    matches = [item for item in functions if item.contract == contract and item.name == name]
    if len(matches) != 1:
        return None
    return matches[0]


def _authorization_edges(
    links: tuple[TrustBoundary, ...],
    edges: list[InteractionEdge],
    functions: list[FunctionNode],
) -> list[InteractionEdge]:
    """A trust-boundary hypothesis needs every required piece of evidence."""
    by_call = {
        (item.function_id, item.call_id): item
        for item in edges
        if item.function_id and item.call_id
    }
    by_function = {item.function_id: item for item in functions}
    found: list[InteractionEdge] = []
    for link in links:
        if link.missing_authorization != "established" or link.privileged_callee != "established":
            continue
        if not all(
            (
                link.caller_controlled,
                link.propagation,
                link.relationship,
                link.provenance,
                link.call_id,
                link.caller_function,
                link.callee_function,
            )
        ):
            continue
        caller = by_function.get(link.caller_function)
        callee = by_function.get(link.callee_function)
        if caller is None or callee is None or caller.contract == callee.contract:
            continue
        call = by_call.get((caller.function_id, link.call_id))
        if call is None or call.target != callee.node_id:
            continue
        found.append(
            InteractionEdge(
                f"{caller.function_id}:{link.call_id}:authorization",
                caller.node_id,
                callee.node_id,
                "authorization_dependency",
                link.provenance,
                caller.function_id,
                link.call_id,
            )
        )
    return found


def _unique_nodes(nodes: list[ContractNode]) -> dict[str, ContractNode]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.contract] = counts.get(node.contract, 0) + 1
    return {node.contract: node for node in nodes if counts[node.contract] == 1}


def _function_node(
    functions: list[FunctionNode], source_file: str, function: SemanticFunction
) -> FunctionNode | None:
    matches = [
        item
        for item in functions
        if item.source_file == source_file
        and item.contract == function.contract
        and item.name == function.name
        and item.line == function.line
        and item.span == function.span
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _storage_edges(
    links: tuple[StorageLink, ...], functions: list[FunctionNode]
) -> list[InteractionEdge]:
    by_id = {item.function_id: item for item in functions}
    edges: list[InteractionEdge] = []
    for link in links:
        if not link.declaration_id or not link.provenance:
            continue
        writer = by_id.get(link.writer_function)
        reader = by_id.get(link.reader_function)
        if writer is None or reader is None or writer.function_id == reader.function_id:
            continue
        if writer.source_file != link.source_file and reader.source_file != link.source_file:
            continue
        edges.append(
            InteractionEdge(
                f"storage:{link.declaration_id}:{writer.function_id}:{reader.function_id}",
                writer.node_id,
                reader.node_id,
                "storage_dependency",
                link.provenance,
                writer.function_id,
                link.declaration_id,
            )
        )
    return edges


def _proxy_edges(
    links: tuple[ImplementationLink, ...],
    nodes: list[ContractNode],
    compiler_configuration: str,
) -> list[InteractionEdge]:
    edges: list[InteractionEdge] = []
    for link in links:
        if not link.provenance or not link.implementation:
            continue
        if link.compiler_configuration and link.compiler_configuration != compiler_configuration:
            continue
        proxy = _node_by(nodes, link.proxy_file, link.proxy)
        implementation = _node_by(nodes, link.implementation_file, link.implementation)
        if proxy is None or implementation is None:
            continue
        edges.append(
            InteractionEdge(
                f"proxy:{proxy.node_id}:{implementation.node_id}",
                proxy.node_id,
                implementation.node_id,
                "proxy_implementation",
                link.provenance,
                "",
                "",
            )
        )
    return edges


def _event_edges(
    links: tuple[EventLink, ...], functions: list[FunctionNode]
) -> list[InteractionEdge]:
    by_id = {item.function_id: item for item in functions}
    edges: list[InteractionEdge] = []
    for link in links:
        function = by_id.get(link.function_identity)
        if function is None or not link.event or not link.declaration_id or not link.provenance:
            continue
        edges.append(
            InteractionEdge(
                f"event:{link.event}:{function.function_id}:{link.declaration_id}",
                function.node_id,
                function.node_id,
                "event_state",
                link.provenance,
                function.function_id,
                link.declaration_id,
            )
        )
    return edges


def _price_edges(
    edges: list[InteractionEdge], links: tuple[PriceLink, ...]
) -> list[InteractionEdge]:
    """A price edge requires oracle, computation, valuation, and asset evidence."""
    by_id = {item.edge_id: item for item in edges}
    assets = {"token_transfer", "transfer_from", "asset_movement"}
    found: list[InteractionEdge] = []
    for link in links:
        if not link.computation_id or not link.valuation_id or not link.provenance or not link.span:
            continue
        if link.provenance == "same-function oracle read and asset call":
            continue
        oracle = by_id.get(link.oracle_operation)
        asset = by_id.get(link.asset_operation)
        if oracle is None or asset is None:
            continue
        if oracle.kind != "oracle_read" or asset.kind not in assets:
            continue
        found.append(
            InteractionEdge(
                f"price:{oracle.edge_id}:{asset.edge_id}:{link.computation_id}",
                oracle.target,
                asset.target,
                "price_dependency",
                link.provenance,
                oracle.function_id,
                link.computation_id,
            )
        )
    return found


def _roles(nodes: list[ContractNode], edges: list[InteractionEdge]) -> list[ContractNode]:
    role_for = {
        "token_transfer": "token",
        "transfer_from": "token",
        "oracle_read": "oracle",
        "proxy_implementation": "proxy",
        "delegatecall": "proxy",
        "callback": "callback",
    }
    updated: list[ContractNode] = []
    for node in nodes:
        roles = {
            role_for[item.kind]
            for item in edges
            if item.kind in role_for and item.target == node.node_id
        }
        if any(
            item.kind == "proxy_implementation" and item.source == node.node_id for item in edges
        ):
            roles.add("proxy")
        if any(
            item.kind == "proxy_implementation" and item.target == node.node_id for item in edges
        ):
            roles.add("implementation")
        updated.append(
            ContractNode(
                node.node_id,
                node.project,
                node.source_snapshot,
                node.compiler_configuration,
                node.source_file,
                node.contract,
                node.kind,
                node.span,
                node.address,
                node.provenance,
                tuple(sorted(roles)),
            )
        )
    return updated


def _hypotheses(
    edges: list[InteractionEdge],
    functions: list[FunctionNode],
    *,
    project: str,
    source_snapshot: str,
    compiler_configuration: str,
) -> tuple[ProtocolHypothesis, ...]:
    del functions
    mapping = {
        "callback": "unsafe callback sequencing",
        "delegatecall": "delegatecall storage/authorization consequences",
        "proxy_implementation": "proxy/implementation semantic mismatch",
        "oracle_read": "oracle dependency leading to asset-sensitive behavior",
        "price_dependency": "reserve/price dependency crossing contracts",
        "token_transfer": "token accounting mismatch across contracts",
        "transfer_from": "allowance/approval misuse across components",
        "approval": "allowance/approval misuse across components",
        "permit": "permit misuse across components",
        "authorization_dependency": "cross-contract authorization confusion",
        "storage_dependency": "cross-contract state transition inconsistencies",
        "event_state": "event/state disagreement",
    }
    found: list[ProtocolHypothesis] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        kind = mapping.get(edge.kind)
        if kind is None or (kind, edge.edge_id) in seen:
            continue
        seen.add((kind, edge.edge_id))
        found.append(
            ProtocolHypothesis(
                kind,
                "candidate",
                kind,
                (edge.source, edge.target),
                (edge.function_id,) if edge.function_id else (),
                (edge.edge_id,),
                ("a hypothesis is not verification", "an edge is not an exploit"),
                "candidate",
                project,
                source_snapshot,
                compiler_configuration,
            )
        )
    return tuple(found)


def _node_by(nodes: list[ContractNode], source_file: str, contract: str) -> ContractNode | None:
    matches = [
        item for item in nodes if item.source_file == source_file and item.contract == contract
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _operations_match(
    operations: tuple[str, str, str], economic_operation: str, overload_ambiguous: bool
) -> bool:
    if overload_ambiguous or any(not item for item in operations):
        return False
    if len(set(operations)) != 1:
        return False
    identity = operations[0]
    if "(" not in identity or "::" not in identity:
        return False
    return not economic_operation or economic_operation == identity


def _step_contract(step: tuple[str, ...]) -> str:
    if len(step) >= 3:
        return step[1]
    return step[0] if step else ""


def _resolve_protocol_step(
    step: tuple[str, ...],
    catalog: tuple[SemanticProgram, ...],
    texts: dict[str, str],
    fallback_source: str,
    fallback_program: SemanticProgram | None,
) -> PlannedCall | None:
    if len(step) == 2:
        file_name, contract, selector = "", step[0], step[1]
    elif len(step) == 3:
        file_name, contract, selector = step
    else:
        return None
    if not contract or not selector:
        return None
    chosen = _choose_program(catalog, file_name, contract)
    if chosen is None:
        return None
    text = texts.get(chosen.file, "")
    if not text and chosen is fallback_program:
        text = fallback_source
    function = _select_function(chosen, contract, selector)
    if function is None:
        return None
    return _call_for_function(function, chosen, text)


def _choose_program(
    catalog: tuple[SemanticProgram, ...], file_name: str, contract: str
) -> SemanticProgram | None:
    if file_name:
        matches = [item for item in catalog if _file_match(item.file, file_name)]
    else:
        matches = [
            item
            for item in catalog
            if any(function.contract == contract for function in item.functions)
            or any(fact.name == contract for fact in item.contracts)
        ]
    if len(matches) != 1:
        return None
    return matches[0]


def _file_match(program_file: str, requested: str) -> bool:
    left = program_file.replace("\\", "/").rstrip("/")
    right = requested.replace("\\", "/").lstrip("./")
    return left == right or left.endswith("/" + right)


def _select_function(
    program: SemanticProgram, contract: str, selector: str
) -> SemanticFunction | None:
    owned = [item for item in program.functions if item.contract == contract]
    by_identity = [item for item in owned if item.identity == selector]
    if len(by_identity) == 1:
        return by_identity[0]
    by_signature = [item for item in owned if _header_signature(item) == selector]
    if len(by_signature) == 1:
        return by_signature[0]
    by_name = [item for item in owned if item.name == selector]
    if len(by_name) == 1:
        return by_name[0]
    return None


def _header_signature(function: SemanticFunction) -> str:
    header = (function.source or "").split("{", 1)[0]
    if "(" not in header:
        return ""
    types = parameter_types(header)
    if types is None:
        return ""
    return f"{function.name}({','.join(types)})"


def _call_for_function(
    function: SemanticFunction, program: SemanticProgram, source: str
) -> PlannedCall | None:
    header = (function.source or "").split("{", 1)[0]
    if "(" in header:
        types = parameter_types(header)
        if types is None:
            return None
        arguments: tuple[ArgumentCandidate, ...] = ()
        if types:
            rows = combine(types)
            if not rows:
                return None
            arguments = rows[0]
        return PlannedCall(
            function.name,
            "user",
            arguments,
            "0",
            "unknown",
            (
                "actor is explicit and is not proof of authorization",
                "a planned call is not a finding",
            ),
            0,
            function.contract,
            function.identity,
        )
    same_name = [
        item
        for item in program.functions
        if item.contract == function.contract and item.name == function.name
    ]
    if len(same_name) != 1:
        return None
    fact = function_signature(source, function.contract, function.name, program)
    if fact is None:
        return None
    parameter_type_names = fact.get("parameter_types")
    if not isinstance(parameter_type_names, tuple):
        return None
    call = describe_call(program, source, function.contract, function.name)
    if call is None or (parameter_type_names and not call.arguments):
        return None
    return PlannedCall(
        call.function,
        call.actor,
        call.arguments,
        call.value,
        call.direction,
        call.assumptions,
        call.score,
        function.contract,
        str(fact.get("identity") or function.identity),
    )
