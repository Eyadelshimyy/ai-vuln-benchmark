"""A derived research evidence graph (Phase 52, Slice D).

This is a *view*, not a second evidence store: it is rebuilt from the campaign's
existing typed state (static candidates, VFCS sequences, property specs, persisted
executions and their check paths, reproduction bundles, and the Phase 49
contradictions). Every edge carries provenance -- which analysis or engine produced
the relation -- and every contradiction is a visible node with ``contradicts`` edges,
never resolved silently. Nothing here verifies anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.discovery.bounty.properties import PropertySpec
from app.discovery.bounty.vfcs import Vfcs
from app.discovery.orchestration.codec import digest
from app.parsing.solidity_research import SemanticCandidate

GRAPH_SCHEMA = "bugforge.evidence_graph/1"
MAX_NODES = 600
MAX_EDGES = 1200


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def node(self, node_id: str, kind: str, **attrs: Any) -> str:
        if node_id not in self.nodes and len(self.nodes) < MAX_NODES:
            self.nodes[node_id] = {"id": node_id, "kind": kind, **attrs}
        return node_id

    def edge(self, source: str, target: str, kind: str, provenance: str, **attrs: Any) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        edge_id = "e_" + digest({"s": source, "t": target, "k": kind})[:16]
        if edge_id not in self.edges and len(self.edges) < MAX_EDGES:
            self.edges[edge_id] = {
                "id": edge_id,
                "source": source,
                "target": target,
                "kind": kind,
                "provenance": provenance,
                **attrs,
            }


def build_evidence_graph(
    *,
    candidates: Iterable[SemanticCandidate],
    sequences: Iterable[Vfcs],
    specs: Mapping[str, PropertySpec],
    executions: Mapping[str, Mapping[str, Any]],
    phase49_contradictions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    graph = _Graph()
    candidates = tuple(candidates)
    for item in candidates:
        key = f"{item.detector}@{item.contract}.{item.function}"
        graph.node(
            f"candidate:{key}",
            "static_candidate",
            detector=item.detector,
            family=item.family,
            file=item.file,
            line=item.line,
            confidence=item.confidence,
            verified=False,
        )
    # different detector families at the same site support (never confirm) each other
    by_site: dict[str, list[SemanticCandidate]] = {}
    for item in candidates:
        by_site.setdefault(f"{item.contract}.{item.function}", []).append(item)
    for site, items in sorted(by_site.items()):
        for a in items:
            for b in items:
                if a.family < b.family:
                    graph.edge(
                        f"candidate:{a.detector}@{site}",
                        f"candidate:{b.detector}@{site}",
                        "supports",
                        "static:same_site_different_family",
                    )
    contradictions: list[dict[str, Any]] = []
    for sequence in sorted(sequences, key=lambda s: s.sequence_id):
        sid = sequence.sequence_id
        seq_node = graph.node(
            f"sequence:{sid}", "vfcs_sequence", template=sequence.template, origin=sequence.origin
        )
        graph.edge(
            f"candidate:{sequence.derived_from}",
            seq_node,
            "derived_sequence",
            f"vfcs:{sequence.origin}",
        )
        spec = specs.get(sid)
        prop_node = ""
        if spec is not None:
            prop_node = graph.node(
                f"property:{spec.property_id}",
                "property",
                family=spec.family.value,
                declaration=spec.declaration.value,
                oracle=spec.oracle.kind.value,
                status=spec.status,
            )
            graph.edge(seq_node, prop_node, "declares", "properties:build_property")
        record = executions.get(sid)
        if record is None:
            continue
        exec_node = graph.node(
            f"execution:{sid}",
            "execution",
            outcome=record.get("outcome"),
            identity_status=record.get("identity_status"),
            replay_mode=record.get("replay_mode"),
            stale=bool(record.get("stale")),
            verified=False,
        )
        graph.edge(seq_node, exec_node, "executed_as", f"stateful:{record.get('replay_mode', '')}")
        violated: list[str] = []
        held: list[str] = []
        for path in record.get("check_paths", []) or []:
            label = str(path.get("path", ""))
            path_node = graph.node(
                f"path:{sid}:{label}",
                "check_path",
                engine=path.get("engine"),
                material=bool(path.get("material")),
                verdict=path.get("verdict"),
            )
            graph.edge(
                prop_node or exec_node,
                path_node,
                "judged_by",
                f"engine:{path.get('engine', '')}",
                verdict=path.get("verdict"),
            )
            if path.get("verdict") == "property_violated":
                violated.append(path_node)
            elif path.get("verdict") == "property_held":
                held.append(path_node)
        for bad in violated:
            for good in held:
                graph.edge(bad, good, "contradicts", "independent_check")
                contradictions.append(
                    {
                        "kind": "independent_paths_disagree",
                        "sequence_id": sid,
                        "violated_by": bad,
                        "held_by": good,
                        "status": "open",
                    }
                )
        if record.get("stale"):
            contradictions.append(
                {
                    "kind": "stale_execution",
                    "sequence_id": sid,
                    "status": "open",
                    "note": "executed against a source set that has since changed",
                }
            )
        bundle = str(record.get("bundle_id", ""))
        if bundle:
            graph.node(f"bundle:{bundle}", "repro_bundle")
            graph.edge(exec_node, f"bundle:{bundle}", "reproduced_by", "stateful:build_bundle")
    for entry in phase49_contradictions:
        node = graph.node(
            f"contradiction:{entry.get('id', '')}",
            "phase49_contradiction",
            status=entry.get("status"),
            identity=entry.get("identity"),
            contradiction_kind=entry.get("kind"),
        )
        contradictions.append(
            {
                "kind": f"phase49:{entry.get('kind', '')}",
                "node": node,
                "status": entry.get("status"),
            }
        )
    nodes = sorted(graph.nodes.values(), key=lambda n: n["id"])
    edges = sorted(graph.edges.values(), key=lambda e: e["id"])
    return {
        "schema": GRAPH_SCHEMA,
        "nodes": nodes,
        "edges": edges,
        "contradictions": contradictions,
        "truncated": len(graph.nodes) >= MAX_NODES or len(graph.edges) >= MAX_EDGES,
        "digest": digest({"n": [n["id"] for n in nodes], "e": [e["id"] for e in edges]}),
        "note": "a derived view of existing campaign state; not a second evidence store",
        "verified": False,
    }
