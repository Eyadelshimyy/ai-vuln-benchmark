"""Every valid bounded protocol path stays represented with a stable identity."""

from __future__ import annotations

import json
from pathlib import Path

from app.adapters.discovery.protocol import ProtocolEngine
from app.discovery.engine import AnalysisRequest
from app.parsing.solidity_protocol import (
    MAX_PROTOCOL_EXPANSION,
    ContractNode,
    InteractionEdge,
    ProtocolBounds,
    ProtocolGraph,
    expand_paths,
    expand_paths_report,
    protocol_domain_evidence,
    protocol_evidence,
)


def _node(node_id: str, contract: str) -> ContractNode:
    return ContractNode(
        node_id,
        "p",
        "snap",
        "cfg",
        f"{contract}.sol",
        contract,
        "contract",
        (1, 2, 3, 4),
        "",
        "parser",
    )


def _graph(extra_nodes: int = 0, snapshot: str = "snap") -> ProtocolGraph:
    nodes = [_node("A", "Router"), _node("B", "Vault"), _node("C", "Pool")]
    edges = [
        InteractionEdge("e1", "A", "B", "external_call", "established", "fn1", "c1"),
        InteractionEdge("e2", "A", "B", "staticcall", "established", "fn2", "c2"),
        InteractionEdge("e3", "B", "C", "external_call", "established", "fn3", "c3"),
        InteractionEdge("e4", "A", "C", "external_call", "established", "fn4", "c4"),
    ]
    for index in range(extra_nodes):
        name = f"N{index}"
        nodes.append(_node(name, name))
        edges.append(
            InteractionEdge(f"x{index}", "C", name, "external_call", "established", "f", "c")
        )
    return ProtocolGraph(tuple(nodes), (), tuple(edges), (), (), False, "", "p", snapshot, "cfg")


def test_every_valid_bounded_path_is_kept_with_a_distinct_stable_identity() -> None:
    graph = _graph()
    report = expand_paths_report(graph)
    ids = [item.path_id for item in report.paths]
    assert len(ids) == len(set(ids))
    assert {item.edges for item in report.paths} >= {
        ("e1",),
        ("e2",),
        ("e4",),
        ("e1", "e3"),
        ("e2", "e3"),
        ("e3",),
    }
    stable = [item.stable_id for item in report.paths]
    assert all(item.startswith("path-") for item in stable)
    assert len(stable) == len(set(stable))
    assert report.truncated is False
    again = expand_paths_report(graph)
    assert [item.stable_id for item in again.paths] == stable


def test_stable_identity_binds_the_source_snapshot_and_ignores_list_position() -> None:
    first = expand_paths_report(_graph())
    other = expand_paths_report(_graph(snapshot="newer"))
    assert {item.path_id for item in first.paths} == {item.path_id for item in other.paths}
    assert {item.stable_id for item in first.paths}.isdisjoint(
        {item.stable_id for item in other.paths}
    )


def test_evidence_preserves_all_paths_and_does_not_verify() -> None:
    graph = _graph()
    report = expand_paths_report(graph)
    evidence = protocol_evidence(graph, paths=report.paths, paths_truncated=report.truncated)
    assert len(evidence.paths) == len(report.paths) > 1
    assert evidence.path == ""
    assert set(evidence.path_ids) == {item.stable_id for item in report.paths}
    assert all(item.status == "candidate" for item in evidence.paths)
    domain = protocol_domain_evidence(evidence)
    assert domain.contributes_to_verification is False
    assert domain.metadata["verified"] == "false"
    assert domain.metadata["path_count"] == str(len(report.paths))
    assert json.loads(domain.metadata["path_ids"]) == sorted(evidence.path_ids)


def test_a_single_path_keeps_its_id_in_the_legacy_field() -> None:
    graph = ProtocolGraph(
        (_node("A", "Router"), _node("B", "Vault")),
        (),
        (InteractionEdge("e1", "A", "B", "external_call", "established", "fn", "c"),),
        (),
        (),
        False,
        "",
        "p",
        "snap",
        "cfg",
    )
    report = expand_paths_report(graph)
    evidence = protocol_evidence(graph, paths=report.paths)
    assert len(report.paths) == 1
    assert evidence.path == report.paths[0].path_id


def test_the_expansion_cap_holds_and_truncation_is_reported() -> None:
    graph = _graph(extra_nodes=30)
    report = expand_paths_report(graph)
    assert len(report.paths) == MAX_PROTOCOL_EXPANSION
    assert report.truncated is True
    assert len(expand_paths(graph)) == MAX_PROTOCOL_EXPANSION
    widened = expand_paths_report(graph, ProtocolBounds(max_expansion=10_000))
    assert len(widened.paths) == MAX_PROTOCOL_EXPANSION
    tight = expand_paths_report(graph, ProtocolBounds(max_expansion=3))
    assert len(tight.paths) == 3 and tight.truncated is True
    exact = expand_paths_report(_graph(), ProtocolBounds(max_expansion=len(expand_paths(_graph()))))
    assert exact.truncated is False


def test_the_engine_reports_every_path_and_never_promotes_one(tmp_path: Path, monkeypatch) -> None:
    graph = _graph()
    (tmp_path / "A.sol").write_text("contract A {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.adapters.discovery.protocol.build_protocol_graph", lambda *args, **kwargs: graph
    )
    result = ProtocolEngine().start_campaign(
        AnalysisRequest(tmp_path, "solidity", target="Router", files=("A.sol",))
    )
    expected = expand_paths(graph)
    evidence = result.protocol_evidence
    assert len(evidence.paths) == len(expected)
    assert result.metadata["paths"] == str(len(expected))
    assert json.loads(result.metadata["path_ids"]) == sorted(item.stable_id for item in expected)
    assert result.metadata["paths_truncated"] == "false"
    assert result.metadata["verified"] == "false"
    assert result.to_evidence().contributes_to_verification is False
    assert all(item.status == "candidate" for item in evidence.paths)
