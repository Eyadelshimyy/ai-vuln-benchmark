"""Phase 37 semantic IR and Phase 36 authority fixes."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_ir import build_semantic_program, render_snapshot
from app.parsing.solidity_project import _version_groups
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine
from app.security_agent.chains import (
    ChainStep,
    assess_chain,
    chain_from_snapshot,
    propose_chain,
    verify_chain,
)
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_agent.leads import new_lead


class _Session:
    def __init__(self, graph: EvidenceGraph) -> None:
        self.id = graph.session_id
        self.project_id = graph.project_id
        self.graph = graph


def _graph(tmp_path: Path, source: str):
    reset_syntax_registry()
    reset_plugin_catalog()
    return parse_source("solidity", tmp_path / "C.sol", source)


def test_fake_evidence_cannot_verify_a_chain() -> None:
    chain = propose_chain(
        "c",
        (
            ChainStep("a", evidence_id="fake", evidence_tier="verified"),
            ChainStep("b", evidence_id="fake-2", evidence_tier="verified"),
        ),
    )
    assert assess_chain(chain).status == "proposed"
    graph = EvidenceGraph(session_id="s", project_id="p")
    assert verify_chain(_Session(graph), chain).verified is False
    restored = chain_from_snapshot({**chain.snapshot(), "status": "verified"})
    assert restored.status == "proposed"
    assert restored.verified is False


def test_chain_status_follows_graph_provenance() -> None:
    graph = EvidenceGraph(session_id="s", project_id="p")
    graph.add(
        kind="reproduction",
        provenance="reproduction",
        summary="reproduced",
        node_id="e1",
        extra={"outcome": "reproduced", "reproduced": "true", "lifecycle": "verified"},
    )
    graph.add(
        kind="reproduction",
        provenance="reproduction",
        summary="reproduced",
        node_id="e2",
        extra={"outcome": "reproduced", "reproduced": "true", "lifecycle": "verified"},
    )
    chain = propose_chain("c", (ChainStep("a", evidence_id="e1"), ChainStep("b", evidence_id="e2")))
    assert verify_chain(_Session(graph), chain).status == "reproduced"
    graph.nodes["e1"].extra["lifecycle"] = "verified"
    graph.nodes["e2"].extra["lifecycle"] = "verified"
    assert verify_chain(_Session(graph), chain).status == "reproduced"
    other = EvidenceGraph(session_id="other", project_id="p")
    other.add(
        kind="reproduction",
        provenance="reproduction",
        summary="x",
        node_id="e1",
        extra={"lifecycle": "verified"},
    )
    assert verify_chain(_Session(other), chain).status == "proposed"


def test_static_evidence_cannot_verify() -> None:
    graph = EvidenceGraph(session_id="s", project_id="p")
    graph.add(kind="static_analysis", provenance="static_analysis", summary="scan", node_id="e1")
    graph.add(kind="scanner", provenance="scanner_result", summary="semgrep", node_id="e2")
    chain = propose_chain("c", (ChainStep("a", evidence_id="e1"), ChainStep("b", evidence_id="e2")))
    assert verify_chain(_Session(graph), chain).status == "static"


def test_unrelated_contracts_are_not_implementation_pairs(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        fallback() external payable { implementation.delegatecall(msg.data); }
    }
    contract Impl { uint256 value; }
    contract Unrelated { uint256 other; }
    """
    path = tmp_path / "C.sol"
    path.write_text(source, encoding="utf-8")
    notes = " ".join(
        item.summary
        for item in SecurityAnalysisEngine().analyze_repository(tmp_path, [path]).observations
        if item.rule_id == "sol.storage_collision"
    )
    assert "Unrelated" not in notes


def test_withdraw_does_not_match_withdraw_all(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        uint256 totalAssets;
        function withdrawAll() external { totalSupply -= 1; return; }
        function withdraw() external { totalSupply -= 1; totalAssets -= 1; }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    assert [item.name for item in program.functions_named("withdraw")] == ["withdraw"]
    assert program.functions_named("withdrawAll")[0].name == "withdrawAll"


def test_compiler_groups_do_not_invent_a_version(tmp_path: Path) -> None:
    groups, omitted = _version_groups({"A.sol": "contract A {}"}, [], tmp_path)
    assert groups == {}
    assert omitted == ("A.sol",)
    mixed, omitted_mixed = _version_groups(
        {
            "A.sol": 'pragma solidity =0.8.25;\nimport "./B.sol";\ncontract A {}',
            "B.sol": "pragma solidity =0.8.28;\ncontract B {}",
        },
        [],
        tmp_path,
    )
    assert "A.sol" in omitted_mixed
    assert "0.8.34" not in mixed


def test_semantic_snapshot_lists_reads_and_calls(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        address owner;
        function withdraw(uint256 amount) external {
            require(msg.sender == owner);
            totalAssets -= amount;
            msg.sender.call("");
        }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    text = render_snapshot(program)
    assert "Function withdraw" in text
    assert "totalAssets" in text
    assert "call" in text
    assert program.compiler_ir_status == "unavailable"
    function = program.functions_named("withdraw", "Vault")[0]
    assert program.authorization(function) == "guarded"
    assert "totalAssets" in program.state_writes(function)
    assert program.external_calls(function) == ("call",)


def test_lead_observation_ids_are_not_hypothesis_ids() -> None:
    lead = new_lead(project_id="p", session_id="s", target="t", hypothesis="h")

    class Hypothesis:
        id = "hyp"
        title = "h"
        target = "t"
        severity = "medium"
        suggested_next_action = "look"
        supporting_evidence_ids = ("e1",)

    class Session:
        project_id = "p"
        id = "s"
        target = "t"
        leads = [lead]

    from app.security_agent.leads import lead_for_hypothesis

    updated = lead_for_hypothesis(Session(), Hypothesis())
    assert updated.observation_ids == []
    assert updated.hypothesis_ids == ["hyp"]
    assert "hyp" not in updated.related_ids
    assert updated.evidence_ids == ["e1"]
