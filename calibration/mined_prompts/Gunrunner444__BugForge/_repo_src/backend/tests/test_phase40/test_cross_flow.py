"""Phase 40 cross-contract security properties and Phase 39 graph integrity."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_cross_dataflow import (
    analyze_authorization,
    analyze_delegatecall,
    analyze_reentrancy,
    graph_edges_are_connected,
)
from app.parsing.solidity_dataflow import analyze_dataflow
from app.parsing.solidity_ir import build_semantic_program
from app.plugins import reset_plugin_catalog
from app.security_agent.chains import ChainStep, propose_chain, verify_chain
from app.security_agent.evidence_graph import EvidenceGraph


def _program(tmp_path: Path, source: str):
    reset_syntax_registry()
    reset_plugin_catalog()
    return build_semantic_program(parse_source("solidity", tmp_path / "C.sol", source))


def test_duplicate_loads_have_distinct_ids(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) balances;
        function run(address user) external {
            uint256 x = balances[user];
            uint256 y = balances[user];
        }
    }
    """
    program = _program(tmp_path, source)
    function = program.functions_named("run", "Vault")[0]
    loads = [item.operation_id for item in function.access_sites if item.kind == "read"]
    assert len(loads) >= 2
    assert len(set(loads)) == len(loads)
    flow = analyze_dataflow(program)
    assert graph_edges_are_connected(flow)
    assert flow.status == "available"


def test_local_arithmetic_dependencies(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function run(uint256 amount, uint256 bonus) external {
            uint256 fee = amount / 10;
            uint256 payout = totalAssets - fee;
            totalAssets = payout;
            fee += bonus;
        }
    }
    """
    flow = analyze_dataflow(_program(tmp_path, source))
    summary = flow._summary("run")
    assert summary is not None
    assert any(edge.kind == "local-dependency" for edge in summary.edges)
    assert any(edge.kind == "state-read-influences-write" for edge in summary.edges)
    assert graph_edges_are_connected(flow)


def test_reentrancy_property_requires_a_reachable_write(tmp_path: Path) -> None:
    vulnerable = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw() external {
            uint256 owed = totalAssets;
            msg.sender.call("");
            totalAssets = owed;
        }
    }
    """
    safe = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw() external {
            totalAssets = 0;
            msg.sender.call("");
        }
    }
    """
    vulnerable_program = _program(tmp_path, vulnerable)
    assert analyze_reentrancy(vulnerable_program, "withdraw").status == "potential"
    safe_program = _program(tmp_path, safe)
    assert analyze_reentrancy(safe_program, "withdraw").status == "unknown"


def test_authorization_property_rejects_unrelated_require(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 admin;
        function set(uint256 amount) external {
            require(amount > 0);
            admin = amount;
        }
        function setOwner(uint256 value) external {
            require(msg.sender == owner);
            admin = value;
        }
    }
    """
    program = _program(tmp_path, source)
    assert analyze_authorization(program, "set").status == "potential"
    assert analyze_authorization(program, "setOwner").status == "satisfied"


def test_delegatecall_property_tracks_caller_target(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        function run(address target, bytes calldata data) external {
            target.delegatecall(data);
        }
    }
    """
    result = analyze_delegatecall(_program(tmp_path, source), "run")
    assert result.status == "potential"
    assert result.provenance == "attacker"


def test_semantic_property_does_not_verify_a_chain() -> None:
    class Session:
        def __init__(self) -> None:
            self.id = "s"
            self.project_id = "p"
            self.graph = EvidenceGraph(session_id="s", project_id="p")
            self.findings = []

    session = Session()
    session.graph.add(
        kind="static_analysis",
        provenance="static_analysis",
        summary="semantic property potential",
        node_id="e1",
        extra={"lifecycle": "verified", "analysis_origin": "semantic"},
    )
    chain = propose_chain("c", (ChainStep("a", evidence_id="e1"), ChainStep("b", evidence_id="e1")))
    assert verify_chain(session, chain).status != "verified"
