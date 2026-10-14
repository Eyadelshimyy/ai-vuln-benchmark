"""Phase 39 path-sensitive value dataflow."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_dataflow import analyze_dataflow
from app.parsing.solidity_expr import occurrences
from app.parsing.solidity_ir import build_semantic_program
from app.plugins import reset_plugin_catalog
from app.security_agent.chains import ChainStep, propose_chain, verify_chain
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_agent.leads import backfill_lead_relationships


def _graph(tmp_path: Path, source: str):
    reset_syntax_registry()
    reset_plugin_catalog()
    return parse_source("solidity", tmp_path / "src" / "Vault.sol", source)


def _program(tmp_path: Path, source: str):
    return build_semantic_program(_graph(tmp_path, source))


def test_scalar_update_has_separate_read_and_write(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 x;
        uint256 y;
        uint256 totalAssets;
        uint256 supply;
        function run(uint256 amount) external {
            x = x + y;
            totalAssets += amount;
            totalAssets -= amount;
            ++supply;
            supply--;
        }
    }
    """
    function = _program(tmp_path, source).functions_named("run", "Vault")[0]
    kinds = {(item.path, item.kind) for item in function.access_sites}
    assert ("x", "read") in kinds
    assert ("x", "write") in kinds
    assert ("totalAssets", "read-modify-write") in kinds
    assert ("supply", "read-modify-write") in kinds
    assert "x" in function.reads and "x" in function.writes


def test_mapping_occurrences_stay_separate(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) balances;
        struct Position { uint256 amount; address owner; }
        mapping(uint256 => Position) positions;
        function run(address user, address user1, address user2, uint256 id, uint256 amount) external {
            balances[user] = balances[user] - amount;
            positions[id].amount = positions[id].amount - amount;
            balances[user1] = 1;
            balances[user2] = 2;
            delete balances[user];
        }
    }
    """
    function = _program(tmp_path, source).functions_named("run", "Vault")[0]
    paths = {item.path for item in function.access_sites}
    assert "balances[user]" in paths
    assert "balances[user1]" in paths
    assert "balances[user2]" in paths
    assert "positions[id].amount" in paths
    assert any(
        item.kind == "read" and item.path == "balances[user]" for item in function.access_sites
    )
    assert any(
        item.kind == "write" and item.path == "balances[user]" for item in function.access_sites
    )
    flow = analyze_dataflow(build_semantic_program(_graph(tmp_path, source)))
    summary = flow._summary("run")
    assert summary is not None
    assert any(edge.kind == "state-read-influences-write" for edge in summary.edges)


def test_parameter_shadows_state_and_nested_local_does_not(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        function shadow(uint256 totalSupply) external {
            uint256 other = totalSupply;
        }
        function nested() external {
            totalSupply = 1;
            {
                uint256 totalSupply = 2;
            }
        }
    }
    """
    program = _program(tmp_path, source)
    shadow = program.functions_named("shadow", "Vault")[0]
    nested = program.functions_named("nested", "Vault")[0]
    assert "totalSupply" not in shadow.reads
    assert "totalSupply" not in shadow.writes
    assert "totalSupply" in nested.writes


def test_authorization_is_per_operation(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 admin;
        function set(uint256 value) external {
            require(msg.sender == owner);
            admin = value;
        }
        function late(uint256 value) external {
            admin = value;
            require(msg.sender == owner);
        }
        function unrelated(uint256 amount) external {
            require(amount > 0);
            admin = amount;
        }
    }
    """
    program = _program(tmp_path, source)
    assert program.functions_named("set", "Vault")[0].authorization == "guarded"
    assert program.functions_named("late", "Vault")[0].authorization == "unguarded"
    assert program.functions_named("unrelated", "Vault")[0].authorization == "unguarded"
    guarded = [
        item.guard_status
        for item in program.functions_named("set", "Vault")[0].access_sites
        if item.kind != "read"
    ]
    assert guarded == ["guarded"]


def test_reads_before_call_are_positional(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 later;
        function withdraw(uint256 amount) external {
            uint256 snapshot = totalAssets;
            msg.sender.call("");
            later = amount;
        }
    }
    """
    program = _program(tmp_path, source)
    function = program.functions_named("withdraw", "Vault")[0]
    call = next(site for site in function.call_sites if site.callee == "call")
    flow = analyze_dataflow(program)
    assert "totalAssets" in flow.reads_before_call(function.identity, call.call_id)
    assert "later" not in flow.reads_before_call(function.identity, call.call_id)
    assert "later" in flow.writes_after_call(function.identity, call.call_id)
    assert "totalAssets" not in flow.writes_after_call(function.identity, call.call_id)


def test_delegatecall_provenance_follows_the_expression(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        function fromParam(address target, bytes calldata data) external {
            target.delegatecall(data);
        }
        function fromState(bytes calldata data) external {
            implementation.delegatecall(data);
        }
        function fromCall(bytes calldata data) external {
            getImplementation().delegatecall(data);
        }
        function getImplementation() internal returns (address) { return implementation; }
    }
    """
    flow = analyze_dataflow(_program(tmp_path, source))
    assert flow.target_provenance("fromParam") == "attacker"
    assert flow.target_provenance("fromState") == "state"
    assert flow.target_provenance("fromCall") == "state"


def test_ambiguous_function_name_is_not_a_fuzzy_match(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function run() external { this.run(); }
    }
    contract B {
        function run() external {}
    }
    """
    flow = analyze_dataflow(_program(tmp_path, source))
    assert flow._summary("run") is None
    assert flow._summary("A.run:0") is None or flow.target_provenance("run") == "unknown"


def test_compiler_profiles_do_not_match_basename_only() -> None:
    from app.parsing.solidity_ir import _same_source

    assert _same_source("/repo/src/Vault.sol", "src/Vault.sol")
    assert not _same_source("/repo/test/Vault.sol", "src/Vault.sol")


def test_legacy_related_id_backfill_does_not_duplicate() -> None:
    first, findings, semantic = backfill_lead_relationships(
        related_ids=["hyp", "e1"],
        evidence_ids=["e1"],
        observation_ids=[],
        chain_ids=[],
        hypothesis_ids=None,
        finding_ids=["f1"],
        semantic_node_ids=[],
    )
    assert first == ["hyp"]
    assert findings == ["f1"]
    again, _, _ = backfill_lead_relationships(
        related_ids=["hyp"],
        evidence_ids=[],
        observation_ids=[],
        chain_ids=[],
        hypothesis_ids=first,
        finding_ids=[],
        semantic_node_ids=semantic,
    )
    assert again == ["hyp"]


def test_fabricated_graph_metadata_does_not_verify() -> None:
    class Session:
        def __init__(self) -> None:
            self.id = "s"
            self.project_id = "p"
            self.graph = EvidenceGraph(session_id="s", project_id="p")
            self.findings = [type("F", (), {"id": "e1", "is_verified": True, "evidence": None})()]

    session = Session()
    session.graph.add(
        kind="execution",
        provenance="execution",
        summary="claimed",
        node_id="e1",
        extra={"lifecycle": "verified"},
    )
    chain = propose_chain("c", (ChainStep("a", evidence_id="e1"), ChainStep("b", evidence_id="e1")))
    assert verify_chain(session, chain).status != "verified"


def test_occurrence_parser_keeps_read_and_write() -> None:
    found = occurrences("balances[user] = balances[user] - amount;", {"balances"})
    assert [item.kind for item in found] == ["read", "write"]


def test_phase38_false_negative_is_executable(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw(uint256 amount) external {
            totalAssets -= amount;
        }
    }
    """
    function = _program(tmp_path, source).functions_named("withdraw", "Vault")[0]
    assert "totalAssets" in function.reads
    assert "totalAssets" in function.writes
    assert any(
        item.kind == "read-modify-write" and item.path == "totalAssets"
        for item in function.access_sites
    )
