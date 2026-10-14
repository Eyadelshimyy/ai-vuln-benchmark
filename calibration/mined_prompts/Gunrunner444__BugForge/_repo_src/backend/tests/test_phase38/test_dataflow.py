"""Phase 38 semantic IR hardening and bounded dataflow."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_dataflow import analyze_dataflow
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_project import _version_groups, project_standard_json
from app.plugins import reset_plugin_catalog
from app.security_agent.chains import ChainStep, propose_chain, verify_chain
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_testing.oob import MockOobProvider, OobProvider, production_provider

FALSE_NEGATIVES = (
    {
        "vulnerability_class": "read-write",
        "source": "totalAssets -= amount",
        "expected_path": "totalAssets read and write",
        "expected_detection": "read_write_same_operation",
        "previously_missed": "writes were removed from reads",
        "phase": "38",
    },
    {
        "vulnerability_class": "storage-proxy",
        "source": "proxy beside one unrelated contract",
        "expected_path": "unresolved implementation",
        "expected_detection": "no implementation pair",
        "previously_missed": "single other contract was treated as the implementation",
        "phase": "38",
    },
)


def _graph(tmp_path: Path, source: str):
    reset_syntax_registry()
    reset_plugin_catalog()
    return parse_source("solidity", tmp_path / "C.sol", source)


def test_assignment_is_both_read_and_write(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 x;
        uint256 y;
        uint256 totalAssets;
        mapping(address => uint256) balances;
        uint256[] items;
        struct Position { uint256 amount; address owner; }
        mapping(uint256 => Position) positions;
        uint128 packed;
        function update(uint256 amount, address user, uint256 id, uint256 index) external {
            x = x + y;
            totalAssets += amount;
            totalAssets -= amount;
            balances[user] = balances[user] - amount;
            items[index] = items[index] + 1;
            positions[id].amount = positions[id].amount - amount;
            packed = packed + 1;
            y = x;
        }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    function = program.functions_named("update", "Vault")[0]
    for name in ("x", "totalAssets", "balances", "items", "positions", "packed"):
        assert name in function.reads
        assert name in function.writes
        assert name in function.read_write_same_operation
    assert "y" in function.reads
    assert function.span[2] > 0
    assert function.declaration_ids
    assert any("balances[" in item for item in function.accesses)
    assert any("positions[" in item and "amount" in item for item in function.accesses)


def test_shadowed_local_is_not_the_state_variable(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        function shadow(uint256 totalSupply) external {
            uint256 other = totalSupply;
            other += 1;
        }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    function = program.functions_named("shadow", "Vault")[0]
    assert "totalSupply" not in function.reads
    assert "totalSupply" not in function.writes
    assert program.declarations[0].declaration_id
    assert program.declarations[0].symbol == "totalSupply"


def test_unrelated_require_is_not_a_guard(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 totalSupply;
        function set(uint256 amount) external {
            require(amount > 0);
            totalSupply = amount;
        }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    function = program.functions_named("set", "Vault")[0]
    assert function.authorization == "unguarded"


def test_owner_require_guards_the_write(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 totalSupply;
        function set(uint256 amount) external {
            require(msg.sender == owner);
            totalSupply = amount;
        }
    }
    """
    function = build_semantic_program(_graph(tmp_path, source)).functions_named("set", "Vault")[0]
    assert function.authorization == "guarded"
    assert function.path_status == "unconditional"


def test_guard_after_write_does_not_dominate(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 totalSupply;
        function set(uint256 amount) external {
            totalSupply = amount;
            require(msg.sender == owner);
        }
    }
    """
    function = build_semantic_program(_graph(tmp_path, source)).functions_named("set", "Vault")[0]
    assert function.authorization == "unguarded"


def test_function_limit_is_explicit(tmp_path: Path, monkeypatch) -> None:
    from app.parsing import solidity_ir

    monkeypatch.setattr(solidity_ir, "FUNCTION_LIMIT", 1)
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        function a() external {}
        function b() external {}
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    assert program.functions_seen == 2
    assert program.functions_indexed == 1
    assert program.incomplete_reason == "function limit reached"
    assert program.status == "partial"
    assert program.function_status("Vault", "b") == "not_indexed"


def test_internal_call_is_represented(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw(uint256 amount) external {
            _pull(amount);
        }
        function _pull(uint256 amount) internal {
            totalAssets -= amount;
        }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    function = program.functions_named("withdraw", "Vault")[0]
    assert any(
        site.callee == "_pull" and site.call_type == "internal" for site in function.call_sites
    )
    flow = analyze_dataflow(program)
    assert flow.reachable_functions("withdraw")
    assert flow.status == "available"


def test_delegatecall_target_provenance(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        function run(address target, bytes calldata data) external {
            target.delegatecall(data);
        }
    }
    """
    program = build_semantic_program(_graph(tmp_path, source))
    flow = analyze_dataflow(program)
    assert flow.target_provenance("run") == "attacker"


def test_cyclic_calls_stop(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Loop {
        function a() external { b(); }
        function b() external { a(); }
    }
    """
    flow = analyze_dataflow(build_semantic_program(_graph(tmp_path, source)))
    assert any(item.cyclic or item.incomplete for item in flow.summaries.values())


def test_unpinned_import_inherits_exact_parent(tmp_path: Path) -> None:
    groups, omitted = _version_groups(
        {
            "A.sol": 'pragma solidity =0.8.20;\nimport "./B.sol";\ncontract A {}',
            "B.sol": "contract B {}",
        },
        [],
        tmp_path,
    )
    assert "B.sol" in groups["0.8.20"]
    assert "B.sol" not in omitted


def test_build_settings_enter_standard_json() -> None:
    payload = project_standard_json(
        {"A.sol": "contract A {}"},
        ["lib/=lib/"],
        {"optimizer": "true", "optimizer_runs": "50", "via_ir": "true", "evm_version": "cancun"},
    )
    settings = payload["settings"]
    assert isinstance(settings, dict)
    assert settings["optimizer"] == {"enabled": True, "runs": 50}
    assert settings["viaIR"] is True
    assert settings["evmVersion"] == "cancun"


def test_fabricated_execution_node_does_not_verify() -> None:
    class Session:
        def __init__(self) -> None:
            self.id = "s"
            self.project_id = "p"
            self.graph = EvidenceGraph(session_id="s", project_id="p")
            self.findings = []

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


def test_mock_oob_is_not_the_production_provider() -> None:
    provider = production_provider()
    assert provider.kind == "none"
    assert provider.available() is False
    assert MockOobProvider().kind == "mock"
    assert OobProvider().kind == "none"


def test_false_negative_record_is_present() -> None:
    assert FALSE_NEGATIVES
    assert {item["phase"] for item in FALSE_NEGATIVES} == {"38"}
