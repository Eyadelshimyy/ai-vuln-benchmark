"""Phase 33 cross-contract relationships and storage compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_cross import (
    analyze_project,
    build_project,
    project_context_active,
    reset_project_context,
    set_project_context,
)
from app.parsing.solidity_proxy import analyze_proxy
from app.parsing.solidity_storage import analyze_storage, compare_storage_layouts
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _graph(tmp_path: Path, source: str, name: str):
    return parse_source("solidity", tmp_path / name, source)


def _ids(tmp_path: Path, source: str) -> set[str]:
    path = tmp_path / "Contract.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert all(item.metadata.get("verified") != "true" for item in result.observations)
    return {item.rule_id for item in result.observations}


def test_direct_interface_ambiguous_unresolved_and_library_calls(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    interface IERC20 { function transferFrom(address from, address to, uint256 amount) external returns (bool); }
    library Math { function add(uint256 a, uint256 b) internal pure returns (uint256) { return a + b; } }
    contract Left { function ping() external pure returns (uint256) { return 1; } }
    contract Right { function ping() external pure returns (uint256) { return 2; } }
    contract Router {
        IERC20 token;
        function direct() external pure returns (uint256) { return Math.add(1, 2); }
        function tokenCall() external { token.transferFrom(msg.sender, address(this), 1); }
        function ambiguous() external pure returns (uint256) { return ping(); }
        function missing() external pure returns (uint256) { return missingHelper(); }
    }
    """
    model = analyze_project({str(tmp_path / "P.sol"): _graph(tmp_path, source, "P.sol")})
    kinds = {(item.caller_function, item.kind, item.status) for item in model.calls}
    assert ("direct", "library", "resolved") in kinds or ("direct", "direct", "resolved") in kinds
    assert ("tokenCall", "token", "resolved") in kinds
    assert ("ambiguous", "direct", "ambiguous") in kinds
    assert ("missing", "direct", "unresolved") in kinds


def test_state_reaches_helper_call_and_branch_stays_unknown(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) balances;
        function withdraw() external {
            uint256 amount = balances[msg.sender];
            _send(amount);
            balances[msg.sender] = 0;
        }
        function _send(uint256 amount) internal {
            msg.sender.call{value: amount}("");
        }
        function maybe(bool flag) external {
            uint256 amount = balances[msg.sender];
            if (flag) { _send(amount); }
        }
    }
    """
    model = analyze_project({str(tmp_path / "V.sol"): _graph(tmp_path, source, "V.sol")})
    known = [item for item in model.flows if item.function == "withdraw" and item.known]
    branched = [item for item in model.flows if item.function == "maybe"]
    assert known
    assert any(item.kind in {"helper", "state_to_call"} for item in known)
    assert branched and all(item.branch_sensitive and not item.known for item in branched)
    assert any(item.kind == "reentrancy" and item.function == "withdraw" for item in model.findings)
    assert "sol.cross_contract" in _ids(tmp_path, source)
    late = """
    pragma solidity ^0.8.20;
    contract VaultLate {
        mapping(address => uint256) balances;
        function withdraw() external {
            uint256 amount = balances[msg.sender];
            _send(amount);
        }
        function _send(uint256 amount) internal {
            msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
    late_model = analyze_project({str(tmp_path / "L.sol"): _graph(tmp_path, late, "L.sol")})
    assert any(
        item.kind == "reentrancy" and item.contract == "VaultLate" for item in late_model.findings
    )


def test_unrelated_call_is_not_reentrancy(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Ping {
        function poke(address other) external {
            other.call("");
        }
    }
    """
    model = analyze_project({str(tmp_path / "N.sol"): _graph(tmp_path, source, "N.sol")})
    assert not any(item.kind == "reentrancy" for item in model.findings)


def test_token_callback_needs_shared_state(tmp_path: Path) -> None:
    linked = """
    pragma solidity ^0.8.20;
    interface IERC20 { function transfer(address to, uint256 amount) external returns (bool); }
    contract Vault {
        IERC20 token;
        mapping(address => uint256) balances;
        function deposit(uint256 amount) external {
            uint256 beforeBalance = balances[msg.sender];
            token.transfer(address(this), beforeBalance);
        }
        function tokensReceived(address from, uint256 amount) external {
            balances[from] = amount;
        }
    }
    """
    model = analyze_project({str(tmp_path / "C.sol"): _graph(tmp_path, linked, "C.sol")})
    assert any(item.kind == "callback" for item in model.calls)
    assert any(item.kind == "reentrancy" for item in model.findings)


def test_proxy_and_yul_edges_stay_structural(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    interface IBeacon { function implementation() external view returns (address); }
    contract Proxy {
        address implementation;
        address beacon;
        bytes32 constant SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
        fallback() external payable {
            address impl = IBeacon(beacon).implementation();
            impl.delegatecall(msg.data);
        }
        function yul() external {
            assembly {
                let impl := sload(SLOT)
                let ok := delegatecall(gas(), impl, 0, 0, 0, 0)
            }
        }
    }
    """
    graph = _graph(tmp_path, source, "Proxy.sol")
    proxy = analyze_proxy(graph)
    assert any(item.provenance == "storage_slot" for item in proxy.delegates)
    assert "beacon-like" in proxy.patterns
    model = analyze_project({str(tmp_path / "Proxy.sol"): graph})
    assert any(item.kind in {"delegate", "proxy"} for item in model.calls)


def test_authorization_relationships(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Ownable {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    contract Child is Ownable {
        address owner;
        function set(uint256 value) external onlyOwner { value = value; }
    }
    contract A { modifier onlyOwner() { require(msg.sender == msg.sender); _; } }
    contract B { modifier onlyOwner() { require(tx.origin == msg.sender); _; } }
    contract Ambiguous is A, B {
        address owner;
        function set(uint256 value) external onlyOwner { value = value; }
    }
    interface IAuth { function hasRole(address who) external view returns (bool); }
    contract Open {
        IAuth auth;
        uint256 value;
        function set(uint256 next) external { require(auth.hasRole(msg.sender)); value = next; }
    }
    """
    model = analyze_project({str(tmp_path / "A.sol"): _graph(tmp_path, source, "A.sol")})
    summaries = " ".join(item.summary for item in model.findings if item.kind == "authorization")
    assert "unknown" in summaries.lower() or "not resolved" in summaries.lower()
    assert any(item.kind == "authorization" and item.known for item in model.flows)


def test_storage_relationships_and_namespace(tmp_path: Path) -> None:
    left = analyze_storage(
        _graph(
            tmp_path,
            "pragma solidity ^0.8.20; contract Token { uint256 value; uint256 amount; }",
            "L.sol",
        )
    )
    right = analyze_storage(
        _graph(
            tmp_path,
            "pragma solidity ^0.8.20; contract Token { address owner; uint256 value; uint256 amount; }",
            "R.sol",
        )
    )
    changed = compare_storage_layouts(left, right, "Token", "Token", "version")
    assert changed.compatible is False
    same = compare_storage_layouts(left, left, "Token", "Token", "version")
    assert same.compatible is True
    packed = analyze_storage(
        _graph(
            tmp_path, "pragma solidity ^0.8.20; contract Token { uint128 a; uint128 b; }", "P.sol"
        )
    )
    wider = analyze_storage(
        _graph(
            tmp_path, "pragma solidity ^0.8.20; contract Token { uint256 a; uint128 b; }", "W.sol"
        )
    )
    packing = compare_storage_layouts(packed, wider, "Token", "Token", "version")
    assert packing.compatible is False
    names = analyze_storage(
        _graph(
            tmp_path,
            "pragma solidity ^0.8.20; contract Names { bytes32 constant SLOT = keccak256(abi.encodePacked(msg.sender)); uint256 value; }",
            "Ns.sol",
        )
    )
    assert names.namespaces and names.namespaces[0].known is False


def test_project_context_does_not_leak(tmp_path: Path) -> None:
    first = {
        "a": _graph(
            tmp_path,
            "pragma solidity ^0.8.20; contract One { function a() external {} }",
            "One.sol",
        )
    }
    second = {
        "b": _graph(
            tmp_path,
            "pragma solidity ^0.8.20; contract Two { function b() external {} }",
            "Two.sol",
        )
    }
    token = set_project_context(first)
    try:
        assert any(item.name == "One" for item in analyze_project().contracts)
    finally:
        reset_project_context(token)
    assert project_context_active() is False
    other = build_project(second)
    assert all(item.name != "One" for item in other.contracts)


def test_economic_edges_use_the_defi_model(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    interface IERC20 { function transferFrom(address from, address to, uint256 amount) external returns (bool); }
    contract Vault {
        IERC20 token;
        mapping(address => uint256) shares;
        function deposit(uint256 amount) external {
            token.transferFrom(msg.sender, address(this), amount);
            shares[msg.sender] += amount;
        }
    }
    """
    model = analyze_project({str(tmp_path / "E.sol"): _graph(tmp_path, source, "E.sol")})
    assert model.economics
    assert any("transferFrom" in item or "in" in item for item in model.economics)


def test_unrelated_callback_names_are_not_reentrancy(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    interface IERC20 { function transfer(address to, uint256 amount) external returns (bool); }
    contract Other {
        mapping(address => uint256) balances;
        function tokensReceived(address from, uint256 amount) external {
            balances[from] = amount;
        }
    }
    contract Vault {
        IERC20 token;
        mapping(address => uint256) balances;
        function deposit() external {
            uint256 beforeBalance = balances[msg.sender];
            token.transfer(msg.sender, beforeBalance);
        }
    }
    """
    model = analyze_project({str(tmp_path / "U.sol"): _graph(tmp_path, source, "U.sol")})
    assert not any(item.kind == "reentrancy" for item in model.findings)


def test_duplicate_symbols_stay_ambiguous(tmp_path: Path) -> None:
    left = "pragma solidity ^0.8.20; contract Math { function add(uint256 a) external {} }"
    right = "pragma solidity ^0.8.20; contract Math { function add(uint256 a) external {} contract User { function go() external { add(1); } } }"
    model = analyze_project(
        {
            str(tmp_path / "A.sol"): _graph(tmp_path, left, "A.sol"),
            str(tmp_path / "B.sol"): _graph(tmp_path, right, "B.sol"),
        }
    )
    assert any(item.status == "ambiguous" for item in model.calls)


def test_project_guard_matches_function_guard(tmp_path: Path) -> None:
    from app.parsing.solidity_guards import reentrancy_guard_holds

    body = "{ require(!locked); locked = true; _; locked = false; }"
    assert reentrancy_guard_holds(body)
    source = f"""
    pragma solidity ^0.8.20;
    contract Vault {{
        mapping(address => uint256) balances;
        bool locked;
        modifier nonReentrant() {{ {body} }}
        function withdraw() external nonReentrant {{
            uint256 amount = balances[msg.sender];
            msg.sender.call{{value: amount}}("");
            balances[msg.sender] = 0;
        }}
    }}
    """
    model = analyze_project({str(tmp_path / "G.sol"): _graph(tmp_path, source, "G.sol")})
    assert not any(
        item.kind == "reentrancy" and item.function == "withdraw" for item in model.findings
    )
