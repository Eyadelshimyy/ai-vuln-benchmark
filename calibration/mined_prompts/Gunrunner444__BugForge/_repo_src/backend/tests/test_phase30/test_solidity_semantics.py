"""Phase 30 Solidity rule matrix: positive, negative, near-miss, and adversarial cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.keccak import function_selector
from app.parsing.solidity_compiler import compiler_semantics
from app.parsing.solidity_links import selector_for_function, unique_abi_aliases
from app.parsing.solidity_loops import loop_is_bounded
from app.parsing.solidity_types import (
    canonical_solidity_type,
    canonical_with_aliases,
    is_dynamic_type,
)
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _ids(tmp_path: Path, source: str) -> set[str]:
    path = tmp_path / "Contract.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert all(item.metadata.get("status") == "potential" for item in result.observations)
    assert all(item.metadata.get("verified") != "true" for item in result.observations)
    return {item.rule_id for item in result.observations}


def test_authorization_uses_modifier_bodies_and_branches(tmp_path: Path) -> None:
    empty = """
    pragma solidity ^0.8.20;
    contract P {
        address owner;
        modifier onlyOwner() { _; }
        function setOwner(address next) external onlyOwner { owner = next; }
    }
    """
    real = """
    pragma solidity ^0.8.20;
    contract P {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
        function setOwner(address next) external onlyOwner { owner = next; }
    }
    """
    helper = """
    pragma solidity ^0.8.20;
    contract P {
        address owner;
        function _auth() internal { require(msg.sender == owner); }
        function setOwner(address next) external { _auth(); owner = next; }
    }
    """
    late_helper = """
    pragma solidity ^0.8.20;
    contract P {
        address owner;
        function _auth() internal { require(msg.sender == owner); }
        function setOwner(address next) external { owner = next; _auth(); }
    }
    """
    business = """
    pragma solidity ^0.8.20;
    contract P {
        mapping(address => uint) balances;
        function deposit() external payable { balances[msg.sender] += msg.value; }
        function withdraw() external view returns (uint) { return balances[msg.sender]; }
    }
    """
    assert "sol.missing_authorization" in _ids(tmp_path, empty)
    assert "sol.missing_authorization" not in _ids(tmp_path, real)
    assert "sol.missing_authorization" not in _ids(tmp_path, helper)
    assert "sol.missing_authorization" in _ids(tmp_path, late_helper)
    assert "sol.missing_authorization" not in _ids(tmp_path, business)


def test_reentrancy_tracks_control_transfer_and_shared_state(tmp_path: Path) -> None:
    cross = """
    pragma solidity ^0.8.20;
    contract V {
        mapping(address => uint) balances;
        function withdraw() external {
            uint amount = balances[msg.sender];
            msg.sender.call{value: amount}("");
            claim();
        }
        function claim() public { balances[msg.sender] = 0; }
    }
    """
    local_helper = """
    pragma solidity ^0.8.20;
    contract V {
        uint counter;
        function foo(address t) external {
            t.call("");
            helper();
        }
        function helper() internal { uint local = counter; local += 1; }
    }
    """
    unrelated = """
    pragma solidity ^0.8.20;
    contract V {
        uint counter;
        function poke(address t) external {
            t.call("");
            counter = 2;
        }
    }
    """
    empty_guard = """
    pragma solidity ^0.8.20;
    contract V {
        mapping(address => uint) balances;
        modifier nonReentrant() { _; }
        function withdraw() external nonReentrant {
            uint amount = balances[msg.sender];
            msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
    real_guard = """
    pragma solidity ^0.8.20;
    contract V {
        bool locked;
        mapping(address => uint) balances;
        modifier nonReentrant() { require(!locked); locked = true; _; locked = false; }
        function withdraw() external nonReentrant {
            uint amount = balances[msg.sender];
            msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
    assert "sol.cross_function_reentrancy" in _ids(tmp_path, cross)
    assert "sol.reentrancy" not in _ids(tmp_path, cross)
    assert "sol.cross_function_reentrancy" not in _ids(tmp_path, local_helper)
    assert "sol.reentrancy" not in _ids(tmp_path, unrelated)
    assert "sol.reentrancy" in _ids(tmp_path, empty_guard)
    assert "sol.reentrancy" not in _ids(tmp_path, real_guard)


def test_callback_and_branch_reentrancy(tmp_path: Path) -> None:
    receiver = """
    pragma solidity ^0.8.20;
    contract N {
        mapping(address => uint) balances;
        function onERC721Received(address, address, uint, bytes calldata) external returns (bytes4) {
            msg.sender.call("");
            balances[msg.sender] += 1;
            return this.onERC721Received.selector;
        }
    }
    """
    branched = """
    pragma solidity ^0.8.20;
    contract V {
        mapping(address => uint) balances;
        uint counter;
        function poke(address t) external {
            if (t == address(0)) { t.call(""); }
            else { balances[msg.sender] = 0; }
        }
    }
    """
    assert "sol.callback_reentrancy" in _ids(tmp_path, receiver)
    assert "sol.reentrancy" not in _ids(tmp_path, branched)


def test_loops_use_the_header_bound(tmp_path: Path) -> None:
    assert loop_is_bounded(
        "for (uint i; i < items.length; i++) { items[i] = 1; }", "uint[4] memory items;"
    )
    assert not loop_is_bounded(
        'for (uint i; i < users.length; i++) { users[i].call(""); }',
        "uint[4] memory decoy; address[] calldata users;",
    )
    dynamic = """
    pragma solidity ^0.8.20;
    contract L {
        function pay(address[] calldata users) external {
            for (uint i; i < users.length; i++) { users[i].call(""); }
        }
    }
    """
    fixed = """
    pragma solidity ^0.8.20;
    contract L {
        function pay(address[4] calldata users) external {
            for (uint i; i < users.length; i++) { users[i].call(""); }
        }
    }
    """
    decoy = """
    pragma solidity ^0.8.20;
    contract L {
        uint[4] stored;
        function pay(address[] calldata users) external {
            for (uint i; i < users.length; i++) { users[i].call(""); }
        }
    }
    """
    capped = """
    pragma solidity ^0.8.20;
    contract L {
        uint constant MAX = 8;
        function pay(address[] calldata users) external {
            for (uint i; i < min(users.length, MAX); i++) { users[i].call(""); }
        }
    }
    """
    spinning = """
    pragma solidity ^0.8.20;
    contract L {
        uint[] items;
        function grow(uint n) external {
            uint i;
            while (i < n) { items.push(i); i++; }
        }
    }
    """
    do_loop = """
    pragma solidity ^0.8.20;
    contract L {
        uint[] items;
        function grow() external {
            uint i;
            do { items.push(i); i++; } while (i < items.length);
        }
    }
    """
    assert "sol.unbounded_loop" in _ids(tmp_path, dynamic)
    assert "sol.unbounded_loop" not in _ids(tmp_path, fixed)
    assert "sol.unbounded_loop" in _ids(tmp_path, decoy)
    assert "sol.unbounded_loop" not in _ids(tmp_path, capped)
    assert "sol.unbounded_state_loop" in _ids(tmp_path, spinning)
    assert "sol.unbounded_loop" not in _ids(tmp_path, spinning)
    assert "sol.unbounded_state_loop" in _ids(tmp_path, do_loop)


def test_signatures_oracles_and_tokens_follow_values(tmp_path: Path) -> None:
    stale_nonce = """
    pragma solidity ^0.8.20;
    contract S {
        uint nonce;
        function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
            h = keccak256(abi.encode(nonce));
            return ecrecover(h, v, r, s);
        }
    }
    """
    unrelated_oracle = """
    pragma solidity ^0.8.20;
    contract O {
        function price(address feed) external view returns (int) {
            uint updatedAt = 1;
            (, int answer,,,) = feed.latestRoundData();
            require(updatedAt != 0);
            return answer;
        }
    }
    """
    fresh_oracle = """
    pragma solidity ^0.8.20;
    contract O {
        function price(address feed) external view returns (int) {
            (, int answer,, uint updatedAt,) = feed.latestRoundData();
            require(updatedAt != 0 && block.timestamp - updatedAt < 1 hours);
            return answer;
        }
    }
    """
    fee = """
    pragma solidity ^0.8.20;
    contract T {
        function pull(address token, uint amount) external {
            uint before = token.balanceOf(address(this));
            token.transfer(msg.sender, amount);
            before += 1;
        }
    }
    """
    measured = """
    pragma solidity ^0.8.20;
    contract T {
        function pull(address token, uint amount) external {
            uint before = token.balanceOf(address(this));
            token.transfer(msg.sender, amount);
            uint afterBal = token.balanceOf(address(this));
            require(afterBal >= before);
        }
    }
    """
    native = """
    pragma solidity ^0.8.20;
    contract T {
        function pay() external {
            uint before = address(this).balance;
            payable(msg.sender).transfer(before);
        }
    }
    """
    donation = """
    pragma solidity ^0.8.20;
    contract V {
        function shares(address token) external view returns (uint) {
            return token.balanceOf(address(this)) * 1e18 / token.totalSupply();
        }
    }
    """
    split = """
    pragma solidity ^0.8.20;
    contract V {
        function noise(address token, uint totalSupply) external view returns (uint) {
            uint bal = token.balanceOf(msg.sender);
            return bal / 2 + totalSupply;
        }
    }
    """
    assert "sol.signature_replay" in _ids(tmp_path, stale_nonce)
    assert "sol.stale_oracle" in _ids(tmp_path, unrelated_oracle)
    assert "sol.stale_oracle" not in _ids(tmp_path, fresh_oracle)
    assert "sol.fee_on_transfer" in _ids(tmp_path, fee)
    assert "sol.fee_on_transfer" not in _ids(tmp_path, measured)
    assert "sol.fee_on_transfer" not in _ids(tmp_path, native)
    assert "sol.donation_inflation" in _ids(tmp_path, donation)
    assert "sol.donation_inflation" not in _ids(tmp_path, split)


def test_proxy_initializer_and_assembly(tmp_path: Path) -> None:
    broken = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address public implementation;
        bytes32 constant SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
        fallback() external payable { implementation.delegatecall(msg.data); }
    }
    """
    slot_only = """
    pragma solidity ^0.8.20;
    contract Proxy {
        bytes32 constant SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
        fallback() external payable {
            assembly {
                let impl := sload(SLOT)
                let ok := delegatecall(gas(), impl, 0, 0, 0, 0)
            }
        }
    }
    """
    empty_init = """
    pragma solidity ^0.8.20;
    contract Boot {
        modifier initializer() { _; }
        function initialize() external initializer {}
    }
    """
    real_init = """
    pragma solidity ^0.8.20;
    contract Boot {
        bool initialized;
        modifier initializer() { require(!initialized); initialized = true; _; }
        function initialize() external initializer {}
    }
    """
    yul = """
    pragma solidity ^0.8.20;
    contract A {
        function touch() external {
            assembly { let x := sload(0) mstore(0, x) }
        }
    }
    """
    broken_ids = _ids(tmp_path, broken)
    assert "sol.storage_collision" in broken_ids
    slot_ids = _ids(tmp_path, slot_only)
    assert "sol.storage_collision" not in slot_ids
    assert "sol.assembly_sensitive" in slot_ids
    assert "sol.initializer" in _ids(tmp_path, empty_init)
    assert "sol.initializer" not in _ids(tmp_path, real_init)
    yul_ids = _ids(tmp_path, yul)
    assert "sol.assembly_sensitive" not in yul_ids


def test_type_canonicalization_and_imported_selectors(tmp_path: Path) -> None:
    assert canonical_solidity_type("address payable") == "address"
    assert canonical_solidity_type("uint") == "uint256"
    assert canonical_solidity_type("bytes32") == "bytes32"
    assert canonical_solidity_type("((uint256,address),bool)") == "((uint256,address),bool)"
    assert canonical_solidity_type("ufixed128x18") == ""
    assert canonical_solidity_type("uint256[") == ""
    assert is_dynamic_type("((uint256,string)[2])") is True
    assert is_dynamic_type("(uint256,address)[2]") is False
    assert (
        canonical_with_aliases("Point[]", {"Point": "(uint256,uint256)"}) == "(uint256,uint256)[]"
    )
    assert canonical_with_aliases("Missing", {}) == ""
    source = """
    pragma solidity ^0.8.20;
    type Price is uint128;
    enum Kind { A, B }
    interface IToken { function ping() external; }
    struct Item { Kind k; uint256 n; }
    contract C {
        function takeKind(Kind k) external pure {}
        function takePrice(Price p) external pure {}
        function takeToken(IToken token) external pure {}
        function takeItem(Item memory item) external pure {}
        function takeFixed(uint256[3] memory items) external pure {}
        function mystery(Unknown memory item) external pure {}
    }
    """
    graph = parse_source("solidity", tmp_path / "C.sol", source)

    def fields(extra: str) -> dict[str, str]:
        return {
            part.split("=", 1)[0]: part.split("=", 1)[1] for part in extra.split("|") if "=" in part
        }

    functions = {
        fields(event.extra)["function"]: fields(event.extra)
        for event in graph.events
        if event.kind == "sol_function"
    }
    assert functions["takeKind"]["params"] == "uint8"
    assert functions["takeKind"]["selector"] == function_selector("takeKind(uint8)")
    assert functions["takePrice"]["selector"] == function_selector("takePrice(uint128)")
    assert functions["takeToken"]["selector"] == function_selector("takeToken(address)")
    assert functions["takeItem"]["params"] == "(uint8,uint256)"
    assert functions["takeFixed"]["selector"] == function_selector("takeFixed(uint256[3])")
    assert functions["mystery"]["selector_status"] == "unresolved"
    point = parse_source(
        "solidity",
        tmp_path / "Point.sol",
        "pragma solidity ^0.8.20; struct Point { uint256 x; uint256 y; }",
    )
    user = parse_source(
        "solidity",
        tmp_path / "User.sol",
        'pragma solidity ^0.8.20; import "./Point.sol"; contract U { function use(Point memory p) external {} }',
    )
    other = parse_source(
        "solidity",
        tmp_path / "Other.sol",
        "pragma solidity ^0.8.20; struct Point { address who; }",
    )
    aliases = unique_abi_aliases({"Point.sol": point, "User.sol": user})
    params, selector = selector_for_function(user, "use", aliases)
    assert params == "(uint256,uint256)"
    assert selector == function_selector("use((uint256,uint256))")
    ambiguous = unique_abi_aliases({"Point.sol": point, "Other.sol": other, "User.sol": user})
    assert "Point" not in ambiguous
    assert selector_for_function(user, "use", ambiguous) == ("", "")


def test_compiler_overlay_does_not_invent_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiler paths are monkeypatched so the suite does not depend on solc or forge."""
    from app.parsing.solidity_compiler import SolidityCompilerStatus

    def unavailable() -> SolidityCompilerStatus:
        return SolidityCompilerStatus("UNAVAILABLE", "", "compiler absent in this test")

    import app.parsing.solidity_compiler as _compiler_mod

    monkeypatch.setattr("app.parsing.solidity_compiler.solidity_compiler_status", unavailable)
    # Check through the module so the monkeypatch is observed even when a real
    # solc is installed on the host (the test-local import is a stale reference).
    assert _compiler_mod.solidity_compiler_status().status == "UNAVAILABLE"
    absent = compiler_semantics("contract C {}")
    assert absent.status == "UNAVAILABLE"
    assert absent.storage == []
    ignored = compiler_semantics("contract C {}", runner=lambda _source: '{"version":"0.8.20"}')
    assert ignored.status == "UNAVAILABLE"
    assert ignored.storage == []

    def available() -> SolidityCompilerStatus:
        return SolidityCompilerStatus("AVAILABLE", "solc", "test double")

    monkeypatch.setattr("app.parsing.solidity_compiler.solidity_compiler_status", available)
    present = compiler_semantics("contract C {}")
    assert present.status == "AVAILABLE"
    assert present.storage == []

    def explode(_source: str) -> str:
        raise OSError("compiler invocation failed")

    invocation = compiler_semantics("contract C {}", runner=explode)
    assert invocation.status == "FAILED"
    assert invocation.storage == []
    malformed = compiler_semantics("contract C {}", runner=lambda _source: "not-json")
    assert malformed.status == "FAILED"
    assert malformed.storage == []
    errors = compiler_semantics(
        "contract C {}",
        runner=lambda _source: '{"errors":[{"severity":"error","message":"parse error"}]}',
    )
    assert errors.status == "FAILED"
    assert errors.storage == []
    payload = """
    {"contracts":{"C.sol":{"C":{"storageLayout":{"storage":[{"label":"owner","slot":"0"}]}}}},
     "version":"0.8.20"}
    """
    parsed = compiler_semantics("contract C {}", runner=lambda _source: payload)
    assert parsed.status == "AVAILABLE"
    assert parsed.storage == [{"label": "owner", "slot": "0"}]
    assert parsed.compiler_version == "0.8.20"
    version_only = compiler_semantics(
        "contract C {}", runner=lambda _source: '{"version":"0.8.26"}'
    )
    assert version_only.status == "AVAILABLE"
    assert version_only.compiler_version == "0.8.26"
    assert version_only.storage == []
