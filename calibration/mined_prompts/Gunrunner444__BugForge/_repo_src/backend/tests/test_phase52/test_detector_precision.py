"""Phase 52 Slice D: detector precision for read-only reentrancy, transient storage and
EIP-7702 assumptions -- positive, safe, near-miss and unknown cases."""

from __future__ import annotations

from app.parsing.solidity_high_value import analyze_high_value
from app.parsing.solidity_research import SemanticCandidate, build_research_model

RO = "read_only_reentrancy.view_observes_mid_update"
TS = "transient_storage.not_cleared"
EOA = "eip7702.eoa_assumption"


def _found(source: str, detector: str) -> list[SemanticCandidate]:
    model = build_research_model({"C.sol": "pragma solidity ^0.8.24;\n" + source})
    return [c for c in analyze_high_value(model) if c.detector == detector]


# ---- read-only reentrancy ---------------------------------------------------------------------

_RO_HELPER = """
contract Vault {
    uint public totalShares;
    function withdraw(uint amt) external {
        _send(msg.sender, amt);
        totalShares -= amt;
    }
    function _send(address to, uint amt) internal {
        (bool ok,) = to.call{value: amt}("");
        require(ok);
    }
    function price() external view returns (uint) { return address(this).balance / totalShares; }
}
"""


def test_ro_positive_through_an_internal_helper() -> None:
    (hit,) = _found(_RO_HELPER, RO)
    assert dict(hit.facts)["external_call_via"] == "helper:_send"
    assert hit.function == "price()" and hit.verified is False


def test_ro_safe_when_the_view_checks_the_lock() -> None:
    source = _RO_HELPER.replace(
        "function price() external view returns (uint) {",
        "function price() external view returns (uint) {\n"
        '        require(!_reentrancyGuardEntered(), "locked");',
    ).replace(
        "uint public totalShares;",
        "uint public totalShares;\n    function _reentrancyGuardEntered() internal view returns (bool) "
        "{ return false; }",
    )
    assert _found(source, RO) == []


def test_ro_near_miss_view_over_a_variable_written_before_the_call() -> None:
    source = _RO_HELPER.replace(
        "_send(msg.sender, amt);\n        totalShares -= amt;",
        "totalShares -= amt;\n        _send(msg.sender, amt);",
    )
    assert _found(source, RO) == []


def test_ro_unknown_external_interface_call_is_not_claimed() -> None:
    # a call to an arbitrary interface (no value, not a transfer) may or may not hand
    # control to an attacker; the detector does not claim it
    source = """
    interface IHook { function ping() external; }
    contract Vault {
        uint public totalShares;
        IHook hook;
        function withdraw(uint amt) external { hook.ping(); totalShares -= amt; }
        function price() external view returns (uint) { return totalShares; }
    }
    """
    assert _found(source, RO) == []


# ---- transient storage ------------------------------------------------------------------------

_TS_PAIR = """
contract T {
    function _lock() internal { assembly { tstore(0x1, 1) } }
    function _unlock() internal { assembly { tstore(0x1, 0) } }
    function enter() external {
        _lock();
        _unlock();
    }
}
"""


def test_ts_positive_entry_point_leaves_a_slot_set_via_a_helper() -> None:
    source = _TS_PAIR.replace("        _unlock();\n", "")
    (hit,) = _found(source, TS)
    assert hit.function == "enter()"
    assert dict(hit.facts)["written_via"] == "helper:_lock"


def test_ts_safe_lock_unlock_pair_is_not_left_set() -> None:
    assert _found(_TS_PAIR, TS) == []


def test_ts_safe_modifier_sets_and_clears() -> None:
    source = """
    contract T {
        modifier guarded() {
            assembly { tstore(0x2, 1) }
            _;
            assembly { tstore(0x2, 0) }
        }
        function enter() external guarded { }
    }
    """
    assert _found(source, TS) == []


def test_ts_near_miss_transient_variable_reset_to_false() -> None:
    source = """
    contract T {
        bool transient entered;
        function enter() external {
            entered = true;
            entered = false;
        }
    }
    """
    assert _found(source, TS) == []
    leaky = source.replace("            entered = false;\n", "")
    assert [c.function for c in _found(leaky, TS)] == ["enter()"]


def test_ts_unknown_computed_slots_are_not_claimed() -> None:
    source = """
    contract T {
        function enter(bytes32 key) external {
            bytes32 slot = keccak256(abi.encode(key));
            assembly { tstore(slot, 1) }
            _clear(key);
        }
        function _clear(bytes32 key) internal {
            bytes32 s = keccak256(abi.encode(key));
            assembly { tstore(s, 0) }
        }
    }
    """
    assert _found(source, TS) == []


# ---- EIP-7702 ---------------------------------------------------------------------------------


def test_eoa_positive_modifier_gate_reaches_the_functions_using_it() -> None:
    source = """
    contract G {
        modifier onlyEOA() { require(msg.sender == tx.origin, "eoa"); _; }
        function mint() external onlyEOA { }
        function burn() external { }
    }
    """
    hits = _found(source, EOA)
    assert [h.function for h in hits] == ["mint()"]
    facts = dict(hits[0].facts)
    assert facts["via"] == "modifier:onlyEOA" and facts["subject"] == "caller"
    assert hits[0].confidence == "medium"


def test_eoa_positive_if_revert_contract_form() -> None:
    source = """
    contract G {
        error NoContracts();
        function act() external {
            if (msg.sender.code.length > 0) revert NoContracts();
        }
    }
    """
    (hit,) = _found(source, EOA)
    assert dict(hit.facts)["gate"] == "if_revert"


def test_eoa_safe_requiring_a_contract_caller() -> None:
    source = """
    contract G {
        function act() external { require(msg.sender != tx.origin, "contracts only"); }
    }
    """
    assert _found(source, EOA) == []


def test_eoa_near_miss_factory_deploy_check_is_not_an_actor_gate() -> None:
    source = """
    contract Factory {
        function create(bytes32 salt) external returns (address predicted) {
            predicted = address(uint160(uint256(salt)));
            if (predicted.code.length == 0) { predicted = address(0); }
            require(predicted.code.length == 0, "already deployed");
        }
    }
    """
    assert _found(source, EOA) == []


def test_eoa_near_miss_branch_that_does_not_revert_and_strings() -> None:
    source = """
    contract G {
        event Direct();
        function act() external {
            if (msg.sender == tx.origin) { emit Direct(); }
            require(true, "msg.sender == tx.origin");
        }
    }
    """
    assert _found(source, EOA) == []


def test_eoa_parameter_subject_is_reported_with_low_confidence() -> None:
    source = """
    contract G {
        function pay(address to) external { require(to.code.length == 0, "eoa only"); }
    }
    """
    (hit,) = _found(source, EOA)
    assert dict(hit.facts)["subject"] == "parameter" and hit.confidence == "low"
