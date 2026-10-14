"""Phase 51 highest-value detectors (owner Phase 6): read-only reentrancy, transient
storage misuse, EIP-7702 EOA assumptions. Evidence-based, never keyword-only."""

from __future__ import annotations

from app.parsing.solidity_high_value import FAMILY, analyze_high_value
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import FAMILIES, run_suite


def _detectors(source: str) -> set[str]:
    model = build_research_model({"C.sol": source})
    return {c.detector for c in analyze_high_value(model)}


# ---- read-only reentrancy -------------------------------------------------------------------

_RO_VULN = """
contract Vault {
    mapping(address => uint) public balances;
    uint public totalShares;
    function withdraw() external {
        uint amt = balances[msg.sender];
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        balances[msg.sender] = 0;
        totalShares -= amt;
    }
    function sharePrice() external view returns (uint) {
        return address(this).balance / totalShares;
    }
}
"""

_RO_SAFE = """
contract Vault {
    mapping(address => uint) public balances;
    uint public totalShares;
    function withdraw() external {
        uint amt = balances[msg.sender];
        balances[msg.sender] = 0;
        totalShares -= amt;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
    function sharePrice() external view returns (uint) {
        return address(this).balance / totalShares;
    }
}
"""


def test_read_only_reentrancy_true_positive() -> None:
    assert "read_only_reentrancy.view_observes_mid_update" in _detectors(_RO_VULN)


def test_read_only_reentrancy_safe_cei_order() -> None:
    # state updated before the external call -> no mid-update exposure
    assert "read_only_reentrancy.view_observes_mid_update" not in _detectors(_RO_SAFE)


# ---- transient storage ----------------------------------------------------------------------

_TS_VULN = """
contract T {
    function enter(uint v) external {
        assembly { tstore(0x1, v) }
        _work();
    }
    function _work() internal {}
}
"""

_TS_SAFE = """
contract T {
    function enter(uint v) external {
        assembly { tstore(0x1, v) }
        _work();
        assembly { tstore(0x1, 0) }
    }
    function _work() internal {}
}
"""


def test_transient_not_cleared_true_positive() -> None:
    assert "transient_storage.not_cleared" in _detectors(_TS_VULN)


def test_transient_cleared_is_safe() -> None:
    assert "transient_storage.not_cleared" not in _detectors(_TS_SAFE)


# ---- EIP-7702 -------------------------------------------------------------------------------

_EOA_VULN = """
contract G {
    function act() external {
        require(msg.sender == tx.origin, "no contracts allowed");
        _run();
    }
    function _run() internal {}
}
"""

_EOA_SAFE = """
contract G {
    event Seen(address origin);
    function act() external {
        emit Seen(tx.origin);
        _run();
    }
    function _run() internal {}
}
"""


def test_eip7702_assumption_true_positive() -> None:
    assert "eip7702.eoa_assumption" in _detectors(_EOA_VULN)


def test_eip7702_no_guard_is_safe() -> None:
    # tx.origin merely logged, not used as a security gate
    assert "eip7702.eoa_assumption" not in _detectors(_EOA_SAFE)


def test_code_length_guard_is_detected() -> None:
    source = """
    contract G {
        function act(address a) external {
            require(a.code.length == 0, "contracts only");
            _run();
        }
        function _run() internal {}
    }
    """
    assert "eip7702.eoa_assumption" in _detectors(source)


# ---- integration ----------------------------------------------------------------------------


def test_family_is_registered_in_the_suite() -> None:
    assert FAMILY in FAMILIES
    model = build_research_model({"C.sol": _RO_VULN})
    result = run_suite(model)
    assert FAMILY in result.families_run
    assert any(c.family == FAMILY for c in result.candidates)


def test_candidates_are_never_verified() -> None:
    model = build_research_model({"C.sol": _RO_VULN})
    for candidate in analyze_high_value(model):
        assert candidate.verified is False
        assert candidate.status == "candidate"


# ---- classification (Phase 51 item 8) -------------------------------------------------------


def _classes(tmp_path, source: str) -> set[str]:
    from app.security.engine import SecurityAnalysisEngine

    path = tmp_path / "C.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    return {
        obs.vulnerability_class.value
        for obs in result.observations
        if obs.rule_id == "sol.research.high_value"
    }


def test_read_only_reentrancy_is_classified_as_reentrancy(tmp_path) -> None:
    assert "reentrancy" in _classes(tmp_path, _RO_VULN)


def test_transient_misuse_is_not_classified_as_reentrancy(tmp_path) -> None:
    classes = _classes(tmp_path, _TS_VULN)
    assert "business_logic_risk" in classes
    assert "reentrancy" not in classes


def test_eip7702_assumption_is_classified_as_authorization(tmp_path) -> None:
    classes = _classes(tmp_path, _EOA_VULN)
    assert "authorization_flaw" in classes
    assert "reentrancy" not in classes
