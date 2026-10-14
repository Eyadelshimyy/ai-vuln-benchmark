"""Authorization through an inherited modifier's internal helper is not a false positive."""

from __future__ import annotations

from app.parsing.solidity_caller_context import has_sender_authorization
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite

PROTECTED = """
pragma solidity ^0.8.20;
contract Guard {
    address public owner;
    function _check() internal view {
        require(msg.sender == owner);
    }
    modifier onlyOwner() {
        _check();
        _;
    }
}
contract Bridge is Guard {
    uint public x;
    function lzReceive(bytes calldata) external onlyOwner {
        x = 1;
    }
}
"""

OPEN = PROTECTED.replace("external onlyOwner", "external")

UNRESOLVED = """
pragma solidity ^0.8.20;
contract Bridge {
    uint public x;
    modifier onlyOwner() {
        _authorize();
        _;
    }
    function lzReceive(bytes calldata) external onlyOwner {
        x = 1;
    }
}
"""


def _hits(text: str) -> list[str]:
    model = build_research_model({"Bridge.sol": text})
    return [
        item.detector
        for item in run_suite(model).candidates
        if item.detector == "bridge.receiver_without_endpoint_check"
    ]


def test_an_inherited_helper_keeps_the_function_from_being_called_unprotected() -> None:
    model = build_research_model({"Bridge.sol": PROTECTED})
    function = model.function("Bridge", "lzReceive")
    assert function is not None
    assert has_sender_authorization(model, function) is True
    assert _hits(PROTECTED) == []


def test_the_same_function_without_the_helper_is_still_reported() -> None:
    model = build_research_model({"Bridge.sol": OPEN})
    function = model.function("Bridge", "lzReceive")
    assert function is not None
    assert has_sender_authorization(model, function) is False
    assert _hits(OPEN) == ["bridge.receiver_without_endpoint_check"]


def test_an_unresolved_auth_helper_is_uncertain_rather_than_unprotected() -> None:
    model = build_research_model({"Bridge.sol": UNRESOLVED})
    function = model.function("Bridge", "lzReceive")
    assert function is not None
    assert has_sender_authorization(model, function) is None
    assert _hits(UNRESOLVED) == []
