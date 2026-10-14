"""Regression tests: swallowed low-level call failures and helper-based modifier guards.

Found while checking a local research run: a receiver that emits an event on a failed
``target.call`` and returns normally produced no candidate at all, and an ``onlyOwner``
modifier whose body calls ``_validateOwnership()`` / ``_checkOwner()`` was reported as
missing authorization. Nothing here runs a tool or touches a network.
"""

from __future__ import annotations

import pytest

from app.parsing.solidity_call_outcome import (
    DETECTOR,
    PROPAGATED,
    failure_outcome,
    gas_guard,
    success_variable,
)
from app.security.engine import SecurityAnalysisEngine
from tests.test_phase50.phase50_support import analyze_text, where

HEAD = "// SPDX-License-Identifier: MIT\npragma solidity 0.8.24;\n"


def _receiver(handling: str, call: str = "target.call(data)", pre: str = "") -> str:
    return HEAD + (
        "contract Receiver {\n"
        "  event Executed(bytes r);\n  event Failed(bytes r);\n  error CallReverted();\n"
        "  function onReport(address target, bytes calldata data) external {\n"
        f"    {pre}\n"
        "    bool success;\n    bytes memory ret;\n"
        f"    (success, ret) = {call};\n"
        f"    {handling}\n"
        "  }\n}\n"
    )


SWALLOWED = "if (success) { emit Executed(ret); } else { emit Failed(ret); }"


def _flagged(text: str) -> set[tuple[str, str]]:
    return where(analyze_text(text, ("caller_context",)), DETECTOR)


# ---- failed call that returns normally vs. a failure the caller can observe ----------------


def test_failed_call_emitted_but_not_propagated_is_a_candidate() -> None:
    result = analyze_text(_receiver(SWALLOWED), ("caller_context",))
    found = [c for c in result.candidates if c.detector == DETECTOR]
    assert [(c.contract, c.function.split("(")[0]) for c in found] == [("Receiver", "onReport")]
    item = found[0]
    facts = dict(item.facts)
    assert facts["success_variable"] == "success"
    assert facts["failure_propagation"] == "swallowed"
    assert "all remaining gas" in facts["gas_forwarded"]
    assert item.verified is False and item.status == "candidate"
    assert item.confidence == "low"
    assert any("upstream caller" in m for m in item.missing)


@pytest.mark.parametrize(
    "handling",
    [
        "require(success, 'call failed');",
        "if (!success) revert CallReverted();",
        "if (!success) { emit Failed(ret); revert CallReverted(); }",
        "if (!success) { assembly { revert(add(ret, 32), mload(ret)) } }",
        "if (success) { emit Executed(ret); } else { revert CallReverted(); }",
        "assert(success);",
    ],
)
def test_failure_that_reverts_is_not_a_candidate(handling: str) -> None:
    assert _flagged(_receiver(handling)) == set()


def test_failure_returned_to_the_caller_is_not_a_candidate() -> None:
    text = HEAD + (
        "contract Receiver {\n"
        "  function exec(address target, bytes calldata data) external returns (bool) {\n"
        "    (bool ok, ) = target.call(data);\n    return ok;\n  }\n}\n"
    )
    assert _flagged(text) == set()


def test_inline_bool_declaration_is_tracked() -> None:
    text = HEAD + (
        "contract Receiver {\n  event Failed();\n"
        "  function exec(address target, bytes calldata data) external {\n"
        "    (bool ok, ) = target.call(data);\n    if (!ok) emit Failed();\n  }\n}\n"
    )
    assert _flagged(text) == {("Receiver", "exec")}


# ---- caller-chosen gas --------------------------------------------------------------------


def test_fixed_stipend_with_gasleft_revert_is_not_a_candidate() -> None:
    guard = "if (gasleft() < 200000 + 200000 / 63 + 7000) { revert CallReverted(); }"
    text = _receiver(SWALLOWED, call="target.call{gas: 200000}(data)", pre=guard)
    assert _flagged(text) == set()


def test_fixed_stipend_without_gasleft_check_is_still_a_candidate() -> None:
    text = _receiver(SWALLOWED, call="target.call{gas: 200000}(data)")
    assert _flagged(text) == {("Receiver", "onReport")}


def test_optional_guard_with_unbounded_fallback_is_a_candidate() -> None:
    # The AutomationReceiver shape: the guard applies only when a limit is configured.
    text = HEAD + (
        "contract Receiver {\n  uint256 limit;\n  error InsufficientGas();\n"
        "  event Executed(bytes r);\n  event Failed(bytes r);\n"
        "  function onReport(address target, bytes calldata data) external {\n"
        "    bool success;\n    bytes memory ret;\n"
        "    if (limit > 0) {\n"
        "      if (gasleft() < limit + limit / 63 + 7000) { revert InsufficientGas(); }\n"
        "      (success, ret) = target.call{gas: limit}(data);\n"
        "    } else {\n      (success, ret) = target.call(data);\n    }\n"
        f"    {SWALLOWED}\n  }}\n}}\n"
    )
    result = analyze_text(text, ("caller_context",))
    found = [c for c in result.candidates if c.detector == DETECTOR]
    assert len(found) == 1
    assert dict(found[0].facts)["bounded_call_paths"] == "1"


def test_unchecked_call_and_view_functions_are_left_to_other_rules() -> None:
    text = HEAD + (
        "contract Receiver {\n"
        "  function fire(address target, bytes calldata data) external { target.call(data); }\n"
        "  function peek(address t) external view returns (bool ok) {\n"
        "    (ok, ) = t.staticcall('');\n  }\n}\n"
    )
    assert _flagged(text) == set()


def test_plain_value_transfer_is_a_labelled_best_effort_candidate() -> None:
    # v3: ERC-4337-style prefund is no longer exempt; it is a low-confidence candidate
    # that says the ignored failure may be deliberate (possible false positive).
    text = HEAD + (
        "contract Account {\n"
        "  function pay(uint256 missing) external {\n"
        '    (bool ok, ) = payable(msg.sender).call{value: missing}("");\n    ok;\n  }\n}\n'
    )
    assert _flagged(text) == {("Account", "pay")}


def test_helpers_are_precise() -> None:
    # Helpers take a function's inner body text, as RFunction.body holds it.
    body = " bool s; bytes memory r; (s, r) = t.call(d); if (!s) revert E(); "
    at = body.index("t.call")
    assert success_variable(body, at) == "s"
    assert failure_outcome(body, at, "s").status == PROPAGATED
    swallowed = " bool s; bytes memory r; (s, r) = t.call(d); emit X(s); "
    assert failure_outcome(swallowed, swallowed.index("t.call"), "s").status == "swallowed"
    guarded = " require(gasleft() > 9 + 9 / 63 + 5000); (bool s, ) = t.call{gas: 9}(d); s; "
    assert gas_guard(guarded, guarded.index("t.call"), "gas: 9").bounded is True
    weak = " require(gasleft() > 9); (bool s, ) = t.call{gas: 9}(d); s; "
    assert gas_guard(weak, weak.index("t.call"), "gas: 9").bounded is False
    assert gas_guard(guarded, guarded.index("t.call"), "").bounded is False


# ---- modifier guards that delegate to a helper ----------------------------------------------


def _owned(check: str, caller: str = "msg.sender") -> str:
    return HEAD + (
        "contract Owned {\n  address internal s_owner;\n"
        f"  function _validateOwnership() internal view {{ {check.replace('CALLER', caller)} }}\n"
        "  modifier onlyOwner() { _validateOwnership(); _; }\n}\n"
        "contract Vault is Owned {\n  address payable s_sink;\n"
        "  function withdraw(uint256 amount) external onlyOwner { s_sink.transfer(amount); }\n"
        "  function setSink(address payable sink) external onlyOwner { s_sink = sink; }\n}\n"
    )


def _missing_auth(tmp_path, files: dict[str, str]) -> list[tuple[str, int]]:
    paths = []
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, paths)
    return [
        (item.file_path, item.line)
        for item in result.observations
        if item.rule_id == "sol.missing_authorization"
    ]


@pytest.mark.parametrize("caller", ["msg.sender", "_msgSender()"])
def test_modifier_calling_a_guard_helper_is_authorized(tmp_path, caller: str) -> None:
    text = _owned('require(CALLER == s_owner, "Only callable by owner");', caller)
    assert _missing_auth(tmp_path, {"Vault.sol": text}) == []


def test_oz_style_check_owner_with_msg_sender_accessor(tmp_path) -> None:
    text = _owned("if (s_owner != CALLER) { revert(); }", "_msgSender()")
    assert _missing_auth(tmp_path, {"Vault.sol": text}) == []


def test_guard_helper_in_a_base_file_is_followed(tmp_path) -> None:
    base = HEAD + (
        "contract Owned {\n  address internal s_owner;\n"
        "  function _validateOwnership() internal view { require(msg.sender == s_owner); }\n"
        "  modifier onlyOwner() { _validateOwnership(); _; }\n}\n"
    )
    child = HEAD + (
        'import {Owned} from "./Owned.sol";\n'
        "contract Vault is Owned {\n  address payable s_sink;\n"
        "  function setSink(address payable sink) external onlyOwner { s_sink = sink; }\n}\n"
    )
    assert _missing_auth(tmp_path, {"Owned.sol": base, "Vault.sol": child}) == []


def test_helper_that_checks_nothing_still_reports(tmp_path) -> None:
    text = _owned("s_owner;")
    assert _missing_auth(tmp_path, {"Vault.sol": text}) != []


def test_unresolvable_base_modifier_still_reports(tmp_path) -> None:
    # Ownable is imported but its source is not analyzed: no guard is assumed.
    text = HEAD + (
        'import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";\n'
        "contract Vault is Ownable {\n  address payable s_sink;\n"
        "  function withdraw(uint256 amount) external onlyOwner { s_sink.transfer(amount); }\n}\n"
    )
    assert _missing_auth(tmp_path, {"Vault.sol": text}) != []
