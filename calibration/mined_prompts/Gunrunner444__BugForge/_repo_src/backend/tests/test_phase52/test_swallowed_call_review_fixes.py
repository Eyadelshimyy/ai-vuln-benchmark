"""Review fixes for ``caller_context.swallowed_call_failure``.

1. Empty calldata is a plain transfer only with an explicit non-zero ``value:``.
2. Failure propagation follows the CFG path taken when the call fails.
3. A gas stipend is bounded only by a dominating ``gasleft()`` check tied to it.

Every result stays a static candidate. Nothing here runs a tool or touches a network.
"""

from __future__ import annotations

import pytest

from app.parsing.solidity_call_outcome import (
    DETECTOR,
    PROPAGATED,
    SWALLOWED,
    UNCERTAIN,
    failure_outcome,
    gas_guard,
)
from app.parsing.solidity_research import SemanticCandidate
from tests.test_phase50.phase50_support import analyze_text

HEAD = "// SPDX-License-Identifier: MIT\npragma solidity 0.8.24;\n"


def _fn(body: str, extra: str = "") -> str:
    return HEAD + (
        "contract C {\n  uint256 limit;\n  address other;\n  bool flagOther;\n"
        "  event Ok(bytes r);\n  event Bad(bytes r);\n  error E();\n"
        f"{extra}"
        "  function run(address target, bytes calldata data, uint256 amount) external {\n"
        f"{body}\n  }}\n}}\n"
    )


def _candidates(text: str) -> list[SemanticCandidate]:
    result = analyze_text(text, ("caller_context",))
    return [c for c in result.candidates if c.detector == DETECTOR]


def _one(text: str) -> SemanticCandidate:
    found = _candidates(text)
    assert len(found) == 1, found
    item = found[0]
    assert item.status == "candidate" and item.verified is False and item.confidence == "low"
    return item


def _outcome(body: str, flag: str = "ok") -> str:
    return failure_outcome(body, body.index("t.call"), flag).status


# ---- 1. empty calldata ------------------------------------------------------------------------


def test_empty_calldata_without_value_is_a_candidate() -> None:
    item = _one(_fn('    (bool ok, ) = target.call("");\n    if (!ok) emit Bad("");'))
    assert dict(item.facts)["call_kind"] == "empty_no_value"


def test_empty_calldata_with_fixed_gas_and_no_value_is_a_candidate() -> None:
    item = _one(_fn('    (bool ok, ) = target.call{gas: 50000}("");\n    ok;'))
    assert "fixed stipend of `50000`" in dict(item.facts)["gas_forwarded"]


def test_empty_calldata_with_variable_value_is_a_labelled_candidate() -> None:
    # v3: no longer exempt; amount may be zero at runtime and the failure is ignored.
    item = _one(_fn('    (bool ok, ) = target.call{value: amount}("");\n    ok;'))
    assert dict(item.facts)["call_kind"] == "eth_transfer_variable"


def test_literal_zero_value_is_not_a_transfer() -> None:
    assert len(_candidates(_fn('    (bool ok, ) = target.call{value: 0}("");\n    ok;'))) == 1


def test_non_empty_calldata_with_value_is_a_candidate() -> None:
    body = "    (bool ok, ) = target.call{value: amount}(data);\n    ok;"
    assert dict(_one(_fn(body)).facts)["call_kind"] == "value_with_calldata"


# ---- 2. failure propagation along the failure path ----------------------------------------------


def test_require_on_success_path_only_does_not_propagate() -> None:
    body = (
        "    (bool ok, bytes memory r) = target.call(data);\n"
        "    if (ok) { require(ok); emit Ok(r); } else { emit Bad(r); }"
    )
    assert dict(_one(_fn(body)).facts)["failure_propagation"] == SWALLOWED


def test_return_flag_on_success_path_only_does_not_propagate() -> None:
    text = HEAD + (
        "contract C {\n  event Bad();\n"
        "  function run(address t, bytes calldata d) external returns (bool) {\n"
        "    (bool ok, ) = t.call(d);\n    if (ok) { return ok; }\n    emit Bad();\n"
        "    return true;\n  }\n}\n"
    )
    item = _one(text)
    assert "returns without the flag" in dict(item.facts)["failure_paths"]


def test_failed_branch_with_non_reverting_assembly_is_a_candidate() -> None:
    body = (
        "    (bool ok, bytes memory r) = target.call(data);\n"
        "    if (!ok) { assembly { let size := mload(r) } emit Bad(r); }"
    )
    assert dict(_one(_fn(body)).facts)["failure_propagation"] == SWALLOWED


def test_failed_branch_that_bubbles_revert_data_propagates() -> None:
    body = (
        "    (bool ok, bytes memory r) = target.call(data);\n"
        "    if (!ok) { assembly { revert(add(r, 32), mload(r)) } }"
    )
    assert _candidates(_fn(body)) == []


def test_conditional_assembly_revert_is_uncertain_and_kept() -> None:
    body = (
        "    (bool ok, bytes memory r) = target.call(data);\n"
        "    if (!ok) { assembly { if gt(mload(r), 0) { revert(add(r, 32), mload(r)) } } }"
    )
    facts = dict(_one(_fn(body)).facts)
    assert facts["failure_propagation"] == SWALLOWED
    assert "only conditionally" in facts["failure_paths"]


@pytest.mark.parametrize(
    "check",
    [
        "require(ok);",
        'require(ok, "call failed");',
        "assert(ok);",
        "if (!ok) revert E();",
        'if (!ok) { emit Bad(""); revert E(); }',
        "if (ok == false) { revert E(); }",
        'if (ok) { emit Ok(""); } else { revert E(); }',
    ],
)
def test_real_failure_check_after_the_call_propagates(check: str) -> None:
    assert _candidates(_fn(f"    (bool ok, ) = target.call(data);\n    {check}")) == []


def test_unrelated_check_on_another_variable_does_not_suppress() -> None:
    body = (
        "    (bool ok, ) = target.call(data);\n    require(flagOther);\n"
        '    if (!flagOther) revert E();\n    emit Bad("");'
    )
    assert dict(_one(_fn(body)).facts)["failure_propagation"] == SWALLOWED


def test_nested_failure_branch_that_always_reverts_propagates() -> None:
    body = (
        "    (bool ok, bytes memory r) = target.call(data);\n"
        "    if (!ok) {\n      if (r.length > 0) { emit Bad(r); revert E(); }\n"
        "      else { revert E(); }\n    }"
    )
    assert _candidates(_fn(body)) == []


def test_nested_failure_branch_with_a_normal_exit_is_a_candidate() -> None:
    body = (
        "    (bool ok, bytes memory r) = target.call(data);\n"
        "    if (!ok) {\n      if (r.length > 0) { revert E(); }\n      emit Bad(r);\n    }"
    )
    assert dict(_one(_fn(body)).facts)["failure_propagation"] == SWALLOWED


def test_compound_condition_on_the_flag_is_uncertain_and_kept() -> None:
    body = "    (bool ok, ) = target.call(data);\n    if (!ok && limit > 0) revert E();"
    facts = dict(_one(_fn(body)).facts)
    # The condition-true branch reverts; the other branch falls through, so it is kept.
    assert facts["failure_propagation"] == SWALLOWED
    assert "compound condition" in facts["failure_paths"]


def test_flag_overwritten_before_check_is_kept() -> None:
    body = "    (bool ok, ) = target.call(data);\n    (ok, ) = other.call(data);\n    require(ok);"
    found = _candidates(_fn(body))
    assert len(found) == 1 and "target.call" in found[0].observed[0]
    assert dict(found[0].facts)["failure_propagation"] == UNCERTAIN


def test_failure_outcome_helper_statuses() -> None:
    assert _outcome(" (bool ok, ) = t.call(d); require(ok); ") == PROPAGATED
    assert _outcome(" (bool ok, ) = t.call(d); if (ok) { require(ok); } ") == SWALLOWED


# ---- 3. gas guard dominance and association ----------------------------------------------------


def _gas(body: str) -> list[SemanticCandidate]:
    return _candidates(_fn(body))


SW = '    if (!ok) emit Bad("");'


def test_dominating_guard_tied_to_the_stipend_bounds_the_call() -> None:
    body = (
        "    if (gasleft() < limit + limit / 63 + 7000) revert E();\n"
        f"    (bool ok, ) = target.call{{gas: limit}}(data);\n{SW}"
    )
    assert _gas(body) == []


def test_guard_through_a_local_variable_bounds_the_call() -> None:
    body = (
        "    uint256 required = limit + limit / 63 + 7000;\n"
        '    require(gasleft() >= required, "gas");\n'
        f"    (bool ok, ) = target.call{{gas: limit}}(data);\n{SW}"
    )
    assert _gas(body) == []


def test_gas_check_after_the_call_does_not_bound_it() -> None:
    body = (
        "    (bool ok, ) = target.call{gas: limit}(data);\n"
        "    if (gasleft() < limit) revert E();\n" + SW
    )
    facts = dict(_one(_fn(body)).facts)
    assert facts["gas_guard"] == "NO_GUARD"
    assert "does not protect this call" in facts["gas_guard_reason"]
    assert "all remaining gas" not in facts["gas_forwarded"]
    assert "fixed stipend of `limit`" in facts["gas_forwarded"]


def test_unrelated_gas_check_does_not_bound_the_call() -> None:
    body = (
        f"    require(gasleft() > 21000);\n    (bool ok, ) = target.call{{gas: limit}}(data);\n{SW}"
    )
    facts = dict(_one(_fn(body)).facts)
    assert facts["gas_guard"] == "NO_GUARD"
    assert "does not reference this call's stipend" in facts["gas_guard_reason"]


def test_guard_that_protects_a_different_call_does_not_bound_this_one() -> None:
    body = (
        "    if (amount > 0) {\n      if (gasleft() < limit + 7000) revert E();\n"
        "      (bool a, ) = other.call{gas: limit}(data);\n      a;\n    }\n"
        f"    (bool ok, ) = target.call{{gas: limit}}(data);\n{SW}"
    )
    found = _one(_fn(body))
    # v3: the guard on other.call is unproven (no /63 term), so other.call is reported
    # first; target.call's own site has no guard protecting it.
    sites = dict(found.facts)["call_sites"]
    assert "GUARD_PRESENT_BUT_INSUFFICIENT_OR_UNPROVEN" in sites
    assert "stipend limit, NO_GUARD" in sites


def test_conditional_guard_with_unbounded_fallback_reports_the_fallback() -> None:
    body = (
        "    bool ok;\n    bytes memory r;\n"
        "    if (limit > 0) {\n      if (gasleft() < limit + limit / 63 + 7000) revert E();\n"
        "      (ok, r) = target.call{gas: limit}(data);\n"
        "    } else {\n      (ok, r) = target.call(data);\n    }\n" + SW
    )
    item = _one(_fn(body))
    facts = dict(item.facts)
    assert item.observed[0].startswith("target.call(data)")
    assert facts["bounded_call_paths"] == "1"
    assert "all remaining gas" in facts["gas_forwarded"]


def test_guard_that_does_not_stop_low_gas_is_not_a_guard() -> None:
    body = (
        '    if (gasleft() < limit) { emit Bad(""); }\n'
        f"    (bool ok, ) = target.call{{gas: limit}}(data);\n{SW}"
    )
    assert len(_gas(body)) == 1


def test_fixed_gas_without_guard_is_described_as_a_stipend() -> None:
    facts = dict(_one(_fn(f"    (bool ok, ) = target.call{{gas: 100000}}(data);\n{SW}")).facts)
    assert facts["gas_guard"] == "NO_GUARD"
    assert facts["gas_guard_reason"] == "no gasleft() check in the function"
    assert "fixed stipend of `100000`" in facts["gas_forwarded"]
    assert "all remaining gas" not in facts["gas_forwarded"]


def test_sufficient_gas_branch_containing_the_call_bounds_it() -> None:
    body = (
        "    bool ok;\n    if (gasleft() >= limit + limit / 63 + 7000) {\n"
        "      (ok, ) = target.call{gas: limit}(data);\n    } else { revert E(); }\n" + SW
    )
    assert _gas(body) == []


def test_gas_guard_helper_reports_no_option() -> None:
    body = " (bool ok, ) = t.call(d); ok; "
    assert gas_guard(body, body.index("t.call"), "").bounded is False


def test_try_catch_on_the_failure_path_is_uncertain_and_kept() -> None:
    body = " (bool ok, ) = t.call(d); try this.f() {} catch {} require(ok); "
    assert failure_outcome(body, body.index("t.call"), "ok").status == UNCERTAIN
