"""V3 corrections for ``caller_context.swallowed_call_failure``.

1. Gas-guard sufficiency: NO_GUARD / GUARD_PRESENT_BUT_INSUFFICIENT_OR_UNPROVEN /
   GUARD_PROVEN_SUFFICIENT under a bounded source-text model (stipend, EIP-150 63/64,
   fixed CALL overhead). Only a proven guard removes a call site.
2. Empty-calldata value calls are analysed, not exempt; ambiguity is labelled.
3. Expected emission is not a vulnerability-detection precision score.
4. A detector emission with a caller-supplied address reaches the existing VFCS
   and local harness path. Forge runs only when it is installed; a missing
   compiler is UNAVAILABLE, never a fabricated pass.
"""

from __future__ import annotations

from app.parsing.solidity_call_outcome import (
    CALL_OVERHEAD,
    DETECTOR,
    GUARD_PROVEN,
    GUARD_UNPROVEN,
    NO_GUARD,
    gas_guard,
)
from app.parsing.solidity_research import SemanticCandidate
from tests.test_phase50.phase50_support import analyze_text

HEAD = "// SPDX-License-Identifier: MIT\npragma solidity 0.8.24;\n"
SW = '\n    if (!ok) emit Bad("");'


def _fn(body: str, extra: str = "") -> str:
    return HEAD + (
        "contract C {\n  uint256 limit;\n  event Ok(bytes r);\n  event Bad(bytes r);\n"
        f"  error E();\n{extra}"
        "  function run(address target, bytes calldata data, uint256 amount) external {\n"
        f"{body}\n  }}\n}}\n"
    )


def _found(text: str) -> list[SemanticCandidate]:
    result = analyze_text(text, ("caller_context",))
    return [c for c in result.candidates if c.detector == DETECTOR]


def _one(text: str) -> dict[str, str]:
    found = _found(text)
    assert len(found) == 1, found
    item = found[0]
    assert item.status == "candidate" and item.verified is False and item.confidence == "low"
    return dict(item.facts)


def _guard(body: str, options: str = "gas: limit") -> str:
    return gas_guard(body, body.index("target.call"), options).status


# ---- 1. gas-guard sufficiency -------------------------------------------------------------------


def test_guard_equal_to_the_stipend_is_not_proven() -> None:
    body = "    require(gasleft() >= limit);\n    (bool ok, ) = target.call{gas: limit}(data);" + SW
    facts = _one(_fn(body))
    assert facts["gas_guard"] == GUARD_UNPROVEN
    assert "EIP-150" in facts["gas_guard_reason"]
    assert "fixed stipend of `limit`" in facts["gas_forwarded"]
    assert "all remaining gas" not in facts["gas_forwarded"]


def test_guard_with_insufficient_overhead_is_not_proven() -> None:
    body = (
        "    if (gasleft() < limit + limit / 63 + 100) revert E();\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    facts = _one(_fn(body))
    assert facts["gas_guard"] == GUARD_UNPROVEN
    assert f"at least {CALL_OVERHEAD}" in facts["gas_guard_reason"]


def test_guard_accounting_for_stipend_and_eip150_is_proven() -> None:
    body = (
        "    if (gasleft() < limit + limit / 63 + 7000) revert E();\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    assert _found(_fn(body)) == []
    assert _guard(body) == GUARD_PROVEN
    note = gas_guard(body, body.index("target.call"), "gas: limit").note
    assert "only under the bounded source-text gas model" in note
    assert "not a runtime safety proof" in note


def test_literal_stipend_folds_numerically() -> None:
    ok = "    require(gasleft() >= 210000);\n    (bool ok, ) = target.call{gas: 200000}(data);" + SW
    low = (
        "    require(gasleft() >= 203000);\n    (bool ok, ) = target.call{gas: 200000}(data);" + SW
    )
    assert _guard(ok, "gas: 200000") == GUARD_PROVEN
    assert _guard(low, "gas: 200000") == GUARD_UNPROVEN


def test_guard_through_a_local_and_named_constant_is_proven() -> None:
    extra = "  uint256 private constant GAS_OVERHEAD = 7000;\n"
    body = (
        "    uint256 required = limit + limit / 63 + GAS_OVERHEAD;\n"
        "    if (gasleft() < required) revert E();\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    assert _found(_fn(body, extra)) == []


def test_unresolved_named_overhead_is_unproven() -> None:
    body = (
        "    uint256 required = limit + limit / 63 + amount;\n"
        "    if (gasleft() < required) revert E();\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    facts = _one(_fn(body))
    assert facts["gas_guard"] == GUARD_UNPROVEN
    assert "cannot be evaluated" in facts["gas_guard_reason"]


def test_guard_that_protects_another_call_does_not_bound_this_one() -> None:
    body = (
        "    if (amount > 0) {\n      if (gasleft() < limit + limit / 63 + 7000) revert E();\n"
        "      (bool a, ) = target.call{gas: limit}(data);\n      if (!a) revert E();\n    }\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    found = _found(_fn(body))
    assert len(found) == 1
    facts = dict(found[0].facts)
    assert facts["gas_guard"] == NO_GUARD
    assert "does not protect this call" in facts["gas_guard_reason"]


def test_required_amount_reassigned_before_use_is_unproven() -> None:
    body = (
        "    uint256 required = limit + limit / 63 + 7000;\n    required = 1;\n"
        "    if (gasleft() < required) revert E();\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    facts = _one(_fn(body))
    assert facts["gas_guard"] == GUARD_UNPROVEN
    assert "assigned 2 times" in facts["gas_guard_reason"]


def test_stipend_reassigned_between_check_and_call_is_unproven() -> None:
    body = (
        "    uint256 g = limit;\n    if (gasleft() < g + g / 63 + 7000) revert E();\n"
        "    g = g * 2;\n    (bool ok, ) = target.call{gas: g}(data);" + SW
    )
    facts = _one(_fn(body))
    assert facts["gas_guard"] == GUARD_UNPROVEN
    assert "reassigned between the check and the call" in facts["gas_guard_reason"]


def test_unbounded_fallback_path_is_reported() -> None:
    body = (
        "    bool ok;\n    if (limit > 0) {\n"
        "      if (gasleft() < limit + limit / 63 + 7000) revert E();\n"
        "      (ok, ) = target.call{gas: limit}(data);\n"
        "    } else {\n      (ok, ) = target.call(data);\n    }" + SW
    )
    facts = _one(_fn(body))
    assert facts["gas_guard"] == NO_GUARD
    assert facts["bounded_call_paths"] == "1"
    assert "all remaining gas" in facts["gas_forwarded"]


def test_guard_whose_adequacy_cannot_be_established_is_kept_and_disclosed() -> None:
    body = (
        "    if (gasleft() < limit * 2 - 5) revert E();\n"
        "    (bool ok, ) = target.call{gas: limit}(data);" + SW
    )
    facts = _one(_fn(body))
    assert facts["gas_guard"] == GUARD_UNPROVEN
    assert facts["gas_guard_reason"].startswith("unproven")
    assert "bounded gas model" in facts["analysis"]


# ---- 2. empty-calldata value calls --------------------------------------------------------------


def test_nonzero_literal_transfer_with_ignored_failure_is_a_candidate() -> None:
    facts = _one(_fn('    (bool ok, ) = target.call{value: 1 ether}("");\n    ok;'))
    assert facts["call_kind"] == "eth_transfer_literal"
    assert "best-effort" in facts["intent"]


def test_variable_value_transfer_may_be_zero_and_is_labelled() -> None:
    facts = _one(_fn('    (bool ok, ) = target.call{value: amount}("");\n    ok;'))
    assert facts["call_kind"] == "eth_transfer_variable"
    assert "cannot tell whether it is zero at runtime" in facts["calldata"]


def test_explicit_zero_value_call_runs_receive_or_fallback() -> None:
    facts = _one(_fn('    (bool ok, ) = target.call{value: 0}("");\n    ok;'))
    assert facts["call_kind"] == "empty_no_value"
    assert "intent" not in facts


def test_empty_calldata_without_value_is_a_candidate() -> None:
    assert _one(_fn('    (bool ok, ) = target.call("");' + SW))["call_kind"] == "empty_no_value"


def test_empty_calldata_with_fixed_gas_stipend_is_a_candidate() -> None:
    facts = _one(_fn('    (bool ok, ) = target.call{gas: 2300}("");' + SW))
    assert facts["call_kind"] == "empty_no_value"
    assert facts["gas_guard"] == NO_GUARD
    assert "fixed stipend of `2300`" in facts["gas_forwarded"]


def test_value_transfer_whose_failure_reverts_is_not_a_candidate() -> None:
    body = '    (bool ok, ) = target.call{value: amount}("");\n    if (!ok) revert E();'
    assert _found(_fn(body)) == []


def test_deliberate_best_effort_transfer_stays_low_confidence_and_explained() -> None:
    body = (
        "    // refund is best-effort by design\n"
        '    (bool ok, ) = payable(msg.sender).call{value: amount, gas: 2300}("");\n    ok;'
    )
    found = _found(_fn(body))
    assert len(found) == 1
    item = found[0]
    facts = dict(item.facts)
    assert item.confidence == "low" and item.verified is False
    assert item.title.startswith("Ether transfer failure may be ignored")
    assert "possible false positive" in facts["intent"]
    assert "possible false positive" in item.summary
    assert facts["evidence_class"] == "possible_false_positive"
    assert facts["pattern"] == "prefund_style_transfer"
    assert facts["vulnerability_claim"] == "none"
    assert item.verified is False


# ---- emission is not precision ------------------------------------------------------------------

HEAD_HOOK = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n"


def _classified(text: str) -> tuple[bool, dict[str, str]]:
    found = (
        _found(text)
        if text.startswith(HEAD)
        else [
            c for c in analyze_text(text, ("caller_context",)).candidates if c.detector == DETECTOR
        ]
    )
    if not found:
        return False, {}
    item = found[0]
    assert item.verified is False and item.confidence == "low"
    assert item.status == "candidate"
    return True, dict(item.facts)


def test_expected_emission_is_not_a_vulnerability_precision_score() -> None:
    """Agreement with the detector is not a precision score and not a confirmed flaw.

    The two hook cases and the four prefund-style transfers are expected emissions
    and possible false positives. None of them is counted as a genuine vulnerability.
    """
    from tests.test_phase50.phase50_support import read

    prefund_safe = {"aa_safe.sol": read("aa_safe.sol")}
    prefund_loose = {"aa_vulnerable.sol": read("aa_vulnerable.sol")}
    literal = _fn('    (bool ok, ) = payable(msg.sender).call{value: 1 ether}("");\n    ok;')
    variable = _fn('    (bool ok, ) = payable(msg.sender).call{value: amount}("");\n    ok;')
    hook_refund = HEAD_HOOK + (
        "contract RefundHook {\n"
        "  function afterSwap(address target) external {\n"
        '    (bool ok, ) = target.call{value: 1 ether}("");\n    ok;\n  }\n}\n'
    )
    hook_notify = HEAD_HOOK + (
        "contract NotifyHook {\n"
        "  function beforeSwap(address target, uint256 amount) external {\n"
        '    (bool ok, ) = target.call{value: amount}("");\n    ok;\n  }\n}\n'
    )
    rows = [
        ("prefund_safe", prefund_safe, "prefund_style_transfer"),
        ("prefund_loose", prefund_loose, "prefund_style_transfer"),
        ("prefund_literal", literal, "prefund_style_transfer"),
        ("prefund_variable", variable, "prefund_style_transfer"),
        ("hook_refund", hook_refund, "best_effort_hook"),
        ("hook_notify", hook_notify, "best_effort_hook"),
    ]
    emitted_as_intended = 0
    genuine_flaws = 0
    for name, source, pattern in rows:
        if isinstance(source, dict):
            from app.parsing.solidity_research import build_research_model
            from app.parsing.solidity_research_suite import run_suite

            hits = [
                c
                for c in run_suite(build_research_model(source)).candidates
                if c.detector == DETECTOR and c.function.startswith("validateUserOp")
            ]
            assert len(hits) == 1, name
            facts = dict(hits[0].facts)
            assert hits[0].verified is False and hits[0].confidence == "low"
        else:
            emitted, facts = _classified(source)
            assert emitted, name
        assert facts["evidence_class"] == "possible_false_positive", name
        assert facts["pattern"] == pattern, name
        assert facts["vulnerability_claim"] == "none", name
        emitted_as_intended += 1
        # An expected candidate is not proof of a security flaw.
        genuine_flaws += 0
    propagated = _fn("    (bool ok, ) = target.call(data);\n    require(ok);")
    assert _found(propagated) == []
    assert emitted_as_intended == 6 and genuine_flaws == 0


def test_mechanism_probe_is_not_a_security_violation() -> None:
    """Outer success is a mechanism observation. Outer revert is failure propagation.

    Neither outcome is a security-property violation, and neither is counted as one.
    ``CALL_MUST_FAIL`` keeps its existing verdict for other templates.
    """
    from app.discovery.bounty.findings import is_strong
    from app.discovery.bounty.properties import (
        OracleKind,
        OracleObservation,
        PropertyDeclaration,
        PropertyVerdict,
        build_property,
        evaluate,
    )
    from app.discovery.bounty.stateful import (
        FeedbackLoopResult,
        Outcome,
        StatefulObservation,
        ToolStatus,
        classify,
    )
    from app.discovery.bounty.vfcs import generate
    from app.parsing.solidity_research import build_research_model

    positive = HEAD_HOOK + (
        "contract Relay {\n"
        "  function relay(address target, bytes calldata data) external {\n"
        "    (bool ok, ) = target.call(data);\n    ok;\n  }\n}\n"
    )
    sources = {"Relay.sol": positive}
    model = build_research_model(sources)
    candidate = next(
        c for c in analyze_text(positive, ("caller_context",)).candidates if c.detector == DETECTOR
    )
    sequence = generate(model, [candidate]).sequences[0]
    spec = build_property(sequence, model, candidate)
    assert spec.declaration is PropertyDeclaration.MECHANISM_OBSERVATION
    assert spec.oracle.kind is OracleKind.MECHANISM_PROBE
    assert spec.family.value == "mechanism"

    seen = evaluate(sequence, spec.oracle, OracleObservation(call_ok=(1, 1)))
    assert seen.verdict is PropertyVerdict.MECHANISM_OBSERVED
    assert "unproven" in seen.reason
    outcome, evaluation, reason = classify(sequence, spec.oracle, OracleObservation(call_ok=(1, 1)))
    assert outcome is Outcome.MECHANISM_OBSERVED
    assert evaluation.verdict is PropertyVerdict.MECHANISM_OBSERVED
    assert "security-property violation" in reason

    propagated = evaluate(sequence, spec.oracle, OracleObservation(call_ok=(1, 0)))
    assert propagated.verdict is PropertyVerdict.FAILURE_PROPAGATED
    failed, _, failed_reason = classify(sequence, spec.oracle, OracleObservation(call_ok=(1, 0)))
    assert failed is Outcome.FAILURE_PROPAGATED
    assert "not reproduced" in failed_reason

    tools = ToolStatus(forge="unavailable", solc="unavailable")
    observation = StatefulObservation(
        sequence_id=sequence.sequence_id,
        outcome=Outcome.MECHANISM_OBSERVED,
        reason=reason,
        pipeline="default",
        executed=True,
        return_code=0,
        reverted_index=-1,
        property_under_test=spec.statement,
        property_checked=True,
        deployment="source",
        tools=tools,
        declaration=spec.declaration.value,
        verdict=seen.verdict.value,
        oracle_kind=spec.oracle.kind.value,
    )
    assert observation.strong_candidate is False
    assert observation.signal() is None
    loop = FeedbackLoopResult(observations=(observation,), mutated=(), signals_emitted=())
    assert loop.violations == ()
    assert loop.mechanism_observations == (observation,)
    assert (
        is_strong(
            {
                "outcome": "mechanism_observed",
                "declaration": "mechanism_observation",
                "verdict": "mechanism_observed",
                "oracle_kind": "mechanism_probe",
                "identity_status": "bound",
            }
        )
        is False
    )

    security = evaluate(
        sequence,
        spec.oracle.__class__(
            kind=OracleKind.CALL_MUST_FAIL,
            probe_role="probe",
            probe_function=spec.oracle.probe_function,
            precondition_roles=("failing_callee",),
            expression="an unauthenticated call must revert",
        ),
        OracleObservation(call_ok=(1, 1)),
    )
    assert security.verdict is PropertyVerdict.PROPERTY_VIOLATED


def test_swallowed_call_candidate_reaches_the_existing_harness() -> None:
    """Positive fixture becomes a sequence and a local harness. Negative control does not.

    Forge executes the harness when it is installed. Otherwise the executor reports
    UNAVAILABLE and the test does not treat that as a completed run or a finding.
    The dedicated integration job sets BUGFORGE_REQUIRE_FOUNDRY so UNAVAILABLE fails.
    """
    import os

    from app.discovery.bounty.properties import (
        OracleKind,
        PropertyDeclaration,
        PropertyVerdict,
        build_property,
    )
    from app.discovery.bounty.stateful import Outcome, StatefulExecutor, build_harness
    from app.discovery.bounty.vfcs import generate, stamp_source_identity
    from app.parsing.solidity_research import build_research_model

    positive = HEAD_HOOK + (
        "contract Relay {\n"
        "  function relay(address target, bytes calldata data) external {\n"
        "    (bool ok, ) = target.call(data);\n    ok;\n  }\n}\n"
    )
    negative = HEAD_HOOK + (
        "contract Relay {\n"
        "  function relay(address target, bytes calldata data) external {\n"
        '    (bool ok, ) = target.call(data);\n    require(ok, "propagated");\n  }\n}\n'
    )
    sources = {"Relay.sol": positive}
    model = build_research_model(sources)
    hits = [
        c for c in analyze_text(positive, ("caller_context",)).candidates if c.detector == DETECTOR
    ]
    assert len(hits) == 1
    candidate = hits[0]
    facts = dict(candidate.facts)
    assert facts["evidence_class"] == "detector_emission"
    assert facts["vulnerability_claim"] == "none"
    assert facts["target_parameter"] == "target"
    assert facts["calldata_parameter"] == "data"
    assert facts["failure_propagation"] == "swallowed"
    assert candidate.verified is False and candidate.confidence == "low"

    built = generate(model, [candidate], sources=sources)
    assert len(built.sequences) == 1, built.skipped
    sequence = built.sequences[0]
    assert sequence.template == "failing-callee→swallowed-call"
    assert sequence.verified is False
    assert sequence.calls[0].primitive and sequence.calls[0].function == "deploy_reverting()"
    assert any(value == "reverting_callee" for _name, value in sequence.calls[1].arguments)

    spec = build_property(sequence, model, candidate)
    assert spec.declaration is PropertyDeclaration.MECHANISM_OBSERVATION
    assert spec.oracle.kind is OracleKind.MECHANISM_PROBE
    assert spec.oracle.executable
    assert spec.as_dict()["verified"] is False

    harness = build_harness(sequence, model, oracle=spec.oracle, sources=sources)
    assert harness.buildable, harness.reason
    assert "BugForgeRevertingCallee" in harness.source
    assert "relay(address,bytes)" in harness.source
    assert "ffi" not in harness.source

    safe_hits = [
        c for c in analyze_text(negative, ("caller_context",)).candidates if c.detector == DETECTOR
    ]
    assert safe_hits == []

    executor = StatefulExecutor()
    observation = executor.execute(sequence, model, sources, spec=spec)
    assert observation.verified is False
    required = os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1"
    if not executor.available():
        assert observation.outcome is Outcome.UNAVAILABLE
        assert observation.result is None or observation.result.test_status == "not_run"
        assert observation.strong_candidate is False
        if required:
            raise AssertionError("forge and solc were required, but the executor is unavailable")
        return

    assert observation.outcome is Outcome.MECHANISM_OBSERVED, observation.reason
    assert observation.verdict == PropertyVerdict.MECHANISM_OBSERVED.value
    assert observation.strong_candidate is False
    assert "unproven" in observation.reason
    result = observation.result
    assert result is not None
    assert result.compiler_status == "ok"
    assert result.test_status == "passed"
    assert result.process_status == "completed"
    assert result.property_status == "mechanism_observed"
    assert result.call_results[1] == 1  # outer call succeeded; the callee revert was swallowed
    negative_sources = {"Relay.sol": negative}
    runs_after_positive = executor.runs
    mismatched = executor.execute(
        sequence,
        build_research_model(negative_sources),
        negative_sources,
        spec=spec,
    )
    assert mismatched.outcome is Outcome.IDENTITY_MISMATCH, mismatched.reason
    assert mismatched.verified is False and mismatched.strong_candidate is False
    assert executor.runs == runs_after_positive  # changed source never reaches Forge
    # The propagating body is not a detector hit. This control rebinds the same
    # plan to that text; it is not a claim that analysis discovered it.
    assert safe_hits == []
    control = stamp_source_identity(sequence, negative_sources)
    control_model = build_research_model(negative_sources)
    control_spec = build_property(control, control_model)
    propagated = executor.execute(control, control_model, negative_sources, spec=control_spec)
    assert propagated.outcome is Outcome.FAILURE_PROPAGATED, propagated.reason
    assert propagated.verified is False
    assert propagated.strong_candidate is False
    assert propagated.result is not None
    assert propagated.result.compiler_status == "ok"
    assert propagated.result.test_status == "passed"
    assert propagated.result.property_status == "failure_propagated"
    assert propagated.result.call_results[1] == 0


def test_unknown_receiver_and_constructor_are_limitations() -> None:
    from app.discovery.bounty.stateful import build_harness
    from app.discovery.bounty.vfcs import generate
    from app.parsing.solidity_research import build_research_model

    self_call = HEAD_HOOK + (
        "contract Box {\n  function poke(bytes calldata data) external {\n"
        "    (bool ok, ) = address(this).call(data);\n    ok;\n  }\n}\n"
    )
    model = build_research_model({"Box.sol": self_call})
    hits = [
        c for c in analyze_text(self_call, ("caller_context",)).candidates if c.detector == DETECTOR
    ]
    assert len(hits) == 1
    assert dict(hits[0].facts)["target_parameter"] == "unknown"
    skipped = generate(model, hits)
    assert skipped.sequences == ()
    assert "not invented" in skipped.skipped[0][1]

    needs_oracle = HEAD_HOOK + (
        "interface IOracle { function price() external view returns (uint256); }\n"
        "contract Relay {\n  constructor(IOracle oracle) {}\n"
        "  function relay(address target, bytes calldata data) external {\n"
        "    (bool ok, ) = target.call(data);\n    ok;\n  }\n}\n"
    )
    sources = {"Relay.sol": needs_oracle}
    relay_model = build_research_model(sources)
    relay_hits = [
        c
        for c in analyze_text(needs_oracle, ("caller_context",)).candidates
        if c.detector == DETECTOR
    ]
    sequence = generate(relay_model, relay_hits).sequences[0]
    harness = build_harness(sequence, relay_model, sources=sources)
    assert harness.buildable is False
    assert harness.reason_code == "unknown_constructor_argument"
