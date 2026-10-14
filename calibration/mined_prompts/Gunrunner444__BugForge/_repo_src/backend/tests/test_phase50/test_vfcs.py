"""VFCS generation, feedback-directed mutation, and bounded minimization."""

from __future__ import annotations

from dataclasses import replace

from app.discovery.bounty import vfcs
from app.discovery.bounty.vfcs import (
    FeedbackSignal,
    SequenceIdentity,
    generate,
    minimize,
    mutate,
)
from app.discovery.sequences import MAX_MUTATIONS, MAX_SEQUENCE_LENGTH
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase50.phase50_support import read

IDENTITY = SequenceIdentity(
    campaign_id="cp_1",
    source_snapshot="abc1234",
    compiler_configuration="solc=0.8.28",
    fork_reference="1:19000000:snap",
    program_context="pc_1",
)


def _build(*names: str):
    model = build_research_model({n: read(n) for n in names})
    suite = run_suite(model)
    return model, suite, generate(model, suite.candidates, IDENTITY)


def _by_template(result, template: str):
    return [s for s in result.sequences if s.template == template]


def test_caps_never_exceed_existing_sequence_limits() -> None:
    assert vfcs.MAX_VFCS_LENGTH == MAX_SEQUENCE_LENGTH
    assert vfcs.MAX_MUTATIONS == MAX_MUTATIONS


def test_only_real_functions_are_used_and_primitives_say_why() -> None:
    model, _suite, result = _build(
        "caller_context_vulnerable.sol", "accounting_vulnerable.sol", "message_vulnerable.sol"
    )
    assert result.sequences
    for sequence in result.sequences:
        assert 1 <= len(sequence.calls) <= MAX_SEQUENCE_LENGTH
        for call in sequence.calls:
            if call.primitive:
                assert call.established_by
                continue
            real = [f for f in model.functions_of(call.contract) if f.signature == call.function]
            assert real, f"{call.identity} is not a function in the sources"


def test_short_templates_are_produced_from_candidates() -> None:
    _m, _s, caller = _build("caller_context_vulnerable.sol")
    assert _by_template(caller, "nested-dispatch")
    assert _by_template(caller, "approve→transferFrom")
    assert _by_template(caller, "forwarded-actor")
    _m, _s, accounting = _build("accounting_vulnerable.sol")
    donate = _by_template(accounting, "deposit→donate→withdraw")
    assert donate
    roles = [c.role for c in donate[0].calls]
    assert roles == ["deposit", "donate", "deposit", "withdraw"]
    _m, _s, aa = _build("aa_vulnerable.sol")
    assert _by_template(aa, "AA validation→execution")
    assert _by_template(aa, "initialize→reinitialize")
    _m, _s, messages = _build("message_vulnerable.sol")
    replay = _by_template(messages, "authorize→execute")
    assert any(s.calls[-1].role == "replay" for s in replay)


def test_no_blind_permutations_and_no_sequence_without_a_candidate() -> None:
    _m, suite, result = _build("caller_context_safe.sol", "accounting_safe.sol")
    assert suite.candidates == ()
    assert result.sequences == ()
    _m, suite, result = _build("accounting_vulnerable.sol")
    assert len(result.sequences) <= len(suite.candidates)
    for sequence in result.sequences:
        assert sequence.derived_from.split("@")[0] in {c.detector for c in suite.candidates}


def test_sequences_keep_identity_and_are_never_verified() -> None:
    _m, _s, result = _build("caller_context_vulnerable.sol")
    for sequence in result.sequences:
        assert sequence.identity == IDENTITY
        assert sequence.verified is False
        assert sequence.status == "candidate"
        assert sequence.sequence_id.startswith("vf_")
    other = generate(
        *(lambda m: (m, run_suite(m).candidates))(
            build_research_model(
                {"caller_context_vulnerable.sol": read("caller_context_vulnerable.sol")}
            )
        ),
        replace(IDENTITY, source_snapshot="other"),
    )
    assert {s.sequence_id for s in other.sequences}.isdisjoint(
        {s.sequence_id for s in result.sequences}
    )


def test_generation_is_deterministic_deduplicated_and_capped() -> None:
    model, suite, first = _build(
        "caller_context_vulnerable.sol", "message_vulnerable.sol", "aa_vulnerable.sol"
    )
    again = generate(model, list(reversed(suite.candidates)), IDENTITY)
    assert len(first.sequences) <= vfcs.MAX_VFCS_CANDIDATES
    ids = [s.sequence_id for s in first.sequences]
    assert len(ids) == len(set(ids))
    assert {s.sequence_id for s in first.sequences} == {
        s.sequence_id for s in again.sequences
    } or first.truncated
    tiny = generate(model, suite.candidates, IDENTITY, limit=2)
    assert len(tiny.sequences) == 2 and tiny.truncated
    assert (
        len(generate(model, suite.candidates, IDENTITY, limit=10_000).sequences)
        <= vfcs.MAX_VFCS_CANDIDATES
    )


def test_unresolvable_candidates_are_skipped_with_a_reason() -> None:
    _m, _s, result = _build("accounting_vulnerable.sol")
    reasons = dict(result.skipped)
    assert any("no sequence template" in reason for reason in reasons.values())


# ---- mutation -------------------------------------------------------------------------------


def _parent():
    _m, _s, result = _build("accounting_vulnerable.sol")
    return _by_template(result, "deposit→donate→withdraw")[0]


def test_mutation_is_bounded_deterministic_and_keeps_real_functions() -> None:
    parent = _parent()
    signals = [
        FeedbackSignal("near_miss", parent.sequence_id, call_index=1),
        FeedbackSignal("coverage_gain", parent.sequence_id, call_index=0),
        FeedbackSignal("caller_context_mismatch", parent.sequence_id, call_index=0),
    ]
    first = mutate([parent], signals)
    second = mutate([parent], list(reversed(signals)))
    assert first and len(first) <= MAX_MUTATIONS
    assert [s.sequence_id for s in first] == [s.sequence_id for s in second]
    functions = {c.identity for c in parent.calls}
    for child in first:
        assert child.origin.startswith("mutation:")
        assert {c.identity for c in child.calls} <= functions
        assert len(child.calls) <= MAX_SEQUENCE_LENGTH
        assert child.sequence_id != parent.sequence_id
        assert child.verified is False


def test_symbolic_counterexample_seeds_arguments() -> None:
    parent = _parent()
    signal = FeedbackSignal(
        "symbolic_counterexample", parent.sequence_id, call_index=0, values=(("amount", "7"),)
    )
    (child,) = mutate([parent], [signal])
    assert dict(child.calls[0].arguments)["amount"] == "7"


def test_unknown_parent_or_exhausted_limit_yields_nothing() -> None:
    parent = _parent()
    assert mutate([parent], [FeedbackSignal("near_miss", "vf_missing")]) == ()
    assert mutate([parent], [FeedbackSignal("near_miss", parent.sequence_id)], limit=0) == ()
    many = [FeedbackSignal("near_miss", parent.sequence_id, call_index=i) for i in range(4)]
    assert len(mutate([parent], many * 6, limit=100)) <= MAX_MUTATIONS


# ---- minimization ---------------------------------------------------------------------------


def _needs(*identities: str):
    def evaluate(calls):
        present = {c.identity for c in calls}
        return all(item in present for item in identities)

    return evaluate


def test_minimization_reduces_to_the_calls_that_matter() -> None:
    parent = _parent()
    keep = (parent.calls[1].identity, parent.calls[3].identity)
    result = minimize(parent, _needs(*keep), evaluator_name="scripted")
    assert result.completed and result.reason == "one_minimal"
    assert {c.identity for c in result.minimized} == set(keep)
    assert len(result.minimized) < len(result.original) == 4
    assert result.original == parent.calls
    assert result.verified is False
    assert result.preserved_property == parent.property_under_test
    assert result.attempts <= vfcs.MAX_MINIMIZATION_ATTEMPTS


def test_minimization_preserves_order_and_never_invents_calls() -> None:
    parent = _parent()
    result = minimize(parent, _needs(parent.calls[0].identity), evaluator_name="scripted")
    originals = list(parent.calls)
    positions = [originals.index(c) for c in result.minimized]
    assert positions == sorted(positions)


def test_unreproduced_original_is_reported_not_minimized() -> None:
    parent = _parent()
    result = minimize(parent, lambda calls: False, evaluator_name="scripted")
    assert result.completed and result.reason == "original_not_reproduced"
    assert result.minimized == parent.calls and result.attempts == 1


def test_inconclusive_evaluator_is_counted_and_never_treated_as_reproduction() -> None:
    parent = _parent()
    result = minimize(parent, lambda calls: None, evaluator_name="scripted")
    assert result.reason == "original_not_reproduced"
    assert result.inconclusive_attempts == result.attempts == 1


def test_minimization_stops_at_the_attempt_budget() -> None:
    parent = _parent()
    result = minimize(
        parent, _needs(parent.calls[3].identity), evaluator_name="scripted", max_attempts=2
    )
    assert result.attempts <= 2
    assert not result.completed and result.reason == "attempt_budget_exhausted"
    zero = minimize(parent, _needs(), evaluator_name="scripted", max_attempts=0)
    assert zero.attempts == 0 and not zero.completed
    capped = minimize(parent, _needs(), evaluator_name="scripted", max_attempts=10_000)
    assert capped.attempts <= vfcs.MAX_MINIMIZATION_ATTEMPTS


def test_an_evaluator_that_raises_is_inconclusive() -> None:
    parent = _parent()

    def broken(calls):
        raise RuntimeError("boom")

    result = minimize(parent, broken, evaluator_name="broken")
    assert result.reason == "original_not_reproduced" and result.inconclusive_attempts == 1
