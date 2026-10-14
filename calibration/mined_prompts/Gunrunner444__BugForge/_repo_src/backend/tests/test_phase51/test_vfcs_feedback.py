"""Phase 51 VFCS stable call-instance identity + bounded fuzzer feedback (owner Phase 5)."""

from __future__ import annotations

from app.discovery.bounty.vfcs import (
    ALLOWED_FEEDBACK_ENGINES,
    FeedbackSignal,
    SequenceIdentity,
    call_instance_id,
    generate,
    incorporate_feedback,
    index_for_instance,
    instance_identities,
)
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase50.phase50_support import read


def _sequences():
    model = build_research_model(
        {"caller_context_vulnerable.sol": read("caller_context_vulnerable.sol")}
    )
    result = generate(model, run_suite(model).candidates, SequenceIdentity(campaign_id="cp_test"))
    return result.sequences


def test_instance_identities_are_stable_and_unique_per_position() -> None:
    seq = _sequences()[0]
    first = instance_identities(seq)
    second = instance_identities(seq)
    assert first == second  # stable across regeneration of the same sequence
    assert len(first) == len(seq.calls)
    assert all(i.startswith("ci_") for i in first)


def test_repeated_calls_get_distinct_instance_ids() -> None:
    # two identical call positions still receive different stable ids (position-bound)
    seq = _sequences()[0]
    ids = instance_identities(seq)
    assert len(set(ids)) == len(ids)


def test_index_for_instance_round_trips() -> None:
    seq = _sequences()[0]
    for index in range(len(seq.calls)):
        cid = call_instance_id(seq, index)
        assert index_for_instance(seq, cid) == index
    assert index_for_instance(seq, "ci_does_not_exist") == -1


def test_feedback_only_from_allowed_fuzzers() -> None:
    sequences = _sequences()
    seq = sequences[0]
    signals = [
        FeedbackSignal(kind="coverage_gain", sequence_id=seq.sequence_id, engine="my_new_fuzzer"),
    ]
    outcome = incorporate_feedback(
        sequences, signals, available_engines=frozenset(ALLOWED_FEEDBACK_ENGINES)
    )
    assert outcome.children == ()
    assert outcome.refused
    assert "only from foundry/echidna/medusa/ityfuzz" in outcome.refused[0][1]


def test_unavailable_fuzzer_feedback_is_not_fabricated() -> None:
    sequences = _sequences()
    seq = sequences[0]
    signals = [FeedbackSignal(kind="coverage_gain", sequence_id=seq.sequence_id, engine="echidna")]
    # echidna is allowed but not available here
    outcome = incorporate_feedback(sequences, signals, available_engines=frozenset())
    assert outcome.children == ()
    assert "echidna" in outcome.unavailable_engines


def test_available_fuzzer_feedback_produces_bounded_children() -> None:
    sequences = _sequences()
    seq = sequences[0]
    signals = [
        FeedbackSignal(
            kind="coverage_gain",
            sequence_id=seq.sequence_id,
            engine="foundry",
            call_instance=call_instance_id(seq, 0),
        )
    ]
    outcome = incorporate_feedback(sequences, signals, available_engines=frozenset({"foundry"}))
    # children are bounded and never marked verified
    assert outcome.verified is False
    assert len(outcome.children) <= 16
    for child in outcome.children:
        assert child.verified is False
        assert len(child.calls) <= 4


def test_call_instance_that_does_not_belong_is_refused() -> None:
    sequences = _sequences()
    seq = sequences[0]
    signals = [
        FeedbackSignal(
            kind="coverage_gain",
            sequence_id=seq.sequence_id,
            engine="foundry",
            call_instance="ci_wrong",
        )
    ]
    outcome = incorporate_feedback(sequences, signals, available_engines=frozenset({"foundry"}))
    assert outcome.children == ()
    assert any("call-instance id does not belong" in r for _ref, r in outcome.refused)
