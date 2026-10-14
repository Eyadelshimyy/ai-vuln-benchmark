from __future__ import annotations

import pytest

from app.discovery.orchestration.codec import (
    CodecError,
    canonical_json,
    digest,
    from_jsonable,
    strip_audit,
)
from app.discovery.orchestration.model import (
    DIMENSIONS,
    RESUMABLE_STOPS,
    STOP_STATE,
    TERMINAL,
    TRANSITIONS,
    BudgetLedger,
    CampaignIdentity,
    EvidenceQuality,
    IllegalTransitionError,
    InvalidDecisionError,
    OrchestratorState,
    ResearchState,
    StopReason,
    transition,
)

S = OrchestratorState


def test_canonical_json_sorts_keys_and_sets() -> None:
    first = {"b": 1, "a": {"y": [3, 2], "x": frozenset({"z", "a"})}}
    second = {"a": {"x": frozenset({"a", "z"}), "y": [3, 2]}, "b": 1}
    assert canonical_json(first) == canonical_json(second)
    assert canonical_json(first) == '{"a":{"x":["a","z"],"y":[3,2]},"b":1}'


def test_audit_fields_never_change_a_digest() -> None:
    assert digest({"a": 1, "created_at": "now"}) == digest({"a": 1, "created_at": "later"})
    assert strip_audit({"timestamp": 1, "k": [{"updated_at": 2, "v": 3}]}) == {"k": [{"v": 3}]}


def test_codec_rejects_unserializable_and_unknown_shapes() -> None:
    with pytest.raises(CodecError):
        canonical_json({"a": object()})
    with pytest.raises(CodecError):
        from_jsonable(CampaignIdentity, {"campaign_id": "x", "unknown": 1})
    with pytest.raises(CodecError):
        from_jsonable(CampaignIdentity, {"campaign_id": 5})
    with pytest.raises(CodecError):
        from_jsonable(EvidenceQuality, "verified")


def test_there_is_no_verified_evidence_quality() -> None:
    assert "verified" not in {item.value for item in EvidenceQuality}
    assert not any("verif" in item.value for item in EvidenceQuality)


def test_state_machine_only_allows_declared_moves() -> None:
    for current, targets in TRANSITIONS.items():
        for target in OrchestratorState:
            if target in targets and current is not S.STOPPED:
                assert transition(current, target) is target
            elif target not in targets:
                with pytest.raises(IllegalTransitionError):
                    transition(current, target)
    for state in TERMINAL:
        assert TRANSITIONS[state] == frozenset()


def test_stopped_only_resumes_when_asked() -> None:
    with pytest.raises(IllegalTransitionError):
        transition(S.STOPPED, S.PLANNING)
    assert transition(S.STOPPED, S.PLANNING, explicit_resume=True) is S.PLANNING


def test_every_stop_reason_maps_to_a_state_and_only_known_ones_resume() -> None:
    assert set(STOP_STATE) == set(StopReason)
    assert RESUMABLE_STOPS <= set(StopReason)
    assert StopReason.BUDGET_EXHAUSTED not in RESUMABLE_STOPS
    assert StopReason.ROUNDS_EXHAUSTED not in RESUMABLE_STOPS
    assert StopReason.SAFETY_BLOCKED not in RESUMABLE_STOPS
    assert StopReason.SCOPE_BLOCKED not in RESUMABLE_STOPS


def test_budget_reserve_settle_and_overrun() -> None:
    ledger = BudgetLedger(limits={"attempts": 3, "rounds": 2})
    ledger.reserve({"attempts": 1, "rounds": 1})
    assert ledger.remaining("attempts") == 2
    ledger.settle({"attempts": 1, "rounds": 1}, {"attempts": 1, "rounds": 1})
    assert ledger.consumed == {"attempts": 1, "rounds": 1}
    assert ledger.reserved == {"attempts": 0, "rounds": 0}
    assert ledger.overrun is False
    ledger.reserve({"attempts": 1})
    ledger.settle({"attempts": 1}, {"attempts": 5})
    assert ledger.overrun is True


def test_budget_refuses_a_reservation_that_does_not_fit_and_names_the_dimension() -> None:
    ledger = BudgetLedger(limits={"attempts": 1, "rounds": 0})
    with pytest.raises(InvalidDecisionError, match="budget:rounds"):
        ledger.reserve({"attempts": 1, "rounds": 1})
    assert ledger.reserved == {}
    assert ledger.shortfall({"unknown_dimension": 1}) == "unknown_dimension"
    assert ledger.remaining("missing") == 0


def test_budget_restore_never_raises_a_limit_or_lowers_consumption() -> None:
    current = BudgetLedger(limits={"rounds": 4, "attempts": 4}, consumed={"rounds": 1})
    saved = BudgetLedger(
        limits={"rounds": 10, "attempts": 2, "extra": 9},
        consumed={"rounds": 3},
        reserved={"rounds": 5},
        overrun=True,
    )
    current.merge_restored(saved)
    assert current.limits["rounds"] == 4
    assert current.limits["attempts"] == 2
    assert current.consumed["rounds"] == 3
    assert current.reserved == {}
    assert current.overrun is True


def test_budget_exhaustion_names_the_first_spent_cap() -> None:
    ledger = BudgetLedger(limits={dimension: 1 for dimension in DIMENSIONS})
    assert ledger.exhausted() == ""
    ledger.consumed["rounds"] = 1
    assert ledger.exhausted() == "rounds"


def test_identity_key_needs_contract_and_function() -> None:
    assert CampaignIdentity("c", function="withdraw").identity_key() == ""
    assert CampaignIdentity("c", contract="Vault").identity_key() == ""
    assert CampaignIdentity("c", contract="Vault", function="withdraw").identity_key() == (
        "Vault.withdraw"
    )
    a = CampaignIdentity("c1", contract="Vault", function="withdraw", source_snapshot="s1")
    b = CampaignIdentity("c2", contract="Vault", function="withdraw", source_snapshot="s1")
    c = CampaignIdentity("c1", contract="Vault", function="withdraw", source_snapshot="s2")
    assert a.target_digest() == b.target_digest() != c.target_digest()


def test_state_hash_changes_with_content_not_with_timestamps() -> None:
    state = ResearchState(identity=CampaignIdentity("c"))
    first = state.hash_now()
    state.round = 1
    assert state.hash_now() != first
    state.round = 0
    assert state.hash_now() == first
