"""Fail-closed contracts for the governed post-pilot item-bank lifecycle."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import (
    CandidateLifecycleState,
    PilotCandidateRecord,
    audit_generated_item_candidate,
    build_pilot_candidate_record,
)
from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleError,
    ItemBankLifecycleRecord,
    ItemBankLifecycleState,
    PolicyCriticality,
    build_item_bank_pilot_record,
    transition_item_bank_record,
)

_AUDIT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)


def _pilot_record() -> PilotCandidateRecord:
    """Return one public replay-verified pilot candidate fixture."""
    candidate = _AUDIT_FIXTURES["_candidate"]()
    report = audit_generated_item_candidate(candidate)
    return build_pilot_candidate_record(
        candidate,
        report,
        **_AUDIT_FIXTURES["_pilot_kwargs"](),
    )


def _evidence(
    kind: ItemBankEvidenceKind,
    suffix: str,
    *,
    fingerprint_character: str = "a",
) -> ItemBankEvidenceReference:
    """Return one source-text-free exact evidence reference."""
    return ItemBankEvidenceReference(
        evidence_kind=kind,
        evidence_id=f"{suffix}_evidence",
        evidence_fingerprint=fingerprint_character * 64,
    )


def _calibration_evidence() -> tuple[ItemBankEvidenceReference, ...]:
    """Return the minimum evidence set required for calibration admission."""
    return (
        _evidence(ItemBankEvidenceKind.CALIBRATION, "calibration", fingerprint_character="a"),
        _evidence(ItemBankEvidenceKind.ITEM_FIT, "item_fit", fingerprint_character="b"),
        _evidence(ItemBankEvidenceKind.DIF, "item_dif", fingerprint_character="c"),
        _evidence(
            ItemBankEvidenceKind.ITEM_INFORMATION,
            "item_information",
            fingerprint_character="d",
        ),
    )


def _calibrated_record():
    """Return one exact pilot-to-calibrated transition fixture."""
    pilot = build_item_bank_pilot_record(
        _pilot_record(),
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.CONJUNCTIVE_GATE,
    )
    return transition_item_bank_record(
        pilot,
        ItemBankLifecycleState.CALIBRATED,
        evidence_references=_calibration_evidence(),
        transition_reason_id="calibration_completed",
    )


def test_verified_pilot_creates_one_deterministic_initial_bank_record() -> None:
    """Only an exact verified pilot record can establish post-pilot provenance."""
    pilot = _pilot_record()
    first = build_item_bank_pilot_record(
        pilot,
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.CONJUNCTIVE_GATE,
    )
    second = build_item_bank_pilot_record(
        pilot,
        item_version="1.0.0",
        policy_criticality="conjunctive_gate",
    )

    assert first == second
    assert first.lifecycle_state is ItemBankLifecycleState.PILOTING
    assert first.policy_criticality is PolicyCriticality.CONJUNCTIVE_GATE
    assert first.item_id == pilot.item_id
    assert first.candidate_fingerprint == pilot.candidate_fingerprint
    assert first.pilot_record_fingerprint == pilot.pilot_record_fingerprint
    assert first.audit_report_fingerprint == pilot.audit_report_fingerprint
    assert first.blueprint_id == pilot.blueprint_id
    assert first.rubric_id == pilot.rubric_id
    assert first.rubric_version == pilot.rubric_version
    assert first.previous_record_fingerprint is None
    assert first.evidence_references == ()
    assert first.approved_use_ids == ()
    assert first.transition_reason_id == "pilot_admission"
    assert first.record_id == f"item_bank_record_{first.record_fingerprint[:32]}"
    assert len(first.record_fingerprint) == 64
    assert first.to_dict()["pilot_record_fingerprint"] == pilot.pilot_record_fingerprint


def test_direct_record_construction_is_not_a_supported_lifecycle_transition() -> None:
    """Callers cannot construct an authoritative lifecycle record directly."""
    pilot = _pilot_record()
    with pytest.raises(ValueError, match="build_item_bank_pilot_record"):
        ItemBankLifecycleRecord(
            item_id=pilot.item_id,
            item_version="1.0.0",
            candidate_fingerprint=pilot.candidate_fingerprint,
            pilot_record_fingerprint=pilot.pilot_record_fingerprint,
            audit_report_fingerprint=pilot.audit_report_fingerprint,
            blueprint_id=pilot.blueprint_id,
            rubric_id=pilot.rubric_id,
            rubric_version=pilot.rubric_version,
            lifecycle_state=ItemBankLifecycleState.PILOTING,
            policy_criticality=PolicyCriticality.ORDINARY,
            approved_use_ids=(),
            evidence_references=(),
            previous_record_fingerprint=None,
            transition_reason_id="pilot_admission",
        )


def test_calibration_requires_all_declared_evidence_kinds() -> None:
    """A pilot cannot be called calibrated from one fit result or raw score alone."""
    pilot = build_item_bank_pilot_record(
        _pilot_record(),
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.REQUIRED,
    )
    incomplete = _calibration_evidence()[:-1]

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            pilot,
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=incomplete,
            transition_reason_id="calibration_completed",
        )

    assert caught.value.code == "missing_transition_evidence"
    assert caught.value.path == "$.evidence_references"
    assert "item_information" in caught.value.message
    assert incomplete[0].evidence_fingerprint not in str(caught.value)


def test_evidence_order_does_not_change_calibrated_identity() -> None:
    """Evidence order is not an item-bank identity or approval decision."""
    pilot = build_item_bank_pilot_record(
        _pilot_record(),
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.ORDINARY,
    )
    evidence = _calibration_evidence()
    first = transition_item_bank_record(
        pilot,
        "calibrated",
        evidence_references=evidence,
        transition_reason_id="calibration_completed",
    )
    second = transition_item_bank_record(
        pilot,
        ItemBankLifecycleState.CALIBRATED,
        evidence_references=tuple(reversed(evidence)),
        transition_reason_id="calibration_completed",
    )

    assert first == second
    assert first.record_fingerprint == second.record_fingerprint
    assert tuple(ref.evidence_kind for ref in first.evidence_references) == tuple(
        sorted(
            (ref.evidence_kind for ref in evidence),
            key=lambda kind: kind.value,
        )
    )
    assert first.previous_record_fingerprint == pilot.record_fingerprint


def test_complete_lifecycle_preserves_evidence_and_policy_criticality() -> None:
    """Transitions append evidence and never average away a critical policy gate."""
    calibrated = _calibrated_record()
    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "approval", fingerprint_character="e"),
        ),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring", "calibration_anchor"),
    )
    active = transition_item_bank_record(
        approved,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(),
        transition_reason_id="release_activation",
    )
    suspended = transition_item_bank_record(
        active,
        ItemBankLifecycleState.SUSPENDED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.SUSPENSION, "suspension", fingerprint_character="f"),
            _evidence(ItemBankEvidenceKind.DRIFT, "drift", fingerprint_character="1"),
        ),
        transition_reason_id="drift_quarantine",
    )
    reactivated = transition_item_bank_record(
        suspended,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "reactivation", fingerprint_character="2"),
            _evidence(ItemBankEvidenceKind.DRIFT, "drift_recheck", fingerprint_character="3"),
        ),
        transition_reason_id="drift_cleared",
    )
    retired = transition_item_bank_record(
        reactivated,
        ItemBankLifecycleState.RETIRED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.RETIREMENT, "retirement", fingerprint_character="4"),
        ),
        transition_reason_id="content_obsolete",
    )

    records = (calibrated, approved, active, suspended, reactivated, retired)
    assert [record.lifecycle_state.value for record in records] == [
        "calibrated",
        "approved",
        "active",
        "suspended",
        "active",
        "retired",
    ]
    assert all(
        record.policy_criticality is PolicyCriticality.CONJUNCTIVE_GATE
        for record in records
    )
    assert all(
        record.approved_use_ids == ("calibration_anchor", "production_scoring")
        for record in records[1:]
    )
    assert len(retired.evidence_references) == 10
    assert retired.previous_record_fingerprint == reactivated.record_fingerprint


def test_approved_state_requires_a_specific_use_scope() -> None:
    """Governance approval without a declared use cannot activate an item."""
    calibrated = _calibrated_record()

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            calibrated,
            ItemBankLifecycleState.APPROVED,
            evidence_references=(
                _evidence(
                    ItemBankEvidenceKind.APPROVAL,
                    "approval",
                    fingerprint_character="e",
                ),
            ),
            transition_reason_id="governance_approval",
        )

    assert caught.value.code == "missing_approved_use"
    assert caught.value.path == "$.approved_use_ids"


def test_skipped_backward_noop_and_retirement_exit_transitions_fail_closed() -> None:
    """The lifecycle cannot skip evidence gates, reverse history, or leave retirement."""
    pilot = build_item_bank_pilot_record(
        _pilot_record(),
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.ORDINARY,
    )
    for target in (
        ItemBankLifecycleState.PILOTING,
        ItemBankLifecycleState.APPROVED,
        ItemBankLifecycleState.ACTIVE,
        ItemBankLifecycleState.SUSPENDED,
        ItemBankLifecycleState.RETIRED,
    ):
        with pytest.raises(ItemBankLifecycleError) as caught:
            transition_item_bank_record(
                pilot,
                target,
                evidence_references=(),
                transition_reason_id="invalid_transition",
                approved_use_ids=("production_scoring",),
            )
        assert caught.value.code == "invalid_lifecycle_transition"

    calibrated = _calibrated_record()
    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            calibrated,
            ItemBankLifecycleState.PILOTING,
            evidence_references=(),
            transition_reason_id="reverse_transition",
        )
    assert caught.value.code == "invalid_lifecycle_transition"

    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "approval", fingerprint_character="e"),
        ),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )
    active = transition_item_bank_record(
        approved,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(),
        transition_reason_id="release_activation",
    )
    retired = transition_item_bank_record(
        active,
        ItemBankLifecycleState.RETIRED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.RETIREMENT, "retirement", fingerprint_character="4"),
        ),
        transition_reason_id="content_obsolete",
    )
    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            retired,
            ItemBankLifecycleState.ACTIVE,
            evidence_references=(),
            transition_reason_id="retirement_exit",
        )
    assert caught.value.code == "invalid_lifecycle_transition"


def test_suspension_and_reactivation_have_distinct_evidence_gates() -> None:
    """A policy action alone cannot disguise absent drift or DIF evidence."""
    calibrated = _calibrated_record()
    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "approval", fingerprint_character="e"),
        ),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )
    active = transition_item_bank_record(
        approved,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(),
        transition_reason_id="release_activation",
    )

    with pytest.raises(ItemBankLifecycleError) as suspension_error:
        transition_item_bank_record(
            active,
            ItemBankLifecycleState.SUSPENDED,
            evidence_references=(
                _evidence(
                    ItemBankEvidenceKind.SUSPENSION,
                    "suspension",
                    fingerprint_character="f",
                ),
            ),
            transition_reason_id="unexplained_quarantine",
        )
    assert suspension_error.value.code == "missing_transition_evidence"

    suspended = transition_item_bank_record(
        active,
        ItemBankLifecycleState.SUSPENDED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.SUSPENSION, "suspension", fingerprint_character="f"),
            _evidence(ItemBankEvidenceKind.DIF, "new_dif", fingerprint_character="1"),
        ),
        transition_reason_id="dif_quarantine",
    )
    with pytest.raises(ItemBankLifecycleError) as reactivation_error:
        transition_item_bank_record(
            suspended,
            ItemBankLifecycleState.ACTIVE,
            evidence_references=(
                _evidence(
                    ItemBankEvidenceKind.APPROVAL,
                    "reactivation",
                    fingerprint_character="2",
                ),
            ),
            transition_reason_id="insufficient_reactivation",
        )
    assert reactivation_error.value.code == "missing_transition_evidence"


def test_conflicting_evidence_identity_and_policy_mutation_are_rejected() -> None:
    """One evidence ID has one digest and criticality cannot change by transition."""
    pilot = build_item_bank_pilot_record(
        _pilot_record(),
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.REQUIRED,
    )
    conflicting = (
        _evidence(ItemBankEvidenceKind.CALIBRATION, "shared", fingerprint_character="a"),
        _evidence(ItemBankEvidenceKind.ITEM_FIT, "shared", fingerprint_character="b"),
        _evidence(ItemBankEvidenceKind.DIF, "item_dif", fingerprint_character="c"),
        _evidence(
            ItemBankEvidenceKind.ITEM_INFORMATION,
            "item_information",
            fingerprint_character="d",
        ),
    )
    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            pilot,
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=conflicting,
            transition_reason_id="calibration_completed",
        )
    assert caught.value.code == "conflicting_evidence_identity"
    assert caught.value.path == "$.evidence_references"

    with pytest.raises(TypeError):
        transition_item_bank_record(
            pilot,
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=_calibration_evidence(),
            transition_reason_id="calibration_completed",
            policy_criticality=PolicyCriticality.ORDINARY,
        )


def test_post_construction_mutation_is_detected_before_transition() -> None:
    """A frozen record changed through object internals cannot acquire new authority."""
    calibrated = _calibrated_record()
    original_item_id = calibrated.item_id
    object.__setattr__(calibrated, "item_id", "mutated_item")

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            calibrated,
            ItemBankLifecycleState.APPROVED,
            evidence_references=(
                _evidence(
                    ItemBankEvidenceKind.APPROVAL,
                    "approval",
                    fingerprint_character="e",
                ),
            ),
            transition_reason_id="governance_approval",
            approved_use_ids=("production_scoring",),
        )

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"
    assert original_item_id not in str(caught.value)
    assert "mutated_item" not in str(caught.value)


def test_public_boundaries_reject_subclasses_and_raw_text_fields() -> None:
    """Lifecycle values remain exact package types and source-text-free metadata."""

    class EvidenceSubclass(ItemBankEvidenceReference):
        """Unsupported extension of a package-owned evidence record."""

    valid = _evidence(ItemBankEvidenceKind.CALIBRATION, "calibration")
    subclass = object.__new__(EvidenceSubclass)
    for name, value in vars(valid).items():
        object.__setattr__(subclass, name, value)

    pilot = build_item_bank_pilot_record(
        _pilot_record(),
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.ORDINARY,
    )
    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            pilot,
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=(subclass, *_calibration_evidence()[1:]),
            transition_reason_id="calibration_completed",
        )
    assert caught.value.code == "invalid_evidence_reference"

    serialized = pilot.to_dict()
    forbidden = ("source_text", "response_text", "prompt_text", "provider_output")
    assert all(name not in serialized for name in forbidden)
    assert serialized["record_fingerprint"] == pilot.record_fingerprint
