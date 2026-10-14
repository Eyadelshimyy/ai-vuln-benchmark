"""Regression coverage for non-psychometric item-bank suspension concerns."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.rubric.item_bank as item_bank_module
from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleError,
    ItemBankLifecycleState,
    transition_item_bank_record,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


def _evidence(kind: ItemBankEvidenceKind | str, suffix: str, character: str):
    """Return one bounded exact evidence identity."""
    return ItemBankEvidenceReference(
        evidence_kind=kind,
        evidence_id=f"{suffix}_evidence",
        evidence_fingerprint=character * 64,
    )


def _active_record():
    """Return one governed active item from the canonical lifecycle fixture."""
    calibrated = _FIXTURES["_calibrated_record"]()
    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "approval", "e"),
        ),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )
    return transition_item_bank_record(
        approved,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(),
        transition_reason_id="release_activation",
    )


def _create_internal_successor(
    current,
    *,
    target_state: ItemBankLifecycleState,
    suspension_concern_kinds: tuple[ItemBankEvidenceKind, ...],
):
    """Exercise factory-only concern invariants without bypassing public transitions."""
    return item_bank_module._create_record(
        item_id=current.item_id,
        item_version=current.item_version,
        candidate_fingerprint=current.candidate_fingerprint,
        pilot_record_fingerprint=current.pilot_record_fingerprint,
        audit_report_fingerprint=current.audit_report_fingerprint,
        blueprint_id=current.blueprint_id,
        rubric_id=current.rubric_id,
        rubric_version=current.rubric_version,
        lifecycle_state=target_state,
        policy_criticality=current.policy_criticality,
        approved_use_ids=current.approved_use_ids,
        evidence_references=current.evidence_references,
        previous_record_fingerprint=current.record_fingerprint,
        transition_reason_id="internal_invariant_probe",
        suspension_concern_kinds=suspension_concern_kinds,
    )


def test_security_privacy_concern_can_suspend_and_reactivate_without_fake_drift() -> None:
    """Security/privacy quarantine uses exact concern evidence, not invented DIF/drift."""
    active = _active_record()
    suspended = transition_item_bank_record(
        active,
        ItemBankLifecycleState.SUSPENDED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.SUSPENSION, "security_quarantine", "f"),
            _evidence("security_privacy", "security_finding", "1"),
        ),
        transition_reason_id="security_privacy_quarantine",
    )

    assert suspended.lifecycle_state is ItemBankLifecycleState.SUSPENDED
    assert suspended.suspension_concern_kinds == (
        ItemBankEvidenceKind.SECURITY_PRIVACY,
    )
    assert {reference.evidence_kind.value for reference in suspended.evidence_references} >= {
        "suspension",
        "security_privacy",
    }

    reactivated = transition_item_bank_record(
        suspended,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.APPROVAL, "security_reactivation", "2"),
            _evidence("security_privacy", "security_recheck", "3"),
        ),
        transition_reason_id="security_privacy_cleared",
    )

    assert reactivated.lifecycle_state is ItemBankLifecycleState.ACTIVE
    assert reactivated.suspension_concern_kinds == ()
    assert reactivated.previous_record_fingerprint == suspended.record_fingerprint


def test_reactivation_must_address_the_suspended_concern_class() -> None:
    """An unrelated concern artifact cannot clear a security/privacy quarantine."""
    active = _active_record()
    suspended = transition_item_bank_record(
        active,
        ItemBankLifecycleState.SUSPENDED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.SUSPENSION, "security_quarantine", "4"),
            _evidence(ItemBankEvidenceKind.SECURITY_PRIVACY, "security_finding", "5"),
        ),
        transition_reason_id="security_privacy_quarantine",
    )

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            suspended,
            ItemBankLifecycleState.ACTIVE,
            evidence_references=(
                _evidence(ItemBankEvidenceKind.APPROVAL, "security_reactivation", "6"),
                _evidence(ItemBankEvidenceKind.DIF, "unrelated_dif_recheck", "7"),
            ),
            transition_reason_id="security_privacy_cleared",
        )

    assert caught.value.code == "missing_transition_evidence"
    assert "security_privacy" in caught.value.message


@pytest.mark.parametrize(
    ("reused_kind", "reused_character", "replacement_suffix"),
    (
        (ItemBankEvidenceKind.APPROVAL, "e", "replacement_approval"),
        (
            ItemBankEvidenceKind.SECURITY_PRIVACY,
            "5",
            "replacement_security_recheck",
        ),
    ),
)
def test_reactivation_rejects_historical_fingerprint_under_new_identity(
    reused_kind: ItemBankEvidenceKind,
    reused_character: str,
    replacement_suffix: str,
) -> None:
    """A new evidence identifier cannot disguise a historical reactivation artifact."""
    suspended = transition_item_bank_record(
        _active_record(),
        ItemBankLifecycleState.SUSPENDED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.SUSPENSION, "security_quarantine", "4"),
            _evidence(ItemBankEvidenceKind.SECURITY_PRIVACY, "security_finding", "5"),
        ),
        transition_reason_id="security_privacy_quarantine",
    )
    additions = {
        ItemBankEvidenceKind.APPROVAL: _evidence(
            ItemBankEvidenceKind.APPROVAL,
            "fresh_approval",
            "6",
        ),
        ItemBankEvidenceKind.SECURITY_PRIVACY: _evidence(
            ItemBankEvidenceKind.SECURITY_PRIVACY,
            "fresh_security_recheck",
            "7",
        ),
    }
    additions[reused_kind] = _evidence(
        reused_kind,
        replacement_suffix,
        reused_character,
    )

    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            suspended,
            ItemBankLifecycleState.ACTIVE,
            evidence_references=tuple(additions.values()),
            transition_reason_id="security_privacy_cleared",
        )

    assert caught.value.code == "reused_transition_evidence"
    assert "fresh evidence fingerprints" in caught.value.message


def test_suspension_concern_metadata_fails_closed_when_factory_invariants_break() -> None:
    """Factory-sealed records cannot omit, misclassify, or leak suspension concerns."""
    active = _active_record()

    with pytest.raises(ValueError, match="suspended records require suspension concern kinds"):
        _create_internal_successor(
            active,
            target_state=ItemBankLifecycleState.SUSPENDED,
            suspension_concern_kinds=(),
        )

    with pytest.raises(
        ValueError,
        match="suspension_concern_kinds must contain only governed concern evidence kinds",
    ):
        _create_internal_successor(
            active,
            target_state=ItemBankLifecycleState.SUSPENDED,
            suspension_concern_kinds=(ItemBankEvidenceKind.APPROVAL,),
        )

    with pytest.raises(
        ValueError,
        match="only suspended records may retain suspension concern kinds",
    ):
        _create_internal_successor(
            active,
            target_state=ItemBankLifecycleState.ACTIVE,
            suspension_concern_kinds=(ItemBankEvidenceKind.SECURITY_PRIVACY,),
        )

    missing = item_bank_module._missing_required_kinds(
        ItemBankLifecycleState.SUSPENDED,
        ItemBankLifecycleState.ACTIVE,
        {ItemBankEvidenceKind.APPROVAL},
        suspension_concern_kinds=frozenset(),
    )
    assert missing == ("suspension_resolution_evidence",)
