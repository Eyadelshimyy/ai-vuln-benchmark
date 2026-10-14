"""Contracts for scientifically explicit DIF applicability evidence."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleError,
    ItemBankLifecycleState,
    build_item_bank_pilot_record,
    transition_item_bank_record,
)

_LIFECYCLE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


def _evidence(
    kind: ItemBankEvidenceKind,
    suffix: str,
    fingerprint_character: str,
) -> ItemBankEvidenceReference:
    """Return one bounded source-text-free lifecycle evidence identity."""
    return ItemBankEvidenceReference(
        evidence_kind=kind,
        evidence_id=f"{suffix}_evidence",
        evidence_fingerprint=fingerprint_character * 64,
    )


def _pilot_record():
    """Return one verified pilot lifecycle record from canonical test fixtures."""
    pilot = _LIFECYCLE_FIXTURES["_pilot_record"]()
    return build_item_bank_pilot_record(pilot, item_version="1.0.0")


def _base_calibration_evidence() -> tuple[ItemBankEvidenceReference, ...]:
    """Return calibration evidence common to both DIF applicability branches."""
    return (
        _evidence(ItemBankEvidenceKind.CALIBRATION, "calibration", "a"),
        _evidence(ItemBankEvidenceKind.ITEM_FIT, "item_fit", "b"),
        _evidence(ItemBankEvidenceKind.ITEM_INFORMATION, "item_information", "c"),
    )


def test_item_bank_evidence_domain_can_represent_dif_not_applicable() -> None:
    """Calibration must not require fabricated DIF when no comparison design exists."""
    evidence_kinds = {kind.value for kind in ItemBankEvidenceKind}

    assert "dif_not_applicable" in evidence_kinds


def test_calibration_accepts_governed_dif_not_applicable_evidence() -> None:
    """A governed N/A determination can satisfy only the DIF applicability gate."""
    calibrated = transition_item_bank_record(
        _pilot_record(),
        ItemBankLifecycleState.CALIBRATED,
        evidence_references=(
            *_base_calibration_evidence(),
            _evidence(
                ItemBankEvidenceKind.DIF_NOT_APPLICABLE,
                "dif_not_applicable",
                "d",
            ),
        ),
        transition_reason_id="calibration_completed",
    )

    assert calibrated.lifecycle_state is ItemBankLifecycleState.CALIBRATED
    assert ItemBankEvidenceKind.DIF_NOT_APPLICABLE in {
        reference.evidence_kind for reference in calibrated.evidence_references
    }
    assert ItemBankEvidenceKind.DIF not in {
        reference.evidence_kind for reference in calibrated.evidence_references
    }


def test_calibration_still_requires_an_explicit_dif_applicability_decision() -> None:
    """Omitting both measured DIF and governed N/A evidence fails closed."""
    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            _pilot_record(),
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=_base_calibration_evidence(),
            transition_reason_id="calibration_completed",
        )

    assert caught.value.code == "missing_transition_evidence"
    assert caught.value.path == "$.evidence_references"
    assert "dif_or_dif_not_applicable" in caught.value.message


def test_calibration_rejects_conflicting_dif_applicability_evidence() -> None:
    """A calibration cannot claim both measured DIF and DIF-not-applicable."""
    with pytest.raises(ItemBankLifecycleError) as caught:
        transition_item_bank_record(
            _pilot_record(),
            ItemBankLifecycleState.CALIBRATED,
            evidence_references=(
                *_base_calibration_evidence(),
                _evidence(ItemBankEvidenceKind.DIF, "dif", "d"),
                _evidence(
                    ItemBankEvidenceKind.DIF_NOT_APPLICABLE,
                    "dif_not_applicable",
                    "e",
                ),
            ),
            transition_reason_id="calibration_completed",
        )

    assert caught.value.code == "conflicting_dif_applicability"
    assert caught.value.path == "$.evidence_references"


def test_existing_measured_dif_calibration_path_is_unchanged() -> None:
    """A comparison design with real DIF evidence retains the prior contract."""
    calibrated = transition_item_bank_record(
        _pilot_record(),
        ItemBankLifecycleState.CALIBRATED,
        evidence_references=(
            *_base_calibration_evidence(),
            _evidence(ItemBankEvidenceKind.DIF, "dif", "d"),
        ),
        transition_reason_id="calibration_completed",
    )

    assert calibrated.lifecycle_state is ItemBankLifecycleState.CALIBRATED
    assert ItemBankEvidenceKind.DIF in {
        reference.evidence_kind for reference in calibrated.evidence_references
    }
