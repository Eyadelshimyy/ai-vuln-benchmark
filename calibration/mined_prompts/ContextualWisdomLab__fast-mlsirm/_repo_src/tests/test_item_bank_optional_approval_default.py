"""Regression coverage for optional item-bank approval provenance."""

from __future__ import annotations

import fast_mlsirm.scoring.item_bank as item_bank


def _fp(char: str) -> str:
    """Return one deterministic SHA-256-shaped test fingerprint."""
    return char * 64


def test_draft_factory_defaults_approval_decision_to_none() -> None:
    """Pre-approval lifecycle states need not spell an absent approval digest."""
    entry = item_bank.build_item_bank_entry(
        entry_id="bank_entry",
        item_id="sample_item",
        item_version="1.0.0",
        rubric_fingerprint=_fp("1"),
        blueprint_fingerprint=_fp("2"),
        generation_contract_fingerprint=_fp("3"),
        item_content_fingerprint=_fp("4"),
        lifecycle_state=item_bank.ItemLifecycleState.DRAFT,
        audit_evidence_fingerprints=(),
        screening_result_fingerprints=(),
        pilot_assignment_fingerprints=(),
        calibration_evidence_fingerprints=(),
        metadata={},
    )

    assert entry.approval_decision_fingerprint is None
