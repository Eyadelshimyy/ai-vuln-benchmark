"""Fail-first contracts for the governed item-bank lifecycle."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
import fast_mlsirm.scoring.item_bank as item_bank


def fp(char: str) -> str:
    """Return one deterministic SHA-256-shaped test fingerprint."""
    return char * 64


def build_entry(**overrides: object) -> item_bank.ItemBankEntry:
    """Return one fully evidenced active item-bank entry."""
    values: dict[str, object] = {
        "entry_id": "bank_entry",
        "item_id": "sample_item",
        "item_version": "1.0.0",
        "rubric_fingerprint": fp("1"),
        "blueprint_fingerprint": fp("2"),
        "generation_contract_fingerprint": fp("3"),
        "item_content_fingerprint": fp("4"),
        "lifecycle_state": item_bank.ItemLifecycleState.ACTIVE,
        "audit_evidence_fingerprints": (fp("5"),),
        "screening_result_fingerprints": (fp("6"),),
        "pilot_assignment_fingerprints": (fp("7"),),
        "calibration_evidence_fingerprints": (fp("8"),),
        "approval_decision_fingerprint": fp("9"),
        "metadata": {"bank_family": "essay_bank"},
    }
    values.update(overrides)
    return item_bank.build_item_bank_entry(**values)


def assert_error(code: str, callback) -> None:
    """Assert one stable package-owned contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_entry_is_content_addressed_deeply_immutable_and_order_stable() -> None:
    """Equivalent evidence order yields one immutable item-bank identity."""
    metadata = {"nested_value": {"second_key": 2, "first_key": [1, 2]}}
    first = build_entry(
        calibration_evidence_fingerprints=(fp("b"), fp("8")),
        metadata=metadata,
    )
    second = build_entry(
        calibration_evidence_fingerprints=(fp("8"), fp("b")),
        metadata={"nested_value": {"first_key": [1, 2], "second_key": 2}},
    )
    original = first.entry_fingerprint
    metadata["nested_value"]["first_key"].append(3)

    assert first.entry_fingerprint == second.entry_fingerprint == original
    assert first.entry_handle == f"item_bank_entry_{original[:32]}"
    assert isinstance(first.metadata, MappingProxyType)
    assert first.calibration_evidence_fingerprints == (fp("8"), fp("b"))
    assert first.to_dict()["entry_fingerprint"] == original


def test_entry_requires_exact_provenance_and_factory_sealing() -> None:
    """Direct construction and malformed provenance fail closed."""
    assert_error(
        "invalid_rubric_fingerprint",
        lambda: build_entry(rubric_fingerprint="not-a-digest"),
    )
    assert_error(
        "unverified_item_bank_entry",
        lambda: item_bank.ItemBankEntry(
            entry_id="bank_entry",
            item_id="sample_item",
            item_version="1.0.0",
            rubric_fingerprint=fp("1"),
            blueprint_fingerprint=fp("2"),
            generation_contract_fingerprint=fp("3"),
            item_content_fingerprint=fp("4"),
            lifecycle_state=item_bank.ItemLifecycleState.DRAFT,
            audit_evidence_fingerprints=(),
            screening_result_fingerprints=(),
            pilot_assignment_fingerprints=(),
            calibration_evidence_fingerprints=(),
            approval_decision_fingerprint=None,
            retirement_decision_fingerprint=None,
            predecessor_entry_fingerprint=None,
            metadata={},
        ),
    )


@pytest.mark.parametrize(
    ("state", "overrides", "code"),
    [
        (item_bank.ItemLifecycleState.AUDITED, {"audit_evidence_fingerprints": ()}, "audit_evidence_required"),
        (item_bank.ItemLifecycleState.SCREENED, {"screening_result_fingerprints": ()}, "screening_evidence_required"),
        (item_bank.ItemLifecycleState.PILOTING, {"pilot_assignment_fingerprints": ()}, "pilot_evidence_required"),
        (item_bank.ItemLifecycleState.CALIBRATED, {"calibration_evidence_fingerprints": ()}, "calibration_evidence_required"),
        (item_bank.ItemLifecycleState.APPROVED, {"approval_decision_fingerprint": None}, "approval_decision_required"),
        (item_bank.ItemLifecycleState.ACTIVE, {"approval_decision_fingerprint": None}, "approval_decision_required"),
        (item_bank.ItemLifecycleState.SUSPENDED, {"approval_decision_fingerprint": None}, "approval_decision_required"),
        (item_bank.ItemLifecycleState.RETIRED, {"retirement_decision_fingerprint": None}, "retirement_decision_required"),
    ],
)
def test_lifecycle_states_require_cumulative_governance_evidence(
    state: item_bank.ItemLifecycleState,
    overrides: dict[str, object],
    code: str,
) -> None:
    """Later lifecycle labels cannot be minted from generated JSON alone."""
    assert_error(code, lambda: build_entry(lifecycle_state=state, **overrides))


def test_draft_needs_provenance_but_not_downstream_evidence() -> None:
    """A generated candidate can be represented without claiming readiness."""
    entry = build_entry(
        lifecycle_state="draft",
        audit_evidence_fingerprints=(),
        screening_result_fingerprints=(),
        pilot_assignment_fingerprints=(),
        calibration_evidence_fingerprints=(),
        approval_decision_fingerprint=None,
    )
    assert entry.lifecycle_state is item_bank.ItemLifecycleState.DRAFT
    assert entry.to_dict()["calibration_evidence_fingerprints"] == []


def test_retired_entry_preserves_approval_and_retirement_provenance() -> None:
    """Retirement is an immutable lifecycle record rather than historical deletion."""
    entry = build_entry(
        lifecycle_state="retired",
        retirement_decision_fingerprint=fp("a"),
    )
    payload = entry.to_dict()
    assert payload["approval_decision_fingerprint"] == fp("9")
    assert payload["retirement_decision_fingerprint"] == fp("a")


def test_release_comparability_requires_predecessor_and_linking_evidence() -> None:
    """Nominal score ranges alone cannot assert cross-version comparability."""
    entry = build_entry()
    assert_error(
        "linking_evidence_required",
        lambda: item_bank.build_item_bank_release(
            release_id="bank_release",
            release_version="2.0.0",
            entry_fingerprints=(entry.entry_fingerprint,),
            predecessor_release_fingerprint=fp("c"),
            cross_version_comparable=True,
            linking_evidence_fingerprints=(),
            metadata={},
        ),
    )
    assert_error(
        "predecessor_release_required",
        lambda: item_bank.build_item_bank_release(
            release_id="bank_release",
            release_version="2.0.0",
            entry_fingerprints=(entry.entry_fingerprint,),
            predecessor_release_fingerprint=None,
            cross_version_comparable=True,
            linking_evidence_fingerprints=(fp("d"),),
            metadata={},
        ),
    )

    release = item_bank.build_item_bank_release(
        release_id="bank_release",
        release_version="2.0.0",
        entry_fingerprints=(entry.entry_fingerprint,),
        predecessor_release_fingerprint=fp("c"),
        cross_version_comparable=True,
        linking_evidence_fingerprints=(fp("e"), fp("d")),
        metadata={"population_scope": "pilot_population"},
    )
    assert release.linking_evidence_fingerprints == (fp("d"), fp("e"))
    assert release.release_handle.startswith("item_bank_release_")
    assert release.to_dict()["cross_version_comparable"] is True


def test_release_rejects_duplicate_entries_and_unverified_construction() -> None:
    """Release membership is bounded, unique, and factory-sealed."""
    entry = build_entry()
    assert_error(
        "duplicate_entry_fingerprints",
        lambda: item_bank.build_item_bank_release(
            release_id="bank_release",
            release_version="1.0.0",
            entry_fingerprints=(entry.entry_fingerprint, entry.entry_fingerprint),
            predecessor_release_fingerprint=None,
            cross_version_comparable=False,
            linking_evidence_fingerprints=(),
            metadata={},
        ),
    )
    assert_error(
        "unverified_item_bank_release",
        lambda: item_bank.ItemBankRelease(
            release_id="bank_release",
            release_version="1.0.0",
            entry_fingerprints=(entry.entry_fingerprint,),
            predecessor_release_fingerprint=None,
            cross_version_comparable=False,
            linking_evidence_fingerprints=(),
            metadata={},
        ),
    )


def test_sensitive_metadata_is_rejected() -> None:
    """Lifecycle provenance must not become a raw prompt/response content store."""
    assert_error(
        "sensitive_metadata_field",
        lambda: build_entry(metadata={"response_text": "secret"}),
    )
