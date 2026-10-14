"""Callback-safety regressions for rubric audit enum-valued controls."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import (
    AuditSeverity,
    CandidateAuditFinding,
    CandidateAuditReport,
    CandidateLifecycleState,
)
from fast_mlsirm.rubric.audit import PilotCandidateRecord as _CorePilotCandidateRecord


class _HostileString(str):
    """String subclass that records any callback dispatch during enum lookup."""

    callbacks = 0

    def __hash__(self) -> int:
        type(self).callbacks += 1
        raise AssertionError("caller __hash__ must not execute")

    def __eq__(self, other: object) -> bool:
        type(self).callbacks += 1
        raise AssertionError("caller __eq__ must not execute")


def _hostile(value: str) -> _HostileString:
    """Return one fresh hostile serialized enum value."""
    _HostileString.callbacks = 0
    return _HostileString(value)


def _core_pilot_record(lifecycle_state: object) -> _CorePilotCandidateRecord:
    """Build the internal pilot record with one caller-controlled state value."""
    return _CorePilotCandidateRecord(
        pilot_study_id="pilot_study_alpha",
        query_testlet_id="query_testlet_alpha",
        generator_family_id="generator_family_alpha",
        judge_policy_id="judge_policy_alpha",
        occasion_id="occasion_window_alpha",
        item_id="generated_item_alpha",
        candidate_fingerprint="a" * 64,
        audit_report_fingerprint="b" * 64,
        audit_policy_id="generated_item_audit",
        audit_policy_version="1.0.0",
        blueprint_id="blueprint_alpha",
        rubric_id="rubric_alpha",
        rubric_version="1.0.0",
        lifecycle_state=lifecycle_state,
    )


@pytest.mark.parametrize(
    ("factory", "serialized"),
    [
        (
            lambda value: CandidateAuditFinding(
                finding_code="audit_finding_alpha",
                severity=value,
                path="$.stem",
                message="review required",
            ),
            "blocking",
        ),
        (
            lambda value: CandidateAuditReport(
                audit_policy_id="generated_item_audit",
                audit_policy_version="1.0.0",
                candidate_fingerprint="a" * 64,
                findings=(),
                lifecycle_state=value,
            ),
            "audited",
        ),
        (_core_pilot_record, "pilot"),
    ],
)
def test_audit_enum_strings_reject_subclasses_without_callback_dispatch(factory, serialized):
    """Caller string subclasses fail closed before Enum lookup can call them."""
    value = _hostile(serialized)
    with pytest.raises(ValueError):
        factory(value)
    assert _HostileString.callbacks == 0


def test_audit_enum_strings_preserve_exact_serialized_values_and_members():
    """Exact strings and exact enum members retain their established semantics."""
    finding = CandidateAuditFinding(
        finding_code="audit_finding_alpha",
        severity="blocking",
        path="$.stem",
        message="review required",
    )
    assert finding.severity is AuditSeverity.BLOCKING

    report = CandidateAuditReport(
        audit_policy_id="generated_item_audit",
        audit_policy_version="1.0.0",
        candidate_fingerprint="a" * 64,
        findings=(),
        lifecycle_state="audited",
    )
    assert report.lifecycle_state is CandidateLifecycleState.AUDITED

    pilot = _core_pilot_record(CandidateLifecycleState.PILOT)
    assert pilot.lifecycle_state is CandidateLifecycleState.PILOT
