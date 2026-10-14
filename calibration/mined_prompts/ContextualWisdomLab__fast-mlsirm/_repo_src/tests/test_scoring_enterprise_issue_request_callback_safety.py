"""Regression tests for callback-free enterprise request record admission."""

from __future__ import annotations

from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CandidateIntervention,
    StakeholderPerspective,
    enterprise_issue_evidence_references,
)
from fast_mlsirm.scoring.enterprise_issue.request import _typed_content_values


def _hostile_record(record_type: type[Any]) -> tuple[Any, list[str]]:
    """Return an uninitialized subclass that records any instance attribute read."""

    callbacks: list[str] = []

    class HostileRecord(record_type):  # type: ignore[misc, valid-type]
        def __getattribute__(self, name: str) -> Any:
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            callbacks.append(name)
            raise AssertionError(f"caller callback executed for {name}")

    return object.__new__(HostileRecord), callbacks


def test_public_issue_helper_rejects_subclass_before_provenance_callbacks() -> None:
    """Issue provenance is not read from a caller-defined record subclass."""

    issue, callbacks = _hostile_record(AtomicIssueRecord)

    with pytest.raises(AssessmentSpecError) as caught:
        enterprise_issue_evidence_references(issue)

    assert caught.value.code == "invalid_atomic_issue"
    assert callbacks == []


@pytest.mark.parametrize(
    ("record_type", "name", "fingerprint_attribute"),
    [
        (
            StakeholderPerspective,
            "stakeholder_perspectives",
            "perspective_fingerprint",
        ),
        (
            CandidateIntervention,
            "candidate_interventions",
            "intervention_fingerprint",
        ),
    ],
)
def test_typed_request_records_reject_subclasses_before_fingerprint_callbacks(
    record_type: type[Any],
    name: str,
    fingerprint_attribute: str,
) -> None:
    """Shared request-record admission rejects subclasses before fingerprint reads."""

    record, callbacks = _hostile_record(record_type)

    with pytest.raises(AssessmentSpecError) as caught:
        _typed_content_values(
            (record,),
            name=name,
            expected_type=record_type,
            fingerprint_attribute=fingerprint_attribute,
            maximum=1,
        )

    assert caught.value.code == f"invalid_{name}"
    assert callbacks == []
