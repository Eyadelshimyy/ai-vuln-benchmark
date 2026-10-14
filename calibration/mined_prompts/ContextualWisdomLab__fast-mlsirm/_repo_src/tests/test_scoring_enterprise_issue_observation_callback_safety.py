"""Regression tests for callback-free enterprise observation admission."""

from __future__ import annotations

from typing import Any

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    ObservationStatus,
    ScoringRequest,
)
from fast_mlsirm.scoring.enterprise_issue.observation import (
    _enterprise_request_context,
    _observation_evidence,
    _observation_status,
)


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


def test_enterprise_request_context_rejects_subclass_before_metadata_callback() -> None:
    """Request provenance is not read from a caller-defined request subclass."""

    request, callbacks = _hostile_record(ScoringRequest)

    with pytest.raises(AssessmentSpecError) as caught:
        _enterprise_request_context(request)

    assert caught.value.code == "invalid_scoring_request"
    assert callbacks == []


def test_observation_evidence_rejects_subclass_before_fingerprint_callback() -> None:
    """Evidence provenance is not read from a caller-defined reference subclass."""

    reference, callbacks = _hostile_record(EvidenceReference)

    with pytest.raises(AssessmentSpecError) as caught:
        _observation_evidence((reference,), available_fingerprints=frozenset())

    assert caught.value.code == "invalid_evidence_reference"
    assert callbacks == []


def test_observation_status_rejects_string_subclass_before_enum_callbacks() -> None:
    """Status parsing rejects caller-defined strings before enum hash/equality lookup."""

    callbacks: list[str] = []

    class HostileStatus(str):
        def __hash__(self) -> int:
            callbacks.append("hash")
            raise AssertionError("caller hash callback executed")

        def __eq__(self, other: object) -> bool:
            callbacks.append("eq")
            raise AssertionError("caller equality callback executed")

    with pytest.raises(AssessmentSpecError) as caught:
        _observation_status(HostileStatus(ObservationStatus.SCORED.value))

    assert caught.value.code == "invalid_observation_status"
    assert callbacks == []
