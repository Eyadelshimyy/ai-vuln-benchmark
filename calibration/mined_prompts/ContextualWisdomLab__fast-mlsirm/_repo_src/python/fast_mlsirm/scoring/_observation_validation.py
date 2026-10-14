"""Canonical validation for package-owned score observations."""

from __future__ import annotations

from typing import Any

from ._contract_safety import bounded_values
from ._validation import assessment_error
from .execution import (
    EngineDescriptor,
    EvidenceReference,
    MAX_EVIDENCE_REFERENCES,
    ScoreObservation,
    ScoringRequest,
    build_score_observation,
)


def _same_concrete_value(actual: Any, expected: Any) -> bool:
    """Return whether validated values have the same concrete representation."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _same_concrete_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _validate_evidence_reference(value: Any, *, path: str) -> EvidenceReference:
    """Reconstruct one evidence reference before reading its derived identity."""
    if type(value) is not EvidenceReference:
        raise assessment_error(
            "invalid_evidence_reference",
            path,
            "evidence entries must be exact EvidenceReference values",
        )
    rebuilt = EvidenceReference(
        source_id=value.source_id,
        span_id=value.span_id,
        content_fingerprint=value.content_fingerprint,
        evidence_role=value.evidence_role,
        schema_version=value.schema_version,
    )
    for field_name in (
        "source_id",
        "span_id",
        "content_fingerprint",
        "evidence_role",
        "schema_version",
    ):
        if not _same_concrete_value(
            getattr(value, field_name),
            getattr(rebuilt, field_name),
        ):
            raise assessment_error(
                "evidence_reference_validation_mismatch",
                path,
                "evidence reference does not match its normalized contract",
            )
    return rebuilt


def validate_score_observation(
    observation: Any,
    *,
    request: ScoringRequest,
    engine: EngineDescriptor,
    path: str,
) -> ScoreObservation:
    """Validate one observation and all nested values against its factory contract."""
    if type(observation) is not ScoreObservation:
        raise assessment_error(
            "invalid_score_observation",
            path,
            "result observations must be exact ScoreObservation values",
        )
    raw_evidence = bounded_values(
        observation.evidence_references,
        "evidence_references",
        minimum=0,
        maximum=MAX_EVIDENCE_REFERENCES,
    )
    evidence = tuple(
        _validate_evidence_reference(
            value,
            path=f"{path}.evidence_references[{index}]",
        )
        for index, value in enumerate(raw_evidence)
    )
    rebuilt = build_score_observation(
        observation_id=observation.observation_id,
        request=request,
        engine=engine,
        criterion_id=observation.criterion_id,
        status=observation.status,
        score_category=observation.score_category,
        reason_code=observation.reason_code,
        evidence_references=evidence,
        confidence_metadata=observation.confidence_metadata,
    )
    for field_name in (
        "observation_id",
        "request_fingerprint",
        "engine_fingerprint",
        "assessment_fingerprint",
        "rubric_fingerprint",
        "construct_id",
        "granularity",
        "criterion_id",
        "status",
        "score_category",
        "reason_code",
        "evidence_references",
        "confidence_metadata",
        "schema_version",
    ):
        if not _same_concrete_value(
            getattr(observation, field_name),
            getattr(rebuilt, field_name),
        ):
            raise assessment_error(
                "score_observation_validation_mismatch",
                path,
                "score observation does not match its normalized factory contract",
            )
    return observation


__all__: list[str] = []
