"""Compile enterprise criterion evidence into shared score observations.

The adapter reuses :class:`fast_mlsirm.scoring.ScoreObservation` and adds only
enterprise-specific provenance and evidence gates. It performs no scoring,
calibration, aggregation, ranking, utility, causal, or sentiment arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._contract_safety import bounded_values, freeze_metadata
from .._validation import assessment_error, fingerprint, thaw_json_value
from ..execution import (
    MAX_EVIDENCE_REFERENCES,
    EngineDescriptor,
    EvidenceReference,
    EvidenceRole,
    ObservationStatus,
    ScoreObservation,
    ScoringRequest,
    build_score_observation,
)

_MANAGED_CONFIDENCE_KEYS = frozenset(
    {
        "enterprise_atomic_issue_fingerprint",
        "enterprise_issue_content_fingerprint",
        "enterprise_observation_evidence_fingerprints",
        "enterprise_supporting_evidence_count",
        "enterprise_counter_evidence_count",
        "enterprise_context_evidence_count",
    }
)


def _observation_status(value: ObservationStatus | str) -> ObservationStatus:
    """Return one supported observation status with a stable failure boundary."""
    if type(value) is ObservationStatus:
        return value
    if type(value) is not str:
        raise assessment_error(
            "invalid_observation_status",
            "$.status",
            "status must be a supported observation status",
        )
    try:
        return ObservationStatus(value)
    except Exception:
        raise assessment_error(
            "invalid_observation_status",
            "$.status",
            "status must be a supported observation status",
        ) from None


def _enterprise_request_context(
    request: ScoringRequest,
) -> tuple[str, str, frozenset[str], bool]:
    """Return verified enterprise provenance carried by one shared request."""
    if type(request) is not ScoringRequest:
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    metadata = thaw_json_value(request.metadata)
    required = (
        "enterprise_atomic_issue_fingerprint",
        "enterprise_issue_content_fingerprint",
        "enterprise_evidence_reference_fingerprints",
        "enterprise_counterevidence_fingerprints",
    )
    if any(key not in metadata for key in required):
        raise assessment_error(
            "missing_enterprise_request_provenance",
            "$.request.metadata",
            "request must be produced by build_enterprise_issue_scoring_request",
        )
    atomic_issue_fingerprint = fingerprint(
        metadata["enterprise_atomic_issue_fingerprint"],
        "enterprise_atomic_issue_fingerprint",
        "$.request.metadata.enterprise_atomic_issue_fingerprint",
    )
    issue_content_fingerprint = fingerprint(
        metadata["enterprise_issue_content_fingerprint"],
        "enterprise_issue_content_fingerprint",
        "$.request.metadata.enterprise_issue_content_fingerprint",
    )
    raw_evidence = bounded_values(
        metadata["enterprise_evidence_reference_fingerprints"],
        "enterprise_evidence_reference_fingerprints",
        minimum=1,
        maximum=MAX_EVIDENCE_REFERENCES,
    )
    evidence_fingerprints = tuple(
        fingerprint(
            value,
            "enterprise_evidence_reference_fingerprints",
            f"$.request.metadata.enterprise_evidence_reference_fingerprints[{index}]",
        )
        for index, value in enumerate(raw_evidence)
    )
    if len(set(evidence_fingerprints)) != len(evidence_fingerprints):
        raise assessment_error(
            "duplicate_enterprise_request_evidence",
            "$.request.metadata.enterprise_evidence_reference_fingerprints",
            "enterprise request evidence fingerprints must be unique",
        )
    raw_counterevidence = bounded_values(
        metadata["enterprise_counterevidence_fingerprints"],
        "enterprise_counterevidence_fingerprints",
        minimum=0,
        maximum=MAX_EVIDENCE_REFERENCES,
    )
    counterevidence_fingerprints = tuple(
        fingerprint(
            value,
            "enterprise_counterevidence_fingerprints",
            f"$.request.metadata.enterprise_counterevidence_fingerprints[{index}]",
        )
        for index, value in enumerate(raw_counterevidence)
    )
    if len(set(counterevidence_fingerprints)) != len(counterevidence_fingerprints):
        raise assessment_error(
            "duplicate_enterprise_request_counterevidence",
            "$.request.metadata.enterprise_counterevidence_fingerprints",
            "enterprise request counterevidence fingerprints must be unique",
        )
    return (
        atomic_issue_fingerprint,
        issue_content_fingerprint,
        frozenset(evidence_fingerprints),
        bool(counterevidence_fingerprints),
    )


def _observation_evidence(
    values: Iterable[EvidenceReference],
    *,
    available_fingerprints: frozenset[str],
) -> tuple[EvidenceReference, ...]:
    """Return unique request-bound evidence in deterministic content order."""
    raw = bounded_values(
        values,
        "evidence_references",
        minimum=0,
        maximum=MAX_EVIDENCE_REFERENCES,
    )
    for index, value in enumerate(raw):
        if type(value) is not EvidenceReference:
            raise assessment_error(
                "invalid_evidence_reference",
                f"$.evidence_references[{index}]",
                "evidence entries must be EvidenceReference values",
            )
    identities = tuple(value.evidence_fingerprint for value in raw)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            "duplicate_evidence_reference",
            "$.evidence_references",
            "evidence references must be unique",
        )
    unknown = set(identities).difference(available_fingerprints)
    if unknown:
        raise assessment_error(
            "unbound_enterprise_observation_evidence",
            "$.evidence_references",
            "observation evidence must be declared by the enterprise scoring request",
        )
    return tuple(sorted(raw, key=lambda value: value.evidence_fingerprint))


def _evidence_counts(
    values: tuple[EvidenceReference, ...],
) -> tuple[int, int, int]:
    """Return supporting, counter, and context evidence counts."""
    supporting = sum(
        value.evidence_role is EvidenceRole.SUPPORTING for value in values
    )
    counter = sum(value.evidence_role is EvidenceRole.COUNTER for value in values)
    context = sum(value.evidence_role is EvidenceRole.CONTEXT for value in values)
    return supporting, counter, context


def _confidence_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    atomic_issue_fingerprint: str,
    issue_content_fingerprint: str,
    evidence_references: tuple[EvidenceReference, ...],
    evidence_counts: tuple[int, int, int],
) -> dict[str, Any]:
    """Merge caller confidence metadata with exact enterprise provenance."""
    caller_metadata = thaw_json_value(
        freeze_metadata({} if metadata is None else metadata)
    )
    if any(key in caller_metadata for key in _MANAGED_CONFIDENCE_KEYS):
        raise assessment_error(
            "reserved_enterprise_confidence_metadata",
            "$.confidence_metadata",
            "enterprise observation provenance is package-managed",
        )
    supporting, counter, context = evidence_counts
    caller_metadata.update(
        {
            "enterprise_atomic_issue_fingerprint": atomic_issue_fingerprint,
            "enterprise_issue_content_fingerprint": issue_content_fingerprint,
            "enterprise_observation_evidence_fingerprints": [
                value.evidence_fingerprint for value in evidence_references
            ],
            "enterprise_supporting_evidence_count": supporting,
            "enterprise_counter_evidence_count": counter,
            "enterprise_context_evidence_count": context,
        }
    )
    return caller_metadata


def build_enterprise_issue_score_observation(
    *,
    observation_id: str,
    request: ScoringRequest,
    engine: EngineDescriptor,
    criterion_id: str,
    status: ObservationStatus | str,
    score_category: int | None = None,
    reason_code: str | None = None,
    evidence_references: Iterable[EvidenceReference] = (),
    confidence_metadata: Mapping[str, Any] | None = None,
) -> ScoreObservation:
    """Build one shared criterion observation with enterprise evidence gates.

    Evidence must be a subset of the exact references compiled into ``request``.
    Every non-abstained observation requires supporting evidence and, when the
    issue declares counterevidence, at least one counterevidence reference.
    Abstention remains available when the evidence is insufficient. The function
    returns the authoritative shared ``ScoreObservation`` and adds no competing
    observation schema or numerical scoring behavior.
    """
    (
        atomic_issue_fingerprint,
        issue_content_fingerprint,
        available_fingerprints,
        counterevidence_declared,
    ) = _enterprise_request_context(request)
    normalized_status = _observation_status(status)
    normalized_evidence = _observation_evidence(
        evidence_references,
        available_fingerprints=available_fingerprints,
    )
    supporting, counter, context = _evidence_counts(normalized_evidence)
    if normalized_status is not ObservationStatus.ABSTAINED:
        if supporting == 0:
            raise assessment_error(
                "missing_enterprise_supporting_evidence",
                "$.evidence_references",
                "non-abstained enterprise observations require supporting evidence",
            )
        if counterevidence_declared and counter == 0:
            raise assessment_error(
                "missing_enterprise_counter_evidence",
                "$.evidence_references",
                "enterprise observations must represent declared counterevidence",
            )
    return build_score_observation(
        observation_id=observation_id,
        request=request,
        engine=engine,
        criterion_id=criterion_id,
        status=normalized_status,
        score_category=score_category,
        reason_code=reason_code,
        evidence_references=normalized_evidence,
        confidence_metadata=_confidence_metadata(
            metadata=confidence_metadata,
            atomic_issue_fingerprint=atomic_issue_fingerprint,
            issue_content_fingerprint=issue_content_fingerprint,
            evidence_references=normalized_evidence,
            evidence_counts=(supporting, counter, context),
        ),
    )


__all__ = ["build_enterprise_issue_score_observation"]
