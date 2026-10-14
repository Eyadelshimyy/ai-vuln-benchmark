"""Compile enterprise issue evidence into shared governed scoring requests.

This module extends the accepted enterprise evidence contracts without defining a
parallel request, observation, result, engine, calibration, ranking, or decision
schema. It performs deterministic provenance validation and marshaling only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from fast_mlsirm.rubric import RubricSpecification

from .._contract_safety import bounded_values, freeze_metadata
from .._validation import assessment_error, thaw_json_value
from ..assessment import AssessmentSpec
from ..authorization import build_scoring_request
from ..execution import EvidenceReference, ObservationGranularity, ScoringRequest
from .contracts import (
    MAX_ENTERPRISE_STAKEHOLDERS,
    AtomicIssueRecord,
    CandidateIntervention,
    StakeholderPerspective,
)

_MANAGED_METADATA_KEYS = frozenset(
    {
        "enterprise_atomic_issue_fingerprint",
        "enterprise_issue_content_fingerprint",
        "enterprise_source_record_fingerprints",
        "enterprise_evidence_span_fingerprints",
        "enterprise_counterevidence_fingerprints",
        "enterprise_evidence_reference_fingerprints",
        "enterprise_assertion_records",
        "enterprise_perspective_fingerprints",
        "enterprise_intervention_fingerprints",
    }
)

_T = TypeVar("_T")


def _typed_content_values(
    values: Iterable[_T],
    *,
    name: str,
    expected_type: type[_T],
    fingerprint_attribute: str,
    maximum: int,
) -> tuple[_T, ...]:
    """Return bounded unique records in deterministic fingerprint order."""
    raw = bounded_values(values, name, minimum=0, maximum=maximum)
    for index, value in enumerate(raw):
        if type(value) is not expected_type:
            raise assessment_error(
                f"invalid_{name}",
                f"$.{name}[{index}]",
                f"{name} entries must be {expected_type.__name__} values",
            )
    identities = tuple(getattr(value, fingerprint_attribute) for value in raw)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} entries must be unique",
        )
    return tuple(sorted(raw, key=lambda value: getattr(value, fingerprint_attribute)))


def _perspective_values(
    issue: AtomicIssueRecord,
    values: Iterable[StakeholderPerspective],
) -> tuple[StakeholderPerspective, ...]:
    """Return exact-issue stakeholder perspectives in deterministic order."""
    perspectives = _typed_content_values(
        values,
        name="stakeholder_perspectives",
        expected_type=StakeholderPerspective,
        fingerprint_attribute="perspective_fingerprint",
        maximum=MAX_ENTERPRISE_STAKEHOLDERS,
    )
    declared_sources = set(issue.source_record_fingerprints)
    for index, perspective in enumerate(perspectives):
        if perspective.issue_content_fingerprint != issue.issue_content_fingerprint:
            raise assessment_error(
                "perspective_issue_mismatch",
                f"$.stakeholder_perspectives[{index}].issue_content_fingerprint",
                "stakeholder perspective does not name the supplied issue revision",
            )
        if (
            perspective.value_judgment_span.source_record_fingerprint
            not in declared_sources
        ):
            raise assessment_error(
                "unbound_perspective_source",
                f"$.stakeholder_perspectives[{index}].value_judgment_span",
                "stakeholder perspective evidence must reference a declared source record",
            )
    return perspectives


def _intervention_values(
    issue: AtomicIssueRecord,
    values: Iterable[CandidateIntervention],
) -> tuple[CandidateIntervention, ...]:
    """Return exact-issue candidate interventions in deterministic order."""
    interventions = _typed_content_values(
        values,
        name="candidate_interventions",
        expected_type=CandidateIntervention,
        fingerprint_attribute="intervention_fingerprint",
        maximum=MAX_ENTERPRISE_STAKEHOLDERS,
    )
    for index, intervention in enumerate(interventions):
        if intervention.issue_content_fingerprint != issue.issue_content_fingerprint:
            raise assessment_error(
                "intervention_issue_mismatch",
                f"$.candidate_interventions[{index}].issue_content_fingerprint",
                "candidate intervention does not name the supplied issue revision",
            )
    return interventions


def _validated_issue(issue: AtomicIssueRecord) -> AtomicIssueRecord:
    """Return one accepted atomic issue or fail before attribute access."""
    if type(issue) is not AtomicIssueRecord:
        raise assessment_error(
            "invalid_atomic_issue",
            "$.issue",
            "issue must be an AtomicIssueRecord",
        )
    return issue


def _evidence_references(
    issue: AtomicIssueRecord,
    perspectives: tuple[StakeholderPerspective, ...],
) -> tuple[EvidenceReference, ...]:
    """Return unique issue and perspective evidence in deterministic order."""
    references = issue.evidence_references() + tuple(
        value.value_judgment_span.to_evidence_reference() for value in perspectives
    )
    identities = tuple(value.evidence_fingerprint for value in references)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            "duplicate_enterprise_evidence_references",
            "$.stakeholder_perspectives",
            "enterprise evidence references must be unique across issue and perspective records",
        )
    return tuple(sorted(references, key=lambda value: value.evidence_fingerprint))


def enterprise_issue_evidence_references(
    issue: AtomicIssueRecord,
    *,
    stakeholder_perspectives: Iterable[StakeholderPerspective] = (),
) -> tuple[EvidenceReference, ...]:
    """Compile all exact issue and perspective spans to shared evidence references.

    The returned references preserve supporting, counter, and contextual roles.
    They are suitable for existing ``ScoreObservation`` values and contain no raw
    enterprise source text.
    """
    normalized_issue = _validated_issue(issue)
    perspectives = _perspective_values(normalized_issue, stakeholder_perspectives)
    return _evidence_references(normalized_issue, perspectives)


def _assertion_records(
    issue: AtomicIssueRecord,
    perspectives: tuple[StakeholderPerspective, ...],
) -> list[dict[str, str]]:
    """Return exact evidence-span identities and epistemic roles for audit replay."""
    spans = list(issue.evidence_spans)
    spans.extend(record.evidence_span for record in issue.counterevidence_records)
    spans.extend(value.value_judgment_span for value in perspectives)
    records = [
        {
            "evidence_span_fingerprint": span.evidence_span_fingerprint,
            "assertion_kind": span.assertion_kind.value,
            "evidence_role": span.evidence_role.value,
        }
        for span in spans
    ]
    return sorted(records, key=lambda value: value["evidence_span_fingerprint"])


def _request_metadata(
    *,
    issue: AtomicIssueRecord,
    perspectives: tuple[StakeholderPerspective, ...],
    interventions: tuple[CandidateIntervention, ...],
    evidence_references: tuple[EvidenceReference, ...],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge caller metadata with package-managed enterprise provenance."""
    caller_metadata = thaw_json_value(
        freeze_metadata({} if metadata is None else metadata)
    )
    if any(key in caller_metadata for key in _MANAGED_METADATA_KEYS):
        raise assessment_error(
            "reserved_enterprise_metadata",
            "$.metadata",
            "enterprise provenance metadata is package-managed",
        )
    caller_metadata.update(
        {
            "enterprise_atomic_issue_fingerprint": issue.atomic_issue_fingerprint,
            "enterprise_issue_content_fingerprint": issue.issue_content_fingerprint,
            "enterprise_source_record_fingerprints": list(
                issue.source_record_fingerprints
            ),
            "enterprise_evidence_span_fingerprints": [
                value.evidence_span_fingerprint for value in issue.evidence_spans
            ],
            "enterprise_counterevidence_fingerprints": [
                value.counterevidence_fingerprint
                for value in issue.counterevidence_records
            ],
            "enterprise_evidence_reference_fingerprints": [
                value.evidence_fingerprint for value in evidence_references
            ],
            "enterprise_assertion_records": _assertion_records(issue, perspectives),
            "enterprise_perspective_fingerprints": [
                value.perspective_fingerprint for value in perspectives
            ],
            "enterprise_intervention_fingerprints": [
                value.intervention_fingerprint for value in interventions
            ],
        }
    )
    return caller_metadata


def build_enterprise_issue_scoring_request(
    *,
    request_id: str,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    issue: AtomicIssueRecord,
    response_id: str,
    task_id: str,
    task_revision_fingerprint: str,
    task_family_id: str,
    occasion_id: str,
    criterion_ids: Iterable[str],
    response_character_count: int,
    response_unit_count: int,
    stakeholder_perspectives: Iterable[StakeholderPerspective] = (),
    candidate_interventions: Iterable[CandidateIntervention] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ScoringRequest:
    """Compile an atomic issue into the authoritative criterion-level request.

    The issue identity becomes the shared respondent identity, while the exact
    issue-content fingerprint and caller-supplied content counts define the
    scored response revision. Stakeholder judgments and candidate interventions
    are retained as provenance only; this function performs no scoring, causal,
    utility, ranking, or queue-routing arithmetic.
    """
    normalized_issue = _validated_issue(issue)
    perspectives = _perspective_values(
        normalized_issue,
        stakeholder_perspectives,
    )
    interventions = _intervention_values(
        normalized_issue,
        candidate_interventions,
    )
    evidence_references = _evidence_references(normalized_issue, perspectives)
    return build_scoring_request(
        request_id=request_id,
        assessment=assessment,
        rubric=rubric,
        granularity=ObservationGranularity.CRITERION_LEVEL,
        respondent_id=normalized_issue.issue_id,
        response_id=response_id,
        task_id=task_id,
        task_revision_fingerprint=task_revision_fingerprint,
        task_family_id=task_family_id,
        occasion_id=occasion_id,
        criterion_ids=criterion_ids,
        response_content_fingerprint=normalized_issue.issue_content_fingerprint,
        response_character_count=response_character_count,
        response_unit_count=response_unit_count,
        metadata=_request_metadata(
            issue=normalized_issue,
            perspectives=perspectives,
            interventions=interventions,
            evidence_references=evidence_references,
            metadata=metadata,
        ),
    )


__all__ = [
    "build_enterprise_issue_scoring_request",
    "enterprise_issue_evidence_references",
]
