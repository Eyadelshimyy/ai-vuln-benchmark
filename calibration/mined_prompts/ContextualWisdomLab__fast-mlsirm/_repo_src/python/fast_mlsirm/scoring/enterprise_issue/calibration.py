"""Replay enterprise issue provenance before shared many-facet calibration.

The adapters verify that governed request/result/engine executions remain bound
to exact :class:`AtomicIssueRecord` revisions and then delegate projection and
bundle assembly to the existing shared criterion-level calibration contracts.
They perform no likelihood, gradient, Hessian, optimization, scoring, ranking,
utility, fairness, validity, or causal arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NoReturn

from .._contract_safety import bounded_values
from .._validation import assessment_error, thaw_json_value
from ..calibration import (
    MAX_SCORING_FACETS_RATINGS,
    ScoringFacetsCalibrationBundle,
    ScoringFacetsRatingRecord,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
)
from ..execution import (
    EngineDescriptor,
    EvidenceRole,
    ObservationStatus,
    ScoreObservation,
    ScoringRequest,
    ScoringResult,
)
from .contracts import AtomicIssueRecord
from .observation import _enterprise_request_context, _evidence_counts

MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS = MAX_SCORING_FACETS_RATINGS
"""Maximum enterprise scoring executions accepted by one bundle assembly."""


def _calibration_error(path: str, message: str) -> NoReturn:
    """Raise one stable enterprise calibration replay error."""
    raise assessment_error(
        "enterprise_calibration_provenance_mismatch",
        path,
        message,
    )


def _request_issue_evidence(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    available_evidence_fingerprints: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Replay issue-owned request metadata and return exact evidence identities."""
    metadata = thaw_json_value(request.metadata)
    expected = {
        "enterprise_source_record_fingerprints": list(issue.source_record_fingerprints),
        "enterprise_evidence_span_fingerprints": [
            value.evidence_span_fingerprint for value in issue.evidence_spans
        ],
        "enterprise_counterevidence_fingerprints": [
            value.counterevidence_fingerprint for value in issue.counterevidence_records
        ],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        _calibration_error(
            "$.request.metadata",
            "request enterprise issue provenance does not replay the supplied issue",
        )

    issue_evidence = issue.evidence_references()
    issue_evidence_fingerprints = frozenset(
        value.evidence_fingerprint for value in issue_evidence
    )
    if not issue_evidence_fingerprints.issubset(available_evidence_fingerprints):
        _calibration_error(
            "$.request.metadata.enterprise_evidence_reference_fingerprints",
            "request evidence does not retain every supplied issue evidence reference",
        )
    supporting = frozenset(
        value.evidence_fingerprint
        for value in issue_evidence
        if value.evidence_role is EvidenceRole.SUPPORTING
    )
    counter = frozenset(
        value.evidence_fingerprint
        for value in issue_evidence
        if value.evidence_role is EvidenceRole.COUNTER
    )
    return supporting, counter


def _observation_metadata(
    *,
    issue: AtomicIssueRecord,
    observation: ScoreObservation,
    evidence_fingerprints: tuple[str, ...],
    evidence_counts: tuple[int, int, int],
    path: str,
) -> None:
    """Replay package-managed enterprise observation provenance."""
    supporting, counter, context = evidence_counts
    expected: dict[str, Any] = {
        "enterprise_atomic_issue_fingerprint": issue.atomic_issue_fingerprint,
        "enterprise_issue_content_fingerprint": issue.issue_content_fingerprint,
        "enterprise_observation_evidence_fingerprints": list(evidence_fingerprints),
        "enterprise_supporting_evidence_count": supporting,
        "enterprise_counter_evidence_count": counter,
        "enterprise_context_evidence_count": context,
    }
    metadata = thaw_json_value(observation.confidence_metadata)
    if any(metadata.get(key) != value for key, value in expected.items()):
        _calibration_error(
            f"{path}.confidence_metadata",
            "observation enterprise provenance does not replay exactly",
        )


def build_enterprise_issue_facets_rating_records(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoringFacetsRatingRecord, ...]:
    """Project one exact enterprise execution into shared facets ratings.

    The function replays the package-managed issue, request, observation, and
    evidence identities before calling
    :func:`~fast_mlsirm.scoring.build_scoring_facets_rating_records`. The shared
    builder remains authoritative for request, result, and engine binding.
    Returned records are the existing shared calibration records; no
    enterprise-specific rating, design, fit, or result schema is introduced.

    Passing this boundary proves provenance consistency only. It does not prove
    that an issue is true or material, that a rating is valid or fair, that the
    many-facet model is adequate, or that any intervention is causally effective.
    """
    if type(issue) is not AtomicIssueRecord:
        raise assessment_error(
            "invalid_atomic_issue",
            "$.issue",
            "issue must be an exact AtomicIssueRecord",
        )
    if type(request) is not ScoringRequest:
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be an exact ScoringRequest",
        )
    if type(result) is not ScoringResult:
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "result must be an exact ScoringResult",
        )
    if type(engine) is not EngineDescriptor:
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an exact EngineDescriptor",
        )

    (
        atomic_issue_fingerprint,
        issue_content_fingerprint,
        available_evidence_fingerprints,
        _,
    ) = _enterprise_request_context(request)
    if atomic_issue_fingerprint != issue.atomic_issue_fingerprint:
        _calibration_error(
            "$.request.metadata.enterprise_atomic_issue_fingerprint",
            "request atomic issue provenance does not match the supplied issue",
        )
    if issue_content_fingerprint != issue.issue_content_fingerprint:
        _calibration_error(
            "$.request.metadata.enterprise_issue_content_fingerprint",
            "request issue content provenance does not match the supplied issue",
        )
    if request.respondent_id != issue.issue_id:
        _calibration_error(
            "$.request.respondent_id",
            "request respondent identity does not match the supplied issue",
        )
    if request.response_content_fingerprint != issue.issue_content_fingerprint:
        _calibration_error(
            "$.request.response_content_fingerprint",
            "request response revision does not match the supplied issue",
        )
    required_supporting, required_counter = _request_issue_evidence(
        issue=issue,
        request=request,
        available_evidence_fingerprints=available_evidence_fingerprints,
    )

    for index, observation in enumerate(result.observations):
        path = f"$.result.observations[{index}]"
        if type(observation) is not ScoreObservation:
            raise assessment_error(
                "invalid_score_observation",
                path,
                "result observations must contain exact ScoreObservation values",
            )
        evidence_fingerprints = tuple(
            value.evidence_fingerprint for value in observation.evidence_references
        )
        observation_evidence = frozenset(evidence_fingerprints)
        if not observation_evidence.issubset(available_evidence_fingerprints):
            _calibration_error(
                f"{path}.evidence_references",
                "observation evidence is not declared by the enterprise request",
            )
        evidence_counts = _evidence_counts(observation.evidence_references)
        if observation.status is not ObservationStatus.ABSTAINED:
            if not observation_evidence.intersection(required_supporting):
                _calibration_error(
                    f"{path}.evidence_references",
                    "non-abstained observations require supporting evidence from the supplied issue",
                )
            if required_counter and not observation_evidence.intersection(
                required_counter
            ):
                _calibration_error(
                    f"{path}.evidence_references",
                    "enterprise observations must retain supplied issue counterevidence",
                )
        _observation_metadata(
            issue=issue,
            observation=observation,
            evidence_fingerprints=evidence_fingerprints,
            evidence_counts=evidence_counts,
            path=path,
        )

    return build_scoring_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )


def build_enterprise_issue_facets_calibration_bundle(
    executions: Iterable[
        tuple[AtomicIssueRecord, ScoringRequest, ScoringResult, EngineDescriptor]
    ],
    *,
    require_connected: bool = True,
) -> ScoringFacetsCalibrationBundle:
    """Assemble exact enterprise scoring executions into one shared bundle.

    Each execution must be an exact four-value tuple containing an
    :class:`AtomicIssueRecord`, its governed :class:`ScoringRequest`, the bound
    :class:`ScoringResult`, and the producing :class:`EngineDescriptor`. Every
    execution first passes the enterprise provenance replay implemented by
    :func:`build_enterprise_issue_facets_rating_records`; the resulting shared
    records are then delegated unchanged to
    :func:`~fast_mlsirm.scoring.build_scoring_facets_calibration_bundle`.

    The returned object is the existing criterion-separated shared calibration
    bundle. This adapter does not fit a model, combine criteria, infer issue
    importance, establish validity or fairness, or justify causal or automated
    high-stakes decisions. Those claims require downstream diagnostics,
    recovery evidence, relation-safe validation, and human review.
    """
    values = bounded_values(
        executions,
        "executions",
        minimum=1,
        maximum=MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS,
    )

    def validated_records() -> Iterable[ScoringFacetsRatingRecord]:
        """Yield replay-validated records for bounded shared consumption."""
        for index, execution in enumerate(values):
            if type(execution) is not tuple or len(execution) != 4:
                raise assessment_error(
                    "invalid_enterprise_calibration_execution",
                    f"$.executions[{index}]",
                    (
                        "each execution must be an exact four-value tuple of issue, "
                        "request, result, and engine"
                    ),
                )
            issue, request, result, engine = execution
            yield from build_enterprise_issue_facets_rating_records(
                issue=issue,
                request=request,
                result=result,
                engine=engine,
            )

    return build_scoring_facets_calibration_bundle(
        validated_records(),
        require_connected=require_connected,
    )


__all__ = [
    "MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS",
    "build_enterprise_issue_facets_calibration_bundle",
    "build_enterprise_issue_facets_rating_records",
]
