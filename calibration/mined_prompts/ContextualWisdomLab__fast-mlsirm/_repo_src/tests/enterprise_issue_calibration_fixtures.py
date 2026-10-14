"""Reusable fixtures for enterprise issue calibration handoff tests."""

from __future__ import annotations

import hashlib
from typing import Any

from fast_mlsirm.scoring import (
    EngineDescriptor,
    EvidenceRole,
    ObservationStatus,
    ScoringRequest,
    ScoringResult,
    build_scoring_request,
    build_scoring_result,
)
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    build_enterprise_issue_score_observation,
    build_enterprise_issue_scoring_request,
    enterprise_issue_evidence_references,
)
from scoring_execution_fixtures import assessment, automated_engine, rubric

CRITERION_IDS = ("claim_support", "source_alignment")
_AUTHORIZATION_METADATA_KEYS = frozenset(
    {
        "engine_policy_fingerprint",
        "allow_human_raters",
        "allow_automated_raters",
        "permitted_engine_ids",
    }
)


def _digest(value: str) -> str:
    """Return one deterministic SHA-256 fixture fingerprint."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _issue(
    label: str,
    *,
    include_counterevidence: bool = False,
) -> AtomicIssueRecord:
    """Return one source-text-free issue revision for calibration tests."""
    issue_content_fingerprint = _digest(f"issue-content:{label}")
    source = EnterpriseSourceRecord(
        source_id=f"source_{label}",
        source_family_id="enterprise_source",
        source_content_fingerprint=_digest(f"source-content:{label}"),
        source_character_count=240,
        metadata={"source_channel": "offline_fixture"},
    )
    supporting = EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id=f"supporting_{label}",
        span_content_fingerprint=_digest(f"supporting-span:{label}"),
        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
        start_offset=10,
        end_offset=30,
        metadata={"fixture_kind": "supporting_evidence"},
    )
    counterevidence_records: tuple[CounterevidenceRecord, ...] = ()
    if include_counterevidence:
        counter_span = EvidenceSpanRecord(
            source_id=source.source_id,
            source_record_fingerprint=source.source_record_fingerprint,
            span_id=f"counter_{label}",
            span_content_fingerprint=_digest(f"counter-span:{label}"),
            assertion_kind=EnterpriseAssertionKind.COUNTEREVIDENCE,
            start_offset=40,
            end_offset=60,
            metadata={"fixture_kind": "counter_evidence"},
        )
        counterevidence_records = (
            CounterevidenceRecord(
                counterevidence_id=f"counter_record_{label}",
                issue_content_fingerprint=issue_content_fingerprint,
                evidence_span=counter_span,
                metadata={"verification_state": "source_verified"},
            ),
        )
    return AtomicIssueRecord(
        issue_id=f"issue_{label}",
        issue_family_id="service_risk",
        issue_content_fingerprint=issue_content_fingerprint,
        source_record_fingerprints=(source.source_record_fingerprint,),
        evidence_spans=(supporting,),
        counterevidence_records=counterevidence_records,
        metadata={"decision_scope": "offline_pilot"},
    )


def _engine(label: str) -> EngineDescriptor:
    """Return one deterministic policy-authorized automated judge identity."""
    engine_id = "alternate_engine" if label == "beta" else "fixture_engine"
    return automated_engine(
        engine_id=engine_id,
        engine_family_id=f"engine_family_{label}",
        model_id=f"model_{label}",
        prompt_template_fingerprint=_digest(f"prompt:{label}"),
    )


def _request(
    issue: AtomicIssueRecord,
    *,
    task_label: str,
    engine_label: str,
) -> ScoringRequest:
    """Return one exact enterprise criterion-level request."""
    return build_enterprise_issue_scoring_request(
        request_id=f"request_{issue.issue_id}_{task_label}_{engine_label}",
        assessment=assessment(),
        rubric=rubric(),
        issue=issue,
        response_id=f"response_{issue.issue_id}_{task_label}",
        task_id=f"task_{task_label}",
        task_revision_fingerprint=_digest(f"task-revision:{task_label}"),
        task_family_id="evidence_review",
        occasion_id="pilot_occasion",
        criterion_ids=CRITERION_IDS,
        response_character_count=160,
        response_unit_count=8,
        metadata={"deployment_stage": "offline_fixture"},
    )


def _enterprise_observations(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    engine: EngineDescriptor,
    scores: tuple[int, int],
    abstain_first: bool = False,
):
    """Return complete enterprise observations for one request."""
    references = enterprise_issue_evidence_references(issue)
    observations = []
    for index, (criterion_id, score) in enumerate(
        zip(CRITERION_IDS, scores, strict=True)
    ):
        abstained = abstain_first and index == 0
        observations.append(
            build_enterprise_issue_score_observation(
                observation_id=(
                    f"observation_{issue.issue_id}_{request.task_id}_"
                    f"{engine.engine_id}_{criterion_id}"
                ),
                request=request,
                engine=engine,
                criterion_id=criterion_id,
                status=(
                    ObservationStatus.ABSTAINED
                    if abstained
                    else ObservationStatus.SCORED
                ),
                score_category=None if abstained else score,
                reason_code="insufficient_evidence" if abstained else None,
                evidence_references=() if abstained else references,
                confidence_metadata={"review_state": "fixture_complete"},
            )
        )
    return tuple(observations)


def _result(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    engine: EngineDescriptor,
    scores: tuple[int, int],
    abstain_first: bool = False,
) -> ScoringResult:
    """Return one complete governed enterprise result."""
    return build_scoring_result(
        result_id=f"result_{request.request_id}_{engine.engine_id}",
        request=request,
        engine=engine,
        observations=_enterprise_observations(
            issue=issue,
            request=request,
            engine=engine,
            scores=scores,
            abstain_first=abstain_first,
        ),
        execution_attempt=1,
        diagnostics={"execution_mode": "offline_fixture"},
    )


def _execution(
    *,
    issue_label: str,
    task_label: str,
    engine_label: str,
    scores: tuple[int, int],
):
    """Return one matched issue, request, result, and engine execution."""
    issue = _issue(issue_label)
    engine = _engine(engine_label)
    request = _request(
        issue,
        task_label=task_label,
        engine_label=engine_label,
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=scores,
    )
    return issue, request, result, engine


def _rebuild_request(
    request: ScoringRequest,
    **overrides: Any,
) -> ScoringRequest:
    """Rebuild one request through the shared factory for adversarial replay."""
    values: dict[str, Any] = {
        "request_id": request.request_id,
        "assessment": assessment(),
        "rubric": rubric(),
        "granularity": request.granularity,
        "respondent_id": request.respondent_id,
        "response_id": request.response_id,
        "task_id": request.task_id,
        "task_revision_fingerprint": request.task_revision_fingerprint,
        "task_family_id": request.task_family_id,
        "occasion_id": request.occasion_id,
        "criterion_ids": request.criterion_ids,
        "response_content_fingerprint": request.response_content_fingerprint,
        "response_character_count": request.response_character_count,
        "response_unit_count": request.response_unit_count,
        "metadata": request.to_dict()["metadata"],
    }
    values.update(overrides)
    metadata = dict(values["metadata"])
    for key in _AUTHORIZATION_METADATA_KEYS:
        metadata.pop(key, None)
    values["metadata"] = metadata
    return build_scoring_request(**values)


def _result_with_replacement(
    *,
    issue: AtomicIssueRecord,
    request: ScoringRequest,
    engine: EngineDescriptor,
    replacement,
) -> ScoringResult:
    """Replace the claim observation while retaining complete result coverage."""
    observations = list(
        _enterprise_observations(
            issue=issue,
            request=request,
            engine=engine,
            scores=(1, 2),
        )
    )
    observations[0] = replacement
    return build_scoring_result(
        result_id=f"result_replaced_{request.task_id}_{engine.engine_id}",
        request=request,
        engine=engine,
        observations=observations,
        execution_attempt=1,
        diagnostics={},
    )


def _managed_observation_metadata(
    issue: AtomicIssueRecord,
    references,
) -> dict[str, object]:
    """Return exact package-managed metadata for adversarial observations."""
    supporting = sum(
        value.evidence_role is EvidenceRole.SUPPORTING for value in references
    )
    counter = sum(value.evidence_role is EvidenceRole.COUNTER for value in references)
    context = sum(value.evidence_role is EvidenceRole.CONTEXT for value in references)
    return {
        "enterprise_atomic_issue_fingerprint": issue.atomic_issue_fingerprint,
        "enterprise_issue_content_fingerprint": issue.issue_content_fingerprint,
        "enterprise_observation_evidence_fingerprints": [
            value.evidence_fingerprint for value in references
        ],
        "enterprise_supporting_evidence_count": supporting,
        "enterprise_counter_evidence_count": counter,
        "enterprise_context_evidence_count": context,
    }
