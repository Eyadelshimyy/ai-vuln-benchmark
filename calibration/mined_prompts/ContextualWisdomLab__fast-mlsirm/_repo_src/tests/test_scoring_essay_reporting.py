"""Tests for provenance-bound essay score reports and review routing."""

from __future__ import annotations

from pathlib import Path
import runpy
from types import MappingProxyType
from typing import Any

import pytest

import fast_mlsirm.scoring.essay.reporting as reporting
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    EvidenceRole,
    FixtureOutcome,
    ObservationGranularity,
    ObservationStatus,
    StaticFixtureEngine,
)
from fast_mlsirm.scoring.essay import (
    EssayEvidenceKind,
    EssayReviewFlag,
    build_essay_prompt,
    build_essay_response_evidence,
    build_essay_scoring_request,
    build_essay_submission,
    score_essay_request,
)
from fast_mlsirm.scoring.essay.reporting import (
    EssayScoreReport,
    build_essay_score_report,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
rubric = _FIXTURES["rubric"]


def shared_evidence(
    *,
    source_id: str = "essay_response",
    span_id: str = "response_span",
) -> EvidenceReference:
    """Return one deterministic source-text-free evidence reference."""
    return EvidenceReference(
        source_id=source_id,
        span_id=span_id,
        content_fingerprint="3" * 64,
        evidence_role=EvidenceRole.SUPPORTING,
    )


def essay_request(
    *,
    review_flags: tuple[EssayReviewFlag, ...] = (),
):
    """Return one deterministic essay request with exact task provenance."""
    prompt = build_essay_prompt(
        prompt_id="argument_prompt",
        task_family_id="essay_review",
        prompt_content_fingerprint="1" * 64,
        language_id="english_language",
        genre_id="argument_genre",
        maximum_response_characters=5_000,
        maximum_response_units=1_000,
        metadata={"administration_stage": "pilot"},
    )
    submission = build_essay_submission(
        submission_id="essay_submission",
        prompt=prompt,
        respondent_id="sample_respondent",
        response_id="essay_response",
        response_content_fingerprint="2" * 64,
        response_character_count=800,
        response_unit_count=120,
        review_flags=review_flags,
        metadata={"administration_mode": "offline"},
    )
    evidence = build_essay_response_evidence(
        prompt=prompt,
        submission=submission,
        evidence_reference=shared_evidence(),
        evidence_kind=EssayEvidenceKind.RESPONSE_SPAN,
        start_offset=10,
        end_offset=30,
        metadata={"extractor_id": "manual_review"},
    )
    return build_essay_scoring_request(
        request_id="essay_scoring_request",
        assessment=assessment(),
        rubric=rubric(),
        prompt=prompt,
        submission=submission,
        occasion_id="initial_occasion",
        criterion_ids=("claim_support", "source_alignment"),
        essay_evidence=(evidence,),
        metadata={"workflow_stage": "pilot_review"},
    )


def result_bundle(
    request,
    *,
    claim_status: ObservationStatus = ObservationStatus.SCORED,
    claim_reason: str | None = None,
    claim_evidence: tuple[EvidenceReference, ...] | None = None,
    alignment_status: ObservationStatus = ObservationStatus.SCORED,
    alignment_reason: str | None = None,
):
    """Return one deterministic engine, descriptor, and governed result."""
    evidence = request.essay_evidence[0].evidence_reference
    descriptor = automated_engine()
    engine = StaticFixtureEngine(
        descriptor=descriptor,
        outcomes=(
            FixtureOutcome(
                criterion_id="claim_support",
                status=claim_status,
                score_category=2 if claim_status is ObservationStatus.SCORED else None,
                reason_code=claim_reason,
                evidence_references=(
                    (evidence,) if claim_evidence is None else claim_evidence
                ),
                confidence_metadata={"confidence_band": "reviewed_band"},
            ),
            FixtureOutcome(
                criterion_id="source_alignment",
                status=alignment_status,
                score_category=(
                    1 if alignment_status is ObservationStatus.SCORED else None
                ),
                reason_code=alignment_reason,
                evidence_references=(evidence,),
                confidence_metadata={"confidence_band": "reviewed_band"},
            ),
        ),
    )
    return engine, descriptor, score_essay_request(engine, request)


def assert_error(code: str, callback) -> None:
    """Assert one stable report-contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_reporting_surface_is_explicit_and_documented() -> None:
    """The reporting module exposes only the reviewed report surface."""
    assert set(reporting.__all__) == {
        "EssayScoreReport",
        "MAX_ESSAY_REPORT_REVIEW_TRIGGERS",
        "build_essay_score_report",
    }
    assert EssayScoreReport.__doc__
    assert build_essay_score_report.__doc__


def test_clean_report_is_content_addressed_without_a_validity_claim() -> None:
    """Fully scored evidenced criteria yield no structural review trigger."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    metadata = {"audit_context": {"second_key": 2, "first_key": [1, 2]}}
    first = build_essay_score_report(
        report_id="essay_score_report",
        request=request,
        result=result,
        engine=descriptor,
        metadata=metadata,
    )
    second = build_essay_score_report(
        report_id="essay_score_report",
        request=request,
        result=result,
        engine=descriptor,
        metadata={"audit_context": {"first_key": [1, 2], "second_key": 2}},
    )
    original = first.report_fingerprint
    metadata["audit_context"]["first_key"].append(3)

    assert first.report_fingerprint == second.report_fingerprint == original
    assert first.report_handle == f"essay_score_report_{original[:32]}"
    assert first.human_review_required is False
    assert first.review_trigger_ids == ()
    assert first.scored_criterion_ids == ("claim_support", "source_alignment")
    assert first.terminal_criterion_ids == ()
    assert isinstance(first.metadata, MappingProxyType)
    payload = first.to_dict()
    assert payload["human_review_required"] is False
    assert payload["essay_request"]["request_fingerprint"] == request.request_fingerprint
    assert payload["scoring_result"]["result_fingerprint"] == result.result_fingerprint
    assert payload["engine_descriptor"]["engine_fingerprint"] == (
        descriptor.engine_fingerprint
    )


def test_report_derives_non_suppressible_review_triggers() -> None:
    """Submission flags, terminal outcomes, and missing evidence remain visible."""
    request = essay_request(
        review_flags=(
            EssayReviewFlag.OFF_TOPIC_RESPONSE,
            EssayReviewFlag.LOW_EVIDENCE_COVERAGE,
        )
    )
    _engine, descriptor, result = result_bundle(
        request,
        claim_evidence=(),
        alignment_status=ObservationStatus.ABSTAINED,
        alignment_reason="insufficient_evidence",
    )
    report = build_essay_score_report(
        report_id="review_required_report",
        request=request,
        result=result,
        engine=descriptor,
        additional_review_trigger_ids=(
            "scorer_disagreement",
            "submission_off_topic_response",
        ),
    )

    assert report.human_review_required is True
    assert report.review_trigger_ids == (
        "observation_abstained_insufficient_evidence",
        "observation_missing_evidence",
        "scorer_disagreement",
        "submission_low_evidence_coverage",
        "submission_off_topic_response",
    )
    assert report.scored_criterion_ids == ("claim_support",)
    assert report.terminal_criterion_ids == ("source_alignment",)
    assert report.to_dict()["review_trigger_ids"] == list(report.review_trigger_ids)


def test_report_rejects_direct_and_untyped_construction() -> None:
    """Factory sealing and all provider-boundary type checks fail closed."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    assert_error(
        "unverified_essay_score_report",
        lambda: EssayScoreReport(
            report_id="essay_score_report",
            essay_request=request,
            engine_descriptor=descriptor,
            scoring_result=result,
            review_trigger_ids=(),
            metadata={},
        ),
    )
    assert_error(
        "invalid_essay_request",
        lambda: build_essay_score_report(
            report_id="essay_score_report",
            request=object(),  # type: ignore[arg-type]
            result=result,
            engine=descriptor,
        ),
    )
    assert_error(
        "invalid_scoring_result",
        lambda: build_essay_score_report(
            report_id="essay_score_report",
            request=request,
            result=object(),  # type: ignore[arg-type]
            engine=descriptor,
        ),
    )
    assert_error(
        "invalid_engine_descriptor",
        lambda: build_essay_score_report(
            report_id="essay_score_report",
            request=request,
            result=result,
            engine=object(),  # type: ignore[arg-type]
        ),
    )


@pytest.mark.parametrize(
    ("scope", "field_name", "replacement", "error_code"),
    (
        ("result", "request_fingerprint", "4" * 64, "essay_report_request_mismatch"),
        ("result", "engine_fingerprint", "5" * 64, "essay_report_engine_mismatch"),
        (
            "result",
            "granularity",
            ObservationGranularity.HOLISTIC,
            "essay_report_granularity_mismatch",
        ),
        (
            "result",
            "requested_criterion_ids",
            ("claim_support",),
            "essay_report_criteria_mismatch",
        ),
        (
            "observation",
            "request_fingerprint",
            "6" * 64,
            "essay_report_observation_request_mismatch",
        ),
        (
            "observation",
            "engine_fingerprint",
            "7" * 64,
            "essay_report_observation_engine_mismatch",
        ),
        (
            "observation",
            "assessment_fingerprint",
            "8" * 64,
            "essay_report_observation_assessment_mismatch",
        ),
        (
            "observation",
            "rubric_fingerprint",
            "9" * 64,
            "essay_report_observation_rubric_mismatch",
        ),
        (
            "observation",
            "construct_id",
            "alternate_construct",
            "essay_report_observation_construct_mismatch",
        ),
        (
            "observation",
            "granularity",
            ObservationGranularity.HOLISTIC,
            "essay_report_observation_granularity_mismatch",
        ),
    ),
)
def test_report_replays_result_and_observation_provenance(
    scope: str,
    field_name: str,
    replacement: Any,
    error_code: str,
) -> None:
    """Post-construction provenance mutation fails before report emission."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    target = result if scope == "result" else result.observations[0]
    object.__setattr__(target, field_name, replacement)

    assert_error(
        error_code,
        lambda: build_essay_score_report(
            report_id="essay_score_report",
            request=request,
            result=result,
            engine=descriptor,
        ),
    )


def test_report_rejects_terminal_observation_without_reason_after_mutation() -> None:
    """A terminal observation cannot enter review routing without a reason."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(
        request,
        alignment_status=ObservationStatus.ABSTAINED,
        alignment_reason="insufficient_evidence",
    )
    object.__setattr__(result.observations[1], "reason_code", None)

    assert_error(
        "essay_report_missing_reason_code",
        lambda: build_essay_score_report(
            report_id="essay_score_report",
            request=request,
            result=result,
            engine=descriptor,
        ),
    )


def test_report_rejects_duplicate_triggers_and_sensitive_metadata() -> None:
    """Trigger and metadata normalization retain deterministic audit safety."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    with pytest.raises(AssessmentSpecError):
        build_essay_score_report(
            report_id="essay_score_report",
            request=request,
            result=result,
            engine=descriptor,
            additional_review_trigger_ids=(
                "scorer_disagreement",
                "scorer_disagreement",
            ),
        )
    assert_error(
        "sensitive_metadata_field",
        lambda: build_essay_score_report(
            report_id="essay_score_report",
            request=request,
            result=result,
            engine=descriptor,
            metadata={"response_text": "secret"},
        ),
    )
