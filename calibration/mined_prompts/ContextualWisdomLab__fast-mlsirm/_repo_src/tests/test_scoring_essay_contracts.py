"""Contract tests for essay adapters over the governed scoring boundary."""

from __future__ import annotations

from pathlib import Path
import runpy
from types import MappingProxyType
from typing import Any

import pytest

import fast_mlsirm.scoring.essay as essay
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    EvidenceRole,
    FixtureOutcome,
    ObservationStatus,
    ScoringEngine,
    StaticFixtureEngine,
)
from fast_mlsirm.scoring.essay import (
    EssayEvidenceKind,
    EssayPrompt,
    EssayResponseEvidence,
    EssayReviewFlag,
    EssayScoringRequest,
    EssaySubmission,
    build_essay_prompt,
    build_essay_response_evidence,
    build_essay_scoring_request,
    build_essay_submission,
    score_essay_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
rubric = _FIXTURES["rubric"]


def prompt(**overrides: Any) -> EssayPrompt:
    """Return one deterministic essay prompt adapter."""
    values: dict[str, Any] = {
        "prompt_id": "argument_prompt",
        "task_family_id": "essay_review",
        "prompt_content_fingerprint": "1" * 64,
        "language_id": "english_language",
        "genre_id": "argument_genre",
        "maximum_response_characters": 5_000,
        "maximum_response_units": 1_000,
        "metadata": {"administration_stage": "pilot"},
    }
    values.update(overrides)
    return build_essay_prompt(**values)


def submission(
    prompt_value: EssayPrompt | None = None,
    **overrides: Any,
) -> EssaySubmission:
    """Return one deterministic essay submission adapter."""
    values: dict[str, Any] = {
        "submission_id": "essay_submission",
        "prompt": prompt_value or prompt(),
        "respondent_id": "sample_respondent",
        "response_id": "essay_response",
        "response_content_fingerprint": "2" * 64,
        "response_character_count": 800,
        "response_unit_count": 120,
        "review_flags": (),
        "metadata": {"administration_mode": "offline"},
    }
    values.update(overrides)
    return build_essay_submission(**values)


def shared_evidence(
    *,
    source_id: str = "essay_response",
    span_id: str = "response_span",
    role: EvidenceRole = EvidenceRole.SUPPORTING,
    content: str = "3",
) -> EvidenceReference:
    """Return one source-text-free shared evidence reference."""
    return EvidenceReference(
        source_id=source_id,
        span_id=span_id,
        content_fingerprint=content * 64,
        evidence_role=role,
    )


def essay_evidence(
    prompt_value: EssayPrompt | None = None,
    submission_value: EssaySubmission | None = None,
    **overrides: Any,
) -> EssayResponseEvidence:
    """Return one response-span essay evidence adapter."""
    prompt_record = prompt_value or prompt()
    submission_record = submission_value or submission(prompt_record)
    values: dict[str, Any] = {
        "prompt": prompt_record,
        "submission": submission_record,
        "evidence_reference": shared_evidence(source_id=submission_record.response_id),
        "evidence_kind": EssayEvidenceKind.RESPONSE_SPAN,
        "start_offset": 10,
        "end_offset": 30,
        "metadata": {"extractor_id": "manual_review"},
    }
    values.update(overrides)
    return build_essay_response_evidence(**values)


def essay_request(**overrides: Any) -> EssayScoringRequest:
    """Return one criterion-level essay scoring adapter."""
    prompt_record = overrides.pop("prompt", prompt())
    submission_record = overrides.pop("submission", submission(prompt_record))
    evidence_record = overrides.pop(
        "evidence_record",
        essay_evidence(prompt_record, submission_record),
    )
    values: dict[str, Any] = {
        "request_id": "essay_scoring_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "prompt": prompt_record,
        "submission": submission_record,
        "occasion_id": "initial_occasion",
        "criterion_ids": ("claim_support", "source_alignment"),
        "essay_evidence": (evidence_record,),
        "metadata": {"workflow_stage": "pilot_review"},
    }
    values.update(overrides)
    return build_essay_scoring_request(**values)


def fixture_engine(request: EssayScoringRequest) -> StaticFixtureEngine:
    """Return a deterministic engine covering the essay request criteria."""
    evidence_reference = request.essay_evidence[0].evidence_reference
    return StaticFixtureEngine(
        descriptor=automated_engine(),
        outcomes=(
            FixtureOutcome(
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
                evidence_references=(evidence_reference,),
            ),
            FixtureOutcome(
                criterion_id="source_alignment",
                status=ObservationStatus.ABSTAINED,
                reason_code="insufficient_evidence",
                evidence_references=(evidence_reference,),
            ),
        ),
    )


def assert_error(code: str, callback) -> None:
    """Assert one stable essay/scoring contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_public_surface_is_explicit_and_documented() -> None:
    """The essay namespace exports only the reviewed adapter surface."""
    expected = {
        "EssayEvidenceKind",
        "EssayFacetsCalibrationReport",
        "EssayPrompt",
        "EssayResponseEvidence",
        "EssayReviewFlag",
        "EssayScoringRequest",
        "EssayScoreReport",
        "EssaySubmission",
        "EssayValidationEvidenceReport",
        "EssayValidationMetric",
        "MAX_ESSAY_EVIDENCE_REFERENCES",
        "MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS",
        "MAX_ESSAY_REPORT_REVIEW_TRIGGERS",
        "MAX_ESSAY_RESPONSE_CHARACTERS",
        "MAX_ESSAY_RESPONSE_UNITS",
        "MAX_ESSAY_REVIEW_FLAGS",
        "MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS",
        "build_essay_facets_calibration_report",
        "build_essay_prompt",
        "build_essay_response_evidence",
        "build_essay_score_report",
        "build_essay_scoring_request",
        "build_essay_submission",
        "build_essay_validation_evidence_report",
        "fit_essay_facets_calibration_report",
        "render_essay_facets_calibration_report_html",
        "render_essay_score_report_html",
        "render_essay_validation_evidence_report_html",
        "score_essay_request",
    }
    assert set(essay.__all__) == expected
    assert all(getattr(essay, name).__doc__ for name in expected if name[0].isupper())


def test_prompt_is_content_addressed_and_deeply_immutable() -> None:
    """Equivalent metadata ordering yields one immutable prompt identity."""
    metadata = {"nested_value": {"second_key": 2, "first_key": [1, 2]}}
    first = prompt(metadata=metadata)
    second = prompt(metadata={"nested_value": {"first_key": [1, 2], "second_key": 2}})
    original = first.prompt_fingerprint
    metadata["nested_value"]["first_key"].append(3)

    assert first.prompt_fingerprint == second.prompt_fingerprint == original
    assert isinstance(first.metadata, MappingProxyType)
    assert first.prompt_handle == f"essay_prompt_{original[:32]}"
    assert first.to_dict()["prompt_fingerprint"] == original


def test_prompt_rejects_unverified_invalid_and_empty_limits() -> None:
    """Prompt construction fails closed on direct or malformed inputs."""
    assert_error(
        "unverified_essay_prompt",
        lambda: EssayPrompt(
            prompt_id="argument_prompt",
            task_family_id="essay_review",
            prompt_content_fingerprint="1" * 64,
            language_id="english_language",
            genre_id="argument_genre",
            maximum_response_characters=5_000,
            maximum_response_units=1_000,
            metadata={},
        ),
    )
    assert_error("empty_response_limit", lambda: prompt(maximum_response_units=0))
    assert_error(
        "invalid_maximum_response_characters",
        lambda: prompt(maximum_response_characters=True),
    )
    assert_error(
        "invalid_maximum_response_units",
        lambda: prompt(maximum_response_units=1_000_001),
    )


def test_submission_normalizes_flags_and_preserves_content_boundary() -> None:
    """Review flags are deterministic and raw response metadata is rejected."""
    prompt_record = prompt()
    first = submission(
        prompt_record,
        review_flags=(
            EssayReviewFlag.PROMPT_COPYING_RISK,
            EssayReviewFlag.OFF_TOPIC_RESPONSE,
        ),
    )
    second = submission(
        prompt_record,
        review_flags=("off_topic_response", "prompt_copying_risk"),
    )
    assert first.review_flags == (
        EssayReviewFlag.OFF_TOPIC_RESPONSE,
        EssayReviewFlag.PROMPT_COPYING_RISK,
    )
    assert first.submission_fingerprint == second.submission_fingerprint
    assert first.submission_handle.startswith("essay_submission_")
    assert first.to_dict()["response_content_fingerprint"] == "2" * 64
    assert_error(
        "sensitive_metadata_field",
        lambda: submission(prompt_record, metadata={"response_text": "secret"}),
    )


def test_submission_rejects_limits_duplicates_and_unverified_values() -> None:
    """Prompt limits, duplicate signals, and direct construction fail closed."""
    prompt_record = prompt(maximum_response_characters=100, maximum_response_units=10)
    assert_error(
        "response_character_limit_exceeded",
        lambda: submission(prompt_record, response_character_count=101),
    )
    assert_error(
        "response_unit_limit_exceeded",
        lambda: submission(
            prompt_record,
            response_character_count=80,
            response_unit_count=11,
        ),
    )
    assert_error(
        "duplicate_review_flag",
        lambda: submission(
            prompt_record,
            response_character_count=80,
            response_unit_count=8,
            review_flags=("off_topic_response", "off_topic_response"),
        ),
    )
    assert_error(
        "invalid_review_flag",
        lambda: submission(
            prompt_record,
            response_character_count=80,
            response_unit_count=8,
            review_flags=("unsupported_flag",),
        ),
    )
    assert_error(
        "unverified_essay_submission",
        lambda: EssaySubmission(
            submission_id="essay_submission",
            respondent_id="sample_respondent",
            response_id="essay_response",
            prompt_fingerprint=prompt_record.prompt_fingerprint,
            response_content_fingerprint="2" * 64,
            response_character_count=80,
            response_unit_count=8,
            review_flags=(),
            metadata={},
        ),
    )


def test_response_and_prompt_evidence_preserve_exact_source_identity() -> None:
    """Evidence spans bind exact prompt/submission identities and offsets."""
    prompt_record = prompt()
    submission_record = submission(prompt_record)
    response_span = essay_evidence(prompt_record, submission_record)
    prompt_span = essay_evidence(
        prompt_record,
        submission_record,
        evidence_reference=shared_evidence(
            source_id=prompt_record.prompt_id,
            span_id="prompt_span",
            content="4",
        ),
        evidence_kind=EssayEvidenceKind.PROMPT_SPAN,
        start_offset=0,
        end_offset=20,
    )
    external_span = essay_evidence(
        prompt_record,
        submission_record,
        evidence_reference=shared_evidence(
            source_id="external_source",
            span_id="external_span",
            role=EvidenceRole.COUNTER,
            content="5",
        ),
        evidence_kind=EssayEvidenceKind.EXTERNAL_SOURCE_SPAN,
        start_offset=5,
        end_offset=15,
    )

    assert response_span.evidence_reference.source_id == submission_record.response_id
    assert prompt_span.evidence_reference.source_id == prompt_record.prompt_id
    assert external_span.evidence_reference.evidence_role is EvidenceRole.COUNTER
    assert response_span.evidence_handle.startswith("essay_evidence_")
    assert response_span.to_dict()["submission_fingerprint"] == (
        submission_record.submission_fingerprint
    )


def test_evidence_rejects_mismatches_offsets_and_unverified_values() -> None:
    """Evidence adapters reject replay, source mismatch, and invalid ranges."""
    prompt_record = prompt()
    submission_record = submission(prompt_record)
    other_prompt = prompt(prompt_content_fingerprint="9" * 64)
    assert_error(
        "submission_prompt_mismatch",
        lambda: build_essay_response_evidence(
            prompt=other_prompt,
            submission=submission_record,
            evidence_reference=shared_evidence(),
            evidence_kind=EssayEvidenceKind.EXTERNAL_SOURCE_SPAN,
            start_offset=0,
            end_offset=1,
        ),
    )
    assert_error(
        "response_evidence_source_mismatch",
        lambda: essay_evidence(
            prompt_record,
            submission_record,
            evidence_reference=shared_evidence(source_id="other_response"),
        ),
    )
    assert_error(
        "prompt_evidence_source_mismatch",
        lambda: essay_evidence(
            prompt_record,
            submission_record,
            evidence_reference=shared_evidence(source_id="other_prompt"),
            evidence_kind=EssayEvidenceKind.PROMPT_SPAN,
        ),
    )
    assert_error(
        "invalid_evidence_offsets",
        lambda: essay_evidence(
            prompt_record,
            submission_record,
            start_offset=30,
            end_offset=30,
        ),
    )
    assert_error(
        "invalid_end_offset",
        lambda: essay_evidence(
            prompt_record,
            submission_record,
            start_offset=10,
            end_offset=801,
        ),
    )
    assert_error(
        "unverified_essay_evidence",
        lambda: EssayResponseEvidence(
            evidence_reference=shared_evidence(),
            prompt_fingerprint=prompt_record.prompt_fingerprint,
            submission_fingerprint=submission_record.submission_fingerprint,
            evidence_kind=EssayEvidenceKind.EXTERNAL_SOURCE_SPAN,
            start_offset=0,
            end_offset=1,
            metadata={},
        ),
    )


def test_essay_request_compiles_to_authoritative_shared_request() -> None:
    """The adapter delegates all score semantics to the shared request type."""
    prompt_record = prompt()
    submission_record = submission(
        prompt_record,
        review_flags=(EssayReviewFlag.LOW_EVIDENCE_COVERAGE,),
    )
    evidence_record = essay_evidence(prompt_record, submission_record)
    request = essay_request(
        prompt=prompt_record,
        submission=submission_record,
        evidence_record=evidence_record,
    )
    shared = request.scoring_request

    assert shared.task_id == prompt_record.prompt_id
    assert shared.task_family_id == prompt_record.task_family_id
    assert shared.respondent_id == submission_record.respondent_id
    assert shared.response_id == submission_record.response_id
    assert shared.response_content_fingerprint == (
        submission_record.response_content_fingerprint
    )
    assert shared.metadata["essay_prompt_fingerprint"] == prompt_record.prompt_fingerprint
    assert shared.metadata["essay_review_flags"] == ("low_evidence_coverage",)
    assert request.essay_evidence == (evidence_record,)
    assert request.request_handle.startswith("essay_request_")
    assert request.to_dict()["scoring_request"]["request_fingerprint"] == (
        shared.request_fingerprint
    )


def test_essay_request_rejects_replay_unknown_family_and_duplicates() -> None:
    """Cross-layer identity and duplicate evidence failures are explicit."""
    prompt_record = prompt()
    submission_record = submission(prompt_record)
    evidence_record = essay_evidence(prompt_record, submission_record)
    other_prompt = prompt(prompt_content_fingerprint="8" * 64)
    other_submission = submission(other_prompt)
    other_evidence = essay_evidence(other_prompt, other_submission)
    second_submission = submission(prompt_record, response_id="other_response")
    second_evidence = essay_evidence(prompt_record, second_submission)

    assert_error(
        "submission_prompt_mismatch",
        lambda: essay_request(prompt=other_prompt, submission=submission_record),
    )
    assert_error(
        "unknown_prompt_task_family",
        lambda: essay_request(prompt=prompt(task_family_id="unknown_family")),
    )
    assert_error(
        "duplicate_essay_evidence",
        lambda: essay_request(
            prompt=prompt_record,
            submission=submission_record,
            evidence_record=evidence_record,
            essay_evidence=(evidence_record, evidence_record),
        ),
    )
    assert_error(
        "essay_evidence_prompt_mismatch",
        lambda: essay_request(
            prompt=prompt_record,
            submission=submission_record,
            evidence_record=evidence_record,
            essay_evidence=(other_evidence,),
        ),
    )
    assert_error(
        "essay_evidence_submission_mismatch",
        lambda: essay_request(
            prompt=prompt_record,
            submission=submission_record,
            evidence_record=evidence_record,
            essay_evidence=(second_evidence,),
        ),
    )


def test_direct_essay_request_construction_is_rejected() -> None:
    """Callers cannot forge a wrapper around a shared scoring request."""
    valid = essay_request()
    assert_error(
        "unverified_essay_request",
        lambda: EssayScoringRequest(
            scoring_request=valid.scoring_request,
            prompt_fingerprint=valid.prompt_fingerprint,
            submission_fingerprint=valid.submission_fingerprint,
            essay_evidence=valid.essay_evidence,
        ),
    )


def test_static_fixture_engine_executes_through_shared_protocol() -> None:
    """Deterministic essay fixtures return the ordinary governed result type."""
    request = essay_request()
    engine = fixture_engine(request)
    assert isinstance(engine, ScoringEngine)
    result = score_essay_request(engine, request)

    assert result.request_fingerprint == request.scoring_request.request_fingerprint
    assert result.engine_fingerprint == engine.descriptor.engine_fingerprint
    assert tuple(value.status for value in result.observations) == (
        ObservationStatus.SCORED,
        ObservationStatus.ABSTAINED,
    )


def test_score_adapter_rejects_invalid_engine_request_and_result_contracts() -> None:
    """The execution wrapper fails closed on malformed provider boundaries."""
    request = essay_request()
    assert_error("invalid_essay_request", lambda: score_essay_request(object(), object()))
    assert_error("invalid_scoring_engine", lambda: score_essay_request(object(), request))

    class InvalidResultEngine:
        @property
        def descriptor(self):
            return automated_engine()

        def score(self, scoring_request):
            return object()

    assert isinstance(InvalidResultEngine(), ScoringEngine)
    assert_error(
        "invalid_scoring_result",
        lambda: score_essay_request(InvalidResultEngine(), request),
    )

    other_prompt = prompt(prompt_content_fingerprint="7" * 64)
    other_submission = submission(other_prompt, response_id="other_response")
    other_request = essay_request(
        request_id="other_scoring_request",
        prompt=other_prompt,
        submission=other_submission,
    )
    other_engine = fixture_engine(other_request)

    class WrongRequestEngine:
        @property
        def descriptor(self):
            return other_engine.descriptor

        def score(self, scoring_request):
            return other_engine.score(other_request.scoring_request)

    assert_error(
        "essay_result_request_mismatch",
        lambda: score_essay_request(WrongRequestEngine(), request),
    )

    result_engine = fixture_engine(request)
    returned_result = result_engine.score(request.scoring_request)

    class WrongDescriptorEngine:
        @property
        def descriptor(self):
            return automated_engine(engine_id="alternate_engine")

        def score(self, scoring_request):
            return returned_result

    assert_error(
        "essay_result_engine_mismatch",
        lambda: score_essay_request(WrongDescriptorEngine(), request),
    )

    class InvalidDescriptorEngine:
        @property
        def descriptor(self):
            return object()

        def score(self, scoring_request):
            return returned_result

    assert_error(
        "invalid_engine_descriptor",
        lambda: score_essay_request(InvalidDescriptorEngine(), request),
    )


def test_reachable_type_and_offset_guards_fail_closed() -> None:
    """Every public builder rejects untyped inputs and impossible spans."""
    bound_prompt = prompt()
    bound_submission = submission(bound_prompt)

    assert_error(
        "invalid_response_character_count",
        lambda: submission(bound_prompt, response_character_count="800"),
    )
    assert_error(
        "invalid_evidence_offsets",
        lambda: essay_evidence(
            prompt_value=bound_prompt,
            submission_value=bound_submission,
            start_offset=10,
            end_offset=10,
        ),
    )
    assert_error(
        "invalid_essay_prompt",
        lambda: build_essay_submission(
            submission_id="essay_submission",
            prompt=object(),  # type: ignore[arg-type]
            respondent_id="sample_respondent",
            response_id="essay_response",
            response_content_fingerprint="2" * 64,
            response_character_count=800,
            response_unit_count=120,
        ),
    )
    assert_error(
        "invalid_essay_prompt",
        lambda: build_essay_response_evidence(
            prompt=object(),  # type: ignore[arg-type]
            submission=bound_submission,
            evidence_reference=shared_evidence(),
            evidence_kind=EssayEvidenceKind.RESPONSE_SPAN,
            start_offset=0,
            end_offset=10,
        ),
    )
    assert_error(
        "invalid_essay_submission",
        lambda: build_essay_response_evidence(
            prompt=bound_prompt,
            submission=object(),  # type: ignore[arg-type]
            evidence_reference=shared_evidence(),
            evidence_kind=EssayEvidenceKind.RESPONSE_SPAN,
            start_offset=0,
            end_offset=10,
        ),
    )
    assert_error(
        "invalid_evidence_reference",
        lambda: build_essay_response_evidence(
            prompt=bound_prompt,
            submission=bound_submission,
            evidence_reference=object(),  # type: ignore[arg-type]
            evidence_kind=EssayEvidenceKind.RESPONSE_SPAN,
            start_offset=0,
            end_offset=10,
        ),
    )


def test_request_compile_guards_reject_untyped_and_unbound_provenance() -> None:
    """Request compilation rejects untyped values and cross-prompt submissions."""
    bound_prompt = prompt()
    bound_submission = submission(bound_prompt)
    other_prompt = prompt(prompt_id="alternate_prompt")

    def compile_request(**overrides):
        values: dict[str, Any] = {
            "request_id": "essay_scoring_request",
            "assessment": assessment(),
            "rubric": rubric(),
            "prompt": bound_prompt,
            "submission": bound_submission,
            "occasion_id": "initial_occasion",
            "criterion_ids": ("claim_support", "source_alignment"),
        }
        values.update(overrides)
        return build_essay_scoring_request(**values)

    assert_error("invalid_essay_prompt", lambda: compile_request(prompt=object()))
    assert_error(
        "invalid_essay_submission", lambda: compile_request(submission=object())
    )
    assert_error(
        "submission_prompt_mismatch",
        lambda: compile_request(submission=submission(other_prompt)),
    )
    assert_error("invalid_rubric", lambda: compile_request(rubric=object()))
    assert_error(
        "invalid_essay_evidence",
        lambda: compile_request(essay_evidence=(object(),)),
    )
