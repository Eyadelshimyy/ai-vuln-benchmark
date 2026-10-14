"""Reusable fixtures for scoring execution-contract tests."""

from __future__ import annotations

from typing import Any

from fast_mlsirm.rubric import (
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
)
from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    AssessmentResponseType,
    CalibrationPolicy,
    ConstructSpec,
    EngineKind,
    EnginePolicy,
    EvidenceReference,
    EvidenceRole,
    FixtureOutcome,
    MonitoringPolicy,
    ObservationGranularity,
    ObservationStatus,
    ReportingPolicy,
    StaticFixtureEngine,
    ValidationPolicy,
    build_assessment_spec,
    build_engine_descriptor,
    build_scoring_request,
)


def rubric() -> RubricSpecification:
    """Return one deterministic three-level rubric."""
    return RubricSpecification(
        rubric_id="evidence_quality_rubric",
        construct_id="evidence_quality",
        construct_definition="Evidence-conditioned response quality.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(
                score=0,
                label="not_supported",
                descriptor="Material claims are unsupported.",
                observable_indicators=("material claim lacks evidence",),
            ),
            RubricLevel(
                score=1,
                label="partly_supported",
                descriptor="Some material claims are supported.",
                observable_indicators=("some claims map to evidence",),
            ),
            RubricLevel(
                score=2,
                label="fully_supported",
                descriptor="Every material claim is supported.",
                observable_indicators=("all claims map to evidence",),
            ),
        ),
        task_families=("evidence_review", "essay_review"),
        evidence_requirements=("retain exact evidence provenance",),
    )


def assessment(response_type: AssessmentResponseType = AssessmentResponseType.MIXED):
    """Return one assessment that binds the fixture rubric."""
    rubric_value = rubric()
    construct = ConstructSpec(
        construct_id="evidence_quality",
        construct_definition="Evidence-conditioned response quality.",
        rubric_fingerprints=(rubric_value.fingerprint,),
    )
    construct_ids = (construct.construct_id,)
    return build_assessment_spec(
        assessment_id="evidence_assessment",
        assessment_version="1.0.0",
        constructs=(construct,),
        rubrics=(rubric_value,),
        response_type=response_type,
        engine_policy=EnginePolicy(
            policy_id="engine_policy",
            engine_ids=("alternate_engine", "fixture_engine"),
            allow_human_raters=True,
            allow_automated_raters=True,
            minimum_raters_per_response=1,
        ),
        calibration_policy=CalibrationPolicy(
            policy_id="calibration_policy",
            model_id="facets_ordinal",
            construct_ids=construct_ids,
        ),
        validation_policy=ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=("exact_agreement",),
            construct_ids=construct_ids,
        ),
        adjudication_policy=AdjudicationPolicy(
            policy_id="adjudication_policy",
            trigger_ids=("scorer_disagreement",),
            construct_ids=construct_ids,
        ),
        monitoring_policy=MonitoringPolicy(
            policy_id="monitoring_policy",
            metric_ids=("severity_drift",),
            construct_ids=construct_ids,
        ),
        reporting_policy=ReportingPolicy(
            policy_id="reporting_policy",
            format_ids=("json_report",),
            construct_ids=construct_ids,
            include_exact_values=True,
        ),
        metadata={"deployment_stage": "pilot"},
    )


def automated_engine(**overrides: Any):
    """Return one prompt-driven automated engine descriptor."""
    values: dict[str, Any] = {
        "engine_id": "fixture_engine",
        "engine_family_id": "fixture_family",
        "provider_id": "local_provider",
        "engine_version": "1.0.0",
        "engine_kind": EngineKind.AUTOMATED,
        "model_id": "fixture_model",
        "prompt_driven": True,
        "prompt_template_fingerprint": "a" * 64,
        "metadata": {"deterministic_mode": True},
    }
    values.update(overrides)
    return build_engine_descriptor(**values)


def human_engine(**overrides: Any):
    """Return one human-rater engine descriptor."""
    values: dict[str, Any] = {
        "engine_id": "human_engine",
        "engine_family_id": "human_family",
        "provider_id": "review_organization",
        "engine_version": "1.0.0",
        "engine_kind": EngineKind.HUMAN,
        "model_id": None,
        "prompt_driven": False,
        "prompt_template_fingerprint": None,
        "metadata": {"training_complete": True},
    }
    values.update(overrides)
    return build_engine_descriptor(**values)


def evidence(
    source_id: str = "source_document",
    span_id: str = "evidence_span",
    role: EvidenceRole = EvidenceRole.SUPPORTING,
):
    """Return one content-addressed evidence reference."""
    return EvidenceReference(
        source_id=source_id,
        span_id=span_id,
        content_fingerprint="b" * 64,
        evidence_role=role,
    )


def criterion_request(**overrides: Any):
    """Return one two-criterion request bound to the fixture assessment."""
    values: dict[str, Any] = {
        "request_id": "scoring_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "granularity": ObservationGranularity.CRITERION_LEVEL,
        "respondent_id": "sample_respondent",
        "response_id": "sample_response",
        "task_id": "sample_task",
        "task_revision_fingerprint": "d" * 64,
        "task_family_id": "evidence_review",
        "occasion_id": "initial_occasion",
        "criterion_ids": ("claim_support", "source_alignment"),
        "response_content_fingerprint": "c" * 64,
        "response_character_count": 128,
        "response_unit_count": 8,
        "metadata": {"language_code": "en"},
    }
    values.update(overrides)
    return build_scoring_request(**values)


def holistic_request(**overrides: Any):
    """Return one holistic request bound to the fixture assessment."""
    values: dict[str, Any] = {
        "request_id": "holistic_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "granularity": ObservationGranularity.HOLISTIC,
        "respondent_id": "sample_respondent",
        "response_id": "sample_response",
        "task_id": "sample_task",
        "task_revision_fingerprint": "d" * 64,
        "task_family_id": "evidence_review",
        "occasion_id": "initial_occasion",
        "criterion_ids": (),
        "response_content_fingerprint": "c" * 64,
        "response_character_count": 128,
        "response_unit_count": 8,
        "metadata": {},
    }
    values.update(overrides)
    return build_scoring_request(**values)


def fixture_engine() -> StaticFixtureEngine:
    """Return an offline engine with complete criterion-level outcomes."""
    return StaticFixtureEngine(
        descriptor=automated_engine(),
        outcomes=(
            FixtureOutcome(
                criterion_id="source_alignment",
                status=ObservationStatus.SCORED,
                score_category=1,
                evidence_references=(
                    evidence("source_document", "alignment_span"),
                ),
                confidence_metadata={"confidence_value": 0.8},
            ),
            FixtureOutcome(
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
                evidence_references=(
                    evidence("source_document", "support_span"),
                ),
                confidence_metadata={"confidence_value": 0.9},
            ),
        ),
    )
