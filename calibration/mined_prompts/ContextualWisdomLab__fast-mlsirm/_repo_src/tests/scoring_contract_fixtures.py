"""Shared deterministic fixtures for assessment-contract tests."""

from __future__ import annotations

from typing import Any

from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification
from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    AssessmentResponseType,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
    build_assessment_spec,
)


def rubric(
    rubric_id: str,
    construct_id: str,
    *,
    rubric_version: str = "1.0.0",
) -> RubricSpecification:
    """Return one deterministic ordinal rubric."""
    return RubricSpecification(
        rubric_id=rubric_id,
        construct_id=construct_id,
        construct_definition=f"Definition for {construct_id}.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(
                score=0,
                label="not_demonstrated",
                descriptor="The construct is not demonstrated.",
                observable_indicators=("Required evidence is absent.",),
            ),
            RubricLevel(
                score=1,
                label="partially_demonstrated",
                descriptor="The construct is partially demonstrated.",
                observable_indicators=("Some required evidence is present.",),
            ),
            RubricLevel(
                score=2,
                label="fully_demonstrated",
                descriptor="The construct is fully demonstrated.",
                observable_indicators=("All required evidence is present.",),
            ),
        ),
        task_families=("analytic_response",),
        evidence_requirements=("Cite the exact response evidence used.",),
        rubric_version=rubric_version,
    )


def policies(
    construct_ids: tuple[str, ...] = ("argument_quality", "evidence_use"),
) -> tuple[
    EnginePolicy,
    CalibrationPolicy,
    ValidationPolicy,
    AdjudicationPolicy,
    MonitoringPolicy,
    ReportingPolicy,
]:
    """Return one complete deterministic scoring-policy family."""
    return (
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=("fixture_engine", "human_adapter"),
            allow_human_raters=True,
            allow_automated_raters=True,
            minimum_raters_per_response=2,
        ),
        CalibrationPolicy(
            policy_id="calibration_policy",
            model_id="facets_ordinal",
            construct_ids=construct_ids,
        ),
        ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=("exact_agreement", "quadratic_weighted_kappa"),
            construct_ids=construct_ids,
        ),
        AdjudicationPolicy(
            policy_id="adjudication_policy",
            trigger_ids=("scorer_disagreement", "insufficient_evidence"),
            construct_ids=construct_ids,
        ),
        MonitoringPolicy(
            policy_id="monitoring_policy",
            metric_ids=("severity_drift", "failure_rate_drift"),
            construct_ids=construct_ids,
        ),
        ReportingPolicy(
            policy_id="reporting_policy",
            format_ids=("json_report", "html_report"),
            construct_ids=construct_ids,
            include_exact_values=True,
        ),
    )


def assessment(
    *,
    constructs: tuple[ConstructSpec, ...] | None = None,
    rubrics: tuple[RubricSpecification, ...] | None = None,
    selected_policies: tuple[
        EnginePolicy,
        CalibrationPolicy,
        ValidationPolicy,
        AdjudicationPolicy,
        MonitoringPolicy,
        ReportingPolicy,
    ]
    | None = None,
    metadata: Any | None = None,
    response_type: AssessmentResponseType | str = AssessmentResponseType.CRITERION_LEVEL,
):
    """Build one complete assessment with two exact rubric bindings."""
    argument_rubric = rubric("argument_rubric", "argument_quality")
    evidence_rubric = rubric("evidence_rubric", "evidence_use")
    selected_rubrics = (
        (argument_rubric, evidence_rubric) if rubrics is None else rubrics
    )
    selected_constructs = (
        (
            ConstructSpec(
                construct_id="argument_quality",
                construct_definition="Quality of the response argument.",
                rubric_fingerprints=(argument_rubric.fingerprint,),
            ),
            ConstructSpec(
                construct_id="evidence_use",
                construct_definition="Quality of cited supporting evidence.",
                rubric_fingerprints=(evidence_rubric.fingerprint,),
            ),
        )
        if constructs is None
        else constructs
    )
    policy_values = policies() if selected_policies is None else selected_policies
    return build_assessment_spec(
        assessment_id="essay_assessment",
        assessment_version="1.0.0",
        constructs=selected_constructs,
        rubrics=selected_rubrics,
        response_type=response_type,
        engine_policy=policy_values[0],
        calibration_policy=policy_values[1],
        validation_policy=policy_values[2],
        adjudication_policy=policy_values[3],
        monitoring_policy=policy_values[4],
        reporting_policy=policy_values[5],
        metadata=(
            {
                "study_name": "Connected sparse pilot",
                "nested_metadata": {
                    "fold_count": 5,
                    "threshold_values": [0.1, 0.2],
                },
                "enabled_flag": True,
                "optional_value": None,
            }
            if metadata is None
            else metadata
        ),
    )
