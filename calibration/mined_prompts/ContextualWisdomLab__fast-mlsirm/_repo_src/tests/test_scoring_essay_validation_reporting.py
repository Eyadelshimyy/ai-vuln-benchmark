"""Tests for governed criterion-specific essay validation evidence."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import MappingProxyType

import numpy as np
import pytest

from fast_mlsirm.rubric import RubricSpecification
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
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.essay import (
    EssayValidationEvidenceReport,
    EssayValidationMetric,
    build_essay_validation_evidence_report,
)
import fast_mlsirm.scoring.essay.validation_reporting as validation_reporting
from fast_mlsirm.validation import ValidationVerdict, validate_judge

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
_fixture_rubric = _FIXTURES["rubric"]
_automated_engine = _FIXTURES["automated_engine"]
_human_engine = _FIXTURES["human_engine"]

_ALL_METRIC_IDS = (
    "adjacent_agreement",
    "exact_agreement",
    "human_machine_degradation",
    "pearson_correlation",
    "quadratic_weighted_kappa",
    "standardized_mean_difference",
    "worst_subgroup_standardized_mean_difference",
)
_BASE_METRIC_IDS = tuple(
    metric_id
    for metric_id in _ALL_METRIC_IDS
    if metric_id
    not in {
        "human_machine_degradation",
        "worst_subgroup_standardized_mean_difference",
    }
)
_AUTOMATED = np.array([0, 1, 2, 2, 1, 0], dtype=np.int64)
_REFERENCE = np.array([0, 1, 2, 1, 1, 0], dtype=np.int64)
_HUMAN_A = np.array([0, 1, 2, 2, 1, 0], dtype=np.int64)
_HUMAN_B = np.array([0, 1, 2, 1, 1, 0], dtype=np.int64)
_SUBGROUP = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)


def assessment(
    *,
    metric_ids: tuple[str, ...] = _ALL_METRIC_IDS,
    validation_construct_ids: tuple[str, ...] = ("evidence_quality",),
    engine_ids: tuple[str, ...] = ("fixture_engine",),
    allow_human: bool = True,
    allow_automated: bool = True,
):
    """Return one exact assessment configured for validation evidence."""
    rubric: RubricSpecification = _fixture_rubric()
    construct = ConstructSpec(
        construct_id="evidence_quality",
        construct_definition="Evidence-conditioned response quality.",
        rubric_fingerprints=(rubric.fingerprint,),
    )
    construct_ids = (construct.construct_id,)
    return build_assessment_spec(
        assessment_id="validation_assessment",
        assessment_version="1.0.0",
        constructs=(construct,),
        rubrics=(rubric,),
        response_type=AssessmentResponseType.CRITERION_LEVEL,
        engine_policy=EnginePolicy(
            policy_id="engine_policy",
            engine_ids=engine_ids,
            allow_human_raters=allow_human,
            allow_automated_raters=allow_automated,
            minimum_raters_per_response=2,
        ),
        calibration_policy=CalibrationPolicy(
            policy_id="calibration_policy",
            model_id="facets_ordinal",
            construct_ids=construct_ids,
        ),
        validation_policy=ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=metric_ids,
            construct_ids=validation_construct_ids,
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
        metadata={"study_stage": "human_anchored_pilot"},
    )


def report_kwargs(*, selected_assessment=None) -> dict[str, object]:
    """Return one complete valid factory argument dictionary."""
    assessment_value = assessment() if selected_assessment is None else selected_assessment
    construct = assessment_value.constructs[0]
    return {
        "report_id": "criterion_validation_report",
        "assessment": assessment_value,
        "construct_id": construct.construct_id,
        "rubric_fingerprint": construct.rubric_fingerprints[0],
        "criterion_id": "claim_support",
        "automated_engine": _automated_engine(),
        "reference_engine": _human_engine(),
        "validation_dataset_fingerprint": "d" * 64,
        "automated_labels": _AUTOMATED,
        "reference_labels": _REFERENCE,
        "category_count": 3,
        "human_human_labels": (_HUMAN_A, _HUMAN_B),
        "subgroup_labels": _SUBGROUP,
        "additional_review_trigger_ids": ("independent_replication_required",),
        "metadata": {"fold_id": "held_out_fold"},
    }


def build_report(**overrides: object) -> EssayValidationEvidenceReport:
    """Build one report while allowing focused field overrides."""
    values = report_kwargs()
    values.update(overrides)
    return build_essay_validation_evidence_report(**values)  # type: ignore[arg-type]


def test_full_report_delegates_to_rust_and_omits_gate_decisions() -> None:
    """The report copies Rust metrics but emits no threshold or pass verdict."""
    report = build_report()
    payload = report.to_dict()
    expected = validate_judge(
        _AUTOMATED,
        _REFERENCE,
        k=3,
        human_human=(_HUMAN_A, _HUMAN_B),
        subgroup=_SUBGROUP,
    )
    expected_values = {
        validation_reporting._METRIC_NAME_MAP[gate["name"]]: gate["value"]
        for gate in expected.gates
    }
    expected_values["exact_agreement"] = expected.exact_agreement
    expected_values["adjacent_agreement"] = expected.adjacent_agreement
    assert set(expected_values) == set(_ALL_METRIC_IDS)
    assert report.metric_ids == _ALL_METRIC_IDS
    assert {metric.metric_id: metric.value for metric in report.metrics} == pytest.approx(
        expected_values
    )
    assert payload["human_review_required"] is True
    assert payload["rust_backend_function_id"] == "mlsirm_core_agreement_validate_scoring"
    assert "human_validation_required" in report.review_trigger_ids
    assert "correlation_descriptive_only" in report.review_trigger_ids
    assert "independent_replication_required" in report.review_trigger_ids
    assert "human_human_baseline_missing" not in report.review_trigger_ids
    assert "subgroup_evidence_missing" not in report.review_trigger_ids
    assert payload["interpretation_boundary_ids"] == list(
        report.interpretation_boundary_ids
    )
    assert report.report_handle.startswith("essay_validation_report_")
    assert report.report_fingerprint == build_report().report_fingerprint
    assert isinstance(report.metadata, MappingProxyType)
    assert "pass" not in payload
    for metric_payload in payload["metrics"]:
        assert "threshold" not in metric_payload
        assert "pass" not in metric_payload
    serialized_payload = json.dumps(payload, sort_keys=True)
    for label_vector in (
        _AUTOMATED,
        _REFERENCE,
        _HUMAN_A,
        _HUMAN_B,
        _SUBGROUP,
    ):
        assert json.dumps(label_vector.tolist()) not in serialized_payload


def test_report_without_optional_comparators_routes_missing_evidence() -> None:
    """Missing baseline and subgroup evidence produce mandatory review triggers."""
    selected = assessment(metric_ids=_BASE_METRIC_IDS)
    report = build_essay_validation_evidence_report(
        **{
            **report_kwargs(selected_assessment=selected),
            "human_human_labels": None,
            "subgroup_labels": None,
            "additional_review_trigger_ids": (),
            "metadata": None,
        }
    )
    assert report.metric_ids == _BASE_METRIC_IDS
    assert "human_human_baseline_missing" in report.review_trigger_ids
    assert "subgroup_evidence_missing" in report.review_trigger_ids
    assert report.to_dict()["metadata"] == {}


def test_empty_validation_policy_scope_covers_the_construct() -> None:
    """An explicitly global validation policy may cover the selected construct."""
    selected = assessment(
        metric_ids=_BASE_METRIC_IDS,
        validation_construct_ids=(),
    )
    report = build_essay_validation_evidence_report(
        **{
            **report_kwargs(selected_assessment=selected),
            "human_human_labels": None,
            "subgroup_labels": None,
        }
    )
    assert report.construct_id == "evidence_quality"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("assessment", object(), "invalid_essay_validation_assessment"),
        ("automated_engine", object(), "invalid_essay_validation_automated_engine"),
        ("reference_engine", object(), "invalid_essay_validation_reference_engine"),
        ("category_count", True, "invalid_essay_validation_category_count"),
        ("category_count", 1, "invalid_essay_validation_category_count"),
        ("category_count", 1_001, "invalid_essay_validation_category_count"),
    ),
)
def test_factory_rejects_wrong_top_level_types(
    field: str,
    value: object,
    code: str,
) -> None:
    """Factory boundaries fail with stable structured errors."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_report(**{field: value})
    assert caught.value.code == code


def test_scope_and_policy_bindings_fail_closed() -> None:
    """Construct, rubric, policy, and metric declarations cannot drift."""
    with pytest.raises(AssessmentSpecError) as construct_error:
        build_report(construct_id="other_construct")
    assert construct_error.value.code == "essay_validation_construct_mismatch"

    with pytest.raises(AssessmentSpecError) as rubric_error:
        build_report(rubric_fingerprint="e" * 64)
    assert rubric_error.value.code == "essay_validation_rubric_mismatch"

    selected = assessment()
    object.__setattr__(
        selected.validation_policy,
        "construct_ids",
        ("other_construct",),
    )
    with pytest.raises(AssessmentSpecError) as scope_error:
        build_report(assessment=selected)
    assert scope_error.value.code == "essay_validation_policy_scope_mismatch"

    with pytest.raises(AssessmentSpecError) as metric_error:
        build_report(assessment=assessment(metric_ids=("exact_agreement",)))
    assert metric_error.value.code == "essay_validation_metric_not_declared"


def test_engine_authorization_and_kinds_fail_closed() -> None:
    """Only an authorized automated engine and human reference are accepted."""
    with pytest.raises(AssessmentSpecError) as automated_kind:
        build_report(automated_engine=_human_engine())
    assert automated_kind.value.code == "invalid_essay_validation_automated_engine"

    with pytest.raises(AssessmentSpecError) as reference_kind:
        build_report(reference_engine=_automated_engine())
    assert reference_kind.value.code == "invalid_essay_validation_reference_engine"

    with pytest.raises(AssessmentSpecError) as unauthorized:
        build_report(automated_engine=_automated_engine(engine_id="other_engine"))
    assert unauthorized.value.code == "essay_validation_engine_not_authorized"

    with pytest.raises(AssessmentSpecError) as automated_disabled:
        build_report(
            assessment=assessment(
                metric_ids=_ALL_METRIC_IDS,
                engine_ids=(),
                allow_automated=False,
            )
        )
    assert automated_disabled.value.code == "essay_validation_automated_engine_disabled"

    selected = assessment()
    object.__setattr__(selected.engine_policy, "allow_human_raters", False)
    with pytest.raises(AssessmentSpecError) as human_disabled:
        build_report(assessment=selected)
    assert human_disabled.value.code == "essay_validation_human_reference_disabled"


def test_verdict_adapter_rejects_malformed_or_unknown_metrics(monkeypatch) -> None:
    """Only known finite uniquely named Rust metric evidence is accepted."""
    with pytest.raises(AssessmentSpecError) as wrong_type:
        validation_reporting._metrics_from_verdict(object())
    assert wrong_type.value.code == "invalid_essay_validation_verdict"

    malformed = ValidationVerdict(
        gates=[{"value": 0.5}],
        exact_agreement=0.5,
        adjacent_agreement=0.8,
        passed=False,
    )
    with pytest.raises(AssessmentSpecError) as malformed_error:
        validation_reporting._metrics_from_verdict(malformed)
    assert malformed_error.value.code == "invalid_essay_validation_gate"

    unknown = replace(
        malformed,
        gates=[{"name": "unknown_gate", "value": 0.5}],
    )
    with pytest.raises(AssessmentSpecError) as unknown_error:
        validation_reporting._metrics_from_verdict(unknown)
    assert unknown_error.value.code == "unknown_essay_validation_metric"

    duplicate = replace(
        malformed,
        gates=[
            {"name": "qwk", "value": 0.5},
            {"name": "qwk", "value": 0.6},
        ],
    )
    with pytest.raises(AssessmentSpecError) as duplicate_error:
        validation_reporting._metrics_from_verdict(duplicate)
    assert duplicate_error.value.code == "duplicate_essay_validation_metric"

    nonfinite = replace(
        malformed,
        gates=[{"name": "qwk", "value": np.nan}],
    )
    with pytest.raises(AssessmentSpecError) as nonfinite_error:
        validation_reporting._metrics_from_verdict(nonfinite)
    assert nonfinite_error.value.code == "nonfinite_essay_validation_metric"

    invalid_value = replace(
        malformed,
        gates=[{"name": "qwk", "value": object()}],
    )
    with pytest.raises(AssessmentSpecError) as invalid_error:
        validation_reporting._metrics_from_verdict(invalid_value)
    assert invalid_error.value.code == "invalid_essay_validation_metric"

    monkeypatch.setattr(validation_reporting, "validate_judge", lambda *args, **kwargs: object())
    with pytest.raises(AssessmentSpecError) as delegated_error:
        build_report()
    assert delegated_error.value.code == "invalid_essay_validation_verdict"


def test_factory_propagates_label_validation_failures() -> None:
    """Malformed labels remain rejected by the existing Rust-wrapper boundary."""
    with pytest.raises(ValueError, match="judge must be a 1-D array"):
        build_report(automated_labels=np.zeros((2, 2), dtype=np.int64))


def test_metric_and_report_constructors_are_factory_sealed() -> None:
    """Public dataclasses cannot be forged through direct construction."""
    with pytest.raises(AssessmentSpecError) as metric_error:
        EssayValidationMetric(
            metric_id="exact_agreement",
            value=1.0,
            interpretation_id="descriptive_agreement_evidence",
        )
    assert metric_error.value.code == "unverified_essay_validation_metric"

    valid = build_report()
    values = {
        "report_id": valid.report_id,
        "assessment_spec": valid.assessment_spec,
        "construct_id": valid.construct_id,
        "rubric_fingerprint": valid.rubric_fingerprint,
        "criterion_id": valid.criterion_id,
        "automated_engine": valid.automated_engine,
        "reference_engine": valid.reference_engine,
        "validation_dataset_fingerprint": valid.validation_dataset_fingerprint,
        "category_count": valid.category_count,
        "paired_observation_count": valid.paired_observation_count,
        "metrics": valid.metrics,
        "review_trigger_ids": valid.review_trigger_ids,
        "metadata": valid.metadata,
    }
    with pytest.raises(AssessmentSpecError) as report_error:
        EssayValidationEvidenceReport(**values)
    assert report_error.value.code == "unverified_essay_validation_evidence_report"


def test_sealed_report_validates_internal_immutable_shape() -> None:
    """Internal construction also rejects invalid counts, metrics, and ordering."""
    valid = build_report()
    base = {
        "report_id": valid.report_id,
        "assessment_spec": valid.assessment_spec,
        "construct_id": valid.construct_id,
        "rubric_fingerprint": valid.rubric_fingerprint,
        "criterion_id": valid.criterion_id,
        "automated_engine": valid.automated_engine,
        "reference_engine": valid.reference_engine,
        "validation_dataset_fingerprint": valid.validation_dataset_fingerprint,
        "category_count": valid.category_count,
        "paired_observation_count": valid.paired_observation_count,
        "metrics": valid.metrics,
        "review_trigger_ids": valid.review_trigger_ids,
        "metadata": valid.metadata,
        "_report_token": validation_reporting._REPORT_TOKEN,
    }
    cases = (
        ({"category_count": 1}, "invalid_essay_validation_category_count"),
        ({"paired_observation_count": 1}, "invalid_essay_validation_observation_count"),
        ({"metrics": []}, "invalid_essay_validation_metrics"),
        ({"metrics": (object(),)}, "invalid_essay_validation_metric"),
        (
            {"metrics": tuple(reversed(valid.metrics))},
            "invalid_essay_validation_metric_order",
        ),
    )
    for overrides, code in cases:
        with pytest.raises(AssessmentSpecError) as caught:
            EssayValidationEvidenceReport(**{**base, **overrides})
        assert caught.value.code == code


def test_metric_value_and_public_surface_are_documented() -> None:
    """Metric normalization covers overflow and the public surface stays explicit."""
    with pytest.raises(AssessmentSpecError) as overflow:
        validation_reporting._finite_metric_value("1e9999", "exact_agreement")
    assert overflow.value.code == "nonfinite_essay_validation_metric"
    assert validation_reporting.__all__ == [
        "MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS",
        "EssayValidationEvidenceReport",
        "EssayValidationMetric",
        "build_essay_validation_evidence_report",
    ]
    assert EssayValidationEvidenceReport.__doc__
    assert EssayValidationMetric.__doc__
    assert build_essay_validation_evidence_report.__doc__
