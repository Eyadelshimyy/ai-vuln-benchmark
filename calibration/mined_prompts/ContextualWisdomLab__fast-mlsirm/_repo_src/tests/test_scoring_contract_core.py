"""Core behavior for shared assessment and scoring-policy contracts."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import runpy
from types import MappingProxyType

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring import (
    ASSESSMENT_SCHEMA_VERSION,
    MAX_METADATA_COLLECTION_VALUES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_NODES,
    AdjudicationPolicy,
    AssessmentResponseType,
    AssessmentSpec,
    AssessmentSpecError,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
    artifact_digest,
    build_assessment_spec,
    canonical_json,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
policies = _FIXTURES["policies"]
rubric = _FIXTURES["rubric"]


def test_assessment_spec_is_deterministic_content_addressed_and_deeply_immutable():
    """Input ordering cannot change identity and nested metadata cannot mutate."""
    first = assessment()
    argument_rubric = rubric("argument_rubric", "argument_quality")
    evidence_rubric = rubric("evidence_rubric", "evidence_use")
    second = assessment(
        constructs=tuple(reversed(first.constructs)),
        rubrics=(evidence_rubric, argument_rubric),
        selected_policies=policies(tuple(reversed(first.construct_ids))),
        metadata={
            "optional_value": None,
            "enabled_flag": True,
            "nested_metadata": {
                "threshold_values": (0.1, 0.2),
                "fold_count": 5,
            },
            "study_name": "Connected sparse pilot",
        },
    )

    assert first == second
    assert first.schema_version == ASSESSMENT_SCHEMA_VERSION
    assert first.construct_ids == ("argument_quality", "evidence_use")
    assert first.rubric_fingerprints == tuple(sorted(first.rubric_fingerprints))
    assert len(first.assessment_fingerprint) == 64
    assert first.assessment_handle == (
        f"assessment_spec_{first.assessment_fingerprint[:32]}"
    )
    payload = first.to_dict()
    assert payload["assessment_handle"] == first.assessment_handle
    assert payload["assessment_fingerprint"] == first.assessment_fingerprint
    assert canonical_json(first) == canonical_json(second)
    assert artifact_digest(first) == first.assessment_fingerprint

    assert isinstance(first.metadata, MappingProxyType)
    nested = first.metadata["nested_metadata"]
    assert isinstance(nested, MappingProxyType)
    assert nested["threshold_values"] == (0.1, 0.2)
    with pytest.raises(TypeError):
        first.metadata["study_name"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        nested["fold_count"] = 10  # type: ignore[index]


def test_canonical_json_and_digest_normalize_supported_artifacts():
    """Mappings, tuples, public contracts, and finite scalars serialize stably."""
    left = {"beta_value": [2, 3], "alpha_value": {"finite_value": 1.5}}
    right = {
        "alpha_value": {"finite_value": 1.5},
        "beta_value": (2, 3),
    }
    assert canonical_json(left) == canonical_json(right)
    assert artifact_digest(left) == artifact_digest(right)
    assert len(artifact_digest(left)) == 64

    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=("a" * 64,),
    )
    assert "argument_quality" in canonical_json(construct)
    assert len(artifact_digest(construct)) == 64

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            canonical_json({"invalid_value": value})
        with pytest.raises(ValueError, match="finite"):
            artifact_digest({"invalid_value": value})


def test_contract_types_normalize_identifiers_collections_and_enum_values():
    """Public policy contracts expose sorted unique immutable values and payloads."""
    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=("b" * 64, "a" * 64),
    )
    assert construct.rubric_fingerprints == ("a" * 64, "b" * 64)
    assert construct.to_dict()["construct_id"] == "argument_quality"

    engine = EnginePolicy(
        policy_id="engine_policy",
        engine_ids=("human_adapter", "fixture_engine"),
        allow_human_raters=True,
        allow_automated_raters=True,
        minimum_raters_per_response=2,
    )
    assert engine.engine_ids == ("fixture_engine", "human_adapter")
    assert engine.to_dict()["minimum_raters_per_response"] == 2

    calibration = CalibrationPolicy(
        policy_id="calibration_policy",
        model_id="facets_ordinal",
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert calibration.construct_ids == ("argument_quality", "evidence_use")
    assert calibration.to_dict()["model_id"] == "facets_ordinal"

    validation = ValidationPolicy(
        policy_id="validation_policy",
        metric_ids=("quadratic_weighted_kappa", "exact_agreement"),
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert validation.metric_ids == (
        "exact_agreement",
        "quadratic_weighted_kappa",
    )
    assert validation.to_dict()["metric_ids"][0] == "exact_agreement"

    adjudication = AdjudicationPolicy(
        policy_id="adjudication_policy",
        trigger_ids=("scorer_disagreement", "insufficient_evidence"),
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert adjudication.trigger_ids == (
        "insufficient_evidence",
        "scorer_disagreement",
    )
    assert adjudication.to_dict()["trigger_ids"][0] == "insufficient_evidence"

    monitoring = MonitoringPolicy(
        policy_id="monitoring_policy",
        metric_ids=("severity_drift", "failure_rate_drift"),
        construct_ids=("evidence_use", "argument_quality"),
    )
    assert monitoring.metric_ids == ("failure_rate_drift", "severity_drift")
    assert monitoring.to_dict()["metric_ids"][0] == "failure_rate_drift"

    reporting = ReportingPolicy(
        policy_id="reporting_policy",
        format_ids=("json_report", "html_report"),
        construct_ids=("evidence_use", "argument_quality"),
        include_exact_values=True,
    )
    assert reporting.format_ids == ("html_report", "json_report")
    assert reporting.include_exact_values is True
    assert reporting.to_dict()["include_exact_values"] is True

    spec = assessment(response_type="criterion_level")
    assert spec.response_type is AssessmentResponseType.CRITERION_LEVEL
    assert spec.to_dict()["response_type"] == "criterion_level"


def test_assessment_defaults_metadata_to_an_empty_immutable_mapping():
    """The public builder supplies an immutable empty mapping when metadata is omitted."""
    argument_rubric = rubric("argument_rubric", "argument_quality")
    policy_values = policies(("argument_quality",))
    spec = build_assessment_spec(
        assessment_id="essay_assessment",
        assessment_version="1.0.0",
        constructs=(
            ConstructSpec(
                construct_id="argument_quality",
                construct_definition="Quality of the response argument.",
                rubric_fingerprints=(argument_rubric.fingerprint,),
            ),
        ),
        rubrics=(argument_rubric,),
        response_type=AssessmentResponseType.HOLISTIC,
        engine_policy=policy_values[0],
        calibration_policy=policy_values[1],
        validation_policy=policy_values[2],
        adjudication_policy=policy_values[3],
        monitoring_policy=policy_values[4],
        reporting_policy=policy_values[5],
    )
    assert isinstance(spec.metadata, MappingProxyType)
    assert dict(spec.metadata) == {}
    assert spec.response_type is AssessmentResponseType.HOLISTIC


def test_assessment_is_factory_sealed():
    """Only the cross-reference-validating builder may create an assessment artifact."""
    spec = assessment()
    with pytest.raises(ValueError, match="build_assessment_spec"):
        AssessmentSpec(
            assessment_id=spec.assessment_id,
            assessment_version=spec.assessment_version,
            constructs=spec.constructs,
            rubric_fingerprints=spec.rubric_fingerprints,
            response_type=spec.response_type,
            engine_policy=spec.engine_policy,
            calibration_policy=spec.calibration_policy,
            validation_policy=spec.validation_policy,
            adjudication_policy=spec.adjudication_policy,
            monitoring_policy=spec.monitoring_policy,
            reporting_policy=spec.reporting_policy,
            metadata=spec.metadata,
        )


def test_scoring_contract_public_exports_and_docstrings_are_complete():
    """The namespace exposes only the documented provider-neutral contract surface."""
    expected = {
        "ASSESSMENT_SCHEMA_VERSION",
        "MAX_METADATA_COLLECTION_VALUES",
        "MAX_METADATA_DEPTH",
        "MAX_METADATA_NODES",
        "AdjudicationPolicy",
        "AssessmentResponseType",
        "AssessmentSpec",
        "AssessmentSpecError",
        "CalibrationPolicy",
        "ConstructSpec",
        "EnginePolicy",
        "MonitoringPolicy",
        "ReportingPolicy",
        "ValidationPolicy",
        "artifact_digest",
        "build_assessment_spec",
        "canonical_json",
    }
    assert set(scoring.__all__) == expected
    for name in expected:
        value = getattr(scoring, name)
        assert value is globals()[name]
        if callable(value):
            assert inspect.getdoc(value)

    assert ASSESSMENT_SCHEMA_VERSION == "1.0"
    assert MAX_METADATA_COLLECTION_VALUES == 64
    assert MAX_METADATA_DEPTH == 8
    assert MAX_METADATA_NODES == 1_024
    error = AssessmentSpecError("sample_error", "$.sample", "sample message")
    assert error.code == "sample_error"
    assert error.path == "$.sample"
    assert error.message == "sample message"
    assert str(error) == "sample_error at $.sample: sample message"
