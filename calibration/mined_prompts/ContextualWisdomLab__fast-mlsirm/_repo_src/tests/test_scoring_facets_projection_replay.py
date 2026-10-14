"""Replay-hardening tests for scoring-to-facets projection provenance."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    ObservationGranularity,
    ObservationStatus,
    build_scoring_facets_rating_records,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
automated_engine = _BASE["automated_engine"]
execution = _BASE["execution"]


def replay_execution():
    """Return one complete criterion-level execution for replay mutation tests."""
    return execution(
        request_id="projection_replay_request",
        response_id="projection_replay_response",
        respondent_id="projection_replay_respondent",
        task_id="projection_replay_task",
        engine=automated_engine(),
        claim_score=1,
        source_score=2,
    )


@pytest.mark.parametrize(
    ("field_name", "mutated_value", "expected_code"),
    (
        (
            "request_fingerprint",
            "a" * 64,
            "calibration_observation_request_mismatch",
        ),
        (
            "engine_fingerprint",
            "b" * 64,
            "calibration_observation_engine_mismatch",
        ),
        (
            "assessment_fingerprint",
            "c" * 64,
            "calibration_observation_assessment_mismatch",
        ),
        (
            "rubric_fingerprint",
            "d" * 64,
            "calibration_observation_rubric_mismatch",
        ),
        (
            "construct_id",
            "mutated_construct",
            "calibration_observation_construct_mismatch",
        ),
        (
            "granularity",
            ObservationGranularity.HOLISTIC,
            "calibration_observation_granularity_mismatch",
        ),
        (
            "criterion_id",
            "undeclared_criterion",
            "calibration_observation_criterion_mismatch",
        ),
    ),
)
def test_projection_rejects_mutated_observation_provenance(
    field_name: str,
    mutated_value,
    expected_code: str,
) -> None:
    """A result cannot silently discard observation-level provenance mutations."""
    request, result, engine = replay_execution()
    observation = result.observations[0]
    object.__setattr__(observation, field_name, mutated_value)

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == expected_code
    assert caught.value.path.endswith(f".{field_name}")
    assert str(mutated_value) not in str(caught.value)


def test_projection_rejects_mutated_result_criterion_scope() -> None:
    """Result-declared criterion scope must replay the supplied request exactly."""
    request, result, engine = replay_execution()
    object.__setattr__(result, "requested_criterion_ids", ("claim_support",))

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "calibration_result_criteria_mismatch"
    assert caught.value.path.endswith(".requested_criterion_ids")


def test_projection_rejects_mutated_result_observation_coverage() -> None:
    """A result cannot drop one requested criterion after factory validation."""
    request, result, engine = replay_execution()
    object.__setattr__(result, "observations", result.observations[:1])

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "calibration_observation_coverage_mismatch"
    assert caught.value.path.endswith(".observations")


def test_projection_rejects_duplicate_observation_criterion() -> None:
    """Duplicate criterion coverage remains distinct from simple omission."""
    request, result, engine = replay_execution()
    object.__setattr__(
        result,
        "observations",
        (*result.observations, result.observations[1]),
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "duplicate_observation_criterion"
    assert caught.value.path.endswith(".observations")


def test_projection_rejects_duplicate_observation_identity() -> None:
    """Distinct criteria cannot reuse one observation identity after mutation."""
    request, result, engine = replay_execution()
    object.__setattr__(
        result.observations[1],
        "observation_id",
        result.observations[0].observation_id,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "duplicate_observation_id"
    assert caught.value.path.endswith(".observations")


def test_projection_rejects_untyped_observation_entry() -> None:
    """A privately replaced child cannot escape as an AttributeError."""
    request, result, engine = replay_execution()
    private_value = object()
    object.__setattr__(
        result,
        "observations",
        (result.observations[0], private_value),
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "invalid_score_observation"
    assert caught.value.path.endswith(".observations[1]")
    assert str(private_value) not in str(caught.value)


def test_projection_rejects_hostile_observation_container_without_iteration() -> None:
    """A package-owned result container is rejected before hostile iteration."""
    request, result, engine = replay_execution()

    class HostileObservations:
        """Raise if the calibration boundary attempts to iterate this object."""

        def __iter__(self):
            raise AssertionError("hostile observations must not be iterated")

    object.__setattr__(result, "observations", HostileObservations())

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "invalid_observations"
    assert caught.value.path == "$.observations"


def test_projection_does_not_mutate_a_valid_result() -> None:
    """Validation and projection leave the caller-owned frozen result untouched."""
    request, result, engine = replay_execution()
    observations = result.observations
    fingerprint = result.result_fingerprint

    records = build_scoring_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )

    assert records
    assert result.observations is observations
    assert result.result_fingerprint == fingerprint


def test_projection_replays_nested_evidence_reference_types() -> None:
    """An untyped nested evidence child fails before canonical serialization."""
    request, result, engine = replay_execution()
    private_value = object()
    object.__setattr__(
        result.observations[0],
        "evidence_references",
        (private_value,),
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "invalid_evidence_reference"
    assert str(private_value) not in str(caught.value)


def test_projection_bounds_nested_evidence_collections() -> None:
    """Oversized evidence collections fail before fingerprint computation."""
    request, result, engine = replay_execution()
    evidence = EvidenceReference(
        source_id="source_record",
        span_id="source_span",
        content_fingerprint="e" * 64,
    )
    object.__setattr__(
        result.observations[0],
        "evidence_references",
        tuple(evidence for _index in range(65)),
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "invalid_evidence_references"
    assert caught.value.path == "$.evidence_references"


def test_projection_replays_observation_status_score_relationship() -> None:
    """Post-construction status mutation cannot bypass observation semantics."""
    request, result, engine = replay_execution()
    object.__setattr__(result.observations[0], "status", ObservationStatus.ABSTAINED)

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "unexpected_score_category"
    assert caught.value.path == "$.score_category"


def test_projection_replays_confidence_metadata_safety() -> None:
    """Sensitive response content cannot enter mutated observation metadata."""
    request, result, engine = replay_execution()
    secret = "private_response_payload"
    object.__setattr__(
        result.observations[0],
        "confidence_metadata",
        {"response_text": secret},
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "sensitive_metadata_field"
    assert secret not in str(caught.value)
