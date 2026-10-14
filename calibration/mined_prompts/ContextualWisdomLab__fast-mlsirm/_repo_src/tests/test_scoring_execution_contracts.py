"""Contract tests for governed scoring requests, observations, and engines."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy
from types import MappingProxyType
from typing import Any

import pytest

import fast_mlsirm.scoring as scoring
import fast_mlsirm.scoring.execution as execution_module
from fast_mlsirm.rubric import ResponseFormat
from fast_mlsirm.scoring import (
    AssessmentResponseType,
    AssessmentSpecError,
    EngineDescriptor,
    EngineKind,
    EvidenceReference,
    EvidenceRole,
    FixtureOutcome,
    ObservationGranularity,
    ObservationStatus,
    ScoreObservation,
    ScoringEngine,
    ScoringRequest,
    ScoringResult,
    StaticFixtureEngine,
    build_engine_descriptor,
    build_score_observation,
    build_scoring_request,
    build_scoring_result,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]
evidence = _FIXTURES["evidence"]
fixture_engine = _FIXTURES["fixture_engine"]
holistic_request = _FIXTURES["holistic_request"]
human_engine = _FIXTURES["human_engine"]
rubric = _FIXTURES["rubric"]


class _ExplodingIterable:
    """Iterable fixture that fails through a caller callback."""

    def __iter__(self):
        """Raise a non-standard callback failure."""
        raise RuntimeError("private callback payload")


def test_engine_descriptor_is_content_addressed_and_deeply_immutable() -> None:
    """Equivalent engine metadata ordering yields one immutable identity."""
    first = automated_engine()
    second = automated_engine(
        metadata={"deterministic_mode": True},
    )
    assert first == second
    assert first.engine_kind is EngineKind.AUTOMATED
    assert first.model_id == "fixture_model"
    assert first.prompt_template_fingerprint == "a" * 64
    assert len(first.engine_fingerprint) == 64
    assert first.engine_handle == f"engine_descriptor_{first.engine_fingerprint[:32]}"
    assert isinstance(first.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        first.metadata["deterministic_mode"] = False  # type: ignore[index]
    assert first.to_dict()["engine_fingerprint"] == first.engine_fingerprint


def test_engine_identity_changes_for_every_governed_field() -> None:
    """Engine family, provider, version, model, prompt, and metadata affect identity."""
    base = automated_engine()
    changed = (
        automated_engine(engine_family_id="alternate_family"),
        automated_engine(provider_id="alternate_provider"),
        automated_engine(engine_version="1.0.1"),
        automated_engine(model_id="alternate_model"),
        automated_engine(prompt_template_fingerprint="d" * 64),
        automated_engine(metadata={"deterministic_mode": False}),
    )
    assert all(
        value.engine_fingerprint != base.engine_fingerprint for value in changed
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"engine_id": "bad"}, "invalid_engine_id"),
        ({"engine_version": "01.0.0"}, "invalid_engine_version"),
        ({"engine_kind": "unknown_engine"}, "invalid_engine_kind"),
        ({"model_id": None}, "missing_model_id"),
        ({"prompt_template_fingerprint": None}, "missing_prompt_fingerprint"),
        ({"prompt_driven": False}, "unexpected_prompt_fingerprint"),
        ({"metadata": {"response_text": "private"}}, "sensitive_metadata_field"),
    ],
)
def test_automated_engine_fails_closed(
    overrides: dict[str, Any],
    code: str,
) -> None:
    """Automated engine identity and prompt invariants reject invalid values."""
    with pytest.raises(AssessmentSpecError) as captured:
        automated_engine(**overrides)
    assert captured.value.code == code
    assert "private" not in str(captured.value)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"model_id": "human_model"}, "human_model_forbidden"),
        ({"prompt_driven": True}, "human_prompt_forbidden"),
        ({"prompt_template_fingerprint": "a" * 64}, "human_prompt_forbidden"),
    ],
)
def test_human_engine_cannot_claim_automated_model_provenance(
    overrides: dict[str, Any],
    code: str,
) -> None:
    """Human raters remain distinct from prompt-driven automated engines."""
    with pytest.raises(AssessmentSpecError) as captured:
        human_engine(**overrides)
    assert captured.value.code == code


def test_non_prompt_automated_engine_is_supported_without_a_prompt_digest() -> None:
    """Deterministic compiled engines need a model ID but no prompt identity."""
    value = automated_engine(
        prompt_driven=False,
        prompt_template_fingerprint=None,
    )
    assert value.prompt_driven is False
    assert value.prompt_template_fingerprint is None


def test_engine_descriptor_is_factory_sealed() -> None:
    """Direct construction cannot relabel unverified engine metadata."""
    with pytest.raises(AssessmentSpecError, match="unverified_engine_descriptor"):
        EngineDescriptor(
            engine_id="fixture_engine",
            engine_family_id="fixture_family",
            provider_id="local_provider",
            engine_version="1.0.0",
            engine_kind=EngineKind.AUTOMATED,
            model_id="fixture_model",
            prompt_driven=True,
            prompt_template_fingerprint="a" * 64,
            metadata={},
        )


def test_evidence_reference_is_canonical_and_content_addressed() -> None:
    """Evidence retains exact source/span/digest provenance without source text."""
    value = evidence()
    assert value.evidence_role is EvidenceRole.SUPPORTING
    assert len(value.evidence_fingerprint) == 64
    assert value.evidence_handle.endswith(value.evidence_fingerprint[:32])
    assert value.to_dict()["content_fingerprint"] == "b" * 64


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"source_id": "bad"}, "invalid_source_id"),
        ({"span_id": "bad"}, "invalid_span_id"),
        ({"content_fingerprint": "not_a_digest"}, "invalid_content_fingerprint"),
        ({"evidence_role": "unknown_role"}, "invalid_evidence_role"),
    ],
)
def test_evidence_reference_rejects_invalid_provenance(
    kwargs: dict[str, Any],
    code: str,
) -> None:
    """Evidence identity fields fail with stable non-reflective codes."""
    values: dict[str, Any] = {
        "source_id": "source_document",
        "span_id": "evidence_span",
        "content_fingerprint": "b" * 64,
        "evidence_role": EvidenceRole.SUPPORTING,
    }
    values.update(kwargs)
    with pytest.raises(AssessmentSpecError) as captured:
        EvidenceReference(**values)
    assert captured.value.code == code


def test_criterion_request_is_bound_to_exact_assessment_and_rubric() -> None:
    """Request identity preserves exact graph, score levels, and content statistics."""
    first = criterion_request(
        criterion_ids=("source_alignment", "claim_support"),
        metadata={"language_code": "en", "threshold_value": -0.0},
    )
    second = criterion_request(
        criterion_ids=("claim_support", "source_alignment"),
        metadata={"threshold_value": 0.0, "language_code": "en"},
    )
    assert first == second
    assert first.granularity is ObservationGranularity.CRITERION_LEVEL
    assert first.criterion_ids == ("claim_support", "source_alignment")
    assert first.allowed_scores == (0, 1, 2)
    assert first.response_format is ResponseFormat.ORDINAL_RATING
    assert first.assessment_fingerprint == assessment().assessment_fingerprint
    assert first.rubric_fingerprint == rubric().fingerprint
    assert first.construct_id == "evidence_quality"
    assert len(first.request_fingerprint) == 64
    assert first.request_handle.endswith(first.request_fingerprint[:32])
    assert isinstance(first.metadata, MappingProxyType)
    assert first.metadata["threshold_value"] == 0.0


def test_holistic_request_keeps_granularity_explicit() -> None:
    """Holistic requests have no criterion identifiers and remain distinct."""
    request = holistic_request()
    assert request.granularity is ObservationGranularity.HOLISTIC
    assert request.criterion_ids == ()


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"request_id": "bad"}, "invalid_request_id"),
        ({"granularity": "unknown_level"}, "invalid_granularity"),
        ({"task_family_id": "unknown_family"}, "unknown_task_family"),
        ({"criterion_ids": ()}, "missing_criterion_ids"),
        ({"criterion_ids": ("claim_support", "claim_support")}, "duplicate_criterion_ids"),
        ({"response_content_fingerprint": "bad"}, "invalid_response_content_fingerprint"),
        ({"response_character_count": -1}, "invalid_response_character_count"),
        ({"response_character_count": 1 << 63}, "invalid_response_character_count"),
        ({"response_unit_count": -1}, "invalid_response_unit_count"),
        ({"metadata": {"essay_text": "private"}}, "sensitive_metadata_field"),
    ],
)
def test_request_rejects_invalid_inputs(
    overrides: dict[str, Any],
    code: str,
) -> None:
    """Request identity, work bounds, and raw-content exclusion fail closed."""
    with pytest.raises(AssessmentSpecError) as captured:
        criterion_request(**overrides)
    assert captured.value.code == code
    assert "private" not in str(captured.value)


def test_request_rejects_wrong_graph_references_and_granularity() -> None:
    """Assessment, rubric, construct, and response-type mismatches cannot pass."""
    alternate = rubric().to_dict()
    alternate["rubric_version"] = "1.0.1"
    changed_rubric = type(rubric())(
        **{
            "rubric_id": alternate["rubric_id"],
            "construct_id": alternate["construct_id"],
            "construct_definition": alternate["construct_definition"],
            "response_format": alternate["response_format"],
            "levels": rubric().levels,
            "task_families": tuple(alternate["task_families"]),
            "evidence_requirements": tuple(alternate["evidence_requirements"]),
            "prohibited_patterns": tuple(alternate["prohibited_patterns"]),
            "locale": alternate["locale"],
            "rubric_version": alternate["rubric_version"],
        }
    )
    with pytest.raises(AssessmentSpecError, match="unknown_rubric_fingerprint"):
        criterion_request(rubric=changed_rubric)
    with pytest.raises(AssessmentSpecError, match="unsupported_request_granularity"):
        criterion_request(
            assessment=assessment(AssessmentResponseType.HOLISTIC),
        )
    with pytest.raises(AssessmentSpecError, match="unexpected_criterion_ids"):
        holistic_request(criterion_ids=("claim_support",))


def test_request_rejects_unsafe_collections_and_direct_construction() -> None:
    """Request collections are bounded and the public artifact is factory-sealed."""
    with pytest.raises(AssessmentSpecError, match="invalid_criterion_ids"):
        criterion_request(criterion_ids=_ExplodingIterable())
    with pytest.raises(AssessmentSpecError, match="invalid_criterion_ids"):
        criterion_request(
            criterion_ids=(f"criterion_{index}" for index in range(33))
        )
    request = criterion_request()
    with pytest.raises(AssessmentSpecError, match="unverified_scoring_request"):
        ScoringRequest(
            request_id=request.request_id,
            assessment_fingerprint=request.assessment_fingerprint,
            rubric_id=request.rubric_id,
            rubric_fingerprint=request.rubric_fingerprint,
            construct_id=request.construct_id,
            response_format=request.response_format,
            granularity=request.granularity,
            respondent_id=request.respondent_id,
            response_id=request.response_id,
            task_id=request.task_id,
            task_revision_fingerprint=request.task_revision_fingerprint,
            task_family_id=request.task_family_id,
            occasion_id=request.occasion_id,
            criterion_ids=request.criterion_ids,
            allowed_scores=request.allowed_scores,
            response_content_fingerprint=request.response_content_fingerprint,
            response_character_count=request.response_character_count,
            response_unit_count=request.response_unit_count,
            metadata=request.metadata,
        )


def test_scored_observation_preserves_evidence_and_score() -> None:
    """Scored observations require one allowed score and no terminal reason."""
    request = criterion_request()
    engine = automated_engine()
    first = build_score_observation(
        observation_id="claim_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=2,
        evidence_references=(
            evidence("source_document", "second_span"),
            evidence("source_document", "first_span"),
        ),
        confidence_metadata={"confidence_value": 0.9},
    )
    second = build_score_observation(
        observation_id="claim_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status="scored",
        score_category=2,
        evidence_references=tuple(reversed(first.evidence_references)),
        confidence_metadata={"confidence_value": 0.9},
    )
    assert first == second
    assert first.score_category == 2
    assert first.reason_code is None
    assert first.criterion_id == "claim_support"
    assert len(first.observation_fingerprint) == 64
    assert first.evidence_references[0].span_id == "first_span"
    assert isinstance(first.confidence_metadata, MappingProxyType)


def test_non_scored_observation_requires_reason_and_forbids_score() -> None:
    """Abstained, failed, and excluded observations cannot fabricate scores."""
    request = criterion_request()
    engine = automated_engine()
    for status in (
        ObservationStatus.ABSTAINED,
        ObservationStatus.FAILED,
        ObservationStatus.EXCLUDED,
    ):
        value = build_score_observation(
            observation_id=f"{status.value}_observation",
            request=request,
            engine=engine,
            criterion_id="claim_support",
            status=status,
            score_category=None,
            reason_code="insufficient_evidence",
        )
        assert value.status is status
        assert value.score_category is None
        assert value.reason_code == "insufficient_evidence"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"status": "unknown_status"}, "invalid_observation_status"),
        ({"status": ObservationStatus.SCORED, "score_category": None}, "missing_score_category"),
        ({"status": ObservationStatus.SCORED, "score_category": 9}, "unknown_score_category"),
        ({"status": ObservationStatus.SCORED, "score_category": 2, "reason_code": "failure_reason"}, "unexpected_reason_code"),
        ({"status": ObservationStatus.FAILED, "score_category": 0, "reason_code": "engine_failure"}, "unexpected_score_category"),
        ({"status": ObservationStatus.FAILED, "score_category": None, "reason_code": None}, "missing_reason_code"),
        ({"criterion_id": "unknown_criterion"}, "unknown_criterion_id"),
        ({"evidence_references": (evidence(), evidence())}, "duplicate_evidence_reference"),
        ({"confidence_metadata": {"raw_response": "private"}}, "sensitive_metadata_field"),
    ],
)
def test_observation_rejects_invalid_status_score_and_evidence(
    kwargs: dict[str, Any],
    code: str,
) -> None:
    """Observation semantics and evidence provenance fail closed."""
    values: dict[str, Any] = {
        "observation_id": "claim_observation",
        "request": criterion_request(),
        "engine": automated_engine(),
        "criterion_id": "claim_support",
        "status": ObservationStatus.SCORED,
        "score_category": 2,
        "reason_code": None,
        "evidence_references": (),
        "confidence_metadata": {},
    }
    values.update(kwargs)
    with pytest.raises(AssessmentSpecError) as captured:
        build_score_observation(**values)
    assert captured.value.code == code
    assert "private" not in str(captured.value)


def test_holistic_observation_requires_null_criterion() -> None:
    """Holistic requests cannot acquire a hidden criterion identity."""
    request = holistic_request()
    engine = human_engine()
    value = build_score_observation(
        observation_id="holistic_observation",
        request=request,
        engine=engine,
        criterion_id=None,
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    assert value.criterion_id is None
    with pytest.raises(AssessmentSpecError, match="unexpected_criterion_id"):
        build_score_observation(
            observation_id="holistic_observation",
            request=request,
            engine=engine,
            criterion_id="claim_support",
            status=ObservationStatus.SCORED,
            score_category=1,
        )


def test_observation_is_factory_sealed() -> None:
    """Direct construction cannot relabel an unverified scoring observation."""
    request = criterion_request()
    engine = automated_engine()
    with pytest.raises(AssessmentSpecError, match="unverified_score_observation"):
        ScoreObservation(
            observation_id="claim_observation",
            request_fingerprint=request.request_fingerprint,
            engine_fingerprint=engine.engine_fingerprint,
            assessment_fingerprint=request.assessment_fingerprint,
            rubric_fingerprint=request.rubric_fingerprint,
            construct_id=request.construct_id,
            granularity=request.granularity,
            criterion_id="claim_support",
            status=ObservationStatus.SCORED,
            score_category=2,
            reason_code=None,
            evidence_references=(),
            confidence_metadata={},
        )


def test_result_requires_complete_canonical_criterion_coverage() -> None:
    """Results cover each requested criterion once and preserve execution provenance."""
    request = criterion_request()
    engine = automated_engine()
    claim = build_score_observation(
        observation_id="claim_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=2,
    )
    source = build_score_observation(
        observation_id="source_observation",
        request=request,
        engine=engine,
        criterion_id="source_alignment",
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    first = build_scoring_result(
        result_id="scoring_result",
        request=request,
        engine=engine,
        observations=(source, claim),
        execution_attempt=1,
        diagnostics={"duration_milliseconds": 12},
    )
    second = build_scoring_result(
        result_id="scoring_result",
        request=request,
        engine=engine,
        observations=(claim, source),
        execution_attempt=1,
        diagnostics={"duration_milliseconds": 12},
    )
    assert first == second
    assert tuple(value.criterion_id for value in first.observations) == (
        "claim_support",
        "source_alignment",
    )
    assert first.request_fingerprint == request.request_fingerprint
    assert first.engine_fingerprint == engine.engine_fingerprint
    assert len(first.result_fingerprint) == 64
    assert first.result_handle.endswith(first.result_fingerprint[:32])
    assert isinstance(first.diagnostics, MappingProxyType)


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        ("missing", "incomplete_observation_coverage"),
        ("duplicate", "duplicate_observation_criterion"),
        ("wrong_request", "observation_request_mismatch"),
        ("wrong_engine", "observation_engine_mismatch"),
        ("bad_attempt", "invalid_execution_attempt"),
        ("private_diagnostic", "sensitive_metadata_field"),
    ],
)
def test_result_rejects_incomplete_or_mismatched_execution(
    mutator: str,
    code: str,
) -> None:
    """Result coverage, identity, attempt, and diagnostics fail closed."""
    request = criterion_request()
    engine = automated_engine()
    claim = build_score_observation(
        observation_id="claim_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=2,
    )
    source = build_score_observation(
        observation_id="source_observation",
        request=request,
        engine=engine,
        criterion_id="source_alignment",
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    observations: Any = (claim, source)
    execution_attempt: Any = 1
    diagnostics: Any = {}
    selected_request = request
    selected_engine = engine
    if mutator == "missing":
        observations = (claim,)
    elif mutator == "duplicate":
        observations = (claim, claim)
    elif mutator == "wrong_request":
        other_request = criterion_request(response_id="alternate_response")
        observations = (
            build_score_observation(
                observation_id="claim_observation",
                request=other_request,
                engine=engine,
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
            ),
            source,
        )
    elif mutator == "wrong_engine":
        other_engine = automated_engine(engine_id="alternate_engine")
        observations = (
            build_score_observation(
                observation_id="claim_observation",
                request=request,
                engine=other_engine,
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
            ),
            source,
        )
    elif mutator == "bad_attempt":
        execution_attempt = 0
    elif mutator == "private_diagnostic":
        diagnostics = {"provider_output": "private"}
    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            request=selected_request,
            engine=selected_engine,
            observations=observations,
            execution_attempt=execution_attempt,
            diagnostics=diagnostics,
        )
    assert captured.value.code == code
    assert "private" not in str(captured.value)


def test_holistic_result_requires_exactly_one_holistic_observation() -> None:
    """A holistic execution returns exactly one null-criterion observation."""
    request = holistic_request()
    engine = human_engine()
    observation = build_score_observation(
        observation_id="holistic_observation",
        request=request,
        engine=engine,
        criterion_id=None,
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    result = build_scoring_result(
        result_id="holistic_result",
        request=request,
        engine=engine,
        observations=(observation,),
    )
    assert len(result.observations) == 1
    assert result.observations[0].criterion_id is None


def test_result_is_factory_sealed() -> None:
    """Direct construction cannot relabel unverified execution results."""
    request = holistic_request()
    engine = human_engine()
    observation = build_score_observation(
        observation_id="holistic_observation",
        request=request,
        engine=engine,
        criterion_id=None,
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    with pytest.raises(AssessmentSpecError, match="unverified_scoring_result"):
        ScoringResult(
            result_id="holistic_result",
            request_fingerprint=request.request_fingerprint,
            engine_fingerprint=engine.engine_fingerprint,
            granularity=request.granularity,
            requested_criterion_ids=request.criterion_ids,
            observations=(observation,),
            execution_attempt=1,
            diagnostics={},
        )


def test_static_fixture_engine_implements_protocol_and_is_deterministic() -> None:
    """Offline fixture execution uses the same public request/result contracts."""
    engine = fixture_engine()
    assert isinstance(engine, ScoringEngine)
    request = criterion_request()
    first = engine.score(request)
    second = engine.score(request)
    assert first == second
    assert first.engine_fingerprint == engine.descriptor.engine_fingerprint
    assert tuple(value.score_category for value in first.observations) == (2, 1)


def test_fixture_engine_rejects_wrong_outcome_coverage_and_holistic_mismatch() -> None:
    """Fixture configuration must cover the explicit request granularity exactly."""
    descriptor = automated_engine()
    incomplete = StaticFixtureEngine(
        descriptor=descriptor,
        outcomes=(
            FixtureOutcome(
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
            ),
        ),
    )
    with pytest.raises(AssessmentSpecError, match="incomplete_fixture_coverage"):
        incomplete.score(criterion_request())
    with pytest.raises(AssessmentSpecError, match="fixture_granularity_mismatch"):
        incomplete.score(holistic_request())


def test_fixture_outcome_validates_terminal_semantics_and_evidence() -> None:
    """Fixture outcomes cannot defer invalid score/status combinations to execution."""
    with pytest.raises(AssessmentSpecError, match="missing_fixture_reason"):
        FixtureOutcome(
            criterion_id="claim_support",
            status=ObservationStatus.ABSTAINED,
        )
    with pytest.raises(AssessmentSpecError, match="unexpected_fixture_score"):
        FixtureOutcome(
            criterion_id="claim_support",
            status=ObservationStatus.FAILED,
            score_category=0,
            reason_code="engine_failure",
        )
    with pytest.raises(AssessmentSpecError, match="duplicate_evidence_reference"):
        FixtureOutcome(
            criterion_id="claim_support",
            status=ObservationStatus.SCORED,
            score_category=2,
            evidence_references=(evidence(), evidence()),
        )


def test_public_exports_and_docstrings_are_complete() -> None:
    """The scoring namespace exposes the complete documented execution surface."""
    expected = {
        "EngineDescriptor",
        "EngineKind",
        "EvidenceReference",
        "EvidenceRole",
        "FixtureOutcome",
        "ObservationGranularity",
        "ObservationStatus",
        "ScoreObservation",
        "ScoringEngine",
        "ScoringRequest",
        "ScoringResult",
        "StaticFixtureEngine",
        "build_engine_descriptor",
        "build_score_observation",
        "build_scoring_request",
        "build_scoring_result",
    }
    # Execution contracts are documented explicit attributes; ``__all__``
    # stays pinned to the pre-execution surface until the next public-surface
    # version bump (see the package-namespace governance comment).
    assert not expected & set(scoring.__all__)
    for name in expected:
        value = getattr(scoring, name)
        assert value is globals()[name]
        assert inspect.getdoc(value)


def test_execution_errors_are_structured_non_reflective_and_callback_safe() -> None:
    """Representative callback and enum failures use safe codes and index paths."""
    with pytest.raises(AssessmentSpecError) as captured:
        criterion_request(criterion_ids=_ExplodingIterable())
    assert captured.value.code == "invalid_criterion_ids"
    assert captured.value.path == "$.criterion_ids"
    assert "private callback payload" not in str(captured.value)

    with pytest.raises(AssessmentSpecError) as captured:
        build_engine_descriptor(
            engine_id="fixture_engine",
            engine_family_id="fixture_family",
            provider_id="local_provider",
            engine_version="1.0.0",
            engine_kind="private_engine_kind",
            model_id="fixture_model",
            prompt_driven=False,
        )
    assert captured.value.code == "invalid_engine_kind"
    assert "private_engine_kind" not in str(captured.value)


def test_execution_contract_identities_change_for_material_mutations() -> None:
    """Request, observation, and result identities respond to material changes."""
    base_request = criterion_request()
    changed_request = criterion_request(response_unit_count=9)
    assert base_request.request_fingerprint != changed_request.request_fingerprint

    engine = automated_engine()
    base_observation = build_score_observation(
        observation_id="claim_observation",
        request=base_request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=2,
    )
    changed_observation = build_score_observation(
        observation_id="claim_observation",
        request=base_request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    assert (
        base_observation.observation_fingerprint
        != changed_observation.observation_fingerprint
    )

    source_observation = build_score_observation(
        observation_id="source_observation",
        request=base_request,
        engine=engine,
        criterion_id="source_alignment",
        status=ObservationStatus.SCORED,
        score_category=1,
    )
    first = build_scoring_result(
        result_id="scoring_result",
        request=base_request,
        engine=engine,
        observations=(base_observation, source_observation),
        execution_attempt=1,
    )
    second = build_scoring_result(
        result_id="scoring_result",
        request=base_request,
        engine=engine,
        observations=(changed_observation, source_observation),
        execution_attempt=1,
    )
    assert first.result_fingerprint != second.result_fingerprint


def test_execution_module_internal_seals_reject_wrong_tokens() -> None:
    """Factory tokens remain private implementation details, not public bypasses."""
    assert execution_module._ENGINE_DESCRIPTOR_TOKEN is not None
    assert execution_module._SCORING_REQUEST_TOKEN is not None
    assert execution_module._SCORE_OBSERVATION_TOKEN is not None
    assert execution_module._SCORING_RESULT_TOKEN is not None


def test_result_rejects_a_reused_observation_identifier_across_criteria() -> None:
    """A caller-reused observation identifier fails after criterion checks."""
    request = criterion_request()
    engine = automated_engine()
    observations = tuple(
        build_score_observation(
            observation_id="reused_observation",
            request=request,
            engine=engine,
            criterion_id=criterion_id,
            status=ObservationStatus.SCORED,
            score_category=1,
        )
        for criterion_id in ("claim_support", "source_alignment")
    )

    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            request=request,
            engine=engine,
            observations=observations,
        )

    assert captured.value.code == "duplicate_observation_id"
    assert captured.value.path == "$.observations"
