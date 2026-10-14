"""Many-facet orchestration contracts for governed RAG scoring executions."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    FixtureOutcome,
    ObservationStatus,
    ScoringFacetsCalibrationBundle,
    ScoringFacetsRatingRecord,
    StaticFixtureEngine,
)
from fast_mlsirm.scoring.rag import build_rag_scoring_request
from fast_mlsirm.scoring.rag_calibration import (
    build_rag_facets_calibration_bundle,
    build_rag_facets_rating_records,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]

QUERY_REVISION_FPS = tuple(
    hashlib.sha256(f"rag-calibration-query-revision-{index:03d}".encode()).hexdigest()
    for index in (1, 2)
)
SYSTEM_FP = hashlib.sha256(b"rag-calibration-system").hexdigest()


def _execution(
    respondent_index: int = 1,
    task_index: int = 1,
    engine_id: str = "fixture_engine",
):
    """Return one deterministic governed RAG request/result/engine execution."""
    respondent_suffix = f"{respondent_index:03d}"
    task_suffix = f"{task_index:03d}"
    scenario_index = (respondent_index - 1) * 2 + (task_index - 1)
    grounded_scores = (2, 1, 0, 2)
    relevance_scores = (1, 2, 0, 1)
    request = build_rag_scoring_request(
        request_id=(
            f"rag_calibration_request_{respondent_suffix}_{task_suffix}"
        ),
        assessment=assessment(),
        rubric=rubric(),
        query_id="refund_policy_query",
        query_revision_fingerprint=QUERY_REVISION_FPS[task_index - 1],
        query_testlet_id="evidence_review",
        evidence_regime="retrieved_context",
        candidate_visibility="candidate_blind",
        system_configuration_id="retrieval_stack_a",
        system_configuration_fingerprint=SYSTEM_FP,
        system_run_id=f"retrieval_stack_a_run_{respondent_suffix}",
        response_id=f"generated_response_{respondent_suffix}_{task_suffix}",
        retrieval_run_fingerprint=hashlib.sha256(
            f"rag-calibration-retrieval-{respondent_suffix}-{task_suffix}".encode()
        ).hexdigest(),
        response_content_fingerprint=hashlib.sha256(
            f"rag-calibration-response-{respondent_suffix}-{task_suffix}".encode()
        ).hexdigest(),
        occasion_id="evaluation_wave_001",
        criterion_ids=("grounded_generation", "answer_relevance"),
        response_character_count=412 + scenario_index,
        response_unit_count=7,
        metadata={"evaluation_split": "offline_holdout"},
    )
    engine = automated_engine(engine_id=engine_id)
    fixture = StaticFixtureEngine(
        descriptor=engine,
        outcomes=(
            FixtureOutcome(
                criterion_id="grounded_generation",
                status=ObservationStatus.SCORED,
                score_category=grounded_scores[scenario_index],
            ),
            FixtureOutcome(
                criterion_id="answer_relevance",
                status=ObservationStatus.SCORED,
                score_category=relevance_scores[scenario_index],
            ),
        ),
    )
    return request, fixture.score(request), engine


def test_rag_facets_projection_reuses_shared_calibration_contracts() -> None:
    """RAG provenance projects into the shared facets record hierarchy only."""
    request, result, engine = _execution()

    records = build_rag_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )

    assert all(type(record) is ScoringFacetsRatingRecord for record in records)
    assert {record.criterion_id for record in records} == {
        "grounded_generation",
        "answer_relevance",
    }
    assert {record.respondent_id for record in records} == {request.respondent_id}
    assert {record.task_id for record in records} == {request.task_id}
    assert {record.task_revision_fingerprint for record in records} == {
        request.task_revision_fingerprint
    }
    assert {record.engine_fingerprint for record in records} == {
        engine.engine_fingerprint
    }


def test_rag_facets_bundle_delegates_to_shared_many_facet_design() -> None:
    """RAG orchestration returns one connected shared many-facet design."""
    executions = (
        _execution(1, 1, "fixture_engine"),
        _execution(1, 2, "alternate_engine"),
        _execution(2, 1, "alternate_engine"),
        _execution(2, 2, "fixture_engine"),
    )

    bundle = build_rag_facets_calibration_bundle(executions)

    assert type(bundle) is ScoringFacetsCalibrationBundle
    assert bundle.criterion_ids == ("answer_relevance", "grounded_generation")
    expected_revisions = tuple(sorted(QUERY_REVISION_FPS))
    for design in bundle.designs:
        assert design.respondent_ids == (
            "retrieval_stack_a_run_001",
            "retrieval_stack_a_run_002",
        )
        assert design.task_revision_fingerprints == expected_revisions
        assert design.task_ids == (
            "refund_policy_query",
            "refund_policy_query",
        )
        assert set(design.rater_engine_ids) == {
            "alternate_engine",
            "fixture_engine",
        }
        assert design.respondent_task_connected
        assert design.task_rater_connected
        assert design.connected
        assert {
            record.score_category
            for record in design.rating_records
            if record.status is ObservationStatus.SCORED
        } == {0, 1, 2}


def test_rag_facets_projection_rejects_non_rag_request() -> None:
    """A generic scoring execution cannot masquerade as governed RAG evidence."""
    generic_request = criterion_request()
    engine = automated_engine()
    result = StaticFixtureEngine(
        descriptor=engine,
        outcomes=(
            FixtureOutcome(
                criterion_id="claim_support",
                status=ObservationStatus.SCORED,
                score_category=2,
            ),
            FixtureOutcome(
                criterion_id="source_alignment",
                status=ObservationStatus.SCORED,
                score_category=1,
            ),
        ),
    ).score(generic_request)

    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_facets_rating_records(
            request=generic_request,
            result=result,
            engine=engine,
        )

    assert caught.value.code == "invalid_rag_scoring_request"


def test_rag_facets_surface_accepts_no_raw_rag_content() -> None:
    """Calibration orchestration remains identity-only at the public boundary."""
    forbidden = {
        "query_text",
        "question_text",
        "context_text",
        "retrieved_text",
        "answer_text",
        "response_text",
        "source_text",
    }
    for function in (
        build_rag_facets_rating_records,
        build_rag_facets_calibration_bundle,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
        assert function.__doc__
