"""Tests for governed criterion-level many-facet calibration handoffs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

import fast_mlsirm.scoring as scoring
import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    FixtureOutcome,
    ObservationStatus,
    StaticFixtureEngine,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
    fit_scoring_facets_bundle,
    fit_scoring_facets_design,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]
holistic_request = _FIXTURES["holistic_request"]
human_engine = _FIXTURES["human_engine"]


def assert_error(code: str, callback) -> None:
    """Assert one stable scoring-contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def execution(
    *,
    request_id: str,
    response_id: str,
    respondent_id: str,
    task_id: str,
    engine,
    claim_score: int | None,
    source_score: int | None,
    task_revision_fingerprint: str = "d" * 64,
    task_family_id: str = "evidence_review",
    claim_status: ObservationStatus = ObservationStatus.SCORED,
    source_status: ObservationStatus = ObservationStatus.SCORED,
    occasion_id: str = "initial_occasion",
):
    """Return one deterministic governed scoring execution."""
    request = criterion_request(
        request_id=request_id,
        response_id=response_id,
        respondent_id=respondent_id,
        task_id=task_id,
        task_revision_fingerprint=task_revision_fingerprint,
        task_family_id=task_family_id,
        occasion_id=occasion_id,
    )
    outcomes = tuple(
        FixtureOutcome(
            criterion_id=criterion_id,
            status=status,
            score_category=score,
            reason_code=(
                None
                if status is ObservationStatus.SCORED
                else "insufficient_evidence"
            ),
        )
        for criterion_id, status, score in (
            ("claim_support", claim_status, claim_score),
            ("source_alignment", source_status, source_score),
        )
    )
    fixture = StaticFixtureEngine(descriptor=engine, outcomes=outcomes)
    return request, fixture.score(request), engine


def project(execution_value) -> tuple[calibration.ScoringFacetsRatingRecord, ...]:
    """Project one execution into governed calibration records."""
    request, result, engine = execution_value
    return build_scoring_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )


def connected_records(*, terminal: bool = False, shared_task_id: bool = False):
    """Return a sparse connected respondent-revision-rater design."""
    records = []
    engines = (automated_engine(), human_engine())
    cells = (
        ("alpha", "alpha", "1" * 64, 0, (0, 1)),
        ("alpha", "beta", "2" * 64, 2, (0,)),
        ("beta", "alpha", "1" * 64, 1, (1,)),
        ("beta", "beta", "2" * 64, 0, (0, 1)),
    )
    for cell_index, (
        respondent_name,
        task_name,
        task_revision_fingerprint,
        base_score,
        rater_indexes,
    ) in enumerate(cells):
        for engine_index in rater_indexes:
            engine = engines[engine_index]
            claim_status = ObservationStatus.SCORED
            claim_score: int | None = (base_score + engine_index) % 3
            if terminal and cell_index == 0 and engine_index == 0:
                claim_status = ObservationStatus.ABSTAINED
                claim_score = None
            task_id = "shared_prompt" if shared_task_id else f"prompt_{task_name}"
            records.extend(
                project(
                    execution(
                        request_id=(
                            f"request_{respondent_name}_{task_name}_{engine_index}"
                        ),
                        response_id=f"response_{respondent_name}_{task_name}",
                        respondent_id=f"respondent_{respondent_name}",
                        task_id=task_id,
                        task_revision_fingerprint=task_revision_fingerprint,
                        engine=engine,
                        claim_score=claim_score,
                        source_score=(2 - base_score + 2 * engine_index) % 3,
                        claim_status=claim_status,
                    )
                )
            )
    return tuple(records)


def disconnected_records():
    """Return an auditable but respondent-revision-disconnected design."""
    records = []
    for respondent_name, task_name, revision, base_score in (
        ("component_one", "component_one", "3" * 64, 0),
        ("component_two", "component_two", "4" * 64, 2),
    ):
        for engine_index, engine in enumerate((automated_engine(), human_engine())):
            records.extend(
                project(
                    execution(
                        request_id=(
                            f"request_{respondent_name}_{task_name}_{engine_index}"
                        ),
                        response_id=f"response_{respondent_name}_{task_name}",
                        respondent_id=f"respondent_{respondent_name}",
                        task_id=f"prompt_{task_name}",
                        task_revision_fingerprint=revision,
                        engine=engine,
                        claim_score=(base_score + engine_index) % 3,
                        source_score=(2 - base_score + engine_index) % 3,
                    )
                )
            )
    return tuple(records)


def record_values(source=None) -> dict[str, Any]:
    """Return fields for a private-token rating invariant test."""
    source = source or connected_records()[0]
    return {
        "assessment_fingerprint": source.assessment_fingerprint,
        "rubric_fingerprint": source.rubric_fingerprint,
        "construct_id": source.construct_id,
        "request_fingerprint": source.request_fingerprint,
        "result_fingerprint": source.result_fingerprint,
        "observation_fingerprint": source.observation_fingerprint,
        "respondent_id": source.respondent_id,
        "response_id": source.response_id,
        "response_content_fingerprint": source.response_content_fingerprint,
        "task_id": source.task_id,
        "task_revision_fingerprint": source.task_revision_fingerprint,
        "task_family_id": source.task_family_id,
        "occasion_id": source.occasion_id,
        "criterion_id": source.criterion_id,
        "engine_id": source.engine_id,
        "engine_family_id": source.engine_family_id,
        "engine_fingerprint": source.engine_fingerprint,
        "status": source.status,
        "score_category": source.score_category,
        "allowed_scores": source.allowed_scores,
        "_rating_token": calibration._RATING_TOKEN,
    }


def direct_record(**overrides: Any):
    """Construct one record through the private token for branch testing."""
    values = record_values()
    values.update(overrides)
    return calibration.ScoringFacetsRatingRecord(**values)


def criterion_group(records, criterion_id=None):
    """Return one criterion group from a complete fixture."""
    selected = criterion_id or records[0].criterion_id
    return tuple(record for record in records if record.criterion_id == selected)


def test_public_surface_is_explicit_and_documented() -> None:
    """New APIs are explicit attributes without changing the pinned star surface."""
    names = {
        "MAX_SCORING_FACETS_CELLS",
        "MAX_SCORING_FACETS_RATINGS",
        "ScoringFacetsCalibrationBundle",
        "ScoringFacetsDesign",
        "ScoringFacetsRatingRecord",
        "build_scoring_facets_calibration_bundle",
        "build_scoring_facets_rating_records",
        "fit_scoring_facets_bundle",
        "fit_scoring_facets_design",
    }
    assert all(hasattr(scoring, name) for name in names)
    assert names.isdisjoint(scoring.__all__)
    assert all(getattr(scoring, name).__doc__ for name in names if name[0].isupper())


def test_projection_retains_exact_provenance_and_terminal_states() -> None:
    """A matched result becomes sorted, content-addressed criterion records."""
    request, result, engine = execution(
        request_id="projection_request",
        response_id="projection_response",
        respondent_id="projection_respondent",
        task_id="projection_prompt",
        task_revision_fingerprint="9" * 64,
        engine=automated_engine(),
        claim_score=2,
        source_score=None,
        source_status=ObservationStatus.ABSTAINED,
    )
    records = build_scoring_facets_rating_records(
        request=request,
        result=result,
        engine=engine,
    )
    assert tuple(record.criterion_id for record in records) == tuple(
        sorted(("claim_support", "source_alignment"))
    )
    scored = next(record for record in records if record.status is ObservationStatus.SCORED)
    terminal = next(
        record for record in records if record.status is ObservationStatus.ABSTAINED
    )
    assert scored.score_category == 2
    assert terminal.score_category is None
    assert scored.request_fingerprint == request.request_fingerprint
    assert scored.result_fingerprint == result.result_fingerprint
    assert scored.engine_fingerprint == engine.engine_fingerprint
    assert scored.response_content_fingerprint == request.response_content_fingerprint
    assert scored.task_revision_fingerprint == request.task_revision_fingerprint
    assert scored.task_family_id == request.task_family_id
    assert scored.rating_handle == f"scoring_facets_rating_{scored.rating_fingerprint[:32]}"
    assert scored.to_dict()["allowed_scores"] == [0, 1, 2]


def test_bundle_is_deterministic_and_preserves_sparse_state() -> None:
    """Order, absent cells, and terminal states remain auditable."""
    records = connected_records(terminal=True)
    first = build_scoring_facets_calibration_bundle(records)
    second = build_scoring_facets_calibration_bundle(reversed(records))
    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first.bundle_handle.startswith("scoring_facets_bundle_")
    assert first.criterion_ids == ("claim_support", "source_alignment")
    assert first.to_dict()["criterion_ids"] == list(first.criterion_ids)
    assert set(first.design_by_criterion()) == set(first.criterion_ids)

    design = first.design_by_criterion()["claim_support"]
    normalized = design.responses_array()
    original = design.original_scores_array()
    states = design.response_states()
    assert normalized.shape == original.shape == (2, 2, 2)
    assert np.isnan(normalized).sum() == 3
    assert np.isnan(original).sum() == 3
    assert set(normalized[np.isfinite(normalized)]) <= {0.0, 1.0, 2.0}
    assert any(
        state is ObservationStatus.ABSTAINED
        for respondent_states in states
        for task_states in respondent_states
        for state in task_states
    )
    assert any(
        state is None
        for respondent_states in states
        for task_states in respondent_states
        for state in task_states
    )
    normalized[0, 0, 0] = 99
    assert not np.any(design.responses_array() == 99)
    assert design.to_fit_facets_kwargs()["n_cat"] == 3
    assert design.design_handle.startswith("scoring_facets_design_")
    assert design.to_dict()["connected"] is True


def test_noncontiguous_scores_map_only_at_estimator_boundary() -> None:
    """Original labels remain visible while estimator categories are zero-based."""
    remapped = []
    for record in connected_records():
        score = None if record.score_category is None else (1, 3, 5)[record.score_category]
        remapped.append(
            replace(
                record,
                score_category=score,
                allowed_scores=(1, 3, 5),
                _rating_token=calibration._RATING_TOKEN,
            )
        )
    design = build_scoring_facets_calibration_bundle(remapped).designs[0]
    original = design.original_scores_array()
    normalized = design.responses_array()
    assert design.category_values == (1, 3, 5)
    assert set(original[np.isfinite(original)]) <= {1.0, 3.0, 5.0}
    assert set(normalized[np.isfinite(normalized)]) <= {0.0, 1.0, 2.0}


def test_fit_helpers_delegate_every_numeric_operation(monkeypatch) -> None:
    """Fitting passes fresh arrays and tuning values to the Rust-backed API."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    calls = []

    def fake_fit_facets(**kwargs):
        calls.append(kwargs)
        return {"shape": kwargs["responses"].shape, "n_cat": kwargs["n_cat"]}

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", fake_fit_facets)
    one = fit_scoring_facets_design(
        bundle.designs[0], q_theta=21, max_iter=77, tol=1e-5
    )
    fitted = fit_scoring_facets_bundle(
        bundle, q_theta=15, max_iter=66, tol=1e-4
    )
    assert one == {"shape": (2, 2, 2), "n_cat": 3}
    assert set(fitted) == set(bundle.criterion_ids)
    assert calls[0]["q_theta"] == 21
    assert calls[0]["max_iter"] == 77
    assert calls[0]["tol"] == 1e-5
    assert all(call["n_cat"] == 3 for call in calls)


def test_disconnected_design_requires_assembly_and_rejects_fit() -> None:
    """Disconnected designs may be audited but never enter the estimator."""
    records = disconnected_records()
    assert_error(
        "unidentified_respondent_task_design",
        lambda: build_scoring_facets_calibration_bundle(records),
    )
    bundle = build_scoring_facets_calibration_bundle(
        records,
        require_connected=False,
    )
    assert all(not design.connected for design in bundle.designs)
    for allow_disconnected in (False, True):
        assert_error(
            "unidentified_respondent_task_design",
            lambda allow_disconnected=allow_disconnected: fit_scoring_facets_design(
                bundle.designs[0], allow_disconnected=allow_disconnected
            ),
        )
    assert_error(
        "invalid_require_connected",
        lambda: build_scoring_facets_calibration_bundle(records, require_connected=1),
    )
    assert_error(
        "invalid_allow_disconnected",
        lambda: fit_scoring_facets_design(
            bundle.designs[0], allow_disconnected=1
        ),
    )


def test_projection_rejects_types_granularity_and_mismatches() -> None:
    """Only matched criterion-level request/result/engine triples are accepted."""
    request, result, engine = execution(
        request_id="validation_request",
        response_id="validation_response",
        respondent_id="validation_respondent",
        task_id="validation_prompt",
        task_revision_fingerprint="7" * 64,
        engine=automated_engine(),
        claim_score=0,
        source_score=1,
    )
    assert_error(
        "invalid_scoring_request",
        lambda: build_scoring_facets_rating_records(
            request=object(), result=result, engine=engine
        ),
    )
    assert_error(
        "invalid_scoring_result",
        lambda: build_scoring_facets_rating_records(
            request=request, result=object(), engine=engine
        ),
    )
    assert_error(
        "invalid_engine_descriptor",
        lambda: build_scoring_facets_rating_records(
            request=request, result=result, engine=object()
        ),
    )

    holistic = holistic_request(
        request_id="holistic_validation_request",
        response_id="holistic_validation_response",
        respondent_id="holistic_validation_respondent",
        task_id="holistic_validation_prompt",
        task_revision_fingerprint="8" * 64,
    )
    holistic_result = StaticFixtureEngine(
        descriptor=engine,
        outcomes=(
            FixtureOutcome(
                criterion_id=None,
                status=ObservationStatus.SCORED,
                score_category=1,
            ),
        ),
    ).score(holistic)
    assert_error(
        "unsupported_calibration_granularity",
        lambda: build_scoring_facets_rating_records(
            request=holistic, result=holistic_result, engine=engine
        ),
    )
    assert_error(
        "unsupported_calibration_granularity",
        lambda: build_scoring_facets_rating_records(
            request=request, result=holistic_result, engine=engine
        ),
    )

    other_request, other_result, _ = execution(
        request_id="other_validation_request",
        response_id="other_validation_response",
        respondent_id="other_validation_respondent",
        task_id="other_validation_prompt",
        task_revision_fingerprint="6" * 64,
        engine=engine,
        claim_score=1,
        source_score=2,
    )
    assert_error(
        "calibration_request_result_mismatch",
        lambda: build_scoring_facets_rating_records(
            request=request, result=other_result, engine=engine
        ),
    )
    assert_error(
        "calibration_engine_result_mismatch",
        lambda: build_scoring_facets_rating_records(
            request=other_request,
            result=other_result,
            engine=human_engine(),
        ),
    )


def test_rating_guards_direct_construction_status_score_and_scale() -> None:
    """Malformed package-owned records fail at stable invariant boundaries."""
    source = connected_records()[0]
    with pytest.raises(AssessmentSpecError) as caught:
        replace(source)
    assert caught.value.code == "unverified_facets_rating"
    assert_error(
        "invalid_observation_status",
        lambda: direct_record(status="unknown_status"),
    )
    assert_error(
        "missing_rating_score",
        lambda: direct_record(score_category=None),
    )
    assert_error(
        "unexpected_rating_score",
        lambda: direct_record(
            status=ObservationStatus.ABSTAINED,
            score_category=1,
        ),
    )
    assert_error(
        "rating_score_out_of_range",
        lambda: direct_record(score_category=9),
    )
    assert_error(
        "invalid_allowed_scores",
        lambda: direct_record(allowed_scores=(0, True, 2)),
    )
    assert_error(
        "invalid_allowed_scores",
        lambda: direct_record(allowed_scores=(0, 2, 1)),
    )


def test_bundle_rejects_collection_duplicates_and_mixed_contracts() -> None:
    """Collection validation fails before criterion design allocation."""
    records = connected_records()
    assert_error(
        "invalid_records",
        lambda: build_scoring_facets_calibration_bundle("not_records"),
    )
    assert_error(
        "invalid_facets_rating_record",
        lambda: build_scoring_facets_calibration_bundle((*records, object())),
    )
    assert_error(
        "duplicate_facets_rating_record",
        lambda: build_scoring_facets_calibration_bundle((*records, records[0])),
    )
    mixed = replace(
        records[0],
        occasion_id="secondary_occasion",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "mixed_facets_calibration_contract",
        lambda: build_scoring_facets_calibration_bundle((mixed, *records[1:])),
    )


def test_design_rejects_duplicate_response_and_rater_provenance() -> None:
    """Cell, response, and engine identities cannot be rebound."""
    group = list(criterion_group(connected_records()))
    first, second = group[0], group[1]

    duplicate_cell = replace(
        second,
        observation_fingerprint="e" * 64,
        result_fingerprint="f" * 64,
        response_id=first.response_id,
        respondent_id=first.respondent_id,
        task_id=first.task_id,
        task_revision_fingerprint=first.task_revision_fingerprint,
        task_family_id=first.task_family_id,
        response_content_fingerprint=first.response_content_fingerprint,
        engine_id=first.engine_id,
        engine_family_id=first.engine_family_id,
        engine_fingerprint=first.engine_fingerprint,
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "duplicate_facets_rating_cell",
        lambda: calibration._build_criterion_design(
            (first, duplicate_cell), require_connected=False
        ),
    )

    response_conflict = replace(
        second,
        response_id=first.response_id,
        respondent_id="conflicting_respondent",
        response_content_fingerprint=first.response_content_fingerprint,
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "response_provenance_conflict",
        lambda: calibration._build_criterion_design(
            (first, response_conflict), require_connected=False
        ),
    )

    rater_conflict = replace(
        second,
        engine_fingerprint=first.engine_fingerprint,
        engine_id="conflicting_engine",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "rater_provenance_conflict",
        lambda: calibration._build_criterion_design(
            (first, rater_conflict), require_connected=False
        ),
    )


def test_design_rejects_insufficient_support_and_dense_amplification(monkeypatch) -> None:
    """Every estimable axis needs observed support before bounded allocation."""
    group = criterion_group(connected_records())

    one_respondent = tuple(
        record for record in group if record.respondent_id == group[0].respondent_id
    )
    assert_error(
        "insufficient_facets_respondents",
        lambda: calibration._build_criterion_design(
            one_respondent, require_connected=False
        ),
    )

    one_task = tuple(
        record
        for record in group
        if record.task_revision_fingerprint == group[0].task_revision_fingerprint
    )
    assert_error(
        "insufficient_facets_tasks",
        lambda: calibration._build_criterion_design(
            one_task, require_connected=False
        ),
    )

    one_rater = tuple(
        record
        for record in group
        if record.engine_fingerprint == group[0].engine_fingerprint
    )
    assert_error(
        "insufficient_facets_raters",
        lambda: calibration._build_criterion_design(
            one_rater, require_connected=False
        ),
    )

    target_respondent = group[0].respondent_id
    terminal_group = tuple(
        replace(
            record,
            status=ObservationStatus.ABSTAINED,
            score_category=None,
            _rating_token=calibration._RATING_TOKEN,
        )
        if record.respondent_id == target_respondent
        else record
        for record in group
    )
    assert_error(
        "unobserved_facets_level",
        lambda: calibration._build_criterion_design(
            terminal_group, require_connected=False
        ),
    )

    one_category = tuple(
        replace(
            record,
            score_category=0,
            _rating_token=calibration._RATING_TOKEN,
        )
        for record in group
    )
    assert_error(
        "single_observed_category",
        lambda: calibration._build_criterion_design(
            one_category, require_connected=False
        ),
    )

    monkeypatch.setattr(calibration, "MAX_SCORING_FACETS_CELLS", 1)
    assert_error(
        "facets_cell_budget_exceeded",
        lambda: calibration._build_criterion_design(
            group, require_connected=False
        ),
    )


def test_factory_and_fit_type_guards() -> None:
    """Design, bundle, and fit entry points reject unverified values."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    with pytest.raises(AssessmentSpecError) as design_error:
        replace(bundle.designs[0])
    assert design_error.value.code == "unverified_facets_design"
    with pytest.raises(AssessmentSpecError) as bundle_error:
        replace(bundle)
    assert bundle_error.value.code == "unverified_facets_bundle"
    assert_error(
        "invalid_facets_design",
        lambda: fit_scoring_facets_design(object()),
    )
    assert_error(
        "invalid_facets_bundle",
        lambda: fit_scoring_facets_bundle(object()),
    )
