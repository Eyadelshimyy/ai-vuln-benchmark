"""Regression tests for reviewed scoring-facets calibration invariants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    ObservationStatus,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
    fit_scoring_facets_bundle,
    fit_scoring_facets_design,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
assert_error = _BASE["assert_error"]
connected_records = _BASE["connected_records"]
disconnected_records = _BASE["disconnected_records"]
execution = _BASE["execution"]


def _four_category_records(
    observed_scores: tuple[int, ...],
) -> tuple[calibration.ScoringFacetsRatingRecord, ...]:
    """Return a connected four-category pilot with selected scored categories."""
    counters: dict[str, int] = {}
    output = []
    for record in connected_records():
        index = counters.get(record.criterion_id, 0)
        counters[record.criterion_id] = index + 1
        output.append(
            replace(
                record,
                score_category=observed_scores[index % len(observed_scores)],
                allowed_scores=(0, 1, 2, 3),
                _rating_token=calibration._RATING_TOKEN,
            )
        )
    return tuple(output)


def test_bundle_fit_forwards_bundle_specific_tuning_values(monkeypatch) -> None:
    """Every criterion fit receives the bundle call's explicit tuning values."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    calls = []

    def fake_fit_facets(**kwargs):
        calls.append(kwargs)
        return kwargs["responses"].shape

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", fake_fit_facets)
    fitted = fit_scoring_facets_bundle(
        bundle,
        q_theta=15,
        max_iter=66,
        tol=1e-4,
    )

    assert set(fitted) == set(bundle.criterion_ids)
    assert len(calls) == len(bundle.designs)
    assert all(call["q_theta"] == 15 for call in calls)
    assert all(call["max_iter"] == 66 for call in calls)
    assert all(call["tol"] == 1e-4 for call in calls)
    assert all(call["n_cat"] == 3 for call in calls)
    response_objects = [id(call["responses"]) for call in calls]
    assert len(set(response_objects)) == len(response_objects)


@pytest.mark.parametrize("observed_scores", [(0, 1), (0, 1, 2)])
def test_sparse_declared_categories_build_but_fail_before_rust_fitting(
    observed_scores: tuple[int, ...],
    monkeypatch,
) -> None:
    """Two- and three-category pilots remain auditable but not estimable."""
    bundle = build_scoring_facets_calibration_bundle(
        _four_category_records(observed_scores)
    )

    def unexpected_fit_facets(**_kwargs):
        raise AssertionError("Rust estimator delegation must not occur")

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", unexpected_fit_facets)
    for design in bundle.designs:
        assert design.category_values == (0, 1, 2, 3)
        assert {
            record.score_category
            for record in design.rating_records
            if record.status is ObservationStatus.SCORED
        } == set(observed_scores)
        assert_error(
            "unobserved_facets_category",
            design.to_fit_facets_kwargs,
        )
        assert_error(
            "unobserved_facets_category",
            lambda design=design: fit_scoring_facets_design(design),
        )
    assert_error(
        "unobserved_facets_category",
        lambda: fit_scoring_facets_bundle(bundle),
    )


def test_disconnected_error_precedes_every_estimator_gate() -> None:
    """A disconnected audit artifact never reaches category or Rust fit gates."""
    records = tuple(
        replace(
            record,
            allowed_scores=(0, 1, 2, 3),
            _rating_token=calibration._RATING_TOKEN,
        )
        for record in disconnected_records()
    )
    bundle = build_scoring_facets_calibration_bundle(
        records,
        require_connected=False,
    )

    for design in bundle.designs:
        assert not design.connected
        for allow_disconnected in (False, True):
            assert_error(
                "unidentified_respondent_task_design",
                lambda design=design, allow_disconnected=allow_disconnected: (
                    fit_scoring_facets_design(
                        design,
                        allow_disconnected=allow_disconnected,
                    )
                ),
            )


def test_rating_fingerprint_tracks_current_normalized_content() -> None:
    """In-process content mutation cannot leave a stale provenance fingerprint."""
    record = connected_records()[0]
    original = record.rating_fingerprint

    object.__setattr__(record, "task_id", "mutated_task")

    assert record.rating_fingerprint != original
    assert record.to_dict()["rating_fingerprint"] == record.rating_fingerprint


def test_missing_criterion_has_a_specific_fail_closed_error() -> None:
    """A corrupted criterion-level result cannot degrade to an identifier error."""
    request, result, engine = execution(
        request_id="missing_criterion_request",
        response_id="missing_criterion_response",
        respondent_id="missing_criterion_respondent",
        task_id="missing_criterion_prompt",
        engine=_BASE["automated_engine"](),
        claim_score=0,
        source_score=1,
    )
    object.__setattr__(result.observations[0], "criterion_id", None)
    assert_error(
        "missing_observation_criterion",
        lambda: build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        ),
    )


def test_bundle_rejects_cross_criterion_response_provenance_conflict() -> None:
    """One response identity cannot be rebound only in another criterion group."""
    records = list(connected_records())
    claim = next(record for record in records if record.criterion_id == "claim_support")
    source_index = next(
        index
        for index, record in enumerate(records)
        if record.criterion_id == "source_alignment"
        and record.response_id != claim.response_id
    )
    records[source_index] = replace(
        records[source_index],
        response_id=claim.response_id,
        respondent_id="conflicting_respondent",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "response_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(records),
    )


def test_bundle_rejects_cross_criterion_rater_provenance_conflict() -> None:
    """One engine fingerprint cannot name another rater in a second criterion."""
    records = list(connected_records())
    claim = next(record for record in records if record.criterion_id == "claim_support")
    source_index = next(
        index
        for index, record in enumerate(records)
        if record.criterion_id == "source_alignment"
        and record.engine_fingerprint != claim.engine_fingerprint
    )
    records[source_index] = replace(
        records[source_index],
        engine_fingerprint=claim.engine_fingerprint,
        engine_id="conflicting_engine",
        _rating_token=calibration._RATING_TOKEN,
    )
    assert_error(
        "rater_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(records),
    )


def test_all_terminal_states_do_not_satisfy_category_observation() -> None:
    """Abstained, failed, and excluded records provide no threshold evidence."""
    terminal_by_position = {
        ("claim_support", 0): ObservationStatus.ABSTAINED,
        ("claim_support", 1): ObservationStatus.FAILED,
        ("source_alignment", 0): ObservationStatus.EXCLUDED,
    }
    counters: dict[str, int] = {}
    records = []
    for record in _four_category_records((0, 1, 2)):
        index = counters.get(record.criterion_id, 0)
        counters[record.criterion_id] = index + 1
        status = terminal_by_position.get((record.criterion_id, index))
        if status is None:
            records.append(record)
        else:
            records.append(
                replace(
                    record,
                    status=status,
                    score_category=None,
                    _rating_token=calibration._RATING_TOKEN,
                )
            )
    bundle = build_scoring_facets_calibration_bundle(records)

    terminal_records = [
        record
        for record in records
        if record.status is not ObservationStatus.SCORED
    ]
    assert {record.status for record in terminal_records} == {
        ObservationStatus.ABSTAINED,
        ObservationStatus.FAILED,
        ObservationStatus.EXCLUDED,
    }
    assert all(record.score_category is None for record in terminal_records)
    for design in bundle.designs:
        assert design.respondent_task_connected is True
        assert design.task_rater_connected is True
        assert {
            record.score_category
            for record in design.rating_records
            if record.status is ObservationStatus.SCORED
        } == {0, 1, 2}
        assert_error(
            "unobserved_facets_category",
            design.to_fit_facets_kwargs,
        )
        assert_error(
            "unobserved_facets_category",
            lambda design=design: fit_scoring_facets_design(design),
        )
