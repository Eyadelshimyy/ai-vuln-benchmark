"""Contracts for provenance-safe pilot observations and facets handoff."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_mlsirm.config import MAX_POLYTOMOUS_CATEGORIES
from fast_mlsirm.rubric import (
    FacetsPilotDesign,
    PilotItemProvenance,
    PilotObservationError,
    PilotObservationRecord,
    PilotResponseState,
    audit_generated_item_candidate,
    build_facets_pilot_design,
    build_pilot_candidate_record,
    build_pilot_observation_record,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)
_candidate = _FIXTURES["_candidate"]
_base_pilot_kwargs = _FIXTURES["_pilot_kwargs"]


def _pilot(
    item_id: str = "generated_item_alpha",
    **pilot_overrides: str,
):
    """Return one replay-verified pilot record with configurable provenance."""

    def mutate(payload):
        payload["item_id"] = item_id

    candidate = _candidate(mutate=mutate)
    report = audit_generated_item_candidate(candidate)
    pilot_kwargs = {**_base_pilot_kwargs(), **pilot_overrides}
    return build_pilot_candidate_record(candidate, report, **pilot_kwargs)


def _observation(
    pilot=None,
    *,
    respondent_id: str = "respondent_alpha",
    rater_id: str = "rater_alpha",
    response_state: PilotResponseState | str = PilotResponseState.OBSERVED,
    category: int | None = 1,
) -> PilotObservationRecord:
    """Build one observation through the production provenance boundary."""
    return build_pilot_observation_record(
        pilot or _pilot(),
        respondent_id=respondent_id,
        rater_id=rater_id,
        response_state=response_state,
        category=category,
    )


def test_observation_builder_binds_complete_pilot_provenance_deterministically():
    """Observed categories retain all admission identifiers and stable hashes."""
    pilot = _pilot()
    first = _observation(pilot)
    second = _observation(pilot)

    assert first == second
    assert first.pilot_study_id == pilot.pilot_study_id
    assert first.item_id == pilot.item_id
    assert first.pilot_record_fingerprint == pilot.pilot_record_fingerprint
    assert first.response_state is PilotResponseState.OBSERVED
    assert first.category == 1
    assert first.observation_id.startswith("pilot_observation_")
    assert first.to_dict()["observation_fingerprint"] == first.observation_fingerprint
    assert len(first.observation_fingerprint) == 64

    with pytest.raises(ValueError, match="build_pilot_observation_record"):
        PilotObservationRecord(**first.__dict__)


def test_nonobserved_states_are_preserved_without_numeric_coercion():
    """Missing, N/A, and insufficient-evidence cells remain distinct and unscored."""
    for state in (
        PilotResponseState.MISSING,
        PilotResponseState.NOT_APPLICABLE,
        PilotResponseState.INSUFFICIENT_EVIDENCE,
    ):
        record = _observation(response_state=state, category=None)
        assert record.response_state is state
        assert record.category is None
        assert record.to_dict()["response_state"] == state.value


def test_observation_builder_rejects_invalid_sources_states_and_categories():
    """Only verified pilots and bounded observed categories may create records."""
    with pytest.raises(TypeError, match="verified PilotCandidateRecord"):
        build_pilot_observation_record(
            object(),
            respondent_id="respondent_alpha",
            rater_id="rater_alpha",
            response_state="observed",
            category=1,
        )
    with pytest.raises(ValueError, match="respondent_id"):
        _observation(respondent_id="respondent")
    with pytest.raises(ValueError, match="rater_id"):
        _observation(rater_id="rater")
    with pytest.raises(ValueError, match="response_state"):
        _observation(response_state="unknown_state")
    with pytest.raises(ValueError, match="category must be an integer"):
        _observation(category=None)
    with pytest.raises(ValueError, match="category must be an integer"):
        _observation(category=True)
    with pytest.raises(ValueError, match="category must be between"):
        _observation(category=-1)
    with pytest.raises(ValueError, match="category must be between"):
        _observation(category=MAX_POLYTOMOUS_CATEGORIES)
    with pytest.raises(ValueError, match="category must be None"):
        _observation(response_state="missing", category=0)


def test_structured_error_and_item_provenance_validation_fail_closed():
    """Structured errors and item metadata reject malformed public identities."""
    error = PilotObservationError(
        "invalid_observation",
        "$.records[0]",
        "observation is invalid",
    )
    assert error.code == "invalid_observation"
    assert error.path == "$.records[0]"
    assert str(error).startswith("invalid_observation at $.records[0]")

    with pytest.raises(ValueError, match="path must be"):
        PilotObservationError("invalid_observation", "records", "invalid")
    with pytest.raises(ValueError, match="path must be"):
        PilotObservationError("invalid_observation", 7, "invalid")
    with pytest.raises(ValueError, match="pilot_record_fingerprint"):
        PilotItemProvenance(
            item_id="generated_item_alpha",
            pilot_record_fingerprint="bad",
            query_testlet_id="query_testlet_alpha",
            generator_family_id="generator_family_alpha",
            judge_policy_id="judge_policy_alpha",
            occasion_id="occasion_window_alpha",
        )


def test_facets_design_is_deterministic_and_preserves_nonobserved_states():
    """Record order does not affect tensor order, state retention, or identity."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot("generated_item_beta", query_testlet_id="query_testlet_beta")
    records = (
        _observation(
            item_beta,
            respondent_id="respondent_alpha",
            rater_id="rater_alpha",
            category=1,
        ),
        _observation(
            item_alpha,
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=2,
        ),
        _observation(
            item_alpha,
            respondent_id="respondent_alpha",
            rater_id="rater_alpha",
            category=0,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            response_state=PilotResponseState.INSUFFICIENT_EVIDENCE,
            category=None,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            rater_id="rater_alpha",
            response_state=PilotResponseState.NOT_APPLICABLE,
            category=None,
        ),
    )
    design = build_facets_pilot_design(records)
    reordered = build_facets_pilot_design(reversed(records))

    assert design == reordered
    assert design.design_fingerprint == reordered.design_fingerprint
    assert design.design_id.startswith("facets_pilot_design_")
    assert design.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert design.item_ids == ("generated_item_alpha", "generated_item_beta")
    assert design.rater_ids == ("rater_alpha", "rater_beta")
    assert design.n_cat == 3
    assert design.response_states[1][1][1] is PilotResponseState.INSUFFICIENT_EVIDENCE
    assert design.response_states[1][1][0] is PilotResponseState.NOT_APPLICABLE
    assert design.response_states[0][0][1] is PilotResponseState.MISSING
    assert design.to_dict()["design_fingerprint"] == design.design_fingerprint

    responses = design.responses_array()
    assert responses.shape == (2, 2, 2)
    assert responses.dtype == np.float64
    assert responses[0, 0, 0] == 0.0
    assert responses[0, 1, 0] == 1.0
    assert responses[1, 0, 1] == 2.0
    assert np.isnan(responses[1, 1, 1])
    assert np.isnan(responses[0, 0, 1])

    kwargs = design.to_fit_facets_kwargs()
    assert kwargs["n_cat"] == 3
    np.testing.assert_equal(kwargs["responses"], responses)
    kwargs["responses"][0, 0, 0] = 2.0
    assert design.responses_array()[0, 0, 0] == 0.0

    with pytest.raises(ValueError, match="build_facets_pilot_design"):
        FacetsPilotDesign(**design.__dict__)


def test_design_rejects_invalid_record_collections_and_mixed_studies():
    """Batch assembly is bounded to typed records from one declared pilot."""
    valid = _observation()
    with pytest.raises(ValueError, match="at least 1 value"):
        build_facets_pilot_design(())
    with pytest.raises(ValueError, match="records must be a collection"):
        build_facets_pilot_design("not_records")
    with pytest.raises(TypeError, match=r"records\[1\]"):
        build_facets_pilot_design((valid, object()))

    other_study = _observation(
        _pilot(pilot_study_id="pilot_study_beta"),
        respondent_id="respondent_beta",
        category=0,
    )
    with pytest.raises(PilotObservationError) as caught:
        build_facets_pilot_design((valid, other_study), n_cat=2)
    assert caught.value.code == "mixed_pilot_study"


def test_design_rejects_conflicting_item_provenance_and_duplicate_cells():
    """One item and one response cell cannot be rebound within a design."""
    original = _observation(category=0)
    conflicting = _observation(
        _pilot(query_testlet_id="query_testlet_beta"),
        respondent_id="respondent_beta",
        category=1,
    )
    with pytest.raises(PilotObservationError) as conflict:
        build_facets_pilot_design((original, conflicting), n_cat=2)
    assert conflict.value.code == "item_provenance_conflict"

    duplicate = _observation(category=1)
    with pytest.raises(PilotObservationError) as duplicate_error:
        build_facets_pilot_design((original, duplicate), n_cat=2)
    assert duplicate_error.value.code == "duplicate_observation_cell"


def test_design_category_contracts_fail_closed():
    """Category inference and declarations cannot admit degenerate or out-of-range data."""
    missing = _observation(response_state="missing", category=None)
    with pytest.raises(PilotObservationError) as no_observed:
        build_facets_pilot_design((missing,))
    assert no_observed.value.code == "no_observed_response"

    zero_only = _observation(category=0)
    with pytest.raises(PilotObservationError) as one_category:
        build_facets_pilot_design((zero_only,))
    assert one_category.value.code == "single_category_design"

    with pytest.raises(ValueError, match="n_cat must be an integer"):
        build_facets_pilot_design((_observation(category=1),), n_cat=True)
    with pytest.raises(ValueError, match="n_cat must be between"):
        build_facets_pilot_design((_observation(category=1),), n_cat=1)
    with pytest.raises(ValueError, match="n_cat must be between"):
        build_facets_pilot_design(
            (_observation(category=1),),
            n_cat=MAX_POLYTOMOUS_CATEGORIES + 1,
        )
    with pytest.raises(PilotObservationError) as outside:
        build_facets_pilot_design((_observation(category=2),), n_cat=2)
    assert outside.value.code == "category_out_of_range"


def test_design_requires_observed_support_for_each_indexed_facet():
    """Respondents, items, and raters without observations are rejected before Rust."""
    observed = _observation(category=1)

    missing_respondent = _observation(
        respondent_id="respondent_beta",
        response_state="missing",
        category=None,
    )
    with pytest.raises(PilotObservationError) as respondent_error:
        build_facets_pilot_design((observed, missing_respondent), n_cat=2)
    assert respondent_error.value.code == "unobserved_respondent"

    missing_item = _observation(
        _pilot("generated_item_beta", query_testlet_id="query_testlet_beta"),
        response_state="missing",
        category=None,
    )
    with pytest.raises(PilotObservationError) as item_error:
        build_facets_pilot_design((observed, missing_item), n_cat=2)
    assert item_error.value.code == "unobserved_item"

    missing_rater = _observation(
        rater_id="rater_beta",
        response_state="missing",
        category=None,
    )
    with pytest.raises(PilotObservationError) as rater_error:
        build_facets_pilot_design((observed, missing_rater), n_cat=2)
    assert rater_error.value.code == "unobserved_rater"


def test_same_item_provenance_may_appear_across_distinct_cells():
    """Repeated item metadata is allowed when respondent-rater cells are unique."""
    records = (
        _observation(
            respondent_id="respondent_alpha",
            rater_id="rater_alpha",
            category=0,
        ),
        _observation(
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=1,
        ),
    )
    design = build_facets_pilot_design(records)
    assert design.responses == (((0, None),), ((None, 1),))
    assert design.n_cat == 2
