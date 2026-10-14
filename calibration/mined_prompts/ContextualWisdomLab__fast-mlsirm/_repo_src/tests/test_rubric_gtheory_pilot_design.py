"""Contracts for the generated-item one-facet G-theory pilot handoff."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_mlsirm import gtheory
import fast_mlsirm.rubric as rubric
import fast_mlsirm.rubric.gtheory_pilot as gtheory_pilot
from fast_mlsirm.rubric import (
    GTheoryPiPilotDesign,
    PilotObservationError,
    PilotResponseState,
    build_gtheory_pi_pilot_design,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _complete_records():
    """Return a deterministic complete 2-person by 2-item pilot design."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot(
        "generated_item_beta",
        query_testlet_id="query_testlet_beta",
    )
    return (
        _observation(
            item_alpha,
            respondent_id="respondent_alpha",
            rater_id="rater_alpha",
            category=0,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_alpha",
            rater_id="rater_alpha",
            category=1,
        ),
        _observation(
            item_alpha,
            respondent_id="respondent_beta",
            rater_id="rater_alpha",
            category=2,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            rater_id="rater_alpha",
            category=1,
        ),
    )


def test_gtheory_pi_design_is_deterministic_and_preserves_provenance():
    """Record order cannot change the disclosed score matrix or design identity."""
    records = _complete_records()
    first = build_gtheory_pi_pilot_design(records)
    second = build_gtheory_pi_pilot_design(tuple(reversed(records)))

    assert first == second
    assert first.design_fingerprint == second.design_fingerprint
    assert first.design_id.startswith("gtheory_pi_pilot_design_")
    assert first.pilot_study_id == "pilot_study_alpha"
    assert first.rater_id == "rater_alpha"
    assert first.occasion_id == "occasion_window_alpha"
    assert first.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert first.item_ids == ("generated_item_alpha", "generated_item_beta")
    assert tuple(entry.item_id for entry in first.item_provenance) == first.item_ids
    assert first.scores == ((0, 1), (2, 1))

    payload = first.to_dict()
    assert payload["design_id"] == first.design_id
    assert payload["design_fingerprint"] == first.design_fingerprint
    assert payload["facets_design"]["design_id"].startswith("facets_pilot_design_")


def test_gtheory_pi_and_phi_lambda_kwargs_match_public_rust_backed_apis():
    """The handoff emits fresh arrays and bounded D-study settings only."""
    design = build_gtheory_pi_pilot_design(_complete_records())

    pi_kwargs = design.to_gtheory_pi_kwargs((2, 4, 8))
    assert set(pi_kwargs) <= set(inspect.signature(gtheory.gtheory_pi).parameters)
    assert pi_kwargs["n_i_prime"] == (2, 4, 8)
    assert pi_kwargs["data"].dtype == np.float64
    np.testing.assert_array_equal(
        pi_kwargs["data"],
        np.asarray([[0.0, 1.0], [2.0, 1.0]], dtype=np.float64),
    )

    phi_kwargs = design.to_phi_lambda_kwargs(1.5, (3, 6))
    assert set(phi_kwargs) <= set(inspect.signature(gtheory.phi_lambda).parameters)
    assert phi_kwargs["cut"] == 1.5
    assert phi_kwargs["n_i_prime"] == (3, 6)
    np.testing.assert_array_equal(phi_kwargs["data"], pi_kwargs["data"])

    first = design.scores_array()
    second = design.scores_array()
    assert first is not second
    first[0, 0] = 99.0
    assert design.scores_array()[0, 0] == 0.0


def test_gtheory_pi_design_rejects_rater_and_occasion_confounding():
    """One-facet labels cannot silently absorb rater or occasion variation."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot(
        "generated_item_beta",
        query_testlet_id="query_testlet_beta",
    )
    multi_rater = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=0),
        _observation(item_beta, respondent_id="respondent_alpha", category=1),
        _observation(
            item_alpha,
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=1,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=0,
        ),
    )
    with pytest.raises(PilotObservationError) as rater_error:
        build_gtheory_pi_pilot_design(multi_rater)
    assert rater_error.value.code == "gtheory_pi_rater_confounded"

    other_occasion_item = _pilot(
        "generated_item_beta",
        query_testlet_id="query_testlet_beta",
        occasion_id="occasion_window_beta",
    )
    mixed_occasion = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=0),
        _observation(
            other_occasion_item,
            respondent_id="respondent_alpha",
            category=1,
        ),
        _observation(item_alpha, respondent_id="respondent_beta", category=1),
        _observation(
            other_occasion_item,
            respondent_id="respondent_beta",
            category=0,
        ),
    )
    with pytest.raises(PilotObservationError) as occasion_error:
        build_gtheory_pi_pilot_design(mixed_occasion)
    assert occasion_error.value.code == "gtheory_pi_occasion_confounded"


def test_gtheory_pi_design_rejects_incomplete_or_degenerate_matrices():
    """The complete balanced Rust contract is enforced without case deletion."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot(
        "generated_item_beta",
        query_testlet_id="query_testlet_beta",
    )
    incomplete = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=0),
        _observation(item_beta, respondent_id="respondent_alpha", category=1),
        _observation(item_alpha, respondent_id="respondent_beta", category=1),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            response_state=PilotResponseState.INSUFFICIENT_EVIDENCE,
            category=None,
        ),
    )
    with pytest.raises(PilotObservationError) as incomplete_error:
        build_gtheory_pi_pilot_design(incomplete)
    assert incomplete_error.value.code == "gtheory_pi_incomplete_design"

    one_respondent = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=0),
        _observation(item_beta, respondent_id="respondent_alpha", category=1),
    )
    with pytest.raises(PilotObservationError) as respondent_error:
        build_gtheory_pi_pilot_design(one_respondent)
    assert respondent_error.value.code == "gtheory_pi_insufficient_respondents"

    one_item = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=0),
        _observation(item_alpha, respondent_id="respondent_beta", category=1),
    )
    with pytest.raises(PilotObservationError) as item_error:
        build_gtheory_pi_pilot_design(one_item)
    assert item_error.value.code == "gtheory_pi_insufficient_items"


def test_gtheory_pi_design_is_factory_sealed_and_schema_bound():
    """Only the public builder may wrap one validated facets design."""
    design = build_gtheory_pi_pilot_design(_complete_records())
    with pytest.raises(ValueError, match="build_gtheory_pi_pilot_design"):
        GTheoryPiPilotDesign(
            facets_design=design.facets_design,
            rater_id=design.rater_id,
            occasion_id=design.occasion_id,
        )

    with pytest.raises(TypeError, match="FacetsPilotDesign"):
        GTheoryPiPilotDesign(
            facets_design=object(),  # type: ignore[arg-type]
            rater_id="rater_alpha",
            occasion_id="occasion_window_alpha",
            _design_token=gtheory_pilot._GTHEORY_PI_DESIGN_TOKEN,
        )

    with pytest.raises(PilotObservationError) as rater_error:
        GTheoryPiPilotDesign(
            facets_design=design.facets_design,
            rater_id="rater_beta",
            occasion_id=design.occasion_id,
            _design_token=gtheory_pilot._GTHEORY_PI_DESIGN_TOKEN,
        )
    assert rater_error.value.code == "gtheory_pi_rater_confounded"

    with pytest.raises(PilotObservationError) as occasion_error:
        GTheoryPiPilotDesign(
            facets_design=design.facets_design,
            rater_id=design.rater_id,
            occasion_id="occasion_window_beta",
            _design_token=gtheory_pilot._GTHEORY_PI_DESIGN_TOKEN,
        )
    assert occasion_error.value.code == "gtheory_pi_occasion_confounded"

    facets_design = design.facets_design
    object.__setattr__(facets_design, "schema_version", "different_schema")
    with pytest.raises(ValueError, match="schema_version"):
        GTheoryPiPilotDesign(
            facets_design=facets_design,
            rater_id=design.rater_id,
            occasion_id=design.occasion_id,
            _design_token=gtheory_pilot._GTHEORY_PI_DESIGN_TOKEN,
        )


def test_gtheory_pi_setting_validation_is_bounded_and_strict():
    """Untrusted D-study sizes and mastery cuts cannot amplify work or coerce types."""
    design = build_gtheory_pi_pilot_design(_complete_records())

    with pytest.raises(ValueError, match="at least 1 value"):
        design.to_gtheory_pi_kwargs(())
    with pytest.raises(ValueError, match="collection"):
        design.to_gtheory_pi_kwargs("not_sizes")
    with pytest.raises(ValueError, match="at most 64 values"):
        design.to_gtheory_pi_kwargs(range(1, 66))
    with pytest.raises(ValueError, match="must be an integer"):
        design.to_gtheory_pi_kwargs((True,))
    with pytest.raises(ValueError, match="must be an integer"):
        design.to_gtheory_pi_kwargs((1.5,))
    with pytest.raises(ValueError, match="between 1"):
        design.to_gtheory_pi_kwargs((0,))
    with pytest.raises(ValueError, match="between 1"):
        design.to_gtheory_pi_kwargs((gtheory_pilot.MAX_GTHEORY_PRIME_SIZE + 1,))

    for cut in (True, "one", np.inf):
        with pytest.raises(ValueError, match="cut must be a finite number"):
            design.to_phi_lambda_kwargs(cut)  # type: ignore[arg-type]


def test_gtheory_pi_pilot_handoff_is_exported_without_star_surface_change():
    """Documented explicit imports work without changing the stable star contract."""
    assert rubric.GTheoryPiPilotDesign is GTheoryPiPilotDesign
    assert rubric.build_gtheory_pi_pilot_design is build_gtheory_pi_pilot_design
    assert "GTheoryPiPilotDesign" not in rubric.__all__
    assert "build_gtheory_pi_pilot_design" not in rubric.__all__
