"""Resource-safety contracts for the pilot-observation facets handoff."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.rubric.pilot_observations as pilot_observations
from fast_mlsirm.rubric import PilotObservationError

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _sparse_two_by_two_by_two_records():
    """Return two observed cells spanning a dense 2 x 2 x 2 design."""
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
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=1,
        ),
    )


def test_sparse_identifiers_cannot_amplify_into_an_oversized_dense_tensor(
    monkeypatch: pytest.MonkeyPatch,
):
    """The full cross-product is rejected before tuple or NumPy allocation."""
    monkeypatch.setattr(pilot_observations, "MAX_FACETS_PILOT_CELLS", 7)

    with pytest.raises(PilotObservationError) as caught:
        pilot_observations.build_facets_pilot_design(
            _sparse_two_by_two_by_two_records(),
            n_cat=2,
        )

    assert caught.value.code == "facets_design_cell_budget_exceeded"
    assert caught.value.path == "$.records"
    assert "requires 8 cells" in caught.value.message
    assert "maximum is 7" in caught.value.message


def test_observed_support_rejection_precedes_dense_budget_rejection(
    monkeypatch: pytest.MonkeyPatch,
):
    """Invalid unobserved facets fail semantically before resource budgeting."""
    observed = _observation(category=1)
    missing_rater = _observation(
        rater_id="rater_beta",
        response_state="missing",
        category=None,
    )
    monkeypatch.setattr(pilot_observations, "MAX_FACETS_PILOT_CELLS", 1)

    with pytest.raises(PilotObservationError) as caught:
        pilot_observations.build_facets_pilot_design(
            (observed, missing_rater),
            n_cat=2,
        )

    assert caught.value.code == "unobserved_rater"
