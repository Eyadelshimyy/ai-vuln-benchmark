"""Contracts for the pilot-observation binary MIRT calibration handoff."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pytest

import fast_mlsirm.rubric.pilot_observations as pilot_observations
from fast_mlsirm.fit import fit
from fast_mlsirm.rubric import (
    MirtPilotDesign,
    PilotObservationError,
    PilotResponseState,
    build_mirt_pilot_design,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _two_testlet_records():
    """Return records spanning two respondents, two testlets, and one gap."""
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
            category=1,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_alpha",
            rater_id="rater_beta",
            response_state=PilotResponseState.INSUFFICIENT_EVIDENCE,
            category=None,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=0,
        ),
    )


def test_mirt_design_is_deterministic_and_preserves_states_and_raters():
    """The design orders cells, maps testlets to factors, and keeps provenance."""
    records = _two_testlet_records()
    first = build_mirt_pilot_design(records)
    second = build_mirt_pilot_design(tuple(reversed(records)))

    assert first == second
    assert first.design_fingerprint == second.design_fingerprint
    assert first.design_id.startswith("mirt_pilot_design_")
    assert first.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert first.item_ids == ("generated_item_alpha", "generated_item_beta")
    assert first.factor_testlet_ids == (
        "query_testlet_alpha",
        "query_testlet_beta",
    )
    assert first.item_factor_ids == (0, 1)
    assert first.responses == ((1, None), (None, 0))
    assert first.response_states == (
        (
            PilotResponseState.OBSERVED,
            PilotResponseState.INSUFFICIENT_EVIDENCE,
        ),
        (PilotResponseState.MISSING, PilotResponseState.OBSERVED),
    )
    assert first.rater_assignments == (
        ("rater_alpha", "rater_beta"),
        (None, "rater_beta"),
    )
    payload = first.to_dict()
    assert payload["design_fingerprint"] == first.design_fingerprint
    assert payload["response_states"][0][1] == "insufficient_evidence"
    assert payload["item_provenance"][0]["item_id"] == "generated_item_alpha"


def test_mirt_design_fit_kwargs_match_the_calibration_api():
    """Handoff kwargs are fresh, NaN-masked, and accepted by ``fit`` by name."""
    design = build_mirt_pilot_design(_two_testlet_records())

    kwargs = design.to_fit_kwargs()
    assert set(kwargs) <= set(inspect.signature(fit).parameters)
    responses = kwargs["responses"]
    assert responses.dtype == np.float64
    assert responses.shape == (2, 2)
    assert responses[0, 0] == 1.0
    assert np.isnan(responses[0, 1])
    assert np.isnan(responses[1, 0])
    assert responses[1, 1] == 0.0
    np.testing.assert_array_equal(
        kwargs["factor_id"], np.asarray([0, 1], dtype=np.int64)
    )
    assert kwargs["responses"] is not design.to_fit_kwargs()["responses"]


def test_mirt_design_rejects_direct_construction_and_invalid_records():
    """Only the validated assembler may mint designs from typed records."""
    design = build_mirt_pilot_design(_two_testlet_records())
    with pytest.raises(ValueError, match="build_mirt_pilot_design"):
        MirtPilotDesign(
            pilot_study_id=design.pilot_study_id,
            respondent_ids=design.respondent_ids,
            item_provenance=design.item_provenance,
            factor_testlet_ids=design.factor_testlet_ids,
            item_factor_ids=design.item_factor_ids,
            responses=design.responses,
            response_states=design.response_states,
            rater_assignments=design.rater_assignments,
        )

    with pytest.raises(ValueError, match="records"):
        build_mirt_pilot_design(())
    with pytest.raises(TypeError, match=r"records\[1\]"):
        build_mirt_pilot_design((_observation(), object()))


def test_mirt_design_rejects_mixed_studies_and_provenance_conflicts():
    """One study and one provenance binding per item are enforced fail-closed."""
    base = _observation()
    other_study = _observation(_pilot(pilot_study_id="pilot_study_beta"))
    with pytest.raises(PilotObservationError) as mixed:
        build_mirt_pilot_design((base, other_study))
    assert mixed.value.code == "mixed_pilot_study"

    conflicting_item = _observation(
        _pilot(query_testlet_id="query_testlet_beta"),
        respondent_id="respondent_beta",
    )
    with pytest.raises(PilotObservationError) as conflict:
        build_mirt_pilot_design((base, conflicting_item))
    assert conflict.value.code == "item_provenance_conflict"


def test_mirt_design_rejects_multi_rater_cells_and_polytomous_categories():
    """Rater aggregation and dichotomization decisions are never made silently."""
    pilot = _pilot()
    first_rater = _observation(pilot, rater_id="rater_alpha", category=1)
    second_rater = _observation(pilot, rater_id="rater_beta", category=0)
    with pytest.raises(PilotObservationError) as duplicated:
        build_mirt_pilot_design((first_rater, second_rater))
    assert duplicated.value.code == "duplicate_person_item_cell"
    assert "build_facets_pilot_design" in duplicated.value.message

    with pytest.raises(PilotObservationError) as polytomous:
        build_mirt_pilot_design((_observation(pilot, category=2),))
    assert polytomous.value.code == "non_binary_observed_category"
    assert "build_facets_pilot_design" in polytomous.value.message


def test_mirt_design_requires_observed_support_for_every_index():
    """Rows and columns without any observed response cannot be calibrated."""
    with pytest.raises(PilotObservationError) as unobserved_all:
        build_mirt_pilot_design(
            (
                _observation(
                    response_state=PilotResponseState.MISSING,
                    category=None,
                ),
            )
        )
    assert unobserved_all.value.code == "no_observed_response"

    pilot = _pilot()
    with pytest.raises(PilotObservationError) as unobserved_person:
        build_mirt_pilot_design(
            (
                _observation(pilot, respondent_id="respondent_alpha", category=1),
                _observation(
                    pilot,
                    respondent_id="respondent_beta",
                    response_state=PilotResponseState.NOT_APPLICABLE,
                    category=None,
                ),
            )
        )
    assert unobserved_person.value.code == "unobserved_respondent"

    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot("generated_item_beta")
    with pytest.raises(PilotObservationError) as unobserved_item:
        build_mirt_pilot_design(
            (
                _observation(item_alpha, category=1),
                _observation(
                    item_beta,
                    response_state=PilotResponseState.MISSING,
                    category=None,
                ),
            )
        )
    assert unobserved_item.value.code == "unobserved_item"


def test_sparse_identifiers_cannot_amplify_into_an_oversized_matrix(
    monkeypatch: pytest.MonkeyPatch,
):
    """The dense persons-by-items budget is enforced before allocation."""
    monkeypatch.setattr(pilot_observations, "MAX_MIRT_PILOT_CELLS", 3)

    with pytest.raises(PilotObservationError) as caught:
        pilot_observations.build_mirt_pilot_design(_two_testlet_records())

    assert caught.value.code == "mirt_design_cell_budget_exceeded"
    assert caught.value.path == "$.records"
    assert "requires 4 cells" in caught.value.message
    assert "maximum is 3" in caught.value.message
