"""Contracts for the generated-item binary testlet calibration handoff."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pytest

import fast_mlsirm.rubric as rubric
import fast_mlsirm.rubric.testlet_pilot as testlet_pilot
from fast_mlsirm.rubric import (
    PilotObservationError,
    PilotResponseState,
    TestletPilotDesign,
    build_testlet_pilot_design,
)
from fast_mlsirm.testlet import fit_testlet

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _testlet_records():
    """Return deterministic binary records with one repeated query testlet."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot("generated_item_beta")
    item_gamma = _pilot(
        "generated_item_gamma",
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
            item_gamma,
            respondent_id="respondent_alpha",
            rater_id="rater_gamma",
            category=0,
        ),
        _observation(
            item_beta,
            respondent_id="respondent_beta",
            rater_id="rater_beta",
            category=0,
        ),
        _observation(
            item_gamma,
            respondent_id="respondent_beta",
            rater_id="rater_gamma",
            category=1,
        ),
    )


def test_testlet_design_is_deterministic_and_discloses_grouping():
    """Query-testlet mappings remain explicit, immutable, and fingerprinted."""
    records = _testlet_records()
    first = build_testlet_pilot_design(records)
    second = build_testlet_pilot_design(tuple(reversed(records)))

    assert first == second
    assert first.design_fingerprint == second.design_fingerprint
    assert first.design_id.startswith("testlet_pilot_design_")
    assert first.pilot_study_id == "pilot_study_alpha"
    assert first.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert first.item_ids == (
        "generated_item_alpha",
        "generated_item_beta",
        "generated_item_gamma",
    )
    assert tuple(entry.item_id for entry in first.item_provenance) == first.item_ids
    assert first.query_testlet_ids == (
        "query_testlet_alpha",
        "query_testlet_beta",
    )
    assert first.item_testlet_ids == (0, 0, 1)
    assert first.responses == ((1, None, 0), (None, 0, 1))
    assert first.response_states == (
        (
            PilotResponseState.OBSERVED,
            PilotResponseState.INSUFFICIENT_EVIDENCE,
            PilotResponseState.OBSERVED,
        ),
        (
            PilotResponseState.MISSING,
            PilotResponseState.OBSERVED,
            PilotResponseState.OBSERVED,
        ),
    )
    assert first.rater_assignments == (
        ("rater_alpha", "rater_beta", "rater_gamma"),
        (None, "rater_beta", "rater_gamma"),
    )

    payload = first.to_dict()
    assert payload["design_id"] == first.design_id
    assert payload["design_fingerprint"] == first.design_fingerprint
    assert payload["query_testlet_ids"] == list(first.query_testlet_ids)
    assert payload["item_testlet_ids"] == list(first.item_testlet_ids)
    assert payload["binary_design"]["design_id"].startswith("mirt_pilot_design_")


def test_testlet_fit_kwargs_match_the_existing_rust_backed_api():
    """The handoff emits fresh arrays and validated testlet-fit settings."""
    design = build_testlet_pilot_design(_testlet_records())

    kwargs = design.to_fit_testlet_kwargs()
    assert set(kwargs) <= set(inspect.signature(fit_testlet).parameters)
    assert kwargs["model"] == "rasch"
    assert kwargs["max_iter"] == 500
    assert kwargs["tol"] == 1e-6
    assert kwargs["q_gamma"] == 21
    assert kwargs["estimate_sigma"] is True
    assert kwargs["init_sigma2"] == 0.5
    assert kwargs["require_convergence"] is False

    responses = kwargs["responses"]
    assert responses.dtype == np.float64
    assert responses.shape == (2, 3)
    assert responses[0, 0] == 1.0
    assert np.isnan(responses[0, 1])
    assert responses[0, 2] == 0.0
    assert np.isnan(responses[1, 0])
    assert responses[1, 1] == 0.0
    assert responses[1, 2] == 1.0
    np.testing.assert_array_equal(
        kwargs["testlet_id"], np.asarray([0, 0, 1], dtype=np.int64)
    )
    assert responses is not design.responses_array()
    assert kwargs["testlet_id"] is not design.testlet_id_array()

    custom = design.to_fit_testlet_kwargs(
        model="2PL",
        max_iter=np.int64(250),
        tol=np.float64(0.0),
        q_gamma=np.int64(31),
        estimate_sigma=np.bool_(False),
        init_sigma2=np.float64(0.0),
        require_convergence=np.bool_(True),
    )
    assert custom["model"] == "2pl"
    assert custom["max_iter"] == 250
    assert custom["tol"] == 0.0
    assert custom["q_gamma"] == 31
    assert custom["estimate_sigma"] is False
    assert custom["init_sigma2"] == 0.0
    assert custom["require_convergence"] is True


def test_testlet_fit_kwargs_reject_invalid_execution_settings():
    """Malformed settings fail before a calibration payload is returned."""
    design = build_testlet_pilot_design(_testlet_records())

    for invalid_model in (object(), "3pl"):
        with pytest.raises(ValueError, match="model"):
            design.to_fit_testlet_kwargs(model=invalid_model)  # type: ignore[arg-type]
    for invalid_max_iter in (True, 0, 100_001, 1.5):
        with pytest.raises(ValueError, match="max_iter"):
            design.to_fit_testlet_kwargs(max_iter=invalid_max_iter)  # type: ignore[arg-type]
    # #1929: no node-count cap; q_gamma=8/42 are now accepted, only
    # non-integer/non-positive values are not.
    for invalid_q_gamma in (True, 0, 7.5):
        with pytest.raises(ValueError, match="q_gamma"):
            design.to_fit_testlet_kwargs(q_gamma=invalid_q_gamma)  # type: ignore[arg-type]
    for name, value in (
        ("tol", True),
        ("tol", -1.0),
        ("tol", np.inf),
        ("tol", "not_numeric"),
        ("init_sigma2", False),
        ("init_sigma2", -0.1),
        ("init_sigma2", np.nan),
        ("init_sigma2", object()),
    ):
        with pytest.raises(ValueError, match=name):
            if name == "tol":
                design.to_fit_testlet_kwargs(tol=value)  # type: ignore[arg-type]
            else:
                design.to_fit_testlet_kwargs(init_sigma2=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="estimate_sigma"):
        design.to_fit_testlet_kwargs(estimate_sigma=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="require_convergence"):
        design.to_fit_testlet_kwargs(require_convergence="yes")  # type: ignore[arg-type]


def test_testlet_design_is_factory_sealed_schema_bound_and_nontrivial():
    """Only a validated binary design with a repeated testlet may be wrapped."""
    design = build_testlet_pilot_design(_testlet_records())
    with pytest.raises(ValueError, match="build_testlet_pilot_design"):
        TestletPilotDesign(binary_design=design.binary_design)

    with pytest.raises(TypeError, match="MirtPilotDesign"):
        TestletPilotDesign(
            binary_design=object(),  # type: ignore[arg-type]
            _design_token=testlet_pilot._TESTLET_DESIGN_TOKEN,
        )

    binary_design = design.binary_design
    object.__setattr__(binary_design, "schema_version", "different_schema")
    with pytest.raises(ValueError, match="schema_version"):
        TestletPilotDesign(
            binary_design=binary_design,
            _design_token=testlet_pilot._TESTLET_DESIGN_TOKEN,
        )

    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot(
        "generated_item_beta",
        query_testlet_id="query_testlet_beta",
    )
    singleton_records = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=1),
        _observation(item_beta, respondent_id="respondent_alpha", category=0),
    )
    with pytest.raises(ValueError, match="two or more items"):
        build_testlet_pilot_design(singleton_records)


def test_testlet_builder_reuses_binary_validation_and_exports():
    """No weaker parser, category, or public-export path is introduced."""
    assert rubric.TestletPilotDesign is TestletPilotDesign
    assert rubric.build_testlet_pilot_design is build_testlet_pilot_design

    with pytest.raises(ValueError, match="records"):
        build_testlet_pilot_design(())
    with pytest.raises(PilotObservationError) as polytomous:
        build_testlet_pilot_design((_observation(category=2),))
    assert polytomous.value.code == "non_binary_observed_category"
