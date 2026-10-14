"""Contracts for the generated-item binary bifactor calibration handoff."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pytest

import fast_mlsirm.rubric as rubric
import fast_mlsirm.rubric.bifactor_pilot as bifactor_pilot
from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit
from fast_mlsirm.rubric import (
    BifactorPilotDesign,
    PilotObservationError,
    PilotResponseState,
    build_bifactor_pilot_design,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _bifactor_records():
    """Return deterministic binary records spanning two specific factors."""
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


def test_bifactor_design_is_deterministic_and_discloses_the_loading_pattern():
    """General and specific factor mappings remain explicit and fingerprinted."""
    records = _bifactor_records()
    first = build_bifactor_pilot_design(records)
    second = build_bifactor_pilot_design(tuple(reversed(records)))

    assert first == second
    assert first.design_fingerprint == second.design_fingerprint
    assert first.design_id.startswith("bifactor_pilot_design_")
    assert first.general_factor_id == "general_factor"
    assert first.pilot_study_id == "pilot_study_alpha"
    assert first.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert first.item_ids == ("generated_item_alpha", "generated_item_beta")
    assert tuple(entry.item_id for entry in first.item_provenance) == first.item_ids
    assert first.general_factor_item_ids == first.item_ids
    assert first.specific_factor_testlet_ids == (
        "query_testlet_alpha",
        "query_testlet_beta",
    )
    assert first.item_specific_factor_ids == (0, 1)
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
    assert payload["design_id"] == first.design_id
    assert payload["design_fingerprint"] == first.design_fingerprint
    assert payload["general_factor_item_ids"] == list(first.item_ids)
    assert payload["binary_design"]["design_id"].startswith("mirt_pilot_design_")


def test_bifactor_fit_kwargs_pin_the_existing_rust_backed_model_contract():
    """The handoff produces fresh arrays and a one-general-factor MMLE config."""
    design = build_bifactor_pilot_design(_bifactor_records())

    kwargs = design.to_fit_kwargs()
    assert set(kwargs) <= set(inspect.signature(fit).parameters)
    assert kwargs["config"].normalized_model() == "BIFAC2PLM"
    assert kwargs["config"].estimator == "mmle"
    assert kwargs["config"].latent_dim == 1
    assert design.default_fit_config() == kwargs["config"]

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
    assert responses is not design.responses_array()
    assert kwargs["factor_id"] is not design.specific_factor_id_array()

    custom = FitConfig(
        model="bifac2plm",
        estimator="mmle",
        latent_dim=1,
        q_theta=7,
        q_xi=7,
    )
    assert design.to_fit_kwargs(custom)["config"] is custom


def test_bifactor_fit_kwargs_reject_mislabeled_or_incompatible_configs():
    """A bifactor audit artifact cannot silently select another fit family."""
    design = build_bifactor_pilot_design(_bifactor_records())

    with pytest.raises(TypeError, match="FitConfig"):
        design.to_fit_kwargs("not_a_fit_config")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BIFAC2PLM"):
        design.to_fit_kwargs(
            FitConfig(model="MIRT", estimator="mmle", latent_dim=1)
        )
    with pytest.raises(ValueError, match="estimator"):
        design.to_fit_kwargs(
            FitConfig(model="BIFAC2PLM", estimator="jmle", latent_dim=1)
        )
    with pytest.raises(ValueError, match="latent_dim"):
        design.to_fit_kwargs(
            FitConfig(model="BIFAC2PLM", estimator="mmle", latent_dim=2)
        )


def test_bifactor_design_is_factory_sealed_and_schema_bound():
    """Only the public builder may wrap one validated binary pilot design."""
    design = build_bifactor_pilot_design(_bifactor_records())
    with pytest.raises(ValueError, match="build_bifactor_pilot_design"):
        BifactorPilotDesign(
            general_factor_id=design.general_factor_id,
            binary_design=design.binary_design,
        )

    with pytest.raises(TypeError, match="MirtPilotDesign"):
        BifactorPilotDesign(
            general_factor_id="general_factor",
            binary_design=object(),  # type: ignore[arg-type]
            _design_token=bifactor_pilot._BIFACTOR_DESIGN_TOKEN,
        )

    binary_design = design.binary_design
    object.__setattr__(binary_design, "schema_version", "different_schema")
    with pytest.raises(ValueError, match="schema_version"):
        BifactorPilotDesign(
            general_factor_id="general_factor",
            binary_design=binary_design,
            _design_token=bifactor_pilot._BIFACTOR_DESIGN_TOKEN,
        )


def test_bifactor_builder_reuses_binary_fail_closed_validation_and_exports():
    """No weaker parser, category, or public-export path is introduced."""
    assert rubric.BifactorPilotDesign is BifactorPilotDesign
    assert rubric.build_bifactor_pilot_design is build_bifactor_pilot_design

    with pytest.raises(ValueError, match="records"):
        build_bifactor_pilot_design(())
    with pytest.raises(ValueError):
        build_bifactor_pilot_design(
            _bifactor_records(),
            general_factor_id="general",
        )
    with pytest.raises(PilotObservationError) as polytomous:
        build_bifactor_pilot_design((_observation(category=2),))
    assert polytomous.value.code == "non_binary_observed_category"
