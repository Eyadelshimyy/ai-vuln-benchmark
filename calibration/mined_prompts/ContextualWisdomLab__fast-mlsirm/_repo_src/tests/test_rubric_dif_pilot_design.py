"""Contracts for the generated-item observed-score DIF pilot handoff."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pytest

import fast_mlsirm.rubric as rubric
import fast_mlsirm.rubric.dif_pilot as dif_pilot
from fast_mlsirm.dif import logistic_dif, mantel_haenszel_dif, sibtest
from fast_mlsirm.rubric import (
    DifPilotDesign,
    PilotObservationError,
    PilotResponseState,
    build_dif_pilot_design,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _complete_records():
    """Return a complete four-person, three-item binary pilot matrix."""
    items = (
        _pilot("generated_item_alpha"),
        _pilot("generated_item_beta", query_testlet_id="query_testlet_beta"),
        _pilot("generated_item_gamma", query_testlet_id="query_testlet_gamma"),
    )
    response_rows = {
        "respondent_alpha": (1, 1, 0),
        "respondent_beta": (1, 0, 0),
        "respondent_gamma": (0, 1, 1),
        "respondent_delta": (0, 0, 1),
    }
    records = []
    for respondent_id, categories in response_rows.items():
        for item, category in zip(items, categories, strict=True):
            records.append(
                _observation(
                    item,
                    respondent_id=respondent_id,
                    rater_id=f"rater_{respondent_id.removeprefix('respondent_')}",
                    category=category,
                )
            )
    return tuple(records)


def _groups():
    """Return explicit descriptive reference and focal assignments."""
    return {
        "respondent_alpha": "reference_group_alpha",
        "respondent_beta": "reference_group_alpha",
        "respondent_gamma": "focal_group_alpha",
        "respondent_delta": "focal_group_alpha",
    }


def _design(records=None, groups=None):
    """Build the standard DIF pilot design used by focused tests."""
    return build_dif_pilot_design(
        records or _complete_records(),
        respondent_groups=groups or _groups(),
        reference_group_id="reference_group_alpha",
        focal_group_id="focal_group_alpha",
    )


def test_dif_design_is_deterministic_and_preserves_governed_provenance():
    """Record and mapping order cannot change the fingerprinted DIF design."""
    records = _complete_records()
    groups = _groups()
    first = _design(records, groups)
    second = _design(tuple(reversed(records)), dict(reversed(tuple(groups.items()))))

    assert first == second
    assert first.design_fingerprint == second.design_fingerprint
    assert first.design_id.startswith("dif_pilot_design_")
    assert first.pilot_study_id == "pilot_study_alpha"
    assert first.respondent_ids == (
        "respondent_alpha",
        "respondent_beta",
        "respondent_delta",
        "respondent_gamma",
    )
    assert first.respondent_group_ids == (
        "reference_group_alpha",
        "reference_group_alpha",
        "focal_group_alpha",
        "focal_group_alpha",
    )
    assert first.item_ids == (
        "generated_item_alpha",
        "generated_item_beta",
        "generated_item_gamma",
    )
    assert tuple(entry.item_id for entry in first.item_provenance) == first.item_ids
    assert first.is_complete_observed_matrix is True
    assert all(
        state is PilotResponseState.OBSERVED
        for row in first.response_states
        for state in row
    )
    assert all(rater is not None for row in first.rater_assignments for rater in row)

    payload = first.to_dict()
    assert payload["design_id"] == first.design_id
    assert payload["design_fingerprint"] == first.design_fingerprint
    assert payload["is_complete_observed_matrix"] is True
    assert payload["reference_group_id"] == "reference_group_alpha"
    assert payload["focal_group_id"] == "focal_group_alpha"
    assert payload["binary_design"]["design_id"].startswith("mirt_pilot_design_")


def test_observed_score_kwargs_match_all_existing_binary_dif_entrypoints():
    """The handoff emits fresh arrays accepted by each observed-score DIF API."""
    design = _design()
    kwargs = design.to_observed_score_dif_kwargs()

    for function in (mantel_haenszel_dif, logistic_dif, sibtest):
        assert set(kwargs) <= set(inspect.signature(function).parameters)

    responses = kwargs["responses"]
    groups = kwargs["group"]
    assert responses.dtype == np.int64
    assert groups.dtype == np.int64
    assert responses.shape == (4, 3)
    np.testing.assert_array_equal(groups, np.asarray([0, 0, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        responses,
        np.asarray(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.int64,
        ),
    )

    responses[0, 0] = 0
    groups[0] = 1
    fresh = design.to_observed_score_dif_kwargs()
    assert fresh["responses"][0, 0] == 1
    assert fresh["group"][0] == 0
    assert design.responses_array().dtype == np.float64
    assert design.group_array().dtype == np.int64


def test_nonobserved_states_remain_auditable_and_block_observed_score_dif():
    """Missingness is retained and never silently dropped or imputed."""
    records = list(_complete_records())
    target = next(
        index
        for index, record in enumerate(records)
        if record.respondent_id == "respondent_delta"
        and record.item_id == "generated_item_gamma"
    )
    pilot = _pilot("generated_item_gamma", query_testlet_id="query_testlet_gamma")
    records[target] = _observation(
        pilot,
        respondent_id="respondent_delta",
        rater_id="rater_delta",
        response_state=PilotResponseState.INSUFFICIENT_EVIDENCE,
        category=None,
    )

    design = _design(tuple(records))
    assert design.is_complete_observed_matrix is False
    assert np.isnan(design.responses_array()[2, 2])
    assert design.to_dict()["is_complete_observed_matrix"] is False

    with pytest.raises(PilotObservationError) as rejection:
        design.to_observed_score_dif_kwargs()
    assert rejection.value.code == "dif_incomplete_response_matrix"
    assert rejection.value.path == "$.binary_design.response_states[2][2]"
    assert "no deletion or imputation" in rejection.value.message


def test_group_assignment_contract_rejects_missing_unknown_and_undeclared_values():
    """Every indexed respondent must have exactly one declared group identity."""
    records = _complete_records()

    with pytest.raises(TypeError, match="respondent_groups must be a mapping"):
        build_dif_pilot_design(
            records,
            respondent_groups=object(),  # type: ignore[arg-type]
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
        )

    missing = _groups()
    missing.pop("respondent_delta")
    with pytest.raises(PilotObservationError) as missing_error:
        _design(records, missing)
    assert missing_error.value.code == "dif_missing_group_assignment"

    surplus = {**_groups(), "respondent_unknown": "focal_group_alpha"}
    with pytest.raises(PilotObservationError) as surplus_error:
        _design(records, surplus)
    assert surplus_error.value.code == "dif_group_assignment_count_exceeded"

    unknown = _groups()
    unknown.pop("respondent_delta")
    unknown["respondent_unknown"] = "focal_group_alpha"
    with pytest.raises(PilotObservationError) as unknown_error:
        _design(records, unknown)
    assert unknown_error.value.code == "dif_unknown_respondent_assignment"

    undeclared = {**_groups(), "respondent_delta": "comparison_group_beta"}
    with pytest.raises(PilotObservationError) as undeclared_error:
        _design(records, undeclared)
    assert undeclared_error.value.code == "dif_unknown_group_assignment"

    all_reference = {
        respondent_id: "reference_group_alpha" for respondent_id in _groups()
    }
    with pytest.raises(PilotObservationError) as empty_focal:
        _design(records, all_reference)
    assert empty_focal.value.code == "dif_empty_focal_group"

    all_focal = {respondent_id: "focal_group_alpha" for respondent_id in _groups()}
    with pytest.raises(PilotObservationError) as empty_reference:
        _design(records, all_focal)
    assert empty_reference.value.code == "dif_empty_reference_group"


def test_group_identifiers_and_factory_seal_fail_closed():
    """Descriptive group identities and factory-only construction are enforced."""
    design = _design()

    with pytest.raises(ValueError, match="build_dif_pilot_design"):
        DifPilotDesign(
            binary_design=design.binary_design,
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
            respondent_group_ids=design.respondent_group_ids,
        )

    with pytest.raises(TypeError, match="MirtPilotDesign"):
        DifPilotDesign(
            binary_design=object(),  # type: ignore[arg-type]
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
            respondent_group_ids=("reference_group_alpha", "focal_group_alpha"),
            _design_token=dif_pilot._DIF_DESIGN_TOKEN,
        )

    with pytest.raises(ValueError, match="must differ"):
        build_dif_pilot_design(
            _complete_records(),
            respondent_groups=_groups(),
            reference_group_id="shared_group_alpha",
            focal_group_id="shared_group_alpha",
        )
    with pytest.raises(ValueError, match="reference_group_id"):
        build_dif_pilot_design(
            _complete_records(),
            respondent_groups=_groups(),
            reference_group_id="reference",
            focal_group_id="focal_group_alpha",
        )

    with pytest.raises(ValueError, match="one-to-one"):
        DifPilotDesign(
            binary_design=design.binary_design,
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
            respondent_group_ids=("reference_group_alpha",),
            _design_token=dif_pilot._DIF_DESIGN_TOKEN,
        )

    object.__setattr__(design.binary_design, "schema_version", "different_schema")
    with pytest.raises(ValueError, match="schema_version"):
        DifPilotDesign(
            binary_design=design.binary_design,
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
            respondent_group_ids=design.respondent_group_ids,
            _design_token=dif_pilot._DIF_DESIGN_TOKEN,
        )


def test_dif_builder_reuses_binary_validation_and_public_exports():
    """No weaker parser, category conversion, or hidden export path is added."""
    assert rubric.DifPilotDesign is DifPilotDesign
    assert rubric.build_dif_pilot_design is build_dif_pilot_design

    with pytest.raises(ValueError, match="records"):
        build_dif_pilot_design(
            (),
            respondent_groups={},
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
        )
    with pytest.raises(PilotObservationError) as polytomous:
        build_dif_pilot_design(
            (_observation(category=2),),
            respondent_groups={"respondent_alpha": "reference_group_alpha"},
            reference_group_id="reference_group_alpha",
            focal_group_id="focal_group_alpha",
        )
    assert polytomous.value.code == "non_binary_observed_category"
