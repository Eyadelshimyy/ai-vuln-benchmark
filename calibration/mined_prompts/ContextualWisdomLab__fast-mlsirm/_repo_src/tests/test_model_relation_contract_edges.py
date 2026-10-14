"""Edge contracts for structural model-relation governance."""

from __future__ import annotations

import pytest

from fast_mlsirm.model_relation import (
    MeasurementModelRelation,
    ModelComparisonProcedure,
    ModelRelationEvidence,
    classify_model_relation,
)


def test_nonnested_relation_without_overlap_classification_stays_unknown() -> None:
    """A non-embedding fact alone cannot choose a nonnested procedure."""
    result = classify_model_relation(
        ModelRelationEvidence(parameter_embedding=False)
    )

    assert result.relation is MeasurementModelRelation.UNKNOWN
    assert (
        result.required_procedure
        is ModelComparisonProcedure.RELATION_CLASSIFICATION
    )
    assert not result.selection_permitted
    assert result.reason_code == "requires_relation_classification"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameter_embedding": None, "null_on_boundary": True},
        {"parameter_embedding": None, "unidentified_under_null": True},
        {"parameter_embedding": None, "nonlinear_constraints": True},
        {"parameter_embedding": None, "parameter_spaces_overlap": False},
        {"parameter_embedding": None, "formal_distinguishability": False},
    ],
)
def test_unknown_embedding_rejects_attached_relation_facts(
    kwargs: dict[str, object],
) -> None:
    """Other facts cannot be interpreted before embedding is established."""
    with pytest.raises(ValueError, match="relation evidence"):
        ModelRelationEvidence(**kwargs)


def test_formally_distinguishable_overlapping_models_allow_vuong_selection() -> None:
    """Overlapping models retain their structural relation after the formal test."""
    result = classify_model_relation(
        ModelRelationEvidence(
            parameter_embedding=False,
            parameter_spaces_overlap=True,
            formal_distinguishability=True,
        )
    )

    assert result.relation is MeasurementModelRelation.OVERLAPPING
    assert result.required_procedure is ModelComparisonProcedure.VUONG_SELECTION
    assert result.selection_permitted
    assert result.reason_code == "formally_distinguishable"


def test_boundary_evidence_takes_precedence_over_nonlinear_constraint() -> None:
    """A boundary-null condition keeps ordinary chi-square disabled."""
    result = classify_model_relation(
        ModelRelationEvidence(
            parameter_embedding=True,
            null_on_boundary=True,
            nonlinear_constraints=True,
        )
    )

    assert result.relation is MeasurementModelRelation.BOUNDARY_NESTED
    assert (
        result.required_procedure
        is ModelComparisonProcedure.PARAMETRIC_BOOTSTRAP_LR
    )
    assert result.reason_code == "nonregular_nested_relation"
