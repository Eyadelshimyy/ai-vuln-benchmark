"""Fail-first structural measurement-model relation contracts for issue #608."""

from __future__ import annotations

import importlib.util

import pytest


def test_model_relation_contract_module_exists() -> None:
    """Structural relation governance must be separate from factor retention."""
    assert importlib.util.find_spec("fast_mlsirm.model_relation") is not None


def _surface():
    """Import the planned contract only after the existence assertion."""
    from fast_mlsirm.model_relation import (
        MeasurementModelRelation,
        ModelComparisonProcedure,
        ModelRelationEvidence,
        classify_model_relation,
    )

    return (
        MeasurementModelRelation,
        ModelComparisonProcedure,
        ModelRelationEvidence,
        classify_model_relation,
    )


def test_regular_embedding_requires_regular_likelihood_ratio() -> None:
    """A regular parameter-space embedding permits the ordinary LR procedure."""
    Relation, Procedure, Evidence, classify = _surface()

    result = classify(Evidence(parameter_embedding=True))

    assert result.relation is Relation.REGULAR_NESTED
    assert result.required_procedure is Procedure.LIKELIHOOD_RATIO
    assert result.selection_permitted
    assert result.reason_code == "regular_nested_relation"


@pytest.mark.parametrize("field_name", ["null_on_boundary", "unidentified_under_null"])
def test_boundary_or_unidentified_null_requires_bootstrap_lr(field_name: str) -> None:
    """Nonregular nested relations must not use ordinary chi-square automatically."""
    Relation, Procedure, Evidence, classify = _surface()
    kwargs = {field_name: True}

    result = classify(Evidence(parameter_embedding=True, **kwargs))

    assert result.relation is Relation.BOUNDARY_NESTED
    assert result.required_procedure is Procedure.PARAMETRIC_BOOTSTRAP_LR
    assert result.selection_permitted
    assert result.reason_code == "nonregular_nested_relation"


def test_nonlinear_constraint_nested_requires_bootstrap_lr() -> None:
    """A nonlinear restriction is represented separately from regular nesting."""
    Relation, Procedure, Evidence, classify = _surface()

    result = classify(
        Evidence(parameter_embedding=True, nonlinear_constraints=True)
    )

    assert result.relation is Relation.NONLINEAR_CONSTRAINT_NESTED
    assert result.required_procedure is Procedure.PARAMETRIC_BOOTSTRAP_LR
    assert result.selection_permitted
    assert result.reason_code == "nonlinear_constraint_relation"


@pytest.mark.parametrize("overlap", [False, True])
def test_nonnested_relation_requires_formal_distinguishability_first(
    overlap: bool,
) -> None:
    """Nonnested or overlapping candidates cannot skip Vuong distinguishability."""
    Relation, Procedure, Evidence, classify = _surface()

    result = classify(
        Evidence(
            parameter_embedding=False,
            parameter_spaces_overlap=overlap,
        )
    )

    assert result.relation is (
        Relation.OVERLAPPING if overlap else Relation.STRICTLY_NON_NESTED
    )
    assert result.required_procedure is Procedure.VUONG_DISTINGUISHABILITY
    assert not result.selection_permitted
    assert result.reason_code == "requires_distinguishability_test"


def test_distinguishable_nonnested_relation_allows_vuong_selection() -> None:
    """A/B selection is allowed only after formal distinguishability succeeds."""
    Relation, Procedure, Evidence, classify = _surface()

    result = classify(
        Evidence(
            parameter_embedding=False,
            parameter_spaces_overlap=False,
            formal_distinguishability=True,
        )
    )

    assert result.relation is Relation.STRICTLY_NON_NESTED
    assert result.required_procedure is Procedure.VUONG_SELECTION
    assert result.selection_permitted
    assert result.reason_code == "formally_distinguishable"


def test_failed_distinguishability_returns_no_forced_winner() -> None:
    """An indistinguishable result must prohibit any pairwise selection claim."""
    Relation, Procedure, Evidence, classify = _surface()

    result = classify(
        Evidence(
            parameter_embedding=False,
            parameter_spaces_overlap=True,
            formal_distinguishability=False,
        )
    )

    assert result.relation is Relation.INDISTINGUISHABLE
    assert result.required_procedure is Procedure.NO_SELECTION
    assert not result.selection_permitted
    assert result.reason_code == "indistinguishable"


def test_unknown_embedding_fails_closed_before_procedure_selection() -> None:
    """Missing relation evidence must return an explicit unknown state."""
    Relation, Procedure, Evidence, classify = _surface()

    result = classify(Evidence(parameter_embedding=None))

    assert result.relation is Relation.UNKNOWN
    assert result.required_procedure is Procedure.RELATION_CLASSIFICATION
    assert not result.selection_permitted
    assert result.reason_code == "requires_relation_classification"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameter_embedding": False, "null_on_boundary": True},
        {"parameter_embedding": False, "unidentified_under_null": True},
        {"parameter_embedding": False, "nonlinear_constraints": True},
        {"parameter_embedding": True, "parameter_spaces_overlap": True},
        {"parameter_embedding": True, "formal_distinguishability": True},
        {
            "parameter_embedding": False,
            "parameter_spaces_overlap": None,
            "formal_distinguishability": True,
        },
    ],
)
def test_contradictory_relation_evidence_is_rejected(kwargs: dict[str, object]) -> None:
    """Constraint and distinguishability facts cannot be attached to the wrong relation."""
    _, _, Evidence, _ = _surface()

    with pytest.raises(ValueError, match="relation evidence"):
        Evidence(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("parameter_embedding", 1),
        ("null_on_boundary", 1),
        ("unidentified_under_null", "false"),
        ("nonlinear_constraints", object()),
        ("parameter_spaces_overlap", 0),
        ("formal_distinguishability", "true"),
    ],
)
def test_relation_facts_require_exact_boolean_or_none(
    field_name: str,
    value: object,
) -> None:
    """Truthiness must not silently classify structural model relations."""
    _, _, Evidence, _ = _surface()
    kwargs: dict[str, object] = {"parameter_embedding": None, field_name: value}

    with pytest.raises(TypeError, match=field_name):
        Evidence(**kwargs)


def test_classifier_rejects_unowned_transport_objects() -> None:
    """The public classifier accepts only its immutable package-owned evidence."""
    _, _, _, classify = _surface()

    with pytest.raises(TypeError, match="ModelRelationEvidence"):
        classify(object())  # type: ignore[arg-type]


def test_classifier_revalidates_post_construction_evidence() -> None:
    """Forged frozen fields must not bypass exact relation-fact validation."""
    _, _, Evidence, classify = _surface()
    evidence = Evidence(parameter_embedding=True)
    object.__setattr__(evidence, "parameter_embedding", "regular")

    with pytest.raises(TypeError, match="parameter_embedding"):
        classify(evidence)
