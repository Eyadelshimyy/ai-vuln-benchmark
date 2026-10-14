"""Edge and fail-closed tests for the structural selection governor."""

from __future__ import annotations

import pytest

from fast_mlsirm.model_relation import ModelComparisonProcedure, ModelRelationEvidence
from fast_mlsirm.structural_selection import (
    StructuralComparisonOutcome,
    StructuralSelectionDecision,
    StructuralSelectionEvidence,
    _missing_comparison_decision,
    govern_structural_selection,
)


def _evidence(**overrides: object) -> StructuralSelectionEvidence:
    """Return an ordinary regular-nested evidence record with safe defaults."""
    kwargs: dict[str, object] = {
        "simpler_candidate_id": "correlated_traits",
        "more_complex_candidate_id": "bifactor_candidate",
        "relation_evidence": ModelRelationEvidence(parameter_embedding=True),
        "comparison_outcome": None,
        "simpler_score_interpretation_supported": True,
        "more_complex_score_interpretation_supported": True,
        "recovery_evidence_sufficient": True,
    }
    kwargs.update(overrides)
    return StructuralSelectionEvidence(**kwargs)


def test_unknown_relation_requires_classification() -> None:
    """Incomplete relation facts must not select a comparison procedure or winner."""
    result = govern_structural_selection(
        _evidence(relation_evidence=ModelRelationEvidence(parameter_embedding=None))
    )

    assert result.decision is StructuralSelectionDecision.REQUIRES_RELATION_CLASSIFICATION
    assert result.selected_candidate_id is None
    assert result.required_procedure is ModelComparisonProcedure.RELATION_CLASSIFICATION
    assert result.reason_code == "requires_relation_classification"


def test_admissible_comparison_can_select_supported_simpler_candidate() -> None:
    """A completed regular comparison may select the simpler supported model."""
    result = govern_structural_selection(
        _evidence(comparison_outcome=StructuralComparisonOutcome.FAVORS_SIMPLER)
    )

    assert result.decision is StructuralSelectionDecision.SELECTED
    assert result.selected_candidate_id == "correlated_traits"
    assert result.reason_code == "comparison_favors_simpler"


def test_practical_equivalence_with_no_supported_score_returns_no_winner() -> None:
    """Practical equivalence cannot rescue candidates with unsupported scores."""
    result = govern_structural_selection(
        _evidence(
            comparison_outcome=StructuralComparisonOutcome.PRACTICALLY_EQUIVALENT,
            simpler_score_interpretation_supported=False,
            more_complex_score_interpretation_supported=False,
        )
    )

    assert result.decision is StructuralSelectionDecision.SCORE_INTERPRETATION_NOT_SUPPORTED
    assert result.selected_candidate_id is None
    assert result.reason_code == "score_interpretation_not_supported"


def test_blank_candidate_identifier_fails_closed() -> None:
    """Blank exact strings are not usable model identities."""
    with pytest.raises(ValueError, match="non-empty"):
        _evidence(simpler_candidate_id="   ")


def test_relation_transport_must_be_package_owned() -> None:
    """Arbitrary relation objects cannot impersonate classified relation evidence."""
    with pytest.raises(TypeError, match="relation_evidence"):
        _evidence(relation_evidence=object())


def test_comparison_outcome_must_use_closed_enum() -> None:
    """Free-form strings cannot assert that a model won a governed comparison."""
    with pytest.raises(TypeError, match="comparison_outcome"):
        _evidence(comparison_outcome="favors_more_complex")


def test_governor_rejects_unowned_evidence_objects() -> None:
    """Only the immutable package-owned evidence transport is accepted."""
    with pytest.raises(TypeError, match="StructuralSelectionEvidence"):
        govern_structural_selection(object())  # type: ignore[arg-type]


def test_invalid_procedure_has_no_missing_comparison_mapping() -> None:
    """No-selection procedures cannot be converted into missing test evidence."""
    with pytest.raises(ValueError, match="does not admit"):
        _missing_comparison_decision(ModelComparisonProcedure.NO_SELECTION)


def test_tampered_comparison_enum_fails_closed() -> None:
    """A forged post-construction comparison value cannot silently select a model."""
    evidence = _evidence(
        comparison_outcome=StructuralComparisonOutcome.FAVORS_MORE_COMPLEX
    )
    object.__setattr__(evidence, "comparison_outcome", object())

    with pytest.raises(ValueError, match="unsupported structural comparison outcome"):
        govern_structural_selection(evidence)


def test_post_construction_recovery_tamper_fails_closed() -> None:
    """Post-construction truthiness must not bypass the independent recovery gate."""
    evidence = _evidence(
        comparison_outcome=StructuralComparisonOutcome.FAVORS_MORE_COMPLEX
    )
    object.__setattr__(evidence, "recovery_evidence_sufficient", 1)

    with pytest.raises(TypeError, match="recovery_evidence_sufficient"):
        govern_structural_selection(evidence)


def test_nested_relation_tamper_is_revalidated() -> None:
    """A forged relation fact must not become a truthy regular-nested assertion."""
    evidence = _evidence()
    object.__setattr__(evidence.relation_evidence, "parameter_embedding", "regular")

    with pytest.raises(TypeError, match="parameter_embedding"):
        govern_structural_selection(evidence)
