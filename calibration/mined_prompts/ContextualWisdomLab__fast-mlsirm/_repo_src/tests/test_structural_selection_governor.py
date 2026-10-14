"""Fail-first contracts for the structural selection governor from issue #608."""

from __future__ import annotations

import importlib.util

import pytest

from fast_mlsirm.model_relation import ModelRelationEvidence


def test_structural_selection_governor_module_exists() -> None:
    """Factor retention and pairwise structural selection must remain separate APIs."""
    assert importlib.util.find_spec("fast_mlsirm.structural_selection") is not None


def _surface():
    """Import the planned governor only after the existence assertion."""
    from fast_mlsirm.structural_selection import (
        StructuralComparisonOutcome,
        StructuralSelectionDecision,
        StructuralSelectionEvidence,
        govern_structural_selection,
    )

    return (
        StructuralComparisonOutcome,
        StructuralSelectionDecision,
        StructuralSelectionEvidence,
        govern_structural_selection,
    )


def _evidence(**overrides: object):
    """Build one ordinary regular-nested pairwise decision record."""
    _, _, Evidence, _ = _surface()
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
    return Evidence(**kwargs)


def test_regular_nested_pair_requires_likelihood_ratio_before_selection() -> None:
    """A regular embedding cannot produce a winner before its LR evidence exists."""
    _, Decision, _, govern = _surface()

    result = govern(_evidence())

    assert result.decision is Decision.REQUIRES_LIKELIHOOD_RATIO
    assert result.selected_candidate_id is None
    assert result.required_procedure.value == "likelihood_ratio"
    assert result.reason_code == "requires_likelihood_ratio"


def test_boundary_nested_pair_preserves_bootstrap_lr_requirement() -> None:
    """A boundary relation must retain the nonregular bootstrap-LR procedure."""
    _, Decision, _, govern = _surface()

    result = govern(
        _evidence(
            relation_evidence=ModelRelationEvidence(
                parameter_embedding=True,
                null_on_boundary=True,
            )
        )
    )

    assert result.decision is Decision.REQUIRES_LIKELIHOOD_RATIO
    assert result.required_procedure.value == "parametric_bootstrap_lr"
    assert result.reason_code == "requires_likelihood_ratio"


def test_nonnested_pair_requires_formal_distinguishability_first() -> None:
    """A numerical pairwise winner cannot precede formal Vuong distinguishability."""
    _, Decision, _, govern = _surface()

    result = govern(
        _evidence(
            relation_evidence=ModelRelationEvidence(
                parameter_embedding=False,
                parameter_spaces_overlap=False,
            )
        )
    )

    assert result.decision is Decision.REQUIRES_DISTINGUISHABILITY_TEST
    assert result.selected_candidate_id is None
    assert result.required_procedure.value == "vuong_distinguishability"


def test_indistinguishable_pair_never_forces_a_winner() -> None:
    """Failed distinguishability remains a terminal no-selection state."""
    _, Decision, _, govern = _surface()

    result = govern(
        _evidence(
            relation_evidence=ModelRelationEvidence(
                parameter_embedding=False,
                parameter_spaces_overlap=True,
                formal_distinguishability=False,
            )
        )
    )

    assert result.decision is Decision.INDISTINGUISHABLE
    assert result.selected_candidate_id is None
    assert result.reason_code == "indistinguishable"


def test_distinguishable_nonnested_pair_requires_vuong_selection_result() -> None:
    """Distinguishability licenses a Vuong selection procedure, not an automatic win."""
    _, Decision, _, govern = _surface()

    result = govern(
        _evidence(
            relation_evidence=ModelRelationEvidence(
                parameter_embedding=False,
                parameter_spaces_overlap=False,
                formal_distinguishability=True,
            )
        )
    )

    assert result.decision is Decision.REQUIRES_VUONG_SELECTION
    assert result.selected_candidate_id is None
    assert result.required_procedure.value == "vuong_selection"


def test_recovery_gate_blocks_a_pairwise_fit_winner() -> None:
    """Pairwise comparison evidence is insufficient without recovery evidence."""
    Outcome, Decision, _, govern = _surface()

    result = govern(
        _evidence(
            comparison_outcome=Outcome.FAVORS_MORE_COMPLEX,
            recovery_evidence_sufficient=False,
        )
    )

    assert result.decision is Decision.INSUFFICIENT_RECOVERY_EVIDENCE
    assert result.selected_candidate_id is None


def test_practically_equivalent_models_prefer_supported_simpler_candidate() -> None:
    """Equivalent candidates select the simpler model only when its score is supported."""
    Outcome, Decision, _, govern = _surface()

    result = govern(
        _evidence(comparison_outcome=Outcome.PRACTICALLY_EQUIVALENT)
    )

    assert result.decision is Decision.PRACTICALLY_EQUIVALENT_PREFER_SIMPLER
    assert result.selected_candidate_id == "correlated_traits"


def test_practical_equivalence_uses_complex_model_when_simpler_score_is_invalid() -> None:
    """Parsimony cannot override an unsupported intended score interpretation."""
    Outcome, Decision, _, govern = _surface()

    result = govern(
        _evidence(
            comparison_outcome=Outcome.PRACTICALLY_EQUIVALENT,
            simpler_score_interpretation_supported=False,
        )
    )

    assert result.decision is Decision.SELECTED
    assert result.selected_candidate_id == "bifactor_candidate"
    assert result.reason_code == "simpler_score_interpretation_not_supported"


@pytest.mark.parametrize(
    "outcome",
    ["favors_simpler", "favors_more_complex"],
)
def test_selected_candidate_must_support_intended_score(outcome: str) -> None:
    """A fit preference cannot license a score interpretation that failed its gate."""
    Outcome, Decision, _, govern = _surface()
    mapped = Outcome(outcome)
    overrides = {
        "simpler_score_interpretation_supported": outcome != "favors_simpler",
        "more_complex_score_interpretation_supported": outcome != "favors_more_complex",
    }

    result = govern(_evidence(comparison_outcome=mapped, **overrides))

    assert result.decision is Decision.SCORE_INTERPRETATION_NOT_SUPPORTED
    assert result.selected_candidate_id is None


def test_materially_better_supported_complex_candidate_can_be_selected() -> None:
    """A completed admissible comparison may select the supported complex candidate."""
    Outcome, Decision, _, govern = _surface()

    result = govern(_evidence(comparison_outcome=Outcome.FAVORS_MORE_COMPLEX))

    assert result.decision is Decision.SELECTED
    assert result.selected_candidate_id == "bifactor_candidate"
    assert result.reason_code == "comparison_favors_more_complex"


def test_comparison_result_cannot_precede_required_distinguishability() -> None:
    """Caller-supplied comparison labels cannot bypass the relation procedure."""
    Outcome, _, _, govern = _surface()

    with pytest.raises(ValueError, match="cannot precede"):
        govern(
            _evidence(
                relation_evidence=ModelRelationEvidence(
                    parameter_embedding=False,
                    parameter_spaces_overlap=False,
                ),
                comparison_outcome=Outcome.FAVORS_SIMPLER,
            )
        )


def test_candidate_identifiers_reject_str_subclasses_without_callbacks() -> None:
    """Identifier validation must not execute caller-defined string methods."""
    _, _, Evidence, _ = _surface()
    callbacks = 0

    class HostileStr(str):
        """String subclass whose normalization callback must remain unreachable."""

        def strip(self, chars=None):  # type: ignore[override]
            """Record an unexpected callback instead of normalizing text."""
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller callback executed")

    with pytest.raises(ValueError, match="candidate_id"):
        Evidence(
            simpler_candidate_id=HostileStr("correlated_traits"),
            more_complex_candidate_id="bifactor_candidate",
            relation_evidence=ModelRelationEvidence(parameter_embedding=True),
            comparison_outcome=None,
            simpler_score_interpretation_supported=True,
            more_complex_score_interpretation_supported=True,
            recovery_evidence_sufficient=True,
        )

    assert callbacks == 0


@pytest.mark.parametrize(
    "field_name",
    [
        "simpler_score_interpretation_supported",
        "more_complex_score_interpretation_supported",
        "recovery_evidence_sufficient",
    ],
)
def test_governance_flags_require_exact_booleans(field_name: str) -> None:
    """Integer truthiness must not silently change structural-selection policy."""
    with pytest.raises(TypeError, match=field_name):
        _evidence(**{field_name: 1})


def test_candidate_identifiers_must_be_distinct() -> None:
    """A pairwise decision requires two distinct candidate identities."""
    with pytest.raises(ValueError, match="distinct"):
        _evidence(more_complex_candidate_id="correlated_traits")
