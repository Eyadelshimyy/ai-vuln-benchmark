"""Govern structural measurement-model selection from already-computed evidence.

Factor retention and structural model selection are intentionally separate
workflows. This module performs no likelihood, prediction, recovery, scoreability,
or test-statistic arithmetic. It combines explicit relation facts with an
already-computed relation-appropriate pairwise outcome and conservative
interpretation/recovery gates. Numerical procedures remain Rust-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model_relation import (
    MeasurementModelRelation,
    ModelComparisonProcedure,
    ModelRelationEvidence,
    classify_model_relation,
)


class StructuralComparisonOutcome(str, Enum):
    """Already-computed substantive outcome of the admissible pairwise procedure."""

    FAVORS_SIMPLER = "favors_simpler"
    PRACTICALLY_EQUIVALENT = "practically_equivalent"
    FAVORS_MORE_COMPLEX = "favors_more_complex"


class StructuralSelectionDecision(str, Enum):
    """Fail-closed public decisions for one structural model pair."""

    SELECTED = "selected"
    PRACTICALLY_EQUIVALENT_PREFER_SIMPLER = (
        "practically_equivalent_prefer_simpler"
    )
    INDISTINGUISHABLE = "indistinguishable"
    REQUIRES_RELATION_CLASSIFICATION = "requires_relation_classification"
    REQUIRES_LIKELIHOOD_RATIO = "requires_likelihood_ratio"
    REQUIRES_DISTINGUISHABILITY_TEST = "requires_distinguishability_test"
    REQUIRES_VUONG_SELECTION = "requires_vuong_selection"
    INSUFFICIENT_RECOVERY_EVIDENCE = "insufficient_recovery_evidence"
    SCORE_INTERPRETATION_NOT_SUPPORTED = "score_interpretation_not_supported"


def _require_candidate_id(value: object, name: str) -> str:
    """Return one exact built-in candidate identifier without caller callbacks."""
    if type(value) is not str:
        raise ValueError(f"{name} candidate_id must be an exact built-in string")
    if not value.strip():
        raise ValueError(f"{name} candidate_id must be non-empty")
    return value


def _require_exact_bool(value: object, name: str) -> bool:
    """Return one policy fact after rejecting integer truthiness substitutes."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


@dataclass(frozen=True, slots=True)
class StructuralSelectionEvidence:
    """Governed evidence needed to interpret one simpler/complex model pair.

    ``relation_evidence`` describes the actual parameter-space relationship and
    is classified by :func:`fast_mlsirm.model_relation.classify_model_relation`.
    ``comparison_outcome`` is deliberately an already-computed result from the
    procedure required by that classifier; this transport never computes an LR,
    bootstrap, Vuong statistic, predictive difference, or practical-equivalence
    threshold in Python.

    Score-interpretation and recovery flags state whether their separate
    governed evidence has passed. They are not inferred from fit preference.
    """

    simpler_candidate_id: str
    more_complex_candidate_id: str
    relation_evidence: ModelRelationEvidence
    comparison_outcome: StructuralComparisonOutcome | None
    simpler_score_interpretation_supported: bool
    more_complex_score_interpretation_supported: bool
    recovery_evidence_sufficient: bool

    def __post_init__(self) -> None:
        """Fail closed on untrusted identifiers, flags, and relation transports."""
        simpler_id = _require_candidate_id(
            self.simpler_candidate_id,
            "simpler",
        )
        complex_id = _require_candidate_id(
            self.more_complex_candidate_id,
            "more_complex",
        )
        if simpler_id == complex_id:
            raise ValueError("structural candidate identifiers must be distinct")
        if type(self.relation_evidence) is not ModelRelationEvidence:
            raise TypeError("relation_evidence must be ModelRelationEvidence")
        if self.comparison_outcome is not None and not isinstance(
            self.comparison_outcome,
            StructuralComparisonOutcome,
        ):
            raise TypeError(
                "comparison_outcome must be StructuralComparisonOutcome or None"
            )
        _require_exact_bool(
            self.simpler_score_interpretation_supported,
            "simpler_score_interpretation_supported",
        )
        _require_exact_bool(
            self.more_complex_score_interpretation_supported,
            "more_complex_score_interpretation_supported",
        )
        _require_exact_bool(
            self.recovery_evidence_sufficient,
            "recovery_evidence_sufficient",
        )


@dataclass(frozen=True, slots=True)
class StructuralSelectionResult:
    """Conservative structural-selection decision and procedure provenance."""

    decision: StructuralSelectionDecision
    selected_candidate_id: str | None
    required_procedure: ModelComparisonProcedure
    reason_code: str


def _pending_relation_decision(
    relation: MeasurementModelRelation,
    procedure: ModelComparisonProcedure,
) -> StructuralSelectionDecision | None:
    """Map incomplete relation/procedure evidence to a fail-closed decision."""
    if relation is MeasurementModelRelation.UNKNOWN:
        return StructuralSelectionDecision.REQUIRES_RELATION_CLASSIFICATION
    if relation is MeasurementModelRelation.INDISTINGUISHABLE:
        return StructuralSelectionDecision.INDISTINGUISHABLE
    if procedure is ModelComparisonProcedure.VUONG_DISTINGUISHABILITY:
        return StructuralSelectionDecision.REQUIRES_DISTINGUISHABILITY_TEST
    return None


def _missing_comparison_decision(
    procedure: ModelComparisonProcedure,
) -> StructuralSelectionDecision:
    """Return the next required comparison evidence when no outcome is supplied."""
    if procedure in {
        ModelComparisonProcedure.LIKELIHOOD_RATIO,
        ModelComparisonProcedure.PARAMETRIC_BOOTSTRAP_LR,
    }:
        return StructuralSelectionDecision.REQUIRES_LIKELIHOOD_RATIO
    if procedure is ModelComparisonProcedure.VUONG_SELECTION:
        return StructuralSelectionDecision.REQUIRES_VUONG_SELECTION
    raise ValueError("relation procedure does not admit a pairwise comparison outcome")


def govern_structural_selection(
    evidence: StructuralSelectionEvidence,
) -> StructuralSelectionResult:
    """Return a conservative selection decision without performing psychometric math.

    The relation classifier determines which numerical comparison is admissible.
    Callers may supply a comparison outcome only after that relation permits the
    corresponding LR/bootstrap/Vuong-selection procedure. Recovery and intended
    score interpretation are independent gates. Practical equivalence selects
    the simpler candidate only when its intended score interpretation is
    supported; otherwise a supported complex candidate may be retained.
    """
    if type(evidence) is not StructuralSelectionEvidence:
        raise TypeError("evidence must be StructuralSelectionEvidence")
    simpler_id = _require_candidate_id(evidence.simpler_candidate_id, "simpler")
    complex_id = _require_candidate_id(
        evidence.more_complex_candidate_id,
        "more_complex",
    )
    if simpler_id == complex_id:
        raise ValueError("structural candidate identifiers must be distinct")
    if type(evidence.relation_evidence) is not ModelRelationEvidence:
        raise TypeError("relation_evidence must be ModelRelationEvidence")
    _require_exact_bool(
        evidence.simpler_score_interpretation_supported,
        "simpler_score_interpretation_supported",
    )
    _require_exact_bool(
        evidence.more_complex_score_interpretation_supported,
        "more_complex_score_interpretation_supported",
    )
    _require_exact_bool(
        evidence.recovery_evidence_sufficient,
        "recovery_evidence_sufficient",
    )
    evidence.relation_evidence.__post_init__()

    relation = classify_model_relation(evidence.relation_evidence)
    pending = _pending_relation_decision(
        relation.relation,
        relation.required_procedure,
    )
    if pending is not None:
        if evidence.comparison_outcome is not None:
            raise ValueError(
                "comparison_outcome cannot precede the required relation procedure"
            )
        return StructuralSelectionResult(
            decision=pending,
            selected_candidate_id=None,
            required_procedure=relation.required_procedure,
            reason_code=(
                "indistinguishable"
                if pending is StructuralSelectionDecision.INDISTINGUISHABLE
                else relation.reason_code
            ),
        )

    if evidence.comparison_outcome is None:
        decision = _missing_comparison_decision(relation.required_procedure)
        return StructuralSelectionResult(
            decision=decision,
            selected_candidate_id=None,
            required_procedure=relation.required_procedure,
            reason_code=decision.value,
        )

    if not relation.selection_permitted:
        raise ValueError(
            "comparison_outcome cannot precede the required relation procedure"
        )

    if not evidence.recovery_evidence_sufficient:
        return StructuralSelectionResult(
            decision=StructuralSelectionDecision.INSUFFICIENT_RECOVERY_EVIDENCE,
            selected_candidate_id=None,
            required_procedure=relation.required_procedure,
            reason_code="insufficient_recovery_evidence",
        )

    outcome = evidence.comparison_outcome
    if outcome is StructuralComparisonOutcome.FAVORS_SIMPLER:
        if not evidence.simpler_score_interpretation_supported:
            return StructuralSelectionResult(
                decision=StructuralSelectionDecision.SCORE_INTERPRETATION_NOT_SUPPORTED,
                selected_candidate_id=None,
                required_procedure=relation.required_procedure,
                reason_code="simpler_score_interpretation_not_supported",
            )
        return StructuralSelectionResult(
            decision=StructuralSelectionDecision.SELECTED,
            selected_candidate_id=evidence.simpler_candidate_id,
            required_procedure=relation.required_procedure,
            reason_code="comparison_favors_simpler",
        )

    if outcome is StructuralComparisonOutcome.PRACTICALLY_EQUIVALENT:
        if evidence.simpler_score_interpretation_supported:
            return StructuralSelectionResult(
                decision=(
                    StructuralSelectionDecision.PRACTICALLY_EQUIVALENT_PREFER_SIMPLER
                ),
                selected_candidate_id=evidence.simpler_candidate_id,
                required_procedure=relation.required_procedure,
                reason_code="practically_equivalent_prefer_simpler",
            )
        if evidence.more_complex_score_interpretation_supported:
            return StructuralSelectionResult(
                decision=StructuralSelectionDecision.SELECTED,
                selected_candidate_id=evidence.more_complex_candidate_id,
                required_procedure=relation.required_procedure,
                reason_code="simpler_score_interpretation_not_supported",
            )
        return StructuralSelectionResult(
            decision=StructuralSelectionDecision.SCORE_INTERPRETATION_NOT_SUPPORTED,
            selected_candidate_id=None,
            required_procedure=relation.required_procedure,
            reason_code="score_interpretation_not_supported",
        )

    if outcome is StructuralComparisonOutcome.FAVORS_MORE_COMPLEX:
        if not evidence.more_complex_score_interpretation_supported:
            return StructuralSelectionResult(
                decision=StructuralSelectionDecision.SCORE_INTERPRETATION_NOT_SUPPORTED,
                selected_candidate_id=None,
                required_procedure=relation.required_procedure,
                reason_code="more_complex_score_interpretation_not_supported",
            )
        return StructuralSelectionResult(
            decision=StructuralSelectionDecision.SELECTED,
            selected_candidate_id=evidence.more_complex_candidate_id,
            required_procedure=relation.required_procedure,
            reason_code="comparison_favors_more_complex",
        )

    raise ValueError("unsupported structural comparison outcome")


__all__ = [
    "StructuralComparisonOutcome",
    "StructuralSelectionDecision",
    "StructuralSelectionEvidence",
    "StructuralSelectionResult",
    "govern_structural_selection",
]
