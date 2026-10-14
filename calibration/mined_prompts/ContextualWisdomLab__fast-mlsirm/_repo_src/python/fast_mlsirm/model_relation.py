"""Classify structural measurement-model relations from explicit constraint facts.

This module does not infer relation from model names and performs no likelihood
or test-statistic arithmetic. It turns caller-supplied parameter-space,
boundary, constraint, overlap, and formal distinguishability facts into the
comparison procedure that is scientifically admissible. Unknown or
indistinguishable relations fail closed rather than forcing a winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeasurementModelRelation(str, Enum):
    """Supported structural relations between two measurement models."""

    REGULAR_NESTED = "regular_nested"
    BOUNDARY_NESTED = "boundary_nested"
    NONLINEAR_CONSTRAINT_NESTED = "nonlinear_constraint_nested"
    STRICTLY_NON_NESTED = "strictly_non_nested"
    OVERLAPPING = "overlapping"
    INDISTINGUISHABLE = "indistinguishable"
    UNKNOWN = "unknown"


class ModelComparisonProcedure(str, Enum):
    """Required next procedure for one governed structural relation."""

    LIKELIHOOD_RATIO = "likelihood_ratio"
    PARAMETRIC_BOOTSTRAP_LR = "parametric_bootstrap_lr"
    VUONG_DISTINGUISHABILITY = "vuong_distinguishability"
    VUONG_SELECTION = "vuong_selection"
    RELATION_CLASSIFICATION = "relation_classification"
    NO_SELECTION = "no_selection"


def _require_optional_bool(value: object, name: str) -> None:
    """Reject truthy substitutes for an optional exact Boolean fact."""
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{name} must be bool or None")


def _require_bool(value: object, name: str) -> None:
    """Reject truthy substitutes for an exact Boolean fact."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class ModelRelationEvidence:
    """Explicit constraint facts used to classify a model pair.

    ``parameter_embedding`` states whether one complete parameter space is
    embedded in the other. Boundary, unidentified-null, and nonlinear facts are
    meaningful only for an embedded pair. ``parameter_spaces_overlap`` and a
    formal distinguishability result are meaningful only for a non-embedded
    pair. ``None`` represents evidence that has not yet been established.
    """

    parameter_embedding: bool | None
    null_on_boundary: bool = False
    unidentified_under_null: bool = False
    nonlinear_constraints: bool = False
    parameter_spaces_overlap: bool | None = None
    formal_distinguishability: bool | None = None

    def __post_init__(self) -> None:
        """Reject untyped or internally contradictory relation evidence."""
        _require_optional_bool(self.parameter_embedding, "parameter_embedding")
        _require_bool(self.null_on_boundary, "null_on_boundary")
        _require_bool(self.unidentified_under_null, "unidentified_under_null")
        _require_bool(self.nonlinear_constraints, "nonlinear_constraints")
        _require_optional_bool(
            self.parameter_spaces_overlap,
            "parameter_spaces_overlap",
        )
        _require_optional_bool(
            self.formal_distinguishability,
            "formal_distinguishability",
        )

        nested_facts = (
            self.null_on_boundary
            or self.unidentified_under_null
            or self.nonlinear_constraints
        )
        if self.parameter_embedding is True:
            if (
                self.parameter_spaces_overlap is not None
                or self.formal_distinguishability is not None
            ):
                raise ValueError(
                    "relation evidence for an embedded pair cannot include "
                    "nonnested overlap or distinguishability facts"
                )
            return

        if self.parameter_embedding is False:
            if nested_facts:
                raise ValueError(
                    "relation evidence for a non-embedded pair cannot include "
                    "nested boundary or constraint facts"
                )
            if (
                self.formal_distinguishability is not None
                and self.parameter_spaces_overlap is None
            ):
                raise ValueError(
                    "relation evidence requires overlap classification before "
                    "formal distinguishability"
                )
            return

        if (
            nested_facts
            or self.parameter_spaces_overlap is not None
            or self.formal_distinguishability is not None
        ):
            raise ValueError(
                "relation evidence requires parameter_embedding before other facts"
            )


@dataclass(frozen=True, slots=True)
class ModelRelationResult:
    """Governed relation and the only admissible next comparison procedure."""

    relation: MeasurementModelRelation
    required_procedure: ModelComparisonProcedure
    selection_permitted: bool
    reason_code: str


def classify_model_relation(evidence: ModelRelationEvidence) -> ModelRelationResult:
    """Classify a model pair without using names or computing a test statistic.

    Boundary or unidentified-null evidence takes conservative precedence when an
    embedded pair also carries nonlinear constraints. For non-embedded pairs,
    an A/B preference remains prohibited until a formal Vuong distinguishability
    test succeeds.
    """
    if type(evidence) is not ModelRelationEvidence:
        raise TypeError("evidence must be ModelRelationEvidence")
    evidence.__post_init__()

    if evidence.parameter_embedding is None:
        return ModelRelationResult(
            relation=MeasurementModelRelation.UNKNOWN,
            required_procedure=ModelComparisonProcedure.RELATION_CLASSIFICATION,
            selection_permitted=False,
            reason_code="requires_relation_classification",
        )

    if evidence.parameter_embedding:
        if evidence.null_on_boundary or evidence.unidentified_under_null:
            return ModelRelationResult(
                relation=MeasurementModelRelation.BOUNDARY_NESTED,
                required_procedure=ModelComparisonProcedure.PARAMETRIC_BOOTSTRAP_LR,
                selection_permitted=True,
                reason_code="nonregular_nested_relation",
            )
        if evidence.nonlinear_constraints:
            return ModelRelationResult(
                relation=MeasurementModelRelation.NONLINEAR_CONSTRAINT_NESTED,
                required_procedure=ModelComparisonProcedure.PARAMETRIC_BOOTSTRAP_LR,
                selection_permitted=True,
                reason_code="nonlinear_constraint_relation",
            )
        return ModelRelationResult(
            relation=MeasurementModelRelation.REGULAR_NESTED,
            required_procedure=ModelComparisonProcedure.LIKELIHOOD_RATIO,
            selection_permitted=True,
            reason_code="regular_nested_relation",
        )

    if evidence.parameter_spaces_overlap is None:
        return ModelRelationResult(
            relation=MeasurementModelRelation.UNKNOWN,
            required_procedure=ModelComparisonProcedure.RELATION_CLASSIFICATION,
            selection_permitted=False,
            reason_code="requires_relation_classification",
        )

    relation = (
        MeasurementModelRelation.OVERLAPPING
        if evidence.parameter_spaces_overlap
        else MeasurementModelRelation.STRICTLY_NON_NESTED
    )
    if evidence.formal_distinguishability is None:
        return ModelRelationResult(
            relation=relation,
            required_procedure=ModelComparisonProcedure.VUONG_DISTINGUISHABILITY,
            selection_permitted=False,
            reason_code="requires_distinguishability_test",
        )
    if not evidence.formal_distinguishability:
        return ModelRelationResult(
            relation=MeasurementModelRelation.INDISTINGUISHABLE,
            required_procedure=ModelComparisonProcedure.NO_SELECTION,
            selection_permitted=False,
            reason_code="indistinguishable",
        )
    return ModelRelationResult(
        relation=relation,
        required_procedure=ModelComparisonProcedure.VUONG_SELECTION,
        selection_permitted=True,
        reason_code="formally_distinguishable",
    )


__all__ = [
    "MeasurementModelRelation",
    "ModelComparisonProcedure",
    "ModelRelationEvidence",
    "ModelRelationResult",
    "classify_model_relation",
]
