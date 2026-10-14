"""Governed essay validation evidence without universal acceptance claims.

The adapter binds one exact assessment, criterion, dataset, automated engine, and
human-reference engine to descriptive metrics computed by the existing Rust
agreement kernel. It intentionally ignores the legacy kernel's hard-coded gate
booleans and performs no scoring, calibration, aggregation, validity inference,
fairness conclusion, or deployment authorization in Python.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
import math
from typing import Any

import numpy as np

from fast_mlsirm.validation import ValidationVerdict, validate_judge

from .._contract_safety import artifact_digest, freeze_metadata, sorted_identifiers
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    fingerprint,
    thaw_json_value,
)
from ..assessment import AssessmentSpec
from ..execution import EngineDescriptor, EngineKind

MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS = 64

_REPORT_TOKEN = object()
_METRIC_TOKEN = object()

_METRIC_NAME_MAP = {
    "qwk": "quadratic_weighted_kappa",
    "pearson_r": "pearson_correlation",
    "smd": "standardized_mean_difference",
    "degradation": "human_machine_degradation",
    "subgroup_smd": "worst_subgroup_standardized_mean_difference",
}
_METRIC_INTERPRETATION = {
    "quadratic_weighted_kappa": "descriptive_agreement_evidence",
    "pearson_correlation": "descriptive_association_only",
    "standardized_mean_difference": "descriptive_location_difference",
    "human_machine_degradation": "descriptive_baseline_difference",
    "worst_subgroup_standardized_mean_difference": "descriptive_subgroup_difference",
    "exact_agreement": "descriptive_agreement_evidence",
    "adjacent_agreement": "descriptive_agreement_evidence",
}
_INTERPRETATION_BOUNDARY_IDS = (
    "correlation_is_descriptive_only",
    "legacy_gate_booleans_are_not_validity_evidence",
    "metric_thresholds_are_not_universal",
    "human_validation_is_required",
    "report_does_not_authorize_deployment",
)


def _finite_metric_value(value: Any, metric_id: str) -> float:
    """Return one finite Rust-produced metric value or fail closed."""
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            "invalid_essay_validation_metric",
            f"$.metrics.{metric_id}",
            "validation metric values must be finite real numbers",
        ) from None
    if not math.isfinite(normalized):
        raise assessment_error(
            "nonfinite_essay_validation_metric",
            f"$.metrics.{metric_id}",
            "validation metric values must be finite real numbers",
        )
    return normalized


@dataclass(frozen=True)
class EssayValidationMetric(CanonicalContract):
    """One descriptive Rust-computed validation metric without a pass decision."""

    metric_id: str
    value: float
    interpretation_id: str
    _metric_token: InitVar[object | None] = None

    def __post_init__(self, _metric_token: object | None) -> None:
        """Reject direct construction and normalize metric evidence."""
        if _metric_token is not _METRIC_TOKEN:
            raise assessment_error(
                "unverified_essay_validation_metric",
                "$",
                "validation metrics must be created by the report factory",
            )
        object.__setattr__(
            self,
            "metric_id",
            descriptive_identifier(self.metric_id, "metric_id"),
        )
        object.__setattr__(
            self,
            "value",
            _finite_metric_value(self.value, self.metric_id),
        )
        object.__setattr__(
            self,
            "interpretation_id",
            descriptive_identifier(self.interpretation_id, "interpretation_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible metric evidence."""
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "interpretation_id": self.interpretation_id,
        }

    _content_dict = to_dict


def _metric(metric_id: str, value: Any) -> EssayValidationMetric:
    """Create one sealed metric with its conservative interpretation boundary."""
    return EssayValidationMetric(
        metric_id=metric_id,
        value=value,
        interpretation_id=_METRIC_INTERPRETATION[metric_id],
        _metric_token=_METRIC_TOKEN,
    )


def _metrics_from_verdict(verdict: ValidationVerdict) -> tuple[EssayValidationMetric, ...]:
    """Copy Rust outputs while discarding legacy threshold and pass fields."""
    if not isinstance(verdict, ValidationVerdict):
        raise assessment_error(
            "invalid_essay_validation_verdict",
            "$.validation_verdict",
            "Rust validation must return ValidationVerdict",
        )
    output: list[EssayValidationMetric] = []
    observed_ids: set[str] = set()
    for index, gate in enumerate(verdict.gates):
        try:
            legacy_name = gate["name"]
            value = gate["value"]
        except (KeyError, TypeError):
            raise assessment_error(
                "invalid_essay_validation_gate",
                f"$.validation_verdict.gates[{index}]",
                "Rust validation gates must expose name and value",
            ) from None
        metric_id = _METRIC_NAME_MAP.get(legacy_name)
        if metric_id is None:
            raise assessment_error(
                "unknown_essay_validation_metric",
                f"$.validation_verdict.gates[{index}].name",
                "Rust validation returned an unsupported metric identity",
            )
        if metric_id in observed_ids:
            raise assessment_error(
                "duplicate_essay_validation_metric",
                f"$.validation_verdict.gates[{index}].name",
                "Rust validation returned a duplicate metric identity",
            )
        observed_ids.add(metric_id)
        output.append(_metric(metric_id, value))
    output.extend(
        (
            _metric("exact_agreement", verdict.exact_agreement),
            _metric("adjacent_agreement", verdict.adjacent_agreement),
        )
    )
    return tuple(sorted(output, key=lambda item: item.metric_id))


def _validate_scope(
    assessment: AssessmentSpec,
    construct_id: str,
    rubric_fingerprint: str,
) -> None:
    """Verify exact construct, rubric, and validation-policy scope bindings."""
    construct = next(
        (
            candidate
            for candidate in assessment.constructs
            if candidate.construct_id == construct_id
        ),
        None,
    )
    if construct is None:
        raise assessment_error(
            "essay_validation_construct_mismatch",
            "$.construct_id",
            "construct_id is not declared by the assessment",
        )
    if rubric_fingerprint not in construct.rubric_fingerprints:
        raise assessment_error(
            "essay_validation_rubric_mismatch",
            "$.rubric_fingerprint",
            "rubric_fingerprint is not bound to the selected construct",
        )
    policy_scope = assessment.validation_policy.construct_ids
    if policy_scope and construct_id not in policy_scope:
        raise assessment_error(
            "essay_validation_policy_scope_mismatch",
            "$.assessment.validation_policy.construct_ids",
            "validation policy does not cover the selected construct",
        )


def _validate_engines(
    assessment: AssessmentSpec,
    automated_engine: EngineDescriptor,
    reference_engine: EngineDescriptor,
) -> None:
    """Verify automated and human engine kinds and assessment authorization."""
    if automated_engine.engine_kind is not EngineKind.AUTOMATED:
        raise assessment_error(
            "invalid_essay_validation_automated_engine",
            "$.automated_engine",
            "automated_engine must have automated_engine kind",
        )
    if reference_engine.engine_kind is not EngineKind.HUMAN:
        raise assessment_error(
            "invalid_essay_validation_reference_engine",
            "$.reference_engine",
            "reference_engine must have human_engine kind",
        )
    policy = assessment.engine_policy
    if not policy.allow_automated_raters:
        raise assessment_error(
            "essay_validation_automated_engine_disabled",
            "$.assessment.engine_policy",
            "assessment policy disables automated raters",
        )
    if not policy.allow_human_raters:
        raise assessment_error(
            "essay_validation_human_reference_disabled",
            "$.assessment.engine_policy",
            "assessment policy disables human raters",
        )
    if automated_engine.engine_id not in policy.engine_ids:
        raise assessment_error(
            "essay_validation_engine_not_authorized",
            "$.automated_engine.engine_id",
            "automated engine is not authorized by the assessment policy",
        )
    if automated_engine.engine_fingerprint == reference_engine.engine_fingerprint:
        raise assessment_error(
            "essay_validation_engine_identity_collision",
            "$.reference_engine.engine_fingerprint",
            "automated and reference engines must have distinct identities",
        )


@dataclass(frozen=True)
class EssayValidationEvidenceReport(CanonicalContract):
    """Factory-sealed criterion validation evidence with no overall pass verdict."""

    report_id: str
    assessment_spec: AssessmentSpec
    construct_id: str
    rubric_fingerprint: str
    criterion_id: str
    automated_engine: EngineDescriptor
    reference_engine: EngineDescriptor
    validation_dataset_fingerprint: str
    category_count: int
    paired_observation_count: int
    metrics: tuple[EssayValidationMetric, ...]
    review_trigger_ids: tuple[str, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _report_token: InitVar[object | None] = None

    def __post_init__(self, _report_token: object | None) -> None:
        """Reject direct construction and normalize audit-only report fields."""
        if _report_token is not _REPORT_TOKEN:
            raise assessment_error(
                "unverified_essay_validation_evidence_report",
                "$",
                "use build_essay_validation_evidence_report",
            )
        object.__setattr__(
            self,
            "report_id",
            descriptive_identifier(self.report_id, "report_id"),
        )
        object.__setattr__(
            self,
            "construct_id",
            descriptive_identifier(self.construct_id, "construct_id"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprint",
            fingerprint(self.rubric_fingerprint, "rubric_fingerprint"),
        )
        object.__setattr__(
            self,
            "criterion_id",
            descriptive_identifier(self.criterion_id, "criterion_id"),
        )
        object.__setattr__(
            self,
            "validation_dataset_fingerprint",
            fingerprint(
                self.validation_dataset_fingerprint,
                "validation_dataset_fingerprint",
            ),
        )
        if type(self.category_count) is not int or not 2 <= self.category_count <= 1_000:
            raise assessment_error(
                "invalid_essay_validation_category_count",
                "$.category_count",
                "category_count must be an exact integer between 2 and 1000",
            )
        if type(self.paired_observation_count) is not int or self.paired_observation_count < 2:
            raise assessment_error(
                "invalid_essay_validation_observation_count",
                "$.paired_observation_count",
                "paired_observation_count must be an exact integer >= 2",
            )
        if not isinstance(self.metrics, tuple) or not self.metrics:
            raise assessment_error(
                "invalid_essay_validation_metrics",
                "$.metrics",
                "metrics must remain a non-empty immutable tuple",
            )
        for index, metric in enumerate(self.metrics):
            if not isinstance(metric, EssayValidationMetric):
                raise assessment_error(
                    "invalid_essay_validation_metric",
                    f"$.metrics[{index}]",
                    "metrics must remain EssayValidationMetric values",
                )
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise assessment_error(
                "invalid_essay_validation_metric_order",
                "$.metrics",
                "metric identities must be unique and sorted",
            )
        object.__setattr__(
            self,
            "review_trigger_ids",
            sorted_identifiers(
                self.review_trigger_ids,
                "review_trigger_ids",
                minimum=1,
                maximum=MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    @property
    def metric_ids(self) -> tuple[str, ...]:
        """Return descriptive metric identities in canonical order."""
        return tuple(metric.metric_id for metric in self.metrics)

    @property
    def interpretation_boundary_ids(self) -> tuple[str, ...]:
        """Return non-suppressible scientific interpretation boundaries."""
        return _INTERPRETATION_BOUNDARY_IDS

    @property
    def human_review_required(self) -> bool:
        """Return true because validation evidence requires human interpretation."""
        return True

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical evidence content without derived public identities."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "assessment_spec": self.assessment_spec.to_dict(),
            "construct_id": self.construct_id,
            "rubric_fingerprint": self.rubric_fingerprint,
            "criterion_id": self.criterion_id,
            "automated_engine": self.automated_engine.to_dict(),
            "reference_engine": self.reference_engine.to_dict(),
            "validation_dataset_fingerprint": self.validation_dataset_fingerprint,
            "category_count": self.category_count,
            "paired_observation_count": self.paired_observation_count,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metric_ids": list(self.metric_ids),
            "review_trigger_ids": list(self.review_trigger_ids),
            "human_review_required": self.human_review_required,
            "interpretation_boundary_ids": list(self.interpretation_boundary_ids),
            "rust_backend_function_id": "mlsirm_core_agreement_validate_scoring",
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def report_fingerprint(self) -> str:
        """Return SHA-256 over the complete normalized validation evidence."""
        return artifact_digest(self)

    @property
    def report_handle(self) -> str:
        """Return a descriptive 128-bit public validation-report handle."""
        return f"essay_validation_report_{self.report_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical evidence and deterministic public identities."""
        return {
            **self._content_dict(),
            "report_handle": self.report_handle,
            "report_fingerprint": self.report_fingerprint,
        }


def build_essay_validation_evidence_report(
    *,
    report_id: str,
    assessment: AssessmentSpec,
    construct_id: str,
    rubric_fingerprint: str,
    criterion_id: str,
    automated_engine: EngineDescriptor,
    reference_engine: EngineDescriptor,
    validation_dataset_fingerprint: str,
    automated_labels: np.ndarray,
    reference_labels: np.ndarray,
    category_count: int,
    human_human_labels: tuple[np.ndarray, np.ndarray] | None = None,
    subgroup_labels: np.ndarray | None = None,
    additional_review_trigger_ids: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssayValidationEvidenceReport:
    """Build criterion-specific descriptive evidence through the Rust kernel.

    The existing Rust agreement implementation computes all metric arithmetic.
    This adapter copies only metric values and deliberately discards its legacy
    threshold and pass fields. The report therefore cannot establish construct
    validity, fairness, reliability, causal utility, or deployment readiness.
    """
    if not isinstance(assessment, AssessmentSpec):
        raise assessment_error(
            "invalid_essay_validation_assessment",
            "$.assessment",
            "assessment must be an AssessmentSpec",
        )
    if not isinstance(automated_engine, EngineDescriptor):
        raise assessment_error(
            "invalid_essay_validation_automated_engine",
            "$.automated_engine",
            "automated_engine must be an EngineDescriptor",
        )
    if not isinstance(reference_engine, EngineDescriptor):
        raise assessment_error(
            "invalid_essay_validation_reference_engine",
            "$.reference_engine",
            "reference_engine must be an EngineDescriptor",
        )
    normalized_construct = descriptive_identifier(construct_id, "construct_id")
    normalized_rubric = fingerprint(rubric_fingerprint, "rubric_fingerprint")
    _validate_scope(assessment, normalized_construct, normalized_rubric)
    _validate_engines(assessment, automated_engine, reference_engine)
    if type(category_count) is not int or not 2 <= category_count <= 1_000:
        raise assessment_error(
            "invalid_essay_validation_category_count",
            "$.category_count",
            "category_count must be an exact integer between 2 and 1000",
        )

    verdict = validate_judge(
        automated_labels,
        reference_labels,
        k=category_count,
        human_human=human_human_labels,
        subgroup=subgroup_labels,
    )
    metrics = _metrics_from_verdict(verdict)
    computed_metric_ids = {metric.metric_id for metric in metrics}
    declared_metric_ids = set(assessment.validation_policy.metric_ids)
    undeclared = computed_metric_ids.difference(declared_metric_ids)
    if undeclared:
        raise assessment_error(
            "essay_validation_metric_not_declared",
            "$.assessment.validation_policy.metric_ids",
            "validation evidence contains metrics not declared by policy: "
            + ", ".join(sorted(undeclared)),
        )

    labels = np.asarray(automated_labels)
    paired_count = int(labels.shape[0])
    mandatory = {
        "human_validation_required",
        "correlation_descriptive_only",
    }
    if human_human_labels is None:
        mandatory.add("human_human_baseline_missing")
    if subgroup_labels is None:
        mandatory.add("subgroup_evidence_missing")
    additional = sorted_identifiers(
        additional_review_trigger_ids,
        "additional_review_trigger_ids",
        minimum=0,
        maximum=MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS,
    )
    triggers = sorted_identifiers(
        tuple(mandatory.union(additional)),
        "review_trigger_ids",
        minimum=1,
        maximum=MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS,
    )
    return EssayValidationEvidenceReport(
        report_id=report_id,
        assessment_spec=assessment,
        construct_id=normalized_construct,
        rubric_fingerprint=normalized_rubric,
        criterion_id=criterion_id,
        automated_engine=automated_engine,
        reference_engine=reference_engine,
        validation_dataset_fingerprint=validation_dataset_fingerprint,
        category_count=category_count,
        paired_observation_count=paired_count,
        metrics=metrics,
        review_trigger_ids=triggers,
        metadata={} if metadata is None else metadata,
        _report_token=_REPORT_TOKEN,
    )


__all__ = [
    "MAX_ESSAY_VALIDATION_REVIEW_TRIGGERS",
    "EssayValidationEvidenceReport",
    "EssayValidationMetric",
    "build_essay_validation_evidence_report",
]
