"""Governed essay many-facet calibration result reports.

The adapter binds one exact criterion-specific scoring-facets design to copied
outputs from the existing Rust-backed many-facet estimator. It validates,
marshals, serializes, and routes human review only. It performs no likelihood,
gradient, Hessian, optimization, scoring, ranking, utility, fairness, validity,
or causal arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from typing import Any

import numpy as np

from ...facets import FacetsFit
from .._contract_safety import artifact_digest, freeze_metadata, sorted_identifiers
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    fingerprint,
    strict_boolean,
    thaw_json_value,
)
from ..calibration import ScoringFacetsDesign, fit_scoring_facets_design

MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS = 64

_REPORT_TOKEN = object()


def _exact_integer(value: Any, field_name: str, *, minimum: int) -> int:
    """Return one bounded exact integer or raise a stable contract error."""
    if type(value) is not int or value < minimum:
        raise assessment_error(
            f"invalid_{field_name}",
            f"$.{field_name}",
            f"{field_name} must be an exact integer >= {minimum}",
        )
    return value


def _finite_vector(
    value: Any,
    field_name: str,
    *,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    """Copy one finite one-dimensional numeric vector into an immutable tuple."""
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise assessment_error(
            f"invalid_{field_name}",
            f"$.{field_name}",
            f"{field_name} must be a one-dimensional numeric vector",
        )
    copied = np.array(raw, dtype=np.float64, copy=True)
    if expected_length is not None and copied.size != expected_length:
        raise assessment_error(
            f"invalid_{field_name}_length",
            f"$.{field_name}",
            f"{field_name} length must equal {expected_length}",
        )
    if copied.size == 0:
        raise assessment_error(
            f"empty_{field_name}",
            f"$.{field_name}",
            f"{field_name} must not be empty",
        )
    if not np.all(np.isfinite(copied)):
        raise assessment_error(
            f"nonfinite_{field_name}",
            f"$.{field_name}",
            f"{field_name} must contain only finite values",
        )
    return tuple(float(item) for item in copied)


def _validate_loglik_trace(values: tuple[float, ...]) -> None:
    """Reject material likelihood decreases as an estimator replay failure."""
    for index, (previous, current) in enumerate(
        zip(values, values[1:], strict=False),
        start=1,
    ):
        tolerance = 1e-8 * (1.0 + abs(previous))
        if current + tolerance < previous:
            raise assessment_error(
                "decreasing_facets_loglik_trace",
                f"$.loglik_trace[{index}]",
                "log-likelihood trace materially decreases between iterations",
            )


def _validate_iteration_trace_length(
    n_iter: int,
    trace_length: int,
    converged: bool,
    *,
    path: str,
) -> None:
    """Require exact Rust trace cardinality without dropping terminal evidence."""
    valid_lengths = {n_iter}
    if not converged:
        valid_lengths.add(n_iter + 1)
    if trace_length not in valid_lengths:
        raise assessment_error(
            "facets_iteration_trace_mismatch",
            path,
            "trace length must equal n_iter, or n_iter + 1 for a "
            "nonconverged terminal evaluation",
        )


def _category_values(values: Iterable[int]) -> tuple[int, ...]:
    """Return one sorted unique estimator category scale of exact integers."""
    output = tuple(values)
    if len(output) < 2 or any(type(value) is not int for value in output):
        raise assessment_error(
            "invalid_facets_category_values",
            "$.category_values",
            "category values must contain at least two exact integers",
        )
    if output != tuple(sorted(set(output))):
        raise assessment_error(
            "invalid_facets_category_values",
            "$.category_values",
            "category values must be sorted and unique",
        )
    return output


@dataclass(frozen=True)
class EssayFacetsCalibrationReport(CanonicalContract):
    """Source-text-free report for one exact essay criterion facets fit."""

    report_id: str
    source_design_fingerprint: str
    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    occasion_id: str
    criterion_id: str
    respondent_ids: tuple[str, ...]
    task_revision_fingerprints: tuple[str, ...]
    task_ids: tuple[str, ...]
    task_family_ids: tuple[str, ...]
    rater_engine_ids: tuple[str, ...]
    rater_engine_family_ids: tuple[str, ...]
    rater_engine_fingerprints: tuple[str, ...]
    category_values: tuple[int, ...]
    item_difficulty: tuple[float, ...]
    rater_severity: tuple[float, ...]
    thresholds: tuple[float, ...]
    respondent_theta: tuple[float, ...]
    loglik_trace: tuple[float, ...]
    n_iter: int
    converged: bool
    design_connected: bool
    fit_connected: bool
    n_parameters: int
    review_trigger_ids: tuple[str, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _report_token: InitVar[object | None] = None

    def __post_init__(self, _report_token: object | None) -> None:
        """Reject direct construction and normalize package-owned report fields."""
        if _report_token is not _REPORT_TOKEN:
            raise assessment_error(
                "unverified_essay_facets_calibration_report",
                "$",
                "use build_essay_facets_calibration_report",
            )
        object.__setattr__(
            self,
            "report_id",
            descriptive_identifier(self.report_id, "report_id"),
        )
        for field_name in (
            "source_design_fingerprint",
            "assessment_fingerprint",
            "rubric_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                fingerprint(getattr(self, field_name), field_name),
            )
        for field_name in ("construct_id", "occasion_id", "criterion_id"):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "review_trigger_ids",
            sorted_identifiers(
                self.review_trigger_ids,
                "review_trigger_ids",
                minimum=0,
                maximum=MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    @property
    def human_review_required(self) -> bool:
        """Return whether structural calibration evidence requires human review."""
        return bool(self.review_trigger_ids)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical report content without derived public identities."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "source_design_fingerprint": self.source_design_fingerprint,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "occasion_id": self.occasion_id,
            "criterion_id": self.criterion_id,
            "respondent_ids": list(self.respondent_ids),
            "task_revision_fingerprints": list(self.task_revision_fingerprints),
            "task_ids": list(self.task_ids),
            "task_family_ids": list(self.task_family_ids),
            "rater_engine_ids": list(self.rater_engine_ids),
            "rater_engine_family_ids": list(self.rater_engine_family_ids),
            "rater_engine_fingerprints": list(self.rater_engine_fingerprints),
            "category_values": list(self.category_values),
            "item_difficulty": list(self.item_difficulty),
            "rater_severity": list(self.rater_severity),
            "thresholds": list(self.thresholds),
            "respondent_theta": list(self.respondent_theta),
            "loglik_trace": list(self.loglik_trace),
            "n_iter": self.n_iter,
            "converged": self.converged,
            "design_connected": self.design_connected,
            "fit_connected": self.fit_connected,
            "n_parameters": self.n_parameters,
            "review_trigger_ids": list(self.review_trigger_ids),
            "human_review_required": self.human_review_required,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def report_fingerprint(self) -> str:
        """Return SHA-256 over the complete normalized calibration report."""
        return artifact_digest(self)

    @property
    def report_handle(self) -> str:
        """Return a descriptive 128-bit public calibration-report handle."""
        return f"essay_facets_report_{self.report_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical report content and deterministic public identities."""
        return {
            **self._content_dict(),
            "report_handle": self.report_handle,
            "report_fingerprint": self.report_fingerprint,
        }


def build_essay_facets_calibration_report(
    *,
    report_id: str,
    design: ScoringFacetsDesign,
    fit: FacetsFit,
    source_design_fingerprint: str,
    additional_review_trigger_ids: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssayFacetsCalibrationReport:
    """Build one provenance-bound report from exact Rust facets output.

    The source design fingerprint is explicit because a bare ``FacetsFit`` has
    no embedded provenance. Shape, finite-value, connectedness, iteration, and
    parameter-count checks are replay/integrity gates only. Passing them is not
    evidence of model adequacy, global optimality, validity, fairness,
    interchangeability, scoreability, or deployment readiness.
    """
    if not isinstance(design, ScoringFacetsDesign):
        raise assessment_error(
            "invalid_scoring_facets_design",
            "$.design",
            "design must be a ScoringFacetsDesign",
        )
    if not isinstance(fit, FacetsFit):
        raise assessment_error(
            "invalid_facets_fit",
            "$.fit",
            "fit must be a FacetsFit returned by the Rust-backed estimator",
        )
    normalized_source = fingerprint(
        source_design_fingerprint,
        "source_design_fingerprint",
    )
    if normalized_source != design.design_fingerprint:
        raise assessment_error(
            "essay_facets_design_fingerprint_mismatch",
            "$.source_design_fingerprint",
            "fit provenance does not match the supplied scoring facets design",
        )

    category_values = _category_values(design.category_values)
    item_difficulty = _finite_vector(
        fit.item_difficulty,
        "item_difficulty",
        expected_length=len(design.task_revision_fingerprints),
    )
    rater_severity = _finite_vector(
        fit.rater_severity,
        "rater_severity",
        expected_length=len(design.rater_engine_fingerprints),
    )
    thresholds = _finite_vector(
        fit.thresholds,
        "thresholds",
        expected_length=len(category_values) - 1,
    )
    respondent_theta = _finite_vector(
        fit.theta,
        "respondent_theta",
        expected_length=len(design.respondent_ids),
    )
    loglik_trace = _finite_vector(fit.loglik_trace, "loglik_trace")
    _validate_loglik_trace(loglik_trace)

    n_iter = _exact_integer(fit.n_iter, "n_iter", minimum=1)
    converged = strict_boolean(fit.converged, "converged")
    _validate_iteration_trace_length(
        n_iter,
        len(loglik_trace),
        converged,
        path="$.n_iter",
    )
    n_parameters = _exact_integer(fit.n_parameters, "n_parameters", minimum=1)
    expected_parameters = (
        len(design.task_revision_fingerprints)
        + len(design.rater_engine_fingerprints)
        - 1
        + len(category_values)
        - 2
    )
    if n_parameters != expected_parameters:
        raise assessment_error(
            "facets_parameter_count_mismatch",
            "$.n_parameters",
            "n_parameters does not match the Rust facets model contract",
        )
    design_connected = strict_boolean(design.connected, "design_connected")
    fit_connected = strict_boolean(fit.connected, "fit_connected")
    if fit_connected is not design_connected:
        raise assessment_error(
            "essay_facets_connectedness_mismatch",
            "$.fit_connected",
            "fit connectedness does not match the exact source design",
        )

    additional = sorted_identifiers(
        additional_review_trigger_ids,
        "additional_review_trigger_ids",
        minimum=0,
        maximum=MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS,
    )
    mandatory: set[str] = set()
    if not converged:
        mandatory.add("calibration_not_converged")
    if not design_connected:
        mandatory.add("calibration_disconnected")
    review_trigger_ids = sorted_identifiers(
        tuple(mandatory.union(additional)),
        "review_trigger_ids",
        minimum=0,
        maximum=MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS,
    )

    return EssayFacetsCalibrationReport(
        report_id=report_id,
        source_design_fingerprint=normalized_source,
        assessment_fingerprint=design.assessment_fingerprint,
        rubric_fingerprint=design.rubric_fingerprint,
        construct_id=design.construct_id,
        occasion_id=design.occasion_id,
        criterion_id=design.criterion_id,
        respondent_ids=tuple(design.respondent_ids),
        task_revision_fingerprints=tuple(design.task_revision_fingerprints),
        task_ids=tuple(design.task_ids),
        task_family_ids=tuple(design.task_family_ids),
        rater_engine_ids=tuple(design.rater_engine_ids),
        rater_engine_family_ids=tuple(design.rater_engine_family_ids),
        rater_engine_fingerprints=tuple(design.rater_engine_fingerprints),
        category_values=category_values,
        item_difficulty=item_difficulty,
        rater_severity=rater_severity,
        thresholds=thresholds,
        respondent_theta=respondent_theta,
        loglik_trace=loglik_trace,
        n_iter=n_iter,
        converged=converged,
        design_connected=design_connected,
        fit_connected=fit_connected,
        n_parameters=n_parameters,
        review_trigger_ids=review_trigger_ids,
        metadata={} if metadata is None else metadata,
        _report_token=_REPORT_TOKEN,
    )


def fit_essay_facets_calibration_report(
    *,
    report_id: str,
    design: ScoringFacetsDesign,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    additional_review_trigger_ids: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssayFacetsCalibrationReport:
    """Fit through the shared Rust boundary and immediately bind the result.

    This helper is the preferred path because it captures the exact design
    fingerprint immediately before delegation and never exposes an unbound fit
    as a governed report. It does not alter estimator tuning or interpretation.
    """
    if not isinstance(design, ScoringFacetsDesign):
        raise assessment_error(
            "invalid_scoring_facets_design",
            "$.design",
            "design must be a ScoringFacetsDesign",
        )
    source_design_fingerprint = design.design_fingerprint
    fit = fit_scoring_facets_design(
        design,
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
    )
    return build_essay_facets_calibration_report(
        report_id=report_id,
        design=design,
        fit=fit,
        source_design_fingerprint=source_design_fingerprint,
        additional_review_trigger_ids=additional_review_trigger_ids,
        metadata=metadata,
    )


__all__ = [
    "MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS",
    "EssayFacetsCalibrationReport",
    "build_essay_facets_calibration_report",
    "fit_essay_facets_calibration_report",
]
