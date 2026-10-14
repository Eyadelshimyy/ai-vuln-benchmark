"""Fit enterprise issue bundles through the shared facets report boundary.

The orchestration validates enterprise executions through the existing governed
bundle assembler and delegates each criterion-specific design to the canonical
Rust-backed scoring-facets report helper. It returns only the shared report type
and performs no likelihood, gradient, Hessian, optimization, scoring, ranking,
validity, fairness, utility, causal, or queue arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._contract_safety import (
    descriptive_identifier,
    freeze_metadata,
    sorted_identifiers,
)
from .._validation import assessment_error, thaw_json_value
from ..calibration_reporting import (
    MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS,
    ScoringFacetsCalibrationReport,
    fit_scoring_facets_calibration_report,
)
from ..execution import (
    MAX_REQUEST_CRITERIA,
    EngineDescriptor,
    ScoringRequest,
    ScoringResult,
)
from .calibration import build_enterprise_issue_facets_calibration_bundle
from .contracts import AtomicIssueRecord

MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS = MAX_REQUEST_CRITERIA
"""Maximum criterion reports emitted by one enterprise calibration workflow."""

_MANAGED_REPORT_METADATA_KEYS = frozenset(
    {
        "enterprise_calibration_bundle_fingerprint",
        "enterprise_calibration_design_fingerprint",
        "enterprise_calibration_criterion_id",
    }
)


def _caller_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return source-free caller metadata without package-managed report keys."""
    primitive = thaw_json_value(freeze_metadata({} if metadata is None else metadata))
    if any(key in primitive for key in _MANAGED_REPORT_METADATA_KEYS):
        raise assessment_error(
            "reserved_enterprise_report_metadata",
            "$.metadata",
            "enterprise calibration report provenance is package-managed",
        )
    return primitive


def fit_enterprise_issue_facets_calibration_reports(
    executions: Iterable[
        tuple[AtomicIssueRecord, ScoringRequest, ScoringResult, EngineDescriptor]
    ],
    *,
    report_id_prefix: str,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    require_connected: bool = True,
    additional_review_trigger_ids: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ScoringFacetsCalibrationReport, ...]:
    """Fit one shared provenance-bound report per enterprise criterion.

    The execution iterable is validated once by
    :func:`build_enterprise_issue_facets_calibration_bundle`. Every resulting
    criterion design is then delegated exactly once to the canonical shared
    scoring-facets report helper, which calls the existing Rust-backed many-facet
    estimator and binds the exact design fingerprint immediately. The returned
    values are the canonical shared reports; no enterprise-specific fit or report
    schema is introduced.

    All derived report identifiers are validated as one batch before any
    estimator delegation. Package-managed metadata binds each report to the exact
    enterprise bundle, criterion design, and criterion identifier. Caller
    metadata must remain source-text-free under the shared metadata contract and
    cannot replace those fields. Review triggers are normalized once and
    forwarded unchanged to every criterion report.

    A successful workflow proves provenance and contract consistency only. It
    does not establish model adequacy, convergence quality, construct validity,
    reliability, fairness, rater interchangeability, predictive validity,
    intervention value, causal effect, or high-stakes deployment readiness.
    """
    normalized_prefix = descriptive_identifier(
        report_id_prefix,
        "report_id_prefix",
        "$.report_id_prefix",
    )
    normalized_metadata = _caller_metadata(metadata)
    review_trigger_ids = sorted_identifiers(
        additional_review_trigger_ids,
        "additional_review_trigger_ids",
        minimum=0,
        maximum=MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS,
    )
    bundle = build_enterprise_issue_facets_calibration_bundle(
        executions,
        require_connected=require_connected,
    )
    report_plans = tuple(
        (
            design,
            descriptive_identifier(
                f"{normalized_prefix}_{design.criterion_id}",
                "report_id",
                f"$.report_ids[{index}]",
            ),
        )
        for index, design in enumerate(bundle.designs)
    )

    reports: list[ScoringFacetsCalibrationReport] = []
    for design, report_id in report_plans:
        report_metadata = dict(normalized_metadata)
        report_metadata.update(
            {
                "enterprise_calibration_bundle_fingerprint": (
                    bundle.bundle_fingerprint
                ),
                "enterprise_calibration_design_fingerprint": (
                    design.design_fingerprint
                ),
                "enterprise_calibration_criterion_id": design.criterion_id,
            }
        )
        reports.append(
            fit_scoring_facets_calibration_report(
                report_id=report_id,
                design=design,
                q_theta=q_theta,
                max_iter=max_iter,
                tol=tol,
                additional_review_trigger_ids=review_trigger_ids,
                metadata=report_metadata,
            )
        )
    return tuple(reports)


__all__ = [
    "MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS",
    "fit_enterprise_issue_facets_calibration_reports",
]
