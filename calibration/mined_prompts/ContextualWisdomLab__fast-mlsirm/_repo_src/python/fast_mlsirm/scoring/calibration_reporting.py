"""Domain-neutral names for the canonical scoring-facets calibration report.

The governed report implementation was first delivered through the essay adapter.
This module exposes shared scoring names without copying the report schema,
validation logic, or estimator delegation.  The aliases are intentionally the
same Python objects as the established essay API so existing serialized handles,
error codes, imports, and ABI expectations remain stable.

New domain adapters should import from this module.  The essay names remain
supported compatibility entry points; changing legacy wire identities requires a
separately versioned migration rather than an in-place rename.
"""

from __future__ import annotations

from .essay.calibration_reporting import (
    MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS,
    EssayFacetsCalibrationReport,
    build_essay_facets_calibration_report,
    fit_essay_facets_calibration_report,
)

MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS = (
    MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS
)
"""Maximum review-trigger count accepted by the canonical facets report."""

ScoringFacetsCalibrationReport = EssayFacetsCalibrationReport
"""Canonical facets report type under its domain-neutral compatibility name."""

build_scoring_facets_calibration_report = build_essay_facets_calibration_report
"""Build a provenance-bound report from one exact Rust facets fit."""

fit_scoring_facets_calibration_report = fit_essay_facets_calibration_report
"""Fit through the Rust-backed shared boundary and bind the resulting report."""

__all__ = [
    "MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS",
    "ScoringFacetsCalibrationReport",
    "build_scoring_facets_calibration_report",
    "fit_scoring_facets_calibration_report",
]
