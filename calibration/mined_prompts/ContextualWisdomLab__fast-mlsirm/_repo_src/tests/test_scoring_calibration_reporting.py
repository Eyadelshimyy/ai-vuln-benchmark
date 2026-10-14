"""Compatibility contracts for shared scoring-facets report names."""

from __future__ import annotations

import inspect

import fast_mlsirm.scoring.calibration_reporting as shared_reporting
import fast_mlsirm.scoring.essay.calibration_reporting as essay_reporting


def test_shared_surface_reuses_the_canonical_report_objects() -> None:
    """Shared names must not create a second schema or validation path."""
    assert shared_reporting.ScoringFacetsCalibrationReport is (
        essay_reporting.EssayFacetsCalibrationReport
    )
    assert shared_reporting.build_scoring_facets_calibration_report is (
        essay_reporting.build_essay_facets_calibration_report
    )
    assert shared_reporting.fit_scoring_facets_calibration_report is (
        essay_reporting.fit_essay_facets_calibration_report
    )
    assert shared_reporting.MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS == (
        essay_reporting.MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS
    )


def test_shared_surface_is_documented_and_explicitly_bounded() -> None:
    """Every shared alias remains discoverable with inherited public docs."""
    expected = {
        "MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS",
        "ScoringFacetsCalibrationReport",
        "build_scoring_facets_calibration_report",
        "fit_scoring_facets_calibration_report",
    }
    assert set(shared_reporting.__all__) == expected
    assert inspect.getdoc(shared_reporting)
    assert inspect.getdoc(shared_reporting.ScoringFacetsCalibrationReport)
    assert inspect.getdoc(shared_reporting.build_scoring_facets_calibration_report)
    assert inspect.getdoc(shared_reporting.fit_scoring_facets_calibration_report)
    assert shared_reporting.ScoringFacetsCalibrationReport.__module__ == (
        "fast_mlsirm.scoring.essay.calibration_reporting"
    )
