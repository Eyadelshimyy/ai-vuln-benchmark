"""Public-package export tests for governed essay reporting surfaces."""

from __future__ import annotations

import fast_mlsirm.scoring.essay as essay
import fast_mlsirm.scoring.essay.calibration_report_html as calibration_report_html
import fast_mlsirm.scoring.essay.report_html as report_html
import fast_mlsirm.scoring.essay.reporting as reporting
import fast_mlsirm.scoring.essay.validation_report_html as validation_report_html


def test_score_reporting_symbols_are_exported_from_essay_package() -> None:
    """The issue-level essay namespace exposes the complete report API."""
    expected = {
        "EssayScoreReport",
        "MAX_ESSAY_REPORT_REVIEW_TRIGGERS",
        "build_essay_score_report",
        "render_essay_facets_calibration_report_html",
        "render_essay_score_report_html",
        "render_essay_validation_evidence_report_html",
    }

    assert expected <= set(essay.__all__)
    assert essay.EssayScoreReport is reporting.EssayScoreReport
    assert (
        essay.MAX_ESSAY_REPORT_REVIEW_TRIGGERS
        == reporting.MAX_ESSAY_REPORT_REVIEW_TRIGGERS
    )
    assert essay.build_essay_score_report is reporting.build_essay_score_report
    assert (
        essay.render_essay_facets_calibration_report_html
        is calibration_report_html.render_essay_facets_calibration_report_html
    )
    assert (
        essay.render_essay_score_report_html
        is report_html.render_essay_score_report_html
    )
    assert (
        essay.render_essay_validation_evidence_report_html
        is validation_report_html.render_essay_validation_evidence_report_html
    )
    assert essay.EssayScoreReport.__doc__
    assert essay.build_essay_score_report.__doc__
    assert essay.render_essay_facets_calibration_report_html.__doc__
    assert essay.render_essay_score_report_html.__doc__
    assert essay.render_essay_validation_evidence_report_html.__doc__
