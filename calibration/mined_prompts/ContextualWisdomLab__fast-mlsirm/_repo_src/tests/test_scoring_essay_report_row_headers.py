"""Regression tests for semantic row headers in essay HTML report tables."""

from __future__ import annotations

from html import escape
from pathlib import Path
import runpy

from fast_mlsirm.scoring.essay import (
    render_essay_facets_calibration_report_html,
    render_essay_validation_evidence_report_html,
)

_FACETS_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_facets_reporting.py"))
)
_VALIDATION_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_validation_reporting.py"))
)


def _row_header(value: object) -> str:
    """Return the exact semantic row-header cell expected for text-like values."""
    return f'<th scope="row">{escape(str(value))}</th>'


def test_facets_report_marks_primary_table_axes_as_row_headers(tmp_path: Path) -> None:
    """Facets table identities and ordered axes remain associated with each row."""
    report = _FACETS_FIXTURES["build_report"]()
    output = tmp_path / "facets-row-headers.html"

    render_essay_facets_calibration_report_html(
        report,
        output,
        output_root=tmp_path,
    )
    html = output.read_text(encoding="utf-8")

    identity_row_headers = (
        report.task_ids[0],
        report.rater_engine_ids[0],
        report.respondent_ids[0],
    )
    for value in identity_row_headers:
        assert _row_header(value) in html
        assert f"<td>{escape(str(value))}</td>" not in html

    assert _row_header(report.category_values[0]) in html
    assert _row_header(1) in html


def test_validation_report_marks_metric_identity_as_row_header(tmp_path: Path) -> None:
    """Validation metric values stay associated with their metric identity."""
    report = _VALIDATION_FIXTURES["build_report"]()
    output = tmp_path / "validation-row-headers.html"

    render_essay_validation_evidence_report_html(report, output)
    html = output.read_text(encoding="utf-8")

    metric_id = report.metrics[0].metric_id
    assert _row_header(metric_id) in html
    assert f"<td>{escape(metric_id)}</td>" not in html


def test_print_styles_keep_scrollable_exact_value_evidence_visible(
    tmp_path: Path,
) -> None:
    """Printed report evidence must not remain clipped inside screen scroll boxes."""
    report = _FACETS_FIXTURES["build_report"]()
    output = tmp_path / "facets-print-evidence.html"

    render_essay_facets_calibration_report_html(
        report,
        output,
        output_root=tmp_path,
    )
    html = output.read_text(encoding="utf-8")

    assert ".table-scroll { overflow-x: auto; }" in html
    assert "@media print {" in html
    assert ".table-scroll, pre { overflow: visible; }" in html
    assert "pre { max-height: none; }" in html
