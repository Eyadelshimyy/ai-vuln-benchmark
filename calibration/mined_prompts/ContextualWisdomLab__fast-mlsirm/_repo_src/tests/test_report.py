import json

import pytest

from fast_mlsirm.io import MAX_JSON_NESTING_DEPTH
from fast_mlsirm.report import render_diagnostics_report


def test_render_fit_diagnostics_report_has_sections(tmp_path):
    """Render populated fit diagnostics with the expected report sections."""
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "itemfit": {"item_id": [0, 1], "outfit_mnsq": [1.0, 1.2], "observed_count": [4, 4]},
                "personfit": {"person_id": [0, 1], "outfit_mnsq": [0.9, 1.1], "observed_count": [2, 2]},
                "factorfit": {},
                "categoryfit": {},
                "groupfit": {},
                "clusterfit": {},
                "group_itemfit": {},
                "cluster_itemfit": {},
                "model_fit": {"loglik": -3.2, "deviance": 6.4},
            }
        ),
        encoding="utf-8",
    )

    rendered = render_diagnostics_report(source, out, title="Example Fit")

    assert rendered == out
    html = out.read_text(encoding="utf-8")
    assert "Example Fit" in html
    assert "Model Fit" in html
    assert '<dl class="metrics-grid">' in html
    assert "<dt>Loglik</dt>" in html
    assert '<dd title="-3.2">-3.2</dd>' in html
    assert "Item Fit" in html
    assert "Diagnostics Coverage" in html
    assert "No row data" in html
    assert "No rows were recorded in this section." not in html
    assert "No chartable values were recorded for this section." not in html
    assert "<table>" in html
    assert 'http-equiv="Content-Security-Policy"' in html
    assert "default-src &#x27;none&#x27;" in html
    assert 'role="region" aria-label="Item Fit diagnostics table" tabindex="0"' in html
    assert "<caption>Item Fit diagnostics table</caption>" in html


def test_render_dimensionality_report_has_best_candidate(tmp_path):
    """Render dimensionality diagnostics with the selected candidate."""
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [
                    {"latent_dim": 1.0, "heldout_loglik": -12.5},
                    {"latent_dim": 2.0, "heldout_loglik": -8.0},
                ],
                "best": {"latent_dim": 2.0, "heldout_loglik": -8.0},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Dimensionality Diagnostics Report" in html
    assert "Best Candidate" in html
    assert "Candidate Comparison" in html
    assert "Latent Dim 2" in html
    assert "Latent Dim 2.0" not in html


def test_render_dimensionality_report_summarizes_empty_candidates(tmp_path):
    """Summarize dimensionality diagnostics when no candidates are available."""
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [],
                "best": {"latent_dim": 2.0, "heldout_loglik": -8.0},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Best Candidate" in html
    assert "Diagnostics Coverage" in html
    assert "Candidate Comparison" in html
    assert "<h2>Candidate Comparison</h2>" not in html
    assert "No rows were recorded in this section." not in html


def test_render_report_summarizes_empty_metric_sections(tmp_path):
    """Summarize an empty metric section without rendering an empty panel."""
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {},
                "itemfit": {"item_id": [0], "outfit_mnsq": [1.0], "observed_count": [4]},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Diagnostics Coverage" in html
    assert "No metric data" in html
    assert "Diagnostics without table rows or metric values are summarized here" in html
    no_metric_column = html[html.index("<h3>No metric data</h3>") :]
    no_metric_column = no_metric_column[: no_metric_column.index("</div>")]
    assert "Model Fit" in no_metric_column
    assert '<h2 id="model-fit">Model Fit</h2>' not in html
    assert "No metrics were recorded in this diagnostics file." not in html


def test_render_dimensionality_report_summarizes_empty_best_candidate(tmp_path):
    """Summarize a dimensionality report whose best candidate has no metrics."""
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [{"latent_dim": 2.0, "heldout_loglik": -8.0}],
                "best": {},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Diagnostics Coverage" in html
    assert "No metric data" in html
    no_metric_column = html[html.index("<h3>No metric data</h3>") :]
    no_metric_column = no_metric_column[: no_metric_column.index("</div>")]
    assert "Best Candidate" in no_metric_column
    assert "Candidate Comparison" in html
    assert '<h2 id="best-candidate">Best Candidate</h2>' not in html
    assert '<h2 id="candidate-comparison">Candidate Comparison</h2>' in html
    assert "No metrics were recorded in this diagnostics file." not in html


def test_render_coverage_omits_empty_rendered_tables_column(tmp_path):
    """Omit the rendered-table coverage column when no table is present."""
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [],
                "best": {},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Diagnostics Coverage" in html
    assert "<h3>Rendered tables</h3>" not in html
    assert "<li>None</li>" not in html
    assert "<h3>No row data</h3>" in html
    assert "<h3>No metric data</h3>" in html


def test_render_table_section_omits_empty_chart_placeholder(tmp_path):
    """Avoid an empty chart placeholder when rows have no numeric values."""
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {"item_id": ["A"], "outfit_mnsq": [None], "observed_count": [4]},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert '<h2 id="item-fit">Item Fit</h2>' in html
    assert "<table>" in html
    assert "No chartable values were recorded for this section." not in html


def test_render_table_section_charts_later_numeric_rows(tmp_path):
    """Include a later numeric row when earlier rows are not chartable."""
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    item_ids = list(range(13))
    outfit = [None] * 12 + [1.2]
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {"item_id": item_ids, "outfit_mnsq": outfit, "observed_count": [4] * 13},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert '<div class="bar-chart" aria-hidden="true">' in html
    assert '<span class="bar-label">Item Id 12</span>' in html
    assert "Showing 12 of 13 rows." in html


def test_render_report_rejects_unknown_payload(tmp_path):
    """Reject diagnostics JSON that has no supported report payload."""
    source = tmp_path / "unknown.json"
    out = tmp_path / "report.html"
    source.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported diagnostics JSON"):
        render_diagnostics_report(source, out)


def test_render_report_rejects_excessive_json_nesting(tmp_path):
    """Reject JSON that exceeds the configured nesting safety limit."""
    source = tmp_path / "deep.json"
    out = tmp_path / "report.html"
    depth = MAX_JSON_NESTING_DEPTH + 1
    source.write_text("[" * depth + "0" + "]" * depth, encoding="utf-8")

    with pytest.raises(ValueError, match="maximum JSON nesting depth"):
        render_diagnostics_report(source, out)

    assert not out.exists()


def test_render_report_requires_html_output(tmp_path):
    """Reject output paths that do not use the HTML extension."""
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.txt"
    source.write_text(json.dumps({"model_fit": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="must end with .html"):
        render_diagnostics_report(source, out)


def test_render_table_region_has_keyboard_focus_style(tmp_path):
    """Keep keyboard focus styling while suppressing mouse-click outlines."""
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [{"latent_dim": 2.0, "heldout_loglik": -8.0}],
                "best": {"latent_dim": 2.0},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert 'aria-label="Candidate Comparison diagnostics table"' in html
    assert 'tabindex="0"' in html
    assert ".table-wrap:focus-visible" in html
    assert ".table-wrap:focus:not(:focus-visible) {" in html
    assert "tbody tr:hover" in html
    assert '<div class="bar-chart" aria-hidden="true">' in html
    assert '<div class="bar-track" aria-hidden="true">' in html
