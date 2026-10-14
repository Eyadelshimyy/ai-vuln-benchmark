"""Accessibility regression tests for generated diagnostics reports."""

import json

from fast_mlsirm.report import render_diagnostics_report


def test_rendered_report_has_semantic_hero_metadata(tmp_path):
    """Keep decorative branding hidden and source metadata machine-readable."""
    source = tmp_path / "fit_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {
                    "item_id": [0],
                    "outfit_mnsq": [1.0],
                    "observed_count": [1],
                },
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, output, title="Example Fit")

    html = output.read_text(encoding="utf-8")
    assert '<p aria-hidden="true">fast-mlsirm diagnostics</p>' in html
    assert '<dl class="hero-meta">' in html
    assert "<dt>Source</dt>" in html
    assert "<dd>fit_diagnostics.json</dd>" in html
