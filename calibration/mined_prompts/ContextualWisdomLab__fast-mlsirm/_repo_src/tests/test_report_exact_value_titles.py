"""Tests for supplemental pointer tooltips and accessible exact-value disclosure."""

from __future__ import annotations

import json

from fast_mlsirm.report import _title_attr, render_diagnostics_report


def test_formatted_float_locations_expose_unrounded_pointer_tooltips(tmp_path) -> None:
    """Metric, chart, and table values retain their Python float representation."""
    source = tmp_path / "dimension_diagnostics.json"
    output = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [
                    {"latent_dim": 1.0, "heldout_loglik": -12.500000000000002},
                    {"latent_dim": 2.0, "heldout_loglik": -8.0},
                ],
                "best": {"latent_dim": 2.0, "heldout_loglik": -8.0},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, output)

    html = output.read_text(encoding="utf-8")
    assert '<dd title="2.0">2</dd>' in html
    assert (
        '<span class="bar-value" title="-12.500000000000002">-12.5</span>'
        in html
    )
    assert '<th scope="row" title="1.0">1</th>' in html
    assert '<td title="-12.500000000000002">-12.5</td>' in html
    assert "heldout_loglik\n1.0,-12.500000000000002\n2.0,-8.0" in html


def test_title_attribute_is_supplemental_and_finite_float_only() -> None:
    """Non-floats and non-finite floats do not gain misleading title attributes."""
    assert _title_attr(1) == ""
    assert _title_attr("1.0") == ""
    assert _title_attr(float("inf")) == ""
    assert _title_attr(float("nan")) == ""
    assert _title_attr(1.2345678901234567) == (
        ' title="1.2345678901234567"'
    )
