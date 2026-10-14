"""Accessibility contracts for diagnostics-report focus and contrast styling."""

from __future__ import annotations

import json
from pathlib import Path

from fast_mlsirm.report import render_diagnostics_report


def _render_report(tmp_path: Path) -> str:
    """Render one realistic diagnostics report and return its complete HTML."""
    source = tmp_path / "fit_diagnostics.json"
    output = tmp_path / "diagnostics_report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {
                    "item_id": ["item_alpha", "item_beta"],
                    "outfit_mnsq": [1.0, 1.2],
                    "observed_count": [120, 117],
                },
            }
        ),
        encoding="utf-8",
    )
    render_diagnostics_report(source, output, title="Accessible Fit Review")
    return output.read_text(encoding="utf-8")


def test_skip_link_is_revealed_for_every_actual_focus_state(tmp_path: Path) -> None:
    """A focused skip link must not depend only on user-agent focus heuristics."""
    html = _render_report(tmp_path)
    selector = ".skip-link:focus,\n.skip-link:focus-visible {"

    assert html.count(selector) == 1
    focus_rule = html.split(selector, maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "top: 0;" in focus_rule
    assert "outline: 3px solid var(--teal);" in focus_rule


def test_hover_does_not_dim_unrelated_chart_or_table_content(tmp_path: Path) -> None:
    """Pointer hover must preserve peer contrast and retain the active-row cue."""
    html = _render_report(tmp_path)

    assert ".bar-chart:hover .bar-row:not(:hover)" not in html
    assert "tbody:hover tr:not(:hover)" not in html
    assert "tbody tr:hover {\n  background: var(--hover-bg);\n}" in html


def test_focus_containers_suppress_mouse_click_outlines(tmp_path: Path) -> None:
    """Pointer-focus suppression must preserve keyboard-visible focus fallback."""
    html = _render_report(tmp_path)
    assert ".table-wrap:focus:not(:focus-visible) {\n  outline: none;\n}" in html
    assert ".export-block pre:focus:not(:focus-visible) {\n  outline: none;\n}" in html
    assert ".table-wrap:focus {\n  outline: none;\n}" not in html
    assert ".export-block pre:focus {\n  outline: none;\n}" not in html
