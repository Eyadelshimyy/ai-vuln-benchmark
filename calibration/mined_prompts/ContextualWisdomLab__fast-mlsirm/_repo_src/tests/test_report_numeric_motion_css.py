"""Rendered-report CSS regressions for numeric comparison and motion cleanup."""

from __future__ import annotations

import json
import re

from fast_mlsirm.report import render_diagnostics_report


def _rule_body(css: str, selector: str) -> str:
    """Return the declaration body for one exact CSS selector."""
    match = re.search(rf"(?:^|\n){re.escape(selector)}\s*\{{(?P<body>.*?)\n\}}", css, re.DOTALL)
    assert match is not None, f"missing CSS rule for {selector!r}"
    return match.group("body")


def test_rendered_report_uses_tabular_numerals_without_opacity_transitions(tmp_path) -> None:
    """Numeric alignment and row cues must survive without peer-dimming transitions."""
    source = tmp_path / "fit_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2, "deviance": 6.4},
                "itemfit": {
                    "item_id": ["A", "B"],
                    "outfit_mnsq": [1.0, 1.2],
                    "observed_count": [4, 4],
                },
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, output)

    html = output.read_text(encoding="utf-8")
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]

    body_rule = _rule_body(style, "body")
    assert "font-variant-numeric: tabular-nums;" in body_rule

    bar_row_rule = _rule_body(style, ".bar-row")
    assert "transition:" not in bar_row_rule
    assert "opacity:" not in bar_row_rule
    assert "padding: 4px 8px;" in bar_row_rule
    assert "border-radius: 4px;" in bar_row_rule

    bar_row_hover_rule = _rule_body(style, ".bar-row:hover")
    assert "background: var(--hover-bg);" in bar_row_hover_rule

    table_row_rule = _rule_body(style, "tbody tr")
    assert "transition: background-color 0.15s ease-in-out;" in table_row_rule
    assert "opacity" not in table_row_rule

    hover_rule = _rule_body(style, "tbody tr:hover")
    assert "background: var(--hover-bg);" in hover_rule

    assert "@media (prefers-reduced-motion: reduce)" in style
    reduced_motion = style.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    reduced_motion = reduced_motion.split("@media (max-width: 720px)", 1)[0]
    assert "transition-duration: 0.01ms !important;" in reduced_motion


def test_rendered_report_preserves_keyboard_focus_on_main(tmp_path) -> None:
    """Pointer focus may be quiet, but keyboard focus on main must stay visible."""
    source = tmp_path / "fit_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2, "deviance": 6.4},
                "itemfit": {
                    "item_id": ["A", "B"],
                    "outfit_mnsq": [1.0, 1.2],
                    "observed_count": [4, 4],
                },
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, output)

    html = output.read_text(encoding="utf-8")
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]

    pointer_rule = _rule_body(style, "main:focus:not(:focus-visible)")
    assert "outline: none;" in pointer_rule
    keyboard_rule = _rule_body(style, "main:focus-visible")
    assert "outline: 3px solid var(--teal);" in keyboard_rule
    assert "outline-offset: 3px;" in keyboard_rule
    assert "outline: none;" not in keyboard_rule
    assert "\nmain:focus {" not in style
