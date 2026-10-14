"""Pointer-versus-keyboard focus regressions for diagnostics disclosures."""

from __future__ import annotations

import json
from pathlib import Path

from fast_mlsirm.report import render_diagnostics_report


def _render_report(tmp_path: Path) -> str:
    """Render one report containing exact-value and export disclosures."""
    source = tmp_path / "diagnostics.json"
    output = tmp_path / "diagnostics.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {
                    "item_id": ["item_alpha"],
                    "outfit_mnsq": [1.0],
                    "observed_count": [120],
                },
            }
        ),
        encoding="utf-8",
    )
    render_diagnostics_report(source, output, title="Disclosure Focus Review")
    return output.read_text(encoding="utf-8")


def test_disclosure_summary_suppresses_pointer_focus_without_blanket_focus_rule(
    tmp_path: Path,
) -> None:
    """Pointer focus is suppressed without erasing generic keyboard focus."""
    html = _render_report(tmp_path)
    selector = (
        ".exact-values > summary:focus:not(:focus-visible),\n"
        ".export-block > summary:focus:not(:focus-visible) {"
    )

    assert html.count(selector) == 1
    assert ".exact-values > summary:focus,\n.export-block > summary:focus {" not in html
    rule = html.split(selector, maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "outline: none;" in rule


def test_disclosure_summary_retains_keyboard_focus_indicator(tmp_path: Path) -> None:
    """The pointer rule must not erase the explicit keyboard focus treatment."""
    html = _render_report(tmp_path)
    pointer_selector = (
        ".exact-values > summary:focus:not(:focus-visible),\n"
        ".export-block > summary:focus:not(:focus-visible) {"
    )
    keyboard_selector = (
        ".exact-values > summary:focus-visible,\n"
        ".export-block > summary:focus-visible {"
    )

    assert html.count(keyboard_selector) == 1
    assert html.index(pointer_selector) < html.index(keyboard_selector)
    rule = html.split(keyboard_selector, maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "outline: 3px solid var(--teal);" in rule
    assert "outline-offset: 2px;" in rule
