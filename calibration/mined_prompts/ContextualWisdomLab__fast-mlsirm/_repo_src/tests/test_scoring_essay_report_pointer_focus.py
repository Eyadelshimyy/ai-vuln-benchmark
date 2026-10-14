"""Fail-first coverage for pointer focus styling in essay HTML reports."""

from __future__ import annotations

from fast_mlsirm.scoring.essay import report_html


def test_main_pointer_focus_suppresses_default_outline_without_hiding_keyboard_focus() -> None:
    """Pointer suppression keeps a fallback focus indicator for older user agents."""
    css = report_html._css()
    assert "main:focus:not(:focus-visible) { outline: none; }" in css
    assert "main:focus { outline: none; }" not in css
    assert "main:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }" in css


def test_scrollable_pointer_focus_is_modality_safe() -> None:
    """Scrollable report regions suppress pointer focus without blanket outline removal."""
    css = report_html._css()
    assert ".table-scroll:focus:not(:focus-visible), pre:focus:not(:focus-visible) { outline: none; }" in css
    assert ".table-scroll:focus, pre:focus { outline: none; }" not in css
    assert ".table-scroll:focus-visible, pre:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }" in css
