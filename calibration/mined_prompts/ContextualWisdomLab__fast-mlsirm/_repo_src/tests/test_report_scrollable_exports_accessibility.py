"""Accessibility contracts for horizontally scrollable report exports."""

from __future__ import annotations

from html.parser import HTMLParser

from fast_mlsirm.report import _css
from fast_mlsirm.report_exact_values import exact_value_disclosure


class _PreRegionParser(HTMLParser):
    """Collect attributes from rendered ``pre`` regions without browser tooling."""

    def __init__(self) -> None:
        """Initialize an empty list of discovered preformatted regions."""
        super().__init__()
        self.pre_attributes: list[dict[str, str | None]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Retain attributes for every preformatted export region."""
        if tag == "pre":
            self.pre_attributes.append(dict(attrs))


def test_export_pre_regions_are_named_and_keyboard_focusable() -> None:
    """JSON and CSV regions expose names and a keyboard scrolling entry point."""
    markup = exact_value_disclosure(
        [{"metric_name": "difficulty", "metric_value": 1.25}],
        section_label="Item Fit",
    )
    parser = _PreRegionParser()
    parser.feed(markup)

    assert parser.pre_attributes == [
        {
            "role": "region",
            "aria-label": "JSON export for Item Fit",
            "tabindex": "0",
        },
        {
            "role": "region",
            "aria-label": "CSV export for Item Fit",
            "tabindex": "0",
        },
    ]


def test_export_region_accessible_name_escapes_hostile_section_text() -> None:
    """Section labels cannot escape the accessible-name attribute context."""
    markup = exact_value_disclosure(
        [{"metric_name": "difficulty"}],
        section_label='"><script>alert(1)</script>',
    )
    parser = _PreRegionParser()
    parser.feed(markup)

    assert len(parser.pre_attributes) == 2
    assert all("<script>" in (attrs["aria-label"] or "") for attrs in parser.pre_attributes)
    assert '<script>alert(1)</script>' not in markup
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in markup


def test_scrollable_export_focus_indicator_respects_motion_preferences() -> None:
    """Focusable export regions have a visible outline and reduced-motion policy."""
    stylesheet = _css()

    assert ".export-block pre:focus-visible" in stylesheet
    assert "outline: 3px solid var(--teal);" in stylesheet
    assert "outline-offset: -2px;" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
    assert "transition-duration: 0.01ms !important;" in stylesheet
