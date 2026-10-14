"""Security regression tests for standalone diagnostics report rendering."""

from html import escape

from fast_mlsirm.report import _metric_section, _table_section


def test_report_section_heading_ids_reject_html_attribute_injection():
    """Keep report section IDs inert when headings contain active HTML syntax."""
    heading = '"><script>alert("xss")</script>'
    rendered_sections = [
        _metric_section(heading, {"loglik": -1.0}),
        _table_section(heading, [{"value": 1.0}]),
    ]

    for section in rendered_sections:
        assert section is not None
        assert "<script>" not in section
        assert 'aria-labelledby=""><' not in section
        assert 'id=""><' not in section
        assert escape(heading) in section
