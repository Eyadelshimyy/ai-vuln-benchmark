"""
Spec validation carries the design lessons as warnings.

A JSON spec is the lane where an agent does not write the page, so it may
never read the `design-` lessons. `ReportSpec.validate()` therefore checks the
spec for the patterns those lessons name and warns — never errors — with the
section path and the lesson to read. Every test asserts both halves: the
warning appears where it should, and a well-made spec stays silent.
"""

import pytest

from tracebi.knowledge import get_lesson
from tracebi.spec import ReportSpec, design_warnings

_Q = {"fact": "f", "measures": ["revenue"], "dimensions": ["dim.region"]}


def _warnings(*sections, name="Report"):
    spec = ReportSpec.from_dict({"name": name, "sections": list(sections)})
    result = spec.validate()
    assert result["ok"], result["errors"]           # design never blocks a spec
    return [w for w in result["warnings"] if ": design — " in w]


def _chart(chart_type="bar", query=None, **extra):
    return {"type": "chart", "title": "Revenue", "chart_type": chart_type,
            "x": "dim.region", "y": "revenue",
            "data": {"model": "m", "query": query or dict(_Q)}, **extra}


def test_unsorted_bars_warn_and_sorted_bars_do_not():
    [w] = _warnings(_chart("bar"))
    assert w.startswith("sections[0]: design — the bars are unsorted")
    assert "design-choose-the-chart" in w
    assert not _warnings(_chart("bar", {**_Q, "order_by": ["-revenue"]}))


def test_a_pie_needs_a_limit_of_five_or_fewer():
    assert any("a pie reads only" in w for w in _warnings(_chart("pie")))
    ok = {**_Q, "order_by": ["-revenue"], "limit": 5}
    assert not _warnings(_chart("pie", ok))


def test_too_many_lines_on_one_chart():
    y = [f"m{i}" for i in range(6)]
    [w] = _warnings(_chart("line", y=y))
    assert "6 lines on one chart" in w
    assert not _warnings(_chart("line", y=y[:3]))


def test_too_many_kpi_cards():
    cards = [{"label": f"K{i}", "value": i} for i in range(6)]
    [w] = _warnings({"type": "metrics", "metrics": cards})
    assert "6 KPI cards" in w and "design-kpis-with-context" in w
    assert not _warnings({"type": "metrics", "metrics": cards[:4]})


def test_a_wide_table_without_a_column_list():
    wide = {"fact": "f", "measures": ["a", "b", "c", "d", "e"],
            "dimensions": ["dim.x", "dim.y"]}
    table = {"type": "table", "title": "Detail",
             "data": {"model": "m", "query": wide}}
    [w] = _warnings(table)
    assert "7 columns" in w and "design-fewer-columns" in w
    # Choosing the columns is the fix, and silences it.
    assert not _warnings({**table, "columns": ["dim.x", "a", "b"]})


def test_a_palette_too_big_to_tell_apart():
    palette = ["#%06x" % (i * 0x111111) for i in range(8)]
    [w] = _warnings(_chart("bar", {**_Q, "order_by": ["-revenue"]},
                           palette=palette))
    assert w.startswith("sections[0].palette:")
    assert "design-accessible-by-default" in w


def test_emoji_in_titles_and_text():
    found = _warnings({"type": "text", "title": "📊 Overview",
                       "content": "All good 🚀"}, name="Sales 💰")
    paths = {w.split(":")[0] for w in found}
    assert paths == {"name", "sections[0].title", "sections[0].content"}


def test_rows_are_checked_at_their_own_paths():
    row = {"type": "row", "sections": [_chart("bar")]}
    [w] = _warnings(row)
    assert w.startswith("sections[0].sections[0]: design —")


def test_every_warning_names_a_lesson_that_exists():
    """A warning that points at a missing lesson is a dead end."""
    import re

    cards = [{"label": f"K{i}", "value": i} for i in range(6)]
    everything = design_warnings("Sales 💰", [
        _chart("bar"), _chart("pie"), _chart("line", y=list("abcdef")),
        {"type": "metrics", "metrics": cards},
        _chart("bar", palette=["#000"] * 8),
    ])
    assert len(everything) >= 6
    for w in everything:
        slug = re.search(r"tracebi knowledge (design-[a-z-]+)\)", w).group(1)
        assert get_lesson(slug) is not None, f"{slug} named by: {w}"


@pytest.mark.parametrize("path", [
    "examples/portfolio_project/reports/portfolio_model/portfolio_dashboard.json",
])
def test_the_reference_spec_follows_its_own_lessons(path):
    """The spec people copy from must not trip the checks it teaches."""
    import json
    from pathlib import Path

    raw = json.loads((Path(__file__).parents[1] / path).read_text())
    spec = ReportSpec.from_dict(raw)
    found = [w for w in spec.validate()["warnings"] if ": design — " in w]
    assert not found, found
