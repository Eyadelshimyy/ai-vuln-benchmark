"""The D3 chart engine, in a real browser.

Every declared chart type (bar, barh, line, area, pie, scatter) is drawn by
the inlined D3 from the stamped rows. These pin what a reader relies on: one
mark per row and no ECharts on the page, a tooltip that shows the row's own
value in the page's format, a click that filters through the page's own
control, colours that follow the series rather than its position, a legend
that cannot hide the last series, and arrow keys that read every value.

Skipped unless Playwright can launch Chromium. The CI job ``ui-smoke``
installs Chromium and runs this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.reports.template_package import TemplatePackage

_SALES = pd.DataFrame({
    "sale_id": list(range(16)),
    "region": ["East", "West", "North", "South"] * 4,
    "quarter": [q for q in ("Q1", "Q2", "Q3", "Q4") for _ in range(4)],
    "product": ["A", "B", "A", "B"] * 4,
    "week": [f"W{i + 1:02d}" for i in range(16)],
    "revenue": [1200, 900, 400, 300, 1500, 800, 450, 350,
                1700, 950, 500, 320, 1600, 1000, 520, 380],
    "cost": [700, 600, 300, 200, 800, 500, 300, 250,
             900, 600, 350, 220, 850, 640, 360, 260],
})


def _model() -> DataModel:
    model = DataModel("sales_model")
    model.add_connector(MemoryConnector("mem", tables={"sales": _SALES}))
    model.add_table("sales", connector="mem", source="sales")
    model.add_dimension("dim_sale", table_name="sales", key_col="sale_id",
                        attributes=["region", "quarter", "product", "week"])
    model.add_fact("fact_sales", table_name="sales",
                   measures=["revenue", "cost"], foreign_keys={})
    model.add_measure("revenue", column="revenue", agg="sum")
    model.add_measure("cost", column="cost", agg="sum")
    model.connect()
    return model


def _query(dims, measures=("revenue",), order=None):
    q = {"fact": "fact_sales", "measures": list(measures), "dimensions": list(dims)}
    if order:
        q["order_by"] = order
    return {"model": "sales_model", "query": q}


_FIGURES = """
<select id="f-region" data-tb-filter data-tb-binding="by_region"
        data-tb-column="dim_sale.region"></select>
<select id="f-product" data-tb-filter data-tb-binding="by_quarter"
        data-tb-column="dim_sale.product"></select>
<div id="bar" data-tb-figure="chart" data-tb-binding="by_region" data-tb-type="bar"
     data-tb-x="dim_sale.region" data-tb-y="revenue" data-tb-value-format="currency0"></div>
<div id="barh" data-tb-figure="chart" data-tb-binding="by_region" data-tb-type="barh"
     data-tb-x="dim_sale.region" data-tb-y="revenue,cost"></div>
<div id="line" data-tb-figure="chart" data-tb-binding="by_quarter" data-tb-type="line"
     data-tb-x="dim_sale.quarter" data-tb-y="revenue" data-tb-color="dim_sale.product"></div>
<div id="area" data-tb-figure="chart" data-tb-binding="totals_by_quarter" data-tb-type="area"
     data-tb-x="dim_sale.quarter" data-tb-y="revenue" data-tb-unit=" u"></div>
<div id="pie" data-tb-figure="chart" data-tb-binding="by_product" data-tb-type="pie"
     data-tb-x="dim_sale.product" data-tb-y="revenue"></div>
<div id="scatter" data-tb-figure="chart" data-tb-binding="by_region" data-tb-type="scatter"
     data-tb-x="cost" data-tb-y="revenue"></div>
<div id="stack" data-tb-figure="chart" data-tb-binding="by_quarter" data-tb-type="bar"
     data-tb-x="dim_sale.quarter" data-tb-y="revenue" data-tb-color="dim_sale.product"
     data-tb-stack></div>
<div id="weekly" data-tb-figure="chart" data-tb-binding="by_week" data-tb-type="area"
     data-tb-x="dim_sale.week" data-tb-y="revenue" data-tb-mark="max,last"
     data-tb-annotate="W05=Launch" data-tb-x-title="Week"></div>
<div id="facet" data-tb-figure="chart" data-tb-binding="by_quarter" data-tb-type="bar"
     data-tb-x="dim_sale.quarter" data-tb-y="revenue" data-tb-facet="dim_sale.product"></div>
"""


def _build(tmp_path: Path) -> Path:
    pkg = tmp_path / "charts"
    pkg.mkdir()
    (pkg / "report.json").write_text(json.dumps({
        "name": "Charts",
        "data": {
            "by_region": _query(["dim_sale.region"], ("revenue", "cost"), ["-revenue"]),
            "by_quarter": _query(["dim_sale.quarter", "dim_sale.product"],
                                 order=["dim_sale.quarter"]),
            "totals_by_quarter": _query(["dim_sale.quarter"], order=["dim_sale.quarter"]),
            "by_product": _query(["dim_sale.product"]),
            "by_week": _query(["dim_sale.week"], order=["dim_sale.week"]),
        },
    }), encoding="utf-8")
    (pkg / "template.html").write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Charts</title>"
        "</head><body><main class=\"tb-page\" style=\"width:900px\">"
        + _FIGURES + "</main></body></html>", encoding="utf-8")
    out = tmp_path / "charts.html"
    TemplatePackage(str(pkg)).render({"sales_model": _model()}, str(out))
    return out


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    html = _build(tmp_path_factory.mktemp("d3"))
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception as exc:
        pw.stop()
        pytest.skip(f"Chromium is not installed: {exc}")
    p = browser.new_page(viewport={"width": 1000, "height": 900},
                         reduced_motion="reduce")
    p.set_default_timeout(30_000)
    errors: list[str] = []
    p.on("pageerror", lambda exc: errors.append(str(exc)))
    p.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    p.goto(html.as_uri())
    p.wait_for_function("() => document.documentElement.getAttribute('data-tb-ready') === '1'")
    p.errors = errors
    yield p
    browser.close()
    pw.stop()


def _reset(page):
    for sel in ("#f-region", "#f-product"):
        if page.locator(sel).input_value() != "All":
            page.select_option(sel, "All")


def test_every_type_draws_one_mark_per_row_and_ships_no_echarts(page):
    assert page.evaluate("typeof window.echarts") == "undefined"
    assert page.evaluate("typeof window.d3") == "object"
    counts = {fig: page.locator(f"#{fig} .tb-mark").count()
              for fig in ("bar", "barh", "pie", "scatter")}
    # 4 regions; barh draws revenue and cost per region; 2 products.
    assert counts == {"bar": 4, "barh": 8, "pie": 2, "scatter": 4}
    assert page.locator("#line path.tb-line").count() == 2      # one per product
    assert page.locator("#line circle.tb-dot").count() == 8     # 4 quarters × 2
    assert page.locator("#area path.tb-area").count() == 1
    assert page.errors == []


def test_the_tooltip_shows_the_rows_own_value_in_the_pages_format(page):
    _reset(page)
    page.locator("#bar .tb-hit").first.hover()
    tip = page.locator("#bar .tb-chart-tip")
    tip.wait_for(state="visible")
    # East has the most revenue (the binding orders by -revenue): 1200+1500+1700+1600.
    assert tip.locator(".tb-chart-tip-title").inner_text() == "East"
    assert tip.locator("strong").inner_text() == "$6,000"


def test_a_click_filters_through_the_pages_control_and_colour_follows_the_series(page):
    _reset(page)
    colour_b = page.locator("#line path.tb-line").nth(1).get_attribute("stroke")
    page.locator("#bar .tb-hit").nth(1).click()          # West
    assert page.locator("#f-region").input_value() == "West"
    page.wait_for_function("() => document.querySelectorAll('#bar path.tb-mark').length === 1")
    page.locator("#bar .tb-hit").first.click()          # the same bar again clears it
    assert page.locator("#f-region").input_value() == "All"
    page.wait_for_function("() => document.querySelectorAll('#bar path.tb-mark').length === 4")
    # Filtered to product B, the B line is the only one left, and it keeps
    # B's colour instead of taking the first slot's.
    page.select_option("#f-product", "B")
    page.wait_for_function("() => document.querySelectorAll('#line path.tb-line').length === 1")
    assert page.locator("#line path.tb-line").get_attribute("stroke") == colour_b
    _reset(page)


def test_the_legend_toggles_series_but_never_hides_the_last(page):
    _reset(page)
    legend = page.locator("#barh .tb-legend-item")
    assert legend.count() == 2
    legend.nth(0).click()
    page.wait_for_function("() => document.querySelectorAll('#barh .tb-mark').length === 4")
    assert legend.nth(0).get_attribute("aria-pressed") == "false"
    legend.nth(1).click()                               # the last visible one stays
    assert legend.nth(1).get_attribute("aria-pressed") == "true"
    assert page.locator("#barh .tb-mark").count() == 4
    legend.nth(0).click()
    page.wait_for_function("() => document.querySelectorAll('#barh .tb-mark').length === 8")


def test_arrow_keys_read_each_value_aloud(page):
    _reset(page)
    page.locator("#pie svg").focus()
    page.keyboard.press("ArrowRight")
    live = page.locator("#pie .tb-sr-only")
    first = live.inner_text()
    page.keyboard.press("ArrowRight")
    second = live.inner_text()
    # Product A: every A row's revenue; B likewise. Each slice is one row.
    assert {first, second} == {"A: Revenue 7,870", "B: Revenue 5,000"}


def _boxes(page, selector):
    return page.eval_on_selector_all(
        selector, "els => els.map(e => { const b = e.getBBox(); return [b.x, b.y, b.width, b.height]; })")


def test_stacked_bars_share_one_slot_and_end_where_their_rows_end(page):
    _reset(page)
    boxes = _boxes(page, "#stack path.tb-mark")
    assert len(boxes) == 8                                   # 4 quarters × 2 products
    lefts = sorted({round(b[0], 1) for b in boxes})
    assert len(lefts) == 4, "each quarter's two segments share one slot"
    # Q1 holds A (1200+400) and B (900+300): the top segment sits on the bottom
    # one, so the stack's top is 2,800 on the axis. Read the tooltip instead of
    # pixels: it lists each row's own value, never a total.
    page.locator("#stack .tb-hit").first.hover()
    rows = page.locator("#stack .tb-chart-tip-row strong").all_inner_texts()
    assert sorted(rows) == ["1,200", "1,600"]
    assert "2,800" not in page.locator("#stack .tb-chart-tip").inner_text()


def test_marks_and_annotations_label_the_rows_own_values(page):
    _reset(page)
    labels = page.locator("#weekly .tb-marked-point text").all_text_contents()
    # W09 has the most revenue (1,700) and W16 is the last week (380).
    assert sorted(labels) == ["380", "High 1,700"]
    assert page.locator("#weekly .tb-annotation text").text_content() == "Launch"
    assert page.locator("#weekly .tb-axis-title").text_content() == "Week"


def test_drag_zooms_to_a_range_and_reset_returns(page):
    _reset(page)
    overlay = page.locator("#weekly .tb-brush .overlay")
    box = overlay.bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] * 0.05, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.45, y, steps=6)
    page.mouse.up()
    page.wait_for_function("() => document.querySelectorAll('#weekly circle.tb-dot').length < 16")
    zoomed = page.locator("#weekly circle.tb-dot").count()
    assert 2 <= zoomed < 16
    page.locator("#weekly .tb-legend-reset").click()
    page.wait_for_function("() => document.querySelectorAll('#weekly circle.tb-dot').length === 16")
    assert page.locator("#weekly .tb-legend-reset").count() == 0


def test_small_multiples_share_one_scale(page):
    _reset(page)
    panels = page.locator("#facet svg.tb-panel")
    assert panels.count() == 2                                # products A and B
    ticks = [panels.nth(i).locator(".tb-axis-v text").all_text_contents() for i in range(2)]
    assert ticks[0] and ticks[0] == ticks[1], "panels compare only on a shared scale"
    assert page.locator("#facet .tb-panel-title").all_text_contents() == ["A", "B"]


def test_currency_ticks_stay_short_and_a_unit_never_rescales(page):
    _reset(page)
    ticks = page.locator("#bar .tb-axis-v text").all_text_contents()
    assert ticks and all(t.startswith("$") and "," not in t for t in ticks), ticks
    assert "$6K" in ticks                                    # East's 6,000 on the axis
    page.locator("#area .tb-hit").first.hover()
    # Q1 revenue is 1200+900+400+300 = 2,800: the unit is a suffix, not a scale.
    assert page.locator("#area .tb-chart-tip strong").inner_text() == "2,800 u"
    assert all(t.endswith(" u") for t in page.locator("#area .tb-axis-v text").all_text_contents())

