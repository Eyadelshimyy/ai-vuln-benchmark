"""Large table figures paint a window; 500 rows stay fully rendered.

Skipped unless Playwright can launch Chromium. The CI job ``ui-smoke``
installs Chromium and runs this file. A normal ``pytest tests/`` skips it
when the browser is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.reports.template_package import TemplatePackage

_BIG = 5000
_CAP = 500


def _frame(n: int) -> pd.DataFrame:
    amounts = [1] * n
    if n > 2500:
        amounts[0] = 2500
        amounts[4000] = 2500
    amounts[-1] = n
    notes = ["row"] * n
    notes[-1] = "LAST"
    if n > 10:
        notes[10] = "findme"
    return pd.DataFrame({
        "row_id": list(range(n)),
        "label": [f"{i:04d}" for i in range(n)],
        "note": notes,
        "amount": amounts,
    })


def _model(df: pd.DataFrame) -> DataModel:
    model = DataModel("window_model")
    model.add_connector(MemoryConnector("mem", tables={"lines": df}))
    model.add_table("lines", connector="mem", source="lines")
    model.add_dimension(
        "dim_line", table_name="lines", key_col="row_id",
        attributes=["label", "note"],
    )
    model.add_fact(
        "fact_lines", table_name="lines", measures=["amount"], foreign_keys={},
    )
    model.add_measure("amount", column="amount", agg="sum")
    model.connect()
    return model


def _build(tmp_path: Path, n: int, *, windowed: bool) -> Path:
    pkg = tmp_path / f"pkg{n}"
    pkg.mkdir()
    data = {
        "lines": {
            "model": "window_model",
            "query": {
                "fact": "fact_lines",
                "measures": ["amount"],
                "dimensions": ["dim_line.label", "dim_line.note"],
                "order_by": ["dim_line.label"],
            },
        },
    }
    if windowed:
        data["lines_total"] = {
            "model": "window_model",
            "query": {"fact": "fact_lines", "measures": ["amount"]},
        }
    (pkg / "report.json").write_text(json.dumps({
        "name": "Window",
        "author": "t",
        "data": data,
    }), encoding="utf-8")
    if windowed:
        template = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>Window</title></head><body>"
            "<select id=\"f\" data-tb-filter data-tb-binding=\"lines\" "
            "data-tb-column=\"dim_line.note\"></select>"
            "<input id=\"q\" data-tb-search data-tb-binding=\"lines\">"
            "<table id=\"big\" data-tb-figure=\"table\" data-tb-binding=\"lines\" "
            "data-tb-sort data-tb-bars=\"amount\" data-tb-formats=\"amount=comma\" "
            "data-tb-totals=\"lines_total\"></table>"
            "</body></html>"
        )
    else:
        template = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>Window</title></head><body>"
            "<table id=\"small\" data-tb-figure=\"table\" "
            "data-tb-binding=\"lines\" data-tb-sort></table>"
            "</body></html>"
        )
    (pkg / "template.html").write_text(template, encoding="utf-8")
    out = tmp_path / f"out{n}.html"
    TemplatePackage(str(pkg)).render({"window_model": _model(_frame(n))}, str(out))
    return out


def _browser():
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception as exc:
        pw.stop()
        pytest.skip(f"Chromium is not installed: {exc}")
    return pw, browser


def _open(browser, html: Path):
    page = browser.new_page()
    page.set_default_timeout(60_000)
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(html.as_uri())
    return page, errors


def test_five_thousand_rows_window_sort_search_bars_and_print(tmp_path: Path) -> None:
    pw, browser = _browser()
    try:
        page, errors = _open(browser, _build(tmp_path, _BIG, windowed=True))
        page.wait_for_function(
            """() => {
              const t = document.querySelector('#big');
              if (!t || !t.classList.contains('tb-window')) return false;
              const n = t.querySelectorAll('tbody tr:not(.tb-window-pad)').length;
              return n > 0 && n < 200;
            }"""
        )
        # The totals cell is the model's sum of every row, not of the window.
        # (The label column is digit-shaped, so the runtime does not write
        # the word Total into that first numeric cell.)
        assert page.evaluate("() => document.querySelector('#big tfoot').hidden") is False
        assert f"{sum(_frame(_BIG)['amount']):,}" in page.evaluate(
            "() => document.querySelector('#big tfoot').textContent"
        )

        def bar_width() -> str | None:
            return page.evaluate(
                """() => {
                  const cell = [...document.querySelectorAll(
                    '#big tbody tr:not(.tb-window-pad) td')].find(
                      td => td.textContent === '2,500');
                  const bar = cell && cell.querySelector('.tb-bar');
                  return bar ? bar.style.width : null;
                }"""
            )

        top = bar_width()
        assert top == "50%"  # 2500 / column max 5000, not the window's max
        page.evaluate(
            """() => {
              const scroller = document.querySelector('#big').closest('.tb-scroll');
              const row = document.querySelector('#big tbody tr:not(.tb-window-pad)');
              const rh = row.getBoundingClientRect().height;
              scroller.scrollTop = Math.round(4000 * rh);
            }"""
        )
        # Row 0 also reads 2,500, so that text alone is true before the scroll
        # handler runs. The label 4000 ("4,000") is only in this window.
        page.wait_for_function(
            """() => {
              const rows = [...document.querySelectorAll(
                '#big tbody tr:not(.tb-window-pad)')];
              const row = rows.find(tr => {
                const cells = [...tr.querySelectorAll('td')].map(td => td.textContent);
                return cells.indexOf('4,000') !== -1 && cells.indexOf('2,500') !== -1;
              });
              if (!row) return false;
              const cell = [...row.querySelectorAll('td')]
                .find(td => td.textContent === '2,500');
              const bar = cell && cell.querySelector('.tb-bar');
              return !!bar && bar.style.width === '50%';
            }"""
        )
        assert bar_width() == top
        page.evaluate(
            """() => {
              const scroller = document.querySelector('#big').closest('.tb-scroll');
              scroller.scrollTop = scroller.scrollHeight;
            }"""
        )
        page.wait_for_function(
            """() => [...document.querySelectorAll(
                 '#big tbody tr:not(.tb-window-pad)')]
               .some(tr => tr.textContent.indexOf('LAST') !== -1)"""
        )
        page.evaluate(
            """() => {
              document.querySelector('#big').closest('.tb-scroll').scrollTop = 0;
            }"""
        )
        # The label column is all digits, so "0000" renders as 0. The first
        # row is the one whose amount is 2,500 and whose note is not LAST.
        page.wait_for_function(
            """() => {
              const row = document.querySelector('#big tbody tr:not(.tb-window-pad)');
              if (!row) return false;
              const text = row.textContent;
              if (text.indexOf('2,500') === -1 || text.indexOf('LAST') !== -1) return false;
              const cell = [...row.querySelectorAll('td')]
                .find(td => td.textContent === '2,500');
              const bar = cell && cell.querySelector('.tb-bar');
              return !!bar && bar.style.width === '50%';
            }"""
        )
        assert bar_width() == top

        amount = page.locator("#big thead th", has_text="Amount").locator("button")
        amount.click()
        amount.click()
        page.wait_for_function(
            """() => {
              const row = document.querySelector('#big tbody tr:not(.tb-window-pad)');
              return row && row.textContent.indexOf('5,000') !== -1
                  && row.textContent.indexOf('LAST') !== -1;
            }"""
        )

        page.locator("#q").fill("findme")
        page.wait_for_function(
            """() => {
              const rows = [...document.querySelectorAll(
                '#big tbody tr:not(.tb-window-pad)')];
              return rows.length === 1 && rows[0].textContent.indexOf('findme') !== -1;
            }"""
        )
        assert page.evaluate("() => document.querySelector('#big tfoot').hidden") is True
        page.locator("#q").fill("")
        page.wait_for_function(
            """() => document.querySelectorAll(
                 '#big tbody tr:not(.tb-window-pad)').length > 1
               && document.querySelectorAll(
                 '#big tbody tr:not(.tb-window-pad)').length < 200"""
        )

        page.locator("#f").select_option("findme")
        page.wait_for_function(
            """() => {
              const rows = [...document.querySelectorAll(
                '#big tbody tr:not(.tb-window-pad)')];
              return rows.length === 1 && rows[0].textContent.indexOf('findme') !== -1;
            }"""
        )
        page.locator("#f").select_option("All")
        page.wait_for_function(
            """() => {
              const n = document.querySelectorAll(
                '#big tbody tr:not(.tb-window-pad)').length;
              return n > 1 && n < 200;
            }"""
        )

        page.emulate_media(media="print")
        page.evaluate("() => window.dispatchEvent(new Event('beforeprint'))")
        page.wait_for_function(
            """() => document.querySelectorAll(
                 '#big tbody tr:not(.tb-window-pad)').length === 5000"""
        )
        page.emulate_media(media="screen")
        page.evaluate("() => window.dispatchEvent(new Event('afterprint'))")
        page.wait_for_function(
            """() => {
              const n = document.querySelectorAll(
                '#big tbody tr:not(.tb-window-pad)').length;
              return n > 0 && n < 200;
            }"""
        )
        assert not errors, "\n".join(errors)
    finally:
        browser.close()
        pw.stop()


def test_five_hundred_rows_render_every_row(tmp_path: Path) -> None:
    pw, browser = _browser()
    try:
        page, errors = _open(browser, _build(tmp_path, _CAP, windowed=False))
        page.wait_for_function(
            """() => {
              const t = document.querySelector('#small');
              if (!t || t.classList.contains('tb-window')) return false;
              const n = t.querySelectorAll('tbody tr').length;
              return !!t.querySelector('thead .tb-sort') && n === 500
                  && !t.querySelector('.tb-window-pad');
            }"""
        )
        assert not errors, "\n".join(errors)
    finally:
        browser.close()
        pw.stop()
