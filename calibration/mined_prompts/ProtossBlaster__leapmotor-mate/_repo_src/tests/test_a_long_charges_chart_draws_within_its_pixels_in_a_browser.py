"""The chart of a long charge draws no more readings than it has pixels, and a short charge draws every one.

A night's charge polled every 30 s brings two or three readings a pixel, and the chart's library pays for every
one of them: the chart took over a second to appear. The axis keeps every K-th reading and the last, so the
line is drawn from real readings at their own instants, as many as the chart's width can show; on a phone
fewer than on a computer. A charge whose axis, the blanks of its pauses included, fits in the chart's width is drawn
as it always was, and a pause still breaks the line where it did.

Measured in a real browser, because the thinning follows the width the chart is drawn at.
"""
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi", reason="web/main.py needs fastapi (absent in the minimal CI env)")
pytest.importorskip("uvicorn", reason="the page has to be SERVED, not rendered in-process")
sync_api = pytest.importorskip("playwright.sync_api", reason="needs playwright + `playwright install chromium`")

from web_in_a_browser import chromium, seed_database, served

VIN = "LVIN0000000000001"
PLUGGED_IN = datetime(2026, 9, 29, 19, 0, tzinfo=timezone.utc)
LONG, SHORT = 2400, 13          # a night's charge polled every 30 s; an hour polled every five minutes
PAUSES = ((1200, 6 * 60), (1800, 6 * 3600))     # after which reading of the third charge, and for how long


def _charges():
    charge = ("INSERT INTO charges (id, vehicle_id, started_at, ended_at, start_soc, end_soc, energy_added_kwh,"
              " location_type) VALUES (?,1,?,?,20,80,40.0,'HOME')")
    sample = ("INSERT INTO positions (vehicle_id, recorded_at, charging, charge_voltage_v, charge_current_a, soc)"
              " VALUES (1,?,1,400,?,?)")
    rows = [("INSERT INTO settings (key, value) VALUES ('timezone', 'UTC')", ())]
    for cid, start, n, step, pauses in ((1, PLUGGED_IN, LONG, 30, ()), (2, PLUGGED_IN + timedelta(days=1), SHORT, 300, ()),
                                        (3, PLUGGED_IN + timedelta(days=2), LONG, 30, PAUSES)):
        at = [start + timedelta(seconds=step * k + sum(p for after, p in pauses if k >= after)) for k in range(n)]
        rows.append((charge, (cid, start.isoformat(), at[-1].isoformat())))
        for k in range(n):
            rows.append((sample, (at[k].isoformat(), -10 - (k % 7), 20 + 60 * k / n)))
    return rows


@pytest.fixture(scope="module")
def mate(tmp_path_factory):
    data = tmp_path_factory.mktemp("mate-chart-pixels")
    db = data / "leapmotor_mate.db"
    seed_database(db, VIN, _charges())
    with served(data, db) as url:
        yield url


_DRAWN = """(cid) => { const el = document.getElementById('pc-' + cid), labels = el._c.w.globals.categoryLabels;
  const power = el._c.w.config.series.find(s => s.name[0] !== '_').data;     // the first drawn line, after the axes' placeholders
  const holes = power.map((v, i) => v == null && power[i - 1] != null ? labels[i - 1] : null).filter(Boolean);
  return {width: el.clientWidth, categories: labels.length, first: labels[0], last: labels[labels.length - 1], holes: holes}; }"""


def _drawn(browser, url, cid, width):
    page = browser.new_page(viewport={"width": width, "height": 900})
    assert page.goto(f"{url}/charges?highlight={cid}").status == 200
    page.wait_for_function(f"() => document.getElementById('pc-{cid}') && document.getElementById('pc-{cid}')._c")
    return page.evaluate(_DRAWN, cid)


@pytest.mark.parametrize("width", [1280, 390])
def test_a_nights_charge_draws_as_many_readings_as_the_chart_is_wide(mate, width):
    pw, browser = chromium(sync_api)
    try:
        d = _drawn(browser, mate, 1, width)
    finally:
        browser.close()
        pw.stop()
    assert 0 < d["width"] < LONG
    assert d["width"] / 2 <= d["categories"] <= d["width"] + 1, d
    assert d["first"] == "19:00" and d["last"] == "14:59", ("the first and the last reading are drawn", d)


def test_a_short_charge_draws_every_reading(mate):
    pw, browser = chromium(sync_api)
    try:
        d = _drawn(browser, mate, 2, 1280)
    finally:
        browser.close()
        pw.stop()
    assert d["categories"] == SHORT and d["first"] == "19:00" and d["last"] == "20:00", d


@pytest.mark.parametrize("width", [1280, 390])
def test_a_pause_in_a_nights_charge_still_breaks_the_line(mate, width):
    """A hole in the recording is found among all the readings, before the thinning, and keeps at least one blank:
    the blanks count against the chart's width like the readings, so a long pause does not blow the budget."""
    pw, browser = chromium(sync_api)
    try:
        d = _drawn(browser, mate, 3, width)
    finally:
        browser.close()
        pw.stop()
    assert d["holes"] == ["04:59", "10:05"], ("the line breaks after the last reading before each pause", d)
    assert d["width"] / 2 <= d["categories"] <= d["width"] + 1 + 3 * len(PAUSES), d
    assert d["first"] == "19:00" and d["last"] == "21:05", d
