"""The times under a charge's chart and the minutes under a trip's do not run into each other on a phone.

The time axis asked for six intervals at any width: on a computer they stand some 90 px apart, but a phone's chart
is a third as wide, and the library leaves labels that overlap where they are on this kind of axis, so a phone
read "19:0021:5100:42". The axis now prints as many labels as its width holds, up to the computer's.

Measured in a real browser, because where the labels land depends on the width the chart is drawn at.
"""
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi", reason="web/main.py needs fastapi (absent in the minimal CI env)")
pytest.importorskip("uvicorn", reason="the page has to be SERVED, not rendered in-process")
sync_api = pytest.importorskip("playwright.sync_api", reason="needs playwright + `playwright install chromium`")

from web_in_a_browser import chromium, seed_database, served

VIN = "LVIN0000000000001"
START = datetime(2026, 9, 26, 19, 0, tzinfo=timezone.utc)


def _a_nights_charge_and_a_long_trip():
    """A charge of 20 hours polled every 30 s, and a drive of two and a half hours with a point a minute."""
    rows = [("INSERT INTO settings (key, value) VALUES ('timezone', 'UTC')", ()),
            ("INSERT INTO settings (key, value) VALUES ('elevation_enabled', '0')", ()),
            ("INSERT INTO charges (id, vehicle_id, started_at, ended_at, start_soc, end_soc, energy_added_kwh,"
             " location_type) VALUES (1,1,?,?,20,80,40.0,'HOME')",
             (START.isoformat(), (START + timedelta(seconds=30 * 2399)).isoformat()))]
    rows += [("INSERT INTO positions (vehicle_id, recorded_at, charging, charge_voltage_v, charge_current_a, soc)"
              " VALUES (1,?,1,400,?,?)", ((START + timedelta(seconds=30 * k)).isoformat(), -10 - k % 7, 20 + k / 40))
             for k in range(2400)]
    drive = START + timedelta(days=2)
    rows.append(("INSERT INTO trips (id, vehicle_id, started_at, ended_at, distance_km, duration_min, start_soc,"
                 " end_soc) VALUES (1,1,?,?,180,150,90,40)", (drive.isoformat(), (drive + timedelta(minutes=150)).isoformat())))
    rows += [("INSERT INTO trip_positions (trip_id, recorded_at, latitude, longitude, speed_kmh, soc, elevation_m)"
              " VALUES (1,?,?,9.0,?,?,?)", ((drive + timedelta(minutes=k)).isoformat(), 45 + k * 0.01, 80 + k % 20,
                                            90 - k / 3, 100 + k)) for k in range(151)]
    return rows


@pytest.fixture(scope="module")
def mate(tmp_path_factory):
    data = tmp_path_factory.mktemp("mate-time-axis")
    db = data / "mate.sqlite"
    seed_database(db, VIN, _a_nights_charge_and_a_long_trip())
    with served(data, db) as url:
        yield url


_AXIS = """(id) => { const el = document.getElementById(id);
  const xs = [...el.querySelectorAll('.apexcharts-xaxis-texts-g text')]
    .map(t => ({text: (t.querySelector('tspan') || t).textContent, box: t.getBoundingClientRect()})).filter(l => l.text);
  return {width: el.clientWidth, labels: xs.map(l => l.text),
          gaps: xs.slice(1).map((l, i) => Math.round(l.box.left - xs[i].box.right))}; }"""

CHARTS = {"charge": ("/charges?highlight=1", "pc-1"), "trip": ("/trips/1", "trip-profile")}


def _axis(url, chart, width):
    path, id_ = CHARTS[chart]
    pw, browser = chromium(sync_api)
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        assert page.goto(url + path).status == 200
        page.wait_for_function(f"() => document.getElementById('{id_}') && document.getElementById('{id_}')._c")
        return page.evaluate(_AXIS, id_)
    finally:
        browser.close()
        pw.stop()


@pytest.mark.parametrize("width", [390, 360])
@pytest.mark.parametrize("chart", ["charge", "trip"])
def test_on_a_phone_the_labels_stand_apart(mate, chart, width):
    a = _axis(mate, chart, width)
    assert len(a["labels"]) >= 3, ("the axis still tells the time", a)
    assert min(a["gaps"]) >= 6, ("no label runs into the next", a)


@pytest.mark.parametrize("chart", ["charge", "trip"])
def test_on_a_computer_the_axis_keeps_its_labels(mate, chart):
    a = _axis(mate, chart, 1280)
    assert len(a["labels"]) >= 7, ("as many labels as before", a)
    assert min(a["gaps"]) >= 6, a
