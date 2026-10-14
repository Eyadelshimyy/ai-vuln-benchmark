"""The Charges page reloads its history when the car ends its charge session, not when the charging flag drops.

The live panel polls every 30 s, and the page reloaded once the panel's charging flag went from on to off.
That was the end of the session while the flag stayed on for the whole of it (#307). The flag now goes off
once the battery is at the car's limit with nothing going in, while the car holds the session open some
minutes more, and the charge is in the history only when the session closes. So the panel carries the open
session, and the page reloads when it is gone.

Measured in a real browser, like the other live-panel tests; skips where it cannot run.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi", reason="web/main.py needs fastapi (absent in the minimal CI env)")
pytest.importorskip("uvicorn", reason="the page has to be SERVED, not rendered in-process")
sync_api = pytest.importorskip("playwright.sync_api", reason="needs playwright + `playwright install chromium`")

from web_in_a_browser import chromium, seed_database, served

VIN = "LVIN0000000000001"
PLUGGED_IN = datetime(2026, 10, 11, 6, 13, tzinfo=timezone.utc)
READING = ("INSERT INTO positions (vehicle_id, recorded_at, plug_connected, charging, charge_voltage_v,"
           " charge_current_a, soc) VALUES (1, ?, 1, ?, 430, ?, ?)")
# The panel's poll, fired and not awaited: a reload it sets off tears the page down under the call.
_POLL = "() => { htmx.ajax('GET', 'api/charging-live', { target: '#charging-live-panel', swap: 'innerHTML' }) }"
_PANEL = "() => document.querySelector('#charging-live-panel').innerText"


def _a_charge_in_progress():
    rows = [("INSERT INTO settings (key, value) VALUES ('timezone', 'UTC')", ()),
            ("INSERT INTO charges (id, vehicle_id, started_at, start_soc) VALUES (9, 1, ?, 77.1)",
             (PLUGGED_IN.isoformat(),))]
    for k in range(6):
        rows.append((READING, ((PLUGGED_IN + timedelta(seconds=30 * k)).isoformat(), 1, -7, 77.1 + 0.1 * k)))
    return rows


@pytest.fixture(scope="module")
def mate(tmp_path_factory):
    data = tmp_path_factory.mktemp("mate-session-reload")
    db = data / "leapmotor_mate.db"
    seed_database(db, VIN, _a_charge_in_progress())
    with served(data, db) as url:
        yield url, db


def _write(db, sql, params):
    con = sqlite3.connect(db)
    con.execute(sql, params)
    con.commit()
    con.close()


def test_the_page_reloads_once_the_session_is_closed_and_not_when_the_battery_is_full(mate):
    url, db = mate
    pw, browser = chromium(sync_api)
    try:
        page = browser.new_page()
        assert page.goto(f"{url}/charges").status == 200
        assert "0 sessions recorded" in page.inner_text("body")
        page.evaluate("() => { window.stays = 1 }")

        # The battery reaches 100 % with nothing going in; the car holds its session open.
        _write(db, READING, ((PLUGGED_IN + timedelta(minutes=40)).isoformat(), 0, 1.1, 100.0))
        page.evaluate(_POLL)
        page.wait_for_function(f"() => ({_PANEL})().includes('Fully charged')")
        page.wait_for_timeout(300)
        assert page.evaluate("() => window.stays") == 1, "the page reloaded before the charge was in the history"

        # The car ends its session: the charge is filed, and the page shows it.
        _write(db, "UPDATE charges SET ended_at=?, end_soc=100, energy_added_kwh=15 WHERE id=9",
               ((PLUGGED_IN + timedelta(minutes=51)).isoformat(),))
        with page.expect_event("load"):
            page.evaluate(_POLL)
        assert page.evaluate("() => window.stays") is None
        assert "1 sessions recorded" in page.inner_text("body")
    finally:
        browser.close()
        pw.stop()
