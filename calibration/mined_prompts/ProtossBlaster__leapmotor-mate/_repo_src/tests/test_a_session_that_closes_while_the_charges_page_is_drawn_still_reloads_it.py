"""A charge session that closes while the Charges page is being drawn still gets the page reloaded.

The page reloads its history once the live panel stops carrying the open session. The page reads the
session before its history, so a session closing in between, during the chart of the charge in progress
(which may wait on Home Assistant), still reads as open on the page, and the panel's next poll reloads it.
Read after the history, the page would show the old totals with no session to lose, and never reload.
"""
import sqlite3

import pytest

pytest.importorskip("httpx", reason="Starlette TestClient needs httpx")

import db_reader
import main
from test_the_cable_shows_while_the_charger_holds import car  # noqa: F401  (car: fixture)

CLOSED = "UPDATE charges SET ended_at='2026-10-11T09:00:00+00:00', end_soc=100, energy_added_kwh=15 WHERE id=9"


def test_the_page_still_carries_the_session_it_was_drawn_for(car, monkeypatch):
    with sqlite3.connect(db_reader.DB_PATH) as c:
        c.execute("INSERT INTO charges (id, vehicle_id, started_at, start_soc) VALUES (9, 1, '2026-10-11T08:00:00+00:00', 80)")
    car.frame(plug_connected=1, soc=100.0, charge_current_a=1.1)
    chart = main._charging_chart

    def the_car_ends_its_session_meanwhile():
        with sqlite3.connect(db_reader.DB_PATH) as c:
            c.execute(CLOSED)
        return chart()

    monkeypatch.setattr(main, "_charging_chart", the_car_ends_its_session_meanwhile)
    page = car.get("/charges")
    assert "0 sessions recorded" in page            # drawn before the close
    assert 'data-session="9"' in page                # so the next poll, with no session, reloads it
    assert 'data-session=""' in car.get("/api/charging-live")
