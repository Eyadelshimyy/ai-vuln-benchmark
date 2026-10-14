"""A charge session left open on a full battery no longer reads as running.

The pages say "charging" while a charge session is open, so that a charge whose current dips under the
detection floor stays on the screen (#307). But the car ends its session some time after the energy
stops, with the cable still in. At the car's limit with no current going in, the charge is over, and the
pages say "Charge complete" while the session waits for the car to end it. Below the limit the session
still reads as running, as before.
"""
import sqlite3

import pytest

pytest.importorskip("httpx", reason="Starlette TestClient needs httpx")

import db_reader
import main
from starlette.testclient import TestClient
from test_the_cable_chip_says_the_charge_is_not_running import _chip
from test_the_cable_shows_while_the_charger_holds import car  # noqa: F401  (car: fixture)


@pytest.fixture
def session(car):
    """The car with a charge session open."""
    with sqlite3.connect(db_reader.DB_PATH) as c:
        c.execute("INSERT INTO charges (vehicle_id, started_at, start_soc) VALUES (1, '2026-10-11T08:00:00+00:00', 80)")
    return car


@pytest.mark.parametrize("soc, amps, charging", [
    (100.0, 1.1, False),
    (100.0, -1.6, True),
    (60.0, 0.0, True),
], ids=["full, nothing going in", "full, current still going in", "below the limit"])
def test_an_open_session_reads_as_running_until_the_battery_is_full(session, soc, amps, charging):
    session.frame(plug_connected=1, soc=soc, charge_current_a=amps)
    assert bool(db_reader.get_latest_status()["charging"]) is charging


def test_the_overview_and_the_charges_page_say_it_is_complete(session):
    session.frame(plug_connected=1, soc=100.0, charge_current_a=1.1)
    assert _chip(session.get("/api/overview-hero")) == "Cable connected · 100% (Charge complete)"
    assert "Fully charged" in session.get("/charges")


def test_the_wallbox_profile_stays_put_while_the_session_is_open(session):
    session.frame(plug_connected=1, soc=100.0, charge_current_a=1.1)
    page = TestClient(main.app).post("api/settings/wallbox-profiles/any/load").text
    assert "Cannot switch profiles while a charge is active" in page
