"""A charge reads complete when the battery is at the car's own charge limit, the cable still in and
nothing flowing — not when signal 3736 says so.

3736 is no completion flag on any car seen: the official app interprets it as isHaveFullCharge (a tip
about calibrating the battery with a charge to 100 %), the BEVs in the data send it as 0 or not at all,
and a B10 REEV holds it at 1 for the whole charge. The limit is the one the car reports, assumed 100 %
if it reports none. Cars stop on it or one point under it (a car whose SoC comes in whole percent).
"""
import pytest

pytest.importorskip("httpx", reason="Starlette TestClient needs httpx")

import db_reader
from test_the_cable_chip_says_the_charge_is_not_running import _chip
from test_the_cable_shows_while_the_charger_holds import VIN, car  # noqa: F401  (car: fixture)


def _limit(pct):
    db_reader.set_setting(f"charge_limit_percent_{VIN.lower()}", str(pct))


@pytest.mark.parametrize("soc, limit, complete", [
    (100.0, None, True),
    (99.0, None, True),
    (98.9, None, False),
    (90.0, 90, True),
    (89.0, 90, True),
    (88.9, 90, False),
], ids=["full, no limit reported", "a point under 100", "short of 100", "at the limit",
        "a point under the limit", "short of the limit"])
def test_complete_is_the_battery_at_the_cars_limit(car, soc, limit, complete):
    if limit:
        _limit(limit)
    car.frame(plug_connected=1, soc=soc)
    assert bool(db_reader.get_latest_status()["charge_completed"]) is complete


def test_the_cars_own_flag_is_kept_but_not_read(car):
    car.frame(plug_connected=1, soc=60.0, charge_completed=1)
    status = db_reader.get_latest_status()
    assert not status["charge_completed"]
    assert status["charge_completed_raw"] == 1


@pytest.mark.parametrize("frame", [
    dict(plug_connected=1, charging=1),
    dict(plug_connected=1, charge_current_a=-1.6),
    dict(plug_connected=0, ac_port_mode=1, charge_current_a=-1.6),
], ids=["the charging flag", "current under the flag", "current under the flag, the cable on the AC port"])
def test_not_complete_while_the_current_flows(car, frame):
    car.frame(soc=100.0, **frame)
    assert not db_reader.get_latest_status()["charge_completed"]


def test_not_complete_without_the_cable(car):
    car.frame(plug_connected=0, soc=100.0)
    assert not db_reader.get_latest_status()["charge_completed"]


def test_the_overview_and_the_charges_page_say_it(car):
    """The session closed, the AC port still reports the cable, the battery at the car's 80 %."""
    _limit(80)
    car.frame(plug_connected=0, ac_port_mode=1, soc=80.0)
    assert _chip(car.get("/api/overview-hero")) == "Cable connected · 80% (Charge complete)"
    assert "Fully charged" in car.get("/charges")
