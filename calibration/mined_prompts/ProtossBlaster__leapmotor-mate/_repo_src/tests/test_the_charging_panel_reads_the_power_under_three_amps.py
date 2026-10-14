"""A C10 charging at 1.7 kW showed no power in the Charges page's "Charging status" (#409, @arzthilfe).

His C10 AWD has an 800 V pack, his B03X a 400 V one, and only the C10 lost its power. The panel
computes |I×V| in `get_latest_status`, and that copy still refused any current under 3 A: at 718 V a
1.7 kW home charge carries 2.37 A, so the panel printed 0.00 kW (and the Overview's chip dropped its
kW) while the chart under it, which has no floor, drew the power. At 400 V the same 3 A floor sits at
1.2 kW, under his B03X's home charge.

It is the copy #307 left behind: the same reporter's 0.00 kW at 718.2 V × 1.599 A was cured in the
poller (`client._charge_power_kw`, see test_the_power_shown_is_not_gated_by_the_detection_floor.py),
not here. Whether the car is charging is decided elsewhere; this only measures how much flows, and
zero current already reads zero.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import db as poller_db
import db_reader
import pytest

T0 = datetime(2026, 10, 11, 1, 0, tzinfo=timezone.utc)


@pytest.fixture
def car(tmp_path, monkeypatch):
    path = str(tmp_path / "i409.db")
    poller_db.Database(path)
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    monkeypatch.setattr(db_reader, "_current_vehicle_id", lambda: 1)
    con = sqlite3.connect(path)
    con.execute("INSERT INTO vehicles (id, vin, car_type) VALUES (1, 'LC10000000000001', 'C10')")
    con.execute("INSERT INTO charges (id, vehicle_id, started_at, start_soc) VALUES (1, 1, ?, 69)",
                (T0.isoformat(),))
    con.commit()
    con.close()
    db_reader.set_setting("language", "en")
    return path


def _reading(path, minute, amps, volts=718.2):
    con = sqlite3.connect(path)
    con.execute("INSERT INTO positions (vehicle_id, recorded_at, soc, charging, plug_connected, "
                "charge_voltage_v, charge_current_a, latitude, longitude) VALUES (1, ?, 90, 1, 1, ?, ?, 52.0, 7.0)",
                ((T0 + timedelta(minutes=minute)).isoformat(), volts, amps))
    con.commit()
    con.close()


def test_a_charge_under_three_amps_has_its_power(car):
    """718.2 V × 2.37 A = 1.70 kW. It read 0.0."""
    _reading(car, 0, -2.37)
    assert db_reader.get_latest_status()["charge_power_kw"] == pytest.approx(1.70, abs=0.005)


def test_the_panel_says_what_the_chart_under_it_draws(car):
    """His polls taper through the old floor; the panel follows them as the chart does."""
    for minute, amps in enumerate((-3.3, -3.4, -2.0, -2.3, -1.6)):
        _reading(car, minute, amps)
        panel = db_reader.get_latest_status()["charge_power_kw"]
        chart = db_reader.get_charge_power_curve(1)["power"][-1]
        assert panel == pytest.approx(chart, abs=0.005), f"{amps} A: panel {panel}, chart {chart}"


def test_the_charging_status_panel_prints_it(car):
    pytest.importorskip("httpx", reason="Starlette's TestClient is built on httpx")
    import main
    from starlette.testclient import TestClient
    _reading(car, 0, -2.37)
    html = TestClient(main.app).get("/api/charging-live").text
    assert "1.70" in html and "0.00" not in html


def test_a_full_speed_charge_is_unchanged(car):
    _reading(car, 0, -15.3)
    assert db_reader.get_latest_status()["charge_power_kw"] == pytest.approx(10.99, abs=0.005)


def test_no_current_is_still_no_power(car):
    _reading(car, 0, 0.0)
    assert db_reader.get_latest_status()["charge_power_kw"] == 0.0


def test_a_reading_the_car_did_not_send_is_not_a_power(car):
    _reading(car, 0, None)
    assert db_reader.get_latest_status()["charge_power_kw"] == 0.0
