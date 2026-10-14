"""The diagnostics' "SoC by day" counts the kilometres a car drove, not the gap to another number.

#338 (@dommi1966, a T03 and an A10 on one account): the T03's bundle read "drove 9228 km" on
28/09 and "drove 8573 km" on 01/10, days it drove a dozen. The section took the day's highest
odometer minus its lowest, and the lowest was either a reading without an odometer, which is stored
as 0, or a reading of the A10 (777 km, 96.8 %) filed under the T03 before 4.14.1. The SoC column
carried the A10's 96.8 % too.

Now a reading without an odometer does not count for the kilometres, and a reading farther from the
day's median odometer than a car drives in a day is left out of the day's figures and counted apart,
so it still shows.
"""
from datetime import datetime, timedelta, timezone

import db as PollerDB
import db_reader
import diagnostics
import pytest


@pytest.fixture
def day(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    pdb = PollerDB.Database(path)
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    pdb.ensure_vehicle("LVIN0000000000001", "T03", 2024)
    d = (datetime.now(timezone.utc) - timedelta(days=2)).replace(hour=8, minute=0, second=0,
                                                                microsecond=0)

    def readings(*pairs):
        for i, (soc, odo) in enumerate(pairs):
            pdb._conn.execute(
                "INSERT INTO positions (vehicle_id, recorded_at, soc, odometer_km) VALUES (1,?,?,?)",
                ((d + timedelta(minutes=i)).isoformat(), soc, odo))
        pdb._conn.commit()
        return [ln for ln in diagnostics._soc_daily_section().splitlines() if d.date().isoformat() in ln][0]

    return readings


def test_a_reading_without_an_odometer_does_not_count(day):
    line = day((47.0, 9500.0), (47.0, 0.0), (46.0, 9512.0), (46.0, None))
    assert "drove 12 km" in line, line


def test_a_reading_of_the_other_car_is_left_out_and_counted_apart(day):
    line = day((41.0, 9350.0), (41.0, 9350.0), (96.8, 777.0), (40.0, 9351.0), (40.0, 9351.0))
    assert "drove 1 km" in line, line
    assert "soc 41.0→40.0" in line, line
    assert "⚠ 1 reading" in line, line


def test_a_long_day_still_counts_in_full(day):
    line = day((90.0, 9000.0), (60.0, 9400.0), (30.0, 9800.0))
    assert "drove 800 km" in line and "soc 90.0→30.0" in line, line
    assert "⚠" not in line, line
