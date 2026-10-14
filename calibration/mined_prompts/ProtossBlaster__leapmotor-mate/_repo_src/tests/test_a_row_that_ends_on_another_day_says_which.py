"""A row's end time carries its day when that is not the day the row started on.

A charge from 22:30 to 00:30 reads "22:30 → 04 Jul 00:30", and one that lasts longer than a day says
which day it ended; without the day, both read as ending earlier than they started, or a day short. The
day is the local one, as the times are. A row that starts and ends on one day reads as before.
"""
import re

import db as D
import db_reader
import pytest

pytest.importorskip("httpx", reason="Starlette's TestClient is built on httpx")
pytest.importorskip("fastapi", reason="web.main needs fastapi (absent in the minimal CI env)")

from test_a_days_heading_sums_up_its_trips import _client
from test_trips_calendar_range import _text

# UTC; the install's zone is two hours ahead in July.
_ROWS = [(1, "2026-07-03T06:00", "2026-07-03T07:00"),     # 08:00 → 09:00
         (2, "2026-07-03T20:30", "2026-07-03T22:30"),     # 22:30 → 00:30 the next day, the same day in UTC
         (3, "2026-07-03T10:00", "2026-07-05T03:12")]     # 12:00 → 05:12 two days later


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    c = D.Database(path)._conn
    c.execute("INSERT INTO vehicles (id, vin, car_type) VALUES (1,'LFZTEST0000000001','B10')")
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('timezone', 'Europe/Warsaw')")
    for i, start, end in _ROWS:
        c.execute("INSERT INTO charges (id, vehicle_id, started_at, ended_at, start_soc, end_soc, energy_added_kwh,"
                  " duration_min, charge_type, location_type) VALUES (?,1,?,?,40,60,10,60,'AC','HOME')",
                  (i, f"{start}:00+00:00", f"{end}:00+00:00"))
        c.execute("INSERT INTO trips (id, vehicle_id, started_at, ended_at, distance_km, start_soc, end_soc,"
                  " duration_min) VALUES (?,1,?,?,10.0,60,55,20)", (i, f"{start}:00+00:00", f"{end}:00+00:00"))
    c.commit()
    c.close()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    monkeypatch.setattr(db_reader, "get_language", lambda: "en")
    return _client()


@pytest.mark.parametrize("page", ["charges", "trips"])
def test_the_end_says_its_day_when_it_is_another(client, page):
    rows = _text(client.get(f"/api/{page}/calendar/day", params={"year": 2026, "month": 7, "day": 3}).text)
    for span in ("08:00 → 09:00", "22:30 → 04 Jul 00:30", "12:00 → 05 Jul 05:12"):
        assert re.search(re.escape(span).replace(r"\ ", r"\s+"), rows), (span, rows)


def test_an_end_that_does_not_parse_prints_as_it_did(client):
    """A time typed in by hand is stored as typed when it is no date; the drawer still opens, and its row with it."""
    c = D.Database(db_reader.DB_PATH)._conn
    c.execute("INSERT INTO charges (id, vehicle_id, started_at, ended_at, start_soc, end_soc, energy_added_kwh,"
              " duration_min, charge_type, location_type, manual_entry) VALUES (9,1,'2026-07-03T15:00:00+00:00',"
              "'2026-07-99T18:00:00',40,60,10,60,'AC','HOME',1)")
    c.commit()
    c.close()
    page = client.get("/api/charges/calendar/day", params={"year": 2026, "month": 7, "day": 3})
    assert page.status_code == 200
    rows = _text(page.text)
    assert "17:00 → 18:00" in rows and "22:30 → 04 Jul 00:30" in rows, rows
