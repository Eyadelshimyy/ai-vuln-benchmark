"""get_last_charge_end() bounds the "since last charge" getEC card (Statistics page): it must
return the most recently COMPLETED charge's end, ignoring in-progress charges (ended_at IS NULL)
and out-of-order rows, and return None when no charge has ever finished."""
import sqlite3

import db_reader


def _db(charges):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE charges (id INT, started_at TEXT, ended_at TEXT, vehicle_id INTEGER DEFAULT 1)")
    con.executemany("INSERT INTO charges (id, started_at, ended_at) VALUES (?,?,?)", charges)
    con.commit()
    return con


def test_returns_most_recent_completed_charge_end(monkeypatch):
    con = _db([
        (1, "2026-06-01T08:00:00+00:00", "2026-06-01T10:00:00+00:00"),
        (2, "2026-06-05T08:00:00+00:00", "2026-06-05T11:30:00+00:00"),   # most recent end
        (3, "2026-06-03T08:00:00+00:00", "2026-06-03T09:00:00+00:00"),  # started later than charge 2 but ended earlier — must not win
    ])
    monkeypatch.setattr(db_reader, "_get", lambda: con)
    dt = db_reader.get_last_charge_end()
    assert dt is not None
    assert dt.astimezone(__import__("datetime").timezone.utc).isoformat() == "2026-06-05T11:30:00+00:00"


def test_ignores_in_progress_charge(monkeypatch):
    con = _db([
        (1, "2026-06-05T08:00:00+00:00", "2026-06-05T11:30:00+00:00"),
        (2, "2026-06-06T08:00:00+00:00", None),   # still charging — must be skipped
    ])
    monkeypatch.setattr(db_reader, "_get", lambda: con)
    dt = db_reader.get_last_charge_end()
    assert dt.astimezone(__import__("datetime").timezone.utc).isoformat() == "2026-06-05T11:30:00+00:00"


def test_none_when_no_charge_ever_completed(monkeypatch):
    con = _db([(1, "2026-06-06T08:00:00+00:00", None)])
    monkeypatch.setattr(db_reader, "_get", lambda: con)
    assert db_reader.get_last_charge_end() is None

    empty = _db([])
    monkeypatch.setattr(db_reader, "_get", lambda: empty)
    assert db_reader.get_last_charge_end() is None
