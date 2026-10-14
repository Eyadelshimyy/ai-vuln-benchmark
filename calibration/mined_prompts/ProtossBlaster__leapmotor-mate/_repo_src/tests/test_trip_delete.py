"""Deleting a trip removes the row AND its GPS track, and is a safe no-op for an
unknown id (the user-facing 'Delete trip' button, confirmed in the UI)."""
import sqlite3

import db as POLLER_DB          # poller/db.py — builds the schema
import db_reader                # web/db_reader.py — the delete path used by the endpoint


def _make_trip(pdb):
    cur = pdb._conn.execute(
        "INSERT INTO trips (vehicle_id,started_at,ended_at,distance_km)"
        " VALUES (1,'2026-06-07T10:00:00+00:00','2026-06-07T10:10:00+00:00',5.0)"
    )
    pdb._conn.commit()
    tid = cur.lastrowid
    pdb._conn.execute(
        "INSERT INTO trip_positions (trip_id,recorded_at,latitude,longitude,speed_kmh,soc)"
        " VALUES (?,?,?,?,?,?)",
        (tid, "2026-06-07T10:01:00+00:00", 45.0, 9.0, 5, 80),
    )
    pdb._conn.commit()
    return tid


def test_delete_trip_removes_row_and_track(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    pdb = POLLER_DB.Database(path)
    tid = _make_trip(pdb)
    pdb._conn.close()

    monkeypatch.setattr(db_reader, "DB_PATH", path)
    assert db_reader.delete_trip(tid) is True
    assert db_reader.delete_trip(tid) is False          # already gone -> safe no-op

    c = sqlite3.connect(path)
    assert c.execute("SELECT COUNT(*) FROM trips WHERE id=?", (tid,)).fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM trip_positions WHERE trip_id=?", (tid,)).fetchone()[0] == 0
    c.close()


def test_delete_unknown_trip_is_noop(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    POLLER_DB.Database(path)._conn.close()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    assert db_reader.delete_trip(999999) is False
