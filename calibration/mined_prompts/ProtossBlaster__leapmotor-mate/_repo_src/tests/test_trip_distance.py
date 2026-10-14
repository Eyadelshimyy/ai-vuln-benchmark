"""Trip distance guards: the odometer-zero bug (a few-metre move logged as the car's
entire mileage, e.g. a 3-min hop showing 6441 km) must not happen on new trips and
must be repaired on existing data."""
import types

import db as D


TINY = [(45.00000, 9.00000), (45.00002, 9.00001), (45.00003, 9.00000)]   # ~few metres
REAL = [(45.0000, 9.0000), (45.0090, 9.0000), (45.0180, 9.0000)]         # ~2 km
WITHZERO = [(45.0000, 9.0000), (0.0, 0.0), (45.0090, 9.0000)]            # (0,0) spurious mid-track


def _add_track(db, tid, pts):
    for lat, lon in pts:
        db._conn.execute(
            "INSERT INTO trip_positions (trip_id,recorded_at,latitude,longitude,speed_kmh,soc)"
            " VALUES (?,?,?,?,?,?)",
            (tid, "2026-06-07T11:51:00+00:00", lat, lon, 5, 80),
        )
    db._conn.commit()


def _open_trip(db, start_odo):
    cur = db._conn.execute(
        "INSERT INTO trips (vehicle_id,started_at,start_lat,start_lon,start_soc,start_odometer_km)"
        " VALUES (1,'2026-06-07T11:50:00+00:00',45,9,80,?)",
        (start_odo,),
    )
    db._conn.commit()
    return cur.lastrowid


def _bad_trip(db, end_odo, dist, pts):
    """A trip as the OLD buggy code stored it: start odo 0, distance == end odometer."""
    cur = db._conn.execute(
        "INSERT INTO trips (vehicle_id,started_at,ended_at,start_soc,end_soc,"
        "start_odometer_km,end_odometer_km,distance_km,efficiency_kwh_100km)"
        " VALUES (1,'2026-06-07T11:50:00+00:00','2026-06-07T11:53:00+00:00',80,79,0,?,?,0.2)",
        (end_odo, dist),
    )
    db._conn.commit()
    tid = cur.lastrowid
    _add_track(db, tid, pts)
    return tid


def test_gps_track_skips_spurious_coords():
    # (0,0) "null island" must not add a transcontinental jump (~10000 km here).
    rows = [{"latitude": la, "longitude": lo} for la, lo in WITHZERO]
    assert D._gps_track_km(rows) < 2.0


def test_finalize_falls_back_to_gps_when_start_odometer_missing(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    tid = _open_trip(db, start_odo=0)            # odometer signal missing at trip start
    _add_track(db, tid, TINY)
    data = types.SimpleNamespace(odometer_km=6441.0, soc=79.0, latitude=45.00003, longitude=9.0)
    assert db.finalize_trip(tid, data) < 1.0     # GPS track, NOT 6441


def test_finalize_uses_odometer_delta_when_valid(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    tid = _open_trip(db, start_odo=1000)
    _add_track(db, tid, REAL)
    data = types.SimpleNamespace(odometer_km=1025.0, soc=75.0, latitude=45.018, longitude=9.0)
    assert abs(db.finalize_trip(tid, data) - 25.0) < 0.01


def test_repair_migration(tmp_path):
    path = str(tmp_path / "t.db")
    db = D.Database(path)
    legit = _bad_trip(db, 6440, 42.0, REAL)      # NOT the signature (distance != end odo)
    db._conn.execute("UPDATE trips SET start_odometer_km=1000, end_odometer_km=1042 WHERE id=?", (legit,))
    db._conn.commit()
    b_del = _bad_trip(db, 6441, 6441.0, TINY)    # few metres -> delete
    b_keep = _bad_trip(db, 6443, 6443.0, REAL)   # real track -> recompute
    b_zero = _bad_trip(db, 6445, 6445.0, WITHZERO)
    # simulate the upgrade: bad trips present, repair flag not yet set
    db._conn.execute("DELETE FROM settings WHERE key='trips_odo_repair_v1'")
    db._conn.commit()
    db._conn.close()

    db2 = D.Database(path)                        # reopen -> runs _repair_odometer_trips
    q = lambda tid: db2._conn.execute("SELECT distance_km FROM trips WHERE id=?", (tid,)).fetchone()
    assert q(b_del) is None                       # tiny hop deleted
    assert 1.5 < q(b_keep)["distance_km"] < 2.5   # recomputed ~2 km, not 6443
    assert q(b_zero)["distance_km"] < 1.5         # (0,0) ignored, no jump
    assert q(legit)["distance_km"] == 42.0        # legit trip untouched
    assert db2.get_setting("trips_odo_repair_v1") == "1"


def test_finalize_keeps_trip_when_distance_unmeasurable(tmp_path):
    # No odometer (start 0) AND no GPS track → distance UNKNOWN → keep the trip (don't drop it as a
    # <0.5 km hop). Regression: 1.11.10's GPS fallback returned 0 for no-GPS cars → trips vanished.
    db = D.Database(str(tmp_path / "t.db"))
    tid = _open_trip(db, start_odo=0)            # odometer missing
    # (no _add_track → car reported no GPS → trip_positions empty)
    data = types.SimpleNamespace(odometer_km=0.0, soc=79.0, latitude=0.0, longitude=0.0)
    assert db.finalize_trip(tid, data) is None   # None ⇒ recorder KEEPS it (0.0 would be deleted)
    row = db._conn.execute("SELECT distance_km, ended_at FROM trips WHERE id=?", (tid,)).fetchone()
    assert row["distance_km"] is None            # stored NULL, not 0
    assert row["ended_at"] is not None           # finalized & preserved (time/SOC kept)


def test_repair_migration_keeps_no_gps_trips(tmp_path):
    # A bug-signature trip with NO GPS must be KEPT (distance cleared to NULL), not deleted.
    path = str(tmp_path / "t.db")
    db = D.Database(path)
    b_nogps = _bad_trip(db, 6449, 6449.0, [])    # bug signature, NO GPS positions
    db._conn.execute("DELETE FROM settings WHERE key='trips_odo_repair_v1'")
    db._conn.commit()
    db._conn.close()
    db2 = D.Database(path)                        # reopen → runs the repair
    row = db2._conn.execute("SELECT distance_km FROM trips WHERE id=?", (b_nogps,)).fetchone()
    assert row is not None                        # NOT deleted (no GPS ⇒ unmeasurable → kept)
    assert row["distance_km"] is None             # bogus 6449 km cleared to NULL
