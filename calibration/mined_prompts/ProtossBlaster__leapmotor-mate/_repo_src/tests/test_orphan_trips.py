"""close_orphan_trips (crash recovery) must filter (0,0)/null-island GPS fixes before summing the
distance. A single bad point slipping in before a crash used to add a virtual round-trip to the
equator and wreck the trip's distance (and skew the distance-weighted stats). CI-safe (memory DB)."""
import db as D


def _orphan(pdb, pts):
    pdb._conn.execute(
        "INSERT INTO trips (id, vehicle_id, started_at, start_soc, start_odometer_km) "
        "VALUES (1, 1, '2026-06-13T08:00:00+00:00', 80, 1000)")     # ended_at NULL → orphan
    for i, (lat, lon) in enumerate(pts):
        pdb._conn.execute(
            "INSERT INTO trip_positions (trip_id, recorded_at, latitude, longitude, soc) "
            "VALUES (1, ?, ?, ?, 79)", (f"2026-06-13T08:0{i}:00+00:00", lat, lon))
    pdb._conn.commit()


def test_null_island_point_does_not_blow_up_distance():
    pdb = D.Database(":memory:")
    # a short Milan hop with a (0,0) fix sandwiched in — raw haversine would be ~10000 km
    _orphan(pdb, [(45.46, 9.19), (0.0, 0.0), (45.47, 9.20)])
    assert pdb.close_orphan_trips(1) == 1
    d = pdb._conn.execute("SELECT distance_km FROM trips WHERE id=1").fetchone()["distance_km"]
    assert d is not None and d < 5        # filtered → a sane sub-5 km hop, not ~10000 km


def test_clean_track_still_measured():
    pdb = D.Database(":memory:")
    _orphan(pdb, [(45.460, 9.190), (45.470, 9.200)])
    pdb.close_orphan_trips(1)
    d = pdb._conn.execute("SELECT distance_km FROM trips WHERE id=1").fetchone()["distance_km"]
    assert 0.5 < d < 3                    # ~1.3 km, unaffected by the change
