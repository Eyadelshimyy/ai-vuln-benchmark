"""Trip distance vs the whole-km odometer — the "24 m manoeuvre logged as 1.0 km" bug.

The odometer reads in integer km, so a tiny driveway shuffle that crosses a km
boundary shows Δodo = 1 (real trip #78: GPS track 24 m, odometer 3406→3407 → stored
1.0 km). trip_distance_km() now cross-checks the ambiguous Δodo == 1 case against the
GPS track and lets the recorder drop sub-0.5 km hops.
"""
from db import trip_distance_km


def test_manoeuvre_crossing_km_boundary_uses_gps():
    # The real #78 numbers: 24 m of GPS over a 3406→3407 odometer tick.
    assert trip_distance_km(0.024, True, 3406, 3407) == 0.024   # → recorder drops it


def test_real_1km_trip_keeps_odometer():
    # A genuine ~1 km drive: GPS reads 0.8-0.9 (corner-cutting) → odometer wins.
    assert trip_distance_km(0.85, True, 3406, 3407) == 1


def test_longer_trips_always_use_odometer():
    assert trip_distance_km(1.4, True, 3400, 3402) == 2     # Δ=2 → no ambiguity
    assert trip_distance_km(120.0, True, 3407, 3541) == 134  # the real #79


def test_delta_zero_falls_back_to_gps():
    assert trip_distance_km(0.3, True, 3406, 3406) == 0.3


def test_bogus_zero_odometer_falls_back_to_gps():
    # Missing signal 1318 → start odo 0: never use the delta (it'd be the car's mileage).
    assert trip_distance_km(2.1, True, 0, 6441) == 2.1


def test_delta_one_without_gps_keeps_odometer():
    assert trip_distance_km(0.0, False, 3406, 3407) == 1


def test_nothing_valid_returns_none_preserving_the_trip():
    assert trip_distance_km(0.0, False, 0, 0) is None


# ── one-shot startup repair of historical quantized manoeuvres ───────────────────

import db as D


def _trip(db, tid, start_odo, end_odo, dist, pts):
    db._conn.execute(
        "INSERT INTO trips (id,vehicle_id,started_at,ended_at,start_odometer_km,"
        "end_odometer_km,distance_km,efficiency_kwh_100km) VALUES (?,1,'2026-06-01T08:00:00+00:00',"
        "'2026-06-01T08:02:00+00:00',?,?,?,15.0)", (tid, start_odo, end_odo, dist))
    for lat, lon in pts:
        db._conn.execute("INSERT INTO trip_positions (trip_id,recorded_at,latitude,longitude) "
                         "VALUES (?,'2026-06-01T08:01:00+00:00',?,?)", (tid, lat, lon))
    db._conn.commit()


def test_repair_fixes_quantized_manoeuvre_keeps_real_1km(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    # 24 m manoeuvre over a km tick (the real #78 shape) → corrected to GPS
    _trip(db, 1, 3406, 3407, 1.0, [(45.4434, 9.1249), (45.4436, 9.1250)])
    # genuine ~1 km drive (GPS ≈ 0.9 km) → untouched
    _trip(db, 2, 3406, 3407, 1.0, [(45.4434, 9.1249), (45.4515, 9.1249)])
    db.set_setting("trips_odo_quantize_repair_v1", "")
    db._repair_quantized_trip_distance()
    r1 = db._conn.execute("SELECT distance_km, efficiency_kwh_100km FROM trips WHERE id=1").fetchone()
    r2 = db._conn.execute("SELECT distance_km, efficiency_kwh_100km FROM trips WHERE id=2").fetchone()
    assert r1["distance_km"] < 0.1 and r1["efficiency_kwh_100km"] is None   # corrected
    assert r2["distance_km"] == 1.0 and r2["efficiency_kwh_100km"] == 15.0  # untouched
    # one-shot: re-running changes nothing even if values are edited back
    db._conn.execute("UPDATE trips SET distance_km=1.0 WHERE id=1"); db._conn.commit()
    db._repair_quantized_trip_distance()
    assert db._conn.execute("SELECT distance_km FROM trips WHERE id=1").fetchone()[0] == 1.0
