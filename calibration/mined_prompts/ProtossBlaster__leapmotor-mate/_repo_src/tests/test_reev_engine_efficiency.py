"""REEV electric consumption — GitHub beta #10 (gm27271, C10 REEV).

A REEV trip where the range-extender RAN has no valid electric kWh/100 km: the generator recharges the
pack mid-drive, so the trip's NET SoC change is not the motor's traction energy. The old code stored the
plain BEV figure — SoC-delta (or getEC) over the FULL distance — which comes out diluted/near-zero
(gm27271 saw 0.5 kWh/100 km where the car reported ~19). Mate now WITHHOLDS it (NULL) at every writer, so
the misleading number never shows and no average is polluted. The trip still shows its fuel L/100 km.

Detection is purely a fuel-level drop past the 0.2 % noise floor → pure-electric REEV trips (fuel flat)
and BEVs (no fuel signal) are provably untouched.

tmp_path DB (poller schema + db_reader pointed at it). No ambient DB → CI-safe.
"""
import types

import db as D
import db_reader


def _vd(fuel=None, soc=80.0, odo=1000.0):
    return types.SimpleNamespace(soc=soc, odometer_km=odo, latitude=45.0, longitude=9.0,
                                 fuel_level_pct=fuel)


def _seed_db(tmp_path, monkeypatch):
    pdb = D.Database(str(tmp_path / "t.db"))
    pdb._conn.execute("INSERT INTO vehicles (id, vin) VALUES (1, 'VIN1')")
    pdb._conn.commit()
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    return pdb


def _ins(pdb, tid, dist, eff, fs=None, fe=None, ec=None, ec_stable=0):
    pdb._conn.execute(
        "INSERT INTO trips (id, vehicle_id, started_at, ended_at, distance_km, duration_min, "
        "efficiency_kwh_100km, fuel_start_pct, fuel_end_pct, ec_kwh, ec_stable) "
        "VALUES (?,1,'2026-07-07T20:00:00+00:00','2026-07-07T20:30:00+00:00',?,30,?,?,?,?,?)",
        (tid, dist, eff, fs, fe, ec, ec_stable))
    pdb._conn.commit()


# ── the predicate ─────────────────────────────────────────────────────────────
def test_extender_ran_predicate():
    assert D._reev_extender_ran(98.4, 96.2) is True     # 2.2 % drop → generator ran
    assert D._reev_extender_ran(80.0, 80.0) is False    # flat → pure electric
    assert D._reev_extender_ran(80.1, 80.0) is False    # 0.1 % = one signal tick, below floor
    assert D._reev_extender_ran(None, None) is False     # BEV — no fuel signal
    assert D._reev_extender_ran(96.2, None) is False     # partial read → don't suppress on a guess


# ── finalize_trip (new trips, poller write path) ──────────────────────────────
def test_finalize_withholds_efficiency_when_extender_ran(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    db.set_battery_capacity(60.0)
    tid = db.create_trip(1, _vd(fuel=98.4, soc=80.0, odo=1000.0))
    db.finalize_trip(tid, _vd(fuel=96.2, soc=75.0, odo=1017.6))   # fuel −2.2 % (engine), SoC −5 %, 17.6 km
    eff = db._conn.execute("SELECT efficiency_kwh_100km FROM trips WHERE id=?", (tid,)).fetchone()[0]
    assert eff is None                    # withheld (would otherwise be ~17 kWh/100 km from net SoC)


def test_finalize_keeps_efficiency_on_pure_electric_reev_trip(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    db.set_battery_capacity(60.0)
    tid = db.create_trip(1, _vd(fuel=98.4, soc=80.0, odo=1000.0))
    db.finalize_trip(tid, _vd(fuel=98.4, soc=75.0, odo=1017.6))   # fuel FLAT → pure electric, like a BEV
    eff = db._conn.execute("SELECT efficiency_kwh_100km FROM trips WHERE id=?", (tid,)).fetchone()[0]
    assert eff is not None and eff > 0


def test_finalize_bev_unaffected(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    db.set_battery_capacity(60.0)
    tid = db.create_trip(1, _vd(fuel=None, soc=80.0, odo=1000.0))   # BEV — no fuel signal at all
    db.finalize_trip(tid, _vd(fuel=None, soc=75.0, odo=1017.6))
    eff = db._conn.execute("SELECT efficiency_kwh_100km FROM trips WHERE id=?", (tid,)).fetchone()[0]
    assert eff is not None and eff > 0


# ── one-time migration (existing rows, incl. gm27271's history) ───────────────
def test_migration_nulls_engine_on_reev_trips(tmp_path, monkeypatch):
    pdb = _seed_db(tmp_path, monkeypatch)
    _ins(pdb, 1, 40.0, 0.5, fs=90.0, fe=88.0)      # REEV engine-on (fuel −2 %) → the diluted 0.5 → NULL it
    _ins(pdb, 2, 40.0, 16.0, fs=90.0, fe=90.0)     # REEV pure-electric (fuel flat) → keep
    _ins(pdb, 3, 40.0, 15.0)                        # BEV (fuel NULL) → keep
    pdb._conn.execute("DELETE FROM settings WHERE key='trips_reev_engine_eff_repair_v1'")
    pdb._conn.commit()
    pdb._repair_reev_engine_efficiency()
    effs = [r[0] for r in pdb._conn.execute("SELECT efficiency_kwh_100km FROM trips ORDER BY id")]
    assert effs == [None, 16.0, 15.0]


# ── getEC override must not resurrect the diluted figure ──────────────────────
def test_ec_override_skips_engine_on_reev(tmp_path, monkeypatch):
    pdb = _seed_db(tmp_path, monkeypatch)
    _ins(pdb, 1, 40.0, None, fs=90.0, fe=88.0, ec=0.2, ec_stable=1)   # engine-on: would be 0.2/40*100 = 0.5
    _ins(pdb, 2, 40.0, None, ec=8.0, ec_stable=1)                     # BEV: 8/40*100 = 20.0
    db_reader.apply_ec_trip_energy()
    effs = [r[0] for r in pdb._conn.execute("SELECT efficiency_kwh_100km FROM trips ORDER BY id")]
    assert effs[0] is None       # engine-on REEV NOT overridden — no diluted 0.5 comes back
    assert effs[1] == 20.0       # BEV overridden by getEC as before


# ── merged trip group (display recompute) ─────────────────────────────────────
def test_merged_group_blanks_efficiency_when_extender_ran():
    """A merged group spanning any generator-on segment → combined electric kWh/100 km is meaningless."""
    def _seg(sid, started, ended, ssoc, esoc, sodo, eodo, fs, fe):
        return {"id": sid, "started_at": started, "ended_at": ended, "start_soc": ssoc, "end_soc": esoc,
                "start_odometer_km": sodo, "end_odometer_km": eodo, "start_lat": 45.0, "start_lon": 9.0,
                "end_lat": 45.1, "end_lon": 9.1, "distance_km": (eodo - sodo), "duration_min": 15.0,
                "regen_kwh": 0.0, "fuel_start_pct": fs, "fuel_end_pct": fe, "ec_stable": 0, "ec_kwh": None}
    parent = _seg(1, "2026-07-07T20:00:00", "2026-07-07T20:15:00", 80.0, 78.0, 1000.0, 1010.0, 90.0, 89.0)
    child  = _seg(2, "2026-07-07T20:20:00", "2026-07-07T20:35:00", 78.0, 74.0, 1010.0, 1025.0, 89.0, 87.0)
    d = db_reader._trip_group_stats(parent, [child])
    assert d["distance_km"] == 25.0             # 1025 − 1000 (odometer span)
    assert d["efficiency_kwh_100km"] is None     # extender ran across the group → blanked (not a bogus number)
