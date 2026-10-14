"""Per-vehicle battery capacity (Tier-1 multi-car hardening).

Capacity is stored per-vehicle (vehicles.capacity_kwh) because energy is written as
ΔSoC × capacity, and a B10's 65 kWh vs a T03's 36 kWh differ ~80% — sharing one global
would corrupt the STORED trip/charge energy of the second car. These tests pin:

  * backfill is a byte-identical no-op on a single-car upgrade (and preserves a user override);
  * the write paths use each car's OWN capacity (isolation), proven end-to-end via a
    reconstructed charge (the one write path that stores kWh directly);
  * the getter is backward-compatible (no id → the legacy global);
  * ensure_vehicle seeds a new car (first → global, additional → model default);
  * the global setter keeps a single car in sync but never clobbers a second one;
  * a web capacity override is mirrored onto the car's row (or the poller would ignore it).
"""
import types

import db as D
import db_reader


def _fresh(tmp_path):
    return D.Database(str(tmp_path / "t.db"))


# ── backfill: no-op on upgrade, preserves override ────────────────────────────
def test_backfill_seeds_existing_vehicle_from_global(tmp_path):
    db = _fresh(tmp_path)
    # A vehicle that predates the per-vehicle column (capacity_kwh NULL) + a legacy global.
    db._conn.execute("INSERT INTO vehicles (id, vin, car_type, capacity_kwh) VALUES (1,'V1','B10',NULL)")
    db.set_setting("battery_capacity_kwh", "62.0")
    db._conn.commit()
    db._backfill_vehicle_capacity()
    assert db._conn.execute("SELECT capacity_kwh FROM vehicles WHERE id=1").fetchone()[0] == 62.0
    assert db.get_battery_capacity(1) == 62.0            # write paths now see the same number


def test_backfill_preserves_user_override_not_model_default(tmp_path):
    db = _fresh(tmp_path)
    db._conn.execute("INSERT INTO vehicles (id, vin, car_type, capacity_kwh) VALUES (1,'V1','B10',NULL)")
    db.set_setting("battery_capacity_kwh", "58.0")       # user overrode a B10 below its 65 default
    db._conn.commit()
    db._backfill_vehicle_capacity()
    assert db.get_battery_capacity(1) == 58.0            # override kept, NOT the 65 model default


# ── the core: per-vehicle isolation in a real write path ──────────────────────
def test_per_vehicle_isolation_reconstructed_charge(tmp_path):
    db = _fresh(tmp_path)
    v_b10 = db.ensure_vehicle("VINB10", "B10")           # first car, no global → default 65
    v_t03 = db.ensure_vehicle("VINT03", "T03")           # additional car → default 36
    assert db.get_battery_capacity(v_b10) == D.default_capacity_for("B10") == 65.0
    assert db.get_battery_capacity(v_t03) == D.default_capacity_for("T03") == 36.0

    data = types.SimpleNamespace(soc=60.0, latitude=45.0, longitude=9.0)
    cid_b10 = db.create_reconstructed_charge(v_b10, 20.0, "2026-06-09T09:00:00+00:00", data)
    cid_t03 = db.create_reconstructed_charge(v_t03, 20.0, "2026-06-09T09:00:00+00:00", data)
    e_b10 = db._conn.execute("SELECT energy_added_kwh FROM charges WHERE id=?", (cid_b10,)).fetchone()[0]
    e_t03 = db._conn.execute("SELECT energy_added_kwh FROM charges WHERE id=?", (cid_t03,)).fetchone()[0]
    # SAME ΔSoC (40%), DIFFERENT stored energy — each car used its OWN pack size.
    assert abs(e_b10 - 0.40 * 65.0) < 1e-6               # 26.0 kWh
    assert abs(e_t03 - 0.40 * 36.0) < 1e-6               # 14.4 kWh  (the +80% bug, now avoided)


# ── getter backward-compatibility ─────────────────────────────────────────────
def test_get_capacity_no_arg_returns_global(tmp_path):
    db = _fresh(tmp_path)
    db.set_setting("battery_capacity_kwh", "70.0")
    assert db.get_battery_capacity() == 70.0             # single-car callers unchanged


def test_get_capacity_unknown_vehicle_falls_back_to_global(tmp_path):
    db = _fresh(tmp_path)
    db.set_setting("battery_capacity_kwh", "70.0")
    assert db.get_battery_capacity(9999) == 70.0         # no such row → legacy fallback, never crash


# ── ensure_vehicle: first inherits global, additional gets model default ──────
def test_ensure_vehicle_first_inherits_global_additional_gets_default(tmp_path):
    db = _fresh(tmp_path)
    db.set_setting("battery_capacity_kwh", "58.0")       # setup-wizard / override value for the 1st car
    v1 = db.ensure_vehicle("VIN1", "B10")
    v2 = db.ensure_vehicle("VIN2", "T03")                # a second, shared car
    assert db.get_battery_capacity(v1) == 58.0           # first car = the global
    assert db.get_battery_capacity(v2) == 36.0           # second car = its OWN default, not 58


# ── global setter mirrors a single car but never a second ─────────────────────
def test_set_battery_capacity_mirrors_single_vehicle(tmp_path):
    db = _fresh(tmp_path)
    vid = db.ensure_vehicle("VIN1", "B10")
    db.set_battery_capacity(48.0)
    assert db.get_battery_capacity(vid) == 48.0          # single car kept in sync with the global


def test_set_battery_capacity_does_not_clobber_second_vehicle(tmp_path):
    db = _fresh(tmp_path)
    v1 = db.ensure_vehicle("VIN1", "B10")                # 65
    v2 = db.ensure_vehicle("VIN2", "T03")                # 36
    db.set_battery_capacity(99.0)                        # legacy global setter, now 2 cars present
    assert db.get_battery_capacity(v1) == 65.0           # neither car touched — guard is count==1
    assert db.get_battery_capacity(v2) == 36.0


# ── web override is mirrored onto the car's row ───────────────────────────────
def test_web_override_mirror_updates_vehicle(tmp_path, monkeypatch):
    db = _fresh(tmp_path)
    vid = db.ensure_vehicle("VIN1", "B10")               # starts at 65
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    db_reader.set_vehicle_capacity_current(60.0, nominal=67.1)
    row = db._conn.execute("SELECT capacity_kwh, capacity_nominal_kwh FROM vehicles WHERE id=?",
                           (vid,)).fetchone()
    assert row["capacity_kwh"] == 60.0                   # poller now reads the override
    assert row["capacity_nominal_kwh"] == 67.1
