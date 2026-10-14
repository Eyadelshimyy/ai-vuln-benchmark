"""Retroactive scan for charges missed while the car was asleep — GitHub #35.

Recovers home charges that happened before live reconstruction existed (or while the
poller was down): a SoC that rose while parked, not covered by any existing charge.
The scan must be safe (no regen-while-driving false positives, no overlap with real
charges) and idempotent (re-running creates no duplicates).
"""
import db as D                 # poller Database — builds the schema + seeds a vehicle
import db_reader               # web side — owns the scan


def _seed(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    db = D.Database(path)
    db._conn.execute("INSERT INTO vehicles (id, vin) VALUES (1, 'VIN1')")
    db.set_battery_capacity(67.1)        # pin the reference so the energy maths is explicit
    db._conn.commit()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    return db


def _pos(db, ts, soc, charging=0, speed=0, odo=1000.0):
    db._conn.execute(
        "INSERT INTO positions (vehicle_id, recorded_at, soc, charging, speed_kmh, odometer_km,"
        " latitude, longitude) VALUES (1,?,?,?,?,?,45.0,9.0)",
        (ts, soc, charging, speed, odo))
    db._conn.commit()


def test_detects_parked_soc_rise(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T22:00:00+00:00", 50.0)
    _pos(db, "2026-06-02T06:00:00+00:00", 80.0)        # +30% while parked, same odometer
    cands = db_reader.scan_missed_charges(apply=False)
    assert len(cands) == 1
    assert cands[0]["start_soc"] == 50.0 and cands[0]["end_soc"] == 80.0
    assert cands[0]["energy_kwh"] == round(30 / 100 * 67.1, 3)
    # Nothing written on a dry run.
    assert db._conn.execute("SELECT COUNT(*) FROM charges").fetchone()[0] == 0


def test_apply_inserts_reconstructed_charge(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T22:00:00+00:00", 50.0)
    _pos(db, "2026-06-02T06:00:00+00:00", 80.0)
    created = db_reader.scan_missed_charges(apply=True)
    assert len(created) == 1
    row = db._conn.execute("SELECT start_soc, end_soc, charge_type, reconstructed FROM charges").fetchone()
    assert row["start_soc"] == 50.0 and row["end_soc"] == 80.0
    assert row["charge_type"] == "AC" and row["reconstructed"] == 1


def test_ignores_rise_while_driving(tmp_path, monkeypatch):
    """Regen on a long descent raises SoC but the odometer moves → not a charge."""
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T10:00:00+00:00", 50.0, odo=1000.0)
    _pos(db, "2026-06-01T10:30:00+00:00", 56.0, speed=60, odo=1040.0)   # drove 40 km
    assert db_reader.scan_missed_charges(apply=False) == []


def test_ignores_rise_covered_by_existing_charge(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T22:00:00+00:00", 50.0)
    _pos(db, "2026-06-02T06:00:00+00:00", 80.0)
    db._conn.execute(
        "INSERT INTO charges (vehicle_id, started_at, ended_at, start_soc, end_soc) "
        "VALUES (1,'2026-06-01T21:00:00+00:00','2026-06-02T07:00:00+00:00',49,81)")
    db._conn.commit()
    assert db_reader.scan_missed_charges(apply=False) == []


def test_idempotent_no_duplicates_on_rerun(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T22:00:00+00:00", 50.0)
    _pos(db, "2026-06-02T06:00:00+00:00", 80.0)
    assert len(db_reader.scan_missed_charges(apply=True)) == 1
    assert db_reader.scan_missed_charges(apply=True) == []          # second run finds nothing new
    assert db._conn.execute("SELECT COUNT(*) FROM charges").fetchone()[0] == 1


def test_below_threshold_ignored(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T22:00:00+00:00", 50.0)
    _pos(db, "2026-06-02T06:00:00+00:00", 51.0)                     # +1% < 2% default
    assert db_reader.scan_missed_charges(apply=False) == []


def test_merges_consecutive_rises_into_one(tmp_path, monkeypatch):
    """One charge seen across several stale polls becomes ONE candidate, not three."""
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-01T22:00:00+00:00", 50.0)
    _pos(db, "2026-06-02T00:00:00+00:00", 60.0)
    _pos(db, "2026-06-02T03:00:00+00:00", 75.0)
    cands = db_reader.scan_missed_charges(apply=False)
    assert len(cands) == 1
    assert cands[0]["start_soc"] == 50.0 and cands[0]["end_soc"] == 75.0


def test_ignores_spurious_zero_soc_start(tmp_path, monkeypatch):
    """The Telegram ghost-charge bug: a poll that returned no SoC parsed as soc=0.0; the scan must
    NOT propose a 'charged from 0%' phantom for a 0→full jump between two real readings."""
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-02T08:32:37+00:00", 70.7)
    _pos(db, "2026-06-02T08:32:47+00:00", 0.0)      # spurious zero (real SoC ~70%, car had range)
    _pos(db, "2026-06-02T08:32:58+00:00", 70.7)
    assert db_reader.scan_missed_charges(apply=False) == []


def test_ignores_impossible_charge_rate(tmp_path, monkeypatch):
    """Even with a non-zero start, a huge SoC jump over seconds implies an impossible charge
    power (thousands of kW) → it's a sensor glitch, not a charge."""
    db = _seed(tmp_path, monkeypatch)
    _pos(db, "2026-06-02T08:00:00+00:00", 5.0)
    _pos(db, "2026-06-02T08:00:20+00:00", 90.0)     # +85% in 20s
    assert db_reader.scan_missed_charges(apply=False) == []
