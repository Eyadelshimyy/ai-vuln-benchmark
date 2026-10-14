"""User-overridable battery capacity — GitHub #35 point 1.

The override changes the energy-calc capacity, but the SoH page must keep measuring
against the as-new spec (battery_capacity_nominal_kwh), otherwise adopting a measured
(already-aged) value would reset SoH to ~100% and hide the ageing. This test pins that
the health denominator prefers the nominal snapshot over the overridden capacity.
"""
import db as D
import db_reader


def _seed(tmp_path, monkeypatch, **settings):
    path = str(tmp_path / "t.db")
    db = D.Database(path)
    db._conn.execute("INSERT INTO vehicles (id, vin) VALUES (1, 'VIN1')")
    for k, v in settings.items():
        db._conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (k, str(v)))
    db._conn.commit()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    return db


def test_soh_denominator_uses_nominal_when_set(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch,
          battery_capacity_kwh="60.0",             # user overrode to a lower value
          battery_capacity_nominal_kwh="67.1")     # original spec snapshot
    assert db_reader.get_battery_health()["nominal_kwh"] == 67.1


def test_soh_denominator_falls_back_to_capacity_when_no_nominal(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, battery_capacity_kwh="60.0")
    assert db_reader.get_battery_health()["nominal_kwh"] == 60.0


def test_energy_calc_capacity_is_the_overridable_key(tmp_path, monkeypatch):
    """finalize uses get_battery_capacity_kwh (the overridable key), independent of the
    SoH nominal — so an override affects new energy figures, not the health reference."""
    _seed(tmp_path, monkeypatch, battery_capacity_kwh="60.0", battery_capacity_nominal_kwh="67.1")
    assert db_reader.get_battery_capacity_kwh() == 60.0
