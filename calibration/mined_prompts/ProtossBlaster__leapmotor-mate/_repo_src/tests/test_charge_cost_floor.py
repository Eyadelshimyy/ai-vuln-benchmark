"""HOME charges are billed on the wallbox energy the poller MEASURED — the running sum of the wallbox
kWh counter's POSITIVE rises over the charge (charges.ac_energy_kwh), accumulated each poll. It is
reset/race-proof: a counter that zeroes mid-session (the drop is ignored) still totals correctly,
whether it's a lifetime counter or resets each session, and no matter WHEN it resets relative to our
polls. No wallbox measurement → battery (DC/SoC) energy. Pure db_reader / poller.db → runs in CI."""
import sqlite3
import types

import db_reader


def _patch_flat(monkeypatch, home_price=0.20):
    monkeypatch.setattr(db_reader, "get_cost_config",
                        lambda: {"mode": "flat", "method": "split", "bands": []})
    monkeypatch.setattr(db_reader, "get_charge_prices", lambda: {"price_home_kwh": home_price})


def test_home_billed_on_wallbox_meter_else_battery(monkeypatch):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE charges (id INT, location_type TEXT, energy_added_kwh REAL, "
                "cost REAL, ac_energy_kwh REAL, started_at TEXT, ended_at TEXT, "
                "is_free INT DEFAULT 0)")   # #120: update_charge_type now also writes is_free
    con.execute("INSERT INTO charges (id, energy_added_kwh, ac_energy_kwh) VALUES (1, 10.0, 11.5)")
    con.commit()
    monkeypatch.setattr(db_reader, "_conn_rw", lambda: con)
    _patch_flat(monkeypatch, 0.20)

    # HOME → billed on the MEASURED wallbox energy (11.5 kWh), cost 11.5 × 0.20
    r = db_reader.update_charge_type(1, "HOME")
    assert round(r["cost"], 2) == 2.30
    # no wallbox measurement → billed on the battery (DC/SoC) energy (10 kWh)
    con.execute("UPDATE charges SET ac_energy_kwh=NULL WHERE id=1")
    con.commit()
    r = db_reader.update_charge_type(1, "HOME")
    assert round(r["cost"], 2) == 2.00


def _charge_total(database, start, readings, hours_ago=5.0):
    from datetime import datetime, timedelta, timezone
    cid = database.create_charge(1, types.SimpleNamespace(soc=20, latitude=1.0, longitude=2.0))
    # Backdate the start so the synthetic counter rises are physically plausible: the per-poll guard
    # caps a single step at the wallbox's power × elapsed time (see _wb_energy_ceiling, GitHub #46).
    past = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    database._conn.execute("UPDATE charges SET started_at=? WHERE id=?", (past, cid))
    database.set_charge_wallbox_start(cid, start)
    for v in readings:
        database.accumulate_wallbox_energy(cid, v)
    return database._conn.execute(
        "SELECT ac_energy_kwh FROM charges WHERE id=?", (cid,)).fetchone()["ac_energy_kwh"]


def test_wallbox_energy_sums_rises_for_every_counter_type(tmp_path):
    import db as D
    database = D.Database(str(tmp_path / "t.db"))

    # cumulative counter (like an odometer): 500 → 510 → 535.5  ⇒ 35.5
    assert _charge_total(database, 500.0, [510.0, 535.5]) == 35.5
    # session counter that reset BEFORE the start read (start ≈ 0): 0.1 → 12 → 35.5  ⇒ 35.4
    assert _charge_total(database, 0.1, [12.0, 35.5]) == 35.4
    # THE RACE — start read the STALE old value, counter then zeroed mid-charge, new session rose:
    # 9.67 (stale, no rise) · 0.1 (reset, drop ignored) · 12 · 35.5  ⇒ 35.4 (the stale 9.67 counts 0)
    assert _charge_total(database, 9.67, [9.67, 0.1, 12.0, 35.5]) == 35.4
