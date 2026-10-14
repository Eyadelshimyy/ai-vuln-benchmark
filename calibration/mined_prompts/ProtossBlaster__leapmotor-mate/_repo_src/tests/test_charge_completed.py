"""Signal 3736 (chargeCompleted) → a persisted `charge_completed` column, read back as
`charge_completed_raw` for the diagnostics bundle.

The poller parses 3736 into VehicleData.charge_completed and writes it to the positions row; the web
reads it back via get_latest_status (SELECT *). It is not what the pages call complete — no car sets
it at a full charge (test_a_charge_is_complete_at_the_cars_limit.py) — so this is the PLUMBING only,
with synthetic values. CI-safe (tmp DB, no network)."""
import sqlite3

import client          # poller/client.py
import db as D         # poller schema + migrations
import db_reader


def _sig(**kw):
    base = {"1010": 0, "1319": 0}   # parked, stationary (so unrelated gates don't fire)
    base.update(kw)
    return base


def test_parse_signal_maps_3736():
    # Truthy mapping: the B10 status flags read 2 (not 1) — see vehicleSecurityActive.
    assert client._parse_signal("VIN", _sig(**{"3736": 2})).charge_completed is True
    assert client._parse_signal("VIN", _sig(**{"3736": 1})).charge_completed is True
    assert client._parse_signal("VIN", _sig(**{"3736": 0})).charge_completed is False
    assert client._parse_signal("VIN", _sig()).charge_completed is False        # absent → False


def test_persists_and_reads_back(tmp_path, monkeypatch):
    D.Database(str(tmp_path / "t.db"))
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    db_reader.upsert_vehicle("VIN", "B10")

    db_reader.save_fresh_signals({"1204": 50, "3736": 1})
    assert db_reader.get_latest_status()["charge_completed_raw"] == 1

    db_reader.save_fresh_signals({"1204": 50, "3736": 0})
    assert db_reader.get_latest_status()["charge_completed_raw"] == 0

    db_reader.save_fresh_signals({"1204": 50})             # signal absent → 0, never NULL/crash
    assert db_reader.get_latest_status()["charge_completed_raw"] == 0


def test_schema_has_column_and_migration_is_idempotent(tmp_path):
    p = str(tmp_path / "m.db")
    D.Database(p)
    cols = {r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(positions)")}
    assert "charge_completed" in cols
    D.Database(p)        # re-init must not raise (the `if not in cols` migration guard)
