"""Integration tests for the REEV fuel-purchase layer — add/list/delete + the snapshot of the tank %
before a refuel + fuel_blended_price_at end-to-end. Uses a throwaway tmp DB (monkeypatched DB_PATH),
never the ambient DB, so it can't be masked by a real database in the test container."""
import sqlite3
import db_reader


def _setup_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE vehicles (id INTEGER PRIMARY KEY, vin TEXT, car_type TEXT)")
    con.execute("INSERT INTO vehicles (id, vin, car_type) VALUES (1, 'VINX', 'C10 REEV')")
    con.execute("CREATE TABLE positions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "vehicle_id INTEGER, recorded_at TEXT, fuel_level_pct REAL)")
    # Fuel-level readings the WAC snapshots as the residual before each refuel.
    con.executemany(
        "INSERT INTO positions (vehicle_id, recorded_at, fuel_level_pct) VALUES (?,?,?)",
        [(1, "2026-07-10T08:00:00+00:00", 80.0),
         (1, "2026-07-12T08:00:00+00:00", 16.0)])   # 16 % = 8 L residual just before the 2nd refuel
    con.commit()
    con.close()


def test_crud_snapshot_and_blend(tmp_path, monkeypatch):
    dbp = str(tmp_path / "t.db")
    _setup_db(dbp)
    monkeypatch.setattr(db_reader, "DB_PATH", dbp)

    # Refuel 1: full tank, priced per-litre → total is derived.
    id1 = db_reader.add_fuel_purchase("2026-07-10T09:00:00+00:00", liters=50, price_per_l=1.70)
    # Refuel 2: 15 L priced by TOTAL → €/L is derived; residual snapshotted from positions (16 %).
    id2 = db_reader.add_fuel_purchase("2026-07-12T09:00:00+00:00", liters=15, total_cost=15 * 1.85)

    rows = db_reader.list_fuel_purchases()
    assert len(rows) == 2
    first = next(r for r in rows if r["id"] == id1)
    second = next(r for r in rows if r["id"] == id2)
    assert abs(first["total_cost"] - 85.0) < 1e-6          # 50 × 1.70 derived
    assert abs(second["price_per_l"] - 1.85) < 1e-6         # 27.75 / 15 derived
    assert abs(second["fuel_before_pct"] - 16.0) < 1e-6     # snapshot from positions

    # Blend after both refuels: 8 L @1.70 + 15 L @1.85 over 23 L.
    p = db_reader.fuel_blended_price_at(1, "2026-07-12T12:00:00+00:00")
    assert abs(p - (8 * 1.70 + 15 * 1.85) / 23) < 1e-3

    # Before the first refuel there's no price yet.
    assert db_reader.fuel_blended_price_at(1, "2026-07-01T00:00:00+00:00") is None

    # Delete removes it from the log.
    assert db_reader.delete_fuel_purchase(id2) is True
    assert len(db_reader.list_fuel_purchases()) == 1


def test_needs_a_price(tmp_path, monkeypatch):
    dbp = str(tmp_path / "t.db")
    _setup_db(dbp)
    monkeypatch.setattr(db_reader, "DB_PATH", dbp)
    for bad in ({}, {"liters": 0, "price_per_l": 1.7}):
        try:
            db_reader.add_fuel_purchase("2026-07-10T09:00:00+00:00", **bad) if bad else \
                db_reader.add_fuel_purchase("2026-07-10T09:00:00+00:00", liters=10)
            assert False, "expected ValueError"
        except (ValueError, TypeError):
            pass
