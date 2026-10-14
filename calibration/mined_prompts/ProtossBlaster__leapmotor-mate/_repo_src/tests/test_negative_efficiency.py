"""Regression for #58: a trip with a NEGATIVE efficiency (SoC ROSE over the trip — e.g. a trip
window mis-bounded across a charge during an offline/session gap) must never show up as the
Statistics 'best efficiency', and a one-time repair nulls such stored values.

Runs on a tmp_path DB (poller schema + db_reader pointed at it) — CI-safe."""
import db as D
import db_reader


def _seed(tmp_path, monkeypatch):
    pdb = D.Database(str(tmp_path / "t.db"))
    pdb._conn.execute("INSERT INTO vehicles (id, vin) VALUES (1, 'VIN1')")
    pdb._conn.commit()
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    return pdb


def _trip(pdb, tid, dist, eff):
    pdb._conn.execute(
        "INSERT INTO trips (id, vehicle_id, started_at, ended_at, distance_km, duration_min,"
        " efficiency_kwh_100km) VALUES (?,1,'2026-06-01T08:00:00+00:00','2026-06-01T08:30:00+00:00',?,30,?)",
        (tid, dist, eff))
    pdb._conn.commit()


def test_best_efficiency_excludes_negative(tmp_path, monkeypatch):
    """The Statistics 'best efficiency' (a MIN) must skip a negative value, not show e.g. -39.3."""
    pdb = _seed(tmp_path, monkeypatch)
    _trip(pdb, 1, 50.0, 16.0)
    _trip(pdb, 2, 40.0, 19.0)
    _trip(pdb, 3, 5.0, -39.3)          # corrupted: SoC rose over the trip
    s = db_reader.get_stats_summary()
    assert s["best_efficiency"] == 16.0   # lowest POSITIVE — never the -39.3


def test_repair_nulls_negative_efficiency(tmp_path, monkeypatch):
    """One-time migration nulls stored negative efficiencies (fixes already-recorded glitchy trips)."""
    pdb = _seed(tmp_path, monkeypatch)
    _trip(pdb, 1, 50.0, 16.0)
    _trip(pdb, 2, 5.0, -39.3)
    # __init__ already ran the repair on the (then empty) DB → clear the gate to run it on our data
    pdb._conn.execute("DELETE FROM settings WHERE key='trips_neg_efficiency_repair_v1'")
    pdb._conn.commit()
    pdb._repair_negative_efficiency()
    effs = [r[0] for r in pdb._conn.execute(
        "SELECT efficiency_kwh_100km FROM trips ORDER BY id")]
    assert effs == [16.0, None]           # positive kept, negative nulled
