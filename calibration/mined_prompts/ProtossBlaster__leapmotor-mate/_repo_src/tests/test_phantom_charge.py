"""Phantom-charge guard: a brief plug / charge-state blip (e.g. the car re-evaluating after a
charge schedule change) can open+close a charge that delivered nothing. finalize_charge drops a
session that gained no SoC AND has no wallbox-measured energy, instead of leaving a fake row.
Pure poller.db → CI-safe."""
import types

import db as D


def _db(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    db.set_battery_capacity(67.1)
    return db


def _open(db, cid, start_soc, ac=None):
    db._conn.execute(
        "INSERT INTO charges (id,vehicle_id,started_at,start_soc,ac_energy_kwh) VALUES (?,1,?,?,?)",
        (cid, "2026-06-01T08:00:00+00:00", start_soc, ac))
    db._conn.commit()


def _exists(db, cid):
    return db._conn.execute("SELECT 1 FROM charges WHERE id=?", (cid,)).fetchone() is not None


def test_phantom_no_soc_no_wallbox_is_dropped(tmp_path):
    db = _db(tmp_path)
    _open(db, 1, start_soc=100.0)                                   # blip at a full battery
    db.finalize_charge(1, types.SimpleNamespace(soc=100.0))         # no SoC gained
    assert not _exists(db, 1)                                        # phantom → dropped


def test_real_charge_is_kept(tmp_path):
    db = _db(tmp_path)
    _open(db, 2, start_soc=40.0)
    db.finalize_charge(2, types.SimpleNamespace(soc=72.0))          # +32% → real energy
    assert _exists(db, 2)


def test_zero_soc_but_wallbox_energy_is_kept(tmp_path):
    """A short top-up the SoC resolution missed but the wallbox actually measured → keep it."""
    db = _db(tmp_path)
    _open(db, 3, start_soc=80.0, ac=0.4)                            # wallbox saw 0.4 kWh
    db.finalize_charge(3, types.SimpleNamespace(soc=80.0))
    assert _exists(db, 3)


def _closed(db, cid, s, e, energy, ac=None, rec=0):
    db._conn.execute(
        "INSERT INTO charges (id,vehicle_id,started_at,ended_at,start_soc,end_soc,"
        "energy_added_kwh,ac_energy_kwh,reconstructed) "
        "VALUES (?,1,'2026-06-01T08:00:00+00:00','2026-06-01T08:30:00+00:00',?,?,?,?,?)",
        (cid, s, e, energy, ac, rec))
    db._conn.commit()


def test_one_time_cleanup_drops_only_phantoms(tmp_path):
    db = _db(tmp_path)                          # __init__ ran the cleanup on the empty DB
    _closed(db, 10, 100.0, 100.0, 0.0, ac=0.02)   # phantom (no SoC, ~0 wallbox) → drop
    _closed(db, 11, 40.0, 72.0, 21.5)             # real (SoC gained)          → keep
    _closed(db, 12, 80.0, 80.0, 0.0, ac=0.4)      # wallbox measured energy     → keep
    _closed(db, 13, 90.0, 90.0, 0.0, rec=1)       # reconstructed               → keep
    db.set_setting("charges_phantom_cleanup_v1", "")   # simulate a pre-cleanup DB
    db._drop_phantom_charges()
    surviving = {r[0] for r in db._conn.execute("SELECT id FROM charges").fetchall()}
    assert surviving == {11, 12, 13}              # ONLY the phantom #10 is gone

    # One-shot: the flag is set now, so a later empty charge is never touched by a re-run.
    _closed(db, 14, 50.0, 50.0, 0.0)
    db._drop_phantom_charges()
    assert _exists(db, 14)
