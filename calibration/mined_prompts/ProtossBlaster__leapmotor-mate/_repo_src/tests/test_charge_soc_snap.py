"""finalize_charge vs the BMS "snap-to-full" — the 107% efficiency artifact.

On charges that terminate at 100% the B10 BMS holds the displayed SoC (e.g. 99.1%)
while still drawing current, then snaps it to 100.0 in the very poll where charging
flips off — the last ~0.9% SoC appears with zero energy delivered, inflating the
ΔSoC×capacity energy estimate ~15% (real charge #41: stored 5.167 kWh vs 4.44 kWh
measured by ∫V·I dt → "Efficienza 107%"). finalize_charge must anchor the energy to
the last SoC seen while charging=1; end_soc keeps the final value for display.
"""
import types

import db as D


def _setup(tmp_path, start_soc, started_at="2026-06-01T08:00:00+00:00"):
    db = D.Database(str(tmp_path / "t.db"))
    db.set_battery_capacity(67.1)        # pin the reference so the maths is explicit
    db._conn.execute(
        "INSERT INTO charges (id,vehicle_id,started_at,start_soc) VALUES (1,1,?,?)",
        (started_at, start_soc))
    db._conn.commit()
    return db


def _pos(db, recorded_at, soc, charging):
    db._conn.execute(
        "INSERT INTO positions (vehicle_id,recorded_at,soc,charging) VALUES (1,?,?,?)",
        (recorded_at, soc, charging))
    db._conn.commit()


def _finalized(db):
    return db._conn.execute("SELECT * FROM charges WHERE id=1").fetchone()


def test_snap_to_full_uses_last_charging_soc(tmp_path):
    """The #41 trace: SoC parked at 99.1 for three charging polls, then 100.0 exactly
    when charging stops → energy from 99.1, NOT from the snapped 100.0."""
    db = _setup(tmp_path, start_soc=92.3)
    _pos(db, "2026-06-01T08:50:00+00:00", 99.1, 1)
    _pos(db, "2026-06-01T08:50:30+00:00", 99.1, 1)
    _pos(db, "2026-06-01T08:51:00+00:00", 99.1, 1)
    _pos(db, "2026-06-01T08:51:30+00:00", 100.0, 0)      # the snap, charging already off
    db.finalize_charge(1, types.SimpleNamespace(soc=100.0), max_power_kw=5.07)
    row = _finalized(db)
    assert row["energy_added_kwh"] == round((99.1 - 92.3) / 100 * 67.1, 3)   # 4.563
    assert row["end_soc"] == 100.0                       # display still shows "reached 100%"


def test_mid_soc_charge_is_a_noop(tmp_path):
    """No snap correction below 100%: the final tick (94.9→95.0 in the poll where
    charging stops) is REAL energy — caught by the replay simulation, where anchoring
    mid-SoC charges to the last charging sample shifted #39/#40 by one tick. The
    stored value must use the final SoC, byte-identical to the pre-fix formula."""
    db = _setup(tmp_path, start_soc=90.0)
    _pos(db, "2026-06-01T08:30:00+00:00", 94.9, 1)       # last charging sample LAGS
    _pos(db, "2026-06-01T08:31:00+00:00", 95.0, 0)
    db.finalize_charge(1, types.SimpleNamespace(soc=95.0))
    assert _finalized(db)["energy_added_kwh"] == round((95.0 - 90.0) / 100 * 67.1, 3)  # 3.355


def test_no_positions_falls_back_to_final_soc(tmp_path):
    """Without samples (poller restart, pruned table) behave exactly as before the fix."""
    db = _setup(tmp_path, start_soc=92.3)
    db.finalize_charge(1, types.SimpleNamespace(soc=100.0))
    assert _finalized(db)["energy_added_kwh"] == round((100.0 - 92.3) / 100 * 67.1, 3)  # 5.167


def test_samples_from_a_previous_charge_are_ignored(tmp_path):
    """charging=1 samples recorded BEFORE this charge started must not leak in."""
    db = _setup(tmp_path, start_soc=92.3, started_at="2026-06-01T08:00:00+00:00")
    _pos(db, "2026-06-01T07:00:00+00:00", 55.0, 1)       # earlier session
    db.finalize_charge(1, types.SimpleNamespace(soc=100.0))
    assert _finalized(db)["energy_added_kwh"] == round((100.0 - 92.3) / 100 * 67.1, 3)


def test_recompute_of_old_charge_ignores_later_charges(tmp_path):
    """Caught by the replay simulation: recomputing an OLD charge must not pick up
    charging samples from charges that happened AFTER it — the window needs the
    upper bound too."""
    db = _setup(tmp_path, start_soc=90.0, started_at="2026-05-31T08:00:00+00:00")
    _pos(db, "2026-05-31T08:30:00+00:00", 95.0, 1)       # this charge
    _pos(db, "2026-06-01T08:50:00+00:00", 99.1, 1)       # NEXT DAY's charge
    last = db._last_charging_soc(1, "2026-05-31T08:00:00+00:00", "2026-05-31T09:00:00+00:00")
    assert last == 95.0                                   # not tomorrow's 99.1


# ── one-time startup repair — production DBs finalized before the fix ───────────

def _closed_pre_fix_row(db, *, energy, cost, ac=None, loc="HOME", reconstructed=0):
    """Turn charge 1 into a pre-fix finalized 100%-ender with the given billing."""
    db._conn.execute(
        "UPDATE charges SET ended_at='2026-06-01T10:00:30+00:00', end_soc=100.0,"
        " energy_added_kwh=?, cost=?, ac_energy_kwh=?, location_type=?, reconstructed=?"
        " WHERE id=1", (energy, cost, ac, loc, reconstructed))
    db._conn.commit()


def _run_repair(db):
    db.set_setting("charges_soc_snap_repair_v1", "")     # simulate a pre-migration DB
    db._repair_snap_to_full_charges()
    return db._conn.execute("SELECT * FROM charges WHERE id=1").fetchone()


def test_repair_fixes_dc_billed_charge_and_rescales_cost(tmp_path):
    """The #38 case: no wallbox figure, cost billed on the inflated DC estimate →
    energy recomputed from the last charging SoC, cost rescaled at the SAME €/kWh."""
    db = _setup(tmp_path, start_soc=63.6)
    _pos(db, "2026-06-01T10:00:00+00:00", 99.1, 1)
    _pos(db, "2026-06-01T10:00:30+00:00", 100.0, 0)
    _closed_pre_fix_row(db, energy=24.424, cost=6.11)
    row = _run_repair(db)
    assert row["energy_added_kwh"] == round((99.1 - 63.6) / 100 * 67.1, 3)       # 23.82
    assert row["cost"] == round(6.11 / 24.424 * row["energy_added_kwh"], 2)      # 5.96
    assert row["end_soc"] == 100.0                       # display value untouched


def test_repair_keeps_wallbox_billed_cost(tmp_path):
    """The #41 case: HOME + wallbox AC figure → cost stays (it's billed on measured
    AC), only the DC estimate is corrected."""
    db = _setup(tmp_path, start_soc=92.3)
    _pos(db, "2026-06-01T10:00:00+00:00", 99.1, 1)
    _closed_pre_fix_row(db, energy=5.167, cost=1.21, ac=4.84)
    row = _run_repair(db)
    assert row["energy_added_kwh"] == round((99.1 - 92.3) / 100 * 67.1, 3)       # 4.563
    assert row["cost"] == 1.21


def test_repair_is_one_shot(tmp_path):
    """Once the flag is set the repair never reruns — a later manual edit survives."""
    db = _setup(tmp_path, start_soc=92.3)
    _pos(db, "2026-06-01T10:00:00+00:00", 99.1, 1)
    _closed_pre_fix_row(db, energy=5.167, cost=1.21, ac=4.84)
    _run_repair(db)
    db._conn.execute("UPDATE charges SET energy_added_kwh=9.999 WHERE id=1")
    db._conn.commit()
    db._repair_snap_to_full_charges()                    # flag already "1" → no-op
    assert db._conn.execute("SELECT energy_added_kwh FROM charges WHERE id=1"
                            ).fetchone()[0] == 9.999


def test_repair_skips_charge_without_samples(tmp_path):
    """No surviving positions (pruned/asleep) → nothing better available, keep as-is."""
    db = _setup(tmp_path, start_soc=92.3)
    _closed_pre_fix_row(db, energy=5.167, cost=1.21, ac=4.84)
    row = _run_repair(db)
    assert row["energy_added_kwh"] == 5.167


def test_repair_skips_reconstructed_and_mid_soc(tmp_path):
    """Reconstructed charges have no live samples by definition; mid-SoC charges
    (end < 100) never suffer the snap — both stay untouched."""
    db = _setup(tmp_path, start_soc=63.6)
    _pos(db, "2026-06-01T10:00:00+00:00", 99.1, 1)
    _closed_pre_fix_row(db, energy=24.424, cost=6.11, reconstructed=1)
    assert _run_repair(db)["energy_added_kwh"] == 24.424
    db._conn.execute("UPDATE charges SET reconstructed=0, end_soc=95.0")
    db._conn.commit()
    assert _run_repair(db)["energy_added_kwh"] == 24.424
