"""Advanced tunables — GitHub #35 point 2.

Covers the two pieces with real logic behind them: the reconstruction-floor clamp
(a hand-edited DB must not be able to drop it below 1% and invent phantom charges)
and the configurable AC/DC power threshold (a 22 kW AC wallbox owner can stop their
sessions being labelled DC).
"""
import types

import db as D
import recorder as R


# ── reconstruction floor clamp ─────────────────────────────────────────────────

def test_reconstruct_floor_clamped_to_1pct():
    rec = R.Recorder(D.Database(":memory:"), vehicle_id=1)
    rec.set_reconstruct_min_pct(0.1)        # hand-edited tiny value
    assert rec._reconstruct_min_pct == 1.0  # floored
    rec.set_reconstruct_min_pct(3.5)        # a sane value passes through
    assert rec._reconstruct_min_pct == 3.5
    rec.set_reconstruct_min_pct(0)          # 0/None ignored, keeps previous
    assert rec._reconstruct_min_pct == 3.5


# ── configurable AC/DC threshold in finalize_charge ────────────────────────────

def _charge(tmp_path, max_kw, dc_min_kw=None):
    db = D.Database(str(tmp_path / "t.db"))
    if dc_min_kw is not None:
        db.set_setting("charge_dc_min_kw", str(dc_min_kw))
    db._conn.execute(
        "INSERT INTO charges (id,vehicle_id,started_at,start_soc) "
        "VALUES (1,1,'2026-06-01T08:00:00+00:00',50)")
    db._conn.commit()
    db.finalize_charge(1, types.SimpleNamespace(soc=60.0), max_power_kw=max_kw)
    return db._conn.execute("SELECT charge_type FROM charges WHERE id=1").fetchone()["charge_type"]


def test_default_threshold_22kw_ac_is_misread_as_dc(tmp_path):
    # The historical default (11): a 22 kW AC session looks like DC.
    assert _charge(tmp_path, max_kw=22.0) == "DC"


def test_raised_threshold_keeps_22kw_ac_as_ac(tmp_path):
    # Raising it to 24 (the 22 kW-wallbox fix) classifies it correctly.
    assert _charge(tmp_path, max_kw=22.0, dc_min_kw=24) == "AC"


def test_real_dc_still_dc_above_raised_threshold(tmp_path):
    assert _charge(tmp_path, max_kw=50.0, dc_min_kw=24) == "DC"


def test_bad_setting_falls_back_to_default(tmp_path):
    assert _charge(tmp_path, max_kw=12.0, dc_min_kw="garbage") == "DC"   # >11 default → DC
