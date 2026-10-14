"""Auto-assign HOME (idea: @hubcasale, PR #47): closed untyped charges with wallbox-measured
AC energy are confirmed as HOME through update_charge_type — the SAME path as a manual badge
confirm — so the cost honours the pricing config (flat AND time-of-use bands) and is billed
on the wallbox AC energy. Opt-in via the wallbox_auto_home setting, off by default.

Everything runs on a tmp_path DB (poller schema + db_reader pointed at it) — no settings DB,
CI-safe."""
import json

import db as D            # poller schema (creates charges/settings tables + migrations)
import db_reader


def _setup(tmp_path, monkeypatch):
    pdb = D.Database(str(tmp_path / "t.db"))
    pdb.set_battery_capacity(67.1)
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    return pdb


def _charge(pdb, cid, *, ac=None, ended="2026-06-02T21:18:36+00:00",
            ctype=None, reconstructed=0):
    pdb._conn.execute(
        "INSERT INTO charges (id, vehicle_id, started_at, ended_at, start_soc, end_soc,"
        " energy_added_kwh, ac_energy_kwh, location_type, reconstructed)"
        " VALUES (?,1,'2026-06-02T16:48:39+00:00',?,40,52,8.0,?,?,?)",
        (cid, ended, ac, ctype, reconstructed))
    pdb._conn.commit()


def _row(pdb, cid):
    return pdb._conn.execute("SELECT * FROM charges WHERE id=?", (cid,)).fetchone()


def test_off_by_default(tmp_path, monkeypatch):
    pdb = _setup(tmp_path, monkeypatch)
    _charge(pdb, 1, ac=10.0)
    assert db_reader.auto_confirm_home_charges() == 0
    assert _row(pdb, 1)["location_type"] is None


def test_confirms_and_bills_on_ac_flat(tmp_path, monkeypatch):
    """Tagged HOME and billed on the wallbox AC energy (10 kWh), not the DC 8 kWh."""
    pdb = _setup(tmp_path, monkeypatch)
    db_reader.set_setting("wallbox_auto_home", "1")
    db_reader.set_setting("price_home_kwh", "0.25")
    _charge(pdb, 1, ac=10.0)
    assert db_reader.auto_confirm_home_charges() == 1
    row = _row(pdb, 1)
    assert row["location_type"] == "HOME"
    assert row["cost"] == 2.5                      # 10 kWh AC × 0.25 — not 8 × 0.25


def test_tou_band_honoured(tmp_path, monkeypatch):
    """The PR-#47 flaw this implementation fixes: with TOU pricing configured, the auto
    assignment must price by the bands (here 0.10), not by the flat base price (0.25)."""
    pdb = _setup(tmp_path, monkeypatch)
    db_reader.set_setting("wallbox_auto_home", "1")
    db_reader.set_setting("price_home_kwh", "0.25")
    db_reader.set_setting("cost_mode", "tou")
    db_reader.set_setting("tou_method", "start")
    db_reader.set_setting("tou_bands", json.dumps(   # start == end → whole-day band,
        [{"start": "00:00", "end": "00:00",          # so the test is timezone-proof
          "prices": {"HOME": 0.10}, "days": [0, 1, 2, 3, 4, 5, 6]}]))
    _charge(pdb, 1, ac=10.0)
    assert db_reader.auto_confirm_home_charges() == 1
    assert _row(pdb, 1)["cost"] == 1.0             # 10 kWh AC × band 0.10, NOT × 0.25


def test_equals_manual_confirm(tmp_path, monkeypatch):
    """Core guarantee: the sweep produces byte-identical results to a manual confirm."""
    pdb = _setup(tmp_path, monkeypatch)
    db_reader.set_setting("price_home_kwh", "0.25")
    _charge(pdb, 1, ac=7.4)
    _charge(pdb, 2, ac=7.4)
    manual = db_reader.update_charge_type(1, "HOME")
    db_reader.set_setting("wallbox_auto_home", "1")
    assert db_reader.auto_confirm_home_charges() == 1   # only #2 is still untyped
    auto = dict(_row(pdb, 2))
    assert (auto["location_type"], auto["cost"]) == (manual["location_type"], manual["cost"])


def test_skips_ineligible(tmp_path, monkeypatch):
    """No wallbox energy / meter jitter / still charging / reconstructed / already typed →
    all left alone for the manual badge."""
    pdb = _setup(tmp_path, monkeypatch)
    db_reader.set_setting("wallbox_auto_home", "1")
    db_reader.set_setting("price_home_kwh", "0.25")
    _charge(pdb, 1, ac=None)                       # public/DC charge: no wallbox energy
    _charge(pdb, 2, ac=0.03)                       # ≤ 0.05 kWh floor: meter jitter
    _charge(pdb, 3, ac=10.0, ended=None)           # still charging
    _charge(pdb, 4, ac=10.0, reconstructed=1)      # reconstructed while offline
    _charge(pdb, 5, ac=10.0, ctype="AC")           # user already picked a type
    assert db_reader.auto_confirm_home_charges() == 0
    for cid, expected in ((1, None), (2, None), (3, None), (4, None), (5, "AC")):
        assert _row(pdb, cid)["location_type"] == expected
