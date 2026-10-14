"""HOME charges on a wallbox are billed on the real AC energy the wallbox delivered (what you pay),
not just the DC energy into the battery. Other charges (no ac_kwh) keep DC billing.

These exercise db_reader.compute_cost directly (no fastapi needed → run in CI's minimal env).
The _session_energy AC plausibility guard lives in tests/test_session_energy.py (needs web.main)."""
import db_reader


def _patch_flat(monkeypatch, home_price=0.25):
    monkeypatch.setattr(db_reader, "get_cost_config",
                        lambda: {"mode": "flat", "method": "split", "bands": []})
    monkeypatch.setattr(db_reader, "get_charge_prices", lambda: {"price_home_kwh": home_price})


CHARGE = {"location_type": "HOME", "energy_added_kwh": 8.0,
          "started_at": "2026-06-02T16:48:39+00:00", "ended_at": "2026-06-02T21:18:36+00:00"}


def test_cost_uses_ac_kwh_when_given(monkeypatch):
    _patch_flat(monkeypatch)
    assert db_reader.compute_cost(CHARGE) == 2.0                  # DC: 8 kWh × 0.25
    assert db_reader.compute_cost(CHARGE, ac_kwh=10.0) == 2.5     # AC: 10 kWh × 0.25 (losses billed)


def test_cost_falls_back_to_dc_without_ac(monkeypatch):
    _patch_flat(monkeypatch)
    assert db_reader.compute_cost(CHARGE, ac_kwh=None) == 2.0     # no wallbox data → DC
    assert db_reader.compute_cost(CHARGE, ac_kwh=0) == 2.0        # 0/invalid → DC
    assert db_reader.compute_cost(CHARGE, ac_kwh=-1) == 2.0       # guard against bad values
