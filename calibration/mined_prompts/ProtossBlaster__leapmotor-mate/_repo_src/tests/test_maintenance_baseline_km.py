"""#112 (@Wartopia): maintenance km bars must start from the real DELIVERY odometer, not the odometer
Mate happened to see when it was installed. With the delivery km set to 0 (a car delivered new), a
first-service bar reflects the whole odometer (e.g. 11 563 km of a 20 000 km interval ≈ 58%), not the
handful of km driven since install (~3%). Also checks the delivery-km value is exposed to pre-fill the
edit form. CI-safe: get_baseline / latest_logs / unit-system stubbed, real B10 service pack."""
import maintenance
import units


def _stub(monkeypatch, baseline):
    monkeypatch.setattr(maintenance, "get_baseline", lambda: baseline)
    monkeypatch.setattr(maintenance, "latest_logs", lambda vid: {})      # no logged services yet
    monkeypatch.setattr(units, "get_unit_system", lambda: "metric")


def test_first_service_km_bar_uses_delivery_km_zero(monkeypatch):
    _stub(monkeypatch, ("2025-11-01", 0.0, True))                        # new car, delivered at 0 km
    out = maintenance.compute({"car_type": "B10", "id": 1}, 11563, "en")
    km_rows = [r for r in out["rows"] if r.get("km_pct") is not None]
    assert km_rows, "expected items with a km interval in the B10 pack"
    # 20 000 km first-service at 11 563 km from 0 → ~58%; the OLD bug (anchor ~11 000) gave ~3% for all.
    assert any(r["km_pct"] >= 40 for r in km_rows)
    assert all(0 <= r["km_pct"] <= 120 for r in km_rows)                 # clamp intact


def test_buggy_install_anchor_would_show_near_empty(monkeypatch):
    # Sanity: with the OLD behaviour (anchor = odometer seen on install, ~11 000) every bar is tiny —
    # this is exactly what @Wartopia saw and what the fix removes.
    _stub(monkeypatch, ("2025-11-01", 11000.0, True))
    out = maintenance.compute({"car_type": "B10", "id": 1}, 11563, "en")
    km_rows = [r for r in out["rows"] if r.get("km_pct") is not None]
    assert km_rows and all(r["km_pct"] < 10 for r in km_rows)            # ~3% — the reported bug


def test_delivery_km_input_exposed_for_the_form(monkeypatch):
    _stub(monkeypatch, ("2025-11-01", 5000.0, True))                     # e.g. a used car bought at 5 000
    out = maintenance.compute({"car_type": "B10", "id": 1}, 11563, "en")
    assert out["baseline_km_input"] == 5000                             # pre-fills the delivery-km field
