"""Dynamic-price sensor picker (#104): candidate entities are found by a price-ish keyword in
the id/name OR a per-kWh currency unit (language-independent — a Nordpool/Tibber/ENTSO-E sensor
always reports one), ranked keyword-matches first. Domain is restricted to sensor/input_number/
number so a device_tracker or switch can never show up. Pure filter over a fake states list →
CI-safe, no network."""
import ha_client as H


def _st(eid, unit="", friendly=None):
    return {"entity_id": eid, "state": "0.30",
            "attributes": {"friendly_name": friendly or eid, "unit_of_measurement": unit}}


def test_matches_by_keyword_regardless_of_unit(monkeypatch):
    monkeypatch.setattr(H, "_fetch_states", lambda: [
        _st("sensor.nordpool_kwh_it_eur", unit=""),
        _st("sensor.unrelated_temp", unit="°C"),
    ])
    ids = [e["entity_id"] for e in H.list_price_entities()]
    assert "sensor.nordpool_kwh_it_eur" in ids
    assert "sensor.unrelated_temp" not in ids


def test_matches_by_per_kwh_currency_unit_even_without_a_price_word(monkeypatch):
    # e.g. a utility's own integration might just be called "sensor.my_provider" but report EUR/kWh
    monkeypatch.setattr(H, "_fetch_states", lambda: [
        _st("sensor.my_provider", unit="EUR/kWh", friendly="My Provider"),
    ])
    ids = [e["entity_id"] for e in H.list_price_entities()]
    assert "sensor.my_provider" in ids


def test_keyword_matches_rank_before_unit_only_matches(monkeypatch):
    monkeypatch.setattr(H, "_fetch_states", lambda: [
        _st("sensor.some_meter", unit="EUR/kWh"),          # unit-only match
        _st("sensor.tibber_price", unit=""),                # keyword match
    ])
    out = H.list_price_entities()
    assert out[0]["entity_id"] == "sensor.tibber_price"


def test_wrong_domain_is_excluded_even_with_price_in_the_name(monkeypatch):
    # a switch/automation named "price_something" is never a usable price SOURCE
    monkeypatch.setattr(H, "_fetch_states", lambda: [
        _st("switch.price_alert_enabled", unit=""),
        _st("input_number.costo_energia", unit="EUR/kWh"),
    ])
    ids = [e["entity_id"] for e in H.list_price_entities()]
    assert "switch.price_alert_enabled" not in ids
    assert "input_number.costo_energia" in ids


def test_no_candidates_returns_empty_list(monkeypatch):
    monkeypatch.setattr(H, "_fetch_states", lambda: [_st("sensor.unrelated", unit="°C")])
    assert H.list_price_entities() == []


def test_not_configured_returns_empty_without_error(monkeypatch):
    monkeypatch.setattr(H, "_fetch_states", lambda: [])   # is_configured()==False path
    assert H.list_price_entities() == []
