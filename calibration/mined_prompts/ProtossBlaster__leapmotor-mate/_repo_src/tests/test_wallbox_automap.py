"""Wallbox entity auto-mapping — GitHub #44.

A Telegram add-on user with a V2C Trydan charger (entities sensor.evse_v2c_trydan_local_*,
Italian friendly names) had to map by hand. The fixtures below are the real entities from
their Home Assistant dump. The tricky part: "Energia fotovoltaica" and "Alimentazione
domestica Inverted" are device_class power too, so they must NOT be picked for the
charging-power role — only the actual "Potenza di carica" should.
"""
import ha_client


def _e(eid, name, device_class, unit):
    return {"entity_id": eid, "name": name, "device_class": device_class, "unit": unit}


P = "sensor.evse_v2c_trydan_local_potenza_di_carica"
EN = "sensor.evse_v2c_trydan_local_energia_di_carica"

V2C = [
    _e("sensor.evse_v2c_trydan_local_alimentazione_domestica_inverted",
       "Alimentazione domestica Inverted", "power", "W"),
    _e(EN, "EVSE v2c-trydan.local Energia di carica", "energy", "kWh"),
    _e("sensor.evse_v2c_trydan_local_energia_fotovoltaica",
       "EVSE v2c-trydan.local Energia fotovoltaica", "power", "W"),
    _e(P, "EVSE v2c-trydan.local Potenza di carica", "power", "W"),
    _e("sensor.evse_v2c_trydan_local_tempo_di_carica",
       "EVSE v2c-trydan.local Tempo di carica", "duration", "s"),
]


def test_v2c_power_role_picks_charging_power_not_solar_or_house():
    mapping = ha_client.auto_map(V2C)
    assert mapping.get("power") == P


def test_v2c_energy_role_picks_charge_energy():
    mapping = ha_client.auto_map(V2C)
    assert mapping.get("energy") == EN


def test_v2c_solar_and_house_power_are_never_charging_metrics():
    # both are device_class power; the negative keywords must keep them out of every role
    for trap in ("energia_fotovoltaica", "alimentazione_domestica_inverted"):
        eid = f"sensor.evse_v2c_trydan_local_{trap}"
        assert all(v != eid for v in ha_client.auto_map(V2C).values())


def test_v2c_entities_pass_the_wallbox_filter_keyword():
    # the "evse"/"v2c"/"trydan" DEFAULT keywords mean these surface without "Show all".
    # Match the entity id (language-independent), not the localized friendly_name.
    # Use _WB_KEYWORDS directly so the test never touches the settings DB (absent in CI).
    assert "v2c" in ha_client._WB_KEYWORDS
    assert any(k in "sensor.evse_v2c_trydan_local_potenza_di_carica"
               for k in ha_client._WB_KEYWORDS)
