"""The energy figure shown for a charge (and summed into the day/month/year/total stats)
is the BILLED energy: the wallbox-measured AC kWh for HOME charges that have a wallbox
reading, else the battery DC (SoC) energy. This keeps the per-charge card, the period
totals and get_charge_stats all consistent with the cost (which bills on the same basis).
"""
import db_reader


def test_billed_kwh_home_with_wallbox_uses_ac():
    assert db_reader._billed_kwh(
        {"location_type": "HOME", "ac_energy_kwh": 38.19, "energy_added_kwh": 34.56}) == 38.19


def test_billed_kwh_home_without_wallbox_falls_back_to_dc():
    assert db_reader._billed_kwh(
        {"location_type": "HOME", "ac_energy_kwh": None, "energy_added_kwh": 34.56}) == 34.56
    # a zero/again-empty wallbox reading also falls back to DC
    assert db_reader._billed_kwh(
        {"location_type": "HOME", "ac_energy_kwh": 0, "energy_added_kwh": 34.56}) == 34.56


def test_billed_kwh_public_charge_uses_dc_even_if_ac_present():
    # Non-HOME types have no wallbox; bill on the battery DC energy regardless of ac_energy_kwh.
    assert db_reader._billed_kwh(
        {"location_type": "FAST", "ac_energy_kwh": 99.0, "energy_added_kwh": 40.0}) == 40.0
    assert db_reader._billed_kwh(
        {"location_type": None, "ac_energy_kwh": None, "energy_added_kwh": 12.3}) == 12.3


def test_billed_kwh_missing_energy_is_zero():
    assert db_reader._billed_kwh({"location_type": "HOME"}) == 0
