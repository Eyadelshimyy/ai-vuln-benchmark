"""Fuel WAC (weighted-average-cost) €/L blend — the REEV FUEL twin of #53's battery WAC. Pure-function
tests for _fuel_wac_blend: deterministic, no DB. Litres cancel (the blend uses fuel-% ratios), so it's
tank-size-free; a refuel adds `liters` at `price_per_l` over a residual `fuel_before_pct`."""
from db_reader import _fuel_wac_blend, _REEV_TANK_L


def _refuel(fuel_before_pct, liters, price_per_l):
    return {"fuel_before_pct": fuel_before_pct, "liters": liters, "price_per_l": price_per_l}


def test_no_refuels_returns_none():
    assert _fuel_wac_blend([]) is None


def test_single_refuel_is_its_own_rate():
    # The first refuel bootstraps the tank to its own €/L, whatever the residual.
    assert abs(_fuel_wac_blend([_refuel(0, 30, 1.70)]) - 1.70) < 1e-9
    assert abs(_fuel_wac_blend([_refuel(20, 25, 1.85)]) - 1.85) < 1e-9


def test_two_prices_blend_by_litres_in_tank():
    # 8 L residual @1.70 + 15 L @1.85 → (8·1.70 + 15·1.85)/23 = 1.7978 €/L. Modelled as: a first fill
    # bootstraps 1.70; before the 2nd fill the tank holds 8 L → fuel_before_pct = 8/50·100 = 16 %.
    first = _refuel(0, _REEV_TANK_L, 1.70)       # bootstraps p = 1.70
    second = _refuel(16.0, 15, 1.85)             # 8 L residual (16 % of 50) + 15 L @1.85
    p = _fuel_wac_blend([first, second])
    assert abs(p - (8 * 1.70 + 15 * 1.85) / 23) < 1e-6


def test_driving_between_refuels_does_not_change_the_blend():
    # Two refuels at the SAME €/L, any residual between → blend stays that rate (consumption-invariant).
    assert abs(_fuel_wac_blend([_refuel(0, _REEV_TANK_L, 1.75), _refuel(30, 20, 1.75)]) - 1.75) < 1e-9


def test_blend_is_bounded_by_prices_paid():
    p = _fuel_wac_blend([_refuel(0, 40, 1.60), _refuel(20, 30, 1.80), _refuel(35, 25, 2.05)])
    assert 1.60 <= p <= 2.05              # always a convex mix of the prices actually paid


def test_unknown_residual_bootstraps_then_carries_forward():
    # fuel_before_pct=None (no car data before the refuel): the first one still bootstraps to its rate;
    # a later None-residual refuel can't weight the mix → carry-forward (blend unchanged).
    assert abs(_fuel_wac_blend([_refuel(None, 30, 1.90)]) - 1.90) < 1e-9
    base = [_refuel(0, _REEV_TANK_L, 1.70)]
    assert _fuel_wac_blend(base) == _fuel_wac_blend(base + [_refuel(None, 20, 2.50)])


def test_zero_or_missing_liters_ignored():
    base = [_refuel(0, _REEV_TANK_L, 1.70)]
    assert _fuel_wac_blend(base + [_refuel(20, 0, 2.99)]) == _fuel_wac_blend(base)
    assert _fuel_wac_blend(base + [_refuel(20, None, 2.99)]) == _fuel_wac_blend(base)


def test_tank_size_free_uses_fuel_pct_ratios():
    # Same %-composition → same blend regardless of the absolute litres. 25 L residual (50 %) @1.70
    # + 25 L (50 % of the 50 L tank) @2.00 → 1.85 €/L.
    p = _fuel_wac_blend([_refuel(0, _REEV_TANK_L, 1.70), _refuel(50.0, _REEV_TANK_L / 2, 2.00)])
    assert abs(p - 1.85) < 1e-6
