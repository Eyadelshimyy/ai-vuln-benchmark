"""On-levelling reported industry losses to the TIV valuation date."""

from __future__ import annotations

from datetime import date

import pytest

from app.services import industry_losses, onlevel
from app.services.industry_losses import EventActuals
from app.services.onlevel import annual_at, factor, onlevel_actuals

E, T = date(2022, 7, 1), date(2025, 7, 1)


@pytest.fixture
def indices(tmp_path, monkeypatch):
    monkeypatch.setattr(industry_losses, "data_dir", lambda: tmp_path)
    folder = tmp_path / "onlevel"
    folder.mkdir()
    (folder / "county_indices.csv").write_text(
        "# comment\n"
        "geographyId,metric,2022,2023,2024,2025\n"
        "US-FL-12071,HU,100,104,108,110\n"
        "US-FL-12071,POP,200,210,220,230\n"
        "US-FL-12071,EMP,1000,1100,1200,1300\n"
        "US-FL-12071,WAGES,10,11,12,14\n"
        "US-FL-12999,EMP,100,100,100,300\n"
        "US-FL-12999,WAGES,1,1,1,9\n"
        "US-FL,HU,1000,1010,1020,1050\n"
        "US-FL,POP,1,1,1,1\n"
        "US-FL,EMP,1000,1000,1000,1100\n"
        "US-FL,WAGES,100,100,100,120\n"
        "US-FL,CONS_PAY,50,52,55,60\n"
        "US-PR,CONS_PAY,20,20,20,25\n",
        encoding="utf-8",
    )
    (folder / "monthly_indices.csv").write_text(
        "month,PPI_RES_INPUTS,PPI_NONRES_INPUTS,CPI_USED_VEHICLES\n"
        "2022-07,100,100,100\n"
        "2025-07,120,110,90\n",
        encoding="utf-8",
    )
    onlevel.clear_caches()
    yield folder
    onlevel.clear_caches()


def test_annual_interpolates_mid_year_and_extrapolates_briefly() -> None:
    s = {2022: 100.0, 2023: 110.0}
    assert annual_at(s, date(2022, 7, 2)) == pytest.approx(100.0, abs=0.1)
    assert annual_at(s, date(2023, 1, 1)) == pytest.approx(105.0, abs=0.2)
    assert annual_at(s, date(2024, 7, 2)) == pytest.approx(121.0, abs=0.2)  # last growth rate
    assert annual_at(s, date(2026, 7, 1)) is None  # too far past the data


def test_residential_is_units_times_replacement_cost(indices) -> None:
    f = factor("US-FL-12071", "RESIDENTIAL", E, T)
    cost = 0.55 * 1.2 + 0.45 * (60 / 50)
    assert f.factor == pytest.approx(1.10 * cost, rel=1e-3)
    assert f.levels == {"HU": "county", "PPI_RES_INPUTS": "national", "CONS_PAY": "state"}


def test_commercial_blends_building_and_payroll(indices) -> None:
    f = factor("US-FL-12071", "COMMERCIAL", E, T)
    cost = 0.55 * 1.1 + 0.45 * 1.2
    assert f.factor == pytest.approx(0.62 * 1.3 * cost + 0.38 * 1.4, rel=1e-3)


def test_auto_is_population_times_used_vehicle_prices(indices) -> None:
    f = factor("US-FL-12071", "AUTO", E, T)
    assert f.factor == pytest.approx(1.15 * 0.9, rel=1e-3)


def test_fallbacks_small_county_state_and_flat(indices) -> None:
    small = factor("US-FL-12999", "COMMERCIAL", E, T)
    assert small.levels["EMP"] == "state"  # below the employment floor
    assert small.components["EMP"] == pytest.approx(1.1, rel=1e-3)
    missing = factor("US-FL-12001", "RESIDENTIAL", E, T)
    assert missing.levels["HU"] == "state" and missing.components["HU"] == pytest.approx(1.05, rel=1e-3)
    pr = factor("US-PR-72127", "RESIDENTIAL", E, T)
    assert pr.levels["HU"] == "flat" and pr.components["CONS_PAY"] == pytest.approx(1.25, rel=1e-3)


def test_onlevel_actuals_scales_rows_and_applies_adjustments(indices) -> None:
    (indices / "adjustments.csv").write_text(
        "stormId,geographyId,segment,factor,reason\nAL092022,US-FL,RESIDENTIAL,0.9,reform\n",
        encoding="utf-8",
    )
    ev = EventActuals(storm_id="AL092022", currency="USD", event_date="2022-07-01",
                      county={("US-FL-12071", "RESIDENTIAL"): 100.0, ("US-FL-12071", "AUTO"): 10.0},
                      unallocated={("FL", "RESIDENTIAL"): 5.0})
    scaled, info = onlevel_actuals(ev, T)
    res = factor("US-FL-12071", "RESIDENTIAL", E, T).factor
    state = factor("US-FL", "RESIDENTIAL", E, T).factor
    assert scaled.county[("US-FL-12071", "RESIDENTIAL")] == pytest.approx(100 * res * 0.9)
    assert scaled.unallocated[("FL", "RESIDENTIAL")] == pytest.approx(5 * state * 0.9)
    assert scaled.county[("US-FL-12071", "AUTO")] == pytest.approx(10 * 1.15 * 0.9, rel=1e-3)
    assert ev.county[("US-FL-12071", "RESIDENTIAL")] == 100.0  # original untouched
    assert info["reportedTotal"] == 115.0
    assert info["factor"] == pytest.approx(scaled.total() / 115.0)
    assert info["adjustments"][0]["reason"] == "reform"


def test_no_indices_returns_actuals_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(industry_losses, "data_dir", lambda: tmp_path)
    onlevel.clear_caches()
    ev = EventActuals(storm_id="AL092022", currency="USD", county={("US-FL-12071", "AUTO"): 1.0})
    assert onlevel_actuals(ev, T) == (ev, None)
    onlevel.clear_caches()
