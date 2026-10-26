"""Observed-wind calibration: land shape, county correction, curve transform."""

from __future__ import annotations

import math

import pytest

from app.services import wind_calibration as wc
from app.services.damage_curves import REF_MARINE_MPH, land_mph
from app.services.hurricane_impact import CountyImpact


def _station(model_kt: float, obs_kt: float, lat: float = 27.0, lon: float = -82.0, **kw) -> dict:
    return {
        "modelWindKt": model_kt,
        "obsMaxSustainedKt": obs_kt / wc.SUSTAINED_TO_1MIN,
        "lat": lat,
        "lon": lon,
        **kw,
    }


def _impact(gid: str, lat: float, lon: float) -> CountyImpact:
    return CountyImpact(
        geoid=gid[-5:], geography_id=gid, name=gid, state_usps="FL",
        centroid_lat=lat, centroid_lon=lon, max_wind_kt=80, max_category=1,
        closest_distance_nm=10.0, rmax_at_closest_nm=20.0, rmax_source="ibtracs",
        tiv=0.0, location_count=0, has_data=False,
    )


def test_fit_land_shape_recovers_power_law() -> None:
    a, b = 0.5, -0.35
    stations = [_station(m, m * a * (m / 64) ** b) for m in range(34, 83, 2)]
    shape = wc.fit_land_shape(stations)
    assert shape.ratio_at_ref == pytest.approx(a, rel=1e-6)
    assert shape.exponent == pytest.approx(b, abs=1e-6)
    # Too few strong-wind stations: the default floor.
    assert shape.floor_ratio == wc.FLOOR_RATIO


def test_floor_is_median_of_strong_stations_and_outages_are_skipped() -> None:
    weak = [_station(m, m * 0.6) for m in range(34, 64)]
    strong = [_station(90 + i, (90 + i) * r) for i, r in enumerate([0.40, 0.45, 0.50] * 8)]
    outage = [_station(120, 10, probableOutage=True)] * 50
    shape = wc.fit_land_shape(weak + strong + outage)
    assert shape.n_stations == len(weak) + len(strong)
    assert shape.floor_ratio == pytest.approx(0.45)
    assert shape.ratio(150) == pytest.approx(0.45)


def test_too_few_stations_raise() -> None:
    with pytest.raises(ValueError):
        wc.fit_land_shape([_station(60, 30)] * 5)


def test_land_mph_keeps_the_axis_at_64_kt() -> None:
    assert land_mph(REF_MARINE_MPH, 0.8, -0.36, 0.91) == pytest.approx(0.8 * REF_MARINE_MPH)
    assert land_mph(50, 0.8, -0.36, 0.91) > 0.8 * 50
    assert land_mph(160, 0.8, -0.36, 0.91) == pytest.approx(0.8 * 160 * 0.91)
    assert land_mph(100, 0.8) == pytest.approx(80)
    assert land_mph(0, 0.8, -0.36, 0.91) == 0.0


def test_county_correction_is_local_and_shrunk() -> None:
    shape = wc.LandShape(ratio_at_ref=0.5, exponent=0.0, floor_ratio=0.0)
    # Every station in one cluster reads 20% above the shape.
    stations = [_station(70, 70 * 0.5 * 1.2, lat=27.0 + 0.01 * i) for i in range(10)]
    corr = wc.storm_correction(stations, shape)
    assert corr is not None
    full = math.log(1.2)
    # Storm mean shrinks as n / (n + 5).
    assert corr.storm_mean == pytest.approx(full * 10 / 15)
    far = corr.at(35.0, -90.0)
    assert far == pytest.approx(corr.storm_mean)
    assert corr.storm_mean < corr.at(27.05, -82.0) <= full


def test_county_multipliers_use_evidence(monkeypatch) -> None:
    stations = [_station(70, 70 * 0.5 * 1.3, lat=27.0, lon=-82.0 + 0.01 * i) for i in range(6)]
    monkeypatch.setattr(wc, "load_evidence", lambda sid: {"stations": stations} if sid == "AL1" else None)
    wc.clear_caches()
    shape = wc.LandShape(ratio_at_ref=0.5, exponent=0.0, floor_ratio=0.0)
    impacts = [_impact("US-FL-12001", 27.0, -82.0), _impact("US-FL-12002", 34.0, -90.0)]
    out = wc.county_multipliers("al1", impacts, shape)
    assert out["US-FL-12001"] > out["US-FL-12002"] > 1.0
    half = wc.county_multipliers("al1", impacts, shape, strength=0.5)
    assert half["US-FL-12001"] == pytest.approx(math.sqrt(out["US-FL-12001"]))
    assert wc.county_multipliers("AL2", impacts, shape) == {}
    wc.clear_caches()


def test_api_applies_observed_wind_only_when_it_can(monkeypatch) -> None:
    from app.api import industry_loss as api

    class Storm:
        storm_id = "AL1"

    class Shaped:
        fit = {"windShape": {"ratioAtRef": 0.5, "exponent": 0.0, "floorRatio": 0.0}}

    stations = [_station(70, 70 * 0.5 * 1.3, lat=27.0, lon=-82.0 + 0.01 * i) for i in range(6)]
    monkeypatch.setattr(wc, "load_evidence", lambda sid: {"stations": stations} if sid == "AL1" else None)
    wc.clear_caches()
    impacts = [_impact("US-FL-12001", 27.0, -82.0)]
    assert api._observed_wind(Storm(), "ibtracs", impacts, Shaped(), False)[1]["reason"] == "switched off"
    assert api._observed_wind(Storm(), "live", impacts, Shaped(), True)[1]["reason"] == "live storm"
    assert api._observed_wind(Storm(), "ibtracs", impacts, object(), True)[0] == {}
    mult, info = api._observed_wind(Storm(), "ibtracs", impacts, Shaped(), True)
    assert info["applied"] and info["counties"] == 1
    assert mult["US-FL-12001"] > 1.0
    wc.clear_caches()


def test_shape_for_reads_the_curve_fit() -> None:
    class Curve:
        fit = {"windShape": {"ratioAtRef": 0.51, "exponent": -0.36, "floorRatio": 0.47, "nStations": 900}}

    shape = wc.shape_for(Curve())
    assert shape == wc.LandShape(0.51, -0.36, 0.47, 900)
    assert wc.shape_for(object()) is None
