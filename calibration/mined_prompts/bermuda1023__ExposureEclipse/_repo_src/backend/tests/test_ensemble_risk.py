"""Unit tests for the ensemble strike-probability + intensity spread service.

Same fixture-driven pattern as ATCF a-decks: hand-crafted ModelTracks so we
don't depend on network or on the county-centroid TopoJSON reachability.
"""

from __future__ import annotations

import pytest

from app.services.atcf_adecks import ModelFix, ModelTrack
from app.services.ensemble_risk import (
    compute_ensemble_risk,
    DEFAULT_STRIKE_THRESHOLD_NM,
)


def _member(tech: str, family: str, lats: list[float], lons: list[float],
            winds: list[int]) -> ModelTrack:
    hours = [12, 24, 36, 48, 72, 96, 120][: len(lats)]
    return ModelTrack(
        tech_id=tech,
        label=tech,
        family=family,
        init_cycle="2024-09-26T12Z",
        fixes=[
            ModelFix(hours_out=h, lat=la, lon=lo, wind_kt=w, pressure_mb=None)
            for h, la, lo, w in zip(hours, lats, lons, winds)
        ],
    )


def _fake_counties() -> dict:
    """A tiny fake county map — used to monkeypatch county_centroids() so
    tests don't reach out for us-atlas TopoJSON."""
    from app.services.hurricane_impact import CountyMeta

    return {
        # A county at 25°N, -80°W (Miami-Dade-ish) — landfall of the
        # synthetic tracks.
        "12086": CountyMeta(
            geoid="12086",
            geography_id="US-FL-12086",
            name="Miami-Dade",
            state_usps="FL",
            centroid_lat=25.5,
            centroid_lon=-80.4,
        ),
        # A county at 30°N, -95°W — far from the tracks; should never be hit.
        "48201": CountyMeta(
            geoid="48201",
            geography_id="US-TX-48201",
            name="Harris",
            state_usps="TX",
            centroid_lat=29.8,
            centroid_lon=-95.3,
        ),
    }


@pytest.fixture(autouse=True)
def _patch_counties(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import ensemble_risk
    monkeypatch.setattr(ensemble_risk, "county_centroids", _fake_counties)


# ─────────────────────────── strike probability ───────────────────────────


def test_strike_probability_all_members_hit_florida_county() -> None:
    # Six ensemble members whose tracks all pass over/near Miami-Dade.
    members = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0],
                lons=[-78.0 - i * 0.1, -79.0 - i * 0.1, -80.0 - i * 0.1,
                      -80.5 - i * 0.1, -81.0 - i * 0.1, -81.5 - i * 0.1,
                      -82.0 - i * 0.1],
                winds=[70, 90, 110, 120, 110, 90, 70])
        for i in range(1, 7)
    ]
    risk = compute_ensemble_risk(members)
    assert risk.ensemble_total == 6
    fl = next(c for c in risk.strike_by_county if c.state_usps == "FL")
    # Every member hits → P = 1.0.
    assert fl.strike_probability == 1.0
    assert fl.member_count == 6
    # Peak intensity of passing members: fixture peaks at 120 kt.
    assert fl.max_intensity_kt == 120
    # Harris County (Texas, ~1000 nm away) is not in the results.
    assert not any(c.state_usps == "TX" for c in risk.strike_by_county)


def test_strike_probability_partial_agreement() -> None:
    # 5 GEFS members: 3 pass over Miami-Dade (25.5°N, -80.4°W), 2 pass far south.
    hit_members = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0],
                lons=[-78.0, -79.0, -80.0, -80.5, -81.0, -81.5, -82.0],
                winds=[70, 90, 100, 110, 100, 90, 70])
        for i in range(1, 4)
    ]
    miss_members = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0],
                lons=[-78.0, -79.0, -80.0, -80.5, -81.0, -81.5, -82.0],
                winds=[70, 90, 100, 110, 100, 90, 70])
        for i in range(4, 6)
    ]
    risk = compute_ensemble_risk(hit_members + miss_members)
    fl = next(c for c in risk.strike_by_county if c.state_usps == "FL")
    assert fl.ensemble_total == 5
    assert fl.member_count == 3
    assert fl.strike_probability == pytest.approx(0.6)


def test_threshold_scaling_controls_which_counties_qualify() -> None:
    # One track passing 200 nm south of Miami-Dade — inside 250 nm, outside 60 nm.
    members = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[22.0, 22.5, 22.5, 22.0, 21.5, 21.0, 20.5],
                lons=[-79.0, -80.0, -80.4, -80.5, -80.8, -81.0, -81.5],
                winds=[70, 90, 100, 110, 100, 90, 70])
        for i in range(1, 6)
    ]
    # Default 60 nm — should NOT hit Miami-Dade.
    tight = compute_ensemble_risk(members, threshold_nm=DEFAULT_STRIKE_THRESHOLD_NM)
    assert all(c.state_usps != "FL" for c in tight.strike_by_county)
    # Loose 250 nm — should hit.
    loose = compute_ensemble_risk(members, threshold_nm=250.0)
    assert any(c.state_usps == "FL" for c in loose.strike_by_county)


def test_ensemble_risk_ignores_non_ensemble_families() -> None:
    # Only OFCL + GFS deterministic — no GEFS / ECMWF-ENS / AI. The service
    # should still return a result but with ensemble_total=0.
    members = [
        _member("OFCL", "official", [25.0, 25.5], [-80.0, -80.5], [90, 100]),
        _member("AVNO", "gfs_det",  [24.5, 25.0], [-80.0, -80.5], [90, 100]),
    ]
    risk = compute_ensemble_risk(members)
    assert risk.ensemble_total == 0
    assert risk.strike_by_county == []


def test_intensity_spread_by_lead() -> None:
    # Members with spreading intensities — verify min/mean/max/std.
    members = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
                lons=[-80.0, -80.5, -81.0, -81.5, -82.0, -82.5, -83.0],
                # Wind at T+72 (index 4) spreads from 60 to 120 (12 kt steps).
                winds=[60 + i * 10] * 7)
        for i in range(0, 6)
    ]
    risk = compute_ensemble_risk(members)
    lead_72 = next(s for s in risk.intensity_by_lead if s.hours_out == 72)
    assert lead_72.min_kt == 60
    assert lead_72.max_kt == 110
    assert lead_72.mean_kt == pytest.approx(85.0)
    assert lead_72.member_count == 6


def test_majority_member_cycle_is_the_one_reported() -> None:
    older = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[25.0, 25.5], lons=[-80.0, -80.4], winds=[80, 90])
        for i in range(1, 5)
    ]
    for t in older:
        t.init_cycle = "2026-10-07T12Z"
    newer = _member("GDMI", "ai", [25.0, 25.5], [-80.0, -80.4], [80, 90])
    newer.init_cycle = "2026-10-07T18Z"
    risk = compute_ensemble_risk([newer, *older])
    assert risk.init_cycle == "2026-10-07T12Z"
    assert risk.ensemble_total == 5


def test_lead_before_24h_does_not_count_as_a_strike() -> None:
    # Passes over the county at 12 h, then runs away. The early segment
    # is the storm's current position, not the forecast.
    member = ModelTrack(
        tech_id="AP01", label="", family="gefs_ens",
        init_cycle="2026-10-07T12Z",
        fixes=[
            ModelFix(hours_out=12, lat=25.5, lon=-80.4, wind_kt=90, pressure_mb=None),
            ModelFix(hours_out=24, lat=28.0, lon=-84.0, wind_kt=90, pressure_mb=None),
            ModelFix(hours_out=48, lat=32.0, lon=-88.0, wind_kt=80, pressure_mb=None),
        ],
    )
    risk = compute_ensemble_risk([member])
    assert all(c.state_usps != "FL" for c in risk.strike_by_county)


def test_short_lead_members_excluded() -> None:
    # A member whose track stops at T+12 shouldn't count for strike probability
    # — MIN_LEAD_HOURS=24. But it will still contribute to intensity spread at
    # T+12 since spread aggregation walks fixes regardless of member drop-out.
    short_members = [
        ModelTrack(
            tech_id=f"AP{i:02d}", label="", family="gefs_ens",
            init_cycle="",
            fixes=[ModelFix(hours_out=12, lat=25.5, lon=-80.4, wind_kt=80, pressure_mb=None)],
        )
        for i in range(1, 4)
    ]
    long_members = [
        _member(f"AP{i:02d}", "gefs_ens",
                lats=[25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0],
                lons=[-80.0, -80.5, -81.0, -81.5, -82.0, -82.5, -83.0],
                winds=[80, 90, 100, 110, 100, 90, 80])
        for i in range(4, 7)
    ]
    risk = compute_ensemble_risk(short_members + long_members)
    fl = next(c for c in risk.strike_by_county if c.state_usps == "FL")
    # Only the 3 long members should count for the strike.
    assert fl.ensemble_total == 6
    assert fl.member_count == 3
    assert fl.strike_probability == 0.5
