"""Live-storm county impact uses NHC official track + wind radii, not HURDAT."""

from __future__ import annotations

from app.services.hurricane_impact import CountyMeta, compute_impact
from app.services.ibtracs import Storm, TrackPoint
from app.services.live_hurricane import storm_for_impact
from app.services.atcf_adecks import OfficialFix


def test_compute_impact_uses_nhc_quadrants(monkeypatch) -> None:
    import app.services.hurricane_impact as hi

    # County ~20 nm due NE of 25N, 80W — inside 40 nm NE R64, outside 5 nm SW.
    county = CountyMeta(
        geoid="12086",
        geography_id="US-FL-12086",
        name="Miami-Dade",
        state_usps="FL",
        centroid_lat=25.25,
        centroid_lon=-79.75,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"12086": county})
    # Keep this unit test independent of real county samples cached by earlier
    # tests in the same process; it is exercising the quadrant/centroid path.
    monkeypatch.setattr(hi, "county_area_samples", lambda: {})

    storm = Storm(
        storm_id="AL012026",
        name="TEST",
        year=2026,
        track=[
            TrackPoint(
                datetime_utc="2026-09-01T12:00:00Z",
                record_id="",
                status="TS",  # would be skipped under the old HU-only IBTrACS filter
                lat=25.0,
                lon=-80.0,
                wind_kt=90,
                pressure_mb=960,
                rmax_nm=18.0,
                r64_quads_nm=(40.0, 10.0, 5.0, 10.0),
                radii_source="nhc",
            )
        ],
    )
    impacts, footprint, _inner, _outer, _rings = compute_impact(storm)
    assert footprint
    assert footprint[0].rmax_source == "nhc"
    assert footprint[0].r64_source == "nhc"
    assert any(i.geoid == "12086" for i in impacts)


def test_r34_reaches_counties_the_core_misses(monkeypatch) -> None:
    """Isaias shape: 64 kt core ~30 nm offshore, 34 kt field over the coast."""
    import app.services.hurricane_impact as hi

    # ~90 nm north of the eye. Inside R34 160, outside R64 30.
    county = CountyMeta(
        geoid="12033",
        geography_id="US-FL-12033",
        name="Escambia",
        state_usps="FL",
        centroid_lat=26.5,
        centroid_lon=-87.6,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"12033": county})
    monkeypatch.setattr(hi, "county_area_samples", lambda: {})
    storm = Storm(
        storm_id="AL092026",
        name="ISAIAS",
        year=2026,
        track=[
            TrackPoint(
                datetime_utc="2026-10-09T12:00:00Z",
                record_id="",
                status="HU",
                lat=25.0,
                lon=-87.6,
                wind_kt=105,
                pressure_mb=959,
                rmax_nm=20.0,
                r64_quads_nm=(30.0, 30.0, 20.0, 20.0),
                r34_quads_nm=(180.0, 110.0, 80.0, 160.0),
                radii_source="nhc",
            )
        ],
    )
    impacts, footprint, *_ = compute_impact(storm)
    assert footprint
    assert len(impacts) == 1
    hit = impacts[0]
    assert hit.max_wind_kt < 64
    assert hit.max_category == 0
    # A second county well outside R34 must not appear.
    far = CountyMeta(
        geoid="36061",
        geography_id="US-NY-36061",
        name="New York",
        state_usps="NY",
        centroid_lat=40.7,
        centroid_lon=-74.0,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"12033": county, "36061": far})
    impacts, *_ = compute_impact(storm)
    assert [i.geoid for i in impacts] == ["12033"]


def test_cone_does_not_stamp_inland_wind(monkeypatch) -> None:
    """The forecast cone is not a wind field. Inland of R34 stays clear."""
    import app.services.hurricane_impact as hi

    inland = CountyMeta(
        geoid="47037",
        geography_id="US-TN-47037",
        name="Davidson",
        state_usps="TN",
        centroid_lat=36.2,
        centroid_lon=-86.8,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"47037": inland})
    monkeypatch.setattr(hi, "county_area_samples", lambda: {})
    storm = Storm(
        storm_id="AL092026",
        name="ISAIAS",
        year=2026,
        track=[
            TrackPoint(
                datetime_utc="2026-10-11T00:00:00Z",
                record_id="",
                status="TS",
                lat=34.4,
                lon=-86.6,
                wind_kt=41,
                pressure_mb=1000,
                rmax_nm=20.0,
                r34_quads_nm=(30.0, 20.0, 14.0, 30.0),
                radii_source="nhc",
                forecast_hour=36,
                cone_radius_nm=55.0,
            )
        ],
        forecast_cone=[(-88.0, 34.0), (-84.0, 34.0), (-84.0, 38.0), (-88.0, 38.0), (-88.0, 34.0)],
    )
    impacts, *_ = compute_impact(storm)
    assert impacts == []


def test_weak_advisory_does_not_paint_hurricane_winds(monkeypatch) -> None:
    """A 35 kt point must not grow a 63 kt ring just because R64 was missing."""
    import app.services.hurricane_impact as hi

    county = CountyMeta(
        geoid="01003",
        geography_id="US-AL-01003",
        name="Baldwin",
        state_usps="AL",
        centroid_lat=30.7,
        centroid_lon=-87.7,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"01003": county})
    monkeypatch.setattr(hi, "county_area_samples", lambda: {})
    storm = Storm(
        storm_id="AL092026",
        name="ISAIAS",
        year=2026,
        track=[
            TrackPoint(
                datetime_utc="2026-10-10T00:00:00Z",
                record_id="",
                status="TS",
                lat=30.7,
                lon=-87.7,
                wind_kt=35,
                pressure_mb=1002,
                rmax_nm=25.0,
                r34_quads_nm=(40.0, 40.0, 40.0, 40.0),
                radii_source="nhc",
                forecast_hour=24,
            )
        ],
    )
    impacts, *_ = compute_impact(storm)
    assert impacts
    assert impacts[0].max_wind_kt <= 35
    assert impacts[0].max_category <= 0


def test_storm_for_impact_prefers_official_adecks(monkeypatch) -> None:
    import app.services.live_hurricane as lh
    import app.services.atcf_adecks as ad

    monkeypatch.setattr(
        lh,
        "_get_live_entry",
        lambda atcf, **_: {"id": atcf, "name": "Gabrielle", "forecastTrack": {}},
    )
    monkeypatch.setattr(lh, "fetch_live_forecast_cone", lambda atcf, **_: [])
    monkeypatch.setattr(lh, "fetch_forecast_track", lambda url, *, refresh=False: [])
    monkeypatch.setattr(
        lh,
        "fetch_official_fixes",
        lambda atcf: [
            OfficialFix(
                hours_out=0,
                lat=24.5,
                lon=-83.0,
                wind_kt=100,
                pressure_mb=950,
                ty="HU",
                rmw_nm=16.0,
                r34_quads=(120, 100, 80, 90),
                r50_quads=(70, 60, 50, 55),
                r64_quads=(40, 30, 25, 35),
                init_cycle="2026090112",
            ),
            OfficialFix(
                hours_out=24,
                lat=26.0,
                lon=-82.0,
                wind_kt=90,
                pressure_mb=960,
                ty="HU",
                rmw_nm=18.0,
                r34_quads=None,
                r50_quads=None,
                r64_quads=(45, 35, 20, 30),
                init_cycle="2026090112",
            ),
        ],
    )
    storm = storm_for_impact("AL072026")
    assert storm is not None
    assert storm.name == "Gabrielle"
    assert [p.radii_source for p in storm.track] == ["nhc", "nhc"]
    assert storm.track[0].rmax_nm == 16.0
    assert storm.track[0].r64_quads_nm == (40.0, 30.0, 25.0, 35.0)
    assert storm.track[0].r34_quads_nm == (120.0, 100.0, 80.0, 90.0)
    assert storm.track[1].lat == 26.0
    # Must not require IBTrACS — a 2026 live id would 404 there.
    assert storm.storm_id == "AL072026"
