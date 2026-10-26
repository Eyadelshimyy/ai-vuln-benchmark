"""Local-wind slices. A grazed county is not stamped with the storm peak."""

from __future__ import annotations

from app.services.county_wind import (
    experienced_wind_kt,
    include_county,
    local_wind_kt,
    mph_band,
    sample_polygons,
    skirt_wind_kt,
    speed_bins_from_winds,
)
from app.services.hurricane_impact import (
    CountyMeta,
    _impacts_from_paint,
    _paint_fix,
    compute_impact,
)
from app.services.ibtracs import Storm, TrackPoint


def test_speed_bins_match_form_v1_edges() -> None:
    # 115 kt is about 132 mph: the 131–140 row, and it is Cat 4.
    one = speed_bins_from_winds([115])
    assert one[0].mph_lo == 131 and one[0].mph_hi == 140
    assert one[0].category == 4
    assert mph_band(115) == (131, 140)
    # Clear samples stay out of the bins. The rest keep their share of the county.
    mixed = speed_bins_from_winds([50, 50, 0, 0])
    assert len(mixed) == 1
    assert mixed[0].mph_lo == 51
    assert mixed[0].category == 0
    assert abs(mixed[0].area_fraction - 0.5) < 1e-9


def test_skirt_is_not_painted_as_the_eyewall() -> None:
    # Irma over the keys: 115 kt, Rmax 10, R64 ~50. The wide ring is not Cat 4.
    skirt = skirt_wind_kt(115, 10.0, 52.5)
    assert 64 <= skirt < 96


def test_islands_do_not_outvote_the_mainland() -> None:
    def square(x0, y0, x1, y1):
        return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]

    # Mainland is ~0.64 deg². Four islands are ~0.09 deg² together, about 12%.
    # A per-island grid would give the islands most of the points.
    mainland = square(-81.0, 25.2, -80.2, 26.0)
    specks = [square(-81.7 + i * 0.25, 24.5, -81.55 + i * 0.25, 24.65) for i in range(4)]
    pts = sample_polygons([mainland, *specks])
    assert len(pts) >= 40
    keys = sum(1 for lat, _lon in pts if lat < 25.0)
    share = keys / len(pts)
    assert 0.03 < share < 0.25


def test_profile_pins_eyewall_and_r64_and_drops_the_far_field() -> None:
    # Instantaneous: calm eye, Vmax at the wall, ~64 kt at R64, nothing past the skirt.
    assert local_wind_kt(0.0, 110, 15.0, 40.0) == 0
    assert local_wind_kt(15.0, 110, 15.0, 40.0) == 110
    at_r64 = local_wind_kt(40.0, 110, 15.0, 40.0)
    assert 60 <= at_r64 <= 68
    assert local_wind_kt(250.0, 110, 15.0, 40.0) == 0
    # A quadrant with no hurricane-force radius never reports hurricane wind.
    assert local_wind_kt(20.0, 110, 15.0, 0.0) <= 63


def test_experienced_wind_is_vmax_once_the_eyewall_has_crossed() -> None:
    # The eye is calm at that instant. The point still felt the wall.
    assert experienced_wind_kt(0.0, 110, 15.0, 40.0) == 110
    assert experienced_wind_kt(15.0, 110, 15.0, 40.0) == 110
    # Outside the wall the peak is the profile, not Vmax.
    assert experienced_wind_kt(60.0, 110, 15.0, 40.0) < 64


def test_graze_splits_the_county_instead_of_using_the_storm_peak() -> None:
    """Miami-Dade shape from the screenshot: eye ~60 nm away, Rmax 15, 109 kt.

    One near-edge sample sits in the outer field. The rest of the county is
    farther out. The chip must not read 109, and most of the area is not the
    storm's category.
    """
    # 1 deg lat = 60 nm. Eye due west of the county.
    eye_lat, eye_lon = 25.6, -81.6
    # Centroid ~62 nm east. Near edge ~38 nm east (outer field). Far side ~95 nm.
    near = (eye_lat, eye_lon + 38.0 / 54.4)
    mid = (eye_lat, eye_lon + 62.0 / 54.4)
    far = (eye_lat, eye_lon + 95.0 / 54.4)
    samples = [near] + [mid] * 4 + [far] * 5  # 10% near, 40% mid, 50% far

    county = CountyMeta(
        geoid="12086",
        geography_id="US-FL-12086",
        name="Miami-Dade",
        state_usps="FL",
        centroid_lat=mid[0],
        centroid_lon=mid[1],
    )
    painted: dict = {}
    _paint_fix(
        painted,
        {"12086": county},
        {"12086": samples},
        eye_lat=eye_lat,
        eye_lon=eye_lon,
        vmax_kt=109,
        rmax=15.0,
        rmax_source="ibtracs",
        r64_nm=40.0,
        r64_quads=(40.0, 40.0, 40.0, 40.0),
    )
    impacts = _impacts_from_paint(painted)
    assert len(impacts) == 1
    hit = impacts[0]
    assert hit.max_wind_kt < 96  # not the 109 kt storm peak, and not Cat 3
    assert hit.max_wind_kt < 109
    by_cat = {b.category: b.area_fraction for b in hit.wind_bands}
    # The outer bit is TS or Cat 1. The rest is not the core.
    core = sum(by_cat.get(c, 0.0) for c in (3, 4, 5))
    assert core == 0.0
    # Mid and far sit outside R64, so they are clear, not a TS skirt.
    assert by_cat.get(-1, 0.0) >= 0.8
    assert include_county(hit.wind_bands, hit.max_wind_kt)


def test_compute_impact_keeps_a_centroid_fallback(monkeypatch) -> None:
    """Tests that stub centroids and forget samples still score that one point."""
    import app.services.hurricane_impact as hi

    county = CountyMeta(
        geoid="12086",
        geography_id="US-FL-12086",
        name="Miami-Dade",
        state_usps="FL",
        centroid_lat=25.25,
        centroid_lon=-79.75,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"12086": county})
    monkeypatch.setattr(hi, "county_area_samples", lambda: {})
    storm = Storm(
        storm_id="AL992026",
        name="GRAZE",
        year=2026,
        track=[
            TrackPoint(
                datetime_utc="2026-10-09T18:00:00Z",
                record_id="",
                status="HU",
                lat=25.0,
                lon=-81.2,
                wind_kt=130,
                pressure_mb=930,
                rmax_nm=15.0,
                r64_quads_nm=(40.0, 40.0, 40.0, 40.0),
                radii_source="nhc",
            )
        ],
    )
    impacts, footprint, *_rest = compute_impact(storm)
    assert footprint and footprint[0].rmax_source == "nhc"
    # The only point is ~80 nm from the eye and R64 is 40 nm, so it is
    # outside the drawn field and the county is not listed.
    assert impacts == []


def test_swath_between_fixes_catches_the_core(monkeypatch) -> None:
    """The eye passes over the county between two fixes. That is the core."""
    import app.services.hurricane_impact as hi

    # Fixes 40 nm north and south of the county. Midpoint is the centroid.
    county = CountyMeta(
        geoid="12071",
        geography_id="US-FL-12071",
        name="Lee",
        state_usps="FL",
        centroid_lat=26.5,
        centroid_lon=-82.0,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"12071": county})
    monkeypatch.setattr(hi, "county_area_samples", lambda: {})
    storm = Storm(
        storm_id="AL982026",
        name="SWATH",
        year=2026,
        track=[
            TrackPoint(
                datetime_utc="2026-10-09T12:00:00Z",
                record_id="",
                status="HU",
                lat=25.8,
                lon=-82.0,
                wind_kt=100,
                pressure_mb=950,
                rmax_nm=15.0,
                r64_quads_nm=(45.0, 45.0, 45.0, 45.0),
                radii_source="nhc",
            ),
            TrackPoint(
                datetime_utc="2026-10-09T18:00:00Z",
                record_id="L",
                status="HU",
                lat=27.2,
                lon=-82.0,
                wind_kt=100,
                pressure_mb=952,
                rmax_nm=15.0,
                r64_quads_nm=(45.0, 45.0, 45.0, 45.0),
                radii_source="nhc",
            ),
        ],
    )
    impacts, *_ = compute_impact(storm)
    assert impacts
    # ~42 nm to either vertex (outside Rmax 15). The filled track crosses the point.
    assert impacts[0].max_wind_kt == 100


def _ts_storm() -> Storm:
    # Historical tropical storm: 55 kt, best-track R34 80 nm, R50 25 nm.
    return Storm(
        storm_id="AL972020",
        name="SKIRT",
        year=2020,
        track=[
            TrackPoint(
                datetime_utc="2020-08-04T00:00:00Z",
                record_id="L",
                status="TS",
                lat=34.0,
                lon=-78.0,
                wind_kt=55,
                pressure_mb=995,
                r50_quads_nm=(25.0, 25.0, 0.0, 0.0),
                r34_quads_nm=(80.0, 80.0, 60.0, 60.0),
            )
        ],
    )


def _stub_county(monkeypatch, lat: float, lon: float) -> None:
    import app.services.hurricane_impact as hi

    county = CountyMeta(
        geoid="37129",
        geography_id="US-NC-37129",
        name="New Hanover",
        state_usps="NC",
        centroid_lat=lat,
        centroid_lon=lon,
    )
    monkeypatch.setattr(hi, "county_centroids", lambda: {"37129": county})
    monkeypatch.setattr(hi, "county_area_samples", dict)
    monkeypatch.setattr(hi, "rmax_nm", lambda *a, **k: (20.0, "willoughby"))
    monkeypatch.setattr(hi, "lookup_r64_nm", lambda *a: None)
    monkeypatch.setattr(hi, "lookup_r64_quads_nm", lambda *a: None)


def test_historical_tropical_storm_paints_its_34kt_field_only_when_asked(monkeypatch) -> None:
    # ~40 nm NE of the centre: inside R34, outside R50.
    _stub_county(monkeypatch, 34.47, -77.43)
    impacts, footprint, *_ = compute_impact(_ts_storm())
    assert impacts == [] and footprint == []

    impacts, footprint, *_ = compute_impact(_ts_storm(), ts_field=True)
    assert footprint == []  # the drawn cone stays hurricane-force only
    assert len(impacts) == 1
    assert 34 <= impacts[0].max_wind_kt < 50


def test_historical_tropical_storm_is_never_given_a_hurricane_ring(monkeypatch) -> None:
    # At the centre a 55 kt storm cannot score hurricane-force wind.
    _stub_county(monkeypatch, 34.0, -78.0)
    impacts, *_ = compute_impact(_ts_storm(), ts_field=True)
    assert impacts and impacts[0].max_wind_kt <= 55


def test_virgin_islands_are_scored(monkeypatch) -> None:
    import app.services.hurricane_impact as hi

    # Only the three USVI districts; no us-atlas download.
    monkeypatch.setattr(hi.urllib.request, "urlopen", _empty_topo)
    hi.county_centroids.cache_clear()
    try:
        centroids = hi.county_centroids()
        assert {"78010", "78020", "78030"} <= set(centroids)
        assert centroids["78010"].geography_id == "US-VI-78010"
        assert hi.county_area_samples()["78030"]
    finally:
        hi.county_centroids.cache_clear()


class _EmptyTopo:
    def __enter__(self):
        import io
        import json

        return io.StringIO(json.dumps({
            "arcs": [], "transform": {"scale": [1, 1], "translate": [0, 0]},
            "objects": {"counties": {"geometries": []}},
        }))

    def __exit__(self, *exc) -> None:
        return None


def _empty_topo(*_a, **_k) -> _EmptyTopo:
    return _EmptyTopo()
