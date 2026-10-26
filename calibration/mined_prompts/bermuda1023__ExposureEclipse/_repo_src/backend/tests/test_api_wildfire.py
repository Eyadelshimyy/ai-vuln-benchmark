"""Tests for the live wildfire endpoint + service.

Network is mocked (monkeypatched ``urlopen`` / fetchers) so these are
deterministic in CI. One optional integration check hits the real WFIGS
service only when RUN_LIVE_WILDFIRE=1 is set.
"""

from __future__ import annotations

import io
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import live_wildfire

client = TestClient(app)


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *a) -> None:
        return None


_FAKE_PERIMETER_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "id": 1,
            "geometry": {"type": "Polygon", "coordinates": [[[-120, 39], [-120, 40], [-119, 40], [-119, 39], [-120, 39]]]},
            "properties": {
                "poly_IncidentName": "Big Burn",
                "poly_GISAcres": 50000.0,
                "poly_IRWINID": "{IRWIN-1}",
                "poly_DateCurrent": 1_726_000_000_000,
                "attr_IncidentSize": 50000.0,
                "attr_PercentContained": 25.0,
                "attr_FireCause": "Natural",
                "attr_FireDiscoveryDateTime": 1_725_000_000_000,
                "attr_POOState": "US-CA",
                "attr_IncidentTypeCategory": "WF",
            },
        },
        {
            # Prescribed burn — must be filtered out.
            "id": 2,
            "geometry": {"type": "Polygon", "coordinates": [[[-121, 38], [-121, 39], [-120, 39], [-120, 38], [-121, 38]]]},
            "properties": {
                "poly_IncidentName": "Planned RX",
                "poly_GISAcres": 10.0,
                "attr_IncidentTypeCategory": "RX",
                "attr_POOState": "US-NV",
            },
        },
    ],
}


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    live_wildfire._PERIM_CACHE.clear()
    live_wildfire._FIRMS_CACHE.clear()


def test_perimeter_parse_filters_prescribed_and_maps_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_wildfire.urllib.request,
        "urlopen",
        lambda req, timeout=0: _FakeResp(json.dumps(_FAKE_PERIMETER_GEOJSON).encode()),
    )
    perims, note = live_wildfire.fetch_perimeters(bbox=(-125, 32, -114, 42))
    assert note is None
    assert len(perims) == 1  # RX dropped
    fp = perims[0]
    assert fp.name == "Big Burn"
    assert fp.gis_acres == 50000.0
    assert fp.percent_contained == 25.0
    assert fp.state == "CA"  # US- prefix stripped
    assert fp.discovery_at and fp.discovery_at.endswith("Z")
    assert fp.geometry["type"] == "Polygon"


def test_firms_disabled_without_key_returns_note() -> None:
    fires, note = live_wildfire.fetch_active_fires(map_key=None, bbox=(-125, 32, -114, 42))
    assert fires == []
    assert note and "FIRMS_MAP_KEY" in note


def test_firms_parses_viirs_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,frp,daynight\n"
        "39.5,-120.2,330.1,0.4,0.4,2026-08-06,2112,N,h,12.3,D\n"
    )
    monkeypatch.setattr(live_wildfire, "FIRMS_SOURCES", ("VIIRS_SNPP_NRT",))
    monkeypatch.setattr(
        live_wildfire.urllib.request,
        "urlopen",
        lambda req, timeout=0: _FakeResp(csv_text.encode()),
    )
    fires, note = live_wildfire.fetch_active_fires(map_key="fake-key", bbox=(-125, 32, -114, 42))
    assert note is None
    assert len(fires) == 1
    a = fires[0]
    assert a.lat == 39.5 and a.lon == -120.2
    assert a.brightness_k == 330.1
    assert a.frp_mw == 12.3
    assert a.acquired_at == "2026-08-06T21:12:00Z"


def test_endpoint_shape_and_rollup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_wildfire.urllib.request,
        "urlopen",
        lambda req, timeout=0: _FakeResp(json.dumps(_FAKE_PERIMETER_GEOJSON).encode()),
    )
    r = client.get("/api/wildfire/active", params={"bbox": "-125,32,-114,42", "includeHeat": "false"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["perimeters"]["type"] == "FeatureCollection"
    assert j["counts"]["perimeters"] == 1
    assert j["affectedStates"] == [{"state": "CA", "fireCount": 1, "acres": 50000.0}]
    assert "generatedAt" in j and "attribution" in j


def test_cluster_heat_shapes_builds_hulls() -> None:
    from app.services.live_wildfire import ActiveFire, cluster_heat_shapes

    def fire(lat: float, lon: float, frp: float) -> ActiveFire:
        return ActiveFire(lat=lat, lon=lon, brightness_k=330.0, frp_mw=frp,
                          confidence="h", satellite="N", source="VIIRS_SNPP_NRT",
                          acquired_at="2026-08-05T21:00:00Z")

    # One dense cluster (>= min points, spread over a few cells) + one isolated
    # point that must NOT form a shape.
    fires = [fire(39.0 + i * 0.01, -120.0 + j * 0.01, 10.0 + i)
             for i in range(4) for j in range(4)]
    fires.append(fire(45.0, -110.0, 5.0))  # loner
    shapes = cluster_heat_shapes(fires, grid_deg=0.02, min_points=5)
    assert len(shapes) == 1
    s = shapes[0]
    assert s.detection_count == 16
    assert s.geometry["type"] in ("Polygon", "MultiPolygon")  # occupied-cell footprint
    assert s.max_frp_mw is not None


def test_endpoint_exposes_heat_shapes_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_wildfire.urllib.request,
        "urlopen",
        lambda req, timeout=0: _FakeResp(json.dumps(_FAKE_PERIMETER_GEOJSON).encode()),
    )
    r = client.get("/api/wildfire/active", params={"bbox": "-125,32,-114,42", "includeHeat": "false"})
    j = r.json()
    assert j["heatShapes"]["type"] == "FeatureCollection"
    assert "heatShapes" in j["counts"] and "activeFiresTotal" in j["counts"]


def test_small_hotspots_cleaned_by_cell_and_count() -> None:
    """A persistent single-cell source (factory) and an isolated one-off must
    be dropped; a spreading multi-cell cluster survives."""
    from app.services.live_wildfire import ActiveFire, cluster_heat_shapes

    def f(lat: float, lon: float) -> ActiveFire:
        return ActiveFire(lat=lat, lon=lon, brightness_k=330.0, frp_mw=20.0,
                          confidence="h", satellite="N", source="VIIRS_SNPP_NRT",
                          acquired_at="2026-08-05T21:00:00Z")

    fires = [f(39.0 + i * 0.01, -120.0 + j * 0.01) for i in range(4) for j in range(4)]  # spread
    fires += [f(45.0, -110.0)] * 30          # factory: 30 hits, ONE cell
    fires += [f(30.0, -95.0)]                 # lone one-off
    shapes = cluster_heat_shapes(fires, grid_deg=0.02, min_points=4, min_cells=2)
    assert len(shapes) == 1  # only the spread cluster; factory (1 cell) + loner dropped


def _signed_area(ring: list) -> float:
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return s / 2.0


def _polys_of(geom: dict) -> list:
    return geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]


@pytest.mark.parametrize(
    "name,cells",
    [
        # Two cells touching only at a corner: the shared lattice vertex used to
        # make ray-casting flip a coin and file one lobe as a hole.
        ("corner_touch", [(0, 0), (1, 1)]),
        ("diagonal_fire_front", [(i, i) for i in range(8)]),
        ("solid_block", [(i, j) for i in range(3) for j in range(3)]),
        ("donut", [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]),
        ("island_in_lake",
         [(i, j) for i in range(7) for j in range(7)
          if not (2 <= i <= 4 and 2 <= j <= 4)] + [(3, 3)]),
        ("two_blobs_two_holes",
         [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
         + [(i + 10, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]),
    ],
)
def test_footprint_never_drops_burned_cells(name: str, cells: list) -> None:
    """Every burned cell must land inside the traced footprint, and the emitted
    rings must be valid GeoJSON (CCW outer, CW holes, closed, >=4 points).

    Regression: interior rings were nested by ray-casting a ring vertex, but
    every vertex lies on the grid lattice where point-in-polygon is undefined.
    Diagonally-adjacent cells are ordinary along a fire front, and misfiling
    them as holes silently removed their TIV from the exposure rollup.
    """
    from app.services.live_wildfire import ActiveFire, FOOTPRINT_GRID_DEG, _footprint_geometry
    from app.services.wildfire_exposure import point_in_geometry

    g = FOOTPRINT_GRID_DEG
    pts = [
        ActiveFire(lat=(j + 0.5) * g, lon=(i + 0.5) * g, brightness_k=330.0, frp_mw=10.0,
                   confidence="h", satellite="N", source="VIIRS_SNPP_NRT",
                   acquired_at="2026-08-05T21:00:00Z")
        for i, j in cells
    ]
    geom = _footprint_geometry(pts, g)
    assert geom is not None

    for a in pts:
        assert point_in_geometry(a.lon, a.lat, geom), f"{name}: burned cell fell outside footprint"

    for poly in _polys_of(geom):
        assert _signed_area(poly[0]) > 0, f"{name}: exterior ring must be CCW"
        for hole in poly[1:]:
            assert _signed_area(hole) < 0, f"{name}: hole ring must be CW"
        for ring in poly:
            assert len(ring) >= 4 and ring[0] == ring[-1], f"{name}: ring must be closed"


def test_footprint_holes_are_genuinely_unburned() -> None:
    """An unburned pocket must be excluded, but an island inside that pocket
    must be included again."""
    from app.services.live_wildfire import ActiveFire, FOOTPRINT_GRID_DEG, _footprint_geometry
    from app.services.wildfire_exposure import point_in_geometry

    g = FOOTPRINT_GRID_DEG

    def build(cells: list) -> dict:
        return _footprint_geometry(
            [ActiveFire(lat=(j + 0.5) * g, lon=(i + 0.5) * g, brightness_k=330.0, frp_mw=10.0,
                        confidence="h", satellite="N", source="VIIRS_SNPP_NRT",
                        acquired_at="2026-08-05T21:00:00Z")
             for i, j in cells],
            g,
        )

    donut = build([(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)])
    assert not point_in_geometry(1.5 * g, 1.5 * g, donut)

    nested = build([(i, j) for i in range(7) for j in range(7)
                    if not (2 <= i <= 4 and 2 <= j <= 4)] + [(3, 3)])
    assert point_in_geometry(3.5 * g, 3.5 * g, nested)       # the island
    assert not point_in_geometry(2.5 * g, 2.5 * g, nested)   # the lake around it


def test_confidence_rank() -> None:
    from app.services.live_wildfire import _confidence_rank
    assert _confidence_rank("l") == 0
    assert _confidence_rank("nominal") == 1
    assert _confidence_rank("h") == 2
    assert _confidence_rank("85") == 2   # MODIS numeric
    assert _confidence_rank("10") == 0


def test_firms_windows_chain_beyond_5_days() -> None:
    from app.services.live_wildfire import _firms_windows
    w = _firms_windows(12)
    assert sum(chunk for chunk, _ in w) == 12
    assert all(chunk <= 5 for chunk, _ in w)
    assert w[0][1] is None  # most-recent window omits the date


def test_point_in_geometry() -> None:
    from app.services.wildfire_exposure import point_in_geometry
    sq = {"type": "Polygon", "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]}
    assert point_in_geometry(0, 0, sq) is True
    assert point_in_geometry(5, 5, sq) is False


def test_exposure_endpoint_rolls_up_by_client() -> None:
    # Big box over Florida — synthetic locations from county TIV should roll up.
    fl = {"type": "Polygon", "coordinates": [[[-82.5, 26.5], [-80.0, 26.5], [-80.0, 28.5], [-82.5, 28.5], [-82.5, 26.5]]]}
    r = client.post("/api/wildfire/exposure", json={"polygons": [{"id": "t1", "name": "Test", "geometry": fl}]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["synthetic"] is True and "synthetic" in j["note"].lower()
    res = j["results"][0]
    assert res["totalTiv"] > 0
    assert len(res["byClient"]) >= 1
    assert res["byClient"][0]["tiv"] >= res["byClient"][-1]["tiv"]  # sorted desc


def test_arcgis_error_body_is_not_cached_as_no_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """ArcGIS answers a rejected query with HTTP 200 + an error body. Treating
    that as an empty feature list cached a blank map for the full TTL."""
    monkeypatch.setattr(
        live_wildfire.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeResp(
            json.dumps({"error": {"code": 400, "message": "Invalid field"}}).encode()
        ),
    )
    perims, note = live_wildfire.fetch_perimeters(bbox=(-125, 32, -114, 42))
    assert perims == []
    assert note and "rejected" in note.lower()
    assert not live_wildfire._PERIM_CACHE


def test_perimeter_truncation_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = dict(_FAKE_PERIMETER_GEOJSON)
    payload["exceededTransferLimit"] = True
    monkeypatch.setattr(
        live_wildfire.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeResp(json.dumps(payload).encode()),
    )
    _perims, note = live_wildfire.fetch_perimeters(bbox=(-125, 32, -114, 42))
    assert note and "truncated" in note.lower()

    # The warning has to survive the cache. Caching only the perimeters meant
    # every caller for the next 10 minutes got the short list with no warning.
    _p2, note2 = live_wildfire.fetch_perimeters(bbox=(-125, 32, -114, 42))
    assert note2 == note


def test_partial_firms_failure_is_noted_and_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short read understates every footprint and every exposed-TIV figure
    built from it, so it must be flagged and must not be cached."""
    csv_body = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "confidence,version,bright_ti5,frp,daynight\n"
        "39.1,-120.2,330.0,0.4,0.4,2026-08-05,2100,N,h,2.0NRT,290.0,15.5,D\n"
    )
    calls = {"n": 0}

    def flaky(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp(csv_body.encode())
        raise OSError("upstream timeout")

    monkeypatch.setattr(live_wildfire.urllib.request, "urlopen", flaky)
    fires, note = live_wildfire.fetch_active_fires(map_key="k" * 20, day_range=15)
    assert fires, "the one good window's detections should still come back"
    assert note and "incomplete" in note.lower()
    assert not live_wildfire._FIRMS_CACHE


def test_firms_quota_text_body_is_a_failure_not_zero_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIRMS returns HTTP 200 with plain text when a key is over quota; that
    must not read as 'no fires burning'."""
    monkeypatch.setattr(
        live_wildfire.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeResp(
            b"You have exceeded your transaction limit. Please try again later."
        ),
    )
    fires, note = live_wildfire.fetch_active_fires(map_key="k" * 20, day_range=1)
    assert fires == []
    assert note and "unavailable" in note.lower()


def test_caches_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_wildfire.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeResp(json.dumps(_FAKE_PERIMETER_GEOJSON).encode()),
    )
    # Cache keys embed caller-supplied query params, so an unbounded dict lets a
    # caller mint entries until the process runs out of memory.
    for i in range(live_wildfire._CACHE_MAX_ENTRIES * 2):
        live_wildfire.fetch_perimeters(bbox=(-125, 32, -114, 42), simplify_deg=0.001 + i * 1e-6)
    assert len(live_wildfire._PERIM_CACHE) <= live_wildfire._CACHE_MAX_ENTRIES


def test_exposure_takes_max_across_perils_and_years_not_sum() -> None:
    """CLAUDE.md rules 3+4: a cedent carrying WS+EQ+CS, or renewing the same
    slot year over year, must not have those TIVs added together."""
    import json as _json
    from app.services import wildfire_exposure as we

    we._load_locations.cache_clear()
    md = we._mock_dir()
    datasets = _json.loads((md / "datasets.json").read_text(encoding="utf-8"))

    # Oracle keyed at the SEGMENT grain (client, county, dataset, peril). It
    # must NOT share the production key shape — an earlier version of this test
    # keyed on (client, county, peril), which is exactly the grain the bug used,
    # so it reproduced the defect as its own expectation and passed while the
    # code summed two treaty years together.
    seg: dict[tuple[str, str, str, str], float] = {}
    for ds in datasets:
        if not ds.get("isIncludedInPortfolio"):
            continue
        fact_plain = md / "exposure_facts" / f"{ds['datasetId']}.json"
        fact_packed = fact_plain.with_name(fact_plain.name + ".gz")
        if fact_plain.exists():
            rows = _json.loads(fact_plain.read_text(encoding="utf-8"))
        else:
            import gzip

            with gzip.open(fact_packed, "rt", encoding="utf-8") as handle:
                rows = _json.load(handle)
        for r in rows:
            if r.get("aggregation") != "COUNTY":
                continue
            gid = (r.get("geographyId") or "").split("-")[-1]
            if not gid or not r.get("tiv"):
                continue
            key = (ds["cedentName"], gid, ds["datasetId"], r.get("peril") or "UNKNOWN")
            seg[key] = seg.get(key, 0.0) + float(r["tiv"])

    def segs(client: str, geoid: str) -> list[float]:
        return [v for k, v in seg.items() if k[0] == client and k[1] == geoid]

    # The fixture must still exercise BOTH collision kinds or this proves
    # nothing: one cedent+county carrying several perils, and one carrying the
    # same peril across two treaty years.
    multi_peril = any(
        len({kk[3] for kk in seg if kk[0] == k[0] and kk[1] == k[1]}) > 1 for k in seg
    )
    multi_year = any(
        len({kk[2] for kk in seg
             if kk[0] == k[0] and kk[1] == k[1] and kk[3] == k[3]}) > 1
        for k in seg
    )
    assert multi_peril, "fixture no longer exercises the multi-peril case"
    assert multi_year, "fixture no longer exercises the same-peril-two-years case"

    client_name = "Farmers Group"
    # Pin the worst known case outright. Farmers Group renews WS on Miami-Dade
    # (12086) year over year; summing those two years reported 13.20bn against
    # a true max of 8.00bn — a 1.65x overstatement straight into the XOL calc.
    assert len(segs(client_name, "12086")) > 1
    assert sum(segs(client_name, "12086")) > max(segs(client_name, "12086"))

    counties = {k[1] for k in seg if k[0] == client_name}
    expected = sum(max(segs(client_name, g)) for g in counties)
    summed = sum(sum(segs(client_name, g)) for g in counties)
    # Guards the assertion below against passing by coincidence.
    assert summed > expected

    loaded = sum(
        loc.tiv for loc in we._load_locations().locations if loc.client == client_name
    )
    assert loaded == pytest.approx(expected, rel=1e-9)


def test_combined_does_not_double_count_overlapping_polygons() -> None:
    fl = {"type": "Polygon",
          "coordinates": [[[-82.5, 26.5], [-80.0, 26.5], [-80.0, 28.5],
                           [-82.5, 28.5], [-82.5, 26.5]]]}
    body = {"polygons": [{"id": "a", "geometry": fl}, {"id": "b", "geometry": fl}]}
    j = client.post("/api/wildfire/exposure", json=body).json()
    assert j["results"][0]["totalTiv"] > 0
    # Same polygon twice: the union must equal one of them, not their sum.
    assert j["combined"]["totalTiv"] == pytest.approx(j["results"][0]["totalTiv"])
    assert j["combined"]["locationCount"] == j["results"][0]["locationCount"]


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Polygon", "coordinates": [[[0, 0], [1e9, 0], [1e9, 1e9], [0, 0]]]},
        {"type": "Polygon", "coordinates": "abc"},
        {"type": "Polygon", "coordinates": [[["a", "b"], ["c", "d"], ["e", "f"], ["a", "b"]]]},
        {"type": "Point", "coordinates": [0, 0]},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]},
        {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
    ],
)
def test_exposure_rejects_hostile_geometry(geometry: dict) -> None:
    """Unbounded coordinates drove a 4e18-iteration grid walk, and non-numeric
    coordinates recursed to death in the bbox walk. Both must be 422s."""
    r = client.post("/api/wildfire/exposure",
                    json={"polygons": [{"id": "x", "geometry": geometry}]})
    assert r.status_code == 422, r.text


def test_exposure_rejects_oversized_requests() -> None:
    sq = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    too_many = {"polygons": [{"id": str(i), "geometry": sq} for i in range(51)]}
    assert client.post("/api/wildfire/exposure", json=too_many).status_code == 422

    ring = [[i * 1e-6, 0.0] for i in range(60_000)] + [[0.0, 1.0], [0.0, 0.0]]
    fat = {"type": "Polygon", "coordinates": [ring]}
    body = {"polygons": [{"id": "a", "geometry": fat}, {"id": "b", "geometry": fat}]}
    assert client.post("/api/wildfire/exposure", json=body).status_code == 422


def test_exposure_rejects_expensive_but_wellformed_geometry() -> None:
    """Per-field validation is not enough: a CONUS-wide ring within the vertex
    cap sweeps every synthetic location, and measured 64s against a 30s
    function budget. The (candidates x vertices) budget must reject it."""
    import math
    n = 20_000
    ring = [
        [-125 + 59 * (0.5 + 0.5 * math.cos(2 * math.pi * i / n)),
         24 + 26 * (0.5 + 0.5 * math.sin(2 * math.pi * i / n))]
        for i in range(n)
    ]
    ring.append(ring[0])
    body = {"polygons": [{"id": "a", "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    started = time.monotonic()
    r = client.post("/api/wildfire/exposure", json=body)
    assert r.status_code == 422, r.text
    # The rejection must be cheap — bailing out after doing the work is no fix.
    assert time.monotonic() - started < 5.0


def test_heat_shape_ids_follow_the_footprint_not_the_list_position() -> None:
    """Shapes are sorted by detection count and the layer refetches every few
    minutes. A positional id re-points at a different fire between responses,
    so clicking a shape deselects an unrelated one and the combined TIV moves
    for no visible reason."""
    from app.api.wildfire import _shape_id

    a = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    b = {"type": "Polygon", "coordinates": [[[5, 5], [6, 5], [6, 6], [5, 5]]]}
    assert _shape_id(a) == _shape_id(dict(a))
    assert _shape_id(a) != _shape_id(b)


def test_endpoint_rejects_bad_bbox() -> None:
    r = client.get("/api/wildfire/active", params={"bbox": "1,2,3"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.skipif(os.environ.get("RUN_LIVE_WILDFIRE") != "1", reason="live network test")
def test_live_wfigs_reachable() -> None:
    perims, _note = live_wildfire.fetch_perimeters(bbox=(-125, 32, -114, 42))
    assert isinstance(perims, list)
