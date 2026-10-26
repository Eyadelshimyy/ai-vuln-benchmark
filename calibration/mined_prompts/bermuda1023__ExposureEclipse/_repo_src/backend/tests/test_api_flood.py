"""Live flood overlay — endpoint + service tests.

Every test drives a fixture rather than the network. That is not just for
determinism: on a quiet day the live feed carries no moderate-or-worse
flooding at all (a probe on 2026-08-08 found 406 gauges with a single `action`
and zero minor/moderate/major), so the severity paths simply cannot be
exercised against production data.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import json

from app.main import app
from app.services import live_flood, nwm_inundation
from app.services.weather_alerts import AlertFeedUnavailable, WeatherAlert

client = TestClient(app)


def _alert(
    *,
    aid: str = "urn:oid:1",
    event: str = "Flash Flood Warning",
    severity: str = "Severe",
    area: str = "Houston, TN; Stewart, TN",
    geometry: dict | None = None,
) -> WeatherAlert:
    return WeatherAlert(
        alert_id=aid,
        event=event,
        headline=f"{event} for somewhere",
        severity=severity,
        urgency="Immediate",
        certainty="Observed",
        sent_at="2026-08-08T12:00:00Z",
        expires_at="2026-08-08T18:00:00Z",
        areas_affected=area,
        geometry=geometry,
    )


def _box(w: float, s: float, e: float, n: float) -> dict:
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


@pytest.fixture(autouse=True)
def _clear_cache():
    """The flood cache is module-level; without this a fixture from one test
    would satisfy the next test's request and mask a real regression."""
    live_flood._CACHE.clear()
    yield
    live_flood._CACHE.clear()


def _patch(monkeypatch, alerts: list[WeatherAlert]) -> dict[str, int]:
    calls = {"n": 0}

    def fake(*, bbox=None, states=None, event_filter=None):
        calls["n"] += 1
        # Mirror the real client's contract: it applies the event filter itself.
        return [a for a in alerts if event_filter is None or a.event in event_filter]

    monkeypatch.setattr(live_flood, "fetch_active_alerts", fake)
    return calls


def test_active_returns_only_polygon_bearing_alerts(monkeypatch) -> None:
    _patch(monkeypatch, [
        _alert(aid="a", geometry=_box(-90, 35, -89, 36)),
        _alert(aid="b", event="Coastal Flood Advisory", severity="Minor", geometry=None),
    ])
    r = client.get("/api/flood/active")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["counts"] == {"alerts": 1, "zoneOnly": 1}
    assert [f["properties"]["alertId"] for f in j["alerts"]["features"]] == ["a"]
    # A zone-coded alert that vanished with no explanation would understate the
    # event; the count alone is not enough, the user needs the reason.
    assert any("zone-coded" in n for n in j["notes"])


def test_min_severity_filters_below_the_floor(monkeypatch) -> None:
    _patch(monkeypatch, [
        _alert(aid="sev", severity="Severe", geometry=_box(-90, 35, -89, 36)),
        _alert(aid="min", event="Flood Advisory", severity="Minor",
               geometry=_box(-80, 30, -79, 31)),
    ])
    both = client.get("/api/flood/active?minSeverity=Unknown").json()
    assert both["counts"]["alerts"] == 2

    live_flood._CACHE.clear()
    severe = client.get("/api/flood/active?minSeverity=Severe").json()
    assert [f["properties"]["alertId"] for f in severe["alerts"]["features"]] == ["sev"]
    # Filtered-out alerts must not resurface as "zone only" — that count means
    # "had no geometry", and conflating the two would misreport the feed.
    assert severe["counts"]["zoneOnly"] == 0


def test_severity_rank_accompanies_severity(monkeypatch) -> None:
    """The map ramp interpolates on the numeric twin, so it has to agree with
    the string the panel displays."""
    _patch(monkeypatch, [
        _alert(aid="a", severity="Extreme", geometry=_box(-90, 35, -89, 36)),
    ])
    p = client.get("/api/flood/active").json()["alerts"]["features"][0]["properties"]
    assert p["severity"] == "Extreme"
    assert p["severityRank"] == live_flood.SEVERITY_RANK["Extreme"] == 4


def test_affected_states_parsed_from_area_description(monkeypatch) -> None:
    _patch(monkeypatch, [
        _alert(aid="a", area="Houston, TN; Stewart, TN", geometry=_box(-90, 35, -89, 36)),
        _alert(aid="b", area="Baxter, AR", geometry=_box(-92, 36, -91, 37)),
    ])
    states = client.get("/api/flood/active").json()["affectedStates"]
    assert {s["state"]: s["alertCount"] for s in states} == {"TN": 1, "AR": 1}


def test_area_description_without_state_code_is_ignored(monkeypatch) -> None:
    """areaDesc is free text. A trailing token that isn't a 2-letter code must
    not become a phantom state in the roll-up."""
    _patch(monkeypatch, [
        _alert(aid="a", area="The Florida Keys", geometry=_box(-82, 24, -81, 25)),
    ])
    assert client.get("/api/flood/active").json()["affectedStates"] == []


def test_repeat_request_is_served_from_cache(monkeypatch) -> None:
    calls = _patch(monkeypatch, [_alert(aid="a", geometry=_box(-90, 35, -89, 36))])
    client.get("/api/flood/active")
    client.get("/api/flood/active")
    assert calls["n"] == 1, "second identical request should not re-hit NWS"


def test_cached_list_still_honours_each_severity_floor(monkeypatch) -> None:
    """The floor filters the cached list rather than keying it, so switching
    chips must not re-fetch — but must also never serve the Severe-only list to
    an unfiltered caller, which would silently hide live alerts."""
    calls = _patch(monkeypatch, [
        _alert(aid="sev", severity="Severe", geometry=_box(-90, 35, -89, 36)),
        _alert(aid="min", event="Flood Advisory", severity="Minor",
               geometry=_box(-80, 30, -79, 31)),
    ])
    assert client.get("/api/flood/active?minSeverity=Severe").json()["counts"]["alerts"] == 1
    assert client.get("/api/flood/active?minSeverity=Unknown").json()["counts"]["alerts"] == 2
    assert calls["n"] == 1, "changing the floor should re-filter, not re-fetch"


def test_upstream_outage_is_not_reported_as_no_flooding(monkeypatch) -> None:
    """An empty list from a dead feed is indistinguishable from a quiet day.
    Reporting it as 'no active flooding' is a confident all-clear during an
    event, so the outage must be stated and must never be cached."""
    def boom(*, bbox=None, states=None, event_filter=None):
        raise AlertFeedUnavailable("connection reset")

    monkeypatch.setattr(live_flood, "fetch_active_alerts", boom)
    j = client.get("/api/flood/active").json()
    assert j["counts"] == {"alerts": 0, "zoneOnly": 0}
    assert any("could not be reached" in n for n in j["notes"])
    assert not any("No active flood alerts" in n for n in j["notes"])
    assert live_flood._CACHE == {}, "an outage must not be cached as an all-clear"


def test_validation_rejects_bad_bbox_and_severity() -> None:
    assert client.get("/api/flood/active?bbox=1,2,3").status_code == 422
    assert client.get("/api/flood/active?bbox=10,0,5,9").status_code == 422
    assert client.get("/api/flood/active?minSeverity=Catastrophic").status_code == 422


# ─────────────────────── exposed TIV ───────────────────────


def test_exposure_union_does_not_double_count_overlap() -> None:
    """Adjacent flood warnings routinely overlap. The union has to count each
    location once, or the combined TIV feeding the XOL calc is inflated."""
    poly = _box(-83.0, 25.0, -80.0, 28.0)  # over Florida, hits synthetic locations
    body = {"polygons": [
        {"id": "a", "name": "Flood Warning", "geometry": poly},
        {"id": "b", "name": "Flood Warning (dup)", "geometry": poly},
    ]}
    r = client.post("/api/flood/exposure", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    single = j["results"][0]["totalTiv"]
    assert single > 0, "fixture bbox no longer covers any synthetic locations"
    assert j["results"][1]["totalTiv"] == single
    assert j["combined"]["totalTiv"] == pytest.approx(single, rel=1e-9)
    assert j["combined"]["locationCount"] == j["results"][0]["locationCount"]


def test_exposure_flags_the_upper_bound_caveat() -> None:
    """Alert polygons are warning areas, not observed water. Reporting exposed
    TIV without saying so would overstate precision to an underwriter."""
    body = {"polygons": [{"id": "a", "geometry": _box(-83.0, 25.0, -80.0, 28.0)}]}
    j = client.post("/api/flood/exposure", json=body).json()
    assert j["synthetic"] is True
    assert "upper bound" in j["note"].lower()
    assert j["currency"]


def test_exposure_rejects_oversized_and_malformed_geometry() -> None:
    """The shared geometry guard must apply on this route too — it is the same
    grid walk the wildfire endpoint protects against."""
    from app.api.geometry_input import MAX_POLYGONS

    too_many = {"polygons": [
        {"id": str(i), "geometry": _box(-83.0, 25.0, -80.0, 28.0)}
        for i in range(MAX_POLYGONS + 1)
    ]}
    assert client.post("/api/flood/exposure", json=too_many).status_code == 422

    off_earth = {"polygons": [{"id": "a", "geometry": _box(-999.0, 25.0, -80.0, 28.0)}]}
    assert client.post("/api/flood/exposure", json=off_earth).status_code == 422

    not_a_polygon = {"polygons": [{"id": "a", "geometry": {"type": "Point", "coordinates": [0, 0]}}]}
    assert client.post("/api/flood/exposure", json=not_a_polygon).status_code == 422


def test_exposure_budget_is_charged_per_request_not_per_polygon() -> None:
    """Per-polygon budgets compose badly: many polygons each just under the
    limit pass every per-field check and the vertex cap, but measured 69s
    against a 30s function budget. The request as a whole must be priced."""
    from app.services import wildfire_exposure

    # A CONUS-wide ring sweeps in every synthetic location, so cost is driven by
    # vertex count. Size one ring at ~1/4 of the budget: a single polygon is
    # comfortably legal, four together are not.
    n_locations = len(wildfire_exposure._load_locations().locations)
    verts = max(8, (wildfire_exposure._MAX_WORK // 4) // max(1, n_locations))
    ring = [[-125.0 + (i % 2) * 59.0, 24.0 + ((i // 2) % 2) * 26.0] for i in range(verts - 1)]
    big = {"type": "Polygon", "coordinates": [[[-125.0, 24.0], *ring, [-125.0, 24.0]]]}

    one = client.post("/api/flood/exposure", json={"polygons": [{"id": "a", "geometry": big}]})
    assert one.status_code == 200, "a single ring this size must still be allowed"

    many = {"polygons": [{"id": str(i), "geometry": big} for i in range(8)]}
    r = client.post("/api/flood/exposure", json=many)
    assert r.status_code == 422
    assert "GEOMETRY_TOO_COMPLEX" in r.text


# ─────────────────── modelled inundation (NWM) ───────────────────


@pytest.fixture(autouse=True)
def _clear_inundation_cache():
    nwm_inundation._CACHE.clear()
    # `_patch_nwm` replaces urlopen on the shared `urllib.request` module, which
    # is also how the county centroids behind the synthetic locations are
    # fetched. Warm them first so a cold cache doesn't try to parse an NWM
    # payload as TopoJSON — a failure that depends on test ordering.
    from app.services import wildfire_exposure
    wildfire_exposure._load_locations()
    yield
    nwm_inundation._CACHE.clear()


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, size: int | None = None) -> bytes:
        return self._body if size is None else self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _reach(coords: list, *, fid: int = 1, ref: str = "2026-08-08 20:00:00") -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": {"feature_id": fid, "streamflow_cfs": 1234.5, "reference_time": ref},
    }


def _patch_nwm(monkeypatch, payload: dict) -> dict[str, int]:
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(nwm_inundation.urllib.request, "urlopen", fake_urlopen)
    return calls


def _ring(w: float, s: float, e: float, n: float) -> list:
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _sliver(w: float, s: float, vertices: int) -> list:
    """A thin, vertex-heavy ring standing in for one real river reach.

    Live reaches carry 50–400 vertices each at the default generalisation, which
    is what makes a wide extent expensive; a 5-point box would not reproduce it.
    """
    half = max(2, vertices // 2)
    step = 0.0006 / half
    top = [[w + i * step, s + 0.0004] for i in range(half)]
    bottom = [[w + i * step, s] for i in range(half - 1, -1, -1)]
    return [*top, *bottom, top[0]]


def test_inundation_requires_a_bbox_and_caps_its_size() -> None:
    """A nationwide request truncates upstream, so an uncapped one would draw a
    partial extent and read as if that were all the water there is."""
    assert client.get("/api/flood/inundation").status_code == 422
    assert client.get("/api/flood/inundation?bbox=1,2,3").status_code == 422
    too_big = "-106,26,-88,36"  # 180 deg², well past MAX_BBOX_DEG2
    r = client.get(f"/api/flood/inundation?bbox={too_big}")
    assert r.status_code == 422
    assert "square degrees" in r.text


def test_inundation_truncation_warning_is_cached_with_the_payload(monkeypatch) -> None:
    """Caching the features but recomputing the caveat would serve a partial
    extent as complete for the rest of the TTL — the bug fixed in 6932561."""
    payload = {
        "type": "FeatureCollection",
        "features": [_reach([_ring(-95.5, 29.5, -95.4, 29.6)])],
        "exceededTransferLimit": True,
    }
    calls = _patch_nwm(monkeypatch, payload)

    first = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()
    second = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()

    assert calls["n"] == 1, "second identical request should not re-hit NOAA"
    assert first["truncated"] is second["truncated"] is True
    assert any("cut off" in n for n in second["notes"])


def test_inundation_hoists_reference_time_and_strips_repeated_fields(monkeypatch) -> None:
    """`reference_time` is identical on every feature; at 2000 features
    repeating it is pure payload, so it belongs on the bundle."""
    _patch_nwm(monkeypatch, {
        "type": "FeatureCollection",
        "features": [_reach([_ring(-95.5, 29.5, -95.4, 29.6)], fid=7)],
    })
    j = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()
    assert j["referenceTime"] == "2026-08-08 20:00:00"
    props = j["inundation"]["features"][0]["properties"]
    assert props == {"reachId": 7, "streamflowCfs": 1234.5}


def test_inundation_drops_features_without_usable_geometry(monkeypatch) -> None:
    _patch_nwm(monkeypatch, {
        "type": "FeatureCollection",
        "features": [
            _reach([_ring(-95.5, 29.5, -95.4, 29.6)]),
            {"type": "Feature", "geometry": None, "properties": {}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
             "properties": {}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []},
             "properties": {}},
        ],
    })
    j = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()
    assert j["counts"]["reaches"] == 1


def test_inundation_outage_is_not_reported_as_dry(monkeypatch) -> None:
    """Same reasoning as the alert feed: an empty map during a flood must not
    read as an all-clear."""
    def boom(req, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(nwm_inundation.urllib.request, "urlopen", boom)
    j = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()
    assert j["counts"]["reaches"] == 0
    assert any("NOT a statement" in n for n in j["notes"])
    assert nwm_inundation._CACHE == {}, "an outage must not be cached"


def test_arcgis_error_inside_a_200_body_is_treated_as_an_outage(monkeypatch) -> None:
    """ArcGIS reports failures in the body as well as by status code; parsing
    that as zero features would invent a dry map out of a server error."""
    _patch_nwm(monkeypatch, {"error": {"code": 500, "message": "Unable to complete"}})
    j = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()
    assert j["counts"]["reaches"] == 0
    assert any("NOT a statement" in n for n in j["notes"])


def test_inundation_exposure_fails_hard_when_the_model_is_down(monkeypatch) -> None:
    """The map route fails soft, this one must not: a zero exposed TIV feeds the
    XOL layer calc and is indistinguishable from a genuine zero."""
    def boom(req, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(nwm_inundation.urllib.request, "urlopen", boom)
    r = client.post("/api/flood/inundation/exposure",
                    json={"bbox": [-95.8, 29.4, -94.9, 30.2]})
    assert r.status_code == 503
    assert "UPSTREAM_UNAVAILABLE" in r.text


def test_inundation_exposure_dissolves_reaches_without_double_counting(monkeypatch) -> None:
    """A flood event is thousands of per-reach polygons, past the 50-polygon cap,
    and neighbouring reaches overlap. Combining them must count each location
    once or the TIV feeding the layer calc is inflated."""
    big = _ring(-83.0, 25.0, -80.0, 28.0)  # over Florida, hits synthetic locations
    _patch_nwm(monkeypatch, {
        "type": "FeatureCollection",
        "features": [_reach([big], fid=1), _reach([big], fid=2)],
    })
    j = client.post("/api/flood/inundation/exposure",
                    json={"bbox": [-83.0, 25.0, -80.0, 28.0]}).json()
    assert j["reaches"] == 2

    single = client.post("/api/flood/exposure", json={
        "polygons": [{"id": "a", "geometry": {"type": "Polygon", "coordinates": [big]}}]
    }).json()["combined"]
    assert single["totalTiv"] > 0, "fixture bbox no longer covers synthetic locations"
    assert j["combined"]["totalTiv"] == pytest.approx(single["totalTiv"], rel=1e-9)
    assert j["combined"]["locationCount"] == single["locationCount"]


def test_sub_resolution_extent_reports_zero_as_unmeasurable(monkeypatch) -> None:
    """Real reaches are river corridors ~1e-4 deg² while the synthetic locations
    sit ~0.1° apart, so a zero is a sampling artifact. Left unflagged an
    underwriter would read it as "no exposure here"."""
    from app.services import wildfire_exposure

    sliver = _ring(-82.0, 26.0, -81.999, 26.001)  # 1e-6 deg², far below the floor
    _patch_nwm(monkeypatch, {
        "type": "FeatureCollection", "features": [_reach([sliver])],
    })
    j = client.post("/api/flood/inundation/exposure",
                    json={"bbox": [-83.0, 25.0, -80.0, 28.0]}).json()
    assert j["combined"]["totalTiv"] == 0.0
    assert j["belowResolution"] is True
    assert any("below the resolution" in n for n in j["notes"])

    nwm_inundation._CACHE.clear()
    _patch_nwm(monkeypatch, {
        "type": "FeatureCollection",
        "features": [_reach([_ring(-83.0, 25.0, -80.0, 28.0)])],
    })
    wide = client.post("/api/flood/inundation/exposure",
                       json={"bbox": [-83.0, 25.0, -80.0, 28.0]}).json()
    assert wide["belowResolution"] is False
    assert wide["combined"]["totalTiv"] > 0

    # The floor is derived from the engine's own constants so the two can't drift.
    assert nwm_inundation.extent_area_deg2(
        {"type": "Polygon", "coordinates": [sliver]}
    ) < wildfire_exposure.resolution_deg2()


def test_a_widespread_extent_is_still_priceable(monkeypatch) -> None:
    """The regression that matters most: the work budget charges (candidates ×
    TOTAL vertices), so an ordinary wide extent — measured live at 1,871 reaches
    and ~200k vertices over the mid-Atlantic — costs 86M against a budget of 8M
    and 422s. That would fail on exactly the widespread floods the button exists
    for. Reaches holding no location are dropped first, which cannot move the
    answer because they contribute no index.
    """
    from app.services import wildfire_exposure

    big = _ring(-83.0, 25.0, -80.0, 28.0)
    # Every sliver lies INSIDE `big`, so the union is unchanged by construction
    # whether or not any of them happens to cover a synthetic location.
    noise = [_reach([_sliver(-82.5 + i * 0.002, 26.0, 220)], fid=100 + i)
             for i in range(300)]
    features = [_reach([big], fid=1), *noise]
    _patch_nwm(monkeypatch, {"type": "FeatureCollection", "features": features})

    undropped = nwm_inundation.dissolve([f["geometry"] for f in features])
    _, work = wildfire_exposure._cost(undropped)
    assert work > wildfire_exposure._MAX_WORK, (
        "fixture no longer reproduces the overrun it guards against"
    )

    r = client.post("/api/flood/inundation/exposure",
                    json={"bbox": [-83.0, 25.0, -80.0, 28.0]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["reaches"] == len(features)

    alone = client.post("/api/flood/exposure", json={
        "polygons": [{"id": "a", "geometry": {"type": "Polygon", "coordinates": [big]}}]
    }).json()["combined"]
    assert alone["totalTiv"] > 0, "fixture bbox no longer covers synthetic locations"
    assert j["combined"]["totalTiv"] == pytest.approx(alone["totalTiv"], rel=1e-9)
    assert j["combined"]["locationCount"] == alone["locationCount"]


def test_malformed_upstream_geometry_cannot_crash_the_rollup(monkeypatch) -> None:
    """This geometry comes off a third-party response and skips the request
    validator, but the engine walks coordinates structurally: a string recurses
    until the stack dies and a length-1 position raises IndexError. Both would
    surface as a 500 on a live flood."""
    _patch_nwm(monkeypatch, {"type": "FeatureCollection", "features": [
        _reach([_ring(-83.0, 25.0, -80.0, 28.0)]),
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": "abc"},
         "properties": {}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0]]]},
         "properties": {}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[["a", "b"]]]},
         "properties": {}},
        {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[[[0]]]]},
         "properties": {}},
    ]})
    j = client.get("/api/flood/inundation?bbox=-83.0,25.0,-80.0,28.0").json()
    assert j["counts"]["reaches"] == 1

    r = client.post("/api/flood/inundation/exposure",
                    json={"bbox": [-83.0, 25.0, -80.0, 28.0]})
    assert r.status_code == 200, r.text
    assert r.json()["combined"]["totalTiv"] > 0


def test_an_outage_is_distinguishable_from_a_dry_view(monkeypatch) -> None:
    """Both are zero reaches and a null reference time. If only the wording of
    `notes` separates them, a caller cannot branch and renders a bold 0 either
    way — which reads as "no water" during an outage."""
    def boom(req, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(nwm_inundation.urllib.request, "urlopen", boom)
    down = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()

    _patch_nwm(monkeypatch, {"type": "FeatureCollection", "features": []})
    dry = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()

    assert down["counts"] == dry["counts"] == {"reaches": 0}
    assert down["referenceTime"] is dry["referenceTime"] is None
    assert down["unavailable"] is True
    assert dry["unavailable"] is False


def test_an_oversized_upstream_body_is_refused(monkeypatch) -> None:
    """The body is read into memory before it is parsed, so an upstream that
    ignores its own feature cap must not be able to size this process."""
    monkeypatch.setattr(nwm_inundation, "MAX_RESPONSE_BYTES", 64)
    _patch_nwm(monkeypatch, {
        "type": "FeatureCollection",
        "features": [_reach([_ring(-95.5, 29.5, -95.4, 29.6)])],
    })
    j = client.get("/api/flood/inundation?bbox=-95.8,29.4,-94.9,30.2").json()
    assert j["unavailable"] is True
    assert j["counts"]["reaches"] == 0


def test_extent_area_subtracts_holes() -> None:
    """NOAA emits interior gaps as holes. Counting them as water would inflate
    the extent and, with it, the resolution judgement built on top of it."""
    donut = {"type": "Polygon", "coordinates": [
        _ring(0.0, 0.0, 1.0, 1.0), _ring(0.25, 0.25, 0.75, 0.75),
    ]}
    assert nwm_inundation.extent_area_deg2(donut) == pytest.approx(1.0 - 0.25)
    assert nwm_inundation.extent_area_deg2(None) == 0.0


def test_flood_events_exclude_non_flood_products() -> None:
    """The filter is what keeps a hurricane or winter storm off the flood map."""
    assert "Flash Flood Warning" in live_flood.FLOOD_EVENTS
    assert "Coastal Flood Warning" in live_flood.FLOOD_EVENTS
    for other in ("Hurricane Warning", "Tornado Warning", "Winter Storm Warning",
                  "Tsunami Warning", "Hydrologic Outlook"):
        assert other not in live_flood.FLOOD_EVENTS
