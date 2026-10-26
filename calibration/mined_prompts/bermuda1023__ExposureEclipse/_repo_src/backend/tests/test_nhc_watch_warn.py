"""Unit tests for the NHC watch/warning split + colouring + exposure endpoint.

Split logic is pure — exercised with hand-crafted WeatherAlert lists so we
don't need to reach out to NWS. The exposure endpoint is smoke-tested with
a small polygon over Florida to verify the wire shape + non-crash behaviour.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.nhc_watch_warn import (
    NHC_WW_COLORS,
    NHC_WW_FAMILY,
    NHC_WW_RANK,
    split_watches_warnings,
)
from app.services.weather_alerts import WeatherAlert


def _alert(event: str, geometry: dict | None = None) -> WeatherAlert:
    return WeatherAlert(
        alert_id=f"id-{event}",
        event=event,
        headline=f"{event} headline",
        severity="Severe",
        urgency="Immediate",
        certainty="Observed",
        sent_at="2026-09-15T12:00:00Z",
        expires_at="2026-09-15T18:00:00Z",
        areas_affected="Test County",
        geometry=geometry,
    )


def _tiny_polygon() -> dict:
    """A ~0.5° box over central Florida — inside the synthetic-locations plane
    so exposure_in_polygons has something to hit."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [-82.0, 27.5],
            [-81.5, 27.5],
            [-81.5, 28.0],
            [-82.0, 28.0],
            [-82.0, 27.5],
        ]],
    }


def test_split_moves_only_nhc_ww_events() -> None:
    alerts = [
        _alert("Hurricane Warning", _tiny_polygon()),
        _alert("Tropical Storm Watch", _tiny_polygon()),
        _alert("Storm Surge Warning", _tiny_polygon()),
        _alert("Flash Flood Warning", _tiny_polygon()),
        _alert("Tornado Warning", _tiny_polygon()),
    ]
    ww, residual = split_watches_warnings(alerts)
    assert {w.event for w in ww} == {
        "Hurricane Warning", "Tropical Storm Watch", "Storm Surge Warning",
    }
    assert {a.event for a in residual} == {"Flash Flood Warning", "Tornado Warning"}


def test_split_applies_nhc_colours_and_families() -> None:
    alerts = [_alert(evt) for evt in NHC_WW_FAMILY]
    ww, _ = split_watches_warnings(alerts)
    for w in ww:
        assert w.color == NHC_WW_COLORS[w.event]
        assert w.family == NHC_WW_FAMILY[w.event]
        assert w.rank == NHC_WW_RANK[w.event]


def test_split_sorts_by_rank_desc() -> None:
    # Order in the input shouldn't matter — output must be Warning > Watch
    # by NHC threat rank (Extreme Wind > Hurricane Warning > Storm Surge
    # Warning > Hurricane Watch > TS Warning / Surge Watch > TS Watch).
    alerts = [
        _alert("Tropical Storm Watch"),
        _alert("Hurricane Warning"),
        _alert("Extreme Wind Warning"),
        _alert("Storm Surge Watch"),
    ]
    ww, _ = split_watches_warnings(alerts)
    ranks = [w.rank for w in ww]
    assert ranks == sorted(ranks, reverse=True)
    assert ww[0].event == "Extreme Wind Warning"
    assert ww[1].event == "Hurricane Warning"


def test_split_preserves_zone_only_ww() -> None:
    # Zone-coded alerts arrive with geometry=None. They must still land in
    # the WW bucket (so the frontend can count them and surface the text)
    # rather than being silently dropped.
    alerts = [_alert("Hurricane Watch", geometry=None)]
    ww, _ = split_watches_warnings(alerts)
    assert len(ww) == 1
    assert ww[0].geometry is None


def test_ww_exposure_endpoint_smoke() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/live/watches-warnings/exposure",
        json={
            "polygons": [
                {
                    "id": "test-ww-1",
                    "name": "Hurricane Warning — Central FL",
                    "geometry": _tiny_polygon(),
                }
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["synthetic"] is True
    assert "upper bound" in body["note"].lower()
    assert body["combined"]["id"] == "combined"
    # Currency reported even for zero-exposure requests (rules 5+6).
    assert body["currency"]
    assert isinstance(body["results"], list) and len(body["results"]) == 1


def test_ww_exposure_accepts_a_full_warning_set() -> None:
    """A live NHC set is one polygon per coastal zone. The shared 50-polygon
    cap used for hand-picked fire and flood shapes must not reject it."""
    from app.api.geometry_input import MAX_POLYGONS, MAX_WW_POLYGONS

    client = TestClient(app)
    count = MAX_POLYGONS + 20
    ok = client.post(
        "/api/live/watches-warnings/exposure",
        json={
            "polygons": [
                {
                    "id": f"ww-{i}",
                    "name": "Tropical Storm Warning",
                    "geometry": _tiny_polygon(),
                }
                for i in range(count)
            ]
        },
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert len(body["results"]) == count
    assert body["combined"]["id"] == "combined"
    # Overlapping copies of one box count each location once.
    assert body["combined"]["locationCount"] == body["results"][0]["locationCount"]

    too_many = client.post(
        "/api/live/watches-warnings/exposure",
        json={
            "polygons": [
                {"id": f"ww-{i}", "geometry": _tiny_polygon()}
                for i in range(MAX_WW_POLYGONS + 1)
            ]
        },
    )
    assert too_many.status_code == 422


def test_ww_exposure_endpoint_rejects_bad_geometry() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/live/watches-warnings/exposure",
        json={
            "polygons": [
                {
                    "id": "bad",
                    "geometry": {"type": "Point", "coordinates": [-82.0, 27.5]},
                }
            ]
        },
    )
    # Polygon / MultiPolygon only — a Point ring can't be walked.
    assert r.status_code == 422
