"""JMA typhoon parser. Fixtures are the 2026-10-08 Nolo bulletin, trimmed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import jma_typhoon

FIXTURE = Path(__file__).parent / "fixtures" / "jma"
client = TestClient(app)


def _load(name: str):
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def _nolo() -> jma_typhoon.JmaStorm:
    storm = jma_typhoon.parse_storm(
        "TC2635", _load("tc2635_forecast.json"), _load("tc2635_spec.json"),
    )
    assert storm is not None
    return storm


def test_nolo_analysis_and_forecast_centers() -> None:
    storm = _nolo()
    assert storm.summary.storm_id == "TC2635"
    assert storm.summary.name == "Nolo"
    assert storm.summary.name_jp == "ノロ"
    assert storm.summary.year == 2026
    assert storm.summary.classification == "STS"
    assert storm.summary.intensity_kt == 60
    assert storm.summary.pressure_mb == 985
    assert storm.summary.lat == pytest.approx(26.7)
    assert storm.summary.lon == pytest.approx(155.8)
    assert storm.wind_ms == pytest.approx(30)
    hours = [fix.hours_out for fix in storm.forecast]
    assert hours == [0, 12, 24, 48, 72]
    low = storm.forecast[-1]
    assert low.category_en == "LOW"
    assert low.intensity is None
    strong = next(fix for fix in storm.forecast if fix.hours_out == 24)
    assert strong.category_en == "TY"
    assert strong.intensity == "strong (強い)"
    assert any(text.startswith("North 330") for text in storm.gale_ranges)
    assert any(text.startswith("South 220") for text in storm.gale_ranges)
    assert any("130 km" in text for text in storm.storm_ranges)


def test_nolo_circles_use_published_radii() -> None:
    storm = _nolo()
    circle = next(
        c for c in storm.circles
        if c.kind == "probability" and c.hours_out == 12
    )
    # 30 nm on the bulletin is 55,560 m in forecast.json.
    assert circle.radius_m == pytest.approx(55_560)
    arc_lat, _arc_lng = jma_typhoon._meters_per_deg(27.7)
    north = max(pt[1] for pt in circle.ring)
    assert north == pytest.approx(27.7 + 55_560 / arc_lat, abs=0.02)
    assert circle.label == "+12h"
    low = next(c for c in storm.circles if c.hours_out == 72 and c.kind == "probability")
    assert "extratropical" in low.label
    assert any(c.kind == "gale" and c.radius_m == pytest.approx(277_800) for c in storm.circles)
    assert any(c.kind == "storm" and c.hours_out == 0 for c in storm.circles)
    # Intermediate storm-warning envelopes are not drawn; only the last one.
    assert any(p.kind == "storm-arc" and p.hours_out == 72 for p in storm.paths)
    assert not any(p.kind == "storm-arc" and p.hours_out == 12 for p in storm.paths)
    assert any(p.kind == "tangent" and p.hours_out == 12 for p in storm.paths)


def test_circle_near_the_dateline_stays_local() -> None:
    ring = jma_typhoon.arc_lonlat(20.0, 179.5, 100_000, 0, 360, step_deg=10)
    lons = [pt[0] for pt in ring]
    assert max(lons) - min(lons) < 5


def test_antimeridian_segment_is_split() -> None:
    parts = jma_typhoon.split_antimeridian([
        [178.0, 10.0],
        [179.5, 10.0],
        [-179.5, 11.0],
        [-178.0, 11.0],
    ])
    assert len(parts) == 2
    assert parts[0] == [[178.0, 10.0], [179.5, 10.0]]
    assert parts[1] == [[-179.5, 11.0], [-178.0, 11.0]]


def test_quiet_list_and_a_missing_detail_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    jma_typhoon.clear_cache()

    def quiet(url: str):
        if url.endswith("targetTc.json"):
            return []
        raise AssertionError(url)

    monkeypatch.setattr(jma_typhoon, "_get_json", quiet)
    assert jma_typhoon.fetch_active_typhoons(refresh=True) == []

    def missing(url: str):
        if url.endswith("targetTc.json"):
            return [{
                "tropicalCyclone": "TC9999",
                "typhoonNumber": "2699",
                "category": "TY",
                "issue": "2026-10-08T00:00:00+09:00",
            }]
        return None

    monkeypatch.setattr(jma_typhoon, "_get_json", missing)
    jma_typhoon.clear_cache()
    assert jma_typhoon.fetch_active_typhoons(refresh=True) == []


def test_one_missing_code_does_not_drop_the_other(monkeypatch: pytest.MonkeyPatch) -> None:
    forecast = _load("tc2635_forecast.json")
    spec = _load("tc2635_spec.json")

    def fake(url: str):
        if url.endswith("targetTc.json"):
            return [
                {"tropicalCyclone": "TC2635", "typhoonNumber": "2628", "category": "STS"},
                {"tropicalCyclone": "TC0001", "typhoonNumber": "2601", "category": "TY"},
            ]
        if "TC2635/forecast" in url:
            return forecast
        if "TC2635/specifications" in url:
            return spec
        return None

    monkeypatch.setattr(jma_typhoon, "_get_json", fake)
    jma_typhoon.clear_cache()
    rows = jma_typhoon.fetch_active_typhoons(refresh=True)
    assert [row.storm_id for row in rows] == ["TC2635"]


def test_live_list_keeps_nhc_when_jma_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import live as live_api
    from app.services.live_hurricane import LiveStormSummary

    jma_typhoon.clear_cache()
    monkeypatch.setattr(jma_typhoon, "_get_json", lambda url: [])
    monkeypatch.setattr(
        live_api,
        "fetch_active_summaries",
        lambda: [LiveStormSummary(
            storm_id="AL092026",
            name="Isaias",
            year=2026,
            classification="TS",
            intensity_kt=45,
            pressure_mb=1000,
            lat=28.0,
            lon=-88.0,
            is_live=True,
            label="Isaias (2026) — live, TS",
        )],
    )
    body = client.get("/api/live/storms").json()
    assert body["active"][0]["stormId"] == "AL092026"
    assert body["typhoons"] == []


def test_jma_bundle_does_not_call_nhc(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import live as live_api

    forecast = _load("tc2635_forecast.json")
    spec = _load("tc2635_spec.json")

    def fake(url: str):
        if "TC2635/forecast" in url:
            return forecast
        if "TC2635/specifications" in url:
            return spec
        return None

    def boom(*_args, **_kwargs):
        raise AssertionError("NHC path should not run for a JMA id")

    monkeypatch.setattr(jma_typhoon, "_get_json", fake)
    jma_typhoon.clear_cache()
    monkeypatch.setattr(live_api, "storm_and_forecasts", boom)
    monkeypatch.setattr(live_api, "fetch_model_tracks", boom)
    monkeypatch.setattr(live_api, "compute_ensemble_risk", boom)

    bundle = client.get("/api/live/storms/TC2635")
    assert bundle.status_code == 200
    body = bundle.json()
    assert body["storm"]["name"] == "Nolo"
    assert body["storm"]["classification"] == "STS"
    assert body["jma"]["nameJp"] == "ノロ"
    assert body["jma"]["windMs"] == 30
    assert body["forecastCone"] is None
    assert body["alerts"] == []
    assert body["watchesWarnings"] == []
    assert body["peakSurge"] == []
    assert any(c["kind"] == "probability" and c["hoursOut"] == 12 for c in body["jma"]["circles"])
    assert "10-minute" in body["jma"]["note"]

    tracks = client.get("/api/live/storms/TC2635/model-tracks")
    assert tracks.status_code == 200
    assert tracks.json()["tracks"] == []
    assert "JMA" in " ".join(tracks.json()["notes"])

    risk = client.get("/api/live/storms/TC2635/ensemble-risk")
    assert risk.status_code == 200
    assert risk.json()["strikeByCounty"] == []
    assert "JMA" in " ".join(risk.json()["notes"])

    missing = client.get("/api/live/storms/TC0001")
    assert missing.status_code == 404
