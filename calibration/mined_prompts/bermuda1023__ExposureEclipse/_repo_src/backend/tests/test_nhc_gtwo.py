"""Unit tests for the NHC GTWO KML parser + /gtwo endpoint.

Uses hand-crafted KML fixtures — no network. NHC's real KML encodes chance
via the placemark's ``<styleUrl>`` (``#0``/``#1``/``#2``/``#3`` matching
gray/yellow/orange/red = none/low/medium/high). We also fall back to a
"N percent" description when present (older vintages), so both paths are
covered.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import nhc_gtwo


def _polygon_placemark(name: str, style_url: str, coords: str,
                       desc: str | None = None) -> str:
    desc_el = f"<description>{desc}</description>" if desc else ""
    return f"""
    <Placemark>
      <name>{name}</name>
      {desc_el}
      <styleUrl>{style_url}</styleUrl>
      <Polygon><outerBoundaryIs><LinearRing>
        <coordinates>{coords}</coordinates>
      </LinearRing></outerBoundaryIs></Polygon>
    </Placemark>
    """


def _point_placemark(style_url: str, lon: float, lat: float) -> str:
    return f"""
    <Placemark>
      <styleUrl>{style_url}</styleUrl>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>
    """


def _kml_doc(placemarks: str, doc_name: str = "GTWO test") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{doc_name}</name>"
        f"{placemarks}"
        "</Document></kml>"
    ).encode()


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    nhc_gtwo.clear_cache()


# ─────────────────────────── parser ───────────────────────────


def test_parser_extracts_bucket_from_style_url_numeric() -> None:
    coords = "-75.0,25.0 -70.0,25.0 -70.0,30.0 -75.0,30.0 -75.0,25.0"
    payload = _kml_doc(
        _polygon_placemark("A1", "#1", coords)
        + _polygon_placemark("A2", "#2", coords)
        + _polygon_placemark("A3", "#3", coords)
        + _polygon_placemark("A0", "#0", coords)
    )
    areas, _ = nhc_gtwo._parse_gtwo(payload, "atl")
    assert len(areas) == 4
    buckets = [a.chance_bucket for a in areas]
    assert buckets == ["low", "medium", "high", "none"]
    pcts = [a.chance_pct for a in areas]
    assert pcts == [20, 50, 80, 0]


def test_parser_closes_open_rings() -> None:
    coords = "-75.0,25.0 -70.0,25.0 -70.0,30.0 -75.0,30.0"
    payload = _kml_doc(_polygon_placemark("Open", "#2", coords))
    areas, _ = nhc_gtwo._parse_gtwo(payload, "atl")
    assert len(areas) == 1
    assert areas[0].ring[0] == areas[0].ring[-1]


def test_parser_falls_back_to_description_percent() -> None:
    # No usable styleUrl but a "N percent" phrase in the description.
    coords = "-75.0,25.0 -70.0,25.0 -70.0,30.0 -75.0,30.0 -75.0,25.0"
    payload = _kml_doc(
        _polygon_placemark("A", "#unknown", coords,
                           desc="Formation chance through 7 days...40 percent")
    )
    areas, _ = nhc_gtwo._parse_gtwo(payload, "atl")
    assert len(areas) == 1
    assert areas[0].chance_pct == 40
    assert areas[0].chance_bucket == "medium"


def test_parser_pairs_polygon_with_following_point_marker() -> None:
    coords = "-75.0,25.0 -70.0,25.0 -70.0,30.0 -75.0,30.0 -75.0,25.0"
    payload = _kml_doc(
        _polygon_placemark("A1", "#3", coords)
        + _point_placemark("#higx", -72.5, 27.5)
    )
    areas, _ = nhc_gtwo._parse_gtwo(payload, "atl")
    assert len(areas) == 1
    assert areas[0].marker == (-72.5, 27.5)


def test_parser_extracts_issued_note_from_doc_name() -> None:
    coords = "-75.0,25.0 -70.0,25.0 -70.0,30.0 -75.0,30.0 -75.0,25.0"
    payload = _kml_doc(
        _polygon_placemark("A", "#1", coords),
        doc_name="GTWO - Mon Aug 10 23:41:16 2026",
    )
    _, issued = nhc_gtwo._parse_gtwo(payload, "atl")
    assert issued == "Mon Aug 10 23:41:16 2026"


# ─────────────────────────── endpoint ───────────────────────────


def test_gtwo_endpoint_smoke_returns_empty_when_upstream_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nhc_gtwo, "_download_kml", lambda url: None)
    client = TestClient(app)
    r = client.get("/api/live/gtwo?basin=atl")
    assert r.status_code == 200
    body = r.json()
    assert body["areas"] == []
    assert body["note"]


def test_gtwo_endpoint_parses_areas(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = "-75.0,25.0 -70.0,25.0 -70.0,30.0 -75.0,30.0 -75.0,25.0"
    kml = _kml_doc(
        _polygon_placemark("A1", "#3", coords)
        + _point_placemark("#higx", -72.5, 27.5)
        + _polygon_placemark("A2", "#1",
                             "-40.0,15.0 -35.0,15.0 -35.0,20.0 -40.0,20.0 -40.0,15.0")
        + _point_placemark("#lowx", -37.5, 17.5)
    )
    monkeypatch.setattr(nhc_gtwo, "_download_kml", lambda url: kml)
    client = TestClient(app)
    r = client.get("/api/live/gtwo?basin=atl")
    body = r.json()
    assert len(body["areas"]) == 2
    assert body["areas"][0]["chanceBucket"] == "high"
    assert body["areas"][0]["marker"] == [-72.5, 27.5]
    assert body["areas"][1]["chanceBucket"] == "low"
    assert body["areas"][1]["marker"] == [-37.5, 17.5]


def test_gtwo_endpoint_rejects_unknown_basin() -> None:
    client = TestClient(app)
    r = client.get("/api/live/gtwo?basin=zz")
    assert r.status_code == 422
