"""Fetch us-atlas TopoJSON, convert to GeoJSON, write to /mockdata/geo/.

Run once (or whenever you want fresher boundaries):
    cd backend && .venv/Scripts/python scripts/build_geo.py

Output files:
  mockdata/geo/us_states.geojson    — 50 states with properties.geographyId = "US-FL" etc.
  mockdata/geo/us_counties.geojson  — ~3140 counties with geographyId = "US-FL-12086"
  mockdata/geo/countries.geojson    — one US bbox feature (kept from original mock)

Source: https://github.com/topojson/us-atlas (10m precision, public-domain).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

STATES_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"
COUNTIES_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"

# FIPS state code (2-digit string) → USPS postal abbreviation.
STATE_FIPS_TO_USPS: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


def fetch(url: str) -> dict:
    print(f"GET {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def decode_arcs(topology: dict) -> list[list[list[float]]]:
    """Decode TopoJSON's delta-encoded arcs into absolute lon/lat coordinate arrays."""
    transform = topology.get("transform")
    if transform is None:
        return [
            [[pt[0], pt[1]] for pt in arc] for arc in topology["arcs"]
        ]
    sx, sy = transform["scale"]
    tx, ty = transform["translate"]
    decoded: list[list[list[float]]] = []
    for arc in topology["arcs"]:
        x = y = 0
        coords: list[list[float]] = []
        for dx, dy in arc:
            x += dx
            y += dy
            coords.append([x * sx + tx, y * sy + ty])
        decoded.append(coords)
    return decoded


def resolve_arc_ring(arc_ids: list[int], arcs: list[list[list[float]]]) -> list[list[float]]:
    """Concatenate arcs by id (negative ids mean reversed; bitwise ~ to get index)."""
    ring: list[list[float]] = []
    for i, aid in enumerate(arc_ids):
        coords = arcs[~aid] if aid < 0 else arcs[aid]
        if aid < 0:
            coords = list(reversed(coords))
        if i == 0:
            ring.extend(coords)
        else:
            # Skip first point — it's the duplicate join point.
            ring.extend(coords[1:])
    return ring


def topo_geom_to_geojson(geom: dict, arcs: list[list[list[float]]]) -> dict:
    t = geom["type"]
    if t == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [resolve_arc_ring(ring, arcs) for ring in geom["arcs"]],
        }
    if t == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [resolve_arc_ring(ring, arcs) for ring in poly]
                for poly in geom["arcs"]
            ],
        }
    raise ValueError(f"Unsupported topo geometry type: {t}")


def topo_object_to_features(
    topology: dict,
    object_key: str,
    transform_props,
) -> list[dict]:
    arcs = decode_arcs(topology)
    obj = topology["objects"][object_key]
    assert obj["type"] == "GeometryCollection", f"Expected GeometryCollection at {object_key}"
    features: list[dict] = []
    for geom in obj["geometries"]:
        if geom.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        props = transform_props(geom)
        if props is None:  # filter (e.g., Puerto Rico if state FIPS unknown)
            continue
        features.append(
            {
                "type": "Feature",
                "id": geom.get("id"),
                "properties": props,
                "geometry": topo_geom_to_geojson(geom, arcs),
            }
        )
    return features


def state_props(geom: dict) -> dict | None:
    fips = str(geom.get("id", "")).zfill(2)
    usps = STATE_FIPS_TO_USPS.get(fips)
    if not usps:
        return None
    name = (geom.get("properties") or {}).get("name", "")
    return {
        "geographyId": f"US-{usps}",
        "geographyName": name,
        "statecode": usps,
        "fips": fips,
    }


def county_props(geom: dict) -> dict | None:
    fips = str(geom.get("id", "")).zfill(5)
    state_fips = fips[:2]
    usps = STATE_FIPS_TO_USPS.get(state_fips)
    if not usps:
        return None
    name = (geom.get("properties") or {}).get("name", "")
    return {
        "geographyId": f"US-{usps}-{fips}",
        "geographyName": name,
        "statecode": usps,
        "fips": fips,
    }


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    geo_dir = (backend_dir / ".." / "mockdata" / "geo").resolve()
    geo_dir.mkdir(parents=True, exist_ok=True)

    states_topo = fetch(STATES_URL)
    states_features = topo_object_to_features(states_topo, "states", state_props)
    states_fc = {"type": "FeatureCollection", "features": states_features}
    out = geo_dir / "us_states.geojson"
    out.write_text(json.dumps(states_fc, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} — {len(states_features)} states ({out.stat().st_size // 1024} KB)")

    counties_topo = fetch(COUNTIES_URL)
    counties_features = topo_object_to_features(counties_topo, "counties", county_props)
    counties_fc = {"type": "FeatureCollection", "features": counties_features}
    out = geo_dir / "us_counties.geojson"
    out.write_text(json.dumps(counties_fc, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} — {len(counties_features)} counties ({out.stat().st_size // 1024} KB)")

    # Keep a tiny country file (real US bbox) so the COUNTRY level still has something to render.
    countries_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"geographyId": "US", "geographyName": "United States"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-125, 24], [-66, 24], [-66, 49], [-125, 49], [-125, 24]
                    ]],
                },
            }
        ],
    }
    out = geo_dir / "countries.geojson"
    out.write_text(json.dumps(countries_fc, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} — 1 country feature")

    return 0


if __name__ == "__main__":
    sys.exit(main())
