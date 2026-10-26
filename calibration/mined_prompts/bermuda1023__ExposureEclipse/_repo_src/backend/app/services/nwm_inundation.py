"""NOAA National Water Model — modelled flood inundation extent.

Second, additive layer under the NWS flood alerts in :mod:`live_flood`. Alerts
are *warning areas*, drawn to county and zone boundaries; this is modelled
water extent at reach resolution, so it answers "where is the water" rather
than "where has a warning been issued".

It cannot replace the alert layer, for three reasons that are properties of the
source rather than of this code: NOAA flags the service EXPERIMENTAL, it covers
roughly 30% of the US population rather than the whole country, and it models
riverine flooding only — storm surge and other coastal processes are absent.
Silence here therefore never means "no flooding", which is why the coverage
caveat rides on every response instead of living in a wiki.

Endpoint choice is load-bearing. ``FeatureServer/0`` returns HTTP 500 for
``f=geojson`` (reproduced 2026-08-08) while ``MapServer/0`` serves it correctly,
so this reads MapServer and takes NOAA's own GeoJSON. The alternative —
requesting Esri JSON and converting rings ourselves — was built and measured
against this same data: it recovered only 10 of the 35 holes NOAA emits,
because ring winding alone cannot always recover which outer a hole belongs to.
Using the upstream conversion is both less code and more accurate.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ..brand import USER_AGENT


class InundationUnavailable(RuntimeError):
    """The NWM inundation service could not be reached or parsed.

    Distinct from "the model shows no water": during a flood those two must
    never collapse into the same empty map.
    """


NWM_INUNDATION_URL = (
    "https://maps.water.noaa.gov/server/rest/services/nwm/"
    "ana_inundation_extent/MapServer/0/query"
)
NWM_ATTRIBUTION = "NOAA/NWS National Water Model — Analysis & Assimilation inundation extent (EXPERIMENTAL)"
NWM_ATTRIBUTION_URL = "https://water.noaa.gov/about/nwm"

# The exposure route spends the rest of its Vercel budget (vercel.json:
# maxDuration 30) dissolving the extent and ray-casting it, so this cannot use
# the whole window: a 20s fetch plus a worst-case rollup overruns the function.
FETCH_TIMEOUT_S = 12
# The model runs hourly (a 20:00 reference time was still current at 20:52), so
# a short TTL only adds load without adding freshness.
CACHE_TTL_S = 600.0
_CACHE_MAX_ENTRIES = 16

# ArcGIS caps a single response at 2000 features and reports the cut with
# `exceededTransferLimit`. Measured: a 12°×11° box hit the cap at 5.3MB.
ESRI_TRANSFER_LIMIT = 2000

# The bbox cap keeps a well-behaved response near 5MB, so 32MB is far above any
# legitimate payload. It bounds what a misbehaving upstream can make this
# process allocate, since the body is read into memory before it is parsed.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024

# A flood event is thousands of small per-reach polygons, so payload is driven
# by area, not by how much water there is. 25 deg² is roughly a large metro or
# a small state — past that the response truncates silently and the map lies by
# omission rather than by being slow.
MAX_BBOX_DEG2 = 25.0

# 0.001° ≈ 110 m. Measured on a Houston box: 4,146 vertices → 787 (9× smaller)
# with no visible change at mapping scales, and 0.005 saved only 27 more.
DEFAULT_SIMPLIFY_DEG = 0.001

_CACHE: dict[tuple, tuple[float, tuple[list[dict], bool, str | None]]] = {}


@dataclass(slots=True, frozen=True)
class InundationBundle:
    features: list[dict]        # GeoJSON Features, ready to render and intersect
    truncated: bool             # upstream hit its feature cap — extent is partial
    reference_time: str | None  # model run the extent came from
    notes: list[str]


def _cache_get(key: tuple) -> tuple[list[dict], bool, str | None] | None:
    hit = _CACHE.get(key)
    if hit is None:
        return None
    stamped, value = hit
    if time.monotonic() - stamped > CACHE_TTL_S:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: tuple, value: tuple[list[dict], bool, str | None]) -> None:
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.monotonic(), value)


def bbox_area_deg2(bbox: tuple[float, float, float, float]) -> float:
    west, south, east, north = bbox
    return abs(east - west) * abs(north - south)


def _well_formed(coords, depth: int) -> bool:
    """True if ``coords`` is a ring nest of numeric positions ``depth`` levels down.

    The exposure engine walks coordinates structurally and trusts what it finds:
    a string there recurses until the stack dies, and a length-1 position raises
    IndexError. Its own callers validate before handing geometry over, but this
    geometry comes off a third-party HTTP response instead, so the boundary
    check has to happen here.
    """
    if not isinstance(coords, list) or not coords:
        return False
    if depth == 0:
        return (
            len(coords) >= 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in coords[:2])
        )
    return all(_well_formed(c, depth - 1) for c in coords)


def _clean(feature: dict) -> dict | None:
    """Keep the geometry and the two attributes worth showing.

    ``reference_time``/``update_time`` are identical on every feature, so at
    2000 features repeating them is pure payload; they are hoisted onto the
    bundle instead.
    """
    geom = feature.get("geometry")
    if not isinstance(geom, dict) or geom.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    # Polygon nests position→ring→polygon; MultiPolygon adds one more level.
    if not _well_formed(geom.get("coordinates"), 2 if geom["type"] == "Polygon" else 3):
        return None
    props = feature.get("properties") or {}
    return {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "reachId": props.get("feature_id"),
            "streamflowCfs": props.get("streamflow_cfs"),
        },
    }


def fetch_inundation(
    *,
    bbox: tuple[float, float, float, float],
    simplify_deg: float = DEFAULT_SIMPLIFY_DEG,
) -> InundationBundle:
    """Modelled inundation polygons inside ``bbox``.

    Args:
        bbox: ``(west, south, east, north)`` in lon/lat. Required — there is no
            nationwide default, because a nationwide request truncates.
        simplify_deg: Server-side generalisation passed as ``maxAllowableOffset``.
            0 requests full resolution.

    Returns:
        InundationBundle. ``truncated`` is cached alongside the features on
        purpose: caching the payload without the caveat would serve a partial
        extent as if it were complete for the rest of the TTL.

    Raises:
        ValueError: if ``bbox`` is larger than :data:`MAX_BBOX_DEG2`.
        InundationUnavailable: if the service could not be reached or parsed.
    """
    area = bbox_area_deg2(bbox)
    if area > MAX_BBOX_DEG2:
        raise ValueError(
            f"bbox covers {area:,.1f} square degrees, limit {MAX_BBOX_DEG2:,.0f}; "
            f"zoom in — the inundation model is reach-resolution and a wider "
            f"area is silently truncated upstream"
        )

    key = (tuple(round(v, 4) for v in bbox), round(simplify_deg, 6))
    cached = _cache_get(key)
    if cached is None:
        params = {
            "where": "1=1",
            "outFields": "feature_id,streamflow_cfs,reference_time",
            "returnGeometry": "true",
            "geometry": ",".join(str(v) for v in bbox),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            # NOAA's own conversion. See the module docstring for why this is
            # MapServer and not FeatureServer.
            "f": "geojson",
        }
        if simplify_deg > 0:
            params["maxAllowableOffset"] = str(simplify_deg)
        req = urllib.request.Request(
            NWM_INUNDATION_URL + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
                raw_body = r.read(MAX_RESPONSE_BYTES + 1)
            if len(raw_body) > MAX_RESPONSE_BYTES:
                raise InundationUnavailable(
                    f"response exceeded {MAX_RESPONSE_BYTES:,} bytes"
                )
            data = json.loads(raw_body.decode("utf-8"))
        except InundationUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InundationUnavailable(str(exc)) from exc
        # ArcGIS reports failures in a 200 body as well as by status code.
        if isinstance(data, dict) and "error" in data:
            raise InundationUnavailable(str(data.get("error")))

        raw = data.get("features") or []
        feats = [c for c in (_clean(f) for f in raw) if c is not None]
        truncated = bool(data.get("exceededTransferLimit")) or len(raw) >= ESRI_TRANSFER_LIMIT
        ref = next(
            (f.get("properties", {}).get("reference_time") for f in raw
             if (f.get("properties") or {}).get("reference_time")),
            None,
        )
        cached = (feats, truncated, ref)
        _cache_put(key, cached)

    features, truncated, reference_time = cached

    notes = [
        "Modelled inundation is EXPERIMENTAL, covers roughly 30% of the US "
        "population, and excludes storm surge and other coastal processes. An "
        "empty extent is not evidence that there is no flooding here."
    ]
    if truncated:
        notes.append(
            f"The extent was cut off at {ESRI_TRANSFER_LIMIT:,} reaches by the "
            f"upstream service, so water is missing from this view. Zoom in for "
            f"a complete picture before reading any exposed TIV from it."
        )
    elif not features:
        notes.append("The model shows no inundation in this view.")

    return InundationBundle(
        features=features,
        truncated=truncated,
        reference_time=reference_time,
        notes=notes,
    )


def _ring_area_deg2(ring: list) -> float:
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(s) / 2.0


def extent_area_deg2(geom: dict | None) -> float:
    """Area of a Polygon/MultiPolygon in square degrees, holes subtracted.

    Only ever compared against another square-degree quantity, so the fact that
    a degree is not a constant distance does not matter here.
    """
    if not geom:
        return 0.0
    parts = (
        [geom.get("coordinates")] if geom.get("type") == "Polygon"
        else geom.get("coordinates") or []
    )
    total = 0.0
    for rings in parts:
        if not rings:
            continue
        total += _ring_area_deg2(rings[0])
        for hole in rings[1:]:
            total -= _ring_area_deg2(hole)
    return max(total, 0.0)


def dissolve(geoms: list[dict]) -> dict | None:
    """Polygon/MultiPolygon geometries combined into one MultiPolygon.

    A flood event arrives as thousands of per-reach polygons, well past the
    50-polygon cap on the exposure endpoint. Combining them keeps the work
    budget meaningful and matches how an underwriter thinks about one event.

    No topological union is performed and none is needed: membership across
    MultiPolygon parts is a boolean OR and the rollup collects location indices
    into a set, so overlapping reaches already count each location once.
    """
    parts: list = []
    for geom in geoms:
        coords = (geom or {}).get("coordinates")
        if not coords:
            continue
        if geom.get("type") == "Polygon":
            parts.append(coords)
        elif geom.get("type") == "MultiPolygon":
            parts.extend(coords)
    if not parts:
        return None
    return {"type": "MultiPolygon", "coordinates": parts}


def dissolved_geometry(features: list[dict]) -> dict | None:
    """Every reach footprint in ``features`` as one MultiPolygon."""
    return dissolve([f.get("geometry") or {} for f in features])


__all__ = [
    "InundationUnavailable",
    "InundationBundle",
    "MAX_BBOX_DEG2",
    "DEFAULT_SIMPLIFY_DEG",
    "NWM_ATTRIBUTION",
    "NWM_ATTRIBUTION_URL",
    "bbox_area_deg2",
    "extent_area_deg2",
    "fetch_inundation",
    "dissolve",
    "dissolved_geometry",
]
