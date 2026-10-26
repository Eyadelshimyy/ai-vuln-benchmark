"""Sea-surface temperature ("ocean energy") layer.

For a v1 demo this returns a deterministic synthetic SST grid around a
bounding box — climatologically plausible for the Atlantic, late-summer
peak: warmer water in the Caribbean/Gulf (~28-30 °C), cooler in the
mid-Atlantic and north (~20-25 °C), with a "warm pool" bias toward the
Loop Current. A real-data hookup (NOAA Coral Reef Watch / ERDDAP MURSST)
can replace ``sst_grid`` later without touching the API shape.

SST is the simplest proxy for "ocean energy" available to a hurricane.
The more rigorous measure is Tropical Cyclone Heat Potential (TCHP) —
integrated heat in the upper 26 °C layer — but that data is harder to
serve in real time. Underwriters mostly want the same intuition (warm
water under the storm = bad) so SST does the job here.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from ..brand import USER_AGENT as MUR_USER_AGENT


@dataclass(slots=True, frozen=True)
class SSTPoint:
    lat: float
    lon: float
    temp_c: float
    favorable_for_intensification: bool  # SST ≥ 26.5 °C, the classic threshold


# Anchors that give a vaguely realistic late-summer Atlantic field. Values
# in °C. Each entry is (lat, lon, sst). We interpolate via inverse-distance
# weighting at query time. Denser anchor net = smoother gradients without
# obvious "pull" toward a single point.
_ANCHORS: tuple[tuple[float, float, float], ...] = (
    # Caribbean / Gulf — the warm pool
    (12.0,  -65.0, 29.0),
    (15.0,  -75.0, 29.5),
    (18.0,  -78.0, 29.6),
    (20.0,  -85.0, 30.0),  # Western Caribbean / Gulf
    (22.0,  -88.0, 30.2),
    (24.0,  -84.0, 29.7),  # Loop Current
    (25.0,  -87.0, 29.5),
    (26.0,  -82.0, 29.5),  # SW Florida shelf
    (27.0,  -90.0, 29.0),
    (25.0,  -78.0, 29.0),  # Bahamas
    (24.0,  -76.0, 28.8),
    # SE US coast
    (30.0,  -80.0, 27.5),
    (32.0,  -78.0, 26.8),
    (34.0,  -76.0, 25.5),
    (30.0,  -86.0, 28.5),  # Gulf Coast FL panhandle
    (29.0,  -93.0, 28.0),  # TX/LA coast
    # Mid-Atlantic
    (36.0,  -73.0, 25.0),
    (38.0,  -71.0, 24.0),
    (40.0,  -70.0, 22.5),
    (42.0,  -68.0, 21.0),
    # Tropical Atlantic (MDR)
    (10.0,  -40.0, 28.0),
    (12.0,  -45.0, 28.5),
    (15.0,  -55.0, 28.7),
    (18.0,  -50.0, 28.5),
    (20.0,  -55.0, 28.0),
    (12.0,  -25.0, 27.5),  # Cape Verde area
    # Subtropical Atlantic
    (28.0,  -60.0, 27.0),
    (32.0,  -55.0, 25.5),
    (35.0,  -60.0, 24.5),
    (38.0,  -50.0, 23.0),
    # Northern
    (44.0,  -55.0, 18.0),
    (47.0,  -45.0, 16.0),
    (45.0,  -35.0, 17.5),
)

INTENSIFICATION_THRESHOLD_C = 26.5


def _interp_sst(lat: float, lon: float) -> float:
    """Inverse-distance-weighted SST from the anchor points (k=2 power)."""
    weights = 0.0
    weighted = 0.0
    for a_lat, a_lon, sst in _ANCHORS:
        d2 = (lat - a_lat) ** 2 + (lon - a_lon) ** 2
        if d2 < 1e-6:
            return sst
        w = 1.0 / (d2 ** 1.0)
        weights += w
        weighted += w * sst
    return weighted / weights if weights else 0.0


def sst_grid(
    *,
    bbox: tuple[float, float, float, float],
    step_deg: float = 1.0,
) -> list[SSTPoint]:
    """Sample SST on a regular lat/lon grid covering ``bbox``."""
    west, south, east, north = bbox
    pts: list[SSTPoint] = []
    lat = south
    while lat <= north + 1e-9:
        lon = west
        while lon <= east + 1e-9:
            t = _interp_sst(lat, lon)
            pts.append(
                SSTPoint(
                    lat=round(lat, 3),
                    lon=round(lon, 3),
                    temp_c=round(t, 1),
                    favorable_for_intensification=t >= INTENSIFICATION_THRESHOLD_C,
                )
            )
            lon += step_deg
        lat += step_deg
    return pts


def sst_at_point(lat: float, lon: float) -> SSTPoint:
    t = _interp_sst(lat, lon)
    return SSTPoint(
        lat=round(lat, 3),
        lon=round(lon, 3),
        temp_c=round(t, 1),
        favorable_for_intensification=t >= INTENSIFICATION_THRESHOLD_C,
    )


# ─────────────────────────── Real MUR SST via ERDDAP ───────────────────────────

# NOAA JPL MUR (Multi-scale Ultra-high Resolution) SST monthly, 0.01° native.
# We request with a stride so the returned grid is ~0.1° (manageable cell
# count). Monthly product is stable + always available with ~6 week lag.
MUR_BASE = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41mday.csv"
)
MUR_TIMEOUT_S = 30
# Peak-hurricane month with reliable late-summer Atlantic warmth — used for
# the demo when we want a "what the ocean looks like during a hurricane".
DEMO_MUR_MONTH = "2024-09-16T00:00:00Z"


def _mur_stride_for_bbox(span_deg: float) -> int:
    """Pick a stride that keeps the response under a few thousand cells:
       span ≤  6° → 0.05° (stride  5)
       span ≤ 12° → 0.10° (stride 10)
       span ≤ 25° → 0.25° (stride 25)
       else       → 0.50° (stride 50)
    """
    if span_deg <= 6:
        return 5
    if span_deg <= 12:
        return 10
    if span_deg <= 25:
        return 25
    return 50


@lru_cache(maxsize=32)
def _mur_csv(west: float, south: float, east: float, north: float, time_iso: str) -> str | None:
    """Cached ERDDAP fetch. Returns CSV body or None on failure."""
    span = max(east - west, north - south)
    stride = _mur_stride_for_bbox(span)
    url = (
        f"{MUR_BASE}?sst[({time_iso})]"
        f"[({south}):{stride}:({north})]"
        f"[({west}):{stride}:({east})]"
    )
    req = urllib.request.Request(url, headers={"User-Agent": MUR_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=MUR_TIMEOUT_S) as r:
            return r.read().decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _try_real_mur(bbox: tuple[float, float, float, float]) -> list[SSTPoint] | None:
    """Hit ERDDAP for real MUR SST in the bbox. Returns None on any failure
    so the caller can fall back to the synthetic field."""
    west, south, east, north = bbox
    body = _mur_csv(round(west, 2), round(south, 2), round(east, 2), round(north, 2), DEMO_MUR_MONTH)
    if not body:
        return None
    pts: list[SSTPoint] = []
    for line in body.splitlines()[2:]:  # skip header + units row
        cols = line.split(",")
        if len(cols) < 4:
            continue
        try:
            lat = float(cols[1])
            lon = float(cols[2])
            sst_str = cols[3].strip()
            if not sst_str or sst_str == "NaN":
                continue
            t = float(sst_str)
        except ValueError:
            continue
        pts.append(
            SSTPoint(
                lat=round(lat, 3),
                lon=round(lon, 3),
                temp_c=round(t, 2),
                favorable_for_intensification=t >= INTENSIFICATION_THRESHOLD_C,
            )
        )
    return pts if pts else None


# Drop SST cells whose centre is within this many nautical miles of any US
# county centroid — proxy for "on land" so the heatmap doesn't visibly
# bleed across the Florida peninsula etc. 25 nm ≈ 28 mi, roughly the
# half-diameter of an average US county.
LAND_MASK_RADIUS_NM = 25.0


def _is_us_land(lat: float, lon: float, county_grid: dict) -> bool:
    """Bbox-coarse + haversine check against US county centroids."""
    # Quick CONUS+PR bbox cull — anything well outside the US is "ocean" as
    # far as we're concerned (Caribbean, Atlantic, Gulf, Pacific).
    if not (15.0 <= lat <= 50.0 and -125.0 <= lon <= -65.0):
        return False
    deg = LAND_MASK_RADIUS_NM / 50.0  # generous bounding box
    lat_lo, lat_hi = lat - deg, lat + deg
    lon_lo, lon_hi = lon - deg, lon + deg
    for c in county_grid.values():
        if not (lat_lo <= c.centroid_lat <= lat_hi and lon_lo <= c.centroid_lon <= lon_hi):
            continue
        # Cheap squared-degree distance suffices at this scale.
        dlat = lat - c.centroid_lat
        dlon = lon - c.centroid_lon
        if (dlat * dlat + dlon * dlon) ** 0.5 * 60 <= LAND_MASK_RADIUS_NM:
            return True
    return False


def _drop_us_land(cells: list[SSTPoint]) -> list[SSTPoint]:
    # Lazy import to avoid a circular dep with hurricane_impact at module load.
    from .hurricane_impact import county_centroids

    grid = county_centroids()
    return [p for p in cells if not _is_us_land(p.lat, p.lon, grid)]


def sst_field(bbox: tuple[float, float, float, float]) -> tuple[list[SSTPoint], str]:
    """Return (cells, source). Tries real MUR first, falls back to synthetic.

    Both sources are passed through a US-land mask so the SST heatmap stays
    over water. MUR already returns NaN over land (we drop those at parse
    time); the synthetic IDW field doesn't know what land is, so the mask
    is essential there. The mask covers CONUS + Alaska + Puerto Rico via
    county centroids; small Caribbean islands fall through (they have no
    US county nearby and will paint as ocean, which is fine at this zoom).

    ``source`` ∈ {``"mur"``, ``"synthetic"``} so the API can disclose it.
    """
    real = _try_real_mur(bbox)
    if real:
        return _drop_us_land(real), "mur"
    # Match the previous synthetic call's behaviour but at the same resolution
    # as the MUR stride would have produced — visually consistent fallback.
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    step = 0.1 if span < 6 else 0.25 if span < 12 else 0.5 if span < 25 else 1.0
    return _drop_us_land(sst_grid(bbox=bbox, step_deg=step)), "synthetic"


# Suppress lint about an unused import — math is intentionally kept available
# for a future real-data swap.
_ = math
