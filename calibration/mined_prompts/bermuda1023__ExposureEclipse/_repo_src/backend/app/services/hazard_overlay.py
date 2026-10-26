"""Hazard overlays — tornado, hail, wildfire.

Tornado is REAL DATA: a 0.4° KDE of NOAA SPC's 1950-2025 SVRGIS
tornado-touchdown shapefile, with a linear recency weight (1950 = 0.5×
→ 2025 = 2.0×) so the heatmap reflects current climatology (Tornado
Alley shifting east) rather than a flat 75-year average. EF3+ get a
small magnitude boost so the damage signal isn't drowned by weak
short-track EF0s. The grid is pre-baked by
``backend/scripts/build_tornado_grid.py`` into
``mockdata/hazard_tornado_grid.json`` — runtime just loads + normalises.

Hail + wildfire still ship as synthetic patterns anchored on published
distributions; same swap pattern when real data is wired in.

Data sources / swap targets:

- **Tornado** (REAL): NOAA SPC SVRGIS 1950-2025 tornado shapefile
  https://www.spc.noaa.gov/gis/svrgis/

- **Hail** (synthetic): NOAA SPC severe weather event DB, ≥1″ reports
  https://www.spc.noaa.gov/wcm/#data

- **Wildfire** (synthetic): USFS Wildfire Hazard Potential (LANDFIRE)
  https://www.firelab.org/project/wildfire-hazard-potential
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

HazardType = Literal["tornado", "hail", "wildfire"]


@dataclass(slots=True, frozen=True)
class HazardGridPoint:
    lat: float
    lon: float
    raw: float        # native units (count or index)
    normalised: float # 0..1 for colour ramp


@dataclass(slots=True, frozen=True)
class HazardLegend:
    title: str
    unit: str         # "tornadoes since 1950" / "≥1″ hail reports since 1955" / "WHP index (1-5)"
    source: str       # human-readable attribution
    source_url: str
    raw_min: float
    raw_max: float
    palette: list[str]
    stops: list[float]  # raw-value thresholds aligned with palette colours
    note: str | None = None


# ─────────────────────────── per-peril scoring ───────────────────────────


def _dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Small-distance flat-earth approximation; fine for ranking inside CONUS.
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat * dlat + dlon * dlon)


_MOCKDATA_DIR = Path(__file__).resolve().parents[3] / "mockdata"


def _load_grid_json(filename: str) -> tuple[list[tuple[float, float, float]], float]:
    """Load a pre-baked KDE grid from mockdata. Returns ``(cells, stepDeg)``
    where cells is ``[(lat, lon, raw), ...]``. Empty list + 0.4° default if
    the file is missing — callers can then fall back to a synthetic.

    Tolerates both the modern payload ``{stepDeg, cells}`` and the legacy
    flat-list form for forward/backward compat across builds."""
    path = _MOCKDATA_DIR / filename
    if not path.exists():
        return [], 0.4
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        step = float(raw.get("stepDeg", 0.4))
        cells = raw.get("cells", [])
    else:
        step = 0.4
        cells = raw
    return [(float(d["lat"]), float(d["lon"]), float(d["raw"])) for d in cells], step


@lru_cache(maxsize=1)
def _tornado_cells() -> tuple[list[tuple[float, float, float]], float]:
    return _load_grid_json("hazard_tornado_grid.json")


@lru_cache(maxsize=1)
def _hail_cells() -> tuple[list[tuple[float, float, float]], float]:
    return _load_grid_json("hazard_hail_grid.json")


@lru_cache(maxsize=1)
def _wildfire_cells() -> tuple[list[tuple[float, float, float]], float]:
    return _load_grid_json("hazard_wildfire_grid.json")


def _tornado_raw_synthetic(lat: float, lon: float) -> float:
    """Fallback synthetic pattern when the pre-baked SPC grid isn't
    available (e.g. local dev without the shapefile build)."""
    if not (24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0):
        return 0.0
    d_alley = _dist_km(lat, lon, 35.0, -98.0)
    d_dixie = _dist_km(lat, lon, 33.0, -88.0)
    d_fl    = _dist_km(lat, lon, 28.5, -81.5)
    score = (
        320 * math.exp(-(d_alley / 380) ** 2) +
        140 * math.exp(-(d_dixie / 320) ** 2) +
         95 * math.exp(-(d_fl    / 220) ** 2)
    )
    if -90.0 <= lon <= -75.0 and 36.0 <= lat <= 43.0:
        score += 15
    if -120.0 <= lon <= -103.0 and lat >= 35.0:
        score *= 0.20
    return max(0.0, score)


def _hail_raw_synthetic(lat: float, lon: float) -> float:
    """Synthetic ≥1″ hail report count per county, 1955-2023, anchored on:
       - Hail Alley centroid ~38°N, -101°W (E. CO / W. KS / NE / W. TX)
       - Secondary peak central Plains ~40°N, -97°W (NE / IA / MO)
    SPC publishes peaks of 700-1200 per county in Hail Alley, 200-500
    Midwest, 50-200 Southeast, <50 in West Coast / NE.
    """
    if not (24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0):
        return 0.0
    d_main = _dist_km(lat, lon, 38.0, -101.0)
    d_mid  = _dist_km(lat, lon, 40.0, -97.0)
    d_se   = _dist_km(lat, lon, 33.0, -85.0)
    score = (
        900 * math.exp(-(d_main / 350) ** 2) +
        450 * math.exp(-(d_mid  / 320) ** 2) +
        180 * math.exp(-(d_se   / 450) ** 2)
    )
    # West coast suppression
    if lon < -115.0:
        score *= 0.15
    return max(0.0, score)


def _wildfire_raw(lat: float, lon: float) -> float:
    """USFS Wildfire Hazard Potential — 1-5 index per county. Anchored on:
       - California chaparral + Sierra: 4-5
       - Mountain West (OR/WA east, ID, NV, AZ, NM, CO, UT, WY): 4-5
       - Florida pine flatwoods: 3-4 (hot pockets)
       - Eastern seaboard / Midwest: 1-2
       - Pacific NW coast: 3-4
    """
    if not (24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0):
        return 1.0
    score = 1.0  # baseline
    # California chaparral / Sierra
    d_ca = _dist_km(lat, lon, 36.5, -119.5)
    score += 4.0 * math.exp(-(d_ca / 350) ** 2)
    # Mountain west
    if -120.0 <= lon <= -104.0 and 32.0 <= lat <= 49.0:
        score += 2.5
    # Pacific NW
    if -125.0 <= lon <= -120.0 and 42.0 <= lat <= 49.0:
        score += 1.5
    # Texas hill country + west TX
    if -106.0 <= lon <= -98.0 and 28.0 <= lat <= 35.0:
        score += 1.2
    # Florida pine flatwoods — small bumps
    if -83.0 <= lon <= -80.0 and 27.0 <= lat <= 30.5:
        score += 1.0
    # Eastern seaboard suppression handled by baseline + lack of bonus
    if lon > -85.0 and lat > 32.0:
        score = min(score, 2.5)
    return max(1.0, min(5.0, score))


# ─────────────────────────── public API ───────────────────────────


def _us_land_filter(lat: float, lon: float, county_grid: dict) -> bool:
    """Return True if (lat, lon) is on US land — proxy via county centroid
    proximity. Same heuristic used by the SST mask, INVERTED purpose: we
    KEEP land here (hail/tornado/wildfire are land phenomena)."""
    if not (24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0):
        return False
    radius_nm = 25.0
    deg = radius_nm / 50.0
    lat_lo, lat_hi = lat - deg, lat + deg
    lon_lo, lon_hi = lon - deg, lon + deg
    for c in county_grid.values():
        if not (lat_lo <= c.centroid_lat <= lat_hi and lon_lo <= c.centroid_lon <= lon_hi):
            continue
        dlat = lat - c.centroid_lat
        dlon = lon - c.centroid_lon
        if (dlat * dlat + dlon * dlon) ** 0.5 * 60 <= radius_nm:
            return True
    return False


_REAL_DATA_LOADERS = {
    "tornado": _tornado_cells,
    "hail": _hail_cells,
    "wildfire": _wildfire_cells,
}


@lru_cache(maxsize=8)
def build_grid(
    hazard: HazardType,
    step_deg: float = 0.4,
) -> tuple[list[HazardGridPoint], HazardLegend, float]:
    """Sample the hazard scoring on a lat/lon grid covering CONUS.
    Returned cells tile contiguously when the frontend draws each as a
    step_deg square fill — a smooth heatmap that ignores county / state
    lines. Cached per peril (deterministic).

    Tornado + hail: load the pre-baked SPC SVRGIS KDE grid directly so
    the heatmap reflects real touchdown / report data (recency-weighted,
    EF/mag-boosted) rather than a synthetic pattern.

    Wildfire: still synthetic until real LANDFIRE WHP raster is wired in.
    """
    if hazard in _REAL_DATA_LOADERS:
        cells, baked_step = _REAL_DATA_LOADERS[hazard]()
        if cells:
            raws = [r for _, _, r in cells]
            raw_min, raw_max = 0.0, max(raws)
            span = max(raw_max - raw_min, 1e-9)
            out = [
                HazardGridPoint(
                    lat=lat, lon=lon, raw=round(r, 2),
                    normalised=round((r - raw_min) / span, 4),
                )
                for lat, lon, r in cells
            ]
            # Return the step the JSON was actually baked at — the frontend
            # uses this to size each fill polygon, so a mismatch leaves
            # gaps or overlaps in the heatmap.
            return out, _legend_for(hazard, raw_min, raw_max), baked_step
        # Fall through to synthetic if the JSON isn't present.

    from .hurricane_impact import county_centroids

    raw_fn = {
        "tornado": _tornado_raw_synthetic,
        "hail": _hail_raw_synthetic,
        "wildfire": _wildfire_raw,
    }[hazard]

    county_grid = county_centroids()
    south, north = 24.0, 49.5
    west, east = -125.0, -66.0

    points: list[HazardGridPoint] = []
    raws_s: list[float] = []
    lat = south
    while lat <= north + 1e-9:
        lon = west
        while lon <= east + 1e-9:
            if _us_land_filter(lat, lon, county_grid):
                r = raw_fn(lat, lon)
                points.append(HazardGridPoint(lat=round(lat, 3), lon=round(lon, 3), raw=r, normalised=0.0))
                raws_s.append(r)
            lon += step_deg
        lat += step_deg

    raw_min = min(raws_s) if raws_s else 0.0
    raw_max = max(raws_s) if raws_s else 1.0
    span = max(raw_max - raw_min, 1e-9)
    out = [
        HazardGridPoint(
            lat=p.lat,
            lon=p.lon,
            raw=round(p.raw, 2),
            normalised=round((p.raw - raw_min) / span, 4),
        )
        for p in points
    ]
    legend = _legend_for(hazard, raw_min, raw_max)
    return out, legend, step_deg


def _legend_for(hazard: HazardType, raw_min: float, raw_max: float) -> HazardLegend:
    if hazard == "tornado":
        # 7-stop ramp picked against the bias-corrected KDE distribution
        # (p50≈18, p75≈51, p90≈78, p95≈94, p99≈122, max≈208 after the
        # population deflator). Bottom half stays pale; top decile
        # escalates orange → red → near-black crimson so the most
        # dangerous regions (Dixie Alley post-correction, rural Plains)
        # actually pop instead of plateauing red.
        palette = [
            "#f8fafc",
            "#fef3c7",
            "#fde047",
            "#fb923c",
            "#dc2626",
            "#7f1d1d",
            "#3b0a0a",
        ]
        stops = [0, 12, 28, 45, 60, 75, 90]
        return HazardLegend(
            title="Tornado hazard index",
            unit="0–100 (blended climatology + history)",
            source="SPC SVRGIS 1950-2025 + Brooks/Tippett climatology",
            source_url="https://www.spc.noaa.gov/gis/svrgis/",
            raw_min=raw_min,
            raw_max=raw_max,
            palette=palette,
            stops=stops,
            note="Blend of two surfaces: 60% smooth climatology prior (anchored on the Brooks et al. 2003 / Tippett et al. 2015 environmental tornado-frequency surfaces — bias-free because derived from atmospheric ingredients, not point reports) + 40% real SPC touchdowns (wide 0.7° KDE so single-city clusters dilute into their region). EF3+ magnitude boost + 0.5× → 2.0× recency weight on the historical side. No city-by-city correction — the blend itself washes out reporting bias.",
        )
    if hazard == "hail":
        # 7-stop ramp picked against the bias-corrected KDE distribution
        # (p50≈97, p75≈223, p90≈385, p95≈514, p99≈737, max≈1887). Pale
        # at bottom so quiet areas read quiet; top decile escalates
        # quickly through indigo into near-black so Black Hills + the
        # Texas Panhandle peaks read as obviously the worst.
        palette = [
            "#f8fafc",
            "#dbeafe",
            "#93c5fd",
            "#60a5fa",
            "#3b82f6",
            "#1e3a8a",
            "#0c1429",
        ]
        stops = [0, 12, 28, 45, 60, 75, 90]
        return HazardLegend(
            title="Severe hail hazard index",
            unit="0–100 (blended climatology + history)",
            source="SPC SVRGIS 1955-2025 + Cintineo/Allen-Tippett climatology",
            source_url="https://www.spc.noaa.gov/wcm/#data",
            raw_min=raw_min,
            raw_max=raw_max,
            palette=palette,
            stops=stops,
            note="Blend of two surfaces: 60% smooth climatology prior (anchored on the Cintineo et al. 2012 + Allen & Tippett 2015 environmental hail-frequency surfaces) + 40% real SPC reports (wide 0.7° KDE so single-city clusters dilute into their region). Magnitude-weighted 1.0× (1″) → 2.5× (4″); mild recency ramp 0.7× → 1.3×. No city-by-city correction — the blend itself washes out reporting bias.",
        )
    # wildfire — real WFIGS perimeter data (acres-weighted KDE)
    palette = [
        "#f8fafc",
        "#fef3c7",
        "#fdba74",
        "#fb923c",
        "#dc2626",
        "#7c2d12",
        "#1c0a05",
    ]
    # Stops against the higher-resolution distribution (p50≈2.5, p90≈31,
    # p95≈60, p99≈152, max≈571). Top stop near max so the recent megafire
    # scars (Calf Canyon NM, August Complex NorCal, Hells Canyon) pop.
    stops = [0, 3, 15, 50, 120, 280, 500]
    return HazardLegend(
        title="Wildfire burn density",
        unit="thousand-acre KDE units / 0.15° cell",
        source="WFIGS Interagency Perimeters 2016-2026 (NIFC), acres-weighted KDE",
        source_url="https://data-nifc.opendata.arcgis.com/",
        raw_min=raw_min,
        raw_max=raw_max,
        palette=palette,
        stops=stops,
        note="Real WFIGS fire perimeters on a 0.15° (~10 mi) grid, Gaussian-smoothed (sigma 0.22°) — tighter than tornado/hail since megafire scars are spatially distinct. Each fire weighted by acres burned (capped at 300k per fire). Coverage is 2020-present — reflects the modern fire regime, not historical baselines.",
    )
