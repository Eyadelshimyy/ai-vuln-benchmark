"""Hurricane wind-field impact analysis.

Given a storm, walk its track and estimate the local sustained wind across
each county, not just at the centroid. Sample points from the county polygon
are scored with a profile pinned to Vmax at Rmax and 64 kt at the directional
R64. The county comes back split into clear / tropical-storm / Cat 1–5 area
fractions so a corner clip does not stamp the whole county with the storm's
peak intensity.

County polygons are the us-atlas 10m topojson, fetched once per process.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from ..brand import USER_AGENT
from .hurdat2 import category_for_wind
from .county_wind import (
    SpeedBin,
    WindBand,
    bands_from_winds,
    experienced_wind_kt,
    include_county,
    radii_wind_kt,
    effective_rmax,
    sample_polygons,
    severity,
    skirt_wind_kt,
    speed_bins_from_winds,
)
from .ibtracs import (
    Storm,
    TrackPoint,
    lookup_r64_nm,
    lookup_r64_quads_nm,
    lookup_rmax_nm,
)

COUNTIES_TOPO_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"
FETCH_TIMEOUT_S = 30
DAMAGING_WIND_MULTIPLIER = 2.5  # radius of damaging winds = multiplier × Rmax

# Counties are listed from the area split (see county_wind.include_county),
# not from this threshold. Kept so older callers still import it.
MIN_IMPACT_WIND_KT = 85

# The wind-footprint VISUALISATION (translucent buffer on the map) spans the
# entire hurricane-strength lifecycle, including post-landfall track when the
# storm is still ≥ Cat 1. County loss uses the local wind at each sample,
# not this gate.
MIN_FOOTPRINT_WIND_KT = 64  # Saffir-Simpson Cat 1 threshold

# Conterminous US + PR + USVI bounding box. Track points outside this never
# touch counties we care about — saves doing 3,000 distance checks per point.
# The eastern / southern edges sit about one wind-field radius past St. Croix
# so a storm passing just south or east of the islands still paints them.
US_BBOX_LAT = (16.5, 50.0)
US_BBOX_LON = (-125.5, -62.5)


# FIPS state code → USPS postal abbreviation. Mirror of backend/scripts/build_geo.py.
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
    "56": "WY", "72": "PR",
}


@dataclass(slots=True, frozen=True)
class CountyMeta:
    geoid: str           # 5-digit FIPS, e.g. "12086"
    geography_id: str    # canonical "US-FL-12086"
    name: str
    state_usps: str
    centroid_lat: float
    centroid_lon: float


# geoid → (lat, lon) samples. Filled by county_centroids from the same download.
_AREA_SAMPLES: dict[str, list[tuple[float, float]]] = {}


def _decode_arcs(topology: dict) -> list[list[tuple[float, float]]]:
    """Decode the topojson `arcs` array into absolute (lon, lat) coordinate chains."""
    transform = topology.get("transform")
    if transform is None:
        return [
            [(pt[0], pt[1]) for pt in arc] for arc in topology["arcs"]
        ]
    sx, sy = transform["scale"]
    tx, ty = transform["translate"]
    decoded: list[list[tuple[float, float]]] = []
    for arc in topology["arcs"]:
        x = y = 0
        coords: list[tuple[float, float]] = []
        for dx, dy in arc:
            x += dx
            y += dy
            coords.append((x * sx + tx, y * sy + ty))
        decoded.append(coords)
    return decoded


def _resolve_ring(arc_ids: list[int], arcs: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Concatenate arcs by index (negative = reversed)."""
    ring: list[tuple[float, float]] = []
    for i, aid in enumerate(arc_ids):
        coords = arcs[~aid] if aid < 0 else arcs[aid]
        if aid < 0:
            coords = list(reversed(coords))
        if i == 0:
            ring.extend(coords)
        else:
            ring.extend(coords[1:])
    return ring


def _polygon_centroid(rings: list[list[tuple[float, float]]]) -> tuple[float, float]:
    """Shoelace centroid of a polygon (outer ring; holes ignored at this scale)."""
    outer = rings[0]
    a = cx = cy = 0.0
    n = len(outer)
    for i in range(n):
        x1, y1 = outer[i]
        x2, y2 = outer[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    a *= 0.5
    if abs(a) < 1e-12:
        # Degenerate polygon — fall back to vertex average.
        lon = sum(p[0] for p in outer) / n
        lat = sum(p[1] for p in outer) / n
        return lat, lon
    cx /= 6.0 * a
    cy /= 6.0 * a
    return cy, cx  # return (lat, lon)


def _county_geom_to_centroid(geom: dict, arcs: list) -> tuple[float, float] | None:
    if geom.get("type") == "Polygon":
        rings = [_resolve_ring(r, arcs) for r in geom["arcs"]]
        return _polygon_centroid(rings)
    if geom.get("type") == "MultiPolygon":
        # Use the largest polygon (by absolute area of the outer ring).
        biggest_area = -1.0
        biggest_rings: list[list[tuple[float, float]]] | None = None
        for poly in geom["arcs"]:
            rings = [_resolve_ring(r, arcs) for r in poly]
            outer = rings[0]
            a = 0.0
            for i in range(len(outer)):
                x1, y1 = outer[i]
                x2, y2 = outer[(i + 1) % len(outer)]
                a += x1 * y2 - x2 * y1
            if abs(a) > biggest_area:
                biggest_area = abs(a)
                biggest_rings = rings
        if biggest_rings is None:
            return None
        return _polygon_centroid(biggest_rings)
    return None


@lru_cache(maxsize=1)
def county_centroids() -> dict[str, CountyMeta]:
    """Live-fetch us-atlas counties + build a {geoid: CountyMeta} index."""
    req = urllib.request.Request(
        COUNTIES_TOPO_URL,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        topo = json.load(resp)
    arcs = _decode_arcs(topo)
    obj = topo["objects"]["counties"]
    out: dict[str, CountyMeta] = {}
    _AREA_SAMPLES.clear()
    for g in obj["geometries"]:
        if g.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        geoid = str(g.get("id", "")).zfill(5)
        state_fips = geoid[:2]
        usps = STATE_FIPS_TO_USPS.get(state_fips)
        if not usps:
            continue
        c = _county_geom_to_centroid(g, arcs)
        if c is None:
            continue
        lat, lon = c
        name = (g.get("properties") or {}).get("name", "")
        _AREA_SAMPLES[geoid] = _samples_for_geom(g, arcs, (lat, lon))
        out[geoid] = CountyMeta(
            geoid=geoid,
            geography_id=f"US-{usps}-{geoid}",
            name=name,
            state_usps=usps,
            centroid_lat=lat,
            centroid_lon=lon,
        )
    for geoid, name, (lat, lon), pts in _VI_DISTRICTS:
        _AREA_SAMPLES[geoid] = list(pts)
        out[geoid] = CountyMeta(
            geoid=geoid,
            geography_id=f"US-VI-{geoid}",
            name=name,
            state_usps="VI",
            centroid_lat=lat,
            centroid_lon=lon,
        )
    return out


# us-atlas carries Puerto Rico but not the US Virgin Islands, so PCS losses
# there (Irma, Maria) could never sit inside a wind field. Approximate island
# centres plus land points spread across each island (lat, lon).
_VI_DISTRICTS: tuple[
    tuple[str, str, tuple[float, float], tuple[tuple[float, float], ...]], ...
] = (
    ("78010", "St. Croix", (17.73, -64.75), (
        (17.73, -64.75), (17.71, -64.87), (17.74, -64.82), (17.70, -64.68),
        (17.75, -64.62), (17.76, -64.70),
    )),
    ("78020", "St. John", (18.33, -64.74), (
        (18.33, -64.74), (18.34, -64.78), (18.33, -64.70), (18.35, -64.72),
    )),
    ("78030", "St. Thomas", (18.34, -64.93), (
        (18.34, -64.93), (18.35, -65.00), (18.33, -64.88), (18.36, -64.95),
    )),
)


def county_area_samples() -> dict[str, list[tuple[float, float]]]:
    """Sample points per county. Empty when centroids were stubbed in tests."""
    county_centroids()
    return _AREA_SAMPLES


def _samples_for_geom(
    geom: dict,
    arcs: list,
    centroid: tuple[float, float],
) -> list[tuple[float, float]]:
    """Grid samples inside the county. Always includes the centroid so a
    sliver that the grid misses still has one point.
    """
    polys = [geom["arcs"]] if geom.get("type") == "Polygon" else list(geom.get("arcs") or [])
    polygons = [[_resolve_ring(r, arcs) for r in poly] for poly in polys]
    pts = sample_polygons(polygons)
    if not any(
        abs(p[0] - centroid[0]) < 1e-4 and abs(p[1] - centroid[1]) < 1e-4 for p in pts
    ):
        pts.append(centroid)
    return pts


# ─────────────────────────── geometry helpers ───────────────────────────


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    r_nm = 3440.065
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r_nm * math.asin(math.sqrt(a))


def rmax_nm(
    wind_kt: int,
    lat: float,
    *,
    storm_id: str | None = None,
    datetime_utc: str | None = None,
) -> tuple[float, str]:
    """Radius of maximum winds in nautical miles + the source we used.

    1. **IBTrACS** measured recon Rmax, if available for this (storm_id, time).
       Returned source ``"ibtracs"``.
    2. **Willoughby et al. (2006)** parametric estimate as fallback:

           Rmax(km) = 46.6 · exp(-0.0155 · Vmax(m/s) + 0.0169 · |lat|)

       Returned source ``"willoughby"``. Pre-1988 storms and missing fixes
       always fall back here since recon data is sparse before the modern era.

    Floors to 8 nm so very-weak/missing-wind points still get a non-zero
    radius for visualisation continuity.
    """
    observed = lookup_rmax_nm(storm_id, datetime_utc)
    if observed is not None and observed > 0:
        return max(8.0, observed), "ibtracs"
    if wind_kt <= 0:
        return 0.0, "willoughby"
    v_ms = wind_kt * 0.5144  # kt → m/s
    rmax_km = 46.6 * math.exp(-0.0155 * v_ms + 0.0169 * abs(lat))
    rmax = rmax_km / 1.852  # km → nm
    return max(8.0, rmax), "willoughby"


def _within_us_bbox(pt: TrackPoint) -> bool:
    return (
        US_BBOX_LAT[0] <= pt.lat <= US_BBOX_LAT[1]
        and US_BBOX_LON[0] <= pt.lon <= US_BBOX_LON[1]
    )


# ─────────────────────────── impact calculation ───────────────────────────


@dataclass(slots=True)
class ProgrammeContribution:
    """Per-programme slice of a single county's impacted TIV. ``projected_loss``
    is the TIV × damage_ratio at the county's max sustained wind."""

    dataset_id: str
    tiv: float
    location_count: int
    projected_loss: float = 0.0


@dataclass(slots=True)
class CountyImpact:
    geoid: str
    geography_id: str
    name: str
    state_usps: str
    centroid_lat: float
    centroid_lon: float
    max_wind_kt: int          # strongest LOCAL wind at any sample in the county
    max_category: int          # Saffir-Simpson of that local wind
    closest_distance_nm: float  # closest approach of the storm's eye
    rmax_at_closest_nm: float   # the Rmax we used for that point
    rmax_source: str            # 'ibtracs' (recon) | 'willoughby' (formula)
    tiv: float                  # joined from the user's selection (all segments)
    location_count: int
    has_data: bool             # true if any fact row exists for this county
    # Informational split. Loss calcs use ``tiv``, which already bundles these.
    residential_tiv: float = 0.0
    commercial_tiv: float = 0.0
    auto_tiv: float = 0.0
    # Area of the county in each local-wind band. The peak storm intensity
    # is NOT applied to the whole county. Empty when the caller built a
    # centroid-only impact (tests).
    wind_bands: list[WindBand] = None  # type: ignore[assignment]
    # Same samples, cut into the 10 mph bands Form V-1 publishes.
    speed_bins: list[SpeedBin] = None  # type: ignore[assignment]
    # Parametric damage ratio + projected ground-up loss for the user's
    # in-scope TIV. The router leaves these at 0; the UI applies the
    # underwriter's per-category ratios to ``wind_bands``.
    damage_ratio: float = 0.0
    projected_loss: float = 0.0
    by_programme: list[ProgrammeContribution] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_programme is None:
            self.by_programme = []
        if self.wind_bands is None:
            self.wind_bands = []
        if self.speed_bins is None:
            self.speed_bins = []


@dataclass(slots=True, frozen=True)
class FootprintPoint:
    """One contributing track point in the impact footprint.

    Carries the eyewall radius (Rmax) plus the asymmetric R64 wind field
    (per-quadrant radii of 64-kt winds, NE/SE/SW/NW). The outer cone uses
    the quadrants to draw a lopsided wind footprint and to capture counties
    using the actual measured wind extent at the bearing to the county.

      - ``rmax_nm``       — radius of maximum winds (eyewall)
      - ``r64_nm``        — mean R64 across non-zero quadrants (symmetric)
      - ``r64_quads_nm``  — (NE, SE, SW, NW) measured R64; None when no
                            IBTrACS data and we fall back to symmetric
      - ``r64_source``    — 'ibtracs' (measured) | 'fallback' (2.5×Rmax)
    """

    lat: float
    lon: float
    wind_kt: int
    rmax_nm: float
    radius_nm: float        # 2.5×Rmax — legacy "damaging-winds" radius
    rmax_source: str        # 'ibtracs' | 'willoughby'
    r64_nm: float
    r64_source: str         # 'ibtracs' (measured) | 'fallback' (2.5×Rmax)
    r64_quads_nm: tuple[float, float, float, float] | None  # NE, SE, SW, NW


@dataclass(slots=True, frozen=True)
class ConeQuad:
    """One tapered-quad segment of the wind-field cone between two adjacent
    footprint points. Four corners in (lon, lat) so the frontend can build a
    GeoJSON polygon directly; ``wind_kt`` drives the color via a Mapbox
    interpolate expression."""

    corners: tuple[
        tuple[float, float],  # left-of-A
        tuple[float, float],  # left-of-B
        tuple[float, float],  # right-of-B
        tuple[float, float],  # right-of-A
    ]
    wind_kt: int          # midpoint of the segment, for color interpolation
    start_wind_kt: int
    end_wind_kt: int


_NM_PER_DEG_LAT = 60.0


def _offset_latlon(
    lat: float, lon: float, distance_nm: float, bearing_deg: float
) -> tuple[float, float]:
    """Move (lat, lon) by ``distance_nm`` along compass bearing — small-distance
    flat-earth approximation. Accurate to <1% within typical Rmax distances."""
    if distance_nm <= 0:
        return lat, lon
    br = math.radians(bearing_deg)
    cos_lat = math.cos(math.radians(lat))
    d_lat = (distance_nm * math.cos(br)) / _NM_PER_DEG_LAT
    d_lon = (distance_nm * math.sin(br)) / (_NM_PER_DEG_LAT * max(cos_lat, 0.01))
    return lat + d_lat, lon + d_lon


def r64_at_bearing(
    quads: tuple[float, float, float, float] | None,
    bearing_deg: float,
    *,
    fallback_nm: float,
) -> float:
    """Linearly interpolate R64 across IBTrACS quadrants at a compass bearing.

    Quadrant centers (NHC convention): NE=45°, SE=135°, SW=225°, NW=315°.
    For an arbitrary bearing we find the two adjacent quadrants and weight
    them by angular distance, giving a smooth lopsided wind footprint
    instead of an abrupt boundary at the quadrant edges.

    Zero quadrants stay zero (correct: "no 64-kt winds in this direction"),
    so the cone tapers naturally to nothing where the storm wasn't a
    hurricane in that direction.
    """
    if quads is None:
        return fallback_nm
    shifted = (bearing_deg - 45.0) % 360.0  # 0=NE, 90=SE, 180=SW, 270=NW
    idx = int(shifted // 90.0) % 4
    frac = (shifted % 90.0) / 90.0
    return quads[idx] * (1 - frac) + quads[(idx + 1) % 4] * frac


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2 (degrees, 0=N, 90=E)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _quads_symmetric(
    footprint: list[FootprintPoint],
    half_width: Callable[[FootprintPoint], float],
) -> list[ConeQuad]:
    """Tapered quads using a single half-width per point (symmetric cone).
    Used for the inner Rmax cone where Rmax is by definition isotropic."""
    quads: list[ConeQuad] = []
    for i in range(len(footprint) - 1):
        a = footprint[i]
        b = footprint[i + 1]
        if a.lat == b.lat and a.lon == b.lon:
            continue
        if haversine_nm(a.lat, a.lon, b.lat, b.lon) > 300:
            continue
        bearing = _bearing_deg(a.lat, a.lon, b.lat, b.lon)
        left_bearing = (bearing - 90.0) % 360.0
        right_bearing = (bearing + 90.0) % 360.0
        wa, wb = half_width(a), half_width(b)
        if wa <= 0 or wb <= 0:
            continue
        la_lat, la_lon = _offset_latlon(a.lat, a.lon, wa, left_bearing)
        ra_lat, ra_lon = _offset_latlon(a.lat, a.lon, wa, right_bearing)
        lb_lat, lb_lon = _offset_latlon(b.lat, b.lon, wb, left_bearing)
        rb_lat, rb_lon = _offset_latlon(b.lat, b.lon, wb, right_bearing)
        quads.append(
            ConeQuad(
                corners=(
                    (la_lon, la_lat),
                    (lb_lon, lb_lat),
                    (rb_lon, rb_lat),
                    (ra_lon, ra_lat),
                ),
                wind_kt=(a.wind_kt + b.wind_kt) // 2,
                start_wind_kt=a.wind_kt,
                end_wind_kt=b.wind_kt,
            )
        )
    return quads


def _quads_asymmetric_r64(footprint: list[FootprintPoint]) -> list[ConeQuad]:
    """Outer cone quads with PER-SIDE width: each side of the track polygon
    uses the R64 value at THAT side's perpendicular bearing, interpolated
    between IBTrACS quadrants. Storms with no measured R64 fall back to
    the symmetric mean (which equals the synthetic 2.5×Rmax for those)."""
    quads: list[ConeQuad] = []
    for i in range(len(footprint) - 1):
        a = footprint[i]
        b = footprint[i + 1]
        if a.lat == b.lat and a.lon == b.lon:
            continue
        if haversine_nm(a.lat, a.lon, b.lat, b.lon) > 300:
            continue
        bearing = _bearing_deg(a.lat, a.lon, b.lat, b.lon)
        left_bearing = (bearing - 90.0) % 360.0
        right_bearing = (bearing + 90.0) % 360.0
        wa_left = r64_at_bearing(a.r64_quads_nm, left_bearing, fallback_nm=a.r64_nm)
        wa_right = r64_at_bearing(a.r64_quads_nm, right_bearing, fallback_nm=a.r64_nm)
        wb_left = r64_at_bearing(b.r64_quads_nm, left_bearing, fallback_nm=b.r64_nm)
        wb_right = r64_at_bearing(b.r64_quads_nm, right_bearing, fallback_nm=b.r64_nm)
        if max(wa_left, wa_right, wb_left, wb_right) <= 0:
            continue
        # Tiny epsilon so a zero-quadrant doesn't draw a degenerate spike.
        wa_left = max(wa_left, 0.1)
        wa_right = max(wa_right, 0.1)
        wb_left = max(wb_left, 0.1)
        wb_right = max(wb_right, 0.1)
        la_lat, la_lon = _offset_latlon(a.lat, a.lon, wa_left, left_bearing)
        ra_lat, ra_lon = _offset_latlon(a.lat, a.lon, wa_right, right_bearing)
        lb_lat, lb_lon = _offset_latlon(b.lat, b.lon, wb_left, left_bearing)
        rb_lat, rb_lon = _offset_latlon(b.lat, b.lon, wb_right, right_bearing)
        quads.append(
            ConeQuad(
                corners=(
                    (la_lon, la_lat),
                    (lb_lon, lb_lat),
                    (rb_lon, rb_lat),
                    (ra_lon, ra_lat),
                ),
                # The annulus is not the eyewall. Color it as the wind
                # halfway out to 64 kt, not as the storm's peak.
                wind_kt=int(round((
                    skirt_wind_kt(a.wind_kt, a.rmax_nm, a.r64_nm)
                    + skirt_wind_kt(b.wind_kt, b.rmax_nm, b.r64_nm)
                ) / 2)),
                start_wind_kt=a.wind_kt,
                end_wind_kt=b.wind_kt,
            )
        )
    return quads


def _asymmetric_ring(
    lat: float,
    lon: float,
    quads: tuple[float, float, float, float] | None,
    fallback_nm: float,
    steps: int = 48,
) -> list[list[float]]:
    """Build a closed (lon, lat) polygon ring whose distance from (lat, lon)
    follows the per-bearing R64 — an asymmetric "egg" around each fix that
    blends with the asymmetric cone quads into a single seamless footprint.
    """
    ring: list[list[float]] = []
    for i in range(steps + 1):
        bearing = (360.0 * i) / steps
        r = r64_at_bearing(quads, bearing, fallback_nm=fallback_nm)
        if r < 0.1:
            r = 0.1
        pt_lat, pt_lon = _offset_latlon(lat, lon, r, bearing)
        ring.append([round(pt_lon, 4), round(pt_lat, 4)])
    return ring


def _build_cones(
    footprint: list[FootprintPoint],
) -> tuple[list[ConeQuad], list[ConeQuad]]:
    """(inner cone @ Rmax eyewall, outer cone @ asymmetric R64)."""
    inner = _quads_symmetric(footprint, lambda p: p.rmax_nm)
    outer = _quads_asymmetric_r64(footprint)
    return inner, outer


@dataclass
class _Paint:
    """Running local-wind samples for one county across the track."""

    meta: CountyMeta
    points: list[tuple[float, float]]
    winds: list[int]
    closest_nm: float = 1e9
    rmax_nm: float = 0.0
    rmax_source: str = ""


def _paint_fix(
    painted: dict[str, _Paint],
    centroids: dict[str, CountyMeta],
    samples: dict[str, list[tuple[float, float]]],
    *,
    eye_lat: float,
    eye_lon: float,
    vmax_kt: int,
    rmax: float,
    rmax_source: str,
    r64_nm: float,
    r64_quads: tuple[float, float, float, float] | None,
    r34_nm: float = 0.0,
    r34_quads: tuple[float, float, float, float] | None = None,
    r50_nm: float = 0.0,
    r50_quads: tuple[float, float, float, float] | None = None,
    model_wind=None,
) -> None:
    """Raise the peak local wind at every sample this fix can still reach.

    Live advisories are pinned to the NHC 64/50/34 kt radii. The forecast
    cone is track uncertainty, not a wind field, so it does not set the
    wind. Outside the 50 kt radius an optional GFS/ECMWF cap can only
    lower the wind, never raise it. Historical storms stay on the drawn
    64 kt field.
    """
    has_r34 = r34_nm > 0 or (r34_quads is not None and any(v > 0 for v in r34_quads))
    has_r50 = r50_nm > 0 or (r50_quads is not None and any(v > 0 for v in r50_quads))
    if has_r34:
        r34_max = r34_nm
        if r34_quads:
            r34_max = max(r34_max, max(r34_quads))
        reach = min(200.0, r34_max + 30.0)
    else:
        r64_max = r64_nm
        if r64_quads:
            r64_max = max(r64_max, max(r64_quads))
        reach = min(180.0, r64_max + 70.0)
    if reach <= 0:
        return
    deg = reach / 50.0
    lat_lo, lat_hi = eye_lat - deg, eye_lat + deg
    lon_lo, lon_hi = eye_lon - deg, eye_lon + deg
    for meta in centroids.values():
        if not (lat_lo <= meta.centroid_lat <= lat_hi and lon_lo <= meta.centroid_lon <= lon_hi):
            continue
        d_cent = haversine_nm(eye_lat, eye_lon, meta.centroid_lat, meta.centroid_lon)
        if d_cent > reach:
            continue
        acc = painted.get(meta.geoid)
        if acc is None:
            pts = samples.get(meta.geoid) or [(meta.centroid_lat, meta.centroid_lon)]
            acc = _Paint(meta=meta, points=pts, winds=[0] * len(pts))
            painted[meta.geoid] = acc
        if d_cent < acc.closest_nm:
            acc.closest_nm = d_cent
            acc.rmax_nm = rmax
            acc.rmax_source = rmax_source
        for i, (lat, lon) in enumerate(acc.points):
            d = haversine_nm(eye_lat, eye_lon, lat, lon)
            bearing = _bearing_deg(eye_lat, eye_lon, lat, lon)
            r64_here = r64_at_bearing(r64_quads, bearing, fallback_nm=r64_nm)
            if has_r34:
                r34_here = r64_at_bearing(r34_quads, bearing, fallback_nm=r34_nm)
                r50_here = (
                    r64_at_bearing(r50_quads, bearing, fallback_nm=r50_nm)
                    if has_r50 else 0.0
                )
                r64_use = r64_here if r64_nm > 0 or r64_quads else 0.0
                rmax_use = effective_rmax(rmax, r64_use, r50_here, r34_here)
                wind = radii_wind_kt(
                    d, vmax_kt, rmax_use,
                    r64_use,
                    r50_here,
                    r34_here,
                )
                if model_wind is not None and (r64_use <= 0 or d > r64_use):
                    # GFS and ECMWF cap everything outside the published
                    # hurricane core. A 65 kt official center with no R64 is
                    # a broad circulation, and both models have that well
                    # under hurricane force once it is inland.
                    cap = model_wind(lat, lon)
                    if cap is not None and cap < wind:
                        wind = int(round(cap))
            else:
                # Outside the drawn hurricane field this sample is clear.
                if r64_here <= 0 or d > r64_here + 0.5:
                    continue
                wind = experienced_wind_kt(d, vmax_kt, rmax, r64_here)
            if wind < 34:
                continue
            if wind > acc.winds[i]:
                acc.winds[i] = wind


def _impacts_from_paint(painted: dict[str, _Paint]) -> list[CountyImpact]:
    impacts: list[CountyImpact] = []
    for acc in painted.values():
        bands = bands_from_winds(acc.winds)
        bins = speed_bins_from_winds(acc.winds)
        peak = max(acc.winds) if acc.winds else 0
        if not include_county(bands, peak):
            continue
        impacts.append(
            CountyImpact(
                geoid=acc.meta.geoid,
                geography_id=acc.meta.geography_id,
                name=acc.meta.name,
                state_usps=acc.meta.state_usps,
                centroid_lat=acc.meta.centroid_lat,
                centroid_lon=acc.meta.centroid_lon,
                max_wind_kt=peak,
                max_category=category_for_wind(peak) if peak >= 34 else -1,
                closest_distance_nm=acc.closest_nm if acc.closest_nm < 1e8 else 0.0,
                rmax_at_closest_nm=acc.rmax_nm,
                rmax_source=acc.rmax_source,
                tiv=0.0,
                location_count=0,
                has_data=False,
                wind_bands=bands,
                speed_bins=bins,
            )
        )
    impacts.sort(key=lambda i: (-severity(i.wind_bands), -i.max_wind_kt))
    return impacts


@dataclass
class _Stamp:
    """One position along the track that paints county samples."""

    lat: float
    lon: float
    vmax_kt: int
    rmax: float
    rmax_source: str
    r64_nm: float
    r64_quads: tuple[float, float, float, float] | None
    r50_nm: float = 0.0
    r50_quads: tuple[float, float, float, float] | None = None
    r34_nm: float = 0.0
    r34_quads: tuple[float, float, float, float] | None = None


def _densify_track(stamps: list[_Stamp], step_nm: float = 12.0) -> list[_Stamp]:
    """Fill gaps between 6-hour fixes so the eyewall swath is continuous.

    A point the eye crosses between two fixes would otherwise be scored at
    whichever vertex is closer, and a 15 nm Rmax can miss it entirely.
    The drawn footprint stays on the original fixes.
    """
    if len(stamps) < 2:
        return list(stamps)
    out: list[_Stamp] = [stamps[0]]
    for a, b in zip(stamps, stamps[1:]):
        dist = haversine_nm(a.lat, a.lon, b.lat, b.lon)
        n = int(dist // step_nm) if step_nm > 0 else 0
        if dist > 300 or n < 2:
            out.append(b)
            continue
        for i in range(1, n):
            t = i / n
            if a.r64_quads is not None and b.r64_quads is not None:
                quads: tuple[float, float, float, float] | None = (
                    a.r64_quads[0] * (1 - t) + b.r64_quads[0] * t,
                    a.r64_quads[1] * (1 - t) + b.r64_quads[1] * t,
                    a.r64_quads[2] * (1 - t) + b.r64_quads[2] * t,
                    a.r64_quads[3] * (1 - t) + b.r64_quads[3] * t,
                )
            else:
                quads = a.r64_quads if t < 0.5 else b.r64_quads
            if a.r34_quads is not None and b.r34_quads is not None:
                q34: tuple[float, float, float, float] | None = (
                    a.r34_quads[0] * (1 - t) + b.r34_quads[0] * t,
                    a.r34_quads[1] * (1 - t) + b.r34_quads[1] * t,
                    a.r34_quads[2] * (1 - t) + b.r34_quads[2] * t,
                    a.r34_quads[3] * (1 - t) + b.r34_quads[3] * t,
                )
            else:
                q34 = a.r34_quads if t < 0.5 else b.r34_quads
            if a.r50_quads is not None and b.r50_quads is not None:
                q50: tuple[float, float, float, float] | None = (
                    a.r50_quads[0] * (1 - t) + b.r50_quads[0] * t,
                    a.r50_quads[1] * (1 - t) + b.r50_quads[1] * t,
                    a.r50_quads[2] * (1 - t) + b.r50_quads[2] * t,
                    a.r50_quads[3] * (1 - t) + b.r50_quads[3] * t,
                )
            else:
                q50 = a.r50_quads if t < 0.5 else b.r50_quads
            out.append(
                _Stamp(
                    lat=a.lat + (b.lat - a.lat) * t,
                    lon=a.lon + (b.lon - a.lon) * t,
                    vmax_kt=int(round(a.vmax_kt + (b.vmax_kt - a.vmax_kt) * t)),
                    rmax=a.rmax + (b.rmax - a.rmax) * t,
                    rmax_source=a.rmax_source if t < 0.5 else b.rmax_source,
                    r64_nm=a.r64_nm + (b.r64_nm - a.r64_nm) * t,
                    r64_quads=quads,
                    r50_nm=a.r50_nm + (b.r50_nm - a.r50_nm) * t,
                    r50_quads=q50,
                    r34_nm=a.r34_nm + (b.r34_nm - a.r34_nm) * t,
                    r34_quads=q34,
                )
            )
        out.append(b)
    return out


def compute_impact(
    storm: Storm,
    *,
    multiplier: float = DAMAGING_WIND_MULTIPLIER,
    model_wind=None,
    ts_field: bool = False,
    centroids: dict[str, CountyMeta] | None = None,
    samples: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[
    list[CountyImpact],
    list[FootprintPoint],
    list[ConeQuad],
    list[ConeQuad],
    list[dict],
]:
    """Walk the storm and return counties that actually sit in its wind field.

    Each county is split by the local wind at sample points across its
    polygon (core, hurricane fringe, tropical-storm skirt, untouched).
    The map footprint is unchanged: it still draws every hurricane-strength
    fix. Returned counties carry no TIV yet.

    ``ts_field`` also paints historical tropical-storm-force wind from the
    IBTrACS best-track R34/R50 quadrants, including fixes that never reach
    hurricane strength. Without it a historical storm only scores its 64 kt
    field, so wind loss in the 34-63 kt skirt (most of Debby, Isaias, inland
    Helene) is invisible to loss estimation and calibration.

    ``centroids`` / ``samples`` replace the county grid, e.g. one pseudo
    county per weather station to read the modelled wind at that point.
    """
    if centroids is None:
        centroids = county_centroids()
        samples = county_area_samples()
    elif samples is None:
        samples = {}
    painted: dict[str, _Paint] = {}
    footprint: list[FootprintPoint] = []
    stamps: list[_Stamp] = []

    for pt in storm.track:
        if not _within_us_bbox(pt):
            continue
        # Two filters for "is this a hurricane right now":
        #   - wind ≥ 64 kt (Cat 1)
        #   - IBTrACS USA_STATUS == "HU" (true tropical hurricane phase)
        # The second filter matters because IBTrACS reports a much larger
        # Rmax once the storm goes extratropical (EX) — Michael 2018 jumps
        # from 15 nm during landfall to 120 nm a few days later as the
        # wind field reorganises. Without the status filter we'd render an
        # absurdly large post-tropical cone.
        live_radii = pt.radii_source == "nhc"
        best_track_radii = ts_field and not live_radii
        r34_quads = pt.r34_quads_nm if live_radii or best_track_radii else None
        has_r34 = bool(r34_quads and any(v > 0 for v in r34_quads))
        # Hurricane-force fixes draw the cone. A live advisory below 64 kt
        # still counts when NHC published a 34 kt field — that is what
        # reaches the coast while the core is offshore.
        if pt.wind_kt < 34:
            continue
        if pt.wind_kt < MIN_FOOTPRINT_WIND_KT and not has_r34:
            continue
        # IBTrACS post-tropical (EX) Rmax inflates wildly; skip those.
        # Live NHC official points carry their own radii and are not EX.
        if best_track_radii:
            # ASSUMPTION: post-tropical / low fixes count only when the best
            # track carries a 34 kt field (Sandy 2012 came ashore as EX).
            # Their wind is then pinned to the radii, so the inflated EX Rmax
            # cannot widen the core.
            if pt.status not in ("HU", "TS", "SS") and not has_r34:
                continue
        elif not live_radii and (pt.status != "HU" or pt.wind_kt < MIN_FOOTPRINT_WIND_KT):
            continue
        if live_radii and pt.rmax_nm and pt.rmax_nm > 0:
            rmax, rmax_src = max(8.0, pt.rmax_nm), "nhc"
        else:
            rmax, rmax_src = rmax_nm(
                pt.wind_kt,
                pt.lat,
                storm_id=None if live_radii else storm.storm_id,
                datetime_utc=None if live_radii else pt.datetime_utc,
            )
        radius = rmax * multiplier
        # R64 — live NHC quadrants first, then IBTrACS, then 2.5×Rmax.
        # Never call IBTrACS for a live NHC storm (current year is not in
        # the archive and the lookup would pull the 70 MB CSV).
        if live_radii and pt.r64_quads_nm and any(v > 0 for v in pt.r64_quads_nm):
            measured_quads = pt.r64_quads_nm
            nonzero = [v for v in measured_quads if v > 0]
            r64 = sum(nonzero) / len(nonzero)
            r64_src = "nhc"
        elif live_radii:
            # No published hurricane radius. Do not invent 2.5×Rmax — that
            # painted a 63 kt ring around a tropical storm.
            measured_quads = None
            r64 = 0.0
            r64_src = "none"
        elif pt.wind_kt < MIN_FOOTPRINT_WIND_KT:
            measured_quads = None
            r64 = 0.0
            r64_src = "none"
        else:
            measured_r64 = lookup_r64_nm(storm.storm_id, pt.datetime_utc)
            measured_quads = lookup_r64_quads_nm(storm.storm_id, pt.datetime_utc)
            if measured_r64 and measured_r64 > 0:
                r64 = measured_r64
                r64_src = "ibtracs"
            else:
                r64 = radius
                r64_src = "fallback"
                measured_quads = None
        r34_mean = 0.0
        if has_r34 and r34_quads is not None:
            nonzero34 = [v for v in r34_quads if v > 0]
            r34_mean = sum(nonzero34) / len(nonzero34)
        r50_quads = pt.r50_quads_nm if live_radii or best_track_radii else None
        r50_mean = 0.0
        if r50_quads and any(v > 0 for v in r50_quads):
            nonzero50 = [v for v in r50_quads if v > 0]
            r50_mean = sum(nonzero50) / len(nonzero50)
        else:
            r50_quads = None
        draws_cone = pt.wind_kt >= MIN_FOOTPRINT_WIND_KT
        if draws_cone:
            footprint.append(
                FootprintPoint(
                    lat=pt.lat,
                    lon=pt.lon,
                    wind_kt=pt.wind_kt,
                    rmax_nm=rmax,
                    radius_nm=radius,
                    rmax_source=rmax_src,
                    r64_nm=r64,
                    r64_source=r64_src,
                    r64_quads_nm=measured_quads,
                )
            )
        stamps.append(
            _Stamp(
                lat=pt.lat,
                lon=pt.lon,
                vmax_kt=pt.wind_kt,
                rmax=rmax,
                rmax_source=rmax_src,
                r64_nm=r64,
                r64_quads=measured_quads,
                r50_nm=r50_mean,
                r50_quads=r50_quads,
                r34_nm=r34_mean,
                r34_quads=r34_quads,
            )
        )

    # Score the swath, not just the 6-hour vertices. A corner of a county can
    # sit in the tropical-storm skirt of a compact major while the centroid
    # never sees the storm's peak.
    for stamp in _densify_track(stamps):
        _paint_fix(
            painted,
            centroids,
            samples,
            eye_lat=stamp.lat,
            eye_lon=stamp.lon,
            vmax_kt=stamp.vmax_kt,
            rmax=stamp.rmax,
            rmax_source=stamp.rmax_source,
            r64_nm=stamp.r64_nm,
            r64_quads=stamp.r64_quads,
            r50_nm=stamp.r50_nm,
            r50_quads=stamp.r50_quads,
            r34_nm=stamp.r34_nm,
            r34_quads=stamp.r34_quads,
            model_wind=model_wind,
        )

    impacts = _impacts_from_paint(painted)
    inner_cone, outer_cone = _build_cones(footprint)
    # Asymmetric "egg" polygons around each footprint point — the caps that
    # blend with the asymmetric cone quads into a seamless wind field.
    outer_rings: list[dict] = []
    for pt in footprint:
        ring = _asymmetric_ring(pt.lat, pt.lon, pt.r64_quads_nm, fallback_nm=pt.r64_nm)
        outer_rings.append(
            {
                "ring": ring,
                "wind_kt": skirt_wind_kt(pt.wind_kt, pt.rmax_nm, pt.r64_nm),
                "r64_nm": pt.r64_nm,
                "r64_source": pt.r64_source,
            }
        )
    return (
        impacts,
        footprint,
        inner_cone,
        outer_cone,
        outer_rings,
    )


def join_tiv(
    impacts: list[CountyImpact],
    facts: Iterable,
) -> list[CountyImpact]:
    """Add TIV + location_count from fact rows whose geography_id matches.

    Facts is iterable of ExposureFactNormalized — we only sum COUNTY-grain
    rows. State-grain rows are intentionally ignored here (state-rollup would
    be misleading for a county-level wind-field).

    Also produces a per-programme breakdown (``by_programme``) so the right-rail
    detail view can expand each county and show which deals contributed.
    """
    by_geo: dict[str, tuple[float, int]] = {}
    # Two-level index: geo -> dataset_id -> (tiv, loc)
    by_geo_prog: dict[str, dict[str, tuple[float, int]]] = {}
    # Informational only. Residential, commercial, and auto are not separate losses.
    by_geo_seg: dict[str, dict[str, float]] = {}
    for f in facts:
        gid = getattr(f, "geography_id", None)
        agg = getattr(f, "aggregation", None)
        if gid is None or agg != "COUNTY":
            continue
        cur_tiv, cur_loc = by_geo.get(gid, (0.0, 0))
        by_geo[gid] = (cur_tiv + (f.tiv or 0.0), cur_loc + (f.location_count or 0))
        ds_id = getattr(f, "dataset_id", "(unknown)")
        prog_map = by_geo_prog.setdefault(gid, {})
        cur_p_tiv, cur_p_loc = prog_map.get(ds_id, (0.0, 0))
        prog_map[ds_id] = (cur_p_tiv + (f.tiv or 0.0), cur_p_loc + (f.location_count or 0))
        seg = getattr(f, "occupancy_segment", None) or ""
        seg = seg.value if hasattr(seg, "value") else str(seg)
        seg_map = by_geo_seg.setdefault(gid, {})
        seg_map[seg] = seg_map.get(seg, 0.0) + float(f.tiv or 0.0)

    # NOTE: damage_ratio + projected_loss are intentionally NOT computed
    # server-side. The user supplies their own mean + SD per Saffir-Simpson
    # category from the impact panel; the frontend applies it to each
    # county's TIV (and per-programme TIV) to produce a probabilistic loss
    # range. The legacy fields stay on the wire for back-compat but are
    # always zero in the API response.
    for imp in impacts:
        if imp.geography_id in by_geo:
            tiv, loc = by_geo[imp.geography_id]
            imp.tiv = tiv
            imp.location_count = loc
            imp.has_data = True
            seg_map = by_geo_seg.get(imp.geography_id, {})
            imp.residential_tiv = float(seg_map.get("RESIDENTIAL", 0.0))
            imp.commercial_tiv = float(seg_map.get("COMMERCIAL", 0.0))
            imp.auto_tiv = float(seg_map.get("AUTO", 0.0))
            prog_map = by_geo_prog.get(imp.geography_id, {})
            imp.by_programme = [
                ProgrammeContribution(dataset_id=ds, tiv=t, location_count=lc)
                for ds, (t, lc) in sorted(prog_map.items(), key=lambda kv: -kv[1][0])
            ]
    return impacts
