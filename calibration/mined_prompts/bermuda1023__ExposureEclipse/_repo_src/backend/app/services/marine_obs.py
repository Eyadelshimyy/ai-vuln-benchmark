"""Marine + surface weather observations from NOAA.

Two sources, both free + no auth:

- **NDBC** (National Data Buoy Center) latest_obs.txt — one fixed-width
  file with every station's latest fix; ~900 stations Atlantic + global.
- **NWS observations** via api.weather.gov — METAR/airport surface
  observations near a point.

Both are filtered down to a bounding box so the response stays small
when overlaid alongside a hurricane's forecast cone.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache

from ..brand import USER_AGENT as NWS_USER_AGENT

NDBC_LATEST_URL = "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt"
NWS_STATIONS_URL = "https://api.weather.gov/stations"
FETCH_TIMEOUT_S = 30


@dataclass(slots=True, frozen=True)
class BuoyObservation:
    station_id: str
    lat: float
    lon: float
    wind_kt: float | None
    wind_dir_deg: float | None
    gust_kt: float | None
    wave_height_ft: float | None
    pressure_mb: float | None
    air_temp_f: float | None
    water_temp_f: float | None
    observed_at: str           # ISO; UTC


@dataclass(slots=True, frozen=True)
class LandObservation:
    station_id: str
    name: str
    lat: float
    lon: float
    wind_kt: float | None
    wind_dir_deg: float | None
    gust_kt: float | None
    pressure_mb: float | None
    temp_f: float | None
    observed_at: str


# ─────────────────────────── NDBC ───────────────────────────


def _mps_to_kt(v: float | None) -> float | None:
    return v * 1.94384 if v is not None else None


def _c_to_f(v: float | None) -> float | None:
    return v * 9 / 5 + 32 if v is not None else None


def _m_to_ft(v: float | None) -> float | None:
    return v * 3.28084 if v is not None else None


def _parse_ndbc_float(s: str) -> float | None:
    s = s.strip()
    if not s or s == "MM":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v


_NDBC_CACHE: dict[str, tuple[float, list[BuoyObservation]]] = {}
# NDBC updates latest_obs.txt roughly every 10 minutes. TTL a bit shorter
# than that so we never serve data more than one publish-cycle stale.
_NDBC_CACHE_TTL_S = 8 * 60


def _ndbc_all() -> list[BuoyObservation]:
    """Parse the NDBC latest_obs.txt. Cached for ~8 min so we never serve
    peak-storm readings that have since fallen off (Vercel serverless
    containers stay warm for many minutes; the previous @lru_cache had no
    TTL, so a KSPR reading of 47 kt gusting 69 during peak Bertha winds
    stayed pinned for the container's whole life even after actual winds
    dropped to 11 gusting 17 kt)."""
    import time as _t
    now = _t.time()
    hit = _NDBC_CACHE.get("all")
    if hit is not None and (now - hit[0]) < _NDBC_CACHE_TTL_S:
        return hit[1]

    req = urllib.request.Request(NDBC_LATEST_URL, headers={"User-Agent": NWS_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        # Serve stale data rather than empty on a transient NDBC blip —
        # a 15-min-old reading is better than nothing.
        return hit[1] if hit is not None else []

    out: list[BuoyObservation] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # Fixed-width-ish; split on whitespace works because the data file
        # uses MM for missing and the columns never overlap.
        parts = line.split()
        # Expected columns:
        # STN LAT LON YYYY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES PTDY ATMP WTMP DEWP VIS TIDE
        if len(parts) < 21:
            continue
        try:
            lat = float(parts[1])
            lon = float(parts[2])
            yyyy, mm, dd, hh, mn = parts[3:8]
            iso = f"{yyyy}-{int(mm):02d}-{int(dd):02d}T{int(hh):02d}:{int(mn):02d}:00Z"
        except (ValueError, IndexError):
            continue
        wdir = _parse_ndbc_float(parts[8])
        wspd_ms = _parse_ndbc_float(parts[9])
        gst_ms = _parse_ndbc_float(parts[10])
        wvht_m = _parse_ndbc_float(parts[11])
        pres = _parse_ndbc_float(parts[15])
        atmp = _parse_ndbc_float(parts[17])
        wtmp = _parse_ndbc_float(parts[18])
        out.append(
            BuoyObservation(
                station_id=parts[0],
                lat=lat,
                lon=lon,
                wind_kt=_mps_to_kt(wspd_ms),
                wind_dir_deg=wdir,
                gust_kt=_mps_to_kt(gst_ms),
                wave_height_ft=_m_to_ft(wvht_m),
                pressure_mb=pres,
                air_temp_f=_c_to_f(atmp),
                water_temp_f=_c_to_f(wtmp),
                observed_at=iso,
            )
        )
    _NDBC_CACHE["all"] = (now, out)
    return out


def buoys_in_bbox(
    west: float, south: float, east: float, north: float
) -> list[BuoyObservation]:
    return [
        b for b in _ndbc_all()
        if south <= b.lat <= north and west <= b.lon <= east
    ]


# ─────────────────────────── NWS land stations ───────────────────────────


def _nws_get(
    path: str,
    params: dict | None = None,
    *,
    timeout_s: int = FETCH_TIMEOUT_S,
) -> dict | None:
    """GET against api.weather.gov; returns parsed JSON or None on error.
    ``timeout_s`` can be overridden by the caller — the /stations pagination
    is happy to wait ~30 s but per-station latest-obs fetches want to fail
    fast so a stuck endpoint doesn't hold up a whole 500-station batch."""
    url = f"https://api.weather.gov{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": NWS_USER_AGENT,
            "Accept": "application/geo+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


# Bounding boxes for every US state / territory NWS serves. Coarse but
# comprehensive — an actual bbox library would be overkill here since we
# just need to know which states' stations to fetch for a storm.
# Values are (west, south, east, north) in lat/lon degrees.
_STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "AL": (-88.5, 30.2, -84.9, 35.0),
    "AR": (-94.6, 33.0, -89.6, 36.5),
    "AZ": (-114.8, 31.3, -109.0, 37.0),
    "CA": (-124.5, 32.5, -114.1, 42.0),
    "CO": (-109.1, 37.0, -102.0, 41.0),
    "CT": (-73.8, 40.9, -71.8, 42.1),
    "DC": (-77.15, 38.79, -76.90, 39.00),
    "DE": (-75.8, 38.4, -75.0, 39.9),
    "FL": (-87.6, 24.5, -80.0, 31.0),
    "GA": (-85.6, 30.4, -80.8, 35.0),
    "IA": (-96.7, 40.4, -90.1, 43.6),
    "ID": (-117.3, 42.0, -111.0, 49.0),
    "IL": (-91.6, 36.9, -87.5, 42.6),
    "IN": (-88.1, 37.8, -84.8, 41.8),
    "KS": (-102.1, 36.9, -94.6, 40.0),
    "KY": (-89.6, 36.5, -81.9, 39.2),
    "LA": (-94.1, 28.9, -88.8, 33.0),
    "MA": (-73.5, 41.2, -69.9, 42.9),
    "MD": (-79.5, 37.9, -75.0, 39.7),
    "ME": (-71.1, 43.0, -66.9, 47.5),
    "MI": (-90.4, 41.7, -82.4, 48.3),
    "MN": (-97.2, 43.4, -89.5, 49.4),
    "MO": (-95.8, 36.0, -89.1, 40.6),
    "MS": (-91.7, 30.2, -88.1, 35.0),
    "MT": (-116.1, 44.4, -104.0, 49.0),
    "NC": (-84.4, 33.8, -75.4, 36.6),
    "ND": (-104.1, 45.9, -96.6, 49.0),
    "NE": (-104.1, 40.0, -95.3, 43.0),
    "NH": (-72.6, 42.7, -70.6, 45.3),
    "NJ": (-75.6, 38.9, -73.9, 41.4),
    "NM": (-109.1, 31.3, -103.0, 37.0),
    "NV": (-120.0, 35.0, -114.0, 42.0),
    "NY": (-79.8, 40.5, -71.9, 45.0),
    "OH": (-84.8, 38.4, -80.5, 42.0),
    "OK": (-103.0, 33.6, -94.4, 37.0),
    "OR": (-124.6, 42.0, -116.5, 46.3),
    "PA": (-80.5, 39.7, -74.7, 42.3),
    "PR": (-67.3, 17.9, -65.2, 18.5),
    "RI": (-71.9, 41.1, -71.1, 42.0),
    "SC": (-83.4, 32.0, -78.5, 35.2),
    "SD": (-104.1, 42.5, -96.4, 45.9),
    "TN": (-90.3, 34.9, -81.6, 36.7),
    "TX": (-106.6, 25.8, -93.5, 36.5),
    "UT": (-114.1, 37.0, -109.0, 42.0),
    "VA": (-83.7, 36.5, -75.2, 39.5),
    "VT": (-73.4, 42.7, -71.5, 45.0),
    "WA": (-124.8, 45.5, -116.9, 49.0),
    "WI": (-92.9, 42.5, -86.8, 47.1),
    "WV": (-82.7, 37.2, -77.7, 40.6),
    "WY": (-111.1, 41.0, -104.1, 45.0),
}


def _states_overlapping_bbox(
    west: float, south: float, east: float, north: float,
) -> list[str]:
    """USPS codes for every state whose bbox overlaps the query bbox."""
    out: list[str] = []
    for code, (w, s, e, n) in _STATE_BBOXES.items():
        if not (e < west or w > east or n < south or s > north):
            out.append(code)
    return out


@lru_cache(maxsize=64)
def _nws_stations_for_state(state: str) -> tuple[dict, ...]:
    """All stations NWS knows about in ``state`` (USPS code). Cached per
    state so a Gulf-storm bbox spanning 6 states = 6 fast API calls."""
    stations: list[dict] = []
    next_url: str | None = None
    for _ in range(20):  # per-state cap; TX has ~2500, most much less
        if next_url is None:
            page = _nws_get("/stations", params={"limit": 500, "state": state})
        else:
            page = _nws_get_absolute(next_url)
        if not page:
            break
        features = page.get("features") or []
        stations.extend(features)
        if len(features) < 500:
            break
        next_url = (page.get("pagination") or {}).get("next") or None
        if not next_url:
            break
    return tuple(stations)


def _nws_stations_in_bbox(
    west: float, south: float, east: float, north: float,
) -> list[dict]:
    """Union of station lists for every state overlapping the bbox.
    Replaces the previous "fetch all NWS stations globally then filter"
    pattern which only reached the numeric-prefixed sites — the K-prefixed
    ASOS/AWOS at every major airport sort after 'C' and never got pulled."""
    states = _states_overlapping_bbox(west, south, east, north)
    if not states:
        return []
    stations: list[dict] = []
    for state in states:
        stations.extend(_nws_stations_for_state(state))
    return stations


# Old API kept for callers that don't yet pass a bbox. Falls back to the
# US mega-bbox so we at least fetch every CONUS + territory state — much
# broader than any real storm bbox but bounded, cached per-state.
def _nws_all_stations() -> list[dict]:
    """Deprecated: prefer ``_nws_stations_in_bbox`` when a bbox is known."""
    return _nws_stations_in_bbox(-180.0, 17.0, -65.0, 50.0)


def _nws_get_absolute(url: str) -> dict | None:
    """Fetch an absolute URL against api.weather.gov — used to follow
    pagination cursors verbatim without our own URL-encoding step
    corrupting them."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": NWS_USER_AGENT,
            "Accept": "application/geo+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _nws_speed_to_kt(field: dict | None) -> float | None:
    """Convert an NWS windSpeed/windGust value block to knots. NWS's default
    unitCode is ``wmoUnit:km_h-1`` (km/h), which the old code was silently
    treating as m/s and then multiplying by 1.94 — so a legitimate 12-kt
    reading came out as 43 kt (12 kt → 22 km/h → 22×1.94 ≈ 43). Some
    stations still report ``wmoUnit:m_s-1`` (m/s), so we branch on the
    actual unit code instead of assuming."""
    if not field:
        return None
    v = field.get("value")
    if v is None:
        return None
    unit = (field.get("unitCode") or "").lower()
    if "km_h" in unit or "kilometer" in unit:
        return float(v) * 0.539957        # km/h → kt
    if "m_s" in unit or "meter" in unit:
        return float(v) * 1.94384         # m/s → kt
    if "mile" in unit or "mph" in unit:
        return float(v) * 0.868976        # mph → kt
    if "kn" in unit:
        return float(v)                   # already kt
    # Unknown unit — assume km/h since that's NWS's documented default.
    return float(v) * 0.539957


def _fetch_latest_obs(sid: str, name: str, lat: float, lon: float) -> LandObservation | None:
    # 6-second per-station cap. NWS station endpoints occasionally hang for
    # the full 30 s default; with 500 stations in parallel a handful of
    # hangs could push total latency past Vercel's serverless timeout.
    obs = _nws_get(f"/stations/{sid}/observations/latest", timeout_s=6)
    if not obs:
        return None
    p = (obs.get("properties") or {})
    wind_kt = _nws_speed_to_kt(p.get("windSpeed"))
    wind_dir = (p.get("windDirection") or {}).get("value")
    gust_kt = _nws_speed_to_kt(p.get("windGust"))
    pres_pa = (p.get("barometricPressure") or {}).get("value")
    temp_c = (p.get("temperature") or {}).get("value")
    return LandObservation(
        station_id=sid,
        name=name,
        lat=lat,
        lon=lon,
        wind_kt=wind_kt,
        wind_dir_deg=wind_dir,
        gust_kt=gust_kt,
        pressure_mb=pres_pa / 100.0 if pres_pa else None,
        temp_f=_c_to_f(temp_c),
        observed_at=p.get("timestamp") or "",
    )


def _spatial_subsample(
    stations: list[tuple[str, str, float, float]],
    west: float, south: float, east: float, north: float,
    target: int,
) -> list[tuple[str, str, float, float]]:
    """Grid-based one-per-cell subsample. Guarantees stations are spread over
    the whole bbox instead of clumping wherever NWS densely instrumented (e.g.
    Texas mesonet). Target is a soft cap: the returned count may be smaller
    when regions of the bbox have no stations (open ocean, sparse states)."""
    if len(stations) <= target:
        return stations
    side = max(2, math.ceil(math.sqrt(target)))
    dw = (east - west) / side
    dh = (north - south) / side
    if dw <= 0 or dh <= 0:
        return stations[:target]
    picked: dict[tuple[int, int], tuple[str, str, float, float]] = {}
    # Sort so the choice within a cell is deterministic across requests.
    for s in sorted(stations, key=lambda x: x[0]):
        sid, name, lat, lon = s
        ix = min(side - 1, max(0, int((lon - west) / dw)))
        iy = min(side - 1, max(0, int((lat - south) / dh)))
        picked.setdefault((ix, iy), s)
    return list(picked.values())


def land_stations_in_bbox(
    west: float, south: float, east: float, north: float,
    *, max_stations: int = 80,
) -> list[LandObservation]:
    """NWS observation stations spread evenly across ``bbox``, each enriched
    with the latest observation.

    Previously this iterated NWS's global station list in whatever order the
    API returned it and stopped after 250 in-bbox candidates — which for a
    bbox spanning e.g. Louisiana + Texas would fill up on the ~300 automated
    Texas mesonet sites before ever reaching Louisiana. We now collect every
    in-bbox candidate and spatially subsample onto a grid so obs are visible
    across the whole storm footprint.

    Concurrent fetch (12 workers). NWS has no rate-limit headers; 12 parallel
    reads is polite.
    """
    all_in_bbox: list[tuple[str, str, float, float]] = []
    for f in _nws_all_stations():
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        lon, lat = coords[0], coords[1]
        if lat is None or lon is None:
            continue
        if not (south <= lat <= north and west <= lon <= east):
            continue
        props = f.get("properties") or {}
        sid = props.get("stationIdentifier") or ""
        if not sid:
            continue
        all_in_bbox.append((sid, props.get("name") or sid, lat, lon))

    candidates = _spatial_subsample(
        all_in_bbox, west, south, east, north, max_stations,
    )

    out: list[LandObservation] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(_fetch_latest_obs, sid, name, lat, lon)
            for sid, name, lat, lon in candidates
        ]
        for fut in as_completed(futures, timeout=30):
            try:
                rec = fut.result()
                if rec is not None:
                    out.append(rec)
            except Exception:  # noqa: BLE001
                continue
    return out
