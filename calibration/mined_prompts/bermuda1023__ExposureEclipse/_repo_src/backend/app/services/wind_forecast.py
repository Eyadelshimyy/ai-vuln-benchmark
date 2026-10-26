"""Point-forecast wind for a lat/lon, sourced from both GFS and ECMWF.

The click-to-inspect feature on the live wind heatmap compares what we're
observing right now (IDW blend of NDBC + NWS obs) against what the two major
global NWP models expect at that spot. It's a quick sanity check for the
underwriter — if obs and both models agree, that's high-confidence; if the
observation is a big outlier vs both models, something is off (dead sensor,
extreme local terrain effect, etc).

Data source: **Open-Meteo** (open-meteo.com) — free, no auth, and it exposes
both the NOAA GFS and the ECMWF IFS through a single JSON endpoint. Cheaper
than parsing GRIB2 files ourselves, and Open-Meteo does its own bilinear
interpolation from the model grid to the requested lat/lon.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from ..brand import USER_AGENT
from .ttl_cache import TtlCache

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FETCH_TIMEOUT_S = 15

# Open-Meteo model keys → display / wire names.
MODELS: tuple[tuple[str, str], ...] = (
    ("gfs_seamless", "gfs"),
    ("ecmwf_ifs025", "ecmwf"),
)


@dataclass(slots=True, frozen=True)
class ModelForecast:
    model: str            # "gfs" | "ecmwf"
    valid_time_utc: str   # ISO
    wind_kt: float
    wind_dir_deg: float | None
    wind_gust_kt: float | None


@dataclass(slots=True, frozen=True)
class PointForecast:
    lat: float
    lon: float
    fetched_at_utc: str
    forecasts: list[ModelForecast]     # one row per model, latest hour


def _mps_to_kt(v: float | None) -> float | None:
    return v * 1.94384 if v is not None else None


# Same "don't cache failures" pattern as _fetch_bulk_chunk. A single 429
# used to poison the popup for the whole session; now failures re-attempt
# on the next click while successful responses stay cached.
_POINT_CACHE_MAX = 256
_point_cache: OrderedDict[tuple[float, float, str], dict] = OrderedDict()


def _fetch_open_meteo_hourly(lat: float, lon: float, model_key: str) -> dict | None:
    """One HTTP GET to Open-Meteo for a single model's hourly wind block.

    Uses the ``hourly`` endpoint rather than ``current`` because ECMWF's IFS
    is only published on hourly cadence, so ``current`` returns nulls for it.
    Cached (successful responses only) by (lat, lon, model) so rapid re
    -clicks near the same spot don't hammer the API."""
    key = (lat, lon, model_key)
    hit = _point_cache.get(key)
    if hit is not None:
        _point_cache.move_to_end(key)
        return hit

    params = {
        "latitude": f"{lat:.3f}",
        "longitude": f"{lon:.3f}",
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "forecast_days": 1,
        "models": model_key,
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None

    _point_cache[key] = data
    while len(_point_cache) > _POINT_CACHE_MAX:
        _point_cache.popitem(last=False)
    return data


def _nearest_hour_index(times: list[str], now: datetime) -> int | None:
    """Pick the index in ``times`` closest to ``now`` UTC. Open-Meteo hourly
    times are ISO-8601 in the requested timezone (we ask for UTC)."""
    best_i: int | None = None
    best_delta: float | None = None
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = abs((dt - now).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_i = i
    return best_i


def _extract_hourly_row(
    data: dict, wire_name: str, now: datetime,
) -> ModelForecast | None:
    hourly = (data or {}).get("hourly") or {}
    times = hourly.get("time") or []
    speeds = hourly.get("wind_speed_10m") or []
    dirs = hourly.get("wind_direction_10m") or []
    gusts = hourly.get("wind_gusts_10m") or []
    if not times or not speeds:
        return None
    idx = _nearest_hour_index(times, now)
    if idx is None or idx >= len(speeds):
        return None
    speed_ms = speeds[idx]
    if speed_ms is None:
        return None
    dir_deg = dirs[idx] if idx < len(dirs) else None
    gust_ms = gusts[idx] if idx < len(gusts) else None
    valid_time = times[idx]
    if valid_time and not valid_time.endswith("Z"):
        valid_time = valid_time + "Z"
    return ModelForecast(
        model=wire_name,
        valid_time_utc=valid_time,
        wind_kt=round(float(_mps_to_kt(float(speed_ms)) or 0.0), 1),
        wind_dir_deg=float(dir_deg) if dir_deg is not None else None,
        wind_gust_kt=(
            round(float(_mps_to_kt(float(gust_ms)) or 0.0), 1)
            if gust_ms is not None else None
        ),
    )


def point_forecast(lat: float, lon: float) -> PointForecast:
    """Return latest-hour GFS + ECMWF wind at (lat, lon). Empty ``forecasts``
    when Open-Meteo is unreachable; caller shows that as 'model data
    unavailable' rather than 5xx'ing the click interaction."""
    now = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Round to 0.05° so lru_cache reuses across sub-cell precision clicks.
    lat_q = round(lat * 20) / 20
    lon_q = round(lon * 20) / 20

    # Fire both model requests in parallel — sequential would double click
    # latency to ~2 s.
    forecasts: list[ModelForecast] = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(_fetch_open_meteo_hourly, lat_q, lon_q, key): wire
            for key, wire in MODELS
        }
        for fut, wire in futures.items():
            try:
                data = fut.result()
            except Exception:  # noqa: BLE001
                data = None
            if data is None:
                continue
            row = _extract_hourly_row(data, wire, now)
            if row is not None:
                forecasts.append(row)

    # Preserve the (gfs, ecmwf) display order.
    order = {wire: i for i, (_k, wire) in enumerate(MODELS)}
    forecasts.sort(key=lambda f: order.get(f.model, 999))
    return PointForecast(
        lat=lat, lon=lon, fetched_at_utc=now_iso, forecasts=forecasts,
    )


# Daily point forecast for the tucked-away "where I am" card. Separate from
# the hourly click popup and from the gridded location budget: two calls,
# not hundreds.
_DAILY_FIELDS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
)
_DAILY_DAYS = 5
_DAILY_CACHE: TtlCache[tuple[float, float, str], dict] = TtlCache(
    ttl_s=30 * 60, maxsize=128,
)


@dataclass(slots=True, frozen=True)
class DailyDay:
    date: str
    weather_code: int | None
    temp_max_f: float | None
    temp_min_f: float | None
    precip_in: float | None
    wind_max_kt: float | None
    gust_max_kt: float | None
    wind_dir_deg: float | None


@dataclass(slots=True, frozen=True)
class ModelDaily:
    model: str
    days: list[DailyDay]


@dataclass(slots=True, frozen=True)
class DailyForecast:
    lat: float
    lon: float
    timezone: str | None
    fetched_at_utc: str
    models: list[ModelDaily]


def _opt_float(v: object) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _opt_int(v: object) -> int | None:
    f = _opt_float(v)
    return int(f) if f is not None else None


def _round(v: float | None, places: int) -> float | None:
    return round(v, places) if v is not None else None


def _fetch_open_meteo_daily(lat: float, lon: float, model_key: str) -> dict | None:
    """One daily forecast. Successful bodies only are cached. A 429 or a
    body with no ``daily.time`` is a miss and is not stored."""
    key = (lat, lon, model_key)
    hit = _DAILY_CACHE.get(key)
    if hit is not None:
        return hit
    params = {
        "latitude": f"{lat:.3f}",
        "longitude": f"{lon:.3f}",
        "daily": ",".join(_DAILY_FIELDS),
        "forecast_days": _DAILY_DAYS,
        "wind_speed_unit": "kn",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "models": model_key,
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    times = ((data or {}).get("daily") or {}).get("time") or []
    if not times:
        return None
    _DAILY_CACHE.set(key, data)
    return data


def _extract_daily(data: dict, wire_name: str) -> ModelDaily | None:
    daily = (data or {}).get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return None
    days: list[DailyDay] = []
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    precip = daily.get("precipitation_sum") or []
    wind = daily.get("wind_speed_10m_max") or []
    gust = daily.get("wind_gusts_10m_max") or []
    direc = daily.get("wind_direction_10m_dominant") or []
    for i, date in enumerate(times[:_DAILY_DAYS]):
        if not isinstance(date, str) or not date:
            continue
        days.append(DailyDay(
            date=date,
            weather_code=_opt_int(codes[i] if i < len(codes) else None),
            temp_max_f=_round(_opt_float(tmax[i] if i < len(tmax) else None), 1),
            temp_min_f=_round(_opt_float(tmin[i] if i < len(tmin) else None), 1),
            precip_in=_round(_opt_float(precip[i] if i < len(precip) else None), 2),
            wind_max_kt=_round(_opt_float(wind[i] if i < len(wind) else None), 1),
            gust_max_kt=_round(_opt_float(gust[i] if i < len(gust) else None), 1),
            wind_dir_deg=_round(_opt_float(direc[i] if i < len(direc) else None), 0),
        ))
    if not days:
        return None
    return ModelDaily(model=wire_name, days=days)


def daily_forecast(lat: float, lon: float) -> DailyForecast:
    """Five local days of GFS and ECMWF at one point.

    Temperatures are Fahrenheit, precipitation is inches, wind is knots.
    Dates are the calendar at that coordinate (Open-Meteo ``timezone=auto``),
    not UTC. An unreachable model is omitted; both missing leaves ``models``
    empty rather than raising.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lat_q = round(lat * 20) / 20
    lon_q = round(lon * 20) / 20
    payloads: list[tuple[str, dict]] = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(_fetch_open_meteo_daily, lat_q, lon_q, key): wire
            for key, wire in MODELS
        }
        for fut, wire in futures.items():
            try:
                data = fut.result()
            except Exception:  # noqa: BLE001
                data = None
            if data is not None:
                payloads.append((wire, data))
    order = {wire: i for i, (_k, wire) in enumerate(MODELS)}
    payloads.sort(key=lambda item: order.get(item[0], 999))
    models: list[ModelDaily] = []
    tz: str | None = None
    for wire, data in payloads:
        if tz is None:
            raw_tz = data.get("timezone")
            if isinstance(raw_tz, str) and raw_tz:
                tz = raw_tz
        row = _extract_daily(data, wire)
        if row is not None:
            models.append(row)
    return DailyForecast(
        lat=lat, lon=lon, timezone=tz, fetched_at_utc=now_iso, models=models,
    )


@dataclass(slots=True, frozen=True)
class WindCoord:
    lat: float
    lon: float


@dataclass(slots=True, frozen=True)
class ModelWindFrame:
    """One forecast time-step for a whole cell grid. ``wind_kt`` and
    ``wind_dir_deg`` are parallel-array to the outer ``ModelWindGrid.cells``
    list so the wire payload doesn't repeat lat/lon at every frame."""
    hour: int              # forecast hours from "now" (0, 6, 12, …)
    valid_time_utc: str
    # ``None`` is a gap at that cell for that hour (a level was missing).
    # A real 0 is calm, not a gap.
    wind_kt: list[float | None]
    wind_dir_deg: list[float | None]


@dataclass(slots=True, frozen=True)
class ModelWindGrid:
    model: str
    step_deg: float
    cells: list[WindCoord]
    frames: list[ModelWindFrame]
    # True when this empty grid was refused by the per-minute location
    # budget (or Open-Meteo answered 429). Not cached. The panel waits
    # out the minute and asks once more.
    rate_limited: bool = False

    @property
    def valid_time_utc(self) -> str:
        """First-frame valid time — kept for backward compat with callers
        that still expect a single-time grid."""
        return self.frames[0].valid_time_utc if self.frames else ""


# Open-Meteo's docs allow up to 5000 coords per request. 400 keeps each URL
# well under any edge-side limit while dramatically reducing the total number
# of parallel requests (a Fausto-sized Pacific bbox went from 15 chunks to 4).
# Sending too many parallel small requests caused visible "empty streaks"
# on the wind heatmap when Open-Meteo rate-limited a handful of the chunks.
# 80 locations is the size that returns promptly. A 400-location URL is
# what made a basin-sized click die at the proxy.
# 200 locations is one or two HTTP calls for a cone that already fits
# the per-minute location budget below. Smaller chunks spent that
# budget on GFS, and the ECMWF grid that followed came back empty.
_CHUNK_SIZE = 200
# Aggressive retry policy — the model-grid fetch is chunked in row-major
# order, so a permanent failure on any single chunk drops a horizontal
# band from the response. Better to hammer the retry for a few extra
# seconds than serve a heatmap with gaps.
_RETRY_ATTEMPTS = 2
_RETRY_BACKOFF_S = 0.6

# Forecast horizon for the timeline slider. NHC issues 5-day forecasts;
# we sample every 6 hours through the same window.
_FORECAST_HOURS: tuple[int, ...] = (0, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 96, 120)
# Open-Meteo's free tier is 600 calls per minute per IP, and a
# multi-location request counts one call per coordinate. A gulf cone at
# 0.75° is about 660 locations: GFS, requested first, succeeds, and
# ECMWF is then rejected for the rest of that minute. 250 leaves room
# for the other model in the same minute. The step coarsens to fit.
_MAX_MODEL_CELLS = 250
# Shared by surface and shear, both models. A third grid in the same
# minute (shear on top of both surface fields) waits rather than 429ing.
_LOCATION_BUDGET = 500
_LOCATION_WINDOW_S = 60.0
_location_events: deque[tuple[float, int]] = deque()
_location_lock = threading.Lock()
# Stop waiting and return whatever chunks finished. Unfinished cells are
# left as gaps, not painted as calm.
_GRID_DEADLINE_S = 28.0
_GRID_CACHE: TtlCache[tuple, "ModelWindGrid"] = TtlCache(ttl_s=180, maxsize=12)


class OpenMeteoRateLimited(Exception):
    """The free-tier minute window is full. Retrying now only extends it."""


def _try_reserve_locations(n: int) -> bool:
    """Claim ``n`` locations against the rolling minute budget.

    A fresh process always admits one grid: the cell cap is under the
    budget. The second and third grids in the same process share what
    is left. Refused grids are not sent, so a 429 does not get worse.
    """
    if n <= 0:
        return True
    now = time.monotonic()
    with _location_lock:
        while _location_events and now - _location_events[0][0] >= _LOCATION_WINDOW_S:
            _location_events.popleft()
        spent = sum(count for _, count in _location_events)
        if spent + n > _LOCATION_BUDGET:
            return False
        _location_events.append((now, n))
        return True


def choose_model_step(
    west: float, south: float, east: float, north: float,
    preferred: float = 0.25,
) -> float:
    """Step that keeps the Open-Meteo request inside ``_MAX_MODEL_CELLS``.

    Native GFS and IFS spacing is 0.25°. A zoomed view that fits uses that.
    A basin-sized box coarsens so both models still fit in one minute.
    """
    span_lat = max(north - south, 0.5)
    span_lon = max(east - west, 0.5)
    step = max(preferred, 0.25)
    while step < 2.0:
        nlat = int(span_lat / step) + 1
        nlon = int(span_lon / step) + 1
        if nlat * nlon <= _MAX_MODEL_CELLS:
            return round(step, 2)
        step = round(step + 0.25, 2)
    return 2.0


def _extract_bulk_frames(
    requested_coords: list[tuple[float, float]],
    items: list[dict],
    now: datetime,
) -> tuple[list[WindCoord], dict[int, tuple[list[float], list[float | None], str]]]:
    """Parse Open-Meteo's multi-location response into (coords, frames).

    ``frames`` is keyed by forecast-hour offset (from ``now``) matching
    ``_FORECAST_HOURS``; each value is a tuple of parallel arrays
    (wind_kt_per_coord, wind_dir_deg_per_coord, valid_time_utc).

    Coordinates emit at the *requested* lat/lon (not Open-Meteo's returned
    lat/lon — that would snap to the model's native grid and break cell
    -alignment with the observed grid on the frontend).

    A cell with no wind speed is omitted, not written as 0 kt. A missing
    Open-Meteo item used to paint a calm stripe. Arrays stay parallel
    because a coord and its winds are appended together. A real 0 m/s
    is kept — that is calm wind, not a missing field."""
    coords: list[WindCoord] = []
    frame_kts: dict[int, list[float]] = {h: [] for h in _FORECAST_HOURS}
    frame_dirs: dict[int, list[float | None]] = {h: [] for h in _FORECAST_HOURS}
    frame_vt: dict[int, str] = {h: "" for h in _FORECAST_HOURS}

    for i, req in enumerate(requested_coords):
        item = items[i] if i < len(items) else {}
        hourly = (item or {}).get("hourly") or {}
        times = hourly.get("time") or []
        speeds = hourly.get("wind_speed_10m") or []
        dirs = hourly.get("wind_direction_10m") or []
        if not any(s is not None for s in speeds):
            continue
        req_lat, req_lon = req
        coords.append(WindCoord(
            lat=round(float(req_lat), 3),
            lon=round(float(req_lon), 3),
        ))

        base_idx = _nearest_hour_index(times, now) if times else None

        for h in _FORECAST_HOURS:
            kt: float = 0.0
            dir_deg_out: float | None = None
            if base_idx is not None and times:
                idx = base_idx + h  # 1-hour steps in Open-Meteo's array
                if idx < len(times):
                    speed_ms = speeds[idx] if idx < len(speeds) else None
                    if speed_ms is not None:
                        kt = round(
                            float(_mps_to_kt(float(speed_ms)) or 0.0), 1,
                        )
                        dir_raw = dirs[idx] if idx < len(dirs) else None
                        dir_deg_out = (
                            round(float(dir_raw), 1)
                            if dir_raw is not None else None
                        )
                        if not frame_vt[h]:
                            t = times[idx]
                            frame_vt[h] = t if t.endswith("Z") else (t + "Z")
            frame_kts[h].append(kt)
            frame_dirs[h].append(dir_deg_out)

    frames_per_hour = {
        h: (frame_kts[h], frame_dirs[h], frame_vt[h])
        for h in _FORECAST_HOURS
    }
    return coords, frames_per_hour


# Manual LRU that ONLY caches successful non-empty results. The previous
# @lru_cache implementation also cached failures — once Open-Meteo returned
# 429 for any chunk (transient rate-limit or blip), the empty tuple got
# baked into the cache and every future request in that serverless
# container returned "no data available" until the process recycled. Users
# saw ECMWF flip from "working" to permanently "unavailable" mid-session.
_BULK_CACHE_MAX = 64
_bulk_cache: OrderedDict[tuple[str, str, str, str], tuple[dict, ...]] = OrderedDict()

_SURFACE_HOURLY = "wind_speed_10m,wind_direction_10m"
_SHEAR_HOURLY = (
    "wind_speed_850hPa,wind_direction_850hPa,"
    "wind_speed_200hPa,wind_direction_200hPa"
)
_SURFACE_FIELDS = ("wind_speed_10m",)
_SHEAR_FIELDS = ("wind_speed_850hPa", "wind_speed_200hPa")
_SHEAR_CACHE: TtlCache[tuple, "ModelWindGrid"] = TtlCache(ttl_s=180, maxsize=12)


def deep_layer_shear(
    kt_850: float, dir_850: float, kt_200: float, dir_200: float,
) -> tuple[float, float | None]:
    """850–200 hPa shear as (magnitude kt, meteorological FROM degrees).

    The vector is the 200 hPa wind minus the 850 hPa wind — the deep-layer
    shear tropical forecasters use, not the 200 hPa wind by itself. FROM
    matches the surface-wind convention, so an arrow rotated +180° points
    downshear (where the upper wind is headed relative to the lower wind).
    Direction is omitted below half a knot; the magnitude stays.
    """
    def uv(kt: float, deg: float) -> tuple[float, float]:
        r = math.radians(deg)
        return -kt * math.sin(r), -kt * math.cos(r)

    u850, v850 = uv(kt_850, dir_850)
    u200, v200 = uv(kt_200, dir_200)
    us, vs = u200 - u850, v200 - v850
    mag = math.hypot(us, vs)
    if mag < 0.5:
        return round(mag, 1), None
    to_deg = math.degrees(math.atan2(us, vs))
    return round(mag, 1), round((to_deg + 180.0) % 360.0, 1)


def _chunk_has_real_wind(
    items: tuple[dict, ...],
    fields: tuple[str, ...] = _SURFACE_FIELDS,
) -> bool:
    """True when Open-Meteo actually returned a wind speed.

    A HTTP 200 whose speeds are all null is an empty model, not a calm
    grid. Caching it made GFS/Euro look permanently unavailable until the
    process restarted. A real 0 m/s is kept — that is calm wind, not a
    missing field. Shear requires both the 850 and the 200 hPa speed.
    """
    for item in items:
        hourly = (item or {}).get("hourly") or {}
        if all(
            any(speed is not None for speed in (hourly.get(field) or []))
            for field in fields
        ):
            return True
    return False


def _fetch_bulk_chunk(
    lat_str: str, lon_str: str, model_key: str, *,
    refresh: bool = False,
    hourly: str = _SURFACE_HOURLY,
    fields: tuple[str, ...] = _SURFACE_FIELDS,
) -> tuple[dict, ...]:
    """Fetch one multi-location Open-Meteo chunk.

    A 429 is raised as ``OpenMeteoRateLimited`` and not retried. 5xx and
    network errors are retried. Successful responses that contain at
    least one real wind value are cached; failures, empty bodies, and
    all-null winds are not. ``refresh`` bypasses the cache read."""
    key = (lat_str, lon_str, model_key, hourly)
    if not refresh:
        hit = _bulk_cache.get(key)
        if hit is not None:
            _bulk_cache.move_to_end(key)
            return hit

    import time as _t
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "hourly": hourly,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        # 6 days covers the full 0..120h forecast horizon we sample for
        # the time slider (NHC ships 5-day forecasts; +1 day of headroom
        # so a T+120 sample never falls off the end).
        "forecast_days": 6,
        "models": model_key,
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT},
    )
    items: tuple[dict, ...] = ()
    for attempt in range(_RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            payload = data if isinstance(data, list) else [data]
            items = tuple(payload)
            break
        except urllib.error.HTTPError as e:
            e.close()
            # A 429 means the minute is already spent. Sleeping a few
            # seconds and calling again still lands inside that minute,
            # and the extra calls keep the window shut. Surface it and
            # let the panel wait the minute out.
            if e.code == 429:
                raise OpenMeteoRateLimited() from e
            # Retry 500-series (transient upstream) and 502/503/504
            # (edge/gateway hiccups). Anything else — 400 for a malformed
            # URL, 404 for an unknown model — is a permanent failure.
            if 500 <= e.code < 600 and attempt < _RETRY_ATTEMPTS:
                _t.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                continue
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ):
            # Network-level errors and truncated JSON both retry — they're
            # exactly the class of hiccup that leaves horizontal gaps in
            # the grid when only one chunk hits them.
            if attempt < _RETRY_ATTEMPTS:
                _t.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                continue
            break
        except Exception:  # noqa: BLE001
            break

    if items and _chunk_has_real_wind(items, fields):
        _bulk_cache[key] = items
        while len(_bulk_cache) > _BULK_CACHE_MAX:
            _bulk_cache.popitem(last=False)
    return items


def _extract_shear_frames(
    requested_coords: list[tuple[float, float]],
    items: list[dict],
    now: datetime,
) -> tuple[list[WindCoord], dict[int, tuple[list[float | None], list[float | None], str]]]:
    """850–200 hPa shear at each forecast hour. A cell with neither level
    is omitted. An hour missing one level is a null, not 0 kt of shear."""
    coords: list[WindCoord] = []
    frame_kts: dict[int, list[float | None]] = {h: [] for h in _FORECAST_HOURS}
    frame_dirs: dict[int, list[float | None]] = {h: [] for h in _FORECAST_HOURS}
    frame_vt: dict[int, str] = {h: "" for h in _FORECAST_HOURS}

    for i, req in enumerate(requested_coords):
        item = items[i] if i < len(items) else {}
        hourly = (item or {}).get("hourly") or {}
        times = hourly.get("time") or []
        s850 = hourly.get("wind_speed_850hPa") or []
        d850 = hourly.get("wind_direction_850hPa") or []
        s200 = hourly.get("wind_speed_200hPa") or []
        d200 = hourly.get("wind_direction_200hPa") or []
        base_idx = _nearest_hour_index(times, now) if times else None
        if base_idx is None:
            continue
        samples: list[tuple[int, tuple[float, float | None] | None, str]] = []
        any_real = False
        for h in _FORECAST_HOURS:
            idx = base_idx + h
            shear: tuple[float, float | None] | None = None
            vt = ""
            if idx < len(times):
                sp850 = s850[idx] if idx < len(s850) else None
                sp200 = s200[idx] if idx < len(s200) else None
                di850 = d850[idx] if idx < len(d850) else None
                di200 = d200[idx] if idx < len(d200) else None
                if (
                    sp850 is not None and sp200 is not None
                    and di850 is not None and di200 is not None
                ):
                    shear = deep_layer_shear(
                        float(_mps_to_kt(float(sp850)) or 0.0),
                        float(di850),
                        float(_mps_to_kt(float(sp200)) or 0.0),
                        float(di200),
                    )
                    t = times[idx]
                    vt = t if str(t).endswith("Z") else (str(t) + "Z")
                    any_real = True
            samples.append((h, shear, vt))
        if not any_real:
            continue
        coords.append(WindCoord(
            lat=round(float(req[0]), 3),
            lon=round(float(req[1]), 3),
        ))
        for h, shear, vt in samples:
            if shear is None:
                frame_kts[h].append(None)
                frame_dirs[h].append(None)
            else:
                frame_kts[h].append(shear[0])
                frame_dirs[h].append(shear[1])
            if vt and not frame_vt[h]:
                frame_vt[h] = vt
    return coords, {
        h: (frame_kts[h], frame_dirs[h], frame_vt[h]) for h in _FORECAST_HOURS
    }


def _assemble_model_grid(
    west: float, south: float, east: float, north: float,
    model_wire: str, *,
    hourly: str,
    fields: tuple[str, ...],
    extract,
    cache: TtlCache[tuple, ModelWindGrid],
    step_deg: float | None = None,
    refresh: bool = False,
) -> ModelWindGrid:
    """Shared chunked fetch for surface wind and deep-layer shear."""
    model_key = next(
        (k for (k, wire) in MODELS if wire == model_wire), None,
    )
    if model_key is None:
        raise ValueError(f"unknown model wire name: {model_wire!r}")

    step_deg = choose_model_step(
        west, south, east, north, preferred=step_deg if step_deg else 0.25,
    )
    cache_key = (
        round(west, 2), round(south, 2), round(east, 2), round(north, 2),
        model_wire, step_deg,
    )
    if not refresh:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    coords: list[tuple[float, float]] = []
    lat = south
    while lat <= north + 1e-9:
        lon = west
        while lon <= east + 1e-9:
            coords.append((round(lat, 3), round(lon, 3)))
            lon += step_deg
        lat += step_deg

    if not coords:
        return ModelWindGrid(
            model=model_wire, step_deg=step_deg, cells=[], frames=[],
        )
    if not _try_reserve_locations(len(coords)):
        return ModelWindGrid(
            model=model_wire, step_deg=step_deg, cells=[], frames=[],
            rate_limited=True,
        )

    now = datetime.now(timezone.utc)
    chunks = [
        coords[i : i + _CHUNK_SIZE]
        for i in range(0, len(coords), _CHUNK_SIZE)
    ]

    # Chunks finish out of order. Each cell's value is appended with its
    # own coord, so the parallel arrays stay aligned without row-major order.
    all_coords: list[WindCoord] = []
    all_kts: dict[int, list[float | None]] = {h: [] for h in _FORECAST_HOURS}
    all_dirs: dict[int, list[float | None]] = {h: [] for h in _FORECAST_HOURS}
    frame_valid_times: dict[int, str] = {h: "" for h in _FORECAST_HOURS}

    # Two at a time. Four parallel chunks plus a second model was enough
    # to trip Open-Meteo's per-minute limit and return an empty ECMWF grid.
    pool = ThreadPoolExecutor(max_workers=min(2, len(chunks)))
    fut_to_chunk: dict = {}
    rate_limited = False
    try:
        for ch in chunks:
            lat_str = ",".join(f"{c[0]:.3f}" for c in ch)
            lon_str = ",".join(f"{c[1]:.3f}" for c in ch)
            fut = pool.submit(
                _fetch_bulk_chunk, lat_str, lon_str, model_key,
                refresh=refresh, hourly=hourly, fields=fields,
            )
            fut_to_chunk[fut] = ch
        done, _pending = wait(set(fut_to_chunk), timeout=_GRID_DEADLINE_S)
        for fut in done:
            ch = fut_to_chunk[fut]
            try:
                items = fut.result()
            except OpenMeteoRateLimited:
                rate_limited = True
                items = ()
            except Exception:  # noqa: BLE001
                items = ()
            if not items or not _chunk_has_real_wind(tuple(items), fields):
                continue
            chunk_coords, chunk_frames = extract(ch, list(items), now)
            all_coords.extend(chunk_coords)
            for h in _FORECAST_HOURS:
                kts, dirs, vt = chunk_frames.get(h, ([], [], ""))
                all_kts[h].extend(kts)
                all_dirs[h].extend(dirs)
                if vt and not frame_valid_times[h]:
                    frame_valid_times[h] = vt
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    frames: list[ModelWindFrame] = []
    for h in _FORECAST_HOURS:
        vt = frame_valid_times[h]
        if not vt:
            continue
        frames.append(ModelWindFrame(
            hour=h,
            valid_time_utc=vt,
            wind_kt=all_kts[h],
            wind_dir_deg=all_dirs[h],
        ))

    has_wind = any(
        kt is not None and kt > 0
        for frame in frames
        for kt in frame.wind_kt
    )
    grid = ModelWindGrid(
        model=model_wire, step_deg=step_deg,
        cells=all_coords, frames=frames,
        rate_limited=rate_limited and not has_wind,
    )
    if has_wind:
        cache.set(cache_key, grid)
    return grid


def fetch_model_wind_grid(
    west: float, south: float, east: float, north: float,
    model_wire: str, *, step_deg: float | None = None, refresh: bool = False,
) -> ModelWindGrid:
    """GFS or ECMWF 10 m wind grid over the bbox, as forecast frames.

    The step coarsens on a large cone so the click finishes inside
    ``_GRID_DEADLINE_S`` instead of dying at the proxy. A finished grid
    with real wind is cached for a few minutes; an empty result is not,
    so Retry can ask Open-Meteo again. ``refresh`` skips the cache read.
    """
    return _assemble_model_grid(
        west, south, east, north, model_wire,
        hourly=_SURFACE_HOURLY, fields=_SURFACE_FIELDS,
        extract=_extract_bulk_frames, cache=_GRID_CACHE,
        step_deg=step_deg, refresh=refresh,
    )


def fetch_model_shear_grid(
    west: float, south: float, east: float, north: float,
    model_wire: str, *, step_deg: float | None = None, refresh: bool = False,
) -> ModelWindGrid:
    """GFS or ECMWF 850–200 hPa shear on the same hours as the surface grid.

    ``wind_kt`` is the shear magnitude. ``wind_dir_deg`` is where it comes
    from, so the map arrow (direction + 180°) points downshear. Same step
    cap and deadline as the surface grid. Cached apart from that grid.
    """
    return _assemble_model_grid(
        west, south, east, north, model_wire,
        hourly=_SHEAR_HOURLY, fields=_SHEAR_FIELDS,
        extract=_extract_shear_frames, cache=_SHEAR_CACHE,
        step_deg=step_deg, refresh=refresh,
    )


# WMO factor from 10-minute (what GFS/IFS 10 m wind is closest to) to the
# 1-minute sustained speed the damage curves use.
_ONE_MIN_FROM_10M = 1.11
_OUTER_CAP_CACHE: TtlCache[tuple, list] = TtlCache(ttl_s=20 * 60, maxsize=4)


def outer_wind_cap(track_points: list[tuple[float, float]]):
    """Max of GFS and ECMWF 1-minute wind over the next 3 days.

    Sampled on a 1.2° grid around the forecast track. Used only to cap the
    outer wind field (beyond the 50 kt radius). Returns a ``(lat, lon) -> kt``
    lookup, or None if neither model answers. The core is left to NHC:
    global models do not resolve the eyewall.
    """
    if len(track_points) < 1:
        return None
    lats = [p[0] for p in track_points]
    lons = [p[1] for p in track_points]
    lat0 = math.floor(min(lats) - 1.2)
    lat1 = math.ceil(max(lats) + 1.2)
    lon0 = math.floor(min(lons) - 1.2)
    lon1 = math.ceil(max(lons) + 1.2)
    # Keep the request small. The outer field does not need eyewall resolution.
    step = 1.2
    grid: list[tuple[float, float]] = []
    lat = lat0
    while lat <= lat1 + 0.01:
        lon = lon0
        while lon <= lon1 + 0.01:
            grid.append((round(lat, 2), round(lon, 2)))
            lon += step
        lat += step
    if len(grid) > 80:
        grid = grid[:80]
    key = tuple(grid)
    cached = _OUTER_CAP_CACHE.get(key)
    if cached is None:
        cached = _fetch_outer_cap(grid)
        if cached:
            _OUTER_CAP_CACHE.set(key, cached)
    if not cached:
        return None

    def _at(lat_q: float, lon_q: float) -> float | None:
        num = 0.0
        den = 0.0
        for glat, glon, kt in cached:
            dlat = lat_q - glat
            dlon = lon_q - glon
            dist2 = dlat * dlat + dlon * dlon
            # Only the local cell. A 2° blend pulls the eyewall into inland counties.
            if dist2 > 0.85 * 0.85:
                continue
            w = 1.0 / max(dist2, 0.04)
            num += w * kt
            den += w
        if den <= 0:
            return None
        return num / den

    return _at


def _fetch_outer_cap(grid: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    """Per grid point, the higher of the two models' peak 1-minute wind."""
    lats = ",".join(f"{lat:.2f}" for lat, _lon in grid)
    lons = ",".join(f"{lon:.2f}" for _lat, lon in grid)
    peaks: list[list[float]] = [[0.0, 0.0] for _ in grid]

    def _one(model_key: str, slot: int) -> None:
        params = {
            "latitude": lats,
            "longitude": lons,
            "hourly": "wind_speed_10m",
            "wind_speed_unit": "kn",
            "timezone": "UTC",
            "forecast_days": 3,
            "models": model_key,
        }
        url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        rows = data if isinstance(data, list) else [data]
        for i, row in enumerate(rows):
            if i >= len(peaks):
                break
            speeds = ((row or {}).get("hourly") or {}).get("wind_speed_10m") or []
            vals = [float(v) for v in speeds if v is not None]
            if vals:
                peaks[i][slot] = max(vals) * _ONE_MIN_FROM_10M

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(_one, "gfs_seamless", 0)
        pool.submit(_one, "ecmwf_ifs025", 1)
    out: list[tuple[float, float, float]] = []
    for (lat, lon), (gfs_kt, ecm_kt) in zip(grid, peaks):
        kt = max(gfs_kt, ecm_kt)
        if kt > 0:
            out.append((lat, lon, kt))
    return out


__all__ = [
    "ModelForecast",
    "ModelWindFrame",
    "ModelWindGrid",
    "PointForecast",
    "WindCoord",
    "choose_model_step",
    "deep_layer_shear",
    "fetch_model_shear_grid",
    "DailyDay",
    "DailyForecast",
    "ModelDaily",
    "daily_forecast",
    "fetch_model_wind_grid",
    "outer_wind_cap",
    "point_forecast",
]
