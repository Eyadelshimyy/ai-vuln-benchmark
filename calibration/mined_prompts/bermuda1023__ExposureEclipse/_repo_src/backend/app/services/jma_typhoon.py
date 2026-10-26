"""Official JMA typhoon track.

The Japan Meteorological Agency does not publish a versioned API. The
typhoon page loads three JSON files:

* ``targetTc.json`` — systems it is currently issuing
* ``{tcCode}/forecast.json`` — track, gale circle, storm arcs, forecast circles
* ``{tcCode}/specifications.json`` — name, 10-minute wind, pressure, radii

A 404 or an empty list means nothing is active. It is not an error, and it
must not take down the NHC storm list. Western Pacific a-decks are not on
the public NHC FTP, so this module never invents a cone or a member vote.

Winds in the bulletin are 10-minute sustained, not the 1-minute NHC scale.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..brand import USER_AGENT
from .ttl_cache import TtlCache

TARGET_URL = "https://www.jma.go.jp/bosai/typhoon/data/targetTc.json"
FORECAST_URL = "https://www.jma.go.jp/bosai/typhoon/data/{code}/forecast.json"
SPEC_URL = "https://www.jma.go.jp/bosai/typhoon/data/{code}/specifications.json"

FETCH_TIMEOUT_S = 12
_ID_RE = re.compile(r"^TC\d+$", re.IGNORECASE)

# JMA's own viewer converts an arc ``[start, end]`` (degrees clockwise from
# north) into a math angle with ``[-end + 90, -start + 90]``. Match that so
# the published tangent segments meet the arc ends.
_A = 6_378_137.0
_F = 298.257223563
_E2 = (2 * _F - 1) / (_F ** 2)

_AREA_EN = {
    "北": "North",
    "南": "South",
    "東": "East",
    "西": "West",
    "北東": "Northeast",
    "北西": "Northwest",
    "南東": "Southeast",
    "南西": "Southwest",
    "全域": "All",
}
_INTENSITY_EN = {
    "強い": "strong",
    "非常に強い": "very strong",
    "猛烈な": "violent",
}

NOTE = (
    "JMA 10-minute sustained wind. Forecast circles are JMA's 70% "
    "center-position circles, not an NHC cone. No ensemble strike vote. "
    "The horizon is 72 hours, and the last point may already be extratropical. "
    "GFS and ECMWF particles are model wind, not this 10-minute maximum."
)

_LIST_CACHE: TtlCache[str, list["JmaSummary"]] = TtlCache(ttl_s=300, maxsize=1)
_STORM_CACHE: TtlCache[str, "JmaStorm"] = TtlCache(ttl_s=300, maxsize=8)


def is_jma_id(storm_id: str) -> bool:
    return bool(_ID_RE.match((storm_id or "").strip()))


def clear_cache() -> None:
    _LIST_CACHE.clear()
    _STORM_CACHE.clear()


@dataclass(slots=True, frozen=True)
class JmaSummary:
    storm_id: str
    name: str
    name_jp: str
    year: int
    classification: str
    intensity_kt: int
    pressure_mb: int | None
    lat: float | None
    lon: float | None
    label: str


@dataclass(slots=True, frozen=True)
class JmaFix:
    hours_out: int
    valid_time: str
    lat: float
    lon: float
    wind_kt: int | None
    wind_ms: float | None
    pressure_mb: int | None
    category_en: str
    category_jp: str
    intensity: str | None


@dataclass(slots=True, frozen=True)
class JmaCircle:
    kind: str                 # gale | storm | probability
    hours_out: int
    label: str
    center_lon: float
    center_lat: float
    radius_m: float
    ring: list[list[float]]   # closed [lon, lat], may sit past ±180


@dataclass(slots=True, frozen=True)
class JmaPath:
    kind: str                 # tangent | storm-arc | storm-line
    hours_out: int
    coordinates: list[list[float]]


@dataclass(slots=True)
class JmaStorm:
    summary: JmaSummary
    issued_at: str
    typhoon_number: str
    category_jp: str
    intensity: str | None
    wind_ms: float | None
    gust_ms: float | None
    gust_kt: int | None
    gale_ranges: list[str] = field(default_factory=list)
    storm_ranges: list[str] = field(default_factory=list)
    note: str = NOTE
    observed: list[tuple[float, float]] = field(default_factory=list)
    forecast: list[JmaFix] = field(default_factory=list)
    circles: list[JmaCircle] = field(default_factory=list)
    paths: list[JmaPath] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (120.0, 0.0, 160.0, 45.0)


def _get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        return None
    except Exception:  # noqa: BLE001 — quiet basin and dead feeds are empty
        return None
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _num(value) -> float | None:
    if value is None or value == "" or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value) -> int | None:
    n = _num(value)
    if n is None:
        return None
    return int(round(n))


def _part_name(part: dict) -> str:
    raw = part.get("part")
    if raw == "title":
        return "title"
    if isinstance(raw, dict):
        en = str(raw.get("en") or "").lower()
        jp = str(raw.get("jp") or "")
        if "analysis" in en or jp == "実況":
            return "analysis"
        if "forecast" in en or "予報" in jp:
            return "forecast"
    hours = part.get("advancedHours")
    if hours == 0:
        return "analysis"
    if isinstance(hours, (int, float)) and hours > 0:
        return "forecast"
    return "other"


def _valid_time(part: dict) -> str:
    clock = part.get("validtime") or part.get("issue") or {}
    if isinstance(clock, dict):
        return str(clock.get("UTC") or clock.get("JST") or "")
    return ""


def _area_label(area) -> str:
    if isinstance(area, dict):
        en = area.get("en")
        if en:
            return str(en)
        jp = str(area.get("jp") or "")
        return _AREA_EN.get(jp, jp)
    if isinstance(area, str):
        return _AREA_EN.get(area, area)
    return ""


def _range_text(item: dict) -> str | None:
    if not isinstance(item, dict):
        return None
    rng = item.get("range") or {}
    km = _num(rng.get("km") if isinstance(rng, dict) else None)
    nm = _num(rng.get("nm") if isinstance(rng, dict) else None)
    if km is None and nm is None:
        return None
    label = _area_label(item.get("area")) or "All"
    if km is not None and nm is not None:
        return f"{label} {km:g} km ({nm:g} nm)"
    if km is not None:
        return f"{label} {km:g} km"
    return f"{label} {nm:g} nm"


def _intensity_label(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text == "-":
        return None
    en = _INTENSITY_EN.get(text)
    return f"{en} ({text})" if en else text


def _category(part: dict) -> tuple[str, str]:
    cat = part.get("category")
    if isinstance(cat, dict):
        return str(cat.get("en") or ""), str(cat.get("jp") or "")
    if isinstance(cat, str):
        return cat, ""
    return "", ""


def _wind(part: dict) -> tuple[float | None, int | None, float | None, int | None]:
    wind = part.get("maximumWind")
    if not isinstance(wind, dict):
        return None, None, None, None
    sustained = wind.get("sustained") if isinstance(wind.get("sustained"), dict) else {}
    gust = wind.get("gust") if isinstance(wind.get("gust"), dict) else {}
    return (
        _num(sustained.get("m/s")),
        _int(sustained.get("kt")),
        _num(gust.get("m/s")),
        _int(gust.get("kt")),
    )


def _meters_per_deg(lat_deg: float) -> tuple[float, float]:
    """Meridional and zonal metres per degree, matching JMA's page script."""
    tmp = 1 - _E2 * (math.radians(lat_deg) ** 2)
    arc_lat = (math.pi * _A * (1 - _E2)) / (180 * tmp ** 1.5)
    arc_lng = (math.pi * _A * math.cos(math.radians(lat_deg))) / (180 * math.sqrt(tmp))
    return arc_lat, arc_lng


def arc_lonlat(
    lat: float,
    lon: float,
    radius_m: float,
    start_deg: float,
    end_deg: float,
    *,
    step_deg: float = 5.0,
) -> list[list[float]]:
    """``[lon, lat]`` along a JMA arc. Longitude is not wrapped into ±180."""
    if radius_m <= 0:
        return []
    math_start = -float(end_deg) + 90.0
    math_end = -float(start_deg) + 90.0
    if math_end < math_start:
        math_end += 360.0
    arc_lat, arc_lng = _meters_per_deg(lat)
    if arc_lat <= 0 or arc_lng <= 0:
        return []
    span = math_end - math_start
    steps = max(1, int(round(span / step_deg)))
    steps = min(steps, 96)
    pts: list[list[float]] = []
    for i in range(steps + 1):
        ang = math_start + span * (i / steps)
        rad = math.radians(ang)
        plat = (radius_m / arc_lat) * math.sin(rad) + lat
        plon = (radius_m / arc_lng) * math.cos(rad) + lon
        pts.append([plon, plat])
    return pts


def split_antimeridian(coords: list[list[float]]) -> list[list[list[float]]]:
    """Break a ``[lon, lat]`` line where a step jumps more than 180°."""
    if len(coords) < 2:
        return []
    parts: list[list[list[float]]] = []
    cur = [coords[0]]
    for pt in coords[1:]:
        if abs(pt[0] - cur[-1][0]) > 180:
            if len(cur) >= 2:
                parts.append(cur)
            cur = [pt]
        else:
            cur.append(pt)
    if len(cur) >= 2:
        parts.append(cur)
    return parts


def _is_full_circle(start: float, end: float) -> bool:
    span = abs(end - start)
    return span >= 359 or (abs(span % 360) < 1 and span > 180)


def _close(ring: list[list[float]]) -> list[list[float]]:
    if len(ring) >= 3 and ring[0] != ring[-1]:
        return [*ring, ring[0]]
    return ring


def _bbox(lats: list[float], lons: list[float]) -> tuple[float, float, float, float]:
    if not lats or not lons:
        return (120.0, 0.0, 160.0, 45.0)
    ref = lons[0]
    shifted: list[float] = []
    for lon in lons:
        x = lon
        while x - ref > 180:
            x -= 360
        while ref - x > 180:
            x += 360
        shifted.append(x)
    west, east = min(shifted) - 2.0, max(shifted) + 2.0
    south, north = min(lats) - 2.0, max(lats) + 2.0
    if east > 180:
        west -= 360
        east -= 360
    west = max(-180.0, west)
    east = min(180.0, east)
    south = max(-85.0, south)
    north = min(85.0, north)
    if east <= west:
        east = min(180.0, west + 2.0)
    return (west, south, east, north)


def _year(number: str, issued: str) -> int:
    if len(number) >= 2 and number[:2].isdigit():
        return 2000 + int(number[:2])
    if len(issued) >= 4 and issued[:4].isdigit():
        return int(issued[:4])
    return 0


def _spec_by_hour(spec: list) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for part in spec:
        if not isinstance(part, dict) or _part_name(part) == "title":
            continue
        hours = part.get("advancedHours")
        if isinstance(hours, (int, float)):
            out[int(hours)] = part
    return out


def _title(parts: list) -> dict:
    for part in parts:
        if isinstance(part, dict) and _part_name(part) == "title":
            return part
    return {}


def parse_storm(
    code: str,
    forecast: list,
    spec: list | None,
    *,
    issued_at: str = "",
    list_category: str = "",
) -> JmaStorm | None:
    """Build one storm from the two JSON documents. ``None`` if it has no center."""
    if not isinstance(forecast, list):
        return None
    spec = spec if isinstance(spec, list) else []
    code = code.upper()
    spec_title = _title(spec)
    fc_title = _title(forecast)
    name = spec_title.get("name") if isinstance(spec_title.get("name"), dict) else {}
    if not name and isinstance(fc_title.get("name"), dict):
        name = fc_title["name"]
    name_en = str(name.get("en") or "").strip()
    name_jp = str(name.get("jp") or "").strip()
    number = str(
        spec_title.get("typhoonNumber")
        or fc_title.get("typhoonNumber")
        or ""
    ).strip()
    issued = (
        issued_at
        or _valid_time(spec_title)
        or _valid_time(fc_title)
    )
    by_hour = _spec_by_hour(spec)
    analysis_spec = by_hour.get(0, {})
    class_en, class_jp = _category(analysis_spec)
    if not class_en:
        class_en, class_jp = _category(spec_title)
    if not class_en:
        class_en = list_category
    wind_ms, wind_kt, gust_ms, gust_kt = _wind(analysis_spec)
    pressure = _int(analysis_spec.get("pressure"))
    intensity = _intensity_label(analysis_spec.get("intensity"))

    observed: list[tuple[float, float]] = []
    fixes: list[JmaFix] = []
    circles: list[JmaCircle] = []
    paths: list[JmaPath] = []
    lats: list[float] = []
    lons: list[float] = []
    analysis_center: tuple[float, float] | None = None
    last_warning: dict | None = None

    for part in forecast:
        if not isinstance(part, dict) or _part_name(part) == "title":
            continue
        hours = part.get("advancedHours")
        if not isinstance(hours, (int, float)):
            continue
        hours_i = int(hours)
        center = part.get("center")
        if (
            isinstance(center, list)
            and len(center) >= 2
            and _num(center[0]) is not None
            and _num(center[1]) is not None
        ):
            lat, lon = float(center[0]), float(center[1])
        else:
            lat = lon = None
        if hours_i == 0:
            track = part.get("track") if isinstance(part.get("track"), dict) else {}
            for key in ("preTyphoon", "typhoon"):
                for pair in track.get(key) or []:
                    if (
                        isinstance(pair, list)
                        and len(pair) >= 2
                        and _num(pair[0]) is not None
                        and _num(pair[1]) is not None
                    ):
                        observed.append((float(pair[0]), float(pair[1])))
            if lat is not None and lon is not None:
                analysis_center = (lat, lon)
                if not observed or observed[-1] != (lat, lon):
                    observed.append((lat, lon))
            gale = part.get("galeWarningArea")
            if isinstance(gale, dict):
                gcenter = gale.get("center")
                glat = glon = None
                if isinstance(gcenter, list) and len(gcenter) >= 2:
                    glat, glon = _num(gcenter[0]), _num(gcenter[1])
                radius = _num(gale.get("radius"))
                if glat is not None and glon is not None and radius and radius > 0:
                    ring = _close(arc_lonlat(glat, glon, radius, 0, 360))
                    circles.append(JmaCircle(
                        "gale", 0, "", glon, glat, radius, ring,
                    ))
                    lats.extend([glat, glat])
                    lons.extend([glon - radius / 111_320, glon + radius / 111_320])
            storm = part.get("stormWarningArea")
            if isinstance(storm, dict):
                for arc in storm.get("arc") or []:
                    drawn = _arc_circle(arc, hours_i, "storm")
                    if drawn is not None:
                        circles.append(drawn)
                        lats.append(drawn.center_lat)
                        lons.append(drawn.center_lon)
        if lat is not None and lon is not None:
            spec_part = by_hour.get(hours_i, {})
            spec_ms, spec_kt, _, _ = _wind(spec_part)
            cat_en, cat_jp = _category(spec_part)
            fixes.append(JmaFix(
                hours_out=hours_i,
                valid_time=_valid_time(part),
                lat=lat,
                lon=lon,
                wind_kt=spec_kt if spec_kt is not None else (wind_kt if hours_i == 0 else None),
                wind_ms=spec_ms if spec_ms is not None else (wind_ms if hours_i == 0 else None),
                pressure_mb=_int(spec_part.get("pressure")) if spec_part else (pressure if hours_i == 0 else None),
                category_en=cat_en or (class_en if hours_i == 0 else ""),
                category_jp=cat_jp or (class_jp if hours_i == 0 else ""),
                intensity=_intensity_label(spec_part.get("intensity")) if spec_part else intensity,
            ))
            lats.append(lat)
            lons.append(lon)
        circle = part.get("probabilityCircle")
        if (
            isinstance(circle, dict)
            and isinstance(circle.get("tangent"), list)
            and lat is not None
            and lon is not None
        ):
            radius = _num(circle.get("radius"))
            if radius and radius > 0:
                cat_en, _cat_jp = _category(by_hour.get(hours_i, {}))
                label = f"+{hours_i}h"
                if cat_en == "LOW":
                    label += " extratropical"
                ring = _close(arc_lonlat(lat, lon, radius, 0, 360))
                circles.append(JmaCircle(
                    "probability", hours_i, label, lon, lat, radius, ring,
                ))
                arc_lat, arc_lng = _meters_per_deg(lat)
                if arc_lat > 0 and arc_lng > 0:
                    lats.extend([lat - radius / arc_lat, lat + radius / arc_lat])
                    lons.extend([lon - radius / arc_lng, lon + radius / arc_lng])
            for seg in circle.get("tangent") or []:
                path = _latlon_path(seg)
                for piece in split_antimeridian(path):
                    paths.append(JmaPath("tangent", hours_i, piece))
        warning = part.get("stormWarningArea")
        if isinstance(warning, dict) and hours_i > 0:
            last_warning = {"hours": hours_i, "area": warning}

    if analysis_center is None and not fixes:
        return None

    if last_warning is not None:
        area = last_warning["area"]
        hours_i = int(last_warning["hours"])
        for arc in area.get("arc") or []:
            for piece in _arc_paths(arc):
                paths.append(JmaPath("storm-arc", hours_i, piece))
        for seg in area.get("line") or []:
            path = _latlon_path(seg)
            for piece in split_antimeridian(path):
                paths.append(JmaPath("storm-line", hours_i, piece))

    for pt in observed:
        lats.append(pt[0])
        lons.append(pt[1])

    gale_ranges = [
        text for text in (
            _range_text(item) for item in (analysis_spec.get("galeWarning") or [])
        ) if text
    ]
    storm_ranges = [
        text for text in (
            _range_text(item) for item in (analysis_spec.get("stormWarning") or [])
        ) if text
    ]
    display = name_en or name_jp or "Unnamed"
    classification = class_en or "TC"
    class_label = "extratropical" if classification == "LOW" else classification
    kt_bit = f", {wind_kt} kt" if wind_kt else ""
    label = f"{display} ({number or code}) — JMA {class_label}{kt_bit}"
    lat0 = analysis_center[0] if analysis_center else (fixes[0].lat if fixes else None)
    lon0 = analysis_center[1] if analysis_center else (fixes[0].lon if fixes else None)
    summary = JmaSummary(
        storm_id=code,
        name=display,
        name_jp=name_jp,
        year=_year(number, issued),
        classification=classification,
        intensity_kt=wind_kt or 0,
        pressure_mb=pressure,
        lat=lat0,
        lon=lon0,
        label=label,
    )
    return JmaStorm(
        summary=summary,
        issued_at=issued,
        typhoon_number=number,
        category_jp=class_jp,
        intensity=intensity,
        wind_ms=wind_ms,
        gust_ms=gust_ms,
        gust_kt=gust_kt,
        gale_ranges=gale_ranges,
        storm_ranges=storm_ranges,
        observed=observed,
        forecast=fixes,
        circles=circles,
        paths=paths,
        bbox=_bbox(lats, lons),
    )


def _latlon_path(seg) -> list[list[float]]:
    """JMA pairs are ``[lat, lon]``. GeoJSON wants ``[lon, lat]``."""
    out: list[list[float]] = []
    if not isinstance(seg, list):
        return out
    for pair in seg:
        if (
            isinstance(pair, list)
            and len(pair) >= 2
            and _num(pair[0]) is not None
            and _num(pair[1]) is not None
        ):
            out.append([float(pair[1]), float(pair[0])])
    return out


def _arc_endpoints(arc) -> tuple[float, float, float, float, float] | None:
    if not isinstance(arc, list) or len(arc) < 2:
        return None
    center, radius = arc[0], _num(arc[1])
    if (
        not isinstance(center, list)
        or len(center) < 2
        or radius is None
        or radius <= 0
        or _num(center[0]) is None
        or _num(center[1]) is None
    ):
        return None
    span = arc[2] if len(arc) >= 3 and isinstance(arc[2], list) and len(arc[2]) >= 2 else [0, 360]
    start, end = _num(span[0]), _num(span[1])
    if start is None or end is None:
        start, end = 0.0, 360.0
    return float(center[0]), float(center[1]), radius, start, end


def _arc_circle(arc, hours: int, kind: str) -> JmaCircle | None:
    parsed = _arc_endpoints(arc)
    if parsed is None:
        return None
    lat, lon, radius, start, end = parsed
    if not _is_full_circle(start, end):
        return None
    ring = _close(arc_lonlat(lat, lon, radius, 0, 360))
    if len(ring) < 4:
        return None
    return JmaCircle(kind, hours, "", lon, lat, radius, ring)


def _arc_paths(arc) -> list[list[list[float]]]:
    parsed = _arc_endpoints(arc)
    if parsed is None:
        return []
    lat, lon, radius, start, end = parsed
    return split_antimeridian(arc_lonlat(lat, lon, radius, start, end, step_deg=8))


def fetch_typhoon(code: str, *, refresh: bool = False) -> JmaStorm | None:
    code = (code or "").upper()
    if not is_jma_id(code):
        return None
    if not refresh:
        hit = _STORM_CACHE.get(code)
        if hit is not None:
            return hit
    forecast = _get_json(FORECAST_URL.format(code=code))
    if not isinstance(forecast, list):
        return None
    spec = _get_json(SPEC_URL.format(code=code))
    storm = parse_storm(code, forecast, spec if isinstance(spec, list) else [])
    if storm is None:
        return None
    _STORM_CACHE.set(code, storm)
    return storm


def fetch_active_typhoons(*, refresh: bool = False) -> list[JmaSummary]:
    """Systems JMA is issuing right now. Empty when the basin is quiet."""
    if not refresh:
        hit = _LIST_CACHE.get("active")
        if hit is not None:
            return hit
    raw = _get_json(TARGET_URL)
    if not isinstance(raw, list):
        return []
    codes: list[str] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        code = str(row.get("tropicalCyclone") or "").upper()
        if is_jma_id(code):
            codes.append(code)
    storms: list[JmaStorm] = []
    if codes:
        workers = min(4, len(codes))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(fetch_typhoon, code, refresh=refresh) for code in codes]
            for fut in as_completed(futs):
                try:
                    storm = fut.result()
                except Exception:  # noqa: BLE001
                    storm = None
                if storm is not None:
                    storms.append(storm)
    storms.sort(key=lambda s: s.summary.storm_id)
    summaries = [s.summary for s in storms]
    _LIST_CACHE.set("active", summaries)
    return summaries
