"""NOAA HURDAT2 fetch + parse — historical Atlantic hurricane tracks.

Format spec:   https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf
Current data:  https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt

The file is plain text:
  - Storm header line:   ``AL{NN}{YYYY}, {NAME}, {N_RECS},``
  - N data rows follow:  ``YYYYMMDD, HHMM, {RID}, {STATUS}, {LAT}, {LON}, {WIND}, {PRES}, …``
    * RID column carries record identifier — ``L`` = landfall.
    * Coordinates have a trailing hemisphere letter (``28.0N`` / ``94.8W``).
    * Missing numerics use ``-99`` / ``-999``.

We fetch the file once per cold-start (lru_cache), keep only storms from
1950+, and expose a slimmed JSON for the frontend map layer.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from ..brand import USER_AGENT

HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
FETCH_TIMEOUT_S = 30
MIN_YEAR = 1950


@dataclass(slots=True)
class TrackPoint:
    datetime_utc: str       # ISO-8601 (e.g. "2017-08-25T18:00:00Z")
    record_id: str          # "" / "L" (landfall) / "P" / "C" / etc.
    status: str             # TD, TS, HU, EX, SD, SS, LO, WV, DB
    lat: float
    lon: float
    wind_kt: int            # 0 if missing
    pressure_mb: int | None


@dataclass(slots=True)
class Storm:
    storm_id: str           # "AL011950"
    name: str               # "ABLE" / "UNNAMED"
    year: int
    track: list[TrackPoint]


# ─────────────────────────── parsing ───────────────────────────


def _parse_lat(s: str) -> float:
    s = s.strip()
    if s.endswith("N"):
        return float(s[:-1])
    if s.endswith("S"):
        return -float(s[:-1])
    return float(s)


def _parse_lon(s: str) -> float:
    s = s.strip()
    if s.endswith("W"):
        return -float(s[:-1])
    if s.endswith("E"):
        return float(s[:-1])
    return float(s)


def _parse_int(s: str) -> int | None:
    s = s.strip()
    if not s or s in {"-99", "-999"}:
        return None
    return int(s)


def _parse(raw: str) -> list[Storm]:
    """Walk the HURDAT2 text and yield Storm records from 1950 onward."""
    storms: list[Storm] = []
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        header = [s.strip() for s in lines[i].split(",")]
        if len(header) < 3 or not header[0].startswith("AL"):
            i += 1
            continue
        storm_id = header[0]
        name = header[1] if header[1] else "UNNAMED"
        try:
            n_rec = int(header[2])
        except ValueError:
            i += 1
            continue
        try:
            year = int(storm_id[4:8])
        except ValueError:
            i += 1
            continue

        if year < MIN_YEAR:
            i += 1 + n_rec
            continue

        track: list[TrackPoint] = []
        for j in range(n_rec):
            row_idx = i + 1 + j
            if row_idx >= len(lines):
                break
            row = [s.strip() for s in lines[row_idx].split(",")]
            if len(row) < 8:
                continue
            try:
                dt, hhmm = row[0], row[1]
                datetime_utc = (
                    f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}T{hhmm[:2]}:{hhmm[2:]}:00Z"
                )
                wind = _parse_int(row[6]) or 0
                pres = _parse_int(row[7])
                track.append(
                    TrackPoint(
                        datetime_utc=datetime_utc,
                        record_id=row[2],
                        status=row[3],
                        lat=_parse_lat(row[4]),
                        lon=_parse_lon(row[5]),
                        wind_kt=wind,
                        pressure_mb=pres,
                    )
                )
            except (ValueError, IndexError):
                continue

        if track:
            storms.append(Storm(storm_id=storm_id, name=name, year=year, track=track))
        i += 1 + n_rec
    return storms


@lru_cache(maxsize=1)
def fetch_and_parse() -> list[Storm]:
    """Live-fetch + parse HURDAT2. Memoised per-process (cold start cost only)."""
    req = urllib.request.Request(
        HURDAT2_URL,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return _parse(raw)


# ─────────────────────────── analytics ───────────────────────────


def category_for_wind(wind_kt: int) -> int:
    """Saffir-Simpson. -1=TD, 0=TS, 1..5=Cat 1..5."""
    if wind_kt >= 137:
        return 5
    if wind_kt >= 113:
        return 4
    if wind_kt >= 96:
        return 3
    if wind_kt >= 83:
        return 2
    if wind_kt >= 64:
        return 1
    if wind_kt >= 34:
        return 0
    return -1


# US coastal-state bounding boxes — first-match wins on landfall lookup.
# Imprecise at borders but plenty good for "which state did Katrina hit".
STATE_BBOXES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    # state: ((lat_min, lon_min), (lat_max, lon_max))
    "FL": ((24.4, -87.7), (31.1, -79.9)),
    "GA": ((30.4, -85.7), (35.0, -80.7)),
    "SC": ((32.0, -83.4), (35.2, -78.5)),
    "NC": ((33.7, -84.4), (36.6, -75.4)),
    "VA": ((36.5, -83.7), (39.5, -75.2)),
    "MD": ((37.9, -79.5), (39.7, -75.0)),
    "DE": ((38.4, -75.8), (39.9, -75.0)),
    "NJ": ((38.8, -75.7), (41.4, -73.8)),
    "NY": ((40.4, -79.8), (45.1, -71.7)),
    "CT": ((40.9, -73.8), (42.1, -71.7)),
    "RI": ((41.1, -71.9), (42.1, -71.0)),
    "MA": ((41.2, -73.6), (43.0, -69.8)),
    "NH": ((42.6, -72.7), (45.3, -70.5)),
    "ME": ((42.9, -71.2), (47.6, -66.8)),
    "AL": ((30.1, -88.6), (35.0, -84.8)),
    "MS": ((30.0, -91.8), (35.1, -88.0)),
    "LA": ((28.8, -94.1), (33.1, -88.7)),
    "TX": ((25.7, -106.7), (36.6, -93.4)),
    "PR": ((17.8, -67.4), (18.6, -65.2)),
}


def state_for_point(lat: float, lon: float) -> str | None:
    for state, ((lat_min, lon_min), (lat_max, lon_max)) in STATE_BBOXES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return state
    return None


def landfall_summary(storm: Storm) -> tuple[int, str | None]:
    """Return (highest SS category at any landfall, state of strongest US landfall).

    A storm with no ``L`` record returns ``(-2, None)``. Category is the
    storm's strongest landfall ANYWHERE — that's the meteorological "landfall
    intensity". The state, however, is taken from the strongest US-bbox
    landfall (if any). So Irma 2017 → Cat 5 + state=FL; Maria 2017 →
    Cat 5 + state=PR; Lorenzo 2019 → Cat 5 + state=None (Azores).
    """
    landfalls = [p for p in storm.track if "L" in p.record_id]
    if not landfalls:
        return -2, None
    overall = max(landfalls, key=lambda p: p.wind_kt)
    overall_cat = category_for_wind(overall.wind_kt)
    us_landfalls = [
        (p, s) for p in landfalls if (s := state_for_point(p.lat, p.lon))
    ]
    if us_landfalls:
        strongest_us = max(us_landfalls, key=lambda ps: ps[0].wind_kt)
        return overall_cat, strongest_us[1]
    return overall_cat, None


def peak_wind(storm: Storm) -> int:
    return max((p.wind_kt for p in storm.track if p.wind_kt > 0), default=0)
