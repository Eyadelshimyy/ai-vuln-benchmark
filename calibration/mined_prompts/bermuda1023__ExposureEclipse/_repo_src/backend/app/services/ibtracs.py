"""NOAA IBTrACS — Atlantic basin hurricane tracks + recon Rmax.

NOAA's International Best Track Archive for Climate Stewardship (IBTrACS)
v04r01 North-Atlantic CSV is the source of BOTH:
  - storm tracks (3-hour interpolated USA fixes — denser than HURDAT2's 6-hour
    native, smoother path lines and better hurricane-strength coverage)
  - recon-measured radius of maximum winds (USA_RMW)

Spec PDF / column docs:
  https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf

CSV file:
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv

The CSV is ~70 MB. One fetch per cold-start, one parse, two indexes built:

  * ``_rmax_index()``  — ``{(usa_atcf_id, 'YYYYMMDDHHMM') -> rmw_nm}``
  * ``_storms_index()`` — ``{usa_atcf_id -> Storm(track=[TrackPoint, ...])}``

Both shapes are intentionally compatible with the previous HURDAT2 module so
callers (hurricanes API, hurricane_impact service) don't need to know which
source the tracks came from.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache

from ..brand import USER_AGENT

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
FETCH_TIMEOUT_S = 120
MIN_SEASON = 1950  # ignore everything older than what we render in the UI


# ───────────────────────── data shapes (HURDAT2-compatible) ─────────────────────────


@dataclass(slots=True)
class TrackPoint:
    datetime_utc: str       # ISO-8601 "YYYY-MM-DDTHH:MM:SSZ"
    record_id: str          # "L" when this fix is a landfall, "" otherwise
    status: str             # IBTrACS USA_STATUS code (HU, TS, TD, EX, SD, SS, LO, WV, DB, ET)
    lat: float
    lon: float
    wind_kt: int            # USA_WIND; 0 if missing
    pressure_mb: int | None
    # Optional wind field. Live NHC a-deck sets all of these (radii_source
    # "nhc"); IBTrACS sets only the best-track R50/R34 quadrants, which
    # impact reads when asked for the tropical-storm field.
    rmax_nm: float | None = None
    r64_quads_nm: tuple[float, float, float, float] | None = None
    r50_quads_nm: tuple[float, float, float, float] | None = None
    r34_quads_nm: tuple[float, float, float, float] | None = None
    radii_source: str | None = None  # "nhc" | None
    # Live forecast only. Hour from the advisory (negative = pre-advisory
    # history) and the NHC cone radius at that lead time.
    forecast_hour: int | None = None
    cone_radius_nm: float | None = None


@dataclass(slots=True)
class Storm:
    storm_id: str           # USA_ATCF_ID, e.g. "AL092022"
    name: str
    year: int
    track: list[TrackPoint] = field(default_factory=list)
    # Live NHC cone of uncertainty, (lon, lat), closed. Empty for history.
    forecast_cone: list[tuple[float, float]] | None = None


# ───────────────────────── single-pass CSV parse ─────────────────────────


def _parse_int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        v = int(float(s))  # IBTrACS occasionally writes "120.0"
    except ValueError:
        return None
    return v if v > -90 else None  # IBTrACS uses negative sentinels for missing


def _parse_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _quads(row: list[str], cols: list[int]) -> tuple[float, float, float, float]:
    out: list[float] = []
    for ci in cols:
        v = row[ci].strip()
        try:
            out.append(max(0.0, float(v)) if v else 0.0)
        except ValueError:
            out.append(0.0)
    return tuple(out)  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _parse_csv() -> tuple[
    dict[tuple[str, str], float],                                # rmax index
    dict[tuple[str, str], float],                                # r64 mean (legacy)
    dict[tuple[str, str], tuple[float, float, float, float]],    # r64 quadrants (NE, SE, SW, NW)
    dict[str, Storm],                                            # storms by ATCF id
]:
    """Single-pass fetch + parse → (rmax_idx, r64_mean_idx, r64_quads_idx, storms)."""
    req = urllib.request.Request(
        IBTRACS_URL, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(raw))
    header = next(reader, None)
    if header is None:
        return {}, {}, {}, {}
    # Row 2 is units; skip.
    _units = next(reader, None)

    try:
        i_atcf = header.index("USA_ATCF_ID")
        i_time = header.index("ISO_TIME")
        i_rmw = header.index("USA_RMW")
        i_season = header.index("SEASON")
        i_name = header.index("NAME")
        i_lat = header.index("USA_LAT")
        i_lon = header.index("USA_LON")
        i_wind = header.index("USA_WIND")
        i_pres = header.index("USA_PRES")
        i_status = header.index("USA_STATUS")
        i_record = header.index("USA_RECORD")
        i_r64 = [header.index(f"USA_R64_{q}") for q in ("NE", "SE", "SW", "NW")]
        i_r50 = [header.index(f"USA_R50_{q}") for q in ("NE", "SE", "SW", "NW")]
        i_r34 = [header.index(f"USA_R34_{q}") for q in ("NE", "SE", "SW", "NW")]
    except ValueError:
        # Header layout drifted — bail out clean; downstream falls back gracefully.
        return {}, {}, {}, {}

    rmax_index: dict[tuple[str, str], float] = {}
    r64_index: dict[tuple[str, str], float] = {}
    r64_quads_index: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    storms: dict[str, Storm] = {}
    max_col = max(
        i_atcf, i_time, i_rmw, i_season, i_name,
        i_lat, i_lon, i_wind, i_pres, i_status, i_record,
        *i_r64, *i_r50, *i_r34,
    )

    for row in reader:
        if not row or len(row) <= max_col:
            continue

        season_raw = row[i_season].strip()
        try:
            season = int(season_raw)
        except ValueError:
            continue
        if season < MIN_SEASON:
            continue

        atcf = row[i_atcf].strip().upper()
        if not atcf:
            continue  # skip storms NHC didn't track — they have no ATCF id

        iso = row[i_time].strip()
        if len(iso) < 16:
            continue
        key_dt = iso[0:4] + iso[5:7] + iso[8:10] + iso[11:13] + iso[14:16]

        # Rmax index — only rows where USA_RMW is set get in.
        rmw_raw = row[i_rmw].strip()
        if rmw_raw:
            try:
                rmw = float(rmw_raw)
                if rmw > 0:
                    rmax_index[(atcf, key_dt)] = rmw
            except ValueError:
                pass

        # R64 — radius of 64-kt winds in each of four quadrants (NE, SE, SW, NW).
        # We keep two indexes:
        #   - r64_index:  mean across non-zero quadrants (legacy, used by the
        #                 detail panel + as the "symmetric" half-width when a
        #                 caller doesn't want per-bearing detail).
        #   - r64_quads_index: the raw four values, used to build the
        #                 ASYMMETRIC outer cone + asymmetric county capture
        #                 via linear interpolation between adjacent quadrants.
        quad_raw = _quads(row, i_r64)
        if any(v > 0 for v in quad_raw):
            r64_quads_index[(atcf, key_dt)] = tuple(quad_raw)  # type: ignore[assignment]
            nonzero = [v for v in quad_raw if v > 0]
            r64_index[(atcf, key_dt)] = sum(nonzero) / len(nonzero)

        # Track point — needs at minimum lat/lon.
        lat = _parse_float(row[i_lat])
        lon = _parse_float(row[i_lon])
        if lat is None or lon is None:
            continue

        wind = _parse_int(row[i_wind]) or 0
        pres = _parse_int(row[i_pres])
        status = row[i_status].strip() or "XX"
        record = row[i_record].strip()  # "L" when landfall
        # ISO from IBTrACS is "YYYY-MM-DD HH:MM:SS"; emit ISO-8601 Z form so
        # downstream consumers (frontend, lookup_rmax_nm) see the same shape
        # the HURDAT2 parser used to produce.
        dt_iso = f"{iso[0:10]}T{iso[11:19]}Z"

        s = storms.get(atcf)
        if s is None:
            name = row[i_name].strip() or "UNNAMED"
            s = Storm(storm_id=atcf, name=name, year=season)
            storms[atcf] = s

        r50 = _quads(row, i_r50)
        r34 = _quads(row, i_r34)
        s.track.append(
            TrackPoint(
                datetime_utc=dt_iso,
                record_id=record,
                status=status,
                lat=lat,
                lon=lon,
                wind_kt=wind,
                pressure_mb=pres,
                r50_quads_nm=r50 if any(v > 0 for v in r50) else None,
                r34_quads_nm=r34 if any(v > 0 for v in r34) else None,
            )
        )

    # Sort each track chronologically — IBTrACS rows usually are, but the
    # USA_* values can be sparse and a defensive sort costs nothing.
    for s in storms.values():
        s.track.sort(key=lambda p: p.datetime_utc)

    return rmax_index, r64_index, r64_quads_index, storms


def _rmax_index() -> dict[tuple[str, str], float]:
    return _parse_csv()[0]


def _r64_index() -> dict[tuple[str, str], float]:
    return _parse_csv()[1]


def _r64_quads_index() -> dict[tuple[str, str], tuple[float, float, float, float]]:
    return _parse_csv()[2]


def _storms_index() -> dict[str, Storm]:
    return _parse_csv()[3]


# ───────────────────────── public API ─────────────────────────


def fetch_storms() -> list[Storm]:
    """All Atlantic storms from MIN_SEASON onward, each with its full track.

    Cheap after the first call (lru_cached single-pass parse). Drop-in
    replacement for ``hurdat2.fetch_and_parse``.
    """
    return list(_storms_index().values())


def _normalize_datetime(dt_raw: str) -> str | None:
    """Reduce any of our datetime formats to the 'YYYYMMDDHHMM' index key.

    Accepts:
      - ``YYYYMMDDHH``         (raw HURDAT2, 10 digits)
      - ``YYYYMMDDHHMM``       (raw HURDAT2 + minutes, 12 digits)
      - ``YYYY-MM-DDTHH:MM:SSZ``  (ISO form our parsers emit)
      - ``YYYY-MM-DD HH:MM:SS`` (IBTrACS native)
    """
    dt = dt_raw.strip()
    digits = "".join(ch for ch in dt if ch.isdigit())
    if len(digits) >= 12:
        return digits[:12]
    if len(digits) == 10:
        return digits + "00"
    return None


def lookup_rmax_nm(storm_id: str | None, datetime_utc: str | None) -> float | None:
    """Return IBTrACS-measured Rmax (nautical miles) for one fix, or None.

    Degrades to None on any IBTrACS fetch/parse failure — the caller (rmax_nm
    in hurricane_impact) falls back to the Willoughby estimate. Prevents the
    live-storm bundle from 5xx'ing when NCEI is slow or the Vercel cold-start
    parse times out."""
    if not storm_id or not datetime_utc:
        return None
    key_dt = _normalize_datetime(datetime_utc)
    if key_dt is None:
        return None
    try:
        return _rmax_index().get((storm_id.upper(), key_dt))
    except Exception:  # noqa: BLE001
        return None


def lookup_r64_nm(storm_id: str | None, datetime_utc: str | None) -> float | None:
    """Return IBTrACS-measured MEAN R64 (nautical miles) for one fix, or None.

    Convenience wrapper that averages the non-zero quadrants. Use
    ``lookup_r64_quads_nm`` if you need the per-quadrant detail. Degrades to
    None on IBTrACS unavailability (same rationale as ``lookup_rmax_nm``).
    """
    if not storm_id or not datetime_utc:
        return None
    key_dt = _normalize_datetime(datetime_utc)
    if key_dt is None:
        return None
    try:
        return _r64_index().get((storm_id.upper(), key_dt))
    except Exception:  # noqa: BLE001
        return None


def lookup_r64_quads_nm(
    storm_id: str | None,
    datetime_utc: str | None,
) -> tuple[float, float, float, float] | None:
    """Return IBTrACS-measured R64 per quadrant ``(NE, SE, SW, NW)`` in nm.

    Zero values are kept (meaning "no 64-kt winds in that quadrant"). NHC
    didn't systematically record R64 before ~2004, so older storms return
    None — caller should fall back to a synthetic / symmetric radius.
    """
    if not storm_id or not datetime_utc:
        return None
    key_dt = _normalize_datetime(datetime_utc)
    if key_dt is None:
        return None
    try:
        return _r64_quads_index().get((storm_id.upper(), key_dt))
    except Exception:  # noqa: BLE001
        return None


def warm_cache() -> tuple[int, int, int]:
    """Prime the cache; returns (n_storms, n_rmax_records, n_r64_records)."""
    rmw = _rmax_index()
    r64 = _r64_index()
    storms = _storms_index()
    return len(storms), len(rmw), len(r64)
