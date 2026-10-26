"""Interpolated surface-wind field for a live-storm bbox.

Blends NDBC buoy + a spatially-uniform sample of NWS land-station reports into
a smoothed inverse-distance-weighted (IDW) wind-speed grid.

The point of doing this in-house instead of just plotting the raw obs is that
NOAA's networks are wildly non-uniform (Texas has ~10× the mesonet density of
Louisiana), individual stations go dead or return stuck-at-zero readings, and
observation times drift. Rendering the raw markers implies precision the data
doesn't have. Smoothing + cleaning produces a fair "current wind footprint"
map the underwriter can actually read.

Cleaning rules, in order:
1. Drop obs older than ``MAX_AGE_HOURS``.
2. Drop obs whose wind_kt is None.
3. Kill suspicious zeros: 0 kt readings whose local (~1.5°) neighborhood
   median is above ``LOCAL_KILL_MEDIAN_KT`` are treated as dead sensors and
   removed rather than dragging the IDW mean toward zero.

Interpolation:
- Adaptive grid step by bbox span (see ``_adaptive_step``).
- IDW power = 2.0, soft influence radius ``IDW_RADIUS_DEG`` in degrees.
- Cells with no obs inside the radius are omitted — the renderer paints
  "no data" as a gap instead of extrapolating.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

from .marine_obs import (
    _fetch_latest_obs,
    _nws_stations_in_bbox,
    _spatial_subsample,
    buoys_in_bbox,
)
from .ttl_cache import TtlCache


@dataclass(slots=True, frozen=True)
class WindGridCell:
    lat: float
    lon: float
    wind_kt: float
    wind_dir_deg: float | None   # meteorological convention: FROM direction
    sources: int                 # obs contributing to this cell
    confidence: float            # 0..1 heuristic — see _cell_confidence
    nearest_obs_km: float | None # closest contributing observation
    # Individual score components (0..1) so the frontend popup can show
    # WHY a cell scored HIGH / MED / LOW instead of just the multiplied
    # composite. Users can then judge whether an "HIGH at 60 km" cell is
    # trustworthy — usually it is when 3+ obs agree tightly.
    dist_score: float
    count_score: float
    agreement_score: float
    contributor_spread_kt: float | None  # σ of contributor speeds


@dataclass(slots=True, frozen=True)
class WindObs:
    """One cleaned surface observation used to build the wind heatmap.
    Shipped alongside the grid so the frontend can highlight which stations
    contributed to any given cell when the user clicks "N sources"."""
    lat: float
    lon: float
    wind_kt: float
    wind_dir_deg: float | None
    source: str          # "buoy" | "land" | "recon"
    station_id: str
    observed_at: str     # ISO
    # IDW multiplier. 1 for a fresh buoy. Recon samples decay with age
    # because the vortex evolves even after we slide them with the center.
    weight: float = 1.0


MAX_AGE_HOURS = 4.0
LOCAL_KILL_MEDIAN_KT = 5.0
# Was 60 with spatial subsampling — that thinned dense metro coverage
# (Houston / DFW have 30+ ASOS+mesonet sites) to a sparse skeleton. Now we
# use every station in the bbox up to a safety cap so IDW has as many
# neighbours as possible. Cap keeps latency bounded when a bbox covers many
# CONUS states.
LAND_STATION_CAP = 800
LAND_STATION_WORKERS = 48
IDW_POWER = 2.0
IDW_RADIUS_DEG = 3.0             # ~330 km at mid-latitudes
# Recon SFMR is a dense transect through the core. A 3° radius would smear
# eyewall winds across the whole basin; keep influence tight so hunters
# fill the hole buoys leave without painting a 300 km ribbon.
IDW_RADIUS_RECON_DEG = 0.8       # ~90 km
NEIGHBOR_RADIUS_DEG = 1.5        # for the dead-sensor check
OUTLIER_MAX_DEV_KT = 25.0        # |obs - local_median| above this ⇒ drop
ABS_MAX_WIND_KT = 200.0          # world-record verifiable sustained ~= 220 kt
# Cell step for the observed heatmap. Matches Open-Meteo's native model
# resolution (GFS ~0.25°, ECMWF IFS 0.25°) so that switching between Obs /
# GFS / ECMWF views doesn't visibly change the tiling and diffs are exact
# cell-to-cell subtractions. Was 0.5° — bumping to 0.25° quadruples the
# cell count but keeps interpolation cost trivial (few hundred obs × few
# thousand cells = tens of ms in Python).
FIXED_STEP_DEG = 0.25
# Confidence thresholds (in output space) are set on the frontend. Below are
# the raw-signal knobs the confidence composite uses.
NEAR_OBS_DIST_DEG = 1.5           # nearest obs within this ⇒ full distance score

# Storm-relative persistence for hunter samples. The plane measured a
# wind in the vortex; the vortex has since moved. We slide the sample
# with the center instead of leaving it in the wake. See advect_recon_obs.
CORE_RADIUS_NM = 200.0             # inside this of the center at obs time → vortex
ADVECT_MAX_AGE_H = 6.0             # older than this, the structure has changed
FRESH_KEEP_H = 1.5                 # no motion estimate: keep only this fresh
AGE_TAU_H = 3.0                    # weight = exp(-age / tau)
EXTRAP_MAX_H = 3.0                 # don't run the motion vector past the forecast
# Buoy/land pool from the last successful heatmap, so a recon poll can
# re-grid without refetching every land station.
_SURFACE_CACHE: TtlCache[tuple[float, float, float, float], list[WindObs]] = (
    TtlCache(ttl_s=45 * 60, maxsize=8)
)


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _adaptive_step(span: float) -> float:
    """Kept for backwards compat but no longer called from the main
    interpolator — see FIXED_STEP_DEG. We standardized on 0.5° so that the
    observed heatmap is cell-aligned with the GFS/ECMWF model grids, which
    lets the frontend compute obs-vs-model diffs cell-by-cell without
    resampling. Small enough to look continuous, large enough to keep
    Open-Meteo bulk requests manageable."""
    if span < 6:
        return 0.15
    if span < 12:
        return 0.25
    if span < 25:
        return 0.4
    return 0.6


def _fetch_land_obs_with_meta(
    west: float, south: float, east: float, north: float,
) -> list[tuple[float, float, float, float | None, str, str]]:
    """Fetch the latest wind observation for every NWS station in the bbox
    (up to ``LAND_STATION_CAP``). Runs in parallel across
    ``LAND_STATION_WORKERS`` threads.

    Previously spatial-subsampled to 60 stations for even distribution, but
    that thinned metro clusters (Houston / DFW have 30+ ASOS + mesonet
    sites) to a small skeleton — the IDW field ended up under-constrained
    exactly where the most ground-truth was available. Now we take every
    station and let the IDW's inverse-square weighting handle the natural
    non-uniform density (a dense city cluster just gives that area
    higher-confidence values, which is the correct behaviour).

    Returns tuples of (lat, lon, wind_kt, wind_dir_deg, observed_at,
    station_id). ``wind_dir_deg`` is None when the station reports speed
    but not direction.
    """
    # Pull only stations for states overlapping the bbox instead of
    # scanning NWS's full global list. Old approach paged /stations
    # globally and quit at 30 pages, which never reached the K-prefixed
    # ASOS/AWOS at every major US airport (they sort after '0'-'9', 'A',
    # 'B', 'C' — an ID range that alone eats 15k+ stations). This yields
    # 3–8 fast per-state calls for a typical storm bbox and includes every
    # ASOS + mesonet the states own.
    all_in_bbox: list[tuple[str, str, float, float]] = []
    for f in _nws_stations_in_bbox(west, south, east, north):
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

    # A Bertha-sized bbox now yields ~8k candidates (was ~120 pre-state-
    # fetch). Fetching latest obs for all of them serially would blow past
    # Vercel's 30 s serverless budget even with 32-way parallelism. Instead:
    #   1. Always include every K-prefixed station (ASOS/AWOS — airport
    #      -grade quality, hourly hourly updates, mission-critical for
    #      aviation so extremely reliable). Typically 500-800 per bbox.
    #   2. Fill the remaining LAND_STATION_CAP budget with a spatially
    #      -uniform sample of everything else (mesonet, RAWS, COOP, etc.)
    #      so mid-bbox holes get covered.
    airport_stations = [c for c in all_in_bbox if c[0].startswith("K")]
    other_stations = [c for c in all_in_bbox if not c[0].startswith("K")]
    airport_stations = airport_stations[:LAND_STATION_CAP]
    remaining_budget = max(0, LAND_STATION_CAP - len(airport_stations))
    if remaining_budget > 0 and other_stations:
        other_subset = _spatial_subsample(
            other_stations, west, south, east, north, remaining_budget,
        )
    else:
        other_subset = []
    candidates = airport_stations + other_subset

    out: list[tuple[float, float, float, float | None, str, str]] = []
    with ThreadPoolExecutor(max_workers=LAND_STATION_WORKERS) as pool:
        futures = [
            pool.submit(_fetch_latest_obs, sid, name, lat, lon)
            for sid, name, lat, lon in candidates
        ]
        for fut in as_completed(futures, timeout=45):
            try:
                rec = fut.result()
            except Exception:  # noqa: BLE001
                continue
            if rec is None or rec.wind_kt is None:
                continue
            out.append((
                rec.lat, rec.lon, float(rec.wind_kt),
                float(rec.wind_dir_deg) if rec.wind_dir_deg is not None else None,
                rec.observed_at, rec.station_id,
            ))
    return out


def _clean_obs(
    obs: list[tuple[float, float, float, float | None, str, str]],
    now: datetime,
) -> list[tuple[float, float, float, float | None, str]]:
    """Multi-stage cleaning pipeline for the raw NDBC + NWS + recon obs pool.

    1. Age: drop obs whose ``observed_at`` is older than ``MAX_AGE_HOURS``
       (recon is pre-filtered to 8 h by the recon fetcher; we still apply
       the 4 h cap to buoy/land).
    2. Absurdity: drop obs with wind_kt above ``ABS_MAX_WIND_KT`` (world
       -record sustained wind ≈ 220 kt — anything higher is anemometer
       error or corrupt data).
    3. Dead-sensor zero: 0 kt reading whose local (1.5°) neighborhood
       median is above ``LOCAL_KILL_MEDIAN_KT`` is dropped as a stuck-at
       -zero sensor rather than pulling the IDW mean toward zero.
       **Skipped for recon** — a calm eye is a real 0.
    4. Outlier: obs whose wind_kt deviates from the local median by more
       than ``OUTLIER_MAX_DEV_KT`` are dropped. Catches lightning-strike
       anemometer spikes and single-station spurious readings without
       needing per-station quality flags.
       **Skipped for recon** — eyewall SFMR is supposed to be tens of kt
       above the surrounding buoy field.
    """
    fresh: list[tuple[float, float, float, float | None, str]] = []
    for lat, lon, kt, dir_deg, iso, source in obs:
        dt = _parse_iso(iso)
        if dt is not None:
            age_h = (now - dt).total_seconds() / 3600.0
            cap = 8.0 if source == "recon" else MAX_AGE_HOURS
            if age_h > cap:
                continue
        if kt > ABS_MAX_WIND_KT:
            continue
        fresh.append((lat, lon, kt, dir_deg, source))

    def _local_median(lat: float, lon: float, exclude_self_kt: float) -> float | None:
        vals = [
            k for (la, lo, k, _d, _s) in fresh
            if abs(la - lat) < NEIGHBOR_RADIUS_DEG
            and abs(lo - lon) < NEIGHBOR_RADIUS_DEG
            and (la, lo, k) != (lat, lon, exclude_self_kt)
        ]
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    cleaned: list[tuple[float, float, float, float | None, str]] = []
    for lat, lon, kt, dir_deg, source in fresh:
        if source == "recon":
            cleaned.append((lat, lon, kt, dir_deg, source))
            continue
        local_med = _local_median(lat, lon, kt)
        if kt == 0 and local_med is not None and local_med > LOCAL_KILL_MEDIAN_KT:
            continue
        if local_med is not None and abs(kt - local_med) > OUTLIER_MAX_DEV_KT:
            continue
        cleaned.append((lat, lon, kt, dir_deg, source))
    return cleaned


# "Full trust" distance for the composite. Was 1.5° (~165 km) which rated
# a cell in inland east Texas as HIGH confidence when the nearest actual
# obs was in the Houston metro 166 km away — clearly wrong for
# hurricane-scale wind fields where the mesoscale correlation length is
# 30-60 km. Tightened to 0.5° so anything past ~55 km starts fading, and
# the fade reaches zero at the 3° IDW radius.
_FULL_TRUST_DIST_DEG = 0.5


def _agreement(
    speeds: list[float],
    weights: list[float] | None = None,
) -> tuple[float, float | None]:
    """How tightly the observations that drive this cell agree.

    The spread is weighted by IDW weight and, when provided, only the
    nearby contributors. A hurricane gradient across 300 km is real
    structure, not a bad measurement — it used to zero the score even
    when a hunter fix sat 7 km away and dominated the blend.
    """
    if len(speeds) < 2:
        return 1.0, None
    if weights is None or len(weights) != len(speeds):
        weights = [1.0] * len(speeds)
    wsum = sum(weights)
    if wsum <= 0:
        return 0.0, None
    mean_s = sum(w * s for w, s in zip(weights, speeds)) / wsum
    var = sum(w * (s - mean_s) ** 2 for w, s in zip(weights, speeds)) / wsum
    std = var ** 0.5
    if std <= 5.0:
        score = 1.0
    elif std >= 20.0:
        score = 0.0
    else:
        score = max(0.0, 1.0 - (std - 5.0) / 15.0)
    return score, std


def _cell_confidence_parts(
    nearest_dist_deg: float | None,
    count: int,
    speeds: list[float],
    weights: list[float] | None = None,
) -> tuple[float, float, float, float, float | None]:
    """Return (composite, dist_score, count_score, agreement_score, std)."""
    if count <= 0 or nearest_dist_deg is None:
        return 0.0, 0.0, 0.0, 0.0, None
    if nearest_dist_deg <= _FULL_TRUST_DIST_DEG:
        dist_score = 1.0
    elif nearest_dist_deg >= IDW_RADIUS_DEG:
        dist_score = 0.0
    else:
        span = IDW_RADIUS_DEG - _FULL_TRUST_DIST_DEG
        dist_score = max(
            0.0, 1.0 - (nearest_dist_deg - _FULL_TRUST_DIST_DEG) / span,
        )
    if count == 1:
        count_score = 0.7
    else:
        count_score = 1.0
    agreement_score, std = _agreement(speeds, weights)
    composite = dist_score * count_score * agreement_score
    return (
        round(composite, 3),
        round(dist_score, 3),
        round(count_score, 3),
        round(agreement_score, 3),
        round(std, 1) if std is not None else None,
    )


def _cell_confidence(
    nearest_dist_deg: float | None,
    count: int,
    speeds: list[float],
    weights: list[float] | None = None,
) -> float:
    """Composite 0..1. See ``_cell_confidence_parts``."""
    composite, *_ = _cell_confidence_parts(
        nearest_dist_deg, count, speeds, weights,
    )
    return composite


def _bbox_key(
    west: float, south: float, east: float, north: float,
) -> tuple[float, float, float, float]:
    return (round(west, 2), round(south, 2), round(east, 2), round(north, 2))


def _nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat2 - lat1) * 60.0
    mean = math.radians((lat1 + lat2) / 2.0)
    dlon = (lon2 - lon1) * 60.0 * max(math.cos(mean), 0.05)
    return math.hypot(dlat, dlon)


def _center_at(
    centers: list[tuple[datetime, float, float]], when: datetime,
) -> tuple[float, float] | None:
    """Center lat/lon at ``when`` along a time-ordered track.

    Between fixes, linear in lat/lon (the track step is a few hours).
    After the last fix, continue the last segment for up to
    ``EXTRAP_MAX_H`` — that segment is the official forecast when the
    caller appended it, which is where the center is between advisories.
    """
    if not centers:
        return None
    if when <= centers[0][0]:
        if (centers[0][0] - when).total_seconds() <= 1800:
            return centers[0][1], centers[0][2]
        return None
    last_t, last_lat, last_lon = centers[-1]
    if when >= last_t:
        gap_h = (when - last_t).total_seconds() / 3600.0
        if gap_h <= 0.5:
            return last_lat, last_lon
        if gap_h > EXTRAP_MAX_H or len(centers) < 2:
            return None
        prev_t, prev_lat, prev_lon = centers[-2]
        dt_h = (last_t - prev_t).total_seconds() / 3600.0
        if dt_h < 0.25:
            return last_lat, last_lon
        return (
            last_lat + (last_lat - prev_lat) / dt_h * gap_h,
            last_lon + (last_lon - prev_lon) / dt_h * gap_h,
        )
    for i in range(1, len(centers)):
        t0, la0, lo0 = centers[i - 1]
        t1, la1, lo1 = centers[i]
        if t0 <= when <= t1:
            span = (t1 - t0).total_seconds()
            if span <= 0:
                return la1, lo1
            frac = (when - t0).total_seconds() / span
            return la0 + frac * (la1 - la0), lo0 + frac * (lo1 - lo0)
    return None


def advect_recon_obs(
    samples: list[tuple[float, float, float, float | None, str, str]],
    centers: list[tuple[str, float, float]],
    now: datetime,
) -> list[WindObs]:
    """Place hunter surface samples where that part of the vortex is now.

    ``samples`` is ``(lat, lon, wind_kt, wind_dir_deg, observed_at,
    station_id)`` at the aircraft's measured position. ``centers`` is
    ``(iso_time, lat, lon)`` for the storm center: the recent observed
    fixes (the live track's back-projection is NHC's motion vector) plus
    the latest official forecast, which says where the center is after
    the current fix.

    What this is: storm-relative persistence. A sample taken in the
    eyewall is slid by the same displacement as the center, so it stays
    in the eyewall instead of being left in the wake. The wind vector is
    not rotated — over a few hours the translation is the first-order
    correction, and a bad rotation is worse than none.

    What this is not: a forecast. Speeds stay the measured SFMR / 0.8×
    flight-level values. Samples farther than ``CORE_RADIUS_NM`` from
    the center at observation time are the environment (the ferry), not
    the vortex, and are not slid. Anything older than
    ``ADVECT_MAX_AGE_H`` is dropped. If the center can't be placed at
    both times, only a sample younger than ``FRESH_KEEP_H`` is kept, and
    it stays where it was measured.

    Weight is ``exp(-age / AGE_TAU_H)`` so a hours-old pass cannot
    outvote a buoy that just reported.
    """
    parsed: list[tuple[datetime, float, float]] = []
    for iso, lat, lon in centers:
        dt = _parse_iso(iso)
        if dt is None:
            continue
        parsed.append((dt, lat, lon))
    parsed.sort(key=lambda p: p[0])

    out: list[WindObs] = []
    for lat, lon, kt, dir_deg, iso, sid in samples:
        dt = _parse_iso(iso)
        if dt is None:
            continue
        age_h = (now - dt).total_seconds() / 3600.0
        if age_h < -0.25 or age_h > ADVECT_MAX_AGE_H:
            continue
        if age_h < 0:
            age_h = 0.0
        weight = math.exp(-age_h / AGE_TAU_H)
        c_then = _center_at(parsed, dt)
        c_now = _center_at(parsed, now)
        use_lat, use_lon = lat, lon
        if c_then is not None and c_now is not None:
            if _nm(lat, lon, c_then[0], c_then[1]) <= CORE_RADIUS_NM:
                use_lat = lat + (c_now[0] - c_then[0])
                use_lon = lon + (c_now[1] - c_then[1])
            elif age_h > FRESH_KEEP_H:
                continue
        elif age_h > FRESH_KEEP_H:
            continue
        out.append(
            WindObs(
                lat=round(use_lat, 4),
                lon=round(use_lon, 4),
                wind_kt=round(kt, 1),
                wind_dir_deg=dir_deg,
                source="recon",
                station_id=sid,
                observed_at=iso,
                weight=round(weight, 3),
            )
        )
    return out


def interpolate_obs(
    west: float, south: float, east: float, north: float,
    obs: list[WindObs],
    step: float = FIXED_STEP_DEG,
) -> list[WindGridCell]:
    """IDW grid from an already-cleaned observation list.

    ``WindObs.weight`` scales the sample (recon age decay). Cells with
    nothing inside the source's radius are omitted.
    """
    if not obs:
        return []
    precomputed: list[tuple[float, float, float, float | None, float | None, float, float]] = []
    for o in obs:
        u: float | None = None
        v: float | None = None
        if o.wind_dir_deg is not None and o.wind_kt > 0:
            r = math.radians(o.wind_dir_deg)
            u = -o.wind_kt * math.sin(r)
            v = -o.wind_kt * math.cos(r)
        radius = IDW_RADIUS_RECON_DEG if o.source == "recon" else IDW_RADIUS_DEG
        precomputed.append((
            o.lat, o.lon, o.wind_kt, u, v, radius * radius, max(o.weight, 0.0),
        ))

    cells: list[WindGridCell] = []
    lat = south
    while lat <= north + 1e-9:
        cos_lat = max(math.cos(math.radians(lat)), 0.05)
        lon = west
        while lon <= east + 1e-9:
            weight_sum = 0.0
            speed_sum = 0.0
            u_sum = 0.0
            v_sum = 0.0
            uv_weight_sum = 0.0
            count = 0
            nearest_d2: float | None = None
            # Agreement uses the local neighborhood only. The rest of the
            # storm is allowed to differ. If nothing is nearby, fall back
            # to the weighted spread of whoever actually entered the blend.
            local_speeds: list[float] = []
            local_weights: list[float] = []
            all_speeds: list[float] = []
            all_weights: list[float] = []
            local_r2 = _FULL_TRUST_DIST_DEG * _FULL_TRUST_DIST_DEG
            for la, lo, kt, u, v, obs_r_sq, ow in precomputed:
                if ow <= 0:
                    continue
                dlat = la - lat
                dlon = (lo - lon) * cos_lat
                d2 = dlat * dlat + dlon * dlon
                if d2 > obs_r_sq:
                    continue
                w = ow / ((d2 + 0.01) ** (IDW_POWER / 2))
                weight_sum += w
                speed_sum += w * kt
                if u is not None and v is not None:
                    u_sum += w * u
                    v_sum += w * v
                    uv_weight_sum += w
                count += 1
                all_speeds.append(kt)
                all_weights.append(w)
                if d2 <= local_r2:
                    local_speeds.append(kt)
                    local_weights.append(w)
                if nearest_d2 is None or d2 < nearest_d2:
                    nearest_d2 = d2
            if count == 0 or weight_sum <= 0:
                lon += step
                continue
            wind_kt = round(speed_sum / weight_sum, 1)
            wind_dir_deg: float | None = None
            if uv_weight_sum > 0:
                u_mean = u_sum / uv_weight_sum
                v_mean = v_sum / uv_weight_sum
                to_deg = math.degrees(math.atan2(u_mean, v_mean))
                wind_dir_deg = round((to_deg + 180.0) % 360.0, 1)
            nearest_dist = (nearest_d2 ** 0.5) if nearest_d2 is not None else None
            nearest_km = (
                round(nearest_dist * 111.0, 0) if nearest_dist is not None else None
            )
            agree_speeds = local_speeds if local_speeds else all_speeds
            agree_weights = local_weights if local_speeds else all_weights
            composite, dist_s, count_s, agree_s, std = _cell_confidence_parts(
                nearest_dist, count, agree_speeds, agree_weights,
            )
            cells.append(
                WindGridCell(
                    lat=round(lat, 3),
                    lon=round(lon, 3),
                    wind_kt=wind_kt,
                    wind_dir_deg=wind_dir_deg,
                    sources=count,
                    confidence=composite,
                    nearest_obs_km=nearest_km,
                    dist_score=dist_s,
                    count_score=count_s,
                    agreement_score=agree_s,
                    contributor_spread_kt=std,
                )
            )
            lon += step
        lat += step
    return cells


def remember_surface_obs(
    west: float, south: float, east: float, north: float,
    obs: list[WindObs],
) -> None:
    """Keep the buoy/land pool so a later recon poll can re-grid."""
    _SURFACE_CACHE.set(
        _bbox_key(west, south, east, north),
        [o for o in obs if o.source != "recon"],
    )


def regrid_with_recon(
    west: float, south: float, east: float, north: float,
    recon_obs: list[WindObs],
) -> tuple[list[WindGridCell], float, list[WindObs]] | None:
    """Blend new hunter samples onto the cached buoy/land field.

    Returns None when there is no cached surface field — the caller
    should leave the heatmap it already drew rather than replace it
    with a hunter-only grid. Rewrites the cache so an open storm does
    not expire mid-session.
    """
    key = _bbox_key(west, south, east, north)
    base = _SURFACE_CACHE.get(key)
    if base is None:
        return None
    _SURFACE_CACHE.set(key, base)
    merged = list(base) + list(recon_obs)
    cells = interpolate_obs(west, south, east, north, merged)
    return cells, FIXED_STEP_DEG, merged


def wind_field_grid(
    west: float, south: float, east: float, north: float,
    *, now: datetime | None = None,
    extra_obs: list[WindObs] | None = None,
) -> tuple[list[WindGridCell], float, list[WindObs]]:
    """Return (grid_cells, cell_step_deg, cleaned_obs_pool). Cells with no
    obs in range are omitted — the renderer paints those as gaps rather
    than fake data. The cleaned obs pool is shipped alongside so the
    frontend can highlight which stations contributed to any given cell.

    Direction is derived from a parallel IDW on the u/v vector components
    (rather than a scalar mean on the angle). Averaging angles directly
    breaks at the 0/360 wraparound; vector-mean handles convergent and
    divergent flow correctly."""
    if now is None:
        now = datetime.now(timezone.utc)

    # Keep the raw obs metadata (station id + source label) so the shipped
    # obs pool can be attributed on the frontend.
    obs_with_meta: list[
        tuple[float, float, float, float | None, str, str, str]
    ] = []  # lat, lon, kt, dir, iso, source, station_id
    for b in buoys_in_bbox(west, south, east, north):
        if b.wind_kt is None:
            continue
        obs_with_meta.append((
            b.lat, b.lon, float(b.wind_kt),
            float(b.wind_dir_deg) if b.wind_dir_deg is not None else None,
            b.observed_at, "buoy", b.station_id,
        ))
    obs_with_meta.extend(
        (lat, lon, kt, d, iso, "land", sid)
        for (lat, lon, kt, d, iso, sid) in _fetch_land_obs_with_meta(
            west, south, east, north,
        )
    )
    if extra_obs:
        for o in extra_obs:
            obs_with_meta.append((
                o.lat, o.lon, float(o.wind_kt), o.wind_dir_deg,
                o.observed_at, o.source, o.station_id,
            ))

    obs = [
        (la, lo, kt, d, iso, src)
        for (la, lo, kt, d, iso, src, _sid) in obs_with_meta
    ]
    step = FIXED_STEP_DEG
    cleaned = _clean_obs(obs, now)
    if not cleaned:
        remember_surface_obs(west, south, east, north, [])
        return [], step, []

    # Attribute each cleaned tuple back to its source station. Cleaning
    # preserves (lat, lon, kt) — dir may be dropped/kept unchanged. Match on
    # (lat, lon, kt) which is unique enough for our purposes.
    meta_by_key: dict[tuple[float, float, float], tuple[str, str, str]] = {}
    weight_by_key: dict[tuple[float, float, float], float] = {}
    for lat, lon, kt, _d, iso, source, sid in obs_with_meta:
        meta_by_key[(lat, lon, kt)] = (source, sid, iso)
    if extra_obs:
        for o in extra_obs:
            weight_by_key[(o.lat, o.lon, float(o.wind_kt))] = o.weight
    obs_pool: list[WindObs] = []
    for lat, lon, kt, d, source in cleaned:
        meta = meta_by_key.get((lat, lon, kt))
        src, sid, iso = meta if meta else (source, "", "")
        obs_pool.append(
            WindObs(
                lat=round(lat, 4),
                lon=round(lon, 4),
                wind_kt=round(kt, 1),
                wind_dir_deg=d,
                source=src,
                station_id=sid,
                observed_at=iso,
                weight=weight_by_key.get((lat, lon, kt), 1.0),
            )
        )

    remember_surface_obs(west, south, east, north, obs_pool)
    cells = interpolate_obs(west, south, east, north, obs_pool, step)
    return cells, step, obs_pool


__all__ = [
    "WindGridCell",
    "WindObs",
    "advect_recon_obs",
    "interpolate_obs",
    "regrid_with_recon",
    "remember_surface_obs",
    "wind_field_grid",
]
