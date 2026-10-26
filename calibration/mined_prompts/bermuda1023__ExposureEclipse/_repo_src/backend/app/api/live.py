"""Live + replay hurricane endpoint.

GET  /api/live/storms                 — picker rows: active NHC storms + replay candidates
GET  /api/live/storms/{atcf_id}       — full bundle for one storm:
                                          observed track, forecast (latest + history),
                                          alerts in cone, buoys + land stations in cone,
                                          SST grid covering the bbox

Replay mode (default for retired storms): synthesises prior-advisory tracks
from the IBTrACS truth. Live mode: just the NHC current summary — full
text-advisory scraping is out of scope for v1.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from ..models.common import CamelModel
from ..services.hurdat2 import category_for_wind
from ..services.live_hurricane import (
    LiveStormSummary,
    build_wind_cones,
    clear_current_storms_cache,
    fetch_active_summaries,
    fetch_live_forecast_cone,
    fetch_live_peak_surge,
    fetch_live_wind_radii,
    storm_and_forecasts,
    synthetic_forecast_cone,
)
from ..services.atcf_adecks import (
    FAMILY_ORDER,
    MODEL_LABEL,
    ModelTrack,
    fetch_model_tracks,
    fetch_official_fixes,
    list_available_cycles,
)
from ..services.ensemble_envelope import build_envelope
from ..services.jma_typhoon import (
    JmaStorm,
    JmaSummary,
    fetch_active_typhoons,
    fetch_typhoon,
    is_jma_id,
)
from ..services.nhc_gtwo import fetch_gtwo
from ..services.ensemble_risk import (
    ATLANTIC_COASTAL_STATES,
    DEFAULT_STRIKE_THRESHOLD_NM,
    compute_ensemble_risk,
)
from ..services.marine_obs import buoys_in_bbox, land_stations_in_bbox
from ..services.nhc_watch_warn import split_watches_warnings
from ..services.sea_surface_temp import sst_field
from ..services.weather_alerts import AlertFeedUnavailable, fetch_active_alerts
from ..services.wind_field_map import (
    WindObs,
    advect_recon_obs,
    interpolate_obs,
    regrid_with_recon,
    wind_field_grid,
)
from ..services.recon_obs import fetch_recon_bundle, recon_for_idw
from ..services.wind_forecast import (
    daily_forecast,
    fetch_model_shear_grid,
    fetch_model_wind_grid,
    point_forecast,
)
from ..services import wildfire_exposure
from .geometry_input import (
    PolygonExposureOut,
    WatchWarnExposureRequest,
    exposure_out,
)

router = APIRouter(prefix="/live", tags=["live"])


# ─────────────────────────── wire types ───────────────────────────


class LiveStormRow(CamelModel):
    storm_id: str
    name: str
    year: int
    classification: str
    intensity_kt: int
    pressure_mb: int | None
    lat: float | None
    lon: float | None
    is_live: bool
    label: str


class LiveStormListResponse(CamelModel):
    active: list[LiveStormRow]
    replay: list[LiveStormRow]
    # Official JMA typhoons (western North Pacific). A separate list so a
    # Japan track is not mistaken for an NHC advisory: no cone, no a-deck,
    # no US county strike vote.
    typhoons: list[LiveStormRow] = []
    has_active: bool
    note: str | None = None


class ObservedFix(CamelModel):
    lat: float
    lon: float
    wind_kt: int
    category: int
    status: str
    datetime: str


class ForecastFix(CamelModel):
    lat: float
    lon: float
    wind_kt: int
    hours_out: int
    valid_time: str


class ForecastAdvisory(CamelModel):
    advisory_number: int
    issued_at: str
    points: list[ForecastFix]
    synthetic: bool


class WeatherAlertOut(CamelModel):
    alert_id: str
    event: str
    headline: str
    severity: str
    urgency: str
    certainty: str
    sent_at: str
    expires_at: str
    areas_affected: str
    geometry: dict | None


class NHCWatchWarnOut(CamelModel):
    """One NHC-issued coastal Tropical Cyclone watch or warning polygon
    (Hurricane / Tropical Storm / Storm Surge × Watch/Warning + Extreme Wind
    Warning). ``geometry`` is null for zone-coded alerts that ship without a
    polygon — surfaced in the count and text but not rendered as a shape."""

    alert_id: str
    event: str
    family: str            # hurricane | tropical_storm | storm_surge | extreme_wind | statement
    color: str             # NHC operational hex — feed straight to the map paint
    rank: int              # higher = more severe (drives map z-order)
    headline: str
    severity: str
    urgency: str
    certainty: str
    sent_at: str
    expires_at: str
    areas_affected: str
    geometry: dict | None


class BuoyOut(CamelModel):
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
    observed_at: str


class LandObsOut(CamelModel):
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


class SSTOut(CamelModel):
    lat: float
    lon: float
    temp_c: float
    favorable_for_intensification: bool


class SSTMeta(CamelModel):
    source: str        # 'mur' | 'synthetic'
    step_deg: float    # native cell size — frontend uses this to size polygons


class ConeQuadOut(CamelModel):
    corners: list[list[float]]   # closed ring [[lon,lat], ...]
    wind_kt: int
    start_wind_kt: int
    end_wind_kt: int


class OuterRingOut(CamelModel):
    corners: list[list[float]]
    wind_kt: int
    r64_nm: float
    r64_source: str


class WindFieldOut(CamelModel):
    inner_cone: list[ConeQuadOut]
    outer_cone: list[ConeQuadOut]
    outer_rings: list[OuterRingOut]


class ForecastConeOut(CamelModel):
    """NHC's official forecast cone of uncertainty. Live storms only — empty
    ring on retired/replay storms."""

    ring: list[list[float]]   # [[lon, lat], ...] closed outer boundary


class SurgePolygonOut(CamelModel):
    """One NHC peak-storm-surge band polygon (e.g. '3-6 ft' inundation)."""

    ring: list[list[float]]
    surge_range: str
    color: str


class WindGridPointOut(CamelModel):
    """One cell of the interpolated surface-wind field.

    ``wind_dir_deg`` uses meteorological FROM convention (0°=N, 90°=E).
    ``sources`` counts contributing observations. ``confidence`` is a 0..1
    composite of source count, distance to nearest obs, and speed agreement
    across contributors. ``nearest_obs_km`` is the raw distance to the
    closest contributing obs — surfaced in the click popup so the user can
    see the underlying signal directly."""

    lat: float
    lon: float
    wind_kt: float
    wind_dir_deg: float | None
    sources: int
    confidence: float
    nearest_obs_km: float | None
    # Score breakdown (all 0..1). Multiply to get confidence.
    dist_score: float
    count_score: float
    agreement_score: float
    contributor_spread_kt: float | None


class WindGridMeta(CamelModel):
    step_deg: float
    obs_max_age_hours: float
    idw_radius_km: float


class ReconObsOut(CamelModel):
    """One hurricane-hunter HDOB point (SFMR surface wind, or 0.8×FL fallback)."""

    lat: float
    lon: float
    observed_at: str
    surface_kt: float
    surface_source: str          # "sfmr" | "fl80"
    fl_wind_kt: float | None
    fl_dir_deg: float | None
    sfmr_kt: float | None
    rain_mm_hr: float | None
    aircraft: str
    storm_name: str
    mission_id: str


class VortexFixOut(CamelModel):
    """Latest vortex data message center fix, when a mission is in the storm."""

    lat: float
    lon: float
    observed_at: str
    pressure_mb: float | None
    max_fl_wind_kt: float | None
    aircraft: str
    storm_id: str
    storm_name: str
    mission_id: str


class WindObsOut(CamelModel):
    """One cleaned surface observation used to build the wind heatmap.
    Shipped in the bundle so the frontend can drill down from a cell click
    ('N sources') to the actual contributing stations."""

    lat: float
    lon: float
    wind_kt: float
    wind_dir_deg: float | None
    source: str          # "buoy" | "land" | "recon"
    station_id: str
    observed_at: str


class WindGridCoordOut(CamelModel):
    """Just a lat/lon pair — one entry per grid cell. Parallel-array to
    every ``WindModelFrameOut.wind_kt``/``wind_dir_deg`` so we don't
    repeat coordinates at every forecast frame."""

    lat: float
    lon: float


class WindModelFrameOut(CamelModel):
    """One forecast time-step for the whole grid. ``wind_kt`` and
    ``wind_dir_deg`` are index-aligned with the top-level ``cells`` list."""

    hour: int              # forecast hours from "now" (0, 6, 12, …)
    valid_time_utc: str
    wind_kt: list[float | None]
    wind_dir_deg: list[float | None]


class WindModelGridOut(CamelModel):
    """Multi-frame GFS/ECMWF wind grid. The frontend picks a frame index
    based on the time-slider position; all frames share the same cell
    coordinates so switching frames is O(1)."""

    model: str            # "gfs" | "ecmwf"
    step_deg: float
    cells: list[WindGridCoordOut]
    frames: list[WindModelFrameOut]
    # Empty because the per-minute location budget was already spent.
    # The panel waits and retries; a genuinely calm or uncovered grid
    # leaves this false.
    rate_limited: bool = False


class ModelForecastOut(CamelModel):
    """One NWP model's wind forecast at a single point + time."""

    model: str            # "gfs" | "ecmwf"
    valid_time_utc: str
    wind_kt: float
    wind_dir_deg: float | None
    wind_gust_kt: float | None


class PointForecastOut(CamelModel):
    """Wind at a lat/lon, per model. Empty ``forecasts`` when Open-Meteo is
    unreachable; the click UI shows 'model data unavailable' in that case."""

    lat: float
    lon: float
    fetched_at_utc: str
    forecasts: list[ModelForecastOut]


class DailyDayOut(CamelModel):
    """One local calendar day. Wind is the daily maximum, not the hour
    the click popup shows."""

    date: str
    weather_code: int | None
    temp_max_f: float | None
    temp_min_f: float | None
    precip_in: float | None
    wind_max_kt: float | None
    gust_max_kt: float | None
    wind_dir_deg: float | None


class DailyModelOut(CamelModel):
    model: str            # "gfs" | "ecmwf"
    days: list[DailyDayOut]


class DailyForecastOut(CamelModel):
    """Five local days at one point, GFS and ECMWF. ``models`` is empty
    when Open-Meteo is unreachable."""

    lat: float
    lon: float
    timezone: str | None
    fetched_at_utc: str
    models: list[DailyModelOut]


class JmaFixOut(CamelModel):
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


class JmaCircleOut(CamelModel):
    """Closed ring. ``kind`` is gale, storm, or probability."""

    kind: str
    hours_out: int
    label: str
    center_lon: float
    center_lat: float
    radius_m: float
    ring: list[list[float]]


class JmaPathOut(CamelModel):
    """Open line. ``kind`` is tangent, storm-arc, or storm-line."""

    kind: str
    hours_out: int
    coordinates: list[list[float]]


class JmaOverlayOut(CamelModel):
    """Official JMA bulletin drawn in place of the NHC cone."""

    name_en: str
    name_jp: str
    typhoon_number: str
    issued_at: str
    category_en: str
    category_jp: str
    intensity: str | None
    wind_ms: float | None
    wind_kt: int | None
    gust_ms: float | None
    gust_kt: int | None
    pressure_mb: int | None
    gale_ranges: list[str]
    storm_ranges: list[str]
    note: str
    fixes: list[JmaFixOut]
    circles: list[JmaCircleOut]
    paths: list[JmaPathOut]


class LiveStormBundle(CamelModel):
    storm: LiveStormRow
    observed_track: list[ObservedFix]
    forecasts: list[ForecastAdvisory]      # latest first
    bbox: list[float]                      # [west, south, east, north]
    alerts: list[WeatherAlertOut]
    # NHC-issued Tropical Cyclone watches/warnings, split out of the generic
    # alerts stream. Same underlying source (NWS CAP feed) but with the NHC
    # operational colour scheme + rank so they render distinctly on the map.
    watches_warnings: list[NHCWatchWarnOut]
    watches_warnings_zone_only: int        # count of zone-coded (no polygon) WWs
    buoys: list[BuoyOut]
    land_stations: list[LandObsOut]
    # Hurricane hunter HDOB (SFMR / flight-level) + latest vortex fix.
    # Empty when no aircraft is in the storm (typical for replay / open-ocean
    # gaps between missions).
    recon: list[ReconObsOut]
    vortex: VortexFixOut | None
    sst: list[SSTOut]
    sst_min_c: float | None
    sst_max_c: float | None
    sst_meta: SSTMeta
    # Wind fields built from the same IBTrACS-driven Rmax + R64 quads we use
    # for historical impact. `observed` covers the track to date; `forecast`
    # is the latest advisory's projected track.
    observed_wind_field: WindFieldOut
    forecast_wind_field: WindFieldOut
    # NHC-issued products, populated only for live storms. `forecast_cone` is
    # the swept-circle envelope from NHC's cone KMZ; `peak_surge` is the
    # per-band coastal inundation polygons from NHC's peak-surge KML.
    forecast_cone: ForecastConeOut | None
    peak_surge: list[SurgePolygonOut]
    # Interpolated surface-wind heatmap (IDW over NDBC buoys + NWS land obs).
    # Empty when include_wind_map is off or no fresh obs are in the bbox.
    wind_map: list[WindGridPointOut]
    wind_map_meta: WindGridMeta
    # The cleaned obs pool that fed the heatmap. Shipped so the click popup
    # can show contributor stations for any given cell.
    wind_obs: list[WindObsOut]
    # Set only for a JMA id. Null on every NHC storm.
    jma: JmaOverlayOut | None = None


# ─────────────────────────── helpers ───────────────────────────


def _jma_row(s: JmaSummary) -> LiveStormRow:
    return LiveStormRow(
        storm_id=s.storm_id,
        name=s.name,
        year=s.year,
        classification=s.classification,
        intensity_kt=s.intensity_kt,
        pressure_mb=s.pressure_mb,
        lat=s.lat,
        lon=s.lon,
        is_live=True,
        label=s.label,
    )


def _bundle_from_jma(storm: JmaStorm) -> LiveStormBundle:
    """Track, circles, and radii only. No NHC products and no US impact."""
    empty_wind = WindFieldOut(inner_cone=[], outer_cone=[], outer_rings=[])
    observed = [
        ObservedFix(
            lat=lat,
            lon=lon,
            wind_kt=storm.summary.intensity_kt if i == len(storm.observed) - 1 else 0,
            category=0,
            status=storm.summary.classification,
            datetime=storm.forecast[0].valid_time if storm.forecast else storm.issued_at,
        )
        for i, (lat, lon) in enumerate(storm.observed)
    ]
    points = [
        ForecastFix(
            lat=fix.lat,
            lon=fix.lon,
            wind_kt=fix.wind_kt or 0,
            hours_out=fix.hours_out,
            valid_time=fix.valid_time,
        )
        for fix in storm.forecast
    ]
    advisory = ForecastAdvisory(
        advisory_number=1,
        issued_at=storm.issued_at,
        points=points,
        synthetic=False,
    ) if points else None
    return LiveStormBundle(
        storm=_jma_row(storm.summary),
        observed_track=observed,
        forecasts=[advisory] if advisory else [],
        bbox=list(storm.bbox),
        alerts=[],
        watches_warnings=[],
        watches_warnings_zone_only=0,
        buoys=[],
        land_stations=[],
        recon=[],
        vortex=None,
        sst=[],
        sst_min_c=None,
        sst_max_c=None,
        sst_meta=SSTMeta(source="none", step_deg=0.0),
        observed_wind_field=empty_wind,
        forecast_wind_field=empty_wind,
        forecast_cone=None,
        peak_surge=[],
        wind_map=[],
        wind_map_meta=WindGridMeta(
            step_deg=0.0, obs_max_age_hours=0.0, idw_radius_km=0.0,
        ),
        wind_obs=[],
        jma=JmaOverlayOut(
            name_en=storm.summary.name,
            name_jp=storm.summary.name_jp,
            typhoon_number=storm.typhoon_number,
            issued_at=storm.issued_at,
            category_en=storm.summary.classification,
            category_jp=storm.category_jp,
            intensity=storm.intensity,
            wind_ms=storm.wind_ms,
            wind_kt=storm.summary.intensity_kt or None,
            gust_ms=storm.gust_ms,
            gust_kt=storm.gust_kt,
            pressure_mb=storm.summary.pressure_mb,
            gale_ranges=storm.gale_ranges,
            storm_ranges=storm.storm_ranges,
            note=storm.note,
            fixes=[
                JmaFixOut(
                    hours_out=fix.hours_out,
                    valid_time=fix.valid_time,
                    lat=fix.lat,
                    lon=fix.lon,
                    wind_kt=fix.wind_kt,
                    wind_ms=fix.wind_ms,
                    pressure_mb=fix.pressure_mb,
                    category_en=fix.category_en,
                    category_jp=fix.category_jp,
                    intensity=fix.intensity,
                )
                for fix in storm.forecast
            ],
            circles=[
                JmaCircleOut(
                    kind=c.kind,
                    hours_out=c.hours_out,
                    label=c.label,
                    center_lon=c.center_lon,
                    center_lat=c.center_lat,
                    radius_m=c.radius_m,
                    ring=c.ring,
                )
                for c in storm.circles
            ],
            paths=[
                JmaPathOut(kind=p.kind, hours_out=p.hours_out, coordinates=p.coordinates)
                for p in storm.paths
            ],
        ),
    )


def _summary_to_row(s: LiveStormSummary) -> LiveStormRow:
    return LiveStormRow(
        storm_id=s.storm_id,
        name=s.name,
        year=s.year,
        classification=s.classification,
        intensity_kt=s.intensity_kt,
        pressure_mb=s.pressure_mb,
        lat=s.lat,
        lon=s.lon,
        is_live=s.is_live,
        label=s.label,
    )


def _bbox_for_storm(observed_track, forecasts) -> tuple[float, float, float, float]:
    """Bbox for fetching nearby live data — last day of observed + all forecast.

    Using the storm's entire historical path would pull most of the Atlantic
    for a long-lived hurricane like Michael; we only need the bbox where
    overlays (alerts, buoys, SST) matter operationally.
    """
    recent = observed_track[-8:] if len(observed_track) >= 8 else observed_track
    lats: list[float] = [p.lat for p in recent]
    lons: list[float] = [p.lon for p in recent]
    for adv in forecasts:
        lats.extend(p.lat for p in adv.points)
        lons.extend(p.lon for p in adv.points)
    if not lats:
        return (-100.0, 10.0, -50.0, 50.0)
    # Pad ~3° so the cone of uncertainty + observation buffer fits.
    west = min(lons) - 3.0
    east = max(lons) + 3.0
    south = min(lats) - 3.0
    north = max(lats) + 3.0
    return (west, south, east, north)


def _gather(
    jobs: dict[str, tuple],
    timeout_s: float,
) -> dict:
    """Run callables together and stop waiting at one shared deadline.

    A hung SST or recon feed must not add its timeout on top of the cone
    fetch. Whatever is still running when the deadline passes comes back
    as the job's default. ``jobs`` values are ``(fn, default)``.
    """
    if not jobs:
        return {}
    ex = ThreadPoolExecutor(max_workers=min(8, len(jobs)))
    futs = {name: ex.submit(fn) for name, (fn, _default) in jobs.items()}
    out: dict = {}
    deadline = time.monotonic() + timeout_s
    try:
        for name, (_fn, default) in jobs.items():
            fut = futs[name]
            remaining = deadline - time.monotonic()
            # A slow earlier job must not discard a feed that already
            # finished. Waiting on the a-deck first used to drop a cone
            # KMZ that had returned seconds earlier.
            if fut.done():
                try:
                    out[name] = fut.result()
                except Exception:  # noqa: BLE001
                    out[name] = default
                continue
            if remaining <= 0:
                out[name] = default
                continue
            try:
                out[name] = fut.result(timeout=remaining)
            except Exception:  # noqa: BLE001
                out[name] = default
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return out


def _center_track(observed_storm, forecasts) -> list[tuple[str, float, float]]:
    """Center positions the advection uses: recent fixes, then the forecast."""
    pts: list[tuple[str, float, float]] = []
    for p in observed_storm.track:
        if p.datetime_utc:
            pts.append((p.datetime_utc, p.lat, p.lon))
    if forecasts:
        latest = max(forecasts, key=lambda f: f.advisory_number)
        for fp in latest.points:
            if fp.valid_time:
                pts.append((fp.valid_time, fp.lat, fp.lon))
    return pts


def _advect_recon_samples(fixes, observed_storm, forecasts) -> list[WindObs]:
    samples = [
        (
            fx.lat, fx.lon, fx.surface_kt, fx.fl_dir_deg, fx.observed_at,
            f"{fx.aircraft}-{fx.observed_at[11:16]} adv",
        )
        for fx in recon_for_idw(fixes)
    ]
    return advect_recon_obs(
        samples, _center_track(observed_storm, forecasts),
        datetime.now(timezone.utc),
    )


def _parse_centers(raw: str) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for part in (raw or "").split("|"):
        bits = part.split(",")
        if len(bits) != 3:
            continue
        try:
            out.append((bits[0], float(bits[1]), float(bits[2])))
        except ValueError:
            continue
    return out


def _states_in_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    """Rough state filter for NWS alerts: returns the USPS codes whose
    bounding boxes overlap ``bbox``. Used to narrow the alerts request.
    Coarse — better to over-request than miss an alert."""
    # Very coarse state bboxes (lon_min, lat_min, lon_max, lat_max). Only
    # hurricane-prone states; everything else falls through to "all".
    STATE_BBOXES = {
        "FL": (-87.6, 24.5, -80.0, 31.0),
        "GA": (-85.6, 30.4, -80.8, 35.0),
        "SC": (-83.4, 32.0, -78.5, 35.2),
        "NC": (-84.4, 33.8, -75.4, 36.6),
        "VA": (-83.7, 36.5, -75.2, 39.5),
        "AL": (-88.5, 30.2, -84.9, 35.0),
        "MS": (-91.7, 30.2, -88.1, 35.0),
        "LA": (-94.0, 28.9, -89.0, 33.0),
        "TX": (-106.6, 25.8, -93.5, 36.5),
        "NY": (-79.8, 40.5, -71.9, 45.0),
        "NJ": (-75.6, 38.9, -73.9, 41.4),
        "MA": (-73.5, 41.2, -69.9, 42.9),
        "PR": (-67.3, 17.9, -65.2, 18.5),
    }
    west, south, east, north = bbox
    states: list[str] = []
    for code, (w, s, e, n) in STATE_BBOXES.items():
        if not (e < west or w > east or n < south or s > north):
            states.append(code)
    return states


# ─────────────────────────── endpoints ───────────────────────────


@router.get("/storms", response_model=LiveStormListResponse)
def list_live_storms() -> LiveStormListResponse:
    """Active NHC storms and JMA typhoons. Replay is not offered."""
    active = [_summary_to_row(s) for s in fetch_active_summaries()]
    try:
        typhoons = [_jma_row(s) for s in fetch_active_typhoons()]
    except Exception:  # noqa: BLE001 — JMA outage must not blank the NHC list
        typhoons = []
    note = None
    if not active and not typhoons:
        note = "No active Atlantic storms or JMA typhoons right now."
    elif not active:
        note = "No active Atlantic storms. JMA typhoons are listed separately."
    return LiveStormListResponse(
        active=active,
        replay=[],
        typhoons=typhoons,
        has_active=bool(active),
        note=note,
    )


@router.get("/storms/{atcf_id}", response_model=LiveStormBundle)
def live_storm_bundle(
    atcf_id: str,
    as_of_index: int | None = Query(default=None, ge=0, alias="asOfIndex"),
    include_obs: bool = Query(default=True, alias="includeObs"),
    include_alerts: bool = Query(default=True, alias="includeAlerts"),
    include_sst: bool = Query(default=True, alias="includeSst"),
    include_land: bool = Query(default=True, alias="includeLand"),
    include_surge: bool = Query(default=True, alias="includeSurge"),
    include_wind_map: bool = Query(default=True, alias="includeWindMap"),
    refresh: bool = Query(default=False, alias="refresh"),
) -> LiveStormBundle:
    """Full data bundle for one storm — track + forecast + obs + alerts + SST.

    Ancillary feeds (alerts, buoys, land, SST, recon, wind map) run together
    under one deadline so a slow one cannot drop the cone and wind field.
    ``refresh`` drops the short-lived NHC caches and fetches again.
    A ``TC*`` id is a JMA typhoon and never touches NHC products.
    """
    if is_jma_id(atcf_id):
        storm = fetch_typhoon(atcf_id, refresh=refresh)
        if storm is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DATASET_NOT_FOUND",
                    "message": (
                        f"Typhoon '{atcf_id}' is not in the current JMA list. "
                        "An empty list means JMA is not issuing a typhoon."
                    ),
                },
            )
        return _bundle_from_jma(storm)
    if refresh:
        clear_current_storms_cache()
    result = storm_and_forecasts(
        atcf_id, as_of_index=as_of_index, refresh=refresh,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DATASET_NOT_FOUND",
                "message": (
                    f"Storm '{atcf_id}' not found in the live NHC feed or the "
                    "replay set."
                ),
            },
        )
    observed_storm, forecasts, is_live = result

    observed_fixes = [
        ObservedFix(
            lat=p.lat,
            lon=p.lon,
            wind_kt=p.wind_kt,
            category=category_for_wind(p.wind_kt),
            status=p.status,
            datetime=p.datetime_utc,
        )
        for p in observed_storm.track
    ]
    forecast_out = [
        ForecastAdvisory(
            advisory_number=adv.advisory_number,
            issued_at=adv.issued_at,
            points=[
                ForecastFix(
                    lat=fp.lat,
                    lon=fp.lon,
                    wind_kt=fp.wind_kt,
                    hours_out=fp.hours_out,
                    valid_time=fp.valid_time,
                )
                for fp in adv.points
            ],
            synthetic=adv.synthetic,
        )
        for adv in forecasts
    ]

    bbox = _bbox_for_storm(observed_storm.track, forecasts)

    def _job_alerts():
        states = _states_in_bbox(bbox)
        try:
            return fetch_active_alerts(bbox=bbox, states=states or None)
        except AlertFeedUnavailable:
            return []

    def _job_buoys():
        return list(buoys_in_bbox(*bbox))

    def _job_land():
        return list(land_stations_in_bbox(*bbox))

    def _job_sst():
        return sst_field(bbox)

    def _job_official():
        if not is_live:
            return []
        return fetch_official_fixes(atcf_id, refresh=refresh)

    def _job_cone():
        return fetch_live_forecast_cone(atcf_id, refresh=refresh)

    def _job_radii():
        return fetch_live_wind_radii(atcf_id, refresh=refresh)

    def _job_surge():
        return fetch_live_peak_surge(atcf_id, refresh=refresh)

    def _job_recon():
        return fetch_recon_bundle(
            bbox, atcf_id=atcf_id, storm_name=observed_storm.name,
        )

    def _job_wind():
        return wind_field_grid(*bbox)

    # Cone and wind radii go in first so they start immediately and the
    # shared deadline is spent on them, not on a hung a-deck or SST call.
    # Each product is its own job: a slow surge KML must not throw away
    # a cone that already parsed.
    jobs: dict[str, tuple] = {}
    if is_live:
        jobs["cone"] = (_job_cone, [])
        jobs["radii"] = (_job_radii, ([], []))
        if include_surge:
            jobs["surge"] = (_job_surge, [])
        jobs["recon"] = (_job_recon, None)
    jobs["official"] = (_job_official, [])
    if include_alerts:
        jobs["alerts"] = (_job_alerts, [])
    if include_obs:
        jobs["buoys"] = (_job_buoys, [])
    if include_land:
        jobs["land"] = (_job_land, [])
    if include_sst:
        jobs["sst"] = (_job_sst, ([], "synthetic"))
    if include_wind_map:
        jobs["wind"] = (_job_wind, ([], 0.5, []))
    # One deadline for every feed. The cone and wind swath are in this set,
    # so they come back even when land stations or the recon archive stall.
    got = _gather(jobs, 14.0)

    alerts_out: list[WeatherAlertOut] = []
    watches_warnings_out: list[NHCWatchWarnOut] = []
    zone_only_ww_count = 0
    if include_alerts:
        # Live alerts as of today — used for demo even when the replay storm
        # is historical, per user instruction.
        live_alerts = got.get("alerts") or []
        # Split NHC Tropical Cyclone watches/warnings out of the generic
        # alerts stream so they render with the operational NHC colour
        # scheme + carry their own exposure rollup. Residual alerts (flood,
        # tornado, ...) stay on the generic severity palette.
        nhc_ww, residual_alerts = split_watches_warnings(live_alerts)
        for a in residual_alerts:
            alerts_out.append(
                WeatherAlertOut(
                    alert_id=a.alert_id,
                    event=a.event,
                    headline=a.headline,
                    severity=a.severity,
                    urgency=a.urgency,
                    certainty=a.certainty,
                    sent_at=a.sent_at,
                    expires_at=a.expires_at,
                    areas_affected=a.areas_affected,
                    geometry=a.geometry,
                )
            )
        for w in nhc_ww:
            if w.geometry is None:
                zone_only_ww_count += 1
            watches_warnings_out.append(
                NHCWatchWarnOut(
                    alert_id=w.alert_id,
                    event=w.event,
                    family=w.family,
                    color=w.color,
                    rank=w.rank,
                    headline=w.headline,
                    severity=w.severity,
                    urgency=w.urgency,
                    certainty=w.certainty,
                    sent_at=w.sent_at,
                    expires_at=w.expires_at,
                    areas_affected=w.areas_affected,
                    geometry=w.geometry,
                )
            )

    buoys_out: list[BuoyOut] = []
    land_out: list[LandObsOut] = []
    if include_obs:
        for b in got.get("buoys") or []:
            buoys_out.append(
                BuoyOut(
                    station_id=b.station_id,
                    lat=b.lat,
                    lon=b.lon,
                    wind_kt=b.wind_kt,
                    wind_dir_deg=b.wind_dir_deg,
                    gust_kt=b.gust_kt,
                    wave_height_ft=b.wave_height_ft,
                    pressure_mb=b.pressure_mb,
                    air_temp_f=b.air_temp_f,
                    water_temp_f=b.water_temp_f,
                    observed_at=b.observed_at,
                )
            )
    if include_land:
        # max_stations=80 is the service default — gives a spatially uniform
        # grid subsample instead of clumping into whichever region has the
        # densest instrumentation.
        for ls in got.get("land") or []:
            land_out.append(
                LandObsOut(
                    station_id=ls.station_id,
                    name=ls.name,
                    lat=ls.lat,
                    lon=ls.lon,
                    wind_kt=ls.wind_kt,
                    wind_dir_deg=ls.wind_dir_deg,
                    gust_kt=ls.gust_kt,
                    pressure_mb=ls.pressure_mb,
                    temp_f=ls.temp_f,
                    observed_at=ls.observed_at,
                )
            )

    sst_out: list[SSTOut] = []
    sst_min = sst_max = None
    sst_source = "synthetic"
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    sst_step = 0.05 if span < 6 else 0.10 if span < 12 else 0.25 if span < 25 else 0.5
    if include_sst:
        grid, sst_source = got.get("sst") or ([], "synthetic")
        sst_out = [
            SSTOut(
                lat=p.lat,
                lon=p.lon,
                temp_c=p.temp_c,
                favorable_for_intensification=p.favorable_for_intensification,
            )
            for p in grid
        ]
        if grid:
            sst_min = round(min(p.temp_c for p in grid), 1)
            sst_max = round(max(p.temp_c for p in grid), 1)

    storm_row = LiveStormRow(
        storm_id=observed_storm.storm_id,
        name=observed_storm.name,
        year=observed_storm.year,
        classification=observed_storm.track[-1].status,
        intensity_kt=max((p.wind_kt for p in observed_storm.track), default=0),
        pressure_mb=observed_storm.track[-1].pressure_mb,
        lat=observed_storm.track[-1].lat,
        lon=observed_storm.track[-1].lon,
        is_live=is_live,
        label=f"{observed_storm.name} ({observed_storm.year})",
    )

    # Wind fields: inner Rmax + outer asymmetric R64, same machinery as
    # historical impact. Built for the OBSERVED track (history) and the
    # LATEST forecast advisory (projection). For LIVE storms we pass
    # storm_id="" to disable the IBTrACS Rmax/R64 lookups — those datasets
    # never contain the current year, and even the failing lookup would
    # trigger a 70 MB CSV fetch that blew Vercel's cold-start budget.
    cone_storm_id = "" if is_live else observed_storm.storm_id
    official_radii = {f.hours_out: f for f in (got.get("official") or [])}
    observed_fixes_for_cone = [
        (p.lat, p.lon, p.wind_kt, p.datetime_utc) for p in observed_storm.track
    ]
    obs_taus = [0] * len(observed_fixes_for_cone)
    obs_fp, obs_inner, obs_outer, obs_rings = build_wind_cones(
        cone_storm_id,
        observed_fixes_for_cone,
        radii_by_tau=official_radii or None,
        taus=obs_taus if official_radii else None,
    )

    if forecasts:
        latest = max(forecasts, key=lambda f: f.advisory_number)
        forecast_fixes_for_cone = [
            (fp.lat, fp.lon, fp.wind_kt, fp.valid_time) for fp in latest.points
        ]
        fcst_taus = [fp.hours_out for fp in latest.points]
        _fp_fcst, fcst_inner, fcst_outer, fcst_rings = build_wind_cones(
            cone_storm_id,
            forecast_fixes_for_cone,
            radii_by_tau=official_radii or None,
            taus=fcst_taus if official_radii else None,
        )
    else:
        fcst_inner, fcst_outer, fcst_rings = [], [], []

    def _q_out(q) -> ConeQuadOut:
        return ConeQuadOut(
            corners=[
                [round(lon, 4), round(lat, 4)] for (lon, lat) in q.corners
            ] + [[round(q.corners[0][0], 4), round(q.corners[0][1], 4)]],
            wind_kt=q.wind_kt,
            start_wind_kt=q.start_wind_kt,
            end_wind_kt=q.end_wind_kt,
        )

    def _r_out(r: dict) -> OuterRingOut:
        return OuterRingOut(
            corners=r["ring"],
            wind_kt=r["wind_kt"],
            r64_nm=round(r["r64_nm"], 1),
            r64_source=r["r64_source"],
        )

    def _nhc_rings(polys: list[tuple[int, list[tuple[float, float]]]]) -> list[OuterRingOut]:
        rings: list[OuterRingOut] = []
        for wind_kt, ring in polys:
            if len(ring) < 4:
                continue
            rings.append(
                OuterRingOut(
                    corners=[[round(lon, 4), round(lat, 4)] for lon, lat in ring],
                    wind_kt=wind_kt,
                    r64_nm=0.0,
                    r64_source="nhc",
                )
            )
        return rings

    nhc_initial: list = []
    nhc_forecast_radii: list = []
    cone_ring: list = []
    surge_polys: list = []
    if is_live:
        cone_ring = got.get("cone") or []
        nhc_initial, nhc_forecast_radii = got.get("radii") or ([], [])
        surge_polys = got.get("surge") or []
        if not cone_ring and forecasts:
            latest_pts = max(forecasts, key=lambda f: f.advisory_number).points
            cone_ring = synthetic_forecast_cone(
                [(p.lat, p.lon, p.hours_out) for p in latest_pts],
                basin=atcf_id[:2].upper(),
            )

    # Prefer NHC's operational 34/50/64 kt swath. The parametric R64 cone is
    # the fallback (and the only product for replay storms). A tropical
    # storm's 34-kt polygon is what was missing when the chip looked empty.
    obs_outer_rings = _nhc_rings(nhc_initial) or [_r_out(r) for r in obs_rings]
    fcst_outer_rings = _nhc_rings(nhc_forecast_radii) or [_r_out(r) for r in fcst_rings]

    observed_wind = WindFieldOut(
        inner_cone=[_q_out(q) for q in obs_inner],
        outer_cone=[_q_out(q) for q in obs_outer],
        outer_rings=obs_outer_rings,
    )
    forecast_wind = WindFieldOut(
        inner_cone=[_q_out(q) for q in fcst_inner],
        outer_cone=[_q_out(q) for q in fcst_outer],
        outer_rings=fcst_outer_rings,
    )

    # NHC-issued products only exist for live storms. Cone is a single ring;
    # peak surge is per-band coloured polygons along the coast.
    forecast_cone_out: ForecastConeOut | None = None
    peak_surge_out: list[SurgePolygonOut] = []
    if is_live and cone_ring:
        forecast_cone_out = ForecastConeOut(
            ring=[[round(lon, 4), round(lat, 4)] for lon, lat in cone_ring],
        )
    if is_live and include_surge:
        for poly in surge_polys:
            peak_surge_out.append(
                SurgePolygonOut(
                    ring=[[round(lon, 5), round(lat, 5)] for lon, lat in poly.coords],
                    surge_range=poly.surge_range,
                    color=poly.color,
                )
            )

    # Hurricane hunters — live missions only. Replay storms have no
    # current HDOB; pulling today's archive would mix in the wrong system.
    recon_out: list[ReconObsOut] = []
    vortex_out: VortexFixOut | None = None
    recon_idw: list[WindObs] = []
    if is_live:
        recon_bundle = got.get("recon")
        if recon_bundle is not None:
            for fx in recon_bundle.fixes:
                recon_out.append(
                    ReconObsOut(
                        lat=fx.lat, lon=fx.lon, observed_at=fx.observed_at,
                        surface_kt=fx.surface_kt, surface_source=fx.surface_source,
                        fl_wind_kt=fx.fl_wind_kt, fl_dir_deg=fx.fl_dir_deg,
                        sfmr_kt=fx.sfmr_kt, rain_mm_hr=fx.rain_mm_hr,
                        aircraft=fx.aircraft, storm_name=fx.storm_name,
                        mission_id=fx.mission_id,
                    )
                )
            if recon_bundle.vortex is not None:
                v = recon_bundle.vortex
                vortex_out = VortexFixOut(
                    lat=v.lat, lon=v.lon, observed_at=v.observed_at,
                    pressure_mb=v.pressure_mb, max_fl_wind_kt=v.max_fl_wind_kt,
                    aircraft=v.aircraft, storm_id=v.storm_id,
                    storm_name=v.storm_name, mission_id=v.mission_id,
                )
            recon_idw = _advect_recon_samples(
                recon_bundle.fixes, observed_storm, forecasts,
            )

    # Interpolated wind heatmap. Runs for any storm (live or replay) since it
    # is purely observation-driven; caller can turn it off if the extra land
    # -station fetch is too slow. Live recon SFMR is mixed in when present.
    wind_map_out: list[WindGridPointOut] = []
    wind_obs_out: list[WindObsOut] = []
    wind_step = 0.5
    if include_wind_map:
        try:
            cells, wind_step, obs_pool = got.get("wind") or ([], 0.5, [])
        except Exception:  # noqa: BLE001
            cells, wind_step, obs_pool = [], 0.5, []
        # Hunter samples are not in the buoy/land fetch. Slide the ones
        # that were in the vortex forward with the center and re-grid.
        # The flight track itself stays where the plane measured it.
        if recon_idw:
            base = [o for o in obs_pool if o.source != "recon"]
            obs_pool = base + recon_idw
            cells = interpolate_obs(bbox[0], bbox[1], bbox[2], bbox[3], obs_pool)
            wind_step = 0.25
        wind_map_out = [
            WindGridPointOut(
                lat=c.lat,
                lon=c.lon,
                wind_kt=c.wind_kt,
                wind_dir_deg=c.wind_dir_deg,
                sources=c.sources,
                confidence=c.confidence,
                nearest_obs_km=c.nearest_obs_km,
                dist_score=c.dist_score,
                count_score=c.count_score,
                agreement_score=c.agreement_score,
                contributor_spread_kt=c.contributor_spread_kt,
            )
            for c in cells
        ]
        wind_obs_out = [
            WindObsOut(
                lat=o.lat, lon=o.lon,
                wind_kt=o.wind_kt, wind_dir_deg=o.wind_dir_deg,
                source=o.source, station_id=o.station_id,
                observed_at=o.observed_at,
            )
            for o in obs_pool
        ]

    return LiveStormBundle(
        storm=storm_row,
        observed_track=observed_fixes,
        forecasts=forecast_out,
        bbox=list(bbox),
        alerts=alerts_out,
        watches_warnings=watches_warnings_out,
        watches_warnings_zone_only=zone_only_ww_count,
        buoys=buoys_out,
        land_stations=land_out,
        recon=recon_out,
        vortex=vortex_out,
        sst=sst_out,
        sst_min_c=sst_min,
        sst_max_c=sst_max,
        sst_meta=SSTMeta(source=sst_source, step_deg=sst_step),
        observed_wind_field=observed_wind,
        forecast_wind_field=forecast_wind,
        forecast_cone=forecast_cone_out,
        peak_surge=peak_surge_out,
        wind_map=wind_map_out,
        wind_map_meta=WindGridMeta(
            step_deg=wind_step, obs_max_age_hours=4.0, idw_radius_km=333.0,
        ),
        wind_obs=wind_obs_out,
    )


class ReconPollOut(CamelModel):
    """Hunter points refreshed without rebuilding the whole live bundle.

    ``wind_map`` / ``wind_obs`` are set only when the buoy/land field from
    the last bundle is still cached. Null means "leave the heatmap as it
    is" — a poll must not wipe buoys just because that cache expired.
    """

    recon: list[ReconObsOut]
    vortex: VortexFixOut | None
    wind_map: list[WindGridPointOut] | None = None
    wind_obs: list[WindObsOut] | None = None
    polled_at: str


@router.get("/storms/{atcf_id}/recon", response_model=ReconPollOut)
def recon_poll(
    atcf_id: str,
    west: float = Query(..., ge=-180.0, le=180.0),
    south: float = Query(..., ge=-90.0, le=90.0),
    east: float = Query(..., ge=-180.0, le=180.0),
    north: float = Query(..., ge=-90.0, le=90.0),
    name: str = Query(default=""),
    centers: str = Query(
        default="",
        description="iso,lat,lon|iso,lat,lon center track used to slide hunter samples",
    ),
) -> ReconPollOut:
    """Latest hurricane-hunter points for a storm the map already has open.

    The panel calls this on a timer. Archive files are cached, so a poll
    only downloads bulletins that were not on the previous pass. The
    flight track is returned where the plane measured it. Samples that
    were inside the vortex are also slid forward with the center track
    and folded into the observed wind grid when that grid is still cached.
    """
    now = datetime.now(timezone.utc)
    bbox = (west, south, east, north)
    bundle = fetch_recon_bundle(
        bbox, atcf_id=atcf_id, storm_name=name or "", now=now,
    )
    recon_out = [
        ReconObsOut(
            lat=fx.lat, lon=fx.lon, observed_at=fx.observed_at,
            surface_kt=fx.surface_kt, surface_source=fx.surface_source,
            fl_wind_kt=fx.fl_wind_kt, fl_dir_deg=fx.fl_dir_deg,
            sfmr_kt=fx.sfmr_kt, rain_mm_hr=fx.rain_mm_hr,
            aircraft=fx.aircraft, storm_name=fx.storm_name,
            mission_id=fx.mission_id,
        )
        for fx in bundle.fixes
    ]
    vortex_out = None
    if bundle.vortex is not None:
        v = bundle.vortex
        vortex_out = VortexFixOut(
            lat=v.lat, lon=v.lon, observed_at=v.observed_at,
            pressure_mb=v.pressure_mb, max_fl_wind_kt=v.max_fl_wind_kt,
            aircraft=v.aircraft, storm_id=v.storm_id,
            storm_name=v.storm_name, mission_id=v.mission_id,
        )
    samples = [
        (
            fx.lat, fx.lon, fx.surface_kt, fx.fl_dir_deg, fx.observed_at,
            f"{fx.aircraft}-{fx.observed_at[11:16]} adv",
        )
        for fx in recon_for_idw(bundle.fixes)
    ]
    advected = advect_recon_obs(samples, _parse_centers(centers), now)
    blended = regrid_with_recon(west, south, east, north, advected)
    wind_map = None
    wind_obs = None
    if blended is not None:
        cells, _step, obs = blended
        wind_map = [
            WindGridPointOut(
                lat=c.lat, lon=c.lon, wind_kt=c.wind_kt,
                wind_dir_deg=c.wind_dir_deg, sources=c.sources,
                confidence=c.confidence, nearest_obs_km=c.nearest_obs_km,
                dist_score=c.dist_score, count_score=c.count_score,
                agreement_score=c.agreement_score,
                contributor_spread_kt=c.contributor_spread_kt,
            )
            for c in cells
        ]
        wind_obs = [
            WindObsOut(
                lat=o.lat, lon=o.lon, wind_kt=o.wind_kt,
                wind_dir_deg=o.wind_dir_deg, source=o.source,
                station_id=o.station_id, observed_at=o.observed_at,
            )
            for o in obs
        ]
    return ReconPollOut(
        recon=recon_out,
        vortex=vortex_out,
        wind_map=wind_map,
        wind_obs=wind_obs,
        polled_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@router.get("/wind-model-grid", response_model=WindModelGridOut)
def wind_model_grid(
    west: float = Query(..., ge=-180.0, le=180.0),
    south: float = Query(..., ge=-90.0, le=90.0),
    east: float = Query(..., ge=-180.0, le=180.0),
    north: float = Query(..., ge=-90.0, le=90.0),
    model: str = Query(..., pattern="^(gfs|ecmwf)$"),
    refresh: bool = Query(default=False, alias="refresh"),
) -> WindModelGridOut:
    """GFS or ECMWF surface-wind grid over a bbox, as forecast frames
    (0 → 120 h). The step starts at 0.5° and coarsens when the cone is
    large, so the request finishes instead of dying at the proxy.
    Degrades to an empty grid on failure rather than 5xx'ing."""
    grid = fetch_model_wind_grid(
        west, south, east, north, model, refresh=refresh,
    )
    return WindModelGridOut(
        model=grid.model,
        step_deg=grid.step_deg,
        cells=[
            WindGridCoordOut(lat=c.lat, lon=c.lon) for c in grid.cells
        ],
        frames=[
            WindModelFrameOut(
                hour=f.hour,
                valid_time_utc=f.valid_time_utc,
                wind_kt=f.wind_kt,
                wind_dir_deg=f.wind_dir_deg,
            )
            for f in grid.frames
        ],
        rate_limited=grid.rate_limited,
    )


@router.get("/wind-shear-grid", response_model=WindModelGridOut)
def wind_shear_grid(
    west: float = Query(..., ge=-180.0, le=180.0),
    south: float = Query(..., ge=-90.0, le=90.0),
    east: float = Query(..., ge=-180.0, le=180.0),
    north: float = Query(..., ge=-90.0, le=90.0),
    model: str = Query(..., pattern="^(gfs|ecmwf)$"),
    refresh: bool = Query(default=False, alias="refresh"),
) -> WindModelGridOut:
    """850–200 hPa deep-layer shear for the same hours as the surface grid.

    Magnitude is in knots. Direction is where the shear comes from, so the
    map arrow points downshear. Not the 200 hPa wind by itself. Empty on
    failure rather than 5xx, same as the surface grid.
    """
    grid = fetch_model_shear_grid(
        west, south, east, north, model, refresh=refresh,
    )
    return WindModelGridOut(
        model=grid.model,
        step_deg=grid.step_deg,
        cells=[
            WindGridCoordOut(lat=c.lat, lon=c.lon) for c in grid.cells
        ],
        frames=[
            WindModelFrameOut(
                hour=f.hour,
                valid_time_utc=f.valid_time_utc,
                wind_kt=f.wind_kt,
                wind_dir_deg=f.wind_dir_deg,
            )
            for f in grid.frames
        ],
        rate_limited=grid.rate_limited,
    )


@router.get("/wind-forecast", response_model=PointForecastOut)
def wind_forecast_at_point(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
) -> PointForecastOut:
    """GFS + ECMWF surface-wind forecast for a single point.

    Powers the click-to-inspect popup on the interpolated wind heatmap —
    users see obs vs both models side-by-side to sanity-check the field
    (agreement = high confidence; large disagreement = the observation is
    unusual or a model is off, either way worth flagging)."""
    result = point_forecast(lat, lon)
    return PointForecastOut(
        lat=result.lat,
        lon=result.lon,
        fetched_at_utc=result.fetched_at_utc,
        forecasts=[
            ModelForecastOut(
                model=f.model,
                valid_time_utc=f.valid_time_utc,
                wind_kt=f.wind_kt,
                wind_dir_deg=f.wind_dir_deg,
                wind_gust_kt=f.wind_gust_kt,
            )
            for f in result.forecasts
        ],
    )


@router.get("/daily-forecast", response_model=DailyForecastOut)
def daily_forecast_at_point(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
) -> DailyForecastOut:
    """Five local days of GFS and ECMWF at one point.

    Powers the tucked-away location card. Not the hourly click popup and
    not the wind-grid fetch. Temperatures are Fahrenheit, rain is inches,
    wind is the daily maximum in knots.
    """
    result = daily_forecast(lat, lon)
    return DailyForecastOut(
        lat=result.lat,
        lon=result.lon,
        timezone=result.timezone,
        fetched_at_utc=result.fetched_at_utc,
        models=[
            DailyModelOut(
                model=m.model,
                days=[
                    DailyDayOut(
                        date=d.date,
                        weather_code=d.weather_code,
                        temp_max_f=d.temp_max_f,
                        temp_min_f=d.temp_min_f,
                        precip_in=d.precip_in,
                        wind_max_kt=d.wind_max_kt,
                        gust_max_kt=d.gust_max_kt,
                        wind_dir_deg=d.wind_dir_deg,
                    )
                    for d in m.days
                ],
            )
            for m in result.models
        ],
    )


# ─────────────────── NHC Tropical Weather Outlook ───────────────────


class GTWOAreaOut(CamelModel):
    """One formation-area polygon from the NHC Graphical Tropical Weather
    Outlook (7-day envelope). ``chance_pct`` is the representative percent
    for the area's bucket (0 / 20 / 50 / 80) — the current NHC KML encodes
    chance only as a bucket via styleUrl, not as an exact percent.
    ``chance_bucket`` matches NHC's legend: none (gray) / low (yellow) /
    medium (orange) / high (red)."""

    basin: str
    chance_pct: int
    chance_bucket: str            # "none" | "low" | "medium" | "high"
    label: str
    description: str
    ring: list[list[float]]       # closed [[lon, lat], ...]
    marker: list[float] | None    # [lon, lat] of NHC's designated point label, if any


class GTWOResponse(CamelModel):
    basin: str
    areas: list[GTWOAreaOut]
    issued_note: str | None       # "Mon Aug 10 23:41:16 2026" — from KML doc name
    note: str | None
    attribution: str = (
        "NHC Graphical Tropical Weather Outlook (issued every 6h). Polygons "
        "represent NHC's assessment of the 7-day formation envelope for each "
        "disturbance. Not a track forecast; the 2-day chance is only in the "
        "text outlook and is not currently wired to this feed."
    )


@router.get("/gtwo", response_model=GTWOResponse)
def gtwo_endpoint(
    basin: str = Query(default="atl", pattern="^(atl|pac|cpac)$"),
) -> GTWOResponse:
    """NHC Tropical Weather Outlook formation-area polygons for one basin.

    This is the pre-invest signal — days before a system gets a numbered
    invest slot and typically a week+ before a name. Empty list on
    unreachable KMZ feed is surfaced with an explanatory note (never
    silently confused with "no active areas")."""
    bundle = fetch_gtwo(basin)
    return GTWOResponse(
        basin=bundle.basin,
        areas=[
            GTWOAreaOut(
                basin=a.basin,
                chance_pct=a.chance_pct,
                chance_bucket=a.chance_bucket,
                label=a.label,
                description=a.description,
                ring=[[round(lon, 4), round(lat, 4)] for (lon, lat) in a.ring],
                marker=(
                    [round(a.marker[0], 4), round(a.marker[1], 4)]
                    if a.marker else None
                ),
            )
            for a in bundle.areas
        ],
        issued_note=bundle.issued_note,
        note=bundle.note,
    )


# ─────────────────── model ensemble spaghetti tracks ───────────────────


class ModelFixOut(CamelModel):
    hours_out: int
    lat: float
    lon: float
    wind_kt: int
    pressure_mb: int | None


class ModelTrackOut(CamelModel):
    """One model's projected track for the storm's chosen init cycle."""

    tech_id: str          # ATCF 4-char id (OFCL, AVNO, ECMF, AP01, GRAP, ...)
    label: str            # human-readable ("NHC Official", "GEFS member 01")
    family: str           # taxonomy bucket: official/consensus/ai/gefs_ens/...
    init_cycle: str       # "YYYY-MM-DDTHHZ"
    fixes: list[ModelFixOut]


class EnvelopeAnchorOut(CamelModel):
    hours_out: int
    lat: float
    lon: float


class EnsembleEnvelopeOut(CamelModel):
    """Data-driven consensus envelope — convex hull of every ensemble
    member's position at every lead-time anchor. Narrow where members
    agree, wide where they disagree. Complements NHC's climatological
    cone; does not replace it."""

    members_used: int
    ring: list[list[float]]      # closed [[lon, lat], ...]
    # Per-anchor hull vertices, exposed so the frontend can dot them out.
    anchor_hulls: dict[str, list[EnvelopeAnchorOut]]


class ModelFamilySummaryOut(CamelModel):
    """One row per family present in the response — drives the legend + chip
    group renders on the frontend."""

    family: str
    track_count: int
    tech_ids: list[str]


class ModelTracksResponse(CamelModel):
    storm_id: str
    init_cycle: str | None       # "YYYY-MM-DDTHHZ", null if a-deck unavailable
    available_cycles: list[str]  # most recent first
    tracks: list[ModelTrackOut]
    families: list[ModelFamilySummaryOut]
    ensemble_envelope: EnsembleEnvelopeOut | None
    ai_envelope: EnsembleEnvelopeOut | None
    notes: list[str]
    attribution: str = (
        "NHC ATCF aid_public a-decks (deterministic + GEFS + ECMWF-ENS + "
        "AI models where NCEP publishes them)."
    )


@router.get(
    "/storms/{atcf_id}/model-tracks",
    response_model=ModelTracksResponse,
)
def model_tracks(
    atcf_id: str,
    init_cycle: str | None = Query(default=None, alias="initCycle"),
    include_baselines: bool = Query(default=False, alias="includeBaselines"),
    refresh: bool = Query(default=False, alias="refresh"),
) -> ModelTracksResponse:
    """Per-model projected tracks for one storm's latest init cycle.

    Fetches the per-storm a-deck (``https://ftp.nhc.noaa.gov/atcf/aid_public
    /aal{NN}{YYYY}.dat.gz``), collapses rows to (tech id × lead time), groups
    by family for the legend, and returns both the individual member tracks
    (spaghetti input) and a data-driven consensus envelope. Degrades to an
    empty response when the a-deck is unavailable (very early in a storm's
    lifecycle) — the frontend must handle empty gracefully.

    ``initCycle`` (YYYYMMDDHH) restricts to a specific model cycle. Default:
    the latest cycle present in the file.
    """
    if is_jma_id(atcf_id):
        return ModelTracksResponse(
            storm_id=atcf_id.upper(),
            init_cycle=None,
            available_cycles=[],
            tracks=[],
            families=[],
            ensemble_envelope=None,
            ai_envelope=None,
            notes=[
                "JMA does not publish an NHC a-deck. There is no ensemble "
                "member vote for this typhoon.",
            ],
            attribution="Japan Meteorological Agency typhoon bulletin.",
        )
    tracks = fetch_model_tracks(
        atcf_id,
        init_cycle=init_cycle,
        include_baselines=include_baselines,
        refresh=refresh,
    )
    cycles = list_available_cycles(atcf_id, limit=8, refresh=refresh)
    notes: list[str] = []

    if not tracks:
        notes.append(
            "No a-deck data available yet. Very-early-lifecycle storms may "
            "not be in the aid_public archive for a cycle or two after "
            "genesis; some subtropical / demonstration storms are also "
            "excluded from the ATCF."
        )
        return ModelTracksResponse(
            storm_id=atcf_id.upper(),
            init_cycle=None,
            available_cycles=cycles,
            tracks=[],
            families=[],
            ensemble_envelope=None,
            ai_envelope=None,
            notes=notes,
        )

    # Group tech ids by family for the legend row.
    family_tech: dict[str, list[str]] = {}
    for t in tracks:
        family_tech.setdefault(t.family, []).append(t.tech_id)

    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    families = [
        ModelFamilySummaryOut(
            family=fam,
            track_count=len(techs),
            tech_ids=sorted(techs),
        )
        for fam, techs in sorted(family_tech.items(), key=lambda kv: fam_rank.get(kv[0], 99))
    ]

    def _envelope_out(env) -> EnsembleEnvelopeOut | None:
        if env is None:
            return None
        return EnsembleEnvelopeOut(
            members_used=env.members_used,
            ring=[[round(lon, 3), round(lat, 3)] for (lon, lat) in env.ring],
            anchor_hulls={
                str(h): [
                    EnvelopeAnchorOut(hours_out=p.hours_out, lat=p.lat, lon=p.lon)
                    for p in verts
                ]
                for h, verts in env.anchor_hulls.items()
            },
        )

    # Full ensemble envelope: GEFS + ECMWF-ENS + AI models together.
    ens_env = build_envelope(tracks)
    # AI-only envelope: interesting stand-alone signal — "how much do the
    # newest AI models agree with each other?". Often tighter than the NWP
    # ensembles' spaghetti, sometimes surprisingly not.
    # One GraphCast run is still a useful envelope (drawn as a corridor
    # when the track is nearly straight). Requiring two AI techs left the
    # chip gray on every storm whose a-deck only carries GDMI.
    ai_env = build_envelope(
        tracks, include_families=frozenset({"ai"}), min_members=1,
    )

    if ens_env is None:
        notes.append(
            "Ensemble consensus envelope not built — fewer than 5 ensemble "
            "members (GEFS, ECMWF-ENS, AI) returned tracks in the last 24h."
        )
    inits = {t.init_cycle for t in tracks}
    if len(inits) > 1:
        notes.append(
            "Each model is shown at its own latest cycle from the last 24 "
            "hours. GEFS and ECMWF often publish a cycle behind the early "
            "NHC aids, so they are not dropped while that cycle is still in."
        )

    tracks_out = [_track_out(t) for t in tracks]
    newest_init = max(inits) if inits else None

    return ModelTracksResponse(
        storm_id=atcf_id.upper(),
        init_cycle=newest_init,
        available_cycles=cycles,
        tracks=tracks_out,
        families=families,
        ensemble_envelope=_envelope_out(ens_env),
        ai_envelope=_envelope_out(ai_env),
        notes=notes,
    )


def _track_out(t: ModelTrack) -> ModelTrackOut:
    return ModelTrackOut(
        tech_id=t.tech_id,
        label=t.label if t.label else MODEL_LABEL.get(t.tech_id, t.tech_id),
        family=t.family,
        init_cycle=t.init_cycle,
        fixes=[
            ModelFixOut(
                hours_out=f.hours_out,
                lat=round(f.lat, 3),
                lon=round(f.lon, 3),
                wind_kt=f.wind_kt,
                pressure_mb=f.pressure_mb,
            )
            for f in t.fixes
        ],
    )


# ─────────────────── ensemble strike-probability + intensity spread ───────────────────


class CountyStrikeProbOut(CamelModel):
    geoid: str
    geography_id: str
    name: str
    state_usps: str
    centroid_lat: float
    centroid_lon: float
    strike_probability: float   # 0..1 — fraction of ensemble members within threshold
    member_count: int           # members whose track passes within threshold
    ensemble_total: int
    max_intensity_kt: int       # peak wind kt of any passing member near the county


class IntensityStatOut(CamelModel):
    hours_out: int
    member_count: int
    min_kt: int
    mean_kt: float
    max_kt: int
    std_kt: float


class EnsembleRiskResponse(CamelModel):
    storm_id: str
    init_cycle: str | None
    ensemble_total: int
    threshold_nm: float
    strike_by_county: list[CountyStrikeProbOut]
    intensity_by_lead: list[IntensityStatOut]
    notes: list[str]
    attribution: str = (
        "Derived from GEFS + ECMWF-ENS + AI member tracks in the NHC "
        "ATCF a-deck. Strike = track passes within threshold nm of the "
        "county centroid at any lead time ≥ 24h; probability is the "
        "fraction of considered members that struck."
    )


@router.get(
    "/storms/{atcf_id}/ensemble-risk",
    response_model=EnsembleRiskResponse,
)
def ensemble_risk_endpoint(
    atcf_id: str,
    threshold_nm: float = Query(
        default=DEFAULT_STRIKE_THRESHOLD_NM,
        ge=10.0, le=200.0,
        alias="thresholdNm",
    ),
    all_states: bool = Query(default=False, alias="allStates"),
    refresh: bool = Query(default=False, alias="refresh"),
) -> EnsembleRiskResponse:
    """Per-county ensemble strike probability + per-lead intensity spread.

    Restricted to Atlantic + Gulf coastal states by default; set
    ``allStates=true`` to walk the full US county set (slower). Threshold
    is in nautical miles from the county centroid to the nearest point on
    each member's track. Default 60 nm ≈ R64 envelope of a mature hurricane.
    Not computed for JMA typhoons: there is no public member file, and the
    vote is over US county centroids.
    """
    if is_jma_id(atcf_id):
        return EnsembleRiskResponse(
            storm_id=atcf_id.upper(),
            init_cycle=None,
            ensemble_total=0,
            threshold_nm=threshold_nm,
            strike_by_county=[],
            intensity_by_lead=[],
            notes=[
                "Strike probability is a US county vote over NHC a-deck "
                "members. It is not computed for a JMA typhoon.",
            ],
        )
    tracks = fetch_model_tracks(atcf_id, refresh=refresh)
    notes: list[str] = []
    if not tracks:
        notes.append(
            "No a-deck data available for this storm. Ensemble-risk aggregates "
            "require at least a few ensemble members' worth of a-deck rows."
        )
        return EnsembleRiskResponse(
            storm_id=atcf_id.upper(),
            init_cycle=None,
            ensemble_total=0,
            threshold_nm=threshold_nm,
            strike_by_county=[],
            intensity_by_lead=[],
            notes=notes,
        )
    coastal = None if all_states else ATLANTIC_COASTAL_STATES
    risk = compute_ensemble_risk(
        tracks,
        threshold_nm=threshold_nm,
        coastal_states=coastal,
    )
    if risk.ensemble_total == 0:
        notes.append(
            "A-deck rows found but no ensemble members (GEFS / ECMWF-ENS / "
            "AI) present — the strike-probability aggregate needs multiple "
            "independent forecasts."
        )
    elif not risk.strike_by_county:
        notes.append(
            "No ensemble member's track passed within "
            f"{threshold_nm:.0f} nm of any coastal county centroid. Increase "
            "the threshold, or the storm may be too far out or on a "
            "sea-only trajectory."
        )
    ofcl_cycles = [
        t.init_cycle for t in tracks
        if t.tech_id == "OFCL" and t.init_cycle
    ]
    newest_ofcl = max(ofcl_cycles) if ofcl_cycles else None
    if (
        risk.init_cycle
        and newest_ofcl
        and risk.init_cycle < newest_ofcl
    ):
        notes.append(
            f"Most of the vote is still the {risk.init_cycle} ensemble. "
            f"The official forecast is {newest_ofcl} and can sit well east "
            f"or west of those members until the new ensemble is in the "
            f"a-deck. The circles follow the members, not the official line, "
            f"and they refresh when that file updates."
        )
    return EnsembleRiskResponse(
        storm_id=atcf_id.upper(),
        init_cycle=risk.init_cycle,
        ensemble_total=risk.ensemble_total,
        threshold_nm=risk.threshold_nm,
        strike_by_county=[
            CountyStrikeProbOut(
                geoid=s.geoid,
                geography_id=s.geography_id,
                name=s.name,
                state_usps=s.state_usps,
                centroid_lat=s.centroid_lat,
                centroid_lon=s.centroid_lon,
                strike_probability=s.strike_probability,
                member_count=s.member_count,
                ensemble_total=s.ensemble_total,
                max_intensity_kt=s.max_intensity_kt,
            )
            for s in risk.strike_by_county
        ],
        intensity_by_lead=[
            IntensityStatOut(
                hours_out=i.hours_out,
                member_count=i.member_count,
                min_kt=i.min_kt,
                mean_kt=i.mean_kt,
                max_kt=i.max_kt,
                std_kt=i.std_kt,
            )
            for i in risk.intensity_by_lead
        ],
        notes=notes,
    )


# ─────────────────── exposed TIV inside NHC watches/warnings ───────────────────


class WatchWarnExposureResponse(CamelModel):
    """Rollup of exposed TIV inside a set of NHC watch/warning polygons.

    Same synthetic-point machinery as wildfire / flood — TIV is derived from
    county-aggregated exposure scattered as deterministic locations, then
    counted by point-in-polygon. Flagged ``synthetic`` on the wire and in the
    UI: the true answer needs location-level exposure, which v1 does not have.
    """

    currency: str
    synthetic: bool
    note: str
    results: list[PolygonExposureOut]
    combined: PolygonExposureOut
    warnings: list[str]


@router.post("/watches-warnings/exposure", response_model=WatchWarnExposureResponse)
def post_watch_warn_exposure(req: WatchWarnExposureRequest) -> WatchWarnExposureResponse:
    """Roll up exposed TIV by client for the supplied NHC watch/warning
    polygons. A full active set is one polygon per coastal zone (well over the
    50-shape cap used when a user picks fire or flood polygons), so this route
    accepts that set. ``combined`` is the deduped union across all polygons
    (each synthetic location counted once) so overlapping Hurricane Warning +
    Storm Surge Warning areas over the same coast do not double-count.

    Zone-coded watches (no polygon) cannot be rolled up here — surface them
    from the bundle's ``watchesWarnings[]`` entries whose ``geometry`` is
    null and count them separately in the UI.
    """
    try:
        per, union = wildfire_exposure.exposure_in_polygons(
            [p.geometry for p in req.polygons]
        )
        currency = wildfire_exposure.currency()
        warnings = list(wildfire_exposure.load_warnings())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={
            "code": "UPSTREAM_UNAVAILABLE",
            "message": f"Exposure data could not be loaded: {exc}",
        }) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "GEOMETRY_TOO_COMPLEX",
            "message": str(exc),
        }) from exc

    results = [exposure_out(p.id, p.name, r) for p, r in zip(req.polygons, per)]
    combined = exposure_out("combined", None, union)

    return WatchWarnExposureResponse(
        currency=currency,
        synthetic=True,
        note=(
            "Estimated from synthetic location points distributed within counties "
            "from aggregate TIV — not real location-level data. Replace the "
            "location source with individual-location exposure for exact "
            "in-warning TIV. Also, watch/warning polygons are threat areas, "
            "not observed damage — so this is an upper bound on the TIV that "
            "will actually see the corresponding wind or surge."
        ),
        results=results,
        combined=combined,
        warnings=warnings,
    )


__all__ = ["router"]
