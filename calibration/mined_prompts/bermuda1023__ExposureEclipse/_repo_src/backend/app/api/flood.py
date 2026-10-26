"""Live flood endpoint — active NWS flood watches, warnings and advisories.

GET /api/flood/active
    ?bbox=west,south,east,north   (optional; omit for nationwide)
    &minSeverity=Minor            (Unknown|Minor|Moderate|Severe|Extreme)

Returns a GeoJSON FeatureCollection of polygon-bearing flood alerts, a
per-state roll-up, and notes explaining anything that was dropped.

POST /api/flood/exposure
    Exposed TIV by client for a set of selected alert polygons — same engine
    and the same rules-3+4 rollup the wildfire overlay uses.

GET /api/flood/inundation
    ?bbox=west,south,east,north   (required — a nationwide request truncates)
    &simplify=0.001               (generalisation in degrees; 0 = full)

POST /api/flood/inundation/exposure
    Exposed TIV inside the modelled extent for a bbox. Takes the bbox rather
    than geometry: the extent is thousands of reach polygons, so posting it
    back would mean shipping megabytes to the browser and straight back again.

Two stacked layers, deliberately not merged. Alerts are *warning areas* issued
by forecasters; inundation is *modelled water* from the National Water Model.
Neither is a superset of the other — the model covers ~30% of the US
population and no coastal processes, and alerts exist where the model is
silent — so they are served separately and drawn separately.

Live overlay, like /live/storms and /wildfire/active — not part of the mock
data plane. Sources: NWS api.weather.gov, NOAA maps.water.noaa.gov.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from ..models.common import CamelModel
from ..services import nwm_inundation, wildfire_exposure
from ..services.live_flood import SEVERITY_ORDER, SEVERITY_RANK, fetch_flood_alerts
from ..services.nwm_inundation import (
    DEFAULT_SIMPLIFY_DEG,
    InundationUnavailable,
    NWM_ATTRIBUTION,
    NWM_ATTRIBUTION_URL,
)
from .geometry_input import ExposureRequest, PolygonExposureOut, exposure_out

router = APIRouter(prefix="/flood", tags=["flood"])


# ─────────────────────────── wire types ───────────────────────────


class AffectedStateOut(CamelModel):
    state: str
    alert_count: int


class FloodCounts(CamelModel):
    alerts: int
    zone_only: int


class FloodAttribution(CamelModel):
    alerts: str = "NOAA / National Weather Service active alerts"
    alerts_url: str = "https://api.weather.gov/alerts/active"


class FloodResponse(CamelModel):
    generated_at: str
    bbox: list[float] | None
    min_severity: str
    alerts: dict  # GeoJSON FeatureCollection
    affected_states: list[AffectedStateOut]
    counts: FloodCounts
    notes: list[str]
    attribution: FloodAttribution


class FloodExposureResponse(CamelModel):
    currency: str
    synthetic: bool
    note: str
    results: list[PolygonExposureOut]
    combined: PolygonExposureOut
    warnings: list[str]


class InundationCounts(CamelModel):
    reaches: int


class InundationAttribution(CamelModel):
    model: str = NWM_ATTRIBUTION
    model_url: str = NWM_ATTRIBUTION_URL


class InundationResponse(CamelModel):
    generated_at: str
    bbox: list[float]
    reference_time: str | None
    # True whenever the upstream cut the extent short. Callers must not read a
    # truncated extent as the full picture, so it rides on the response rather
    # than only appearing in prose.
    truncated: bool
    # The model could not be reached. Without this an outage and a genuinely dry
    # view are the same response — zero reaches, null reference time — and the
    # only thing separating them is prose in `notes`, which a caller cannot
    # branch on.
    unavailable: bool = False
    inundation: dict  # GeoJSON FeatureCollection
    counts: InundationCounts
    notes: list[str]
    attribution: InundationAttribution


class InundationExposureRequest(CamelModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    simplify: float = Field(default=DEFAULT_SIMPLIFY_DEG, ge=0.0, le=0.05)


class InundationExposureResponse(CamelModel):
    currency: str
    synthetic: bool
    note: str
    reaches: int
    truncated: bool
    # The extent is narrower than the synthetic method can resolve, so a zero
    # here means "too small to sample", not "no exposure". See
    # `wildfire_exposure.resolution_deg2`.
    below_resolution: bool
    combined: PolygonExposureOut
    warnings: list[str]
    notes: list[str]


# ─────────────────────────── helpers ───────────────────────────


def _parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR",
                    "message": "bbox must be 'west,south,east,north'."},
        )
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "bbox values must be numbers."},
        )
    if not (west < east and south < north):
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR",
                    "message": "bbox must satisfy west<east and south<north."},
        )
    return (west, south, east, north)


# ─────────────────────────── routes ───────────────────────────


@router.get("/active", response_model=FloodResponse)
def get_active_flood(
    bbox: str | None = Query(default=None, description="west,south,east,north (lon/lat)"),
    min_severity: str = Query(default="Unknown", alias="minSeverity"),
) -> FloodResponse:
    """Active NWS flood alerts that carry mappable geometry.

    Only polygon-bearing alerts are returned. Zone-coded alerts (Coastal,
    Lakeshore, and most Watch products) are counted in ``counts.zoneOnly`` and
    explained in ``notes`` rather than silently dropped — they have no geometry
    to draw or intersect.

    Alert polygons are *warning areas*, often drawn to county or zone
    boundaries rather than observed water extent, so exposed TIV derived from
    them is an upper bound. See docs/API.md §GET /api/flood/active.

    Args:
        bbox: Bounding box ``"west,south,east,north"`` (lon/lat, EPSG:4326).
            Omit for nationwide.
        min_severity: Drop alerts below this NWS CAP severity. Use ``Severe``
            to approximate "major flooding only" — it is the only severity
            signal that arrives attached to the geometry.

    Returns:
        FloodResponse with the alert FeatureCollection, per-state roll-up,
        counts, notes and attribution.

    Raises:
        HTTPException 422: If ``bbox`` is malformed or ``minSeverity`` is not
            a recognised CAP severity.
    """
    box = _parse_bbox(bbox)
    if min_severity not in SEVERITY_RANK:
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR",
            "message": f"minSeverity must be one of {', '.join(SEVERITY_ORDER)}.",
        })

    bundle = fetch_flood_alerts(bbox=box, min_severity=min_severity)

    features = [
        {
            "type": "Feature",
            # NWS alert ids are stable URNs, so unlike the wildfire heat shapes
            # there is nothing to hash — the upstream id is the selection key.
            "id": a.alert_id,
            "geometry": a.geometry,
            "properties": {
                "alertId": a.alert_id,
                "event": a.event,
                "headline": a.headline,
                "severity": a.severity,
                # Numeric twin of `severity` so the map ramp can interpolate
                # without a string match expression.
                "severityRank": SEVERITY_RANK.get(a.severity, 0),
                "urgency": a.urgency,
                "certainty": a.certainty,
                "sentAt": a.sent_at,
                "expiresAt": a.expires_at,
                "areaDesc": a.areas_affected,
            },
        }
        for a in bundle.alerts
    ]

    return FloodResponse(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bbox=list(box) if box else None,
        min_severity=min_severity,
        alerts={"type": "FeatureCollection", "features": features},
        affected_states=[
            AffectedStateOut(state=st, alert_count=n) for (st, n) in bundle.affected_states
        ],
        counts=FloodCounts(alerts=len(bundle.alerts), zone_only=bundle.zone_only_count),
        notes=bundle.notes,
        attribution=FloodAttribution(),
    )


@router.post("/exposure", response_model=FloodExposureResponse)
def post_flood_exposure(req: ExposureRequest) -> FloodExposureResponse:
    """Roll up exposed TIV by client for the supplied flood-alert polygons.

    ``results`` holds one entry per polygon. ``combined`` is the union across
    all of them with each location counted once — adjacent flood warnings
    routinely overlap, and summing the per-polygon totals would double-count
    the shared ground.

    TIV is combined max-across-perils at the (client, county) grain per
    CLAUDE.md rules 3+4. v1 uses synthetic location points derived from
    county-aggregated TIV — see ``synthetic``/``note``.
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
        # Geometry passed per-field validation but would cost too much to
        # evaluate; that is a bad request, not a server fault.
        raise HTTPException(status_code=422, detail={
            "code": "GEOMETRY_TOO_COMPLEX",
            "message": str(exc),
        }) from exc

    results = [exposure_out(p.id, p.name, r) for p, r in zip(req.polygons, per)]
    combined = exposure_out("combined", None, union)

    return FloodExposureResponse(
        currency=currency,
        synthetic=True,
        note=("Estimated from synthetic location points distributed within counties "
              "from aggregate TIV — not real location-level data. Alert polygons are "
              "warning areas rather than observed water extent, so this is an upper "
              "bound on the affected area."),
        results=results,
        combined=combined,
        warnings=warnings,
    )


# ─────────────── modelled inundation extent (NWM) ───────────────


def _require_bbox(raw: str | None) -> tuple[float, float, float, float]:
    box = _parse_bbox(raw)
    if box is None:
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR",
            "message": "bbox is required for inundation — a nationwide request "
                       "is truncated upstream and would under-report the extent.",
        })
    return box


@router.get("/inundation", response_model=InundationResponse)
def get_flood_inundation(
    bbox: str | None = Query(default=None, description="west,south,east,north (lon/lat)"),
    simplify: float = Query(default=DEFAULT_SIMPLIFY_DEG, ge=0.0, le=0.05),
) -> InundationResponse:
    """Modelled flood inundation extent from the National Water Model.

    This is modelled water at reach resolution, not a warning area, so it is
    far more precise than the alert layer — but it is EXPERIMENTAL, covers
    roughly 30% of the US population, and models riverine flooding only. An
    empty extent is therefore never evidence that there is no flooding, and
    this layer supplements the alert layer rather than replacing it.

    Fails soft: if the model service cannot be reached the response is empty
    with an explanatory note, never an implied all-clear.

    Args:
        bbox: Required bounding box ``"west,south,east,north"`` (lon/lat).
            Capped at ``MAX_BBOX_DEG2`` square degrees.
        simplify: ``maxAllowableOffset`` in degrees. The 0.001 default cuts
            vertices ~9× with no visible change at mapping scales.

    Returns:
        InundationResponse with the reach FeatureCollection, the model
        reference time, a ``truncated`` flag and notes.

    Raises:
        HTTPException 422: If ``bbox`` is missing, malformed, or too large.
    """
    box = _require_bbox(bbox)
    try:
        bundle = nwm_inundation.fetch_inundation(bbox=box, simplify_deg=simplify)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR", "message": str(exc),
        }) from exc
    except InundationUnavailable:
        # An empty map during a flood must not be mistaken for a dry one.
        return InundationResponse(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            bbox=list(box),
            reference_time=None,
            truncated=False,
            unavailable=True,
            inundation={"type": "FeatureCollection", "features": []},
            counts=InundationCounts(reaches=0),
            notes=["The National Water Model service could not be reached, so no "
                   "modelled inundation could be loaded. This is NOT a statement "
                   "that there is no flooding — treat the layer as empty, not "
                   "clear, and retry."],
            attribution=InundationAttribution(),
        )

    return InundationResponse(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bbox=list(box),
        reference_time=bundle.reference_time,
        truncated=bundle.truncated,
        inundation={"type": "FeatureCollection", "features": bundle.features},
        counts=InundationCounts(reaches=len(bundle.features)),
        notes=bundle.notes,
        attribution=InundationAttribution(),
    )


@router.post("/inundation/exposure", response_model=InundationExposureResponse)
def post_inundation_exposure(req: InundationExposureRequest) -> InundationExposureResponse:
    """Exposed TIV by client inside the modelled inundation extent for a bbox.

    The extent arrives as thousands of per-reach polygons — far past the
    50-polygon cap on ``/flood/exposure`` — so they are combined here into a
    single multipart geometry before the rollup. That keeps the work budget
    meaningful and matches how an underwriter thinks about one flood event.
    Overlapping reaches do not double-count: the rollup collects location
    indices into a set.

    Unlike the map route this does NOT fail soft. Returning zero exposed TIV
    because the model was unreachable would be indistinguishable from a genuine
    zero, and that number feeds a layer calc.

    Raises:
        HTTPException 422: bbox malformed, too large, or the extent too complex.
        HTTPException 503: the model service or the exposure plane is unavailable.
    """
    box = (req.bbox[0], req.bbox[1], req.bbox[2], req.bbox[3])
    if not (box[0] < box[2] and box[1] < box[3]):
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR",
            "message": "bbox must satisfy west<east and south<north.",
        })

    try:
        bundle = nwm_inundation.fetch_inundation(bbox=box, simplify_deg=req.simplify)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR", "message": str(exc),
        }) from exc
    except InundationUnavailable as exc:
        raise HTTPException(status_code=503, detail={
            "code": "UPSTREAM_UNAVAILABLE",
            "message": f"The National Water Model service is unavailable: {exc}",
        }) from exc

    geoms = [g for f in bundle.features if (g := f.get("geometry"))]
    dissolved = nwm_inundation.dissolve(geoms)
    if dissolved is None:
        empty = exposure_out("inundation", "Modelled inundation", (0.0, 0, {}))
        return InundationExposureResponse(
            currency=wildfire_exposure.currency(),
            synthetic=True,
            note="The model shows no inundation in this view.",
            reaches=0,
            truncated=bundle.truncated,
            below_resolution=False,
            combined=empty,
            warnings=list(wildfire_exposure.load_warnings()),
            notes=bundle.notes,
        )

    try:
        # Price only the reaches that could hold a location. The work budget
        # charges (candidates × total vertices), and an ordinary 25 deg² extent
        # is ~1,900 reaches carrying ~200k vertices — 86M charged operations
        # against a budget of 8M — so without this the button 422s on exactly
        # the widespread floods it exists for. Dropping the rest cannot move the
        # answer; see `wildfire_exposure.geometries_with_candidates`.
        scoreable = nwm_inundation.dissolve(
            wildfire_exposure.geometries_with_candidates(geoms)
        )
        _, union = (
            wildfire_exposure.exposure_in_polygons([scoreable])
            if scoreable is not None
            else ([], (0.0, 0, {}))
        )
        currency = wildfire_exposure.currency()
        warnings = list(wildfire_exposure.load_warnings())
        # Measured on the FULL extent, not the priced subset: this describes how
        # big the modelled water actually is against what the method can sample,
        # which is a property of the water rather than of the optimisation.
        #
        # Only ever qualifies a zero, and only when the extent really is too
        # small to sample. Dropping the first conjunct would suppress a real
        # non-zero TIV behind an "unmeasurable" label; dropping the second
        # would call a genuine absence of exposure unmeasurable.
        # `extent_area_deg2` sums parts without a topological union, so it
        # overstates when reaches overlap — measured at 1.046× on a live
        # Houston extent, against a 133× margin to the floor.
        below_resolution = (
            union[1] == 0
            and nwm_inundation.extent_area_deg2(dissolved)
            < wildfire_exposure.resolution_deg2()
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={
            "code": "UPSTREAM_UNAVAILABLE",
            "message": f"Exposure data could not be loaded: {exc}",
        }) from exc
    except ValueError as exc:
        # The caller supplied a bbox, not geometry, so "simplify it or submit
        # fewer polygons" is advice they cannot act on. Zooming in is: it cuts
        # the reach count and the candidate sweep at the same time.
        raise HTTPException(status_code=422, detail={
            "code": "GEOMETRY_TOO_COMPLEX",
            "message": f"The modelled extent in this view is too detailed to "
                       f"price ({exc}). Zoom in and try again.",
        }) from exc

    return InundationExposureResponse(
        currency=currency,
        synthetic=True,
        note=("Estimated from synthetic location points distributed within counties "
              "from aggregate TIV — not real location-level data. The extent is "
              "modelled water rather than a warning area, so it is more precise than "
              "the alert layer, but the model is EXPERIMENTAL and excludes coastal "
              "flooding."),
        reaches=len(bundle.features),
        truncated=bundle.truncated,
        below_resolution=below_resolution,
        combined=exposure_out("inundation", "Modelled inundation", union),
        warnings=warnings,
        notes=(
            bundle.notes + [
                "The modelled extent is narrower than the synthetic location "
                "spacing, so this figure is below the resolution of the method: "
                "read it as unmeasurable, not as an absence of exposure. Use the "
                "alert layer for a bounded estimate here."
            ]
            if below_resolution else bundle.notes
        ),
    )


__all__ = ["router"]
