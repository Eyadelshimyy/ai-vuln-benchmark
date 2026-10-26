"""Live wildfire endpoint — real burn-area perimeters + satellite heat.

GET /api/wildfire/active
    ?bbox=west,south,east,north   (optional; default CONUS)
    &dayRange=3                   (FIRMS look-back days, 1-30; chained ≤5-day windows)
    &includeHeat=true             (fetch NASA FIRMS active-fire pixels)

Returns:
  - perimeters   GeoJSON FeatureCollection of current WFIGS fire polygons
                 (properties: name, acres, percentContained, cause, state, …)
  - activeFires  satellite thermal detections (VIIRS/MODIS) as points
  - affectedStates  per-state roll-up (join key to the exposure/TIV plane)
  - notes        degraded-layer explanations (e.g. FIRMS key missing)

Live overlay, like /live/storms — not part of the mock data plane. Sources:
NIFC/WFIGS (perimeters) + NASA FIRMS (heat).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..models.common import CamelModel
from ..services.live_wildfire import build_wildfire_bundle
from ..services import wildfire_exposure
from .geometry_input import ExposureRequest, PolygonExposureOut, exposure_out

router = APIRouter(prefix="/wildfire", tags=["wildfire"])


def _shape_id(geometry: dict) -> str:
    """Identity derived from the footprint itself, never its list position.

    Heat shapes are sorted by detection count and the layer refetches every few
    minutes, so a positional id silently re-points at a different fire: clicking
    a shape would deselect whichever fire happened to hold that index before,
    changing the combined TIV with no visible cause. Hashing the geometry means
    the id only changes when the footprint does, and a footprint that has grown
    simply reads as a new shape.
    """
    payload = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    return "heat-" + hashlib.sha1(payload.encode()).hexdigest()[:12]


# ─────────────────────────── wire types ───────────────────────────


class ActiveFireOut(CamelModel):
    lat: float
    lon: float
    brightness_k: float | None
    frp_mw: float | None
    confidence: str | None
    satellite: str
    source: str
    acquired_at: str


class AffectedStateOut(CamelModel):
    state: str
    fire_count: int
    acres: float


class WildfireCounts(CamelModel):
    perimeters: int
    active_fires: int
    active_fires_total: int
    heat_shapes: int


class WildfireAttribution(CamelModel):
    perimeters: str = "NIFC / WFIGS Interagency Perimeters (Current)"
    perimeters_url: str = (
        "https://data-nifc.opendata.arcgis.com/datasets/nifc::wfigs-interagency-perimeters-current"
    )
    active_fires: str = "NASA FIRMS (VIIRS/MODIS active fire, NRT)"
    active_fires_url: str = "https://firms.modaps.eosdis.nasa.gov/"


class WildfireResponse(CamelModel):
    generated_at: str
    bbox: list[float] | None
    day_range: int
    perimeters: dict     # GeoJSON FeatureCollection (official WFIGS)
    heat_shapes: dict    # GeoJSON FeatureCollection (built from FIRMS clusters)
    active_fires: list[ActiveFireOut]
    affected_states: list[AffectedStateOut]
    counts: WildfireCounts
    notes: list[str]
    attribution: WildfireAttribution


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


# ─────────────────────────── route ───────────────────────────


@router.get("/active", response_model=WildfireResponse)
def get_active_wildfire(
    bbox: str | None = Query(default=None, description="west,south,east,north (lon/lat)"),
    day_range: int = Query(default=3, ge=1, le=30, alias="dayRange"),  # chained ≤5d FIRMS windows
    include_heat: bool = Query(default=True, alias="includeHeat"),
    include_perimeters: bool = Query(default=True, alias="includePerimeters"),
    simplify: float = Query(
        default=0.005, ge=0.0, le=0.05,
        description="Perimeter generalisation in degrees (0 = full resolution).",
    ),
    min_cells: int = Query(default=2, ge=1, le=20, alias="minCells"),
    min_detections: int = Query(default=4, ge=1, le=200, alias="minDetections"),
    min_confidence: str = Query(default="nominal", alias="minConfidence"),
    min_frp: float = Query(default=0.0, ge=0.0, alias="minFrp"),
) -> WildfireResponse:
    """Assemble and return the current live wildfire bundle.

    Fetches NIFC/WFIGS burn-area perimeters and (optionally) NASA FIRMS
    satellite thermal detections, clusters detections into heat-shape footprints,
    and rolls up perimeter coverage by point-of-origin state.

    Both upstream sources fail soft — if either is unreachable the endpoint
    returns whatever it obtained plus an explanatory entry in ``notes[]``.

    See docs/CALCULATIONS.md §Live wildfire overlay for the clustering and
    footprint-tracing algorithms.
    See docs/API.md §GET /api/wildfire/active for the full parameter reference.

    Args:
        bbox: Bounding box as ``"west,south,east,north"`` (lon/lat, EPSG:4326).
            Omit for the CONUS default (-125.0, 24.0, -66.5, 50.0).
        day_range: FIRMS look-back in days (1–30). Internally chained as
            consecutive ≤5-day windows (FIRMS API cap).
        include_heat: When False, skip the FIRMS fetch entirely (faster; returns
            perimeters + affected-state roll-up only).
        include_perimeters: When False, skip the WFIGS fetch.
        simplify: Server-side generalisation tolerance in degrees
            (``maxAllowableOffset`` sent to ArcGIS). 0.005° ≈ 550 m reduces
            payload ~70× with negligible shape error at mapping scales.
        min_cells: Minimum distinct 0.02°-grid cells a detection cluster must
            occupy to be kept. Removes point sources (factories, gas flares).
        min_detections: Minimum FIRMS detections per cluster.
        min_confidence: Drop detections below this VIIRS/MODIS confidence band
            (``low | nominal | high``).
        min_frp: Drop detections with fire radiative power below this value (MW).

    Returns:
        WildfireResponse containing the perimeter FeatureCollection, heat-shape
        FeatureCollection, individual detection points (thinned to 8 000 max
        for rendering), affected-state roll-up, counts, notes, and attribution.

    Raises:
        HTTPException 422: If ``bbox`` is malformed or ``minConfidence`` is
            not one of ``low | nominal | high``.
    """
    box = _parse_bbox(bbox)
    if min_confidence not in ("low", "nominal", "high"):
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR",
            "message": "minConfidence must be low, nominal, or high.",
        })
    settings = get_settings()
    bundle = build_wildfire_bundle(
        map_key=settings.firms_map_key,
        bbox=box,
        day_range=day_range,
        include_heat=include_heat,
        simplify_deg=simplify,
        min_cells=min_cells,
        min_detections=min_detections,
        min_confidence=min_confidence,
        min_frp=min_frp,
        include_perimeters=include_perimeters,
    )

    features = [
        {
            "type": "Feature",
            "id": fp.incident_id,
            "geometry": fp.geometry,
            "properties": {
                "incidentId": fp.incident_id,
                "name": fp.name,
                "gisAcres": fp.gis_acres,
                "incidentSizeAcres": fp.incident_size_acres,
                "percentContained": fp.percent_contained,
                "cause": fp.cause,
                "discoveryAt": fp.discovery_at,
                "perimeterUpdatedAt": fp.perimeter_updated_at,
                "state": fp.state,
            },
        }
        for fp in bundle.perimeters
    ]

    heat_features = [
        {
            "type": "Feature",
            "id": i,
            "geometry": hs.geometry,
            "properties": {
                # Stable handle for click-to-select — see _shape_id. Without it
                # the UI derived a selection id from the cursor position, so
                # clicking the same shape twice selected it twice.
                "shapeId": _shape_id(hs.geometry),
                "detectionCount": hs.detection_count,
                "maxFrpMw": hs.max_frp_mw,
                "sumFrpMw": hs.sum_frp_mw,
                "firstDetectedAt": hs.first_detected_at,
                "lastDetectedAt": hs.last_detected_at,
            },
        }
        for i, hs in enumerate(bundle.heat_shapes)
    ]

    return WildfireResponse(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bbox=list(box) if box else None,
        day_range=day_range,
        perimeters={"type": "FeatureCollection", "features": features},
        heat_shapes={"type": "FeatureCollection", "features": heat_features},
        active_fires=[
            ActiveFireOut(
                lat=a.lat, lon=a.lon, brightness_k=a.brightness_k, frp_mw=a.frp_mw,
                confidence=a.confidence, satellite=a.satellite, source=a.source,
                acquired_at=a.acquired_at,
            )
            for a in bundle.active_fires
        ],
        affected_states=[
            AffectedStateOut(state=st, fire_count=n, acres=ac)
            for (st, n, ac) in bundle.affected_states
        ],
        counts=WildfireCounts(
            perimeters=len(bundle.perimeters),
            active_fires=len(bundle.active_fires),
            active_fires_total=bundle.detections_total,
            heat_shapes=len(bundle.heat_shapes),
        ),
        notes=bundle.notes,
        attribution=WildfireAttribution(),
    )


# ─────────────────── exposed TIV inside fire polygons ───────────────────


class WildfireExposureResponse(CamelModel):
    currency: str
    synthetic: bool
    note: str
    results: list[PolygonExposureOut]
    combined: PolygonExposureOut
    warnings: list[str]


@router.post("/exposure", response_model=WildfireExposureResponse)
def post_wildfire_exposure(req: ExposureRequest) -> WildfireExposureResponse:
    """Roll up exposed TIV by client for the supplied fire polygons.

    ``results`` holds one entry per polygon. ``combined`` is the union across
    all of them with each location counted once — selecting an official
    perimeter and the heat shape over the same fire is normal, and summing the
    per-polygon totals would double-count the overlap.

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

    return WildfireExposureResponse(
        currency=currency,
        synthetic=True,
        note=("Estimated from synthetic location points distributed within counties "
              "from aggregate TIV — not real location-level data. Replace the location "
              "source with individual-location exposure for exact in-perimeter TIV."),
        results=results,
        combined=combined,
        warnings=warnings,
    )


__all__ = ["router"]
