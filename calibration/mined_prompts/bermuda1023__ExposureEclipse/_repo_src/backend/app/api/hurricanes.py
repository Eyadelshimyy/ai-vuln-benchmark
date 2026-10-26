"""Historical Atlantic hurricane tracks (NOAA HURDAT2).

  GET /api/hurricanes?year_min=1950&year_max=2025&min_category=1&landfall_only=true

Filters:
  - year_min / year_max  → inclusive bounds on storm year
  - min_category         → Saffir-Simpson at landfall (1–5). Storms whose
                            strongest landfall < min_category are dropped.
  - landfall_only        → if true (default), drop storms that never made
                            landfall in any state we can detect.

Response is shaped for direct rendering as a Mapbox source.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from ..models.exposure import MapRequest
from ..providers import ExposureDataProvider, get_provider
from ..services.calculations import apply_filters
from ..services.export_excel import build_hurricane_impact_xlsx
from ..services.hurdat2 import category_for_wind, landfall_summary, peak_wind
from ..services.ibtracs import Storm, fetch_storms
from ..services.hurricane_impact import compute_impact, join_tiv


def _effective_category(landfall_cat: int, peak_cat: int) -> int:
    """The strength used for the min-category filter.

    With landfall (cat >= -1) we keep the landfall intensity. Without a
    detectable landfall (-2) we fall back to the storm's peak intensity over
    its lifetime — otherwise a Cat 5 storm that stayed at sea would compare
    as ``-2 < 1`` and get dropped from "Cat 1+" filters.
    """
    return landfall_cat if landfall_cat != -2 else peak_cat

router = APIRouter(prefix="/hurricanes", tags=["hurricanes"])


def _serialize(
    storm: Storm,
    landfall_cat: int,
    landfall_state: str | None,
    effective_cat: int,
) -> dict:
    return {
        "stormId": storm.storm_id,
        "name": storm.name,
        "year": storm.year,
        "landfallCategory": landfall_cat,  # -2 = no landfall, -1 = TD, 0 = TS, 1-5 = SS
        "landfallState": landfall_state,
        "peakWindKt": peak_wind(storm),
        # The category used by the strength filter — landfall if it had one,
        # else the storm's peak. Frontend uses this to decide colour for the
        # storm's overall classification in the tooltip.
        "effectiveCategory": effective_cat,
        "track": [
            {
                "lat": p.lat,
                "lon": p.lon,
                "windKt": p.wind_kt,
                "category": category_for_wind(p.wind_kt),
                "status": p.status,
                "datetime": p.datetime_utc,
                "isLandfall": "L" in p.record_id,
            }
            for p in storm.track
        ],
    }


@router.get("")
def list_hurricanes(
    year_min: int = Query(1950, ge=1950, le=2100, alias="yearMin"),
    year_max: int = Query(2100, ge=1950, le=2100, alias="yearMax"),
    min_category: int = Query(1, ge=-2, le=5, alias="minCategory"),
    landfall_only: bool = Query(True, alias="landfallOnly"),
    landfall_states: str | None = Query(
        default=None,
        alias="landfallStates",
        description="Comma-separated USPS state codes (e.g. 'LA,MS,AL'). Only storms whose strongest US landfall hit one of these states are returned. Implies landfallOnly=true.",
    ),
) -> dict:
    if year_max < year_min:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "yearMax must be >= yearMin.",
            },
        )

    state_filter: set[str] | None = None
    if landfall_states:
        state_filter = {
            s.strip().upper() for s in landfall_states.split(",") if s.strip()
        } or None

    try:
        storms = fetch_storms()
    except Exception as exc:  # noqa: BLE001 — surface NOAA fetch failures cleanly
        raise HTTPException(
            status_code=502,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Failed to fetch IBTrACS data from NOAA.",
                "details": {"error": str(exc)},
            },
        ) from exc

    out: list[dict] = []
    for s in storms:
        if s.year < year_min or s.year > year_max:
            continue
        cat, state = landfall_summary(s)
        if landfall_only and cat == -2:
            continue
        if state_filter is not None:
            # A state filter implicitly requires a US landfall.
            if state is None or state not in state_filter:
                continue
        peak_cat = category_for_wind(peak_wind(s))
        effective = _effective_category(cat, peak_cat)
        if effective < min_category:
            continue
        out.append(_serialize(s, cat, state, effective))

    return {
        "storms": out,
        "count": len(out),
        "filters": {
            "yearMin": year_min,
            "yearMax": year_max,
            "minCategory": min_category,
            "landfallOnly": landfall_only,
            "landfallStates": sorted(state_filter) if state_filter else None,
        },
    }


# ───────────────────────── impact endpoint ─────────────────────────
#
# POST /api/hurricanes/{storm_id}/impact
#
# Body shape mirrors MapRequest (programmeId | chainId | chainIds[] | cedentId
# | datasetId | datasetGroupId + filters + perils). Returns the set of US
# counties whose centroid falls within `multiplier × Rmax` of any on-land
# point in the storm's track, joined to TIV from the user's current selection.


def _compute_impact_payload(
    storm_id: str,
    payload: MapRequest,
    multiplier: float,
    provider: ExposureDataProvider,
) -> dict:
    """Shared impact computation — used by both the JSON and xlsx endpoints."""
    # Import here to avoid a circular dep with exposures router at module load.
    from .exposures import _apply_peril_filter
    from ..services.view_resolve import (
        resolve_view as _resolve_view,
        require_at_most_one_target as _require_exactly_one_target,
    )

    _require_exactly_one_target(payload)
    from ..services.live_hurricane import storm_for_impact

    storm = None
    track_source = "ibtracs"
    live = storm_for_impact(storm_id)
    if live is not None and live.track:
        storm = live
        track_source = "live"
    else:
        try:
            storms = fetch_storms()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "INTERNAL_ERROR",
                    "message": "Failed to fetch IBTrACS data from NOAA.",
                    "details": {"error": str(exc)},
                },
            ) from exc
        storm = next((s for s in storms if s.storm_id.upper() == storm_id.upper()), None)
    if storm is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DATASET_NOT_FOUND",
                "message": (
                    f"Storm '{storm_id}' is not in the live NHC feed or the "
                    "historical IBTrACS/HURDAT set."
                ),
                "details": {"stormId": storm_id},
            },
        )

    # Resolve the user's selection → fact rows, apply peril multi-select +
    # filter block (so the TIV in the impact panel matches the rest of the
    # workbench).
    resolved = _resolve_view(provider, payload)
    facts = _apply_peril_filter(resolved.facts, payload.perils)
    facts = apply_filters(facts, payload.filters)

    from ..services.wind_forecast import outer_wind_cap

    model_wind = None
    if track_source == "live":
        model_wind = outer_wind_cap(
            [(p.lat, p.lon) for p in storm.track if p.wind_kt >= 34]
        )
    impacts, footprint, cone, outer_cone, outer_rings = compute_impact(
        storm, multiplier=multiplier, model_wind=model_wind,
    )
    impacts = join_tiv(impacts, facts)

    total_tiv = sum(i.tiv for i in impacts)
    total_loc = sum(i.location_count for i in impacts)
    counties_with_data = sum(1 for i in impacts if i.has_data)

    # Bounding box of impacted-county centroids, padded ~0.3deg so fitBounds
    # leaves a comfortable margin around the wind footprint.
    bbox: list[float] | None = None
    if impacts:
        lats = [i.centroid_lat for i in impacts]
        lons = [i.centroid_lon for i in impacts]
        bbox = [min(lons) - 0.3, min(lats) - 0.3, max(lons) + 0.3, max(lats) + 0.3]

    return {
        "stormId": storm.storm_id,
        "stormName": storm.name,
        "year": storm.year,
        "trackSource": track_source,
        "currency": resolved.currency,
        "multiplier": multiplier,
        "bbox": bbox,  # [west, south, east, north] or null when no impact
        "footprint": [
            {
                "lat": fp.lat,
                "lon": fp.lon,
                "windKt": fp.wind_kt,
                "rmaxNm": round(fp.rmax_nm, 1),
                "radiusNm": round(fp.radius_nm, 1),
                "rmaxSource": fp.rmax_source,
                "r64Nm": round(fp.r64_nm, 1),
                "r64Source": fp.r64_source,
                # Per-quadrant R64 (NE, SE, SW, NW) in nautical miles; null
                # when no IBTrACS measurement — caller uses the symmetric r64Nm.
                "r64QuadsNm": (
                    [round(v, 1) for v in fp.r64_quads_nm]
                    if fp.r64_quads_nm is not None
                    else None
                ),
            }
            for fp in footprint
        ],
        # Asymmetric outer-footprint polygons (one per fix). Frontend renders
        # these directly so the "egg" matches the asymmetric outer cone seam.
        "outerFootprint": [
            {
                "corners": r["ring"],
                "windKt": r["wind_kt"],
                "r64Nm": round(r["r64_nm"], 1),
                "r64Source": r["r64_source"],
            }
            for r in outer_rings
        ],
        # Inner cone (Rmax / eyewall) and outer cone (R64 / hurricane-wind
        # extent). Each quad is a closed GeoJSON ring colored by wind speed.
        "cone": [
            {
                "corners": [
                    [round(lon, 4), round(lat, 4)] for (lon, lat) in q.corners
                ]
                + [[round(q.corners[0][0], 4), round(q.corners[0][1], 4)]],
                "windKt": q.wind_kt,
                "startWindKt": q.start_wind_kt,
                "endWindKt": q.end_wind_kt,
            }
            for q in cone
        ],
        "outerCone": [
            {
                "corners": [
                    [round(lon, 4), round(lat, 4)] for (lon, lat) in q.corners
                ]
                + [[round(q.corners[0][0], 4), round(q.corners[0][1], 4)]],
                "windKt": q.wind_kt,
                "startWindKt": q.start_wind_kt,
                "endWindKt": q.end_wind_kt,
            }
            for q in outer_cone
        ],
        "summary": {
            "countiesImpacted": len(impacts),
            "countiesWithData": counties_with_data,
            "totalTiv": total_tiv,
            "totalLocationCount": total_loc,
            # Informational. totalTiv already bundles every segment.
            "residentialTiv": sum(i.residential_tiv for i in impacts),
            "commercialTiv": sum(i.commercial_tiv for i in impacts),
            "autoTiv": sum(i.auto_tiv for i in impacts),
        },
        "counties": [
            {
                "geographyId": i.geography_id,
                "geoid": i.geoid,
                "name": i.name,
                "state": i.state_usps,
                "centroidLat": round(i.centroid_lat, 4),
                "centroidLon": round(i.centroid_lon, 4),
                "maxWindKt": i.max_wind_kt,
                "maxCategory": i.max_category,
                "closestDistanceNm": round(i.closest_distance_nm, 1),
                "rmaxAtClosestNm": round(i.rmax_at_closest_nm, 1),
                "rmaxSource": i.rmax_source,
                "windBands": [
                    {
                        "category": b.category,
                        "areaFraction": round(b.area_fraction, 4),
                        "maxWindKt": b.max_wind_kt,
                    }
                    for b in i.wind_bands
                    if b.area_fraction > 0
                ],
                "speedBins": [
                    {
                        "mphLo": b.mph_lo,
                        "mphHi": b.mph_hi,
                        "category": b.category,
                        "areaFraction": round(b.area_fraction, 4),
                        "maxWindKt": b.max_wind_kt,
                    }
                    for b in i.speed_bins
                    if b.area_fraction > 0
                ],
                "tiv": i.tiv,
                "residentialTiv": i.residential_tiv,
                "commercialTiv": i.commercial_tiv,
                "autoTiv": i.auto_tiv,
                "locationCount": i.location_count,
                "hasData": i.has_data,
                "byProgramme": [
                    {
                        "datasetId": p.dataset_id,
                        "tiv": p.tiv,
                        "locationCount": p.location_count,
                    }
                    for p in i.by_programme
                ],
            }
            for i in impacts
        ],
    }


@router.post("/{storm_id}/impact")
def hurricane_impact(
    storm_id: str,
    payload: MapRequest,
    multiplier: float = Query(2.5, ge=1.0, le=6.0, alias="multiplier"),
    provider: ExposureDataProvider = Depends(get_provider),
) -> dict:
    return _compute_impact_payload(storm_id, payload, multiplier, provider)


@router.post("/{storm_id}/impact/export")
def hurricane_impact_export(
    storm_id: str,
    payload: MapRequest,
    multiplier: float = Query(2.5, ge=1.0, le=6.0, alias="multiplier"),
    provider: ExposureDataProvider = Depends(get_provider),
) -> Response:
    """Download the impact result as an .xlsx workbook (Summary + counties)."""
    impact = _compute_impact_payload(storm_id, payload, multiplier, provider)
    xlsx = build_hurricane_impact_xlsx(impact)
    safe_name = "".join(c for c in (impact.get("stormName") or "storm") if c.isalnum()).lower() or "storm"
    filename = f"impact_{safe_name}_{impact.get('year')}_{storm_id}.xlsx"
    return Response(
        content=xlsx,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
