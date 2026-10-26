"""Industry loss for a storm, supplied actuals, and backtests.

  GET  /api/damage-curves
  POST /api/hurricanes/{storm_id}/industry-loss
  GET  /api/industry-losses
  GET  /api/industry-losses/{storm_id}/backtest?curveSetId=&multiplier=&observedWind=

Industry loss is independent of the left-rail selection: it always runs on
the industry TIV (supplied vintage for the storm year, else the built-in
Industry book).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..services.damage_curves import default_curve_set_id, get_curve_set, load_curve_sets
from ..services.event_evidence import load_evidence
from ..services.event_loss import event_loss, lognormal_band, resolve_storm, storm_impacts
from ..services.industry_losses import (
    LOSS_SEGMENTS,
    EventActuals,
    IndustryDataError,
    actuals_for,
    load_actuals,
    tiv_for_year,
)
from ..services.loss_calibration import backtest
from ..services.onlevel import onlevel_actuals
from ..services.wind_calibration import county_multipliers, shape_for

router = APIRouter(tags=["industry-loss"])


class IndustryLossRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    curve_set_id: str | None = None
    multiplier: float = Field(2.5, ge=1.0, le=6.0)
    # geoid or geographyId -> exposed fraction 0..1 (the impact panel overrides).
    exposed_fraction: dict[str, float] = Field(default_factory=dict)
    # Scale reported losses to the TIV valuation date (services/onlevel.py).
    on_level: bool = True
    # Correct a historical storm's wind county by county to its ASOS stations
    # (services/wind_calibration.py). Off by default: it tightens county fits
    # but widens event-total error.
    observed_wind: bool = False


def _observed_wind(storm, track_source: str, impacts, curve, wanted: bool) -> tuple[dict[str, float], dict]:
    """County wind multipliers from the storm's stations, and a wire summary.
    Only for IBTrACS storms with an evidence file and a curve fitted on the
    observed-wind shape; a live storm has no observations yet."""
    shape = shape_for(curve)
    info: dict = {"applied": False, "reason": None}
    if not wanted:
        info["reason"] = "switched off"
    elif track_source != "ibtracs":
        info["reason"] = "live storm"
    elif shape is None:
        info["reason"] = "curve set has no observed-wind shape"
    else:
        mult = county_multipliers(storm.storm_id, impacts, shape)
        if not mult:
            info["reason"] = "no station evidence for this storm"
        else:
            vals = sorted(mult.values())
            info = {
                "applied": True, "reason": None, "counties": len(vals),
                "median": vals[len(vals) // 2], "min": vals[0], "max": vals[-1],
            }
            return mult, info
    return {}, info


def _error(status: int, code: str, message: str, **details) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, "details": details})


def _curve(curve_set_id: str | None):
    try:
        return get_curve_set(curve_set_id)
    except KeyError as exc:
        raise _error(404, "VALIDATION_ERROR", f"Unknown curve set {curve_set_id!r}.",
                     available=sorted(load_curve_sets())) from exc


def _storm(storm_id: str):
    try:
        found = resolve_storm(storm_id)
    except Exception as exc:
        raise _error(502, "UPSTREAM_UNAVAILABLE", "Failed to fetch storm tracks.", error=str(exc)) from exc
    if found is None:
        raise _error(404, "DATASET_NOT_FOUND", f"Storm '{storm_id}' was not found.", stormId=storm_id)
    return found


def _actuals(storm_id: str):
    try:
        return actuals_for(storm_id)
    except IndustryDataError as exc:
        raise _error(422, "VALIDATION_ERROR", str(exc)) from exc


def _tiv(year: int):
    try:
        return tiv_for_year(year)
    except IndustryDataError as exc:
        raise _error(422, "VALIDATION_ERROR", str(exc)) from exc


def _tiv_source(tiv) -> dict:
    return {"kind": tiv.kind, "year": tiv.year, "file": tiv.file, "currency": tiv.currency,
            "asOfDate": tiv.as_of.isoformat() if tiv.as_of else None}


def _actual_basis(actuals: EventActuals) -> dict:
    """Reported totals plus what the figure does and does not include."""
    return {
        "total": actuals.total(),
        "bySegment": actuals.segment_totals(),
        "unallocated": actuals.unallocated_total(),
        "eventDate": actuals.event_date,
        "claimsBySegment": actuals.claims or None,
        "states": actuals.states(),
        "asOfDate": actuals.as_of,
        "source": actuals.source,
        "includesFlood": actuals.includes_flood,
        "includesSurge": actuals.includes_surge,
        "includesNfip": actuals.includes_nfip,
        "includesLae": actuals.includes_lae,
        "fx": actuals.fx,
    }


@router.get("/damage-curves")
def list_damage_curves() -> dict:
    return {
        "defaultId": default_curve_set_id(),
        "curveSets": [c.to_wire() for c in load_curve_sets().values()],
    }


@router.post("/hurricanes/{storm_id}/industry-loss")
def industry_loss(storm_id: str, payload: IndustryLossRequest | None = None) -> dict:
    req = payload or IndustryLossRequest()
    curve = _curve(req.curve_set_id)
    storm, track_source = _storm(storm_id)
    tiv = _tiv(storm.year)
    actuals = _actuals(storm.storm_id)
    impacts = storm_impacts(storm, track_source, req.multiplier)
    wind_mult, wind_info = _observed_wind(storm, track_source, impacts, curve, req.observed_wind)
    result = event_loss(impacts, tiv, curve, req.exposed_fraction, wind_mult)
    currency = tiv.currency

    notes: list[str] = []
    if wind_info["applied"]:
        notes.append(
            f"Wind corrected to airport observations: county multipliers "
            f"x{wind_info['min']:.2f} to x{wind_info['max']:.2f} (median x{wind_info['median']:.2f})."
        )
    if tiv.kind == "book":
        notes.append(
            "Using the built-in Industry book (supplied 2026 windstorm TIV by county, "
            "23 states). Supply industry_tiv/<year>.csv to use a storm-year vintage."
        )
    elif tiv.year != storm.year:
        notes.append(f"No {storm.year} TIV supplied; using the {tiv.year} vintage.")
    for seg in result.segments_without_tiv:
        notes.append(f"No {seg.lower()} TIV in this vintage, so no {seg.lower()} loss is modelled.")
    if result.missing_tiv:
        notes.append(f"{len(result.missing_tiv)} counties in the wind field have no industry TIV.")
    if curve.kind == "vendor":
        ded = ", ".join(f"{seg.lower()} {v:.1%}" for seg, v in curve.deductible.items())
        notes.append(
            f"Uncalibrated Form V-1 curve. Storm wind is marine exposure, taken to land at "
            f"x{curve.wind_factor:.2f}. Deductibles as a share of TIV: {ded}. Commercial uses the "
            "residential curve and auto 15% of it. The band is a default spread, not a fitted one."
        )
    if actuals is not None and actuals.currency != currency:
        raise _error(422, "CURRENCY_MISMATCH",
                     f"Actuals are {actuals.currency} but TIV is {currency}.")

    sig_c, sig_e = curve.sigma_log_county, curve.sigma_log_event
    by_seg = {
        seg: {
            "tiv": result.tiv_by_segment[seg],
            **lognormal_band(result.by_segment[seg], sig_e),
        }
        for seg in LOSS_SEGMENTS
    }
    actual_block = None
    if actuals is not None:
        on_level_info = None
        if req.on_level:
            actuals, on_level_info = onlevel_actuals(actuals, tiv.as_of)
        actual_block = {**_actual_basis(actuals), "onLevel": on_level_info}
        if on_level_info and on_level_info["factor"]:
            notes.append(
                f"Reported loss on-levelled x{on_level_info['factor']:.2f} from "
                f"{on_level_info['eventDate']} to the TIV valuation date {on_level_info['targetDate']} "
                "(county housing units, employment and payroll; construction cost; used-vehicle prices)."
            )
        if actuals.fx:
            fx = actuals.fx
            notes.append(
                f"Reported loss was supplied in {fx['reportedCurrency']} and converted to "
                f"{actuals.currency} at {fx['fxRate']:g} {actuals.currency}/{fx['reportedCurrency']}"
                + (f" ({fx['fxSource']}, {fx['fxDate']})" if fx.get("fxSource") else "") + "."
            )
        unalloc = actuals.unallocated_total()
        if unalloc > 0.01 * actuals.total():
            notes.append(
                f"{unalloc / actuals.total():.0%} of reported loss is not placed in a county; it "
                "counts in the totals but not in the county backtest."
            )
        if actuals.includes_flood or actuals.includes_surge:
            notes.append(
                "Reported loss includes private-market flood and surge; the modelled loss is wind only."
            )

    return {
        "stormId": storm.storm_id,
        "stormName": storm.name,
        "year": storm.year,
        "trackSource": track_source,
        "currency": currency,
        "multiplier": req.multiplier,
        "curveSet": {"id": curve.id, "label": curve.label, "kind": curve.kind,
                     "windFactor": curve.wind_factor, "windExponent": curve.wind_exponent,
                     "windFloor": curve.wind_floor, "deductible": curve.deductible,
                     "sigmaLogCounty": sig_c, "sigmaLogEvent": sig_e},
        "tivSource": _tiv_source(tiv),
        "formula": (
            "loss = exposedFraction x TIV[segment] x sum_bands(areaFraction x "
            "insuredDR[segment](windFactor x marineMph x max(windFloor, (marineMph/73.6)^windExponent)"
            " x observedWindMultiplier[county]))"
        ),
        "observedWind": wind_info,
        "summary": {
            "total": lognormal_band(result.total, sig_e),
            "bySegment": by_seg,
            "countiesWithLoss": sum(1 for c in result.counties if c.total > 0),
        },
        "actuals": actual_block,
        "counties": [
            {
                "geographyId": c.impact.geography_id,
                "geoid": c.impact.geoid,
                "name": c.impact.name,
                "state": c.impact.state_usps,
                "maxWindKt": c.impact.max_wind_kt,
                "observedWindMultiplier": wind_mult.get(c.impact.geography_id),
                "exposedFraction": c.exposed_fraction,
                "segments": {
                    seg: {"tiv": c.tiv[seg], "loss": c.loss[seg], "dr": c.dr[seg]}
                    for seg in LOSS_SEGMENTS
                },
                **lognormal_band(c.total, sig_c),
            }
            for c in result.counties
            if c.total > 0
        ],
        "notes": notes,
    }


@router.get("/industry-losses")
def list_industry_losses() -> dict:
    try:
        events = load_actuals()
    except IndustryDataError as exc:
        raise _error(422, "VALIDATION_ERROR", str(exc)) from exc
    return {
        "events": [
            {
                "stormId": e.storm_id,
                "currency": e.currency,
                **_actual_basis(e),
                "counties": len({gid for gid, _ in e.county}),
                "files": e.files,
            }
            for e in sorted(events.values(), key=lambda ev: ev.storm_id)
        ]
    }


@router.get("/industry-losses/{storm_id}/backtest")
def industry_loss_backtest(
    storm_id: str,
    curve_set_id: str | None = Query(None, alias="curveSetId"),
    multiplier: float = Query(2.5, ge=1.0, le=6.0),
    on_level: bool = Query(True, alias="onLevel"),
    observed_wind: bool = Query(False, alias="observedWind"),
) -> dict:
    actuals = _actuals(storm_id)
    if actuals is None:
        raise _error(404, "DATASET_NOT_FOUND", f"No industry losses supplied for '{storm_id}'.",
                     stormId=storm_id)
    curve = _curve(curve_set_id)
    storm, track_source = _storm(storm_id)
    tiv = _tiv(storm.year)
    if actuals.currency != tiv.currency:
        raise _error(422, "CURRENCY_MISMATCH",
                     f"Actuals are {actuals.currency} but TIV is {tiv.currency}.")
    impacts = storm_impacts(storm, track_source, multiplier)
    on_level_info = None
    if on_level:
        actuals, on_level_info = onlevel_actuals(actuals, tiv.as_of)
    wind_mult, wind_info = _observed_wind(storm, track_source, impacts, curve, observed_wind)
    result = backtest(actuals, impacts, tiv, curve, load_evidence(storm.storm_id), wind_mult)
    return {
        "stormId": storm.storm_id,
        "stormName": storm.name,
        "year": storm.year,
        "currency": tiv.currency,
        "curveSet": {"id": curve.id, "label": curve.label, "kind": curve.kind},
        "tivSource": _tiv_source(tiv),
        "onLevel": on_level_info,
        "observedWind": wind_info,
        **result,
    }
