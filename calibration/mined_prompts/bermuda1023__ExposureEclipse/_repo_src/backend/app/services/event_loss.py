"""Industry loss for one storm: county wind mix x segment TIV x damage curve.

    county_loss[seg] = exposed_fraction x tiv[seg]
                       x sum_b(areaFraction_b x insuredDR[seg](land(mph_b) x obs_c))

``areaFraction_b`` is the share of the county polygon whose peak local wind
sits in 10 mph band ``b`` (``CountyImpact.speed_bins``); ``mph_b`` is the
band midpoint in marine exposure. ``windFactor`` takes it to wind over land
and the deductible turns ground-up into insured DR (``damage_curves``);
``land`` is the curve set's wind transform and ``obs_c`` the county's
observed-wind correction for a historical storm (``wind_calibration``). The point estimate
is treated as the mean of a lognormal; P10/P50/P90 use the curve set's
county or event log-spread. Event bands are NOT the sum of county bands
(that would assume every county errs the same way).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from .damage_curves import CurveSet
from .hurricane_impact import CountyImpact, compute_impact
from .ibtracs import Storm, fetch_storms
from .industry_losses import LOSS_SEGMENTS, IndustryTiv

Z90 = 1.2815515655446004


def lognormal_band(mean: float, sigma: float) -> dict[str, float]:
    if mean <= 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    sigma = max(0.0, sigma)
    median = mean * math.exp(-0.5 * sigma * sigma)
    return {
        "mean": mean,
        "p10": median * math.exp(-Z90 * sigma),
        "p50": median,
        "p90": median * math.exp(Z90 * sigma),
    }


KT_TO_MPH = 1.15078


def county_bins(impact: CountyImpact) -> list[tuple[float, float]]:
    """(marine mph, area_fraction) for one county, at each 10 mph band midpoint.
    The open-ended top band uses its strongest sample. Centroid-only impacts
    use the county's max wind."""
    out: list[tuple[float, float]] = []
    if impact.speed_bins:
        for b in impact.speed_bins:
            if b.area_fraction <= 0 or b.mph_lo <= 0:
                continue
            mph = b.mph_lo + 4.5
            if b.mph_lo >= 161:
                mph = max(mph, b.max_wind_kt * KT_TO_MPH)
            out.append((mph, b.area_fraction))
        return out
    if impact.max_wind_kt >= 34:
        return [(impact.max_wind_kt * KT_TO_MPH, 1.0)]
    return []


def effective_dr(
    bins: list[tuple[float, float]], curve: CurveSet, segment: str, wind_multiplier: float = 1.0,
) -> float:
    """Insured damage ratio on the whole county TIV for this wind mix."""
    return sum(frac * curve.dr(segment, mph, wind_multiplier) for mph, frac in bins)


@dataclass(slots=True)
class CountyLoss:
    impact: CountyImpact
    tiv: dict[str, float]
    loss: dict[str, float]
    dr: dict[str, float]
    exposed_fraction: float = 1.0

    @property
    def total(self) -> float:
        return sum(self.loss.values())


@dataclass(slots=True)
class EventLoss:
    counties: list[CountyLoss]
    by_segment: dict[str, float]
    tiv_by_segment: dict[str, float]
    missing_tiv: list[str] = field(default_factory=list)
    segments_without_tiv: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.by_segment.values())


def event_loss(
    impacts: list[CountyImpact],
    tiv: IndustryTiv,
    curve: CurveSet,
    exposed_fraction: dict[str, float] | None = None,
    wind_multiplier: dict[str, float] | None = None,
) -> EventLoss:
    """``wind_multiplier``: observed-wind correction per county
    (``wind_calibration.county_multipliers``), historical storms only."""
    overrides = exposed_fraction or {}
    wind_mult = wind_multiplier or {}
    counties: list[CountyLoss] = []
    by_seg = {seg: 0.0 for seg in LOSS_SEGMENTS}
    tiv_seg = {seg: 0.0 for seg in LOSS_SEGMENTS}
    missing: list[str] = []
    for imp in impacts:
        segs = tiv.county.get(imp.geography_id)
        if not segs:
            missing.append(imp.geography_id)
            continue
        scale = overrides.get(imp.geoid, overrides.get(imp.geography_id, 1.0))
        scale = max(0.0, min(1.0, float(scale)))
        bins = county_bins(imp)
        loss: dict[str, float] = {}
        dr: dict[str, float] = {}
        seg_tiv: dict[str, float] = {}
        for seg in LOSS_SEGMENTS:
            value = float(segs.get(seg, 0.0))
            seg_tiv[seg] = value
            ratio = effective_dr(bins, curve, seg, wind_mult.get(imp.geography_id, 1.0))
            dr[seg] = ratio
            loss[seg] = value * ratio * scale
            by_seg[seg] += loss[seg]
            tiv_seg[seg] += value
        counties.append(CountyLoss(impact=imp, tiv=seg_tiv, loss=loss, dr=dr, exposed_fraction=scale))
    counties.sort(key=lambda c: -c.total)
    present = tiv.segments_present()
    return EventLoss(
        counties=counties,
        by_segment=by_seg,
        tiv_by_segment=tiv_seg,
        missing_tiv=missing,
        segments_without_tiv=[s for s in LOSS_SEGMENTS if s not in present],
    )


# ─────────────────────────────── storm + impact ───────────────────────────────


def resolve_storm(storm_id: str) -> tuple[Storm, str] | None:
    """Live NHC storm first, then IBTrACS history. Returns (storm, track_source)."""
    from .live_hurricane import storm_for_impact

    live = storm_for_impact(storm_id)
    if live is not None and live.track:
        return live, "live"
    storm = next((s for s in fetch_storms() if s.storm_id.upper() == storm_id.upper()), None)
    if storm is None:
        return None
    return storm, "ibtracs"


@lru_cache(maxsize=64)
def _historical_impacts(storm_id: str, multiplier: float) -> tuple[CountyImpact, ...]:
    storm = next(s for s in fetch_storms() if s.storm_id.upper() == storm_id.upper())
    impacts, *_ = compute_impact(storm, multiplier=multiplier, ts_field=True)
    return tuple(impacts)


def storm_impacts(storm: Storm, track_source: str, multiplier: float) -> list[CountyImpact]:
    """County wind mix for the storm. Historical results are cached; live ones are not."""
    if track_source == "ibtracs":
        return list(_historical_impacts(storm.storm_id, multiplier))
    from .wind_forecast import outer_wind_cap

    model_wind = outer_wind_cap([(p.lat, p.lon) for p in storm.track if p.wind_kt >= 34])
    impacts, *_ = compute_impact(storm, multiplier=multiplier, model_wind=model_wind)
    return impacts
