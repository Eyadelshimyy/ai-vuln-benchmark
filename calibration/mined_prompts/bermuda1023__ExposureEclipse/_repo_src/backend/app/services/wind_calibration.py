"""Calibrate the modelled wind field to airport observations.

Two layers, both from the ASOS stations in ``mockdata/event_evidence/``:

1. **Land-wind shape** (every storm, live included). Pooled over the 15 PCS
   storms, observed land wind / modelled marine wind falls as the storm gets
   stronger: about 0.64 at 34 kt, 0.56 at 50 kt, 0.51 at 64 kt, then flat at
   0.47 from about 83 kt (2,126 stations: A = 0.513, b = -0.358). It is fitted
   as ``ratio = A x (marine / 64 kt) ** b``, floored where the strong-wind
   stations stop falling. The damage curve keeps its axis
   (``windFactor`` x marine at 64 kt), so only the shape is taken:

       curve_mph = windFactor x marine x max(floor, (marine / 64 kt) ** b)
       floor     = FLOOR_RATIO / A

2. **County correction** (historical storms with evidence). The log residual
   ``log(observed / shaped model)`` at each station is shrunk to a storm mean,
   and the station-to-mean differences are smoothed onto county centroids
   with a Gaussian kernel. A county far from any station gets the storm mean;
   a storm with few stations gets almost nothing.

Observed 1-minute land wind at a station is 1.05 x the ASOS 2-minute
sustained peak (ASSUMPTION: the usual 2-to-1-minute factor). A gust-based
estimate (gust / 1.4) was tried and fitted the PCS losses slightly worse.
Stations flagged ``probableOutage`` are floors, not peaks, and are left out.

The county correction tightens the county-level fit (holdout county log RMS
1.79 -> 1.69 at half strength) but makes event totals worse (loss-weighted
event log RMS 0.82 -> 0.91): wind corrections of +/-20% move loss about
2.5x through the curve, and station detail is not yet more predictive of
PCS loss than the smooth field. It is therefore opt-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .event_evidence import load_evidence
from .hurricane_impact import CountyImpact

SUSTAINED_TO_1MIN = 1.05
REF_KT = 64.0
MIN_MODEL_KT = 34.0
# The shape stops falling at the median observed / modelled ratio of stations
# with at least FLOOR_MIN_MODEL_KT modelled (37 stations in the 15 storms):
# the strong-wind ratio stops falling there, and a power law would keep
# cutting the cores, where failed anemometers already bias the record low.
FLOOR_MIN_MODEL_KT = 83.0
FLOOR_MIN_STATIONS = 20
FLOOR_RATIO = 0.45
# ASSUMPTION: smoothing constants picked by leave-one-station-out error over
# the 15 storms (0.36 log RMS with a flat 0.80 factor, 0.28 with the shape,
# 0.24 with this correction). Storm mean shrinks as n / (n + 5); a county
# needs about one station within 40 km to move off the storm mean.
STORM_SHRINK_N = 5.0
KERNEL_KM = 40.0
KERNEL_PRIOR = 1.0
MAX_LOG_CORRECTION = 0.5


def observed_kt(station: dict) -> float:
    return SUSTAINED_TO_1MIN * float(station.get("obsMaxSustainedKt") or 0.0)


def usable(station: dict) -> bool:
    return (
        not station.get("probableOutage")
        and float(station.get("modelWindKt") or 0.0) >= MIN_MODEL_KT
        and observed_kt(station) > 0
    )


@dataclass(slots=True, frozen=True)
class LandShape:
    ratio_at_ref: float  # A: observed / modelled at 64 kt
    exponent: float      # b
    floor_ratio: float = FLOOR_RATIO
    n_stations: int = 0

    @property
    def floor(self) -> float:
        """Floor on the shape term, relative to the 64 kt ratio."""
        return self.floor_ratio / self.ratio_at_ref

    def ratio(self, marine_kt: float) -> float:
        return self.ratio_at_ref * max(self.floor, (marine_kt / REF_KT) ** self.exponent)


def fit_land_shape(stations: list[dict]) -> LandShape:
    """Least squares on log(observed / modelled) = log A + b log(modelled / 64 kt)."""
    pts = [
        (math.log(float(s["modelWindKt"]) / REF_KT), math.log(observed_kt(s) / float(s["modelWindKt"])))
        for s in stations if usable(s)
    ]
    if len(pts) < 10:
        raise ValueError(f"only {len(pts)} usable stations")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else 0.0
    strong = sorted(
        observed_kt(s) / float(s["modelWindKt"])
        for s in stations if usable(s) and float(s["modelWindKt"]) >= FLOOR_MIN_MODEL_KT
    )
    floor = strong[len(strong) // 2] if len(strong) >= FLOOR_MIN_STATIONS else FLOOR_RATIO
    return LandShape(ratio_at_ref=math.exp(my - b * mx), exponent=b, floor_ratio=floor, n_stations=n)


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.radians
    a = (math.sin(p(lat2 - lat1) / 2) ** 2
         + math.cos(p(lat1)) * math.cos(p(lat2)) * math.sin(p(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


@dataclass(slots=True)
class StormCorrection:
    storm_mean: float                  # shrunk mean log residual
    stations: list[tuple[float, float, float]]  # (lat, lon, residual - mean)

    def at(self, lat: float, lon: float) -> float:
        num = den = 0.0
        for s_lat, s_lon, dev in self.stations:
            d = _km(lat, lon, s_lat, s_lon)
            if d > 4 * KERNEL_KM:
                continue
            k = math.exp(-0.5 * (d / KERNEL_KM) ** 2)
            num += k * dev
            den += k
        c = self.storm_mean + num / (den + KERNEL_PRIOR)
        return max(-MAX_LOG_CORRECTION, min(MAX_LOG_CORRECTION, c))


def storm_correction(stations: list[dict], shape: LandShape) -> StormCorrection | None:
    res = [
        (float(s["lat"]), float(s["lon"]),
         math.log(observed_kt(s) / (float(s["modelWindKt"]) * shape.ratio(float(s["modelWindKt"])))))
        for s in stations if usable(s)
    ]
    if not res:
        return None
    mean = sum(r for *_, r in res) / (len(res) + STORM_SHRINK_N)
    return StormCorrection(mean, [(lat, lon, r - mean) for lat, lon, r in res])


@lru_cache(maxsize=32)
def _correction(storm_id: str, ratio_at_ref: float, exponent: float, floor_ratio: float) -> StormCorrection | None:
    doc = load_evidence(storm_id)
    if not doc:
        return None
    return storm_correction(doc.get("stations") or [], LandShape(ratio_at_ref, exponent, floor_ratio))


def county_multipliers(
    storm_id: str, impacts: list[CountyImpact], shape: LandShape, strength: float = 1.0,
) -> dict[str, float]:
    """Observed-wind multiplier per county for a historical storm; empty
    when the storm has no evidence file. ``strength`` scales the log correction."""
    corr = _correction(storm_id.upper(), shape.ratio_at_ref, shape.exponent, shape.floor_ratio)
    if corr is None:
        return {}
    return {
        imp.geography_id: math.exp(strength * corr.at(imp.centroid_lat, imp.centroid_lon))
        for imp in impacts
    }


def shape_for(curve) -> LandShape | None:
    """The observed-wind shape a calibrated curve set was fitted with."""
    raw = (getattr(curve, "fit", None) or {}).get("windShape")
    if not raw:
        return None
    return LandShape(
        ratio_at_ref=float(raw["ratioAtRef"]),
        exponent=float(raw["exponent"]),
        floor_ratio=float(raw.get("floorRatio", FLOOR_RATIO)),
        n_stations=int(raw.get("nStations", 0)),
    )


def clear_caches() -> None:
    _correction.cache_clear()
