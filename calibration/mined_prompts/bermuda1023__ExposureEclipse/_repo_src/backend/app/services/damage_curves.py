"""Damage-ratio curve sets by segment, on the Form V-1 10 mph bands.

A curve set maps ``segment -> [DR at each band midpoint]`` (45.5, 55.5 …
165.5 mph), as decimals. DR between midpoints is linear; below the first
midpoint it ramps to zero at 39 mph.

The storm wind field is marine exposure: IBTrACS / NHC intensity is the
1-minute sustained wind over open water. Form V-1 curves are defined on wind
over land, so each set carries ``wind_factor``:

    curve_mph = marine_mph x wind_factor

Vendor sets are ground-up building damage. They also carry a deductible per
segment (fraction of TIV), applied per location with a beta spread of DR
around the mean, so the output is insured loss (the PCS basis). Calibrated
sets are fitted straight to insured loss, so their deductible is zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .industry_losses import LOSS_SEGMENTS, data_dir

BAND_MPH_LO: tuple[int, ...] = (41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161)
BAND_MID: tuple[float, ...] = tuple(lo + 4.5 for lo in BAND_MPH_LO)
ZERO_DAMAGE_MPH = 39.0

# Florida Commission Form V-1 Part A, building damage / building exposure (%).
# Same table as frontend/src/state/damageAssumptions.ts.
_FORM_V1_PCT: dict[str, tuple[float, ...]] = {
    "air": (0.05, 0.12, 0.46, 0.96, 1.89, 3.55, 7.34, 20.6, 32.8, 42.1, 54.2, 61.8, 71.7),
    "kcc": (0.11, 0.26, 0.73, 1.68, 3.79, 7.66, 14.6, 26.1, 32.6, 43.5, 52.4, 60.8, 72.1),
    "rms": (0.13, 0.33, 1.43, 3.22, 6.97, 14.7, 25.4, 44.6, 57.5, 76.6, 85.5, 90.0, 93.9),
}
_VENDOR_LABEL = {"air": "AIR (Form V-1)", "kcc": "KCC (Form V-1)", "rms": "RMS (Form V-1)"}

# ASSUMPTION: Form V-1 is residential building damage. Until calibrated,
# commercial uses the same curve and auto uses 15% of it. Form V-1 does not
# model vehicles; the scale is only an uncalibrated placeholder.
UNCALIBRATED_SEGMENT_SCALE: dict[str, float] = {"RESIDENTIAL": 1.0, "COMMERCIAL": 1.0, "AUTO": 0.15}

# ASSUMPTION: marine 1-minute sustained wind to wind over land. Published
# marine-to-open-terrain ratios run about 0.83-0.87; airport observations
# under landfalling eyewalls (Ian at Fort Myers and Punta Gorda, Michael)
# sit nearer 0.75. 0.80 is the middle of that range. Calibration replaces it.
DEFAULT_WIND_FACTOR = 0.80

# ASSUMPTION: deductible as a fraction of segment TIV. Residential: 2% of
# Coverage A hurricane deductible, Coverage A about 65% of HO TIV. Commercial:
# named-storm deductibles of 2-5% of insured value. Auto: comprehensive
# deductible on an average vehicle.
DEFAULT_DEDUCTIBLE: dict[str, float] = {"RESIDENTIAL": 0.013, "COMMERCIAL": 0.03, "AUTO": 0.03}

# ASSUMPTION: location DR around the county mean is Beta with precision 9
# (CV about 2 at a 2% mean, about 0.5 at 30%).
DEFAULT_BETA_PRECISION = 9.0

# ASSUMPTION: lognormal spread around the point estimate before any
# calibration exists. Replaced by fitted residual spreads in calibrated sets.
DEFAULT_SIGMA_COUNTY = 0.9
DEFAULT_SIGMA_EVENT = 0.45

CURVES_DIR = "damage_curves"


REF_MARINE_MPH = 64 * 1.15078


def land_mph(marine_mph: float, factor: float, exponent: float = 0.0, floor: float = 0.0) -> float:
    """Marine 1-minute wind to curve wind:
    ``factor x marine x max(floor, (marine / 64 kt) ** exponent)``.
    With exponent 0 this is the flat ``factor``; a fitted exponent makes the
    land share fall as the storm gets stronger, as the ASOS record shows."""
    if marine_mph <= 0:
        return 0.0
    shape = (marine_mph / REF_MARINE_MPH) ** exponent if exponent else 1.0
    return factor * marine_mph * max(floor, shape)


def interp_weights(curve_mph: float) -> list[tuple[int, float]]:
    """Linear weights on band midpoints so DR(mph) = sum(w x DR_b)."""
    if curve_mph <= ZERO_DAMAGE_MPH:
        return []
    if curve_mph <= BAND_MID[0]:
        return [(0, (curve_mph - ZERO_DAMAGE_MPH) / (BAND_MID[0] - ZERO_DAMAGE_MPH))]
    if curve_mph >= BAND_MID[-1]:
        return [(len(BAND_MID) - 1, 1.0)]
    i = int((curve_mph - BAND_MID[0]) // 10)
    t = (curve_mph - BAND_MID[i]) / (BAND_MID[i + 1] - BAND_MID[i])
    return [(i, 1.0 - t), (i + 1, t)]


def _beta_excess(mu: float, d: float, phi: float) -> float:
    """E[max(0, X - d)] for X ~ Beta(mean mu, precision phi)."""
    if d <= 0:
        return mu
    if mu <= 0:
        return 0.0
    if mu >= 1:
        return 1.0 - d
    a, b = mu * phi, (1.0 - mu) * phi
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    # Simpson on [d, 1). The integrand is finite at d; near 1 it can be
    # singular when b < 1, which only happens above a 89% mean.
    n = 400
    hi = 1.0 - 1e-9
    h = (hi - d) / n
    total = 0.0
    for k in range(n + 1):
        x = d + k * h
        val = (x - d) * math.exp(log_norm + (a - 1) * math.log(x) + (b - 1) * math.log1p(-x))
        total += val * (1 if k in (0, n) else (4 if k % 2 else 2))
    return max(0.0, min(mu, total * h / 3.0))


_MU_GRID: tuple[float, ...] = tuple(10 ** (-5 + 5 * k / 240) for k in range(241))


@lru_cache(maxsize=32)
def _excess_table(d: float, phi: float) -> tuple[float, ...]:
    return tuple(_beta_excess(min(mu, 0.999), d, phi) for mu in _MU_GRID)


def insured_dr(mu: float, deductible: float, phi: float = DEFAULT_BETA_PRECISION) -> float:
    """Mean insured DR for a ground-up mean ``mu`` after a per-location deductible."""
    if deductible <= 0 or mu <= 0:
        return max(0.0, mu)
    table = _excess_table(round(deductible, 6), round(phi, 4))
    if mu <= _MU_GRID[0]:
        return table[0] * mu / _MU_GRID[0]
    if mu >= _MU_GRID[-1]:
        return table[-1]
    x = (math.log10(mu) + 5) * 240 / 5
    i = min(int(x), len(_MU_GRID) - 2)
    t = x - i
    return table[i] * (1 - t) + table[i + 1] * t


@dataclass(slots=True)
class CurveSet:
    id: str
    label: str
    kind: str  # "vendor" | "calibrated"
    segments: dict[str, list[float]]  # segment -> DR at each BAND_MID (0..1)
    wind_factor: float = DEFAULT_WIND_FACTOR
    deductible: dict[str, float] = field(default_factory=dict)  # segment -> fraction of TIV
    beta_precision: float = DEFAULT_BETA_PRECISION
    sigma_log_county: float = DEFAULT_SIGMA_COUNTY
    sigma_log_event: float = DEFAULT_SIGMA_EVENT
    created_at: str | None = None
    fit: dict = field(default_factory=dict)
    # Observed-wind shape (services/wind_calibration.py). 0 / 0 = flat windFactor.
    wind_exponent: float = 0.0
    wind_floor: float = 0.0

    def land_mph(self, marine_mph: float, multiplier: float = 1.0) -> float:
        """Curve wind for a marine wind; ``multiplier`` is a county's observed correction."""
        return land_mph(marine_mph, self.wind_factor, self.wind_exponent, self.wind_floor) * multiplier

    def weights(self, marine_mph: float, multiplier: float = 1.0) -> list[tuple[int, float]]:
        return interp_weights(self.land_mph(marine_mph, multiplier))

    def ground_up_dr(self, segment: str, marine_mph: float, multiplier: float = 1.0) -> float:
        curve = self.segments.get(segment)
        if not curve:
            return 0.0
        return sum(w * curve[b] for b, w in self.weights(marine_mph, multiplier))

    def dr(self, segment: str, marine_mph: float, multiplier: float = 1.0) -> float:
        """Insured DR at one marine wind speed."""
        return insured_dr(
            self.ground_up_dr(segment, marine_mph, multiplier),
            self.deductible.get(segment, 0.0),
            self.beta_precision,
        )

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "bandsMphLo": list(BAND_MPH_LO),
            "segments": {seg: [round(v, 6) for v in vals] for seg, vals in self.segments.items()},
            "windFactor": self.wind_factor,
            "windExponent": self.wind_exponent,
            "windFloor": self.wind_floor,
            "deductible": self.deductible,
            "betaPrecision": self.beta_precision,
            "sigmaLogCounty": self.sigma_log_county,
            "sigmaLogEvent": self.sigma_log_event,
            "createdAt": self.created_at,
            "fit": self.fit,
        }


def vendor_curve(vendor: str) -> list[float]:
    return [p / 100.0 for p in _FORM_V1_PCT[vendor]]


def _vendor_set(vendor: str) -> CurveSet:
    base = vendor_curve(vendor)
    return CurveSet(
        id=vendor,
        label=_VENDOR_LABEL[vendor],
        kind="vendor",
        segments={seg: [v * UNCALIBRATED_SEGMENT_SCALE[seg] for v in base] for seg in LOSS_SEGMENTS},
        deductible=dict(DEFAULT_DEDUCTIBLE),
    )


def insured_prior(curve: CurveSet, segment: str) -> list[float]:
    """The set's curve after its deductible, at each band midpoint. Used as the
    prior for a fit, whose target is insured loss."""
    d = curve.deductible.get(segment, 0.0)
    return [
        max(1e-5, insured_dr(v, d, curve.beta_precision))
        for v in curve.segments[segment]
    ]


def curve_set_from_json(payload: dict) -> CurveSet:
    segments = {
        seg: [float(v) for v in vals]
        for seg, vals in (payload.get("segments") or {}).items()
        if seg in LOSS_SEGMENTS
    }
    for seg, vals in segments.items():
        if len(vals) != len(BAND_MPH_LO):
            raise ValueError(f"curve {payload.get('id')}: {seg} has {len(vals)} bands")
    return CurveSet(
        id=str(payload["id"]),
        label=str(payload.get("label") or payload["id"]),
        kind=str(payload.get("kind") or "calibrated"),
        segments=segments,
        wind_factor=float(payload.get("windFactor", DEFAULT_WIND_FACTOR)),
        wind_exponent=float(payload.get("windExponent", 0.0)),
        wind_floor=float(payload.get("windFloor", 0.0)),
        deductible={k: float(v) for k, v in (payload.get("deductible") or {}).items()},
        beta_precision=float(payload.get("betaPrecision", DEFAULT_BETA_PRECISION)),
        sigma_log_county=float(payload.get("sigmaLogCounty", DEFAULT_SIGMA_COUNTY)),
        sigma_log_event=float(payload.get("sigmaLogEvent", DEFAULT_SIGMA_EVENT)),
        created_at=payload.get("createdAt"),
        fit=payload.get("fit") or {},
    )


def curves_dir() -> Path:
    return data_dir() / CURVES_DIR


@lru_cache(maxsize=1)
def load_curve_sets() -> dict[str, CurveSet]:
    out = {v: _vendor_set(v) for v in ("kcc", "air", "rms")}
    folder = curves_dir()
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            curve = curve_set_from_json(json.loads(path.read_text(encoding="utf-8")))
            # Segments the calibration had no data for keep the insured KCC shape.
            for seg in LOSS_SEGMENTS:
                if seg not in curve.segments:
                    curve.segments[seg] = insured_prior(out["kcc"], seg)
            out[curve.id] = curve
    return out


def default_curve_set_id() -> str:
    """Newest calibrated set, else KCC."""
    calibrated = [c for c in load_curve_sets().values() if c.kind == "calibrated"]
    if not calibrated:
        return "kcc"
    return max(calibrated, key=lambda c: (c.created_at or "", c.id)).id


def get_curve_set(curve_set_id: str | None) -> CurveSet:
    sets = load_curve_sets()
    key = curve_set_id or default_curve_set_id()
    if key not in sets:
        raise KeyError(key)
    return sets[key]
