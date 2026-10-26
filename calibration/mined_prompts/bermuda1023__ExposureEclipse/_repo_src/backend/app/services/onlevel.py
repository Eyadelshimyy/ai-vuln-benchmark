"""On-level reported industry losses to the valuation date of the industry TIV.

A PCS county loss is event-year dollars on event-year exposure. Comparing it
with (or fitting a damage ratio against) a later TIV vintage needs the loss
the same event would cause on that later exposure::

    onLevelLoss = reportedLoss x insuredValue(target) / insuredValue(event)

The damage ratio itself is not trended: it is what calibration learns. The
insured-value growth per county and segment comes from public indices baked
by ``backend/scripts/build_onlevel_indices.py`` into ``mockdata/onlevel/``:

  RESIDENTIAL  units x replacement cost
               = HU x (m x PPI residential inputs + (1-m) x state construction pay)
               Contents and ALE are a share of Coverage A, so they follow it.
  COMMERCIAL   b x EMP x (m x PPI nonresidential inputs + (1-m) x construction pay)
               + (1-b) x county private payroll (contents + BI are declared values)
  AUTO         POP x CPI used cars and trucks (comprehensive pays actual cash value)

House prices and GDP deflators are deliberately not used: land is not insured,
and insured value is replacement cost.

Each component falls back county -> state -> flat (1.0) and the level used is
reported, so every factor is traceable.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from datetime import date
from functools import lru_cache
from itertools import pairwise

from . import industry_losses
from .industry_losses import EventActuals

ONLEVEL_DIR = "onlevel"

# ASSUMPTION: materials vs labour share of replacement cost (RSMeans-style
# square-foot estimates put materials at roughly half to 60%).
MATERIAL_SHARE = 0.55
# ASSUMPTION: commercial TIV is 62% building, 38% contents + BI, the split in
# the built-in Industry book.
COMMERCIAL_BUILDING_SHARE = 0.62
# Below this many private employees a county's employment ratio is too noisy;
# use the state's.
MIN_COUNTY_EMPLOYMENT = 500
# Annual series are extrapolated at their last growth rate by at most this
# many years past the newest point.
MAX_EXTRAPOLATION_YEARS = 1.5


@dataclass(frozen=True, slots=True)
class Indices:
    annual: dict[tuple[str, str], dict[int, float]]  # (geographyId, metric) -> year -> value
    monthly: dict[str, dict[str, float]]  # YYYY-MM -> key -> value
    months: tuple[str, ...]
    sources: dict


@lru_cache(maxsize=1)
def load_indices() -> Indices | None:
    folder = industry_losses.data_dir() / ONLEVEL_DIR
    county_path, monthly_path = folder / "county_indices.csv", folder / "monthly_indices.csv"
    if not county_path.exists() or not monthly_path.exists():
        return None
    annual: dict[tuple[str, str], dict[int, float]] = {}
    with county_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        header = [c.strip() for c in next(reader)]
        years = [int(y) for y in header[2:]]
        for raw in reader:
            row = [c.strip() for c in raw]
            values = {y: float(v) for y, v in zip(years, row[2:], strict=False) if v}
            if values:
                annual[(row[0], row[1])] = values
    monthly: dict[str, dict[str, float]] = {}
    with monthly_path.open("r", encoding="utf-8", newline="") as handle:
        lines = (line.strip() for line in handle if line.strip())
        for row in csv.DictReader(lines):
            month = row.pop("month")
            monthly[month] = {k: float(v) for k, v in row.items()}
    sources_path = folder / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    return Indices(annual, monthly, tuple(sorted(monthly)), sources)


def _year_fraction(d: date) -> float:
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def annual_at(series: dict[int, float], d: date) -> float | None:
    """Annual values sit mid-year (PEP July-1 estimates, QCEW annual averages)."""
    if not series:
        return None
    t = _year_fraction(d)
    pts = sorted((y + 0.5, v) for y, v in series.items())
    if len(pts) == 1:
        return pts[0][1] if abs(t - pts[0][0]) <= MAX_EXTRAPOLATION_YEARS else None
    if t <= pts[0][0]:
        (t0, v0), (t1, v1) = pts[0], pts[1]
        if pts[0][0] - t > MAX_EXTRAPOLATION_YEARS or v0 <= 0 or v1 <= 0:
            return None
        return v0 * (v0 / v1) ** ((t0 - t) / (t1 - t0))
    if t >= pts[-1][0]:
        (t0, v0), (t1, v1) = pts[-2], pts[-1]
        if t - pts[-1][0] > MAX_EXTRAPOLATION_YEARS or v0 <= 0 or v1 <= 0:
            return None
        return v1 * (v1 / v0) ** ((t - t1) / (t1 - t0))
    for (t0, v0), (t1, v1) in pairwise(pts):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return None


def monthly_at(ix: Indices, key: str, d: date) -> float | None:
    """Value for the month of ``d``; past the newest month the newest value holds."""
    month = f"{d.year:04d}-{d.month:02d}"
    if month < ix.months[0]:
        return None
    month = min(month, ix.months[-1])
    return ix.monthly.get(month, {}).get(key)


@dataclass(slots=True)
class OnLevelFactor:
    factor: float
    components: dict[str, float] = field(default_factory=dict)
    levels: dict[str, str] = field(default_factory=dict)  # component -> county | state | national | flat


def _growth(ix: Indices, gid: str, metric: str, d0: date, d1: date) -> tuple[float, str]:
    """Quantity ratio for one metric, county first, then state, else flat."""
    parts = gid.split("-")
    candidates = [(gid, "county")] if len(parts) == 3 else []
    candidates.append((f"{parts[0]}-{parts[1]}", "state"))
    for key, level in candidates:
        series = ix.annual.get((key, metric))
        if not series:
            continue
        a, b = annual_at(series, d0), annual_at(series, d1)
        if a is None or b is None or a <= 0:
            continue
        if level == "county" and metric in ("EMP", "WAGES"):
            emp = ix.annual.get((key, "EMP"), {})
            if min(emp.values(), default=0) < MIN_COUNTY_EMPLOYMENT:
                continue
        return b / a, level
    return 1.0, "flat"


def _price(ix: Indices, key: str, d0: date, d1: date) -> float:
    a, b = monthly_at(ix, key, d0), monthly_at(ix, key, d1)
    return b / a if a and b else 1.0


def _cost(ix: Indices, gid: str, ppi_key: str, d0: date, d1: date) -> tuple[float, dict, dict]:
    ppi = _price(ix, ppi_key, d0, d1)
    pay, pay_level = _growth(ix, "-".join(gid.split("-")[:2]), "CONS_PAY", d0, d1)
    cost = MATERIAL_SHARE * ppi + (1 - MATERIAL_SHARE) * pay
    return cost, {ppi_key: ppi, "CONS_PAY": pay}, {ppi_key: "national", "CONS_PAY": pay_level}


@lru_cache(maxsize=65536)
def _factor_cached(gid: str, segment: str, d0: date, d1: date) -> OnLevelFactor | None:
    ix = load_indices()
    if ix is None:
        return None
    if segment == "RESIDENTIAL":
        hu, hu_level = _growth(ix, gid, "HU", d0, d1)
        cost, comps, levels = _cost(ix, gid, "PPI_RES_INPUTS", d0, d1)
        return OnLevelFactor(hu * cost, {"HU": hu, **comps, "replacementCost": cost},
                             {"HU": hu_level, **levels})
    if segment == "COMMERCIAL":
        emp, emp_level = _growth(ix, gid, "EMP", d0, d1)
        wages, wages_level = _growth(ix, gid, "WAGES", d0, d1)
        cost, comps, levels = _cost(ix, gid, "PPI_NONRES_INPUTS", d0, d1)
        f = COMMERCIAL_BUILDING_SHARE * emp * cost + (1 - COMMERCIAL_BUILDING_SHARE) * wages
        return OnLevelFactor(f, {"EMP": emp, "WAGES": wages, **comps, "replacementCost": cost},
                             {"EMP": emp_level, "WAGES": wages_level, **levels})
    if segment == "AUTO":
        pop, pop_level = _growth(ix, gid, "POP", d0, d1)
        acv = _price(ix, "CPI_USED_VEHICLES", d0, d1)
        return OnLevelFactor(pop * acv, {"POP": pop, "CPI_USED_VEHICLES": acv},
                             {"POP": pop_level, "CPI_USED_VEHICLES": "national"})
    raise ValueError(f"unknown segment {segment!r}")


def factor(geography_id: str, segment: str, event_date: date, target_date: date) -> OnLevelFactor | None:
    """Insured-value growth for a county (``US-FL-12071``) or state (``US-FL``)."""
    return _factor_cached(geography_id, segment, event_date, target_date)


# ─────────────────────────────── adjustments ───────────────────────────────


@dataclass(frozen=True, slots=True)
class Adjustment:
    storm_id: str  # or *
    geography: str  # US-FL, US-FL-12071, or *
    segment: str  # or *
    factor: float
    reason: str

    def matches(self, storm_id: str, gid: str, segment: str) -> bool:
        return (
            self.storm_id in ("*", storm_id)
            and self.segment in ("*", segment)
            and (self.geography == "*" or gid == self.geography or gid.startswith(self.geography + "-"))
        )


@lru_cache(maxsize=1)
def load_adjustments() -> tuple[Adjustment, ...]:
    """Optional analyst multipliers (e.g. litigation creep) in onlevel/adjustments.csv."""
    path = industry_losses.data_dir() / ONLEVEL_DIR / "adjustments.csv"
    if not path.exists():
        return ()
    out: list[Adjustment] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")):
            out.append(Adjustment(
                storm_id=(row.get("stormId") or "*").strip().upper(),
                geography=(row.get("geographyId") or "*").strip().upper(),
                segment=(row.get("segment") or "*").strip().upper(),
                factor=float(row["factor"]),
                reason=(row.get("reason") or "").strip(),
            ))
    return tuple(out)


# ─────────────────────────────── actuals ───────────────────────────────


def event_date_for(actuals: EventActuals) -> date:
    """PCS first event date, else July 1 of the storm year (ATCF id suffix)."""
    if actuals.event_date:
        return date.fromisoformat(actuals.event_date)
    year = int(actuals.storm_id[-4:]) if actuals.storm_id[-4:].isdigit() else date.today().year
    return date(year, 7, 1)


def onlevel_actuals(actuals: EventActuals, target_date: date) -> tuple[EventActuals, dict | None]:
    """Scale every reported row to ``target_date`` exposure.

    Returns the original actuals and ``None`` when no indices are installed.
    Claim counts are left as reported.
    """
    ix = load_indices()
    if ix is None:
        return actuals, None
    d0 = event_date_for(actuals)
    adjustments = load_adjustments()
    used: dict[Adjustment, float] = {}
    levels: dict[str, int] = {}

    def scale(gid: str, seg: str, loss: float) -> float:
        f = factor(gid, seg, d0, target_date)
        if f is None:
            return loss
        quantity_metric = {"RESIDENTIAL": "HU", "COMMERCIAL": "EMP", "AUTO": "POP"}[seg]
        lvl = f.levels.get(quantity_metric, "flat")
        levels[lvl] = levels.get(lvl, 0) + 1
        out = loss * f.factor
        for adj in adjustments:
            if adj.matches(actuals.storm_id, gid, seg):
                used[adj] = used.get(adj, 0.0) + out * (adj.factor - 1)
                out *= adj.factor
        return out

    county = {(gid, seg): scale(gid, seg, v) for (gid, seg), v in actuals.county.items()}
    state_rows = {(st, seg): scale(f"US-{st}", seg, v) for (st, seg), v in actuals.state_rows.items()}
    unallocated = {(st, seg): scale(f"US-{st}", seg, v) for (st, seg), v in actuals.unallocated.items()}
    scaled = replace(actuals, county=county, state_rows=state_rows, unallocated=unallocated)

    reported = actuals.segment_totals()
    onlevel = scaled.segment_totals()
    summary = {
        "eventDate": d0.isoformat(),
        "targetDate": target_date.isoformat(),
        "reportedTotal": actuals.total(),
        "reportedBySegment": reported,
        "factor": (scaled.total() / actuals.total()) if actuals.total() > 0 else None,
        "factorBySegment": {
            seg: (onlevel[seg] / reported[seg]) if reported[seg] > 0 else None for seg in reported
        },
        "rowsByQuantityLevel": levels,
        "adjustments": [
            {"stormId": a.storm_id, "geographyId": a.geography, "segment": a.segment,
             "factor": a.factor, "reason": a.reason, "lossEffect": effect}
            for a, effect in used.items()
        ],
        "indicesBuiltAt": ix.sources.get("builtAt"),
        "latestMonth": ix.months[-1],
        "method": (
            f"RES = HU x ({MATERIAL_SHARE:.0%} PPI res inputs + {1 - MATERIAL_SHARE:.0%} state "
            f"construction pay); COM = {COMMERCIAL_BUILDING_SHARE:.0%} EMP x nonres cost + "
            f"{1 - COMMERCIAL_BUILDING_SHARE:.0%} county payroll; AUTO = POP x CPI used vehicles"
        ),
    }
    return scaled, summary


def clear_caches() -> None:
    load_indices.cache_clear()
    load_adjustments.cache_clear()
    _factor_cached.cache_clear()
