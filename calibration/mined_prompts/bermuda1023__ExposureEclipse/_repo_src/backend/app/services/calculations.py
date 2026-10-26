"""Calculation services — the single source of truth for every formula in
`docs/CALCULATION_RULES.md`. Map, detail, pivot, and export call into here;
they MUST NOT recompute these metrics elsewhere (CLAUDE.md rule "single source
of truth").

Design rules (all derived from CALCULATION_RULES.md + CLAUDE.md):

* **Pure functions.** No I/O, no HTTP, no global state. Operate on iterables
  of `ExposureFactNormalized`.
* **Aggregate, then divide.** Ratios are taken from the summed numerator and
  denominator at the target grain — ratios are never averaged.
* **`null` on cannot-compute.** Never substitute `0` for missing. Returns
  `None` (the caller attaches the warning to the response). The one exception
  is :func:`client_market_share`, which owns its own domain-specific warning
  code (`WARN_IED_DENOMINATOR_MISSING`).
* **Divide-by-zero never raises.** Always degrades to ``None``.
* **No currency mixing inside aggregate_*.** The router is responsible for
  partitioning facts by currency (or applying an explicit conversion
  assumption) before calling these helpers; the math here does not look at
  ``fact.currency``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from typing import Any

from app.models.enums import (
    AggregationLevel,
    Measure,
    OccupancySegment,
    Peril,
    WarningCode,
    YoyStatus,
)
from app.models.exposure import ExposureFactNormalized, ExposureFilters, IEDIndustryRow
from app.models.warnings import Warning, make_warning

# ───────────────────────── Aggregations ─────────────────────────

# Mapping from Measure → attribute name on ExposureFactNormalized. Measures with
# no direct numeric column (none today) would need a custom reducer here.
_MEASURE_TO_ATTR: dict[Measure, str] = {
    Measure.TIV: "tiv",
    Measure.BUILDING: "building",
    Measure.CONTENTS: "contents",
    Measure.BI: "bi",
    Measure.EXPLIM_GR: "explim_gross",
    Measure.EXPLIM_NET: "explim_net",
    Measure.LOCATION_COUNT: "location_count",
    Measure.ACCOUNT_COUNT: "account_count",
    Measure.INVALID_TIV: "invalid_tiv",
    Measure.INVALID_COUNT: "invalid_count",
}

_INTEGER_MEASURES: frozenset[Measure] = frozenset(
    {Measure.LOCATION_COUNT, Measure.ACCOUNT_COUNT, Measure.INVALID_COUNT}
)


def _grain_key(fact: ExposureFactNormalized, by: Sequence[str]) -> tuple[Any, ...]:
    """Build the group-key tuple for one fact across attribute names ``by``.

    Unknown attribute names map to ``None`` so callers see a stable shape rather
    than a hard failure on a typo. (The result is still a stable tuple key, so
    aggregates work — but the caller would notice everything collapses into
    one bucket and likely catches the mistake.)

    Special case: when the caller groups by the bare ``county`` attribute we
    qualify the value with the state code ("Charlotte, FL") because many
    states share county names. The map/detail paths key by ``geography_id``
    instead (US-FL-12086), so they're unaffected — only pivot rows/columns
    that pick the COUNTY dimension flow through here.
    """
    out: list[Any] = []
    for attr in by:
        if attr == "county":
            county = getattr(fact, "county", None)
            state = getattr(fact, "statecode", None)
            out.append(f"{county}, {state}" if county and state else county)
        else:
            out.append(getattr(fact, attr, None))
    return tuple(out)


def aggregate_tiv(
    facts: Iterable[ExposureFactNormalized], by: Sequence[str]
) -> dict[tuple, float]:
    """Σ ``fact.tiv`` grouped by the tuple of attribute names in ``by``.

    Caller MUST partition by currency before calling; this helper does not
    inspect ``fact.currency`` (CALCULATION_RULES.md "no silent currency mixing").
    """
    out: dict[tuple, float] = defaultdict(float)
    for f in facts:
        out[_grain_key(f, by)] += float(f.tiv or 0.0)
    return dict(out)


def aggregate_location_count(
    facts: Iterable[ExposureFactNormalized], by: Sequence[str]
) -> dict[tuple, int]:
    """Σ ``fact.location_count`` grouped by ``by``."""
    out: dict[tuple, int] = defaultdict(int)
    for f in facts:
        out[_grain_key(f, by)] += int(f.location_count or 0)
    return dict(out)


def aggregate_measure(
    facts: Iterable[ExposureFactNormalized],
    measure: Measure,
    by: Sequence[str],
) -> dict[tuple, float | int]:
    """Generic aggregator across canonical :class:`Measure` values (CONTRACTS §1b).

    All measures here are additive (Σ over the grain). ``None`` values are
    treated as zero contributions (rather than poisoning the sum) — they're a
    data-quality signal the provider already surfaces via warnings. Caller
    must partition by currency for monetary measures.
    """
    if measure not in _MEASURE_TO_ATTR:
        raise ValueError(f"Unsupported measure: {measure!r}")
    attr = _MEASURE_TO_ATTR[measure]
    is_int = measure in _INTEGER_MEASURES

    if is_int:
        out_i: dict[tuple, int] = defaultdict(int)
        for f in facts:
            v = getattr(f, attr, None)
            if v is not None:
                out_i[_grain_key(f, by)] += int(v)
        return dict(out_i)

    out_f: dict[tuple, float] = defaultdict(float)
    for f in facts:
        v = getattr(f, attr, None)
        if v is not None:
            out_f[_grain_key(f, by)] += float(v)
    return dict(out_f)


# ───────────────────────── Ratios ─────────────────────────


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Shared null-on-cannot-compute guard for every ratio metric."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def deal_share_of_portfolio_in_geography(
    deal_geo_tiv: float | None, portfolio_geo_tiv: float | None
) -> float | None:
    """``deal geo TIV ÷ portfolio geo TIV`` (CALCULATION_RULES.md §Deal Share…).

    Both inputs MUST be at the same geography + grain. ``None`` if either is
    missing or denominator is zero — caller pairs with the appropriate warning.
    """
    return _safe_ratio(deal_geo_tiv, portfolio_geo_tiv)


def geography_share_of_total_portfolio(
    portfolio_geo_tiv: float | None, total_portfolio_tiv: float | None
) -> float | None:
    """``portfolio geo TIV ÷ total portfolio TIV`` (CALCULATION_RULES.md §Geography Share…)."""
    return _safe_ratio(portfolio_geo_tiv, total_portfolio_tiv)


def selected_deal_geography_concentration(
    deal_geo_tiv: float | None, total_deal_tiv: float | None
) -> float | None:
    """``deal geo TIV ÷ deal total TIV`` (CALCULATION_RULES.md §Selected Deal Concentration)."""
    return _safe_ratio(deal_geo_tiv, total_deal_tiv)


def client_market_share(
    client_tiv: float | None,
    industry_tiv: float | None,
) -> tuple[float | None, Warning | None]:
    """``client TIV ÷ RMS IED industry TIV`` (CALCULATION_RULES.md §Client Market Share).

    Returns ``(share, warning?)``. When the industry denominator is missing the
    share is ``None`` and a ``WARN_IED_DENOMINATOR_MISSING`` warning is
    returned so the caller can attach it to the response body (this is the one
    domain-specific warning calculations.py owns directly).

    Denominator zero degrades to ``(None, None)`` — the data is present but
    the ratio is mathematically undefined; the caller decides what to surface.
    """
    if industry_tiv is None:
        return None, make_warning(WarningCode.WARN_IED_DENOMINATOR_MISSING)
    share = _safe_ratio(client_tiv, industry_tiv)
    return share, None


def lookup_industry_tiv(
    ied: list[IEDIndustryRow],
    geography_id: str,
    occupancy_segment: OccupancySegment | Collection[OccupancySegment] | None = None,
) -> float | None:
    """Find the matching IED row for the given ``geography_id`` (+ optional segments).

    Sums any matching rows (geometry-level IED is typically one row per key but
    we don't rely on that). A collection sums those segments only. ``UNKNOWN``
    is treated literally — never force-mapped onto another segment
    (CONTRACTS §6 / CALCULATION_RULES.md).
    """
    if occupancy_segment is None:
        wanted: set[str] | None = None
    elif isinstance(occupancy_segment, (str, OccupancySegment)):
        wanted = {str(getattr(occupancy_segment, "value", occupancy_segment))}
    else:
        wanted = {str(getattr(s, "value", s)) for s in occupancy_segment}
    total: float | None = None
    for row in ied:
        if row.geography_id != geography_id:
            continue
        if wanted is not None and str(getattr(row.occupancy_segment, "value", row.occupancy_segment)) not in wanted:
            continue
        total = (total or 0.0) + float(row.industry_tiv or 0.0)
    return total


# ───────────────────────── Year-over-Year ─────────────────────────


def yoy_change(
    current_value: float | None, prior_value: float | None
) -> tuple[float | None, YoyStatus]:
    """Implements the YoY truth table in CALCULATION_RULES.md §Year-over-Year Change.

    Returns ``(change, status)``:

    * both present, prior ≠ 0 → ``(value, OK)``
    * current present, prior missing → ``(None, NEW)``
    * prior present, current missing → ``(None, REMOVED)``
    * prior == 0 → ``(None, NA)`` (divide-by-zero is never an exception)
    * both missing → ``(None, NA)`` (defensive — caller usually short-circuits)

    The "no prior dataset selected" case is the router's responsibility
    (it emits ``WARN_PRIOR_DATASET_NOT_SELECTED`` and never calls this).
    """
    if current_value is None and prior_value is None:
        return None, YoyStatus.NA
    if current_value is not None and prior_value is None:
        return None, YoyStatus.NEW
    if current_value is None and prior_value is not None:
        return None, YoyStatus.REMOVED
    # Both present
    assert current_value is not None and prior_value is not None  # for type checkers
    if prior_value == 0:
        return None, YoyStatus.NA
    return (float(current_value) - float(prior_value)) / float(prior_value), YoyStatus.OK


# ───────────────────────── Filtering ─────────────────────────


def apply_filters(
    facts: Iterable[ExposureFactNormalized], filters: ExposureFilters
) -> list[ExposureFactNormalized]:
    """Apply :class:`ExposureFilters` to ``facts`` per API_SPEC.md filter contract.

    Rules:

    * ``peril == Peril.ALL`` is a pass-through (CONTRACTS §5 — ``ALL`` is a
      filter sentinel, not a fact value).
    * Empty list on any string dimension = no filter on that dimension.
    * Otherwise the fact's value must be in the supplied list.
    """
    out: list[ExposureFactNormalized] = []
    occupancy = set(filters.occupancy)
    segments = {str(getattr(s, "value", s)) for s in filters.occupancy_segment}
    dtc = set(filters.distance_to_coast)
    geocoding = set(filters.geocoding)
    construction = set(filters.construction)
    stories = set(filters.number_of_stories)
    year_built = set(filters.year_built)
    peril_filter = filters.peril

    for f in facts:
        if peril_filter != Peril.ALL and f.peril != peril_filter:
            continue
        if occupancy and (f.occupancy or "") not in occupancy:
            continue
        if segments and str(getattr(f.occupancy_segment, "value", f.occupancy_segment)) not in segments:
            continue
        if dtc and (f.distance_to_coast or "") not in dtc:
            continue
        if geocoding and (f.geocoding_quality or "") not in geocoding:
            continue
        if construction and (f.construction or "") not in construction:
            continue
        if stories and (f.number_of_stories or "") not in stories:
            continue
        if year_built and (f.year_built or "") not in year_built:
            continue
        out.append(f)
    return out


def filter_to_geography_level(
    facts: Iterable[ExposureFactNormalized], level: AggregationLevel
) -> list[ExposureFactNormalized]:
    """Keep only facts whose ``aggregation``/``geography_level`` matches ``level``.

    The provider may serve multiple levels in one stream (Country + State +
    County rows). Selecting a view at one level should drop the others rather
    than double-counting.
    """
    return [f for f in facts if f.aggregation == level and f.geography_level == level]


__all__ = [
    "aggregate_location_count",
    "aggregate_measure",
    "aggregate_tiv",
    "apply_filters",
    "client_market_share",
    "deal_share_of_portfolio_in_geography",
    "filter_to_geography_level",
    "geography_share_of_total_portfolio",
    "lookup_industry_tiv",
    "selected_deal_geography_concentration",
    "yoy_change",
]
