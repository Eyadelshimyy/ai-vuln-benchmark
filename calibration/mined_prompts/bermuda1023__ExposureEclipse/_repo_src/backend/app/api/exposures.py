"""Exposure endpoints — `/map`, `/detail`, `/pivot` (API_SPEC.md §Exposure APIs).

Routers are thin: pull facts from the provider, apply filters, hand off to the
calculation / grouping services, attach warnings, return the response. No
formula lives here (CLAUDE.md rule "single source of truth").

Source-of-truth notes:

* **Grain semantics** (CONTRACTS.md §13): the view grain is the union of
  geography + active dimensions. Map uses ``[geography_attr]``; pivot uses
  ``rows + columns``; detail uses ``[geography_attr]`` (one row).
* **Group combination** (CLAUDE.md rule 3, CONTRACTS.md §4): the default
  ``MAX_ACROSS_PERILS_AT_VIEW_GRAIN`` is computed by
  :func:`grouping.combine_at_grain` at the requested grain. The location count
  follows :func:`grouping.location_count_at_max_peril` so we don't double-count.
* **County fallback** (CLAUDE.md rule 11): when ``COUNTY`` is requested and a
  state has no county rows in the data, we fall back to STATE rows for that
  state and attach ``WARN_COUNTY_DATA_UNAVAILABLE`` once.
* **Geometry availability**: features whose ``geographyId`` is not in
  ``provider.get_geometry_availability()`` get ``hasGeometry: false`` and a
  per-feature ``WARN_MAP_GEOMETRY_MISSING`` (info severity).
* **YoY**: only when ``comparisonDatasetId`` is supplied; else attach a
  top-level ``WARN_PRIOR_DATASET_NOT_SELECTED`` and leave the yoy fields
  ``None`` (CLAUDE.md rule "null on cannot-compute").
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException

from ..models.enums import (
    AggregationLevel,
    CombinationMethod,
    ErrorCode,
    Measure,
    MetricKey,
    OccupancySegment,
    Peril,
    WarningCode,
    YoyStatus,
)
from ..models.exposure import (
    BreakdownRow,
    DealVsPortfolio,
    DetailBreakdowns,
    DetailRequest,
    DetailResponse,
    DetailSummary,
    ExposureFactNormalized,
    MapFeature,
    MapRequest,
    MapResponse,
    MarketShareDetail,
    PivotCell,
    PivotRequest,
    PivotResponse,
    YoyDetail,
)
from ..models.warnings import Warning, make_warning
from ..providers import ExposureDataProvider, get_provider
from ..services.calculations import (
    aggregate_location_count,
    aggregate_measure,
    aggregate_tiv,
    apply_filters,
    client_market_share,
    deal_share_of_portfolio_in_geography,
    filter_to_geography_level,
    geography_share_of_total_portfolio,
    lookup_industry_tiv,
    selected_deal_geography_concentration,
    yoy_change,
)
from ..services.grouping import combine_at_grain, location_count_at_max_peril

router = APIRouter(prefix="/exposures", tags=["exposures"])


# ───────────────────────────── helpers ─────────────────────────────


# Pivot dimension key → fact attribute. Includes the AggregationLevel codes
# (so rows/columns may carry "STATE", "COUNTY", etc.) and the broader dimension
# names listed in API_SPEC.md §pivot.
_PIVOT_DIM_TO_ATTR: dict[str, str] = {
    # geography levels (also acceptable as rows/columns)
    AggregationLevel.COUNTRY.value: "country",
    AggregationLevel.STATE.value: "statecode",
    AggregationLevel.COUNTY.value: "county",
    AggregationLevel.CRESTA.value: "cresta",
    # dimension keys
    "PERIL": "peril",
    "OCCUPANCY": "occupancy",
    "OCCUPANCY_GROUP": "occupancy_group",
    "OCCUPANCY_SEGMENT": "occupancy_segment",
    "CONSTRUCTION": "construction",
    "DISTANCE_TO_COAST": "distance_to_coast",
    "GEOCODING": "geocoding_quality",
    "NUMBER_OF_STORIES": "number_of_stories",
    "YEAR_BUILT": "year_built",
    "DATASET": "dataset_id",
    "DATASET_GROUP": "dataset_group_id",
    "CURRENCY": "currency",
    "TREATY_YEAR": "portname",  # close-enough proxy; v1 doesn't carry treaty year on facts
}


from ..services.view_resolve import (
    ResolvedView as _ResolvedView,
    require_at_most_one_target as _require_exactly_one_target,
    resolve_view as _resolve_view,
)


def _apply_peril_filter(
    facts: list[ExposureFactNormalized],
    perils: list[Peril],
) -> list[ExposureFactNormalized]:
    """Top-of-page peril multi-select. Empty / contains ALL → pass-through."""
    if not perils or Peril.ALL in perils:
        return facts
    wanted = {p.value if hasattr(p, "value") else str(p) for p in perils}
    return [f for f in facts if f.peril in wanted]


def _county_fallback_if_needed(
    facts: list[ExposureFactNormalized],
    level: AggregationLevel,
) -> tuple[list[ExposureFactNormalized], bool]:
    """If COUNTY view but some state has no county rows, fall back to STATE.

    Per CLAUDE.md rule 11 (graceful degradation, never fabrication). Returns
    ``(facts_used, county_fallback_applied)``.

    The fallback adds STATE-level rows for any state that has zero COUNTY-level
    rows. COUNTY rows for states with data are kept as-is.
    """
    if level != AggregationLevel.COUNTY:
        return filter_to_geography_level(facts, level), False

    county_rows = filter_to_geography_level(facts, AggregationLevel.COUNTY)
    states_with_counties = {f.statecode for f in county_rows if f.statecode}
    all_state_rows = filter_to_geography_level(facts, AggregationLevel.STATE)
    fallback_rows = [
        f for f in all_state_rows if f.statecode and f.statecode not in states_with_counties
    ]
    return county_rows + fallback_rows, bool(fallback_rows)


def _attach_geometry_warnings(
    features: Sequence[MapFeature],
    available: set[str],
) -> None:
    """Mark each feature without geometry + add a per-feature warning."""
    for feature in features:
        if feature.geography_id not in available:
            feature.has_geometry = False
            feature.warnings.append(
                make_warning(
                    WarningCode.WARN_MAP_GEOMETRY_MISSING,
                    context={"geographyId": feature.geography_id},
                )
            )


def _geography_name_for(
    geography_id: str,
    facts: Sequence[ExposureFactNormalized],
    level: AggregationLevel,
) -> str | None:
    """Pick a display name from facts (first match wins).

    Detects the row's actual level from the matched fact rather than trusting the
    requested level — county-fallback rows live in a COUNTY response but carry a
    state-grain geographyId ("US-NC"), and we want "NORTH CAROLINA" not None.
    """
    for f in facts:
        if f.geography_id != geography_id:
            continue
        # Pick the most-specific attribute that has a value.
        for attr in ("county_name", "cresta_name", "state_name", "country_name"):
            name = getattr(f, attr, None)
            if name:
                return name
    return None


def _segment_for_market_share(
    facts: Sequence[ExposureFactNormalized],
    requested: Sequence[OccupancySegment | str] = (),
) -> list[OccupancySegment] | None:
    """Resolve the occupancy segments for IED lookup, else ``None`` (= ALL).

    An explicit ``filters.occupancySegment`` wins so the denominator covers the
    same segments as the numerator. Otherwise a single segment on the facts is
    used, and anything mixed falls back to ALL.

    `use_enum_values=True` on the Pydantic config means `fact.occupancy_segment`
    is a raw string at runtime, not an OccupancySegment instance. Coerce back to
    the enum here so callers can rely on `.value` and `==` against enum members.
    """
    if requested:
        return sorted({OccupancySegment(s) for s in requested}, key=lambda s: s.value)
    raw = {f.occupancy_segment for f in facts if f.occupancy_segment}
    segments: set[OccupancySegment] = set()
    for value in raw:
        try:
            segments.add(OccupancySegment(value))
        except ValueError:
            continue  # unrecognized segment string — treat as ambiguous
    if len(segments) == 1:
        only = next(iter(segments))
        if only != OccupancySegment.UNKNOWN:
            return [only]
    return None


def _segment_label(segments: Sequence[OccupancySegment] | None) -> str:
    return "+".join(s.value for s in segments) if segments else "ALL"


def _scope_aggregates(
    facts: Sequence[ExposureFactNormalized],
    grain: Sequence[str],
    method: CombinationMethod | None,
    *,
    base_dataset_id: str | None = None,
) -> tuple[dict[tuple, float], dict[tuple, int]]:
    """Aggregate one scope consistently for current, prior, and portfolio facts."""
    if method is not None and method != CombinationMethod.KEEP_PERILS_SEPARATE:
        tiv = combine_at_grain(
            facts,
            grain,
            method,
            distinct_segments_confirmed=True,
            base_dataset_id=base_dataset_id,
        )
        locations = (
            location_count_at_max_peril(facts, grain)
            if method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
            else aggregate_location_count(facts, grain)
        )
        return tiv, locations
    return aggregate_tiv(facts, grain), aggregate_location_count(facts, grain)


def _is_portfolio_request(payload: MapRequest | DetailRequest) -> bool:
    return not any(
        (
            payload.programme_id,
            payload.chain_id,
            payload.chain_ids,
            payload.cedent_id,
            payload.dataset_id,
            payload.dataset_group_id,
        )
    )


def _portfolio_aggregates(
    provider: ExposureDataProvider,
    payload: MapRequest | DetailRequest,
    resolved: _ResolvedView,
    level: AggregationLevel,
    grain: Sequence[str],
) -> tuple[dict[tuple, float], list[Warning]]:
    """Build a currency-safe denominator using the same peril/filter semantics."""
    portfolio_facts = (
        resolved.facts if _is_portfolio_request(payload) else provider.get_portfolio_facts()
    )
    portfolio_facts = _apply_peril_filter(portfolio_facts, payload.perils)

    currencies = {f.currency for f in portfolio_facts if f.currency}
    excluded = sorted(currency for currency in currencies if currency != resolved.currency)
    portfolio_facts = [f for f in portfolio_facts if f.currency == resolved.currency]

    portfolio_at_level, _ = _county_fallback_if_needed(portfolio_facts, level)
    portfolio_filtered = apply_filters(portfolio_at_level, payload.filters)
    portfolio_tiv, _ = _scope_aggregates(
        portfolio_filtered,
        grain,
        CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
    )
    warnings: list[Warning] = []
    if excluded:
        warnings.append(
            make_warning(
                WarningCode.WARN_CURRENCY_MISMATCH,
                context={
                    "includedCurrency": resolved.currency,
                    "excludedCurrencies": excluded,
                    "scope": "portfolio denominator",
                },
            )
        )
    return portfolio_tiv, warnings


# ───────────────────────────── /map ─────────────────────────────


@router.post("/map", response_model=MapResponse)
def exposures_map(
    payload: MapRequest,
    provider: ExposureDataProvider = Depends(get_provider),
) -> MapResponse:
    """Compute the map view (API_SPEC.md `POST /api/exposures/map`)."""
    _require_exactly_one_target(payload)

    resolved = _resolve_view(provider, payload)
    raw_facts = _apply_peril_filter(resolved.facts, payload.perils)
    currency = resolved.currency
    combination_method = resolved.combination_method
    base_dataset_id = resolved.base_dataset_id

    level = payload.aggregation_level
    facts_at_level, county_fallback = _county_fallback_if_needed(raw_facts, level)

    grain: tuple[str, ...] = ("geography_id",)

    deal_facts = apply_filters(facts_at_level, payload.filters)
    top_warnings: list[Warning] = list(resolved.warnings)

    if county_fallback:
        top_warnings.append(make_warning(WarningCode.WARN_COUNTY_DATA_UNAVAILABLE))

    if not deal_facts:
        top_warnings.append(make_warning(WarningCode.WARN_FILTERS_RETURN_NO_ROWS))
        return MapResponse(
            aggregation_level=level,
            metric=payload.metric,
            currency=currency,
            features=[],
            warnings=top_warnings,
        )

    # ─── Deal-side TIV / location-count (group-aware) ───
    try:
        deal_tiv_by_geo, deal_loc_by_geo = _scope_aggregates(
            deal_facts,
            grain,
            combination_method,
            base_dataset_id=base_dataset_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": str(exc),
            },
        ) from exc

    # ─── Portfolio denominators (geo-only) ───
    portfolio_tiv_by_geo, portfolio_warnings = _portfolio_aggregates(
        provider, payload, resolved, level, grain
    )
    top_warnings.extend(portfolio_warnings)
    total_portfolio_tiv = sum(portfolio_tiv_by_geo.values())
    total_deal_tiv = sum(deal_tiv_by_geo.values())

    # ─── YoY (optional) ───
    prior_tiv_by_geo: dict[tuple, float] = {}
    prior_loc_by_geo: dict[tuple, int] = {}
    prior_total_deal_tiv: float = 0.0
    comparison_dataset_id = resolved.comparison_dataset_id
    if comparison_dataset_id:
        prior_prog = provider.get_programme_by_dataset_id(comparison_dataset_id)
        if prior_prog is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": ErrorCode.PRIOR_DB_NOT_FOUND.value,
                    "message": (
                        "Prior-year database was not found. Check the server/database "
                        "name or select another prior-year dataset."
                    ),
                    "details": {"datasetId": comparison_dataset_id},
                },
            )
        prior_raw = _apply_peril_filter(
            provider.get_facts_for_dataset(comparison_dataset_id), payload.perils
        )
        prior_at_level, _ = _county_fallback_if_needed(prior_raw, level)
        prior_filtered = apply_filters(prior_at_level, payload.filters)
        prior_tiv_by_geo, prior_loc_by_geo = _scope_aggregates(
            prior_filtered,
            grain,
            CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
        )
        prior_total_deal_tiv = sum(prior_tiv_by_geo.values())
    else:
        top_warnings.append(make_warning(WarningCode.WARN_PRIOR_DATASET_NOT_SELECTED))

    # ─── IED industry rows for market share ───
    ied_rows = provider.get_ied_industry()
    segment = _segment_for_market_share(deal_facts, payload.filters.occupancy_segment)

    # ─── Assemble features ───
    features: list[MapFeature] = []
    for (geography_id,), deal_tiv in deal_tiv_by_geo.items():
        feature_warnings: list[Warning] = []
        loc_count = int(deal_loc_by_geo.get((geography_id,), 0))
        portfolio_geo_tiv = portfolio_tiv_by_geo.get((geography_id,), 0.0)

        deal_share = deal_share_of_portfolio_in_geography(deal_tiv, portfolio_geo_tiv)
        geo_share = geography_share_of_total_portfolio(portfolio_geo_tiv, total_portfolio_tiv)
        concentration = selected_deal_geography_concentration(deal_tiv, total_deal_tiv)

        industry_tiv = lookup_industry_tiv(ied_rows, geography_id, segment)
        share, share_warning = client_market_share(deal_tiv, industry_tiv)
        if share_warning is not None:
            share_warning_with_ctx = make_warning(
                share_warning.code,
                context={"geographyId": geography_id, "segment": _segment_label(segment)},
            )
            feature_warnings.append(share_warning_with_ctx)

        yoy_value: float | None = None
        yoy_st: YoyStatus | None = None
        if comparison_dataset_id:
            prior_geo_tiv = prior_tiv_by_geo.get((geography_id,))
            yoy_value, yoy_st = yoy_change(deal_tiv, prior_geo_tiv)

        metric_value = _metric_value_for(
            payload.metric,
            tiv=deal_tiv,
            location_count=loc_count,
            deal_share=deal_share,
            geo_share=geo_share,
            concentration=concentration,
            market_share=share,
            yoy=yoy_value,
        )

        # Always compute the prior-period metric value when a comparison dataset
        # is set — feeds both the YoY view-mode override AND the prior-vs-current
        # mini-table the tooltip renders. For ratio metrics, the prior-period
        # denominator uses the CURRENT portfolio (no prior portfolio in v1) —
        # approximation documented in OPEN_QUESTIONS. TIV/LOCATION_COUNT exact.
        prior_metric_value: float | None = None
        if comparison_dataset_id:
            prior_geo_tiv = prior_tiv_by_geo.get((geography_id,))
            prior_geo_loc = prior_loc_by_geo.get((geography_id,))
            prior_industry = lookup_industry_tiv(ied_rows, geography_id, segment)
            prior_metric_value = _metric_value_for(
                payload.metric,
                tiv=prior_geo_tiv,
                location_count=prior_geo_loc,
                deal_share=deal_share_of_portfolio_in_geography(
                    prior_geo_tiv, portfolio_geo_tiv
                ),
                geo_share=geo_share,
                concentration=selected_deal_geography_concentration(
                    prior_geo_tiv, prior_total_deal_tiv
                ),
                market_share=client_market_share(prior_geo_tiv, prior_industry)[0],
                yoy=None,
            )
            if payload.yoy_mode:
                metric_value, _ = yoy_change(metric_value, prior_metric_value)

        features.append(
            MapFeature(
                geography_id=geography_id,
                geography_name=_geography_name_for(geography_id, deal_facts, level),
                metric_value=metric_value,
                prior_metric_value=prior_metric_value,
                tiv=deal_tiv,
                location_count=loc_count,
                deal_share_of_portfolio_in_geography=deal_share,
                geography_share_of_total_portfolio=geo_share,
                selected_deal_geography_concentration=concentration,
                client_market_share=share,
                yoy_change=yoy_value,
                yoy_status=yoy_st,
                has_geometry=True,
                warnings=feature_warnings,
            )
        )

    # ─── Add features for prior-only geographies (REMOVED) when YoY is on ───
    if comparison_dataset_id:
        for (geography_id,), prior_tiv in prior_tiv_by_geo.items():
            if (geography_id,) in deal_tiv_by_geo:
                continue
            yoy_value, yoy_st = yoy_change(None, prior_tiv)
            features.append(
                MapFeature(
                    geography_id=geography_id,
                    geography_name=None,
                    metric_value=None,
                    tiv=None,
                    location_count=None,
                    yoy_change=yoy_value,
                    yoy_status=yoy_st,
                )
            )

    geometry_available = provider.get_geometry_availability()
    _attach_geometry_warnings(features, geometry_available)

    return MapResponse(
        aggregation_level=level,
        metric=payload.metric,
        currency=currency,
        features=features,
        warnings=top_warnings,
    )


def _metric_value_for(
    metric: MetricKey,
    *,
    tiv: float | None,
    location_count: int | None,
    deal_share: float | None,
    geo_share: float | None,
    concentration: float | None,
    market_share: float | None,
    yoy: float | None,
) -> float | None:
    """Mirror the requested metric into ``metricValue`` (API_SPEC.md note)."""
    mapping: dict[MetricKey, float | int | None] = {
        MetricKey.TIV: tiv,
        MetricKey.LOCATION_COUNT: location_count,
        MetricKey.DEAL_SHARE_OF_PORTFOLIO_IN_GEOGRAPHY: deal_share,
        MetricKey.GEOGRAPHY_SHARE_OF_TOTAL_PORTFOLIO: geo_share,
        MetricKey.SELECTED_DEAL_GEOGRAPHY_CONCENTRATION: concentration,
        MetricKey.CLIENT_MARKET_SHARE: market_share,
        MetricKey.YOY_CHANGE: yoy,
    }
    value = mapping.get(metric)
    return float(value) if value is not None else None


# ───────────────────────────── /detail ─────────────────────────────


@router.post("/detail", response_model=DetailResponse)
def exposures_detail(
    payload: DetailRequest,
    provider: ExposureDataProvider = Depends(get_provider),
) -> DetailResponse:
    """Side-panel detail for one geography (API_SPEC.md `POST /api/exposures/detail`)."""
    _require_exactly_one_target(payload)

    resolved = _resolve_view(provider, payload)
    raw_facts = _apply_peril_filter(resolved.facts, payload.perils)
    currency = resolved.currency
    combination_method = resolved.combination_method
    base_dataset_id = resolved.base_dataset_id

    level = payload.aggregation_level
    facts_at_level, county_fallback = _county_fallback_if_needed(raw_facts, level)
    deal_facts = apply_filters(facts_at_level, payload.filters)

    geo_id = payload.geography_id
    geo_facts = [f for f in deal_facts if f.geography_id == geo_id]

    top_warnings: list[Warning] = list(resolved.warnings)
    if county_fallback:
        top_warnings.append(make_warning(WarningCode.WARN_COUNTY_DATA_UNAVAILABLE))
    if not deal_facts:
        top_warnings.append(make_warning(WarningCode.WARN_FILTERS_RETURN_NO_ROWS))

    grain: tuple[str, ...] = ("geography_id",)

    # Deal TIV / loc-count (group-aware)
    deal_tiv_by_geo, deal_loc_by_geo = _scope_aggregates(
        deal_facts,
        grain,
        combination_method,
        base_dataset_id=base_dataset_id,
    )

    deal_tiv = deal_tiv_by_geo.get((geo_id,), 0.0)
    loc_count = int(deal_loc_by_geo.get((geo_id,), 0))
    total_deal_tiv = sum(deal_tiv_by_geo.values())

    portfolio_tiv_by_geo, portfolio_warnings = _portfolio_aggregates(
        provider, payload, resolved, level, grain
    )
    top_warnings.extend(portfolio_warnings)
    portfolio_geo_tiv = portfolio_tiv_by_geo.get((geo_id,), 0.0)
    total_portfolio_tiv = sum(portfolio_tiv_by_geo.values())

    deal_share = deal_share_of_portfolio_in_geography(deal_tiv, portfolio_geo_tiv)
    geo_share = geography_share_of_total_portfolio(portfolio_geo_tiv, total_portfolio_tiv)
    concentration = selected_deal_geography_concentration(deal_tiv, total_deal_tiv)

    segment = _segment_for_market_share(geo_facts, payload.filters.occupancy_segment)
    industry_tiv = lookup_industry_tiv(provider.get_ied_industry(), geo_id, segment)
    share, share_warning = client_market_share(deal_tiv, industry_tiv)

    feature_warnings: list[Warning] = []
    if share_warning is not None:
        feature_warnings.append(
            make_warning(
                share_warning.code,
                context={"geographyId": geo_id, "segment": _segment_label(segment)},
            )
        )

    yoy_value: float | None = None
    yoy_st: YoyStatus = YoyStatus.NA
    prior_geo_tiv: float | None = None
    comparison_dataset_id = resolved.comparison_dataset_id
    if comparison_dataset_id:
        prior_prog = provider.get_programme_by_dataset_id(comparison_dataset_id)
        if prior_prog is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": ErrorCode.PRIOR_DB_NOT_FOUND.value,
                    "message": "Prior-year database was not found.",
                    "details": {"datasetId": comparison_dataset_id},
                },
            )
        prior_raw = _apply_peril_filter(
            provider.get_facts_for_dataset(comparison_dataset_id), payload.perils
        )
        prior_at_level, _ = _county_fallback_if_needed(prior_raw, level)
        prior_filtered = apply_filters(prior_at_level, payload.filters)
        prior_tiv_by_geo, _ = _scope_aggregates(
            prior_filtered,
            grain,
            CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
        )
        prior_geo_tiv = prior_tiv_by_geo.get((geo_id,))
        yoy_value, yoy_st = yoy_change(deal_tiv, prior_geo_tiv)
    else:
        top_warnings.append(make_warning(WarningCode.WARN_PRIOR_DATASET_NOT_SELECTED))

    # ─── Breakdowns (within the selected geography) ───
    breakdowns = DetailBreakdowns(
        peril=_breakdown_rows(geo_facts, "peril", combination_method, base_dataset_id),
        occupancy=_breakdown_rows(
            geo_facts, "occupancy_segment", combination_method, base_dataset_id
        ),
        distance_to_coast=_breakdown_rows(
            geo_facts, "distance_to_coast", combination_method, base_dataset_id
        ),
        geocoding=_breakdown_rows(
            geo_facts, "geocoding_quality", combination_method, base_dataset_id
        ),
        stories=_breakdown_rows(
            geo_facts, "number_of_stories", combination_method, base_dataset_id
        ),
        construction=_breakdown_rows(
            geo_facts, "construction", combination_method, base_dataset_id
        ),
    )

    return DetailResponse(
        geography_id=geo_id,
        geography_name=_geography_name_for(geo_id, geo_facts or deal_facts, level),
        aggregation_level=level,
        currency=currency,
        summary=DetailSummary(
            tiv=deal_tiv,
            location_count=loc_count,
            deal_share_of_portfolio_in_geography=deal_share,
            geography_share_of_total_portfolio=geo_share,
            selected_deal_geography_concentration=concentration,
            client_market_share=share,
            yoy_change=yoy_value,
            yoy_status=yoy_st,
        ),
        deal_vs_portfolio=DealVsPortfolio(
            deal_tiv=deal_tiv,
            portfolio_tiv=portfolio_geo_tiv,
        ),
        market_share=MarketShareDetail(
            client_tiv=deal_tiv if industry_tiv is not None else None,
            industry_tiv=industry_tiv,
            share=share,
            segment=_segment_label(segment),
        ),
        yoy=YoyDetail(
            current_tiv=deal_tiv,
            prior_tiv=prior_geo_tiv,
            change=yoy_value,
            status=yoy_st,
        ),
        breakdowns=breakdowns,
        active_filters=payload.filters.model_dump(by_alias=True),
        warnings=top_warnings + feature_warnings,
    )


def _breakdown_rows(
    facts: Sequence[ExposureFactNormalized],
    attr: str,
    method: CombinationMethod | None,
    base_dataset_id: str | None,
) -> list[BreakdownRow]:
    """Group facts by ``attr`` and emit TIV + loc count per key."""
    usable = [f for f in facts if getattr(f, attr, None) not in (None, "")]
    tiv_by_key, loc_by_key = _scope_aggregates(
        usable,
        (attr,),
        method,
        base_dataset_id=base_dataset_id,
    )
    tiv = {
        (key[0].value if hasattr(key[0], "value") else str(key[0])): value
        for key, value in tiv_by_key.items()
    }
    loc = {
        (key[0].value if hasattr(key[0], "value") else str(key[0])): value
        for key, value in loc_by_key.items()
    }
    return [
        BreakdownRow(key=k, tiv=tiv[k], location_count=loc[k])
        for k in sorted(tiv.keys())
    ]


# ───────────────────────────── /pivot ─────────────────────────────


@router.post("/pivot", response_model=PivotResponse)
def exposures_pivot(
    payload: PivotRequest,
    provider: ExposureDataProvider = Depends(get_provider),
) -> PivotResponse:
    """Compute a pivot grid (API_SPEC.md `POST /api/exposures/pivot`)."""
    _require_exactly_one_target(payload)

    resolved = _resolve_view(provider, payload)
    raw_facts = _apply_peril_filter(resolved.facts, payload.perils)
    currency = resolved.currency
    combination_method = resolved.combination_method
    base_dataset_id = resolved.base_dataset_id

    # The pivot view doesn't pin to a single AggregationLevel — pick the finest
    # geography level mentioned in rows+columns, else default to the data as-is.
    geo_levels = [
        AggregationLevel(dim)
        for dim in [*payload.rows, *payload.columns]
        if dim in AggregationLevel.__members__
    ]
    if geo_levels:
        # finest mentioned (COUNTY > STATE > COUNTRY by enum order in CONTRACTS.md §2)
        order = {
            AggregationLevel.COUNTRY: 0,
            AggregationLevel.STATE: 1,
            AggregationLevel.COUNTY: 2,
            AggregationLevel.CRESTA: 1,  # CRESTA is mid-level; treat as STATE-ish
        }
        level = max(geo_levels, key=lambda lvl: order.get(lvl, 0))
        facts_at_level, county_fallback = _county_fallback_if_needed(raw_facts, level)
    else:
        facts_at_level = raw_facts
        county_fallback = False

    deal_facts = apply_filters(facts_at_level, payload.filters)

    top_warnings: list[Warning] = list(resolved.warnings)
    if county_fallback:
        top_warnings.append(make_warning(WarningCode.WARN_COUNTY_DATA_UNAVAILABLE))

    method = payload.combination_method or combination_method
    if method is not None:
        if method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN:
            top_warnings.append(
                make_warning(WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS)
            )
        elif method == CombinationMethod.SUM_DISTINCT_SEGMENTS:
            top_warnings.append(make_warning(WarningCode.WARN_DATASET_GROUP_SUMMED))

    if not deal_facts:
        top_warnings.append(make_warning(WarningCode.WARN_FILTERS_RETURN_NO_ROWS))
        return PivotResponse(
            rows=payload.rows,
            columns=payload.columns,
            measures=payload.measures,
            currency=currency,
            cells=[],
            grand_total={m.value: 0 for m in payload.measures},
            warnings=top_warnings,
        )

    # Translate dimension names → fact attributes for the grain key.
    def _to_attrs(dims: Sequence[str]) -> list[str]:
        out: list[str] = []
        for d in dims:
            attr = _PIVOT_DIM_TO_ATTR.get(d)
            if attr is None:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": ErrorCode.VALIDATION_ERROR.value,
                        "message": f"Unknown pivot dimension: {d!r}",
                        "details": {"dimension": d},
                    },
                )
            out.append(attr)
        return out

    row_attrs = _to_attrs(payload.rows)
    col_attrs = _to_attrs(payload.columns)
    grain_attrs = row_attrs + col_attrs

    # ─── Per-measure aggregations ───
    # NB: `payload.measures` arrives as plain `str` values (the model uses
    # `use_enum_values=True`); coerce back into the enum for comparisons that
    # need it, and key cells by the canonical wire string.
    cell_values: dict[tuple[tuple, tuple], dict[str, float | int]] = defaultdict(dict)

    for measure in payload.measures:
        measure_enum = Measure(measure) if not isinstance(measure, Measure) else measure
        measure_key = measure_enum.value
        if (
            measure_enum == Measure.TIV
            and method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
        ):
            grained = combine_at_grain(
                deal_facts, grain_attrs, method, distinct_segments_confirmed=True
            )
        elif measure_enum == Measure.TIV and method == CombinationMethod.SUM_DISTINCT_SEGMENTS:
            grained = combine_at_grain(
                deal_facts, grain_attrs, method, distinct_segments_confirmed=True
            )
        elif (
            measure_enum == Measure.LOCATION_COUNT
            and method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
        ):
            grained = location_count_at_max_peril(deal_facts, grain_attrs)
        else:
            grained = aggregate_measure(deal_facts, measure_enum, grain_attrs)

        for key, value in grained.items():
            row_key = key[: len(row_attrs)]
            col_key = key[len(row_attrs) :]
            cell_values[(row_key, col_key)][measure_key] = value

    cells: list[PivotCell] = []
    grand: dict[str, float | int] = {
        (m if isinstance(m, str) else m.value): 0 for m in payload.measures
    }
    for (row_key, col_key), values in cell_values.items():
        cells.append(
            PivotCell(
                row_key=[_stringify(v) for v in row_key],
                col_key=[_stringify(v) for v in col_key],
                values={k: (v if v is not None else None) for k, v in values.items()},
            )
        )
        for k, v in values.items():
            if v is None:
                continue
            grand[k] = grand.get(k, 0) + v

    return PivotResponse(
        rows=payload.rows,
        columns=payload.columns,
        measures=payload.measures,
        currency=currency,
        cells=cells,
        grand_total=grand,
        warnings=top_warnings,
    )


def _stringify(value: object) -> str:
    """Pivot keys go on the wire as strings; coerce enums + None safely."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


__all__ = ["router"]
