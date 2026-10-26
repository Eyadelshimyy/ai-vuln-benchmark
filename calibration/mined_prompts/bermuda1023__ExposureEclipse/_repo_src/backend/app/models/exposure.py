"""Exposure request/response models — shapes per API_SPEC.md + DATA_MODEL.md.

`ExposureFactNormalized` is the conceptual shape every provider must serve and every
calculation operates on. `MapRequest` / `MapResponse` etc are the wire contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .common import CamelModel
from .enums import (
    AggregationLevel,
    CombinationMethod,
    Measure,
    MetricKey,
    OccupancySegment,
    Peril,
    PortfolioScope,
    YoyStatus,
)
from .warnings import Warning


# ───────────────────────── Normalized internal shape ─────────────────────────


class ExposureFactNormalized(CamelModel):
    """Per DATA_MODEL.md — the analytical row calc operates on, provider-independent.

    Closest real analogue is the ERT `Evolution` cut (ERT_OUTPUT_FORMAT.md §2).
    Numeric values are in `currency`. Counts are integers.
    """

    dataset_id: str
    dataset_group_id: str | None = None
    portname: str  # MMDDYYYY snapshot (ERT PORTNAME)

    source_server_name: str
    source_database_name: str
    source_table_name: str

    aggregation: AggregationLevel
    geography_level: AggregationLevel

    country: str | None = None
    country_name: str | None = None
    statecode: str | None = None
    state_name: str | None = None
    county: str | None = None
    county_name: str | None = None
    cresta: str | None = None
    cresta_name: str | None = None
    geography_id: str  # canonical key — `US`, `US-FL`, `US-FL-12086`, `CRESTA-…`

    peril: Peril
    occupancy: str | None = None
    occupancy_group: str | None = None
    occupancy_segment: OccupancySegment = OccupancySegment.UNKNOWN

    construction: str | None = None
    year_built: str | None = None
    distance_to_coast: str | None = None
    geocoding_quality: str | None = None
    number_of_stories: str | None = None

    building: float = 0.0
    contents: float = 0.0
    bi: float = 0.0
    tiv: float = 0.0
    explim_gross: float = 0.0
    explim_net: float = 0.0
    location_count: int = 0
    account_count: int | None = None
    invalid_tiv: float | None = None
    invalid_count: int | None = None

    currency: str
    exposure_data_cutoff_date: datetime | None = None


# ───────────────────────── Filter / view inputs ─────────────────────────


class ExposureFilters(CamelModel):
    """Mirror `filters` block in API_SPEC.md POST bodies. `peril` is `Peril | "ALL"`."""

    peril: Peril = Peril.ALL
    occupancy: list[str] = []
    # Empty = every segment. The market-share denominator follows the same set.
    occupancy_segment: list[OccupancySegment] = []
    distance_to_coast: list[str] = []
    geocoding: list[str] = []
    construction: list[str] = []
    number_of_stories: list[str] = []
    year_built: list[str] = []


# ───────────────────────── Map endpoint ─────────────────────────


class MapRequest(CamelModel):
    # ── Primary navigation: exactly one of these identifies what to view ──
    # Pick a programme, a chain (auto-pairs latest vs prior), or a cedent
    # (combines all the cedent's chains under MAX_ACROSS_PERILS_AT_VIEW_GRAIN).
    # `dataset_id` / `dataset_group_id` are legacy and still accepted.
    programme_id: str | None = None
    chain_id: str | None = None
    chain_ids: list[str] = Field(default_factory=list)
    cedent_id: str | None = None
    dataset_id: str | None = None
    dataset_group_id: str | None = None

    portfolio_scope: PortfolioScope = PortfolioScope.ALL_LOADED_DATASETS
    aggregation_level: AggregationLevel
    metric: MetricKey = MetricKey.TIV
    filters: ExposureFilters = Field(default_factory=ExposureFilters)

    # ── Comparison (YoY) selection ──
    # When a chain is picked, the prior programme in the chain is used by
    # default. `comparison_programme_id` overrides which year to compare to.
    # `comparison_dataset_id` is the legacy escape hatch.
    comparison_programme_id: str | None = None
    comparison_dataset_id: str | None = None

    # ── Peril multi-select (top-of-page filter) ──
    # Empty list or [ALL] means "all perils". Multiple perils combine under
    # MAX_ACROSS_PERILS_AT_VIEW_GRAIN per CLAUDE.md rule 3.
    perils: list[Peril] = Field(default_factory=list)

    currency_assumption: dict[str, float] | None = None  # {fromCurr: rate-to-display-curr}
    # When True AND a comparison is set, `metricValue` becomes the YoY change
    # of the chosen metric (current vs prior at the same grain).
    yoy_mode: bool = False


class MapFeature(CamelModel):
    geography_id: str
    geography_name: str | None = None
    metric_value: float | None = None  # mirrors the requested `metric` (or YoY of it)
    # Prior-period value of the selected metric at the same grain. Populated
    # ONLY when a comparison dataset is set. Lets the UI show "current → prior"
    # and the dollar/count delta without recomputing on the client.
    prior_metric_value: float | None = None
    tiv: float | None = None
    location_count: int | None = None
    deal_share_of_portfolio_in_geography: float | None = None
    geography_share_of_total_portfolio: float | None = None
    selected_deal_geography_concentration: float | None = None
    client_market_share: float | None = None
    yoy_change: float | None = None
    yoy_status: YoyStatus | None = None
    has_geometry: bool = True
    warnings: list[Warning] = []


class MapResponse(CamelModel):
    aggregation_level: AggregationLevel
    metric: MetricKey
    currency: str
    features: list[MapFeature]
    warnings: list[Warning] = []


# ───────────────────────── Detail endpoint ─────────────────────────


class DetailRequest(MapRequest):
    geography_id: str


class DetailSummary(CamelModel):
    tiv: float | None = None
    location_count: int | None = None
    deal_share_of_portfolio_in_geography: float | None = None
    geography_share_of_total_portfolio: float | None = None
    selected_deal_geography_concentration: float | None = None
    client_market_share: float | None = None
    yoy_change: float | None = None
    yoy_status: YoyStatus | None = None


class DealVsPortfolio(CamelModel):
    deal_tiv: float
    portfolio_tiv: float


class MarketShareDetail(CamelModel):
    client_tiv: float | None
    industry_tiv: float | None
    share: float | None
    segment: str = "ALL"


class YoyDetail(CamelModel):
    current_tiv: float | None
    prior_tiv: float | None
    change: float | None
    status: YoyStatus


class BreakdownRow(CamelModel):
    key: str
    tiv: float
    location_count: int


class DetailBreakdowns(CamelModel):
    peril: list[BreakdownRow] = []
    occupancy: list[BreakdownRow] = []
    distance_to_coast: list[BreakdownRow] = []
    geocoding: list[BreakdownRow] = []
    stories: list[BreakdownRow] = []
    construction: list[BreakdownRow] = []


class DetailResponse(CamelModel):
    geography_id: str
    geography_name: str | None = None
    aggregation_level: AggregationLevel
    currency: str
    summary: DetailSummary
    deal_vs_portfolio: DealVsPortfolio
    market_share: MarketShareDetail
    yoy: YoyDetail
    breakdowns: DetailBreakdowns
    active_filters: dict[str, Any] = {}
    warnings: list[Warning] = []


# ───────────────────────── Pivot endpoint ─────────────────────────


class PivotRequest(CamelModel):
    # Same one-of identifier surface as MapRequest.
    programme_id: str | None = None
    chain_id: str | None = None
    chain_ids: list[str] = Field(default_factory=list)
    cedent_id: str | None = None
    dataset_id: str | None = None
    dataset_group_id: str | None = None
    portfolio_scope: PortfolioScope = PortfolioScope.ALL_LOADED_DATASETS
    rows: list[str] = []  # aggregation levels and/or dimension keys
    columns: list[str] = []
    measures: list[Measure]
    filters: ExposureFilters = Field(default_factory=ExposureFilters)
    comparison_programme_id: str | None = None
    comparison_dataset_id: str | None = None
    combination_method: CombinationMethod | None = None
    perils: list[Peril] = Field(default_factory=list)
    currency_assumption: dict[str, float] | None = None


class PivotCell(CamelModel):
    row_key: list[str]
    col_key: list[str]
    values: dict[str, float | int | None]


class PivotResponse(CamelModel):
    rows: list[str]
    columns: list[str]
    measures: list[Measure]
    currency: str
    cells: list[PivotCell]
    row_totals: list[PivotCell] = []
    column_totals: list[PivotCell] = []
    grand_total: dict[str, float | int | None] = {}
    warnings: list[Warning] = []


# ───────────────────────── Industry-TIV (IED) row ─────────────────────────


class IEDIndustryRow(CamelModel):
    """RMS IED denominator row — internal shape, not currently on the public wire."""

    geography_level: AggregationLevel
    geography_id: str
    occupancy_segment: OccupancySegment = OccupancySegment.UNKNOWN
    industry_tiv: float
    currency: str
    source_year: int | None = None
