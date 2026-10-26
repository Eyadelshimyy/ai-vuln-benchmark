"""Tests for ``app.services.calculations`` — the formulas in CALCULATION_RULES.md.

Each test name traces back to a row in the calc rules / contracts so a
failure points directly at the relevant rule. Heavy coverage lives here per
TEST_PLAN.md.
"""

from __future__ import annotations

from app.models.enums import (
    AggregationLevel,
    OccupancySegment,
    Peril,
    WarningCode,
    YoyStatus,
)
from app.models.exposure import ExposureFactNormalized, ExposureFilters, IEDIndustryRow
from app.services.calculations import (
    aggregate_location_count,
    aggregate_tiv,
    apply_filters,
    client_market_share,
    deal_share_of_portfolio_in_geography,
    geography_share_of_total_portfolio,
    lookup_industry_tiv,
    selected_deal_geography_concentration,
    yoy_change,
)


# ───────────────────────── Fixture helpers ─────────────────────────


def fact(
    *,
    statecode: str = "FL",
    state_name: str | None = None,
    peril: Peril = Peril.WS,
    tiv: float = 0.0,
    locations: int = 0,
    occupancy: str | None = None,
    occupancy_segment: OccupancySegment = OccupancySegment.UNKNOWN,
    distance_to_coast: str | None = None,
    geocoding_quality: str | None = None,
    construction: str | None = None,
    number_of_stories: str | None = None,
    year_built: str | None = None,
    aggregation: AggregationLevel = AggregationLevel.STATE,
    dataset_id: str = "ds-test",
    currency: str = "USD",
) -> ExposureFactNormalized:
    """Minimal :class:`ExposureFactNormalized` with sensible defaults.

    ``geography_id`` is derived as ``US-<statecode>`` for state-grain rows so
    callers don't have to repeat it everywhere.
    """
    return ExposureFactNormalized(
        dataset_id=dataset_id,
        portname="09302025",
        source_server_name="srv",
        source_database_name="db",
        source_table_name="tbl",
        aggregation=aggregation,
        geography_level=aggregation,
        country="US",
        country_name="United States",
        statecode=statecode,
        state_name=state_name or statecode,
        geography_id=f"US-{statecode}" if statecode else "US",
        peril=peril,
        occupancy=occupancy,
        occupancy_segment=occupancy_segment,
        construction=construction,
        year_built=year_built,
        distance_to_coast=distance_to_coast,
        geocoding_quality=geocoding_quality,
        number_of_stories=number_of_stories,
        tiv=tiv,
        location_count=locations,
        currency=currency,
    )


# ───────────────────────── aggregate_tiv ─────────────────────────


def test_aggregate_tiv_groups_correctly_across_perils_and_states() -> None:
    facts = [
        fact(statecode="FL", peril=Peril.WS, tiv=12_400_000_000),
        fact(statecode="FL", peril=Peril.EQ, tiv=9_100_000_000),
        fact(statecode="FL", peril=Peril.CS, tiv=7_800_000_000),
        fact(statecode="CA", peril=Peril.WS, tiv=3_000_000_000),
        fact(statecode="CA", peril=Peril.EQ, tiv=14_200_000_000),
    ]
    by_state = aggregate_tiv(facts, by=("statecode",))
    # Aggregation sums across perils within a state — this is *raw* aggregation
    # (no peril-combination logic). Caller decides whether to combine.
    assert by_state[("FL",)] == 12_400_000_000 + 9_100_000_000 + 7_800_000_000
    assert by_state[("CA",)] == 3_000_000_000 + 14_200_000_000

    by_state_peril = aggregate_tiv(facts, by=("statecode", "peril"))
    assert by_state_peril[("FL", Peril.WS)] == 12_400_000_000
    assert by_state_peril[("CA", Peril.EQ)] == 14_200_000_000


def test_aggregate_location_count_sums_integers() -> None:
    facts = [
        fact(statecode="FL", locations=100),
        fact(statecode="FL", locations=50),
        fact(statecode="CA", locations=10),
    ]
    counts = aggregate_location_count(facts, by=("statecode",))
    assert counts[("FL",)] == 150
    assert counts[("CA",)] == 10


# ───────────────────────── Ratios ─────────────────────────


def test_deal_share_of_portfolio_in_geography_basic() -> None:
    assert deal_share_of_portfolio_in_geography(25.0, 100.0) == 0.25


def test_deal_share_of_portfolio_returns_none_when_denominator_zero() -> None:
    assert deal_share_of_portfolio_in_geography(25.0, 0) is None


def test_deal_share_of_portfolio_returns_none_when_denominator_missing() -> None:
    assert deal_share_of_portfolio_in_geography(25.0, None) is None


def test_geography_share_of_total_portfolio_returns_none_when_total_zero() -> None:
    assert geography_share_of_total_portfolio(50.0, 0) is None


def test_geography_share_of_total_portfolio_basic() -> None:
    assert geography_share_of_total_portfolio(50.0, 200.0) == 0.25


def test_selected_deal_geography_concentration_basic_and_zero() -> None:
    assert selected_deal_geography_concentration(40.0, 100.0) == 0.4
    assert selected_deal_geography_concentration(40.0, 0) is None


# ───────────────────────── Client market share ─────────────────────────


def test_client_market_share_present_returns_ratio_no_warning() -> None:
    share, warning = client_market_share(20.0, 100.0)
    assert share == 0.2
    assert warning is None


def test_client_market_share_missing_industry_returns_warning() -> None:
    share, warning = client_market_share(20.0, None)
    assert share is None
    assert warning is not None
    assert warning.code == WarningCode.WARN_IED_DENOMINATOR_MISSING


def test_client_market_share_zero_industry_returns_none_no_warning() -> None:
    # Industry data was present but zero — that's a math problem, not a
    # missing-denominator problem; no IED warning.
    share, warning = client_market_share(20.0, 0)
    assert share is None
    assert warning is None


# ───────────────────────── lookup_industry_tiv ─────────────────────────


def _ied(
    geography_id: str,
    segment: OccupancySegment,
    tiv: float,
) -> IEDIndustryRow:
    return IEDIndustryRow(
        geography_level=AggregationLevel.STATE,
        geography_id=geography_id,
        occupancy_segment=segment,
        industry_tiv=tiv,
        currency="USD",
    )


def test_lookup_industry_tiv_matches_geography_and_segment() -> None:
    rows = [
        _ied("US-FL", OccupancySegment.RESIDENTIAL, 1_000),
        _ied("US-FL", OccupancySegment.COMMERCIAL, 500),
        _ied("US-CA", OccupancySegment.RESIDENTIAL, 2_000),
    ]
    assert lookup_industry_tiv(rows, "US-FL", OccupancySegment.RESIDENTIAL) == 1_000
    assert lookup_industry_tiv(rows, "US-CA", OccupancySegment.RESIDENTIAL) == 2_000


def test_lookup_industry_tiv_without_segment_sums_all_segments() -> None:
    rows = [
        _ied("US-FL", OccupancySegment.RESIDENTIAL, 1_000),
        _ied("US-FL", OccupancySegment.COMMERCIAL, 500),
        _ied("US-FL", OccupancySegment.UNKNOWN, 100),
    ]
    assert lookup_industry_tiv(rows, "US-FL") == 1_600


def test_lookup_industry_tiv_unknown_segment_not_coerced() -> None:
    # UNKNOWN must not be force-mapped (CONTRACTS §6 / CLAUDE.md). Asking for
    # RESIDENTIAL must NOT pick up the UNKNOWN row.
    rows = [_ied("US-FL", OccupancySegment.UNKNOWN, 999)]
    assert lookup_industry_tiv(rows, "US-FL", OccupancySegment.RESIDENTIAL) is None
    # And UNKNOWN itself stays addressable.
    assert lookup_industry_tiv(rows, "US-FL", OccupancySegment.UNKNOWN) == 999


def test_lookup_industry_tiv_missing_geography_returns_none() -> None:
    rows = [_ied("US-FL", OccupancySegment.RESIDENTIAL, 1_000)]
    assert lookup_industry_tiv(rows, "US-TX", OccupancySegment.RESIDENTIAL) is None


# ───────────────────────── YoY truth table ─────────────────────────


def test_yoy_change_ok() -> None:
    change, status = yoy_change(120.0, 100.0)
    assert status == YoyStatus.OK
    assert change == 0.2


def test_yoy_change_new_when_prior_missing() -> None:
    change, status = yoy_change(120.0, None)
    assert status == YoyStatus.NEW
    assert change is None


def test_yoy_change_removed_when_current_missing() -> None:
    change, status = yoy_change(None, 100.0)
    assert status == YoyStatus.REMOVED
    assert change is None


def test_yoy_change_na_when_prior_is_zero() -> None:
    change, status = yoy_change(120.0, 0)
    assert status == YoyStatus.NA
    assert change is None


def test_yoy_change_both_missing_yields_na() -> None:
    change, status = yoy_change(None, None)
    assert status == YoyStatus.NA
    assert change is None


# ───────────────────────── Filters ─────────────────────────


def test_apply_filters_peril_all_is_passthrough() -> None:
    facts = [
        fact(peril=Peril.WS, tiv=1),
        fact(peril=Peril.EQ, tiv=2),
        fact(peril=Peril.CS, tiv=3),
    ]
    out = apply_filters(facts, ExposureFilters(peril=Peril.ALL))
    assert len(out) == 3


def test_apply_filters_specific_peril_drops_others() -> None:
    facts = [
        fact(peril=Peril.WS, tiv=1),
        fact(peril=Peril.EQ, tiv=2),
        fact(peril=Peril.CS, tiv=3),
    ]
    out = apply_filters(facts, ExposureFilters(peril=Peril.EQ))
    assert [f.peril for f in out] == [Peril.EQ]


def test_apply_filters_occupancy_list_filters() -> None:
    facts = [
        fact(occupancy="Permanent", tiv=1),
        fact(occupancy="Seasonal", tiv=2),
        fact(occupancy="Permanent", tiv=3),
    ]
    out = apply_filters(facts, ExposureFilters(occupancy=["Permanent"]))
    assert len(out) == 2
    assert all(f.occupancy == "Permanent" for f in out)


def test_apply_filters_empty_lists_are_no_op() -> None:
    facts = [
        fact(occupancy="Permanent", tiv=1, construction="Wood", year_built="1980 to 2000"),
        fact(occupancy="Seasonal", tiv=2, construction="Masonry", year_built="Unknown"),
    ]
    # All list dimensions empty → no filtering on those dims.
    out = apply_filters(facts, ExposureFilters())
    assert len(out) == 2


def test_apply_filters_combines_multiple_dimensions() -> None:
    facts = [
        fact(peril=Peril.WS, occupancy="Permanent", construction="Wood", tiv=1),
        fact(peril=Peril.WS, occupancy="Permanent", construction="Masonry", tiv=2),
        fact(peril=Peril.EQ, occupancy="Permanent", construction="Wood", tiv=3),
    ]
    out = apply_filters(
        facts,
        ExposureFilters(peril=Peril.WS, occupancy=["Permanent"], construction=["Wood"]),
    )
    assert len(out) == 1
    assert out[0].tiv == 1
