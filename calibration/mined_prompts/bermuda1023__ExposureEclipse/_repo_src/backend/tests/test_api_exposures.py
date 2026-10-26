"""Tests for `/api/exposures/map`, `/detail`, and `/pivot`."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from app.main import app
from app.models.enums import (
    AggregationLevel,
    Measure,
    MetricKey,
    WarningCode,
)
from app.providers import get_provider


client = TestClient(app)


# ───────────────────────── /map ─────────────────────────


def test_map_state_returns_features_with_metric_value_mirror() -> None:
    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["aggregationLevel"] == AggregationLevel.STATE.value
    assert body["currency"] == "USD"
    assert len(body["features"]) > 0
    fl = next((f for f in body["features"] if f["geographyId"] == "US-FL"), None)
    assert fl is not None
    # metricValue mirrors the requested metric (here = TIV)
    assert fl["metricValue"] == fl["tiv"]
    assert fl["tiv"] > 0


def test_single_multi_peril_dataset_uses_max_across_perils() -> None:
    provider = get_provider()
    facts = [
        fact
        for fact in provider.get_facts_for_dataset("ds-farmers-bda-2027")
        if fact.aggregation == AggregationLevel.STATE and fact.geography_id == "US-FL"
    ]
    by_peril: dict[str, float] = {}
    for fact in facts:
        by_peril[str(fact.peril)] = by_peril.get(str(fact.peril), 0.0) + fact.tiv

    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
        },
    )
    assert resp.status_code == 200, resp.text
    fl = next(f for f in resp.json()["features"] if f["geographyId"] == "US-FL")
    assert len(by_peril) > 1
    assert fl["tiv"] == max(by_peril.values())
    assert fl["tiv"] < sum(by_peril.values())


def test_mixed_currency_scope_is_blocked() -> None:
    resp = client.post(
        "/api/exposures/map",
        json={
            "chainIds": ["chain-farmers-bda", "chain-munich-lon-ws"],
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "CURRENCY_MISMATCH"


def test_portfolio_denominator_matches_selected_peril_and_currency() -> None:
    provider = get_provider()
    portfolio_tiv = sum(
        fact.tiv
        for fact in provider.get_portfolio_facts()
        if fact.aggregation == AggregationLevel.STATE
        and fact.geography_id == "US-FL"
        and str(fact.peril) == "WS"
        and fact.currency == "USD"
    )
    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.DEAL_SHARE_OF_PORTFOLIO_IN_GEOGRAPHY.value,
            "perils": ["WS"],
        },
    )
    assert resp.status_code == 200, resp.text
    fl = next(f for f in resp.json()["features"] if f["geographyId"] == "US-FL")
    assert portfolio_tiv > 0
    assert math.isclose(
        fl["dealShareOfPortfolioInGeography"],
        fl["tiv"] / portfolio_tiv,
        rel_tol=1e-12,
    )


def test_map_county_with_ied_gap_emits_market_share_warning() -> None:
    """Los Angeles is outside the supplied windstorm industry sheet."""
    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.COUNTY.value,
            "metric": MetricKey.CLIENT_MARKET_SHARE.value,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    gap = next((f for f in body["features"] if f["geographyId"] == "US-CA-06037"), None)
    assert gap is not None
    assert gap["clientMarketShare"] is None
    codes = {w["code"] for w in gap["warnings"]}
    assert WarningCode.WARN_IED_DENOMINATOR_MISSING.value in codes


def test_map_over_filtering_returns_no_rows_warning() -> None:
    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
            "filters": {
                "peril": "ALL",
                "occupancy": ["NO_SUCH_OCCUPANCY"],
                "distanceToCoast": [],
                "geocoding": [],
                "construction": [],
                "numberOfStories": [],
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["features"] == []
    codes = {w["code"] for w in body["warnings"]}
    assert WarningCode.WARN_FILTERS_RETURN_NO_ROWS.value in codes


def test_map_without_comparison_emits_prior_not_selected_warning() -> None:
    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    codes = {w["code"] for w in body["warnings"]}
    assert WarningCode.WARN_PRIOR_DATASET_NOT_SELECTED.value in codes


def test_map_allows_zero_targets_returns_portfolio() -> None:
    """No selection target → portfolio mode (union of all in-force programmes)."""
    resp = client.post(
        "/api/exposures/map",
        json={
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["features"]) > 0


def test_map_rejects_multiple_targets() -> None:
    """Two targets are still ambiguous and must 422."""
    resp = client.post(
        "/api/exposures/map",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "cedentId": "ced-farmers",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
        },
    )
    assert resp.status_code == 422


# ───────────────────────── /detail ─────────────────────────


def test_detail_returns_summary_and_breakdowns() -> None:
    resp = client.post(
        "/api/exposures/detail",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": MetricKey.TIV.value,
            "geographyId": "US-FL",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["geographyId"] == "US-FL"
    assert body["summary"]["tiv"] > 0
    assert body["summary"]["locationCount"] > 0
    assert body["dealVsPortfolio"]["portfolioTiv"] >= body["dealVsPortfolio"]["dealTiv"]
    assert "peril" in body["breakdowns"]


# ───────────────────────── /pivot ─────────────────────────


def test_pivot_returns_at_least_one_cell() -> None:
    resp = client.post(
        "/api/exposures/pivot",
        json={
            "datasetId": "ds-farmers-bda-2027",
            "rows": ["STATE"],
            "columns": ["PERIL"],
            "measures": [Measure.TIV.value, Measure.LOCATION_COUNT.value],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["currency"] == "USD"
    assert len(body["cells"]) >= 1
    sample = body["cells"][0]
    assert "rowKey" in sample
    assert "colKey" in sample
    assert Measure.TIV.value in sample["values"]
