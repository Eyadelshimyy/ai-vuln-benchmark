"""Industry cedant: residential, commercial, and auto are informational.

The book is a selectable WS reference cedant. It is not part of the silent
in-force portfolio. Map TIV, detail summary, and hurricane-impact TIV are the
sum of the three segments. The segment amounts stay on the wire for display.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.models.enums import AggregationLevel, OccupancySegment, Peril
from app.models.exposure import ExposureFactNormalized
from app.providers.mock import MockExposureDataProvider
from app.services.hurricane_impact import CountyImpact, join_tiv


client = TestClient(app)

MIAMI = "US-FL-12086"
GLADES = "US-FL-12043"
# Digits read from the supplied workbook. A rebuild that drifts fails here.
MIAMI_RES = 386_320_674_139
MIAMI_COM = 677_318_120_794
MIAMI_AUTO = 36_785_091_163
MIAMI_BUNDLED = MIAMI_RES + MIAMI_COM + MIAMI_AUTO


@pytest.fixture(scope="module")
def provider() -> MockExposureDataProvider:
    return MockExposureDataProvider(get_settings().mock_data_dir)


def _value(raw: object) -> str:
    return raw.value if hasattr(raw, "value") else str(raw)


def _fact(
    *,
    geography_id: str,
    aggregation: AggregationLevel,
    segment: OccupancySegment,
    tiv: float,
) -> ExposureFactNormalized:
    return ExposureFactNormalized(
        dataset_id="ds-industry-ws",
        portname="01012026",
        source_server_name="REFERENCE",
        source_database_name="Industry_US_Property_WS",
        source_table_name="industry_exposure_estimate",
        aggregation=aggregation,
        geography_level=aggregation,
        geography_id=geography_id,
        peril=Peril.WS,
        occupancy="All",
        occupancy_group="All",
        occupancy_segment=segment,
        building=tiv * 0.6,
        contents=tiv * 0.3,
        bi=tiv * 0.1,
        tiv=tiv,
        currency="USD",
    )


def _source_rows() -> list[dict[str, str]]:
    path = Path(get_settings().mock_data_dir) / "usws_tiv_by_county.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bare_impact(geography_id: str = MIAMI) -> CountyImpact:
    return CountyImpact(
        geoid=geography_id.rsplit("-", 1)[-1],
        geography_id=geography_id,
        name="Miami-Dade",
        state_usps="FL",
        centroid_lat=25.6,
        centroid_lon=-80.4,
        max_wind_kt=110,
        max_category=3,
        closest_distance_nm=12.0,
        rmax_at_closest_nm=18.0,
        rmax_source="nhc",
        tiv=0.0,
        location_count=0,
        has_data=False,
    )


def test_industry_cedent_is_selectable_and_excluded_from_portfolio(
    provider: MockExposureDataProvider,
) -> None:
    cedents = provider.list_cedents()
    assert "ced-industry" in {c.cedent_id for c in cedents}
    industry = provider.get_cedent("ced-industry")
    assert industry is not None
    assert industry.cedent_name == "Industry"
    assert industry.region == "Reference"
    assert len(industry.chains) == 1
    chain = industry.chains[0]
    assert chain.office == "REF"
    assert chain.chain_id == "chain-industry-us-ws"
    assert len(chain.programmes) == 1
    programme = chain.programmes[0]
    assert programme.dataset_id == "ds-industry-ws"
    assert programme.include_in_portfolio is False
    assert programme.signed_share == 1.0
    assert [_value(p) for p in programme.perils] == ["WS"]
    assert programme.is_in_force(datetime(2026, 10, 8, tzinfo=timezone.utc))

    for other in cedents:
        if other.cedent_id == "ced-industry":
            continue
        assert other.region != "Reference"
        for other_chain in other.chains:
            assert other_chain.office != "REF"

    assert "ds-industry-ws" not in provider.portfolio_dataset_ids(in_force_only=True)
    loaded = {fact.dataset_id for fact in provider.get_portfolio_facts()}
    assert "ds-industry-ws" not in loaded

    datasets_path = Path(get_settings().mock_data_dir) / "datasets.json"
    datasets = json.loads(datasets_path.read_text(encoding="utf-8"))
    assert all(row.get("datasetId") != "ds-industry-ws" for row in datasets)


def test_industry_facts_identity_and_exact_rollups(
    provider: MockExposureDataProvider,
) -> None:
    facts = provider.get_facts_for_dataset("ds-industry-ws")
    assert facts
    summary = json.loads(
        (Path(get_settings().mock_data_dir) / "industry_exposure_summary.json").read_text(
            encoding="utf-8"
        )
    )

    county = [f for f in facts if _value(f.aggregation) == AggregationLevel.COUNTY.value]
    state = [f for f in facts if _value(f.aggregation) == AggregationLevel.STATE.value]
    country = [f for f in facts if _value(f.aggregation) == AggregationLevel.COUNTRY.value]

    expected_segments = {
        OccupancySegment.RESIDENTIAL.value,
        OccupancySegment.COMMERCIAL.value,
        OccupancySegment.AUTO.value,
    }
    by_county: dict[str, set[str]] = {}
    totals = {segment: 0.0 for segment in expected_segments}
    for fact in facts:
        assert _value(fact.peril) == Peril.WS.value
        assert fact.occupancy == "All"
        assert fact.occupancy_group == "All"
        assert fact.building == fact.tiv
        assert fact.contents == 0
        assert fact.bi == 0
        assert round(fact.tiv, 2) == round(fact.building + fact.contents + fact.bi, 2)
        assert round(fact.explim_gross, 2) == round(fact.tiv, 2)
        assert round(fact.explim_net, 2) == round(fact.tiv, 2)
        segment = _value(fact.occupancy_segment)
        assert segment in expected_segments

    for fact in county:
        segment = _value(fact.occupancy_segment)
        by_county.setdefault(fact.geography_id, set()).add(segment)
        totals[segment] += fact.tiv

    source = _source_rows()
    assert len(by_county) == summary["counties"] == len(source)
    assert all(segments == expected_segments for segments in by_county.values())
    assert round(totals["RESIDENTIAL"], 2) == summary["residentialTiv"]
    assert round(totals["COMMERCIAL"], 2) == summary["commercialTiv"]
    assert round(totals["AUTO"], 2) == summary["autoTiv"]
    assert round(sum(totals.values()), 2) == summary["totalTiv"]
    assert summary["totalTiv"] > 5e12

    county_by_state: dict[str, float] = {}
    for fact in county:
        assert fact.statecode
        county_by_state[fact.statecode] = county_by_state.get(fact.statecode, 0.0) + fact.tiv
    state_by: dict[str, float] = {}
    for fact in state:
        assert fact.statecode
        state_by[fact.statecode] = state_by.get(fact.statecode, 0.0) + fact.tiv
    assert set(state_by) == set(county_by_state)
    for code, tiv in state_by.items():
        assert round(tiv, 2) == round(county_by_state[code], 2), code
    assert {"FL", "NY", "TX", "LA", "HI"}.issubset(state_by)
    assert "CA" not in state_by
    assert "AK" not in state_by
    assert "PR" not in state_by
    assert round(sum(fact.tiv for fact in country), 2) == round(sum(fact.tiv for fact in county), 2)
    assert len(country) == 3
    miami_source = next(row for row in source if row["geographyId"] == MIAMI)
    assert int(miami_source["residentialTiv"]) == MIAMI_RES
    assert int(miami_source["commercialTiv"]) == MIAMI_COM
    assert int(miami_source["autoTiv"]) == MIAMI_AUTO


def test_industry_map_and_detail_bundle_three_segments() -> None:
    mapped = client.post(
        "/api/exposures/map",
        json={
            "cedentId": "ced-industry",
            "aggregationLevel": AggregationLevel.COUNTY.value,
            "metric": "TIV",
        },
    )
    assert mapped.status_code == 200, mapped.text
    features = {row["geographyId"]: row for row in mapped.json()["features"]}
    miami = features[MIAMI]
    glades = features[GLADES]
    assert miami["metricValue"] == miami["tiv"]
    assert miami["tiv"] > glades["tiv"] * 20

    detail = client.post(
        "/api/exposures/detail",
        json={
            "cedentId": "ced-industry",
            "aggregationLevel": AggregationLevel.COUNTY.value,
            "metric": "TIV",
            "geographyId": MIAMI,
        },
    )
    assert detail.status_code == 200, detail.text
    body = detail.json()
    occupancy = {row["key"]: row["tiv"] for row in body["breakdowns"]["occupancy"]}
    assert set(occupancy) == {"RESIDENTIAL", "COMMERCIAL", "AUTO"}
    assert occupancy["RESIDENTIAL"] == MIAMI_RES
    assert occupancy["COMMERCIAL"] == MIAMI_COM
    assert occupancy["AUTO"] == MIAMI_AUTO
    bundled = occupancy["RESIDENTIAL"] + occupancy["COMMERCIAL"] + occupancy["AUTO"]
    assert bundled == MIAMI_BUNDLED
    assert round(body["summary"]["tiv"], 2) == round(bundled, 2)
    assert round(miami["tiv"], 2) == round(bundled, 2)

    glades_detail = client.post(
        "/api/exposures/detail",
        json={
            "cedentId": "ced-industry",
            "aggregationLevel": AggregationLevel.COUNTY.value,
            "metric": "TIV",
            "geographyId": GLADES,
        },
    )
    assert glades_detail.status_code == 200, glades_detail.text
    glades_occ = {
        row["key"]: row["tiv"] for row in glades_detail.json()["breakdowns"]["occupancy"]
    }
    assert occupancy["RESIDENTIAL"] > glades_occ["RESIDENTIAL"] * 20

    portfolio = client.post(
        "/api/exposures/map",
        json={
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": "TIV",
        },
    )
    assert portfolio.status_code == 200, portfolio.text
    industry_state = client.post(
        "/api/exposures/map",
        json={
            "cedentId": "ced-industry",
            "aggregationLevel": AggregationLevel.STATE.value,
            "metric": "TIV",
        },
    )
    assert industry_state.status_code == 200, industry_state.text
    portfolio_fl = next(
        (row for row in portfolio.json()["features"] if row["geographyId"] == "US-FL"),
        None,
    )
    industry_fl = next(row for row in industry_state.json()["features"] if row["geographyId"] == "US-FL")
    portfolio_fl_tiv = 0.0 if portfolio_fl is None else portfolio_fl["tiv"]
    assert industry_fl["tiv"] > portfolio_fl_tiv * 10


def _industry_map(segments: list[str], level: AggregationLevel = AggregationLevel.COUNTY) -> dict:
    response = client.post(
        "/api/exposures/map",
        json={
            "cedentId": "ced-industry",
            "aggregationLevel": level.value,
            "metric": "TIV",
            "filters": {"occupancySegment": segments},
        },
    )
    assert response.status_code == 200, response.text
    return {row["geographyId"]: row for row in response.json()["features"]}


def test_industry_map_segment_filter_selects_segments() -> None:
    assert _industry_map(["AUTO"])[MIAMI]["tiv"] == MIAMI_AUTO
    assert _industry_map(["RESIDENTIAL"])[MIAMI]["tiv"] == MIAMI_RES
    assert _industry_map(["RESIDENTIAL", "COMMERCIAL"])[MIAMI]["tiv"] == MIAMI_RES + MIAMI_COM
    assert _industry_map([])[MIAMI]["tiv"] == MIAMI_BUNDLED

    summary = json.loads(
        (Path(get_settings().mock_data_dir) / "industry_exposure_summary.json").read_text()
    )
    country = _industry_map(["AUTO"], AggregationLevel.COUNTRY)
    assert round(sum(row["tiv"] for row in country.values())) == summary["autoTiv"]


def test_industry_segment_filter_keeps_market_share_on_the_same_segments() -> None:
    for segments in (["AUTO"], ["RESIDENTIAL", "COMMERCIAL"], []):
        miami = _industry_map(segments)[MIAMI]
        assert miami["clientMarketShare"] == pytest.approx(1.0)

    detail = client.post(
        "/api/exposures/detail",
        json={
            "cedentId": "ced-industry",
            "aggregationLevel": AggregationLevel.COUNTY.value,
            "metric": "TIV",
            "geographyId": MIAMI,
            "filters": {"occupancySegment": ["COMMERCIAL", "AUTO"]},
        },
    )
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["summary"]["tiv"] == MIAMI_COM + MIAMI_AUTO
    assert body["marketShare"]["industryTiv"] == MIAMI_COM + MIAMI_AUTO
    assert body["marketShare"]["segment"] == "AUTO+COMMERCIAL"
    assert {row["key"] for row in body["breakdowns"]["occupancy"]} == {"COMMERCIAL", "AUTO"}


def test_join_tiv_bundles_segments_and_ignores_state_rows() -> None:
    """Impact TIV is one number. The three segments stay informational."""
    residential = 100.0
    commercial = 250.0
    auto = 40.0
    facts = [
        _fact(
            geography_id=MIAMI,
            aggregation=AggregationLevel.COUNTY,
            segment=OccupancySegment.RESIDENTIAL,
            tiv=residential,
        ),
        _fact(
            geography_id=MIAMI,
            aggregation=AggregationLevel.COUNTY,
            segment=OccupancySegment.COMMERCIAL,
            tiv=commercial,
        ),
        _fact(
            geography_id=MIAMI,
            aggregation=AggregationLevel.COUNTY,
            segment=OccupancySegment.AUTO,
            tiv=auto,
        ),
        _fact(
            geography_id="US-FL",
            aggregation=AggregationLevel.STATE,
            segment=OccupancySegment.RESIDENTIAL,
            tiv=9_999.0,
        ),
    ]
    impact = join_tiv([_bare_impact()], facts)[0]
    assert impact.has_data is True
    assert impact.tiv == residential + commercial + auto
    assert impact.residential_tiv == residential
    assert impact.commercial_tiv == commercial
    assert impact.auto_tiv == auto
    # The server does not emit a second loss. Callers multiply this one tiv.
    assert impact.projected_loss == 0.0
    assert impact.location_count == 0
    assert len(impact.by_programme) == 1
    assert impact.by_programme[0].tiv == residential + commercial + auto


def test_ied_denominator_matches_industry_book_and_stays_sane(
    provider: MockExposureDataProvider,
) -> None:
    """Market share uses the Industry client, not the old trillion-scale fixture."""
    summary = json.loads(
        (Path(get_settings().mock_data_dir) / "industry_exposure_summary.json").read_text(
            encoding="utf-8"
        )
    )
    ied = provider.get_ied_industry()

    def total(level: str, gid: str, segment: str | None = None) -> float:
        matched = [
            row.industry_tiv
            for row in ied
            if _value(row.geography_level) == level
            and row.geography_id == gid
            and (segment is None or _value(row.occupancy_segment) == segment)
        ]
        assert matched, f"missing IED {level} {gid} {segment}"
        return sum(matched)

    assert total("COUNTRY", "US", "RESIDENTIAL") == summary["residentialTiv"]
    assert total("COUNTRY", "US", "COMMERCIAL") == summary["commercialTiv"]
    assert total("COUNTRY", "US", "AUTO") == summary["autoTiv"]
    assert total("COUNTRY", "US") == summary["totalTiv"]
    # No UNKNOWN row. Summing every segment must not double-count the country.
    assert {_value(row.occupancy_segment) for row in ied} == {"RESIDENTIAL", "COMMERCIAL", "AUTO"}

    facts = provider.get_facts_for_dataset("ds-industry-ws")
    fl_facts = [
        fact.tiv
        for fact in facts
        if _value(fact.aggregation) == AggregationLevel.COUNTY.value and fact.statecode == "FL"
    ]
    assert round(total("STATE", "US-FL"), 2) == round(sum(fl_facts), 2)

    county_ids = {
        row.geography_id
        for row in ied
        if _value(row.geography_level) == AggregationLevel.COUNTY.value
    }
    assert "US-FL-12086" in county_ids
    assert total("COUNTY", "US-FL-12086") == MIAMI_BUNDLED
    assert "US-CA-06037" not in county_ids
    assert "US-FL-12011" in county_ids


def test_county_reference_reports_industry_book_tiv() -> None:
    ref = client.get("/api/counties/US-FL-12086/reference")
    assert ref.status_code == 200, ref.text
    body = ref.json()
    assert body["industryResidentialTiv"] == MIAMI_RES
    assert body["industryCommercialTiv"] == MIAMI_COM
    assert body["industryAutoTiv"] == MIAMI_AUTO
    assert body["industryTiv"] == MIAMI_BUNDLED
    assert body["industryTiv"] == (
        body["industryResidentialTiv"] + body["industryCommercialTiv"] + body["industryAutoTiv"]
    )


def test_industry_pivot_is_tiv_by_county() -> None:
    pivoted = client.post(
        "/api/exposures/pivot",
        json={
            "cedentId": "ced-industry",
            "rows": ["STATE", "COUNTY"],
            "columns": ["OCCUPANCY_SEGMENT"],
            "measures": ["TIV"],
        },
    )
    assert pivoted.status_code == 200, pivoted.text
    cells = pivoted.json()["cells"]
    miami = [
        cell
        for cell in cells
        if cell["rowKey"][0] == "FL" and cell["rowKey"][1] == "Miami-Dade, FL"
    ]
    assert {cell["colKey"][0] for cell in miami} == {"RESIDENTIAL", "COMMERCIAL", "AUTO"}
    bundled = sum(cell["values"]["TIV"] for cell in miami)
    assert bundled == MIAMI_BUNDLED
    summary = json.loads(
        (Path(get_settings().mock_data_dir) / "industry_exposure_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert pivoted.json()["grandTotal"]["TIV"] == summary["totalTiv"]


def test_join_tiv_on_industry_county_matches_three_segments(
    provider: MockExposureDataProvider,
) -> None:
    facts = [
        fact
        for fact in provider.get_facts_for_dataset("ds-industry-ws")
        if fact.geography_id == MIAMI
    ]
    impact = join_tiv([_bare_impact()], facts)[0]
    assert impact.residential_tiv == MIAMI_RES
    assert impact.commercial_tiv == MIAMI_COM
    assert impact.auto_tiv == MIAMI_AUTO
    assert round(impact.tiv, 2) == round(
        impact.residential_tiv + impact.commercial_tiv + impact.auto_tiv, 2
    )
    assert impact.projected_loss == 0.0
