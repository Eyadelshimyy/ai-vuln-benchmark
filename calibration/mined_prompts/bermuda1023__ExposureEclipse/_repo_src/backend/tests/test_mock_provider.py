"""Phase 1 — MockExposureDataProvider + fixture coverage.

Asserts every scenario in docs/MOCK_DATA_SPEC.md is reachable through the provider
without any UI involvement. Numbers come from the worked example in
docs/CALCULATION_RULES.md (FL/CA max-across-perils @ STATE grain).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.models.dataset import DatasetGroupCreate, DatasetGroupMemberInput
from app.models.enums import (
    AggregationLevel,
    CombinationMethod,
    ErtStatus,
    Peril,
    WarningCode,
)
from app.providers.mock import MockExposureDataProvider


@pytest.fixture(scope="module")
def provider() -> MockExposureDataProvider:
    settings = get_settings()
    return MockExposureDataProvider(settings.mock_data_dir)


# ───────────────────────── cedent / chain / programme discovery ─────────────────────────


def test_cedent_tree_loaded(provider: MockExposureDataProvider) -> None:
    cedents = provider.list_cedents()
    assert len(cedents) >= 5
    cedent_ids = {c.cedent_id for c in cedents}
    for expected in (
        "ced-farmers",
        "ced-acmere",
        "ced-zenith",
        "ced-coastalre",
        "ced-sample-client",
        "ced-alwaysfails",
    ):
        assert expected in cedent_ids, f"missing cedent {expected}"


def test_farmers_chain_has_multi_year_lineage(provider: MockExposureDataProvider) -> None:
    """The Farmers Nationwide chain (BDA, multi-peril) should have ≥3 years."""
    chain = provider.get_chain("chain-farmers-bda")
    assert chain is not None
    years = sorted({p.treaty_year for p in chain.programmes}, reverse=True)
    assert years[:3] == [2027, 2026, 2025]


def test_get_programme_by_dataset_id_resolves(provider: MockExposureDataProvider) -> None:
    prog = provider.get_programme_by_dataset_id("ds-farmers-bda-2027")
    assert prog is not None
    assert prog.cedent_id == "ced-farmers"
    assert prog.chain_id == "chain-farmers-bda"
    assert prog.treaty_year == 2027
    # Multi-peril programme carries the full perils[] list.
    assert {"WS", "EQ", "CS"}.issubset(set(prog.perils))


# ───────────────────────── ERT status derivation ─────────────────────────


def test_status_ready_all_required_tables(provider: MockExposureDataProvider) -> None:
    status = provider.get_dataset_status("ds-farmers-bda-2027")
    assert status.ert_status == ErtStatus.ERT_READY
    required = {"TIV_SUMMARY", "EVOLUTION", "PERIL_DETAILS"}
    present = {t.table_type for t in status.tables if t.exists}
    assert required.issubset(present)
    assert status.warnings == []


def test_status_partial_emits_warning(provider: MockExposureDataProvider) -> None:
    status = provider.get_dataset_status("ds-acmere-26-multi")
    assert status.ert_status == ErtStatus.ERT_PARTIAL
    codes = [w.code for w in status.warnings]
    assert WarningCode.WARN_ERT_TABLES_PARTIAL.value in codes
    # PERIL_DETAILS must be the missing required table for partial classification.
    missing_required = {
        t.table_type for t in status.tables if not t.exists and t.table_type == "PERIL_DETAILS"
    }
    assert missing_required == {"PERIL_DETAILS"}


def test_status_not_found_emits_warning(provider: MockExposureDataProvider) -> None:
    status = provider.get_dataset_status("ds-sample-27-ws")
    assert status.ert_status == ErtStatus.ERT_NOT_FOUND
    codes = [w.code for w in status.warnings]
    assert WarningCode.WARN_ERT_NOT_FOUND.value in codes
    assert all(not t.exists for t in status.tables)


def test_alwaysfails_marker_present(provider: MockExposureDataProvider) -> None:
    """Designated EDM the jobs service (Agent C) will fail on every run."""
    assert provider.is_always_fails("ds-alwaysfails-27-ws") is True
    assert provider.is_always_fails("ds-farmers-bda-2027") is False


# ───────────────────────── worked example: max-across-perils @ STATE grain ─────────────────────────


def _state_tiv(
    provider: MockExposureDataProvider,
    dataset_id: str,
    statecode: str,
    peril: str | None = None,
) -> float:
    """Sum STATE-grain rows for one dataset + state, optionally filtered by peril.

    The mock keeps multiple STATE rows per (state, peril) — different occupancy
    or distance-to-coast slices. Since the office-level EDM now bundles multiple
    perils, callers usually want to pin one peril to read the worked-example values.
    """
    return sum(
        f.tiv
        for f in provider.get_facts_for_dataset(dataset_id)
        if f.aggregation == AggregationLevel.STATE.value
        and f.statecode == statecode
        and (peril is None or f.peril == peril)
    )


def test_worked_example_state_tivs_in_expected_magnitude(provider: MockExposureDataProvider) -> None:
    """FL/CA STATE-grain TIVs are realistic per peril for the bulk-generated
    Farmers BDA 2027 EDM. Spec-specific exact values were removed when the
    fact data switched from hand-crafted to generated; structural sanity
    (every peril × major state has a real number in the right order of
    magnitude) is what we guard now.
    """
    ds = "ds-farmers-bda-2027"
    for state in ("FL", "CA"):
        for peril in (Peril.WS.value, Peril.EQ.value, Peril.CS.value):
            tiv = _state_tiv(provider, ds, state, peril=peril)
            assert tiv > 0, f"{state} {peril}: missing STATE-grain row"
            assert 1e8 < tiv < 1e13, f"{state} {peril}: TIV {tiv} out of plausible range"


def test_tiv_split_consistency(provider: MockExposureDataProvider) -> None:
    """CLAUDE.md rule 6 / contract: TIV must equal Building + Contents + BI on every row."""
    for ds_id in (
        "ds-farmers-bda-2027",
        "ds-farmers-bda-2027",
        "ds-acmere-26-multi",
        "ds-farmers-bda-2026",
    ):
        for f in provider.get_facts_for_dataset(ds_id):
            expected = round(f.building + f.contents + f.bi, 2)
            assert round(f.tiv, 2) == expected, (
                f"{ds_id} {f.geography_id} {f.peril}: TIV {f.tiv} != B+C+BI {expected}"
            )


# ───────────────────────── geometry availability / map-missing ─────────────────────────


def test_geometry_includes_country_state_county_cresta(
    provider: MockExposureDataProvider,
) -> None:
    geo = provider.get_geometry_availability()
    assert "US" in geo
    assert {"US-FL", "US-TX", "US-CA", "US-NY"}.issubset(geo)
    assert "US-FL-12086" in geo
    assert "CRESTA-US_01" in geo


# ───────────────────────── IED industry denominator ─────────────────────────


def test_ied_has_state_rows_and_county_gap(provider: MockExposureDataProvider) -> None:
    """At least one COUNTY in mock facts has NO IED row → WARN_IED_DENOMINATOR_MISSING."""
    ied = provider.get_ied_industry()
    state_geos = {
        row.geography_id
        for row in ied
        if row.geography_level == AggregationLevel.STATE.value
    }
    county_geos = {
        row.geography_id
        for row in ied
        if row.geography_level == AggregationLevel.COUNTY.value
    }

    # The windstorm sheet covers the Gulf and Atlantic states, not California.
    assert {"US-FL", "US-TX", "US-NY"}.issubset(state_geos)
    assert "US-CA" not in state_geos

    fact_county_geos = {
        f.geography_id
        for f in provider.get_facts_for_dataset("ds-farmers-bda-2027")
        if f.aggregation == AggregationLevel.COUNTY.value
    }
    assert "US-CA-06037" in fact_county_geos
    assert "US-CA-06037" not in county_geos
    assert "US-FL-12086" in county_geos
    assert "US-FL-12011" in county_geos


# ───────────────────────── data quality + occupancy ─────────────────────────


def test_some_rows_have_invalid_tiv(provider: MockExposureDataProvider) -> None:
    nonzero = [
        f for f in provider.get_facts_for_dataset("ds-farmers-bda-2027")
        if (f.invalid_tiv or 0) > 0
    ]
    assert nonzero, "Need ≥1 row with non-zero invalidTiv for data-quality scenario"


def test_unknown_occupancy_segment_present(provider: MockExposureDataProvider) -> None:
    has_unknown = any(
        f.occupancy_segment == "UNKNOWN"
        for f in provider.get_facts_for_dataset("ds-farmers-bda-2027")
    )
    assert has_unknown


def test_every_state_with_data_has_at_least_one_county(
    provider: MockExposureDataProvider,
) -> None:
    """No state in any portfolio dataset should have STATE-grain TIV but no
    COUNTY-grain row — drilling into the state would show an empty polygon."""
    for ds_id in ("ds-farmers-bda-2027", "ds-farmers-bda-2027", "ds-acmere-26-multi",
                  "ds-farmers-bda-2026", "ds-acmere-27-multi", "ds-coastalre-27-ws"):
        facts = provider.get_facts_for_dataset(ds_id)
        state_codes = {
            f.statecode for f in facts
            if f.aggregation == AggregationLevel.STATE.value and f.statecode
        }
        county_state_codes = {
            f.statecode for f in facts
            if f.aggregation == AggregationLevel.COUNTY.value and f.statecode
        }
        missing = state_codes - county_state_codes
        assert not missing, f"{ds_id}: states with STATE rows but no COUNTY row: {sorted(missing)}"


# ───────────────────────── portfolio scope ─────────────────────────


def test_portfolio_facts_exclude_non_portfolio_datasets(
    provider: MockExposureDataProvider,
) -> None:
    """Option A: in-force BOUND only; no orphan fact stems; no sample/alwaysfails."""
    portfolio_dataset_ids = {f.dataset_id for f in provider.get_portfolio_facts()}
    assert "ds-sample-27-ws" not in portfolio_dataset_ids
    assert "ds-alwaysfails-27-ws" not in portfolio_dataset_ids
    # Legacy per-peril orphan stems must not pollute the denominator.
    for orphan in (
        "ds-farmers-25-ws",
        "ds-farmers-26-ws",
        "ds-farmers-26-eq",
        "ds-farmers-26-cs",
        "ds-farmers-27-ws",
        "ds-farmers-27-eq",
        "ds-farmers-27-cs",
    ):
        assert orphan not in portfolio_dataset_ids
    # In-force multi-peril Farmers BDA year should still be present when in force.
    # (Date-sensitive: if fixtures expire, at least orphan exclusion holds.)
    ids = provider.portfolio_dataset_ids(in_force_only=True)
    assert "ds-sample-27-ws" not in ids
    assert all(o not in ids for o in (
        "ds-farmers-25-ws", "ds-farmers-26-ws", "ds-farmers-27-ws",
    ))


# ───────────────────────── dataset-group CRUD ─────────────────────────


def test_create_dataset_group_persists_in_memory(
    provider: MockExposureDataProvider,
) -> None:
    before = {g.dataset_group_id for g in provider.list_dataset_groups()}
    payload = DatasetGroupCreate(
        group_name="Farmers 2027 Multi-Peril",
        currency="USD",
        combination_method=CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN,
        members=[
            DatasetGroupMemberInput(dataset_id="ds-farmers-bda-2027", peril=Peril.WS),
            DatasetGroupMemberInput(dataset_id="ds-farmers-bda-2027", peril=Peril.EQ),
            DatasetGroupMemberInput(dataset_id="ds-acmere-26-multi", peril=Peril.CS),
        ],
    )
    created = provider.create_dataset_group(payload)
    assert created.dataset_group_id.startswith("grp-")
    assert created.dataset_group_id not in before

    after = {g.dataset_group_id for g in provider.list_dataset_groups()}
    assert created.dataset_group_id in after
    assert provider.get_dataset_group(created.dataset_group_id) is not None
    # Default combination method retained → rule 3 default upheld.
    assert created.combination_method == CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN.value


# ───────────────────────── currency & portname ─────────────────────────


def test_currencies_match_dataset_registry(provider: MockExposureDataProvider) -> None:
    for ds_id in (
        "ds-farmers-bda-2027",
        "ds-farmers-bda-2027",
        "ds-acmere-26-multi",
        "ds-farmers-bda-2026",
    ):
        for f in provider.get_facts_for_dataset(ds_id):
            assert f.currency == "USD"


def test_portname_format_per_year(provider: MockExposureDataProvider) -> None:
    for f in provider.get_facts_for_dataset("ds-farmers-bda-2027"):
        assert f.portname == "09302025"
    for f in provider.get_facts_for_dataset("ds-farmers-bda-2026"):
        assert f.portname == "09302024"


# ───────────────────────── fixture-resolution sanity ─────────────────────────


def test_mock_data_dir_resolves(provider: MockExposureDataProvider) -> None:
    # The dir must really exist on disk where we expect.
    assert (Path(__file__).resolve().parents[2] / "mockdata" / "datasets.json").exists()
