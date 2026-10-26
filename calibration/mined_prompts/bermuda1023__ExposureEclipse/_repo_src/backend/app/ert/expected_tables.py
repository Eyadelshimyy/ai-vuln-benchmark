"""ExpectedERTTable registry — 7 canonical ERT cuts (ERT_OUTPUT_FORMAT.md).

CLAUDE.md rule 8: SQL table names are NOT hardcoded. The runtime name comes from a
pattern resolved per (EDM, year, currency, peril, level). v1 mock provider uses simple
synthetic names; the SQL provider will plug in the real naming convention in Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.enums import AggregationLevel


@dataclass(frozen=True)
class ExpectedERTTable:
    expected_table_id: str
    table_type: str  # one of the 7 TableType codes below
    description: str
    required_for_v1: bool
    aggregation_level: AggregationLevel | None = None
    table_name_pattern: str = "{edm}__{table_type}"  # placeholder until SQL naming confirmed


# The seven CALC.Get_* cuts from ERT_OUTPUT_FORMAT.md (TableType codes are the contract).
EXPECTED_ERT_TABLES: tuple[ExpectedERTTable, ...] = (
    ExpectedERTTable(
        expected_table_id="ert-tiv-summary",
        table_type="TIV_SUMMARY",
        description="Geography × peril TIV split (CALC.Get_TIV_Summary)",
        required_for_v1=True,
    ),
    ExpectedERTTable(
        expected_table_id="ert-evolution",
        table_type="EVOLUTION",
        description="Full-grain normalized fact, drives YoY (CALC.Get_Evolution)",
        required_for_v1=True,
    ),
    ExpectedERTTable(
        expected_table_id="ert-construction-summary",
        table_type="CONSTRUCTION_SUMMARY",
        description="Construction × peril TIV (CALC.Get_Construction_Summary)",
        required_for_v1=False,
    ),
    ExpectedERTTable(
        expected_table_id="ert-yearbuilt-summary",
        table_type="YEARBUILT_SUMMARY",
        description="Year-built × peril TIV (CALC.Get_YearBuilt_Summary)",
        required_for_v1=False,
    ),
    ExpectedERTTable(
        expected_table_id="ert-numberofstories-summary",
        table_type="NUMBEROFSTORIES_SUMMARY",
        description="Stories × peril TIV (CALC.Get_NumberOfStories_Summary)",
        required_for_v1=False,
    ),
    ExpectedERTTable(
        expected_table_id="ert-peril-details",
        table_type="PERIL_DETAILS",
        description="Peril × geocoding detail (CALC.GET_Peril_Details)",
        required_for_v1=True,
    ),
    ExpectedERTTable(
        expected_table_id="ert-distance-to-coast",
        table_type="DISTANCE_TO_COAST",
        description="WS-focused distance-to-coast bands (CALC.Get_Distance_To_Coast)",
        required_for_v1=False,
    ),
)

REQUIRED_TABLE_TYPES = {t.table_type for t in EXPECTED_ERT_TABLES if t.required_for_v1}
ALL_TABLE_TYPES = {t.table_type for t in EXPECTED_ERT_TABLES}
