"""Warning + Error envelope models — shape per CONTRACTS.md §10 / §11."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .enums import ErrorCode, WarningCode, WarningSeverity

# Default warning messages — match the table in CONTRACTS.md §10 verbatim so the wire is
# consistent. UI is free to swap to its own copy if needed (code is the stable identifier).
WARNING_DEFAULT_MESSAGES: dict[WarningCode, str] = {
    WarningCode.WARN_COUNTY_DATA_UNAVAILABLE: (
        "County-level data is not available for this dataset. Showing state-level results."
    ),
    WarningCode.WARN_CURRENCY_ASSUMED: "Currency was manually assumed for this dataset.",
    WarningCode.WARN_CURRENCY_MISMATCH: (
        "Selected datasets use different currencies. "
        "Provide a conversion assumption or compare separately."
    ),
    WarningCode.WARN_IED_DENOMINATOR_MISSING: (
        "Market share cannot be calculated; the RMS IED table has no matching geography."
    ),
    WarningCode.WARN_PRIOR_DATASET_NOT_SELECTED: (
        "No prior-year dataset selected; YoY is unavailable."
    ),
    WarningCode.WARN_AGGREGATION_LEVEL_MISMATCH: (
        "Current and prior datasets differ in aggregation level; aggregated to the common level."
    ),
    WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS: (
        "This group contains multiple peril EDMs. "
        "Max-across-perils is used at the current viewed grain."
    ),
    WarningCode.WARN_DATASET_GROUP_SUMMED: (
        "This group sums TIV across EDMs marked as distinct segments."
    ),
    WarningCode.WARN_ERT_TABLES_PARTIAL: (
        "Some ERT output tables are missing; results may be incomplete."
    ),
    WarningCode.WARN_ERT_NOT_FOUND: "No ERT output tables found for this dataset.",
    WarningCode.WARN_MAP_GEOMETRY_MISSING: (
        "Map geometry is unavailable for some features at this level."
    ),
    WarningCode.WARN_FILTERS_RETURN_NO_ROWS: "No exposure records match the current filters.",
    WarningCode.WARN_EXPORT_TOO_LARGE: (
        "Export exceeds the size limit; refine filters or aggregation."
    ),
    WarningCode.WARN_EDM_LOAD_FAILED: (
        "One or more EDM fact loads failed; results may understate exposure."
    ),
    WarningCode.WARN_EDM_MOCK_FALLBACK: (
        "One or more EDMs fell back to mock fixture data instead of SQL."
    ),
}

_WARNING_SEVERITY: dict[WarningCode, WarningSeverity] = {
    WarningCode.WARN_COUNTY_DATA_UNAVAILABLE: WarningSeverity.WARN,
    WarningCode.WARN_CURRENCY_ASSUMED: WarningSeverity.WARN,
    WarningCode.WARN_CURRENCY_MISMATCH: WarningSeverity.WARN,
    WarningCode.WARN_IED_DENOMINATOR_MISSING: WarningSeverity.WARN,
    WarningCode.WARN_PRIOR_DATASET_NOT_SELECTED: WarningSeverity.INFO,
    WarningCode.WARN_AGGREGATION_LEVEL_MISMATCH: WarningSeverity.WARN,
    WarningCode.WARN_DATASET_GROUP_MAX_ACROSS_PERILS: WarningSeverity.INFO,
    WarningCode.WARN_DATASET_GROUP_SUMMED: WarningSeverity.WARN,
    WarningCode.WARN_ERT_TABLES_PARTIAL: WarningSeverity.WARN,
    WarningCode.WARN_ERT_NOT_FOUND: WarningSeverity.WARN,
    WarningCode.WARN_MAP_GEOMETRY_MISSING: WarningSeverity.INFO,
    WarningCode.WARN_FILTERS_RETURN_NO_ROWS: WarningSeverity.INFO,
    WarningCode.WARN_EXPORT_TOO_LARGE: WarningSeverity.WARN,
    WarningCode.WARN_EDM_LOAD_FAILED: WarningSeverity.WARN,
    WarningCode.WARN_EDM_MOCK_FALLBACK: WarningSeverity.WARN,
}


class _CamelModel(BaseModel):
    """Pydantic base — serializes snake_case → camelCase on the wire."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        use_enum_values=True,
    )


class Warning(_CamelModel):
    code: WarningCode
    severity: WarningSeverity
    message: str
    context: dict[str, Any] | None = None


def make_warning(
    code: WarningCode,
    *,
    message: str | None = None,
    context: dict[str, Any] | None = None,
) -> Warning:
    """Build a Warning with the canonical message + severity unless overridden."""
    return Warning(
        code=code,
        severity=_WARNING_SEVERITY[code],
        message=message or WARNING_DEFAULT_MESSAGES[code],
        context=context,
    )


class ErrorEnvelopeBody(_CamelModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    trace_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorEnvelope(_CamelModel):
    error: ErrorEnvelopeBody
