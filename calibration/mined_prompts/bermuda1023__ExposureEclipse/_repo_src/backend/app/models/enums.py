"""Canonical enums — single source of truth on the BACKEND side.

⭐ MIRRORED from docs/CONTRACTS.md. If you add/change a value here you MUST
update docs/CONTRACTS.md AND frontend/src/types/contracts.ts in the same change.
No ad-hoc string literals on either side of the wire (CLAUDE.md rule 10).

Wire format: UPPER_SNAKE_CASE string literals. We use str-Enum so values
serialize as plain strings via FastAPI/Pydantic.
"""

from __future__ import annotations

from enum import Enum


class MetricKey(str, Enum):
    """CONTRACTS.md §1 — what the map/tooltip/pivot can measure or color by."""

    TIV = "TIV"
    LOCATION_COUNT = "LOCATION_COUNT"
    DEAL_SHARE_OF_PORTFOLIO_IN_GEOGRAPHY = "DEAL_SHARE_OF_PORTFOLIO_IN_GEOGRAPHY"
    GEOGRAPHY_SHARE_OF_TOTAL_PORTFOLIO = "GEOGRAPHY_SHARE_OF_TOTAL_PORTFOLIO"
    SELECTED_DEAL_GEOGRAPHY_CONCENTRATION = "SELECTED_DEAL_GEOGRAPHY_CONCENTRATION"
    CLIENT_MARKET_SHARE = "CLIENT_MARKET_SHARE"
    YOY_CHANGE = "YOY_CHANGE"


class Measure(str, Enum):
    """CONTRACTS.md §1b — raw ERT value columns selectable as pivot/export measures."""

    TIV = "TIV"
    BUILDING = "BUILDING"
    CONTENTS = "CONTENTS"
    BI = "BI"
    EXPLIM_GR = "EXPLIM_GR"
    EXPLIM_NET = "EXPLIM_NET"
    LOCATION_COUNT = "LOCATION_COUNT"
    ACCOUNT_COUNT = "ACCOUNT_COUNT"
    INVALID_TIV = "INVALID_TIV"
    INVALID_COUNT = "INVALID_COUNT"


class AggregationLevel(str, Enum):
    """CONTRACTS.md §2 — geography levels, coarse → fine."""

    COUNTRY = "COUNTRY"
    STATE = "STATE"
    COUNTY = "COUNTY"
    CRESTA = "CRESTA"


class ErtStatus(str, Enum):
    """CONTRACTS.md §3."""

    ERT_NOT_FOUND = "ERT_NOT_FOUND"
    ERT_PARTIAL = "ERT_PARTIAL"
    ERT_READY = "ERT_READY"
    ERT_READY_PRIOR_RUN_DETECTED = "ERT_READY_PRIOR_RUN_DETECTED"
    ERT_ERROR = "ERT_ERROR"


class ProgrammeStatus(str, Enum):
    """Deal lifecycle. Currently in-force = ``BOUND`` and date between inception
    and expiry. ``EXPIRED`` is set automatically by the provider when expiry has
    passed. ``DECLINED`` / ``NTU`` deals don't enter the in-force portfolio."""

    BOUND = "BOUND"
    QUOTED = "QUOTED"
    DECLINED = "DECLINED"
    NTU = "NTU"
    EXPIRED = "EXPIRED"


class CombinationMethod(str, Enum):
    """CONTRACTS.md §4 — default is MAX_ACROSS_PERILS_AT_VIEW_GRAIN (CLAUDE.md rule 3)."""

    MAX_ACROSS_PERILS_AT_VIEW_GRAIN = "MAX_ACROSS_PERILS_AT_VIEW_GRAIN"
    SUM_DISTINCT_SEGMENTS = "SUM_DISTINCT_SEGMENTS"
    SELECTED_EDM_AS_BASE = "SELECTED_EDM_AS_BASE"
    KEEP_PERILS_SEPARATE = "KEEP_PERILS_SEPARATE"
    CUSTOM = "CUSTOM"


class Peril(str, Enum):
    """CONTRACTS.md §5 — canonical = real ERT codes. ALL is filter-only."""

    EQ = "EQ"
    WS = "WS"
    CS = "CS"
    FL = "FL"
    FR = "FR"
    TR = "TR"
    ALL = "ALL"


class OccupancySegment(str, Enum):
    """CONTRACTS.md §6 — market-share segmentation roll-up. UNKNOWN never force-mapped."""

    RESIDENTIAL = "RESIDENTIAL"
    COMMERCIAL = "COMMERCIAL"
    INDUSTRIAL = "INDUSTRIAL"
    AUTO = "AUTO"
    UNKNOWN = "UNKNOWN"


class JobStatus(str, Enum):
    """CONTRACTS.md §7 — lowercase on the wire."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PortfolioScope(str, Enum):
    """CONTRACTS.md §8 — v1 supports ALL_LOADED_DATASETS only."""

    ALL_LOADED_DATASETS = "ALL_LOADED_DATASETS"
    BOUND_DEALS = "BOUND_DEALS"
    CUSTOM = "CUSTOM"


class YoyStatus(str, Enum):
    """CONTRACTS.md §9."""

    NEW = "NEW"
    REMOVED = "REMOVED"
    NA = "NA"
    OK = "OK"


class WarningCode(str, Enum):
    """CONTRACTS.md §10 — stable codes. UI renders message; logs/exports include code."""

    WARN_COUNTY_DATA_UNAVAILABLE = "WARN_COUNTY_DATA_UNAVAILABLE"
    WARN_CURRENCY_ASSUMED = "WARN_CURRENCY_ASSUMED"
    WARN_CURRENCY_MISMATCH = "WARN_CURRENCY_MISMATCH"
    WARN_IED_DENOMINATOR_MISSING = "WARN_IED_DENOMINATOR_MISSING"
    WARN_PRIOR_DATASET_NOT_SELECTED = "WARN_PRIOR_DATASET_NOT_SELECTED"
    WARN_AGGREGATION_LEVEL_MISMATCH = "WARN_AGGREGATION_LEVEL_MISMATCH"
    WARN_DATASET_GROUP_MAX_ACROSS_PERILS = "WARN_DATASET_GROUP_MAX_ACROSS_PERILS"
    WARN_DATASET_GROUP_SUMMED = "WARN_DATASET_GROUP_SUMMED"
    WARN_ERT_TABLES_PARTIAL = "WARN_ERT_TABLES_PARTIAL"
    WARN_ERT_NOT_FOUND = "WARN_ERT_NOT_FOUND"
    WARN_MAP_GEOMETRY_MISSING = "WARN_MAP_GEOMETRY_MISSING"
    WARN_FILTERS_RETURN_NO_ROWS = "WARN_FILTERS_RETURN_NO_ROWS"
    WARN_EXPORT_TOO_LARGE = "WARN_EXPORT_TOO_LARGE"
    WARN_EDM_LOAD_FAILED = "WARN_EDM_LOAD_FAILED"
    WARN_EDM_MOCK_FALLBACK = "WARN_EDM_MOCK_FALLBACK"


class WarningSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"


class ErrorCode(str, Enum):
    """CONTRACTS.md §11 — see docs/ERROR_HANDLING.md for envelope."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    DATASET_GROUP_NOT_FOUND = "DATASET_GROUP_NOT_FOUND"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    PRIOR_DB_NOT_FOUND = "PRIOR_DB_NOT_FOUND"
    IED_GEOGRAPHY_MISSING = "IED_GEOGRAPHY_MISSING"
    ERT_JOB_FAILED = "ERT_JOB_FAILED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    EXPORT_TOO_LARGE = "EXPORT_TOO_LARGE"
    GEOMETRY_TOO_COMPLEX = "GEOMETRY_TOO_COMPLEX"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GeocodingQuality(str, Enum):
    """CONTRACTS.md §14 — real ERT vocabulary, exact labels preserved."""

    COORDINATE = "Coordinate"
    STREET_PARCEL = "Street/Parcel"
    POSTAL_CODE = "Postal code"
    BLOCK_GROUP = "Block Group"


class DistanceToCoastBand(str, Enum):
    """CONTRACTS.md §14 — keep the `a..g` prefix; it's the canonical sort key."""

    A_AT_COAST = "a=> At the Coast"
    B_0_TO_0_5 = "b=> 0 - 0.5 Miles from Coast"
    C_0_5_TO_1 = "c=> 0.5 - 1 Miles from Coast"
    D_1_TO_2 = "d=> 1.0 - 2 Miles from Coast"
    E_2_TO_5 = "e=> 2.0 - 5 Miles from Coast"
    F_5_TO_10 = "f=> 5.0 - 10 Miles from Coast"
    G_10_PLUS = "g=> +10 Miles from Coast"


class YearBuiltBand(str, Enum):
    """CONTRACTS.md §14."""

    PRE_1930 = "1930 and before"
    Y_1930_1960 = "1930 to 1960"
    Y_1960_1980 = "1960 to 1980"
    Y_1980_2000 = "1980 to 2000"
    Y_2000_PRESENT = "2000 to Present"
    UNKNOWN = "Unknown"


# NumberOfStoriesBand and ConstructionClass are data-driven (OPEN_QUESTIONS #27).
# Treat as `str` until the ERT data confirms the full set. Do NOT hardcode.
NumberOfStoriesBand = str  # type alias on purpose
ConstructionClass = str  # type alias on purpose


__all__ = [
    "AggregationLevel",
    "CombinationMethod",
    "ConstructionClass",
    "DistanceToCoastBand",
    "ErrorCode",
    "ErtStatus",
    "GeocodingQuality",
    "JobStatus",
    "Measure",
    "MetricKey",
    "NumberOfStoriesBand",
    "OccupancySegment",
    "Peril",
    "PortfolioScope",
    "WarningCode",
    "WarningSeverity",
    "YearBuiltBand",
    "YoyStatus",
]
