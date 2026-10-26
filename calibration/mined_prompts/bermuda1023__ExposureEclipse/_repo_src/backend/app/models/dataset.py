"""Dataset / dataset-group / ERT-status models — shapes per DATA_MODEL.md + API_SPEC.md."""

from __future__ import annotations

from datetime import datetime

from .common import CamelModel
from .enums import AggregationLevel, CombinationMethod, ErtStatus, Peril
from .warnings import Warning


class DatasetSummary(CamelModel):
    dataset_id: str
    server_name: str
    edm_database_name: str
    treaty_year: int
    currency: str
    ert_status: ErtStatus
    available_granularity: list[AggregationLevel]
    is_included_in_portfolio: bool
    last_generated_at: datetime | None = None
    cedent_name: str | None = None
    programme_name: str | None = None
    exposure_data_cutoff_date: datetime | None = None
    prior_exposure_data_cutoff_date: datetime | None = None


class DatasetListResponse(CamelModel):
    datasets: list[DatasetSummary]


class ExpectedTableStatus(CamelModel):
    """One row in the ERT-status `tables` array."""

    table_type: str
    name: str
    exists: bool
    row_count: int


class DatasetStatusResponse(CamelModel):
    dataset_id: str
    ert_status: ErtStatus
    tables: list[ExpectedTableStatus]
    warnings: list[Warning] = []


class DatasetGroupMemberInput(CamelModel):
    dataset_id: str
    peril: Peril


class DatasetGroupCreate(CamelModel):
    group_name: str
    currency: str
    combination_method: CombinationMethod = CombinationMethod.MAX_ACROSS_PERILS_AT_VIEW_GRAIN
    members: list[DatasetGroupMemberInput]
    cedent_name: str | None = None
    programme_name: str | None = None
    year_of_account: int | None = None
    distinct_segments_confirmed: bool = False
    notes: str | None = None


class DatasetGroupCreated(CamelModel):
    dataset_group_id: str
    warnings: list[Warning] = []


class DatasetGroup(CamelModel):
    dataset_group_id: str
    group_name: str
    currency: str
    combination_method: CombinationMethod
    members: list[DatasetGroupMemberInput]
    distinct_segments_confirmed: bool
    created_at: datetime
    cedent_name: str | None = None
    programme_name: str | None = None
    year_of_account: int | None = None
    notes: str | None = None


class DatasetGroupListResponse(CamelModel):
    dataset_groups: list[DatasetGroup]
