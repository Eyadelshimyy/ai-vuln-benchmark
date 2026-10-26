"""Background-job models — shapes per BACKGROUND_JOBS_SPEC.md + API_SPEC.md."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .common import CamelModel
from .enums import AggregationLevel, JobStatus, Peril


class ErtJobRunRequest(CamelModel):
    server_name: str
    edm_database_name: str
    treaty_year: int
    currency: str
    peril: Peril = Peril.ALL
    aggregation_levels: list[AggregationLevel]
    rerun: bool = False
    started_by: str | None = None


class ErtJobAcceptedResponse(CamelModel):
    job_id: str
    status: JobStatus


class ErtJobErrorTechnical(CamelModel):
    server_name: str
    database_name: str
    procedure_name: str
    input_parameters: dict[str, Any]
    timestamp: datetime
    log_id: str | None = None
    tables_checked: list[str] = []
    tables_generated_before_failure: list[str] = []


class ErtJobError(CamelModel):
    message: str
    technical: ErtJobErrorTechnical
    email_sent: bool = False


class ErtJobStatusResponse(CamelModel):
    job_id: str
    status: JobStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_tables_generated: list[str] = []
    rows_generated: int = 0
    error: ErtJobError | None = None


class ErtJobCancelResponse(CamelModel):
    job_id: str
    status: JobStatus
