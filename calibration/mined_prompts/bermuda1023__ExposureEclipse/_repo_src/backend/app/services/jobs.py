"""In-process asyncio ERT job store (BACKGROUND_JOBS_SPEC.md).

v1 implementation per the spec:

* In-memory dict keyed by ``jobId``. No external broker, no SQLite persistence.
* Lifecycle: ``queued`` → (delay) → ``running`` → (delay) → ``completed`` / ``failed``.
* Duplicate-concurrent guard: a submit for an ``(server_name, edm_database_name)``
  that already has a ``queued`` / ``running`` job returns the *existing* jobId.
* On failure, sends a (no-op in dev) error report via :mod:`.email` and records
  ``email_sent`` on the error block (CLAUDE.md rule "graceful degradation").
* The mock fails when the EDM database name contains ``"AlwaysFails"`` —
  this matches the ``Re_BER_27_AlwaysFails_WS_EDM_01`` fixture
  (MOCK_DATA_SPEC.md) without depending on a specific provider attribute.

Calculations / business logic do NOT live here — only job orchestration. The
router (``app/api/ert_jobs.py``) just delegates.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..brand import BRAND_NAME
from ..config import get_settings
from ..models.enums import AggregationLevel, JobStatus, Peril
from ..models.jobs import (
    ErtJobError,
    ErtJobErrorTechnical,
    ErtJobRunRequest,
    ErtJobStatusResponse,
)
from .email import EmailService, get_email_service

logger = logging.getLogger(__name__)

# Mock timings — kept short so tests stay fast but long enough to observe the
# queued → running → terminal transitions in a status poll.
_QUEUED_DELAY_RANGE = (0.05, 0.2)
_RUNNING_DELAY_RANGE = (0.1, 0.3)

# Substring marker used by the mock to choose which EDM names always fail.
_ALWAYS_FAILS_MARKER = "AlwaysFails"

# A representative slice of ERT output tables the mock pretends to generate
# (real names come from the ERT spec — these are illustrative for the wire).
_MOCK_SUCCESS_TABLES = [
    "TIV_SUMMARY",
    "EVOLUTION",
    "CONSTRUCTION_SUMMARY",
    "YEARBUILT_SUMMARY",
    "NUMBEROFSTORIES_SUMMARY",
    "PERIL_DETAILS",
    "DISTANCE_TO_COAST",
]


@dataclass
class JobRecord:
    """In-memory job record — mirrors DATA_MODEL.md ``BackgroundJob`` + housekeeping."""

    job_id: str
    server_name: str
    edm_database_name: str
    treaty_year: int
    currency: str
    peril: Peril
    aggregation_levels: list[AggregationLevel]
    started_by: str | None
    rerun: bool
    input_parameters_json: dict[str, Any]

    status: JobStatus = JobStatus.QUEUED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_tables_generated: list[str] = field(default_factory=list)
    rows_generated: int = 0
    tables_checked: list[str] = field(default_factory=list)
    tables_generated_before_failure: list[str] = field(default_factory=list)
    error_message: str | None = None
    email_sent: bool = False
    log_id: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Asyncio task handle so the registry can cancel an in-flight run.
    task: asyncio.Task[None] | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_response(self) -> ErtJobStatusResponse:
        """Project to the wire shape (API_SPEC.md /ert-jobs/status)."""
        error: ErtJobError | None = None
        if self.status == JobStatus.FAILED:
            error = ErtJobError(
                message=self.error_message
                or f"The ERT routine failed for {self.edm_database_name}.",
                technical=ErtJobErrorTechnical(
                    server_name=self.server_name,
                    database_name=self.edm_database_name,
                    procedure_name="dbo.usp_GenerateExposureReportTables",
                    input_parameters=self.input_parameters_json,
                    timestamp=self.completed_at or self.updated_at,
                    log_id=self.log_id,
                    tables_checked=list(self.tables_checked),
                    tables_generated_before_failure=list(self.tables_generated_before_failure),
                ),
                email_sent=self.email_sent,
            )
        return ErtJobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            started_at=self.started_at,
            completed_at=self.completed_at,
            output_tables_generated=list(self.output_tables_generated),
            rows_generated=self.rows_generated,
            error=error,
        )


class JobRegistry:
    """In-process registry — single source of truth for job state in v1."""

    def __init__(self, email_service: EmailService | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self._email = email_service or get_email_service()

    # ───── Public API used by the router ─────

    def submit(self, request: ErtJobRunRequest) -> str:
        """Create + start a job. Returns the jobId.

        Duplicate-concurrent guard: if a job for ``(server_name, edm_database_name)``
        is already ``queued`` or ``running``, returns the existing jobId instead
        of starting a second (BACKGROUND_JOBS_SPEC.md §Concurrency).
        """
        existing = self._find_active(request.server_name, request.edm_database_name)
        if existing is not None:
            return existing.job_id

        record = JobRecord(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            server_name=request.server_name,
            edm_database_name=request.edm_database_name,
            treaty_year=request.treaty_year,
            currency=request.currency,
            peril=request.peril,
            aggregation_levels=list(request.aggregation_levels),
            started_by=request.started_by,
            rerun=request.rerun,
            input_parameters_json=request.model_dump(mode="json", by_alias=True),
        )
        self._jobs[record.job_id] = record
        # Schedule the lifecycle simulation on the running event loop.
        record.task = asyncio.create_task(self._run_lifecycle(record))
        return record.job_id

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Returns ``True`` if the job was cancellable.

        Already-terminal jobs (``completed`` / ``failed`` / ``cancelled``) are
        not re-cancelled — they keep their terminal status and we return
        ``False``. That's a no-op semantically, not an error.
        """
        record = self._jobs.get(job_id)
        if record is None:
            return False
        if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        if record.task is not None and not record.task.done():
            record.task.cancel()
        record.status = JobStatus.CANCELLED
        record.completed_at = datetime.now(timezone.utc)
        record.touch()
        return True

    # ───── Internals ─────

    def _find_active(self, server_name: str, edm_database_name: str) -> JobRecord | None:
        for r in self._jobs.values():
            if (
                r.server_name == server_name
                and r.edm_database_name == edm_database_name
                and r.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            ):
                return r
        return None

    async def _run_lifecycle(self, record: JobRecord) -> None:
        """Mock lifecycle: queued → running → completed/failed."""
        try:
            await asyncio.sleep(random.uniform(*_QUEUED_DELAY_RANGE))
            record.status = JobStatus.RUNNING
            record.started_at = datetime.now(timezone.utc)
            record.touch()

            await asyncio.sleep(random.uniform(*_RUNNING_DELAY_RANGE))

            if _ALWAYS_FAILS_MARKER in record.edm_database_name:
                await self._fail(record)
            else:
                self._succeed(record)
        except asyncio.CancelledError:
            # Cooperative cancel — status was already set in cancel().
            logger.info("Job %s cancelled mid-flight", record.job_id)
            return

    def _succeed(self, record: JobRecord) -> None:
        record.status = JobStatus.COMPLETED
        record.completed_at = datetime.now(timezone.utc)
        record.output_tables_generated = list(_MOCK_SUCCESS_TABLES)
        record.tables_checked = list(_MOCK_SUCCESS_TABLES)
        # A mock row count so the response shape carries a sensible number.
        record.rows_generated = 1500 + len(record.aggregation_levels) * 50
        record.touch()

    async def _fail(self, record: JobRecord) -> None:
        record.status = JobStatus.FAILED
        record.completed_at = datetime.now(timezone.utc)
        # Pretend the routine started writing one table before blowing up.
        record.tables_checked = list(_MOCK_SUCCESS_TABLES)
        record.tables_generated_before_failure = [_MOCK_SUCCESS_TABLES[0]]
        record.error_message = (
            f"The ERT routine failed for {record.edm_database_name}. "
            "Stored procedure raised: simulated mock failure."
        )
        record.log_id = f"log-{uuid.uuid4().hex[:10]}"
        record.touch()

        settings = get_settings()
        try:
            record.email_sent = self._email.send_error_report(
                subject=f"[{BRAND_NAME}] ERT job failed: {record.edm_database_name}",
                technical={
                    "serverName": record.server_name,
                    "databaseName": record.edm_database_name,
                    "procedureName": "dbo.usp_GenerateExposureReportTables",
                    "inputParameters": record.input_parameters_json,
                    "timestamp": record.completed_at.isoformat()
                    if record.completed_at
                    else None,
                    "logId": record.log_id,
                    "tablesChecked": record.tables_checked,
                    "tablesGeneratedBeforeFailure": record.tables_generated_before_failure,
                    "errorMessage": record.error_message,
                },
                recipient=settings.support_error_email,
            )
        except Exception:  # pragma: no cover — best-effort send must never mask original failure
            logger.exception("Failed to send ERT-failure email for job %s", record.job_id)
            record.email_sent = False


# ───── Module-level singleton (test code may reset by re-importing) ─────


_registry: JobRegistry | None = None


def get_registry() -> JobRegistry:
    """Return (and lazily create) the singleton :class:`JobRegistry`."""
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry


def reset_registry_for_tests() -> None:
    """Test helper — fresh registry so jobs from one test don't leak into another."""
    global _registry
    _registry = JobRegistry()


__all__ = [
    "JobRecord",
    "JobRegistry",
    "get_registry",
    "reset_registry_for_tests",
]
