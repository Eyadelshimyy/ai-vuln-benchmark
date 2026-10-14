from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from gvas.domain.identifiers import BusinessId
from gvas.domain.reporting import (
    FieldNoteCaseSnapshot,
    FieldNotesReportDocument,
    FieldNotesReportVersion,
    LostReportLeaseError,
    ReportClaimResult,
    ReportGenerationClaim,
    ReportStatus,
    field_notes_report_id,
    field_notes_report_version_id,
)
from gvas.infrastructure.models import FieldNoteReport, FieldNoteReportVersion


def _validate_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _rowcount(result: Result[Any]) -> int:
    return cast(CursorResult[Any], result).rowcount


def _as_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class SqlFieldNotesReportRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _version_record(
        report: FieldNoteReport,
        version: FieldNoteReportVersion,
    ) -> FieldNotesReportVersion:
        return FieldNotesReportVersion(
            report_id=report.id,
            report_version_id=version.id,
            business_id=BusinessId(report.business_id),
            case_id=report.case_id,
            version=version.version,
            source_fingerprint=version.source_fingerprint,
            generated_at=_as_aware(version.generated_at),
            document=FieldNotesReportDocument.model_validate(version.document),
        )

    async def claim(
        self,
        source: FieldNoteCaseSnapshot,
        source_fingerprint: str,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> ReportGenerationClaim:
        _validate_aware(now, "now")
        _validate_aware(stale_before, "stale_before")
        report_id = field_notes_report_id(source.business_id, source.case_id)
        row = await self.session.scalar(
            select(FieldNoteReport)
            .where(
                FieldNoteReport.business_id == source.business_id,
                FieldNoteReport.case_id == source.case_id,
            )
            .with_for_update()
        )
        if row is None:
            row = FieldNoteReport(
                id=report_id,
                business_id=source.business_id,
                case_id=source.case_id,
                status=ReportStatus.GENERATING.value,
                attempts=1,
                started_at=now,
                leased_at=now,
                lease_token=uuid4(),
                source_fingerprint=source_fingerprint,
                current_version=0,
            )
            try:
                async with self.session.begin_nested():
                    self.session.add(row)
                    await self.session.flush()
            except IntegrityError:
                row = await self.session.scalar(
                    select(FieldNoteReport)
                    .where(
                        FieldNoteReport.business_id == source.business_id,
                        FieldNoteReport.case_id == source.case_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    raise
                return await self._claim_existing(
                    row,
                    source_fingerprint,
                    now=now,
                    stale_before=stale_before,
                )
            return self._acquired_claim(row)
        return await self._claim_existing(
            row,
            source_fingerprint,
            now=now,
            stale_before=stale_before,
        )

    async def _claim_existing(
        self,
        row: FieldNoteReport,
        source_fingerprint: str,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> ReportGenerationClaim:
        existing_version = await self.session.scalar(
            select(FieldNoteReportVersion).where(
                FieldNoteReportVersion.report_id == row.id,
                FieldNoteReportVersion.source_fingerprint == source_fingerprint,
            )
        )
        if existing_version is not None:
            completed = self._version_record(row, existing_version)
            return ReportGenerationClaim(
                result=ReportClaimResult.TERMINAL,
                report_id=row.id,
                business_id=BusinessId(row.business_id),
                case_id=row.case_id,
                status=ReportStatus.COMPLETED,
                attempts=row.attempts,
                report_version=completed.version,
                source_fingerprint=source_fingerprint,
                completed_version=completed,
            )

        leased_at = None if row.leased_at is None else _as_aware(row.leased_at)
        if (
            row.status == ReportStatus.GENERATING.value
            and leased_at is not None
            and leased_at > stale_before
        ):
            return ReportGenerationClaim(
                result=ReportClaimResult.BUSY,
                report_id=row.id,
                business_id=BusinessId(row.business_id),
                case_id=row.case_id,
                status=ReportStatus.GENERATING,
                attempts=row.attempts,
                report_version=row.current_version + 1,
                source_fingerprint=row.source_fingerprint,
            )

        row.status = ReportStatus.GENERATING.value
        row.attempts += 1
        row.leased_at = now
        row.lease_token = uuid4()
        row.source_fingerprint = source_fingerprint
        row.finished_at = None
        row.error = None
        return self._acquired_claim(row)

    @staticmethod
    def _acquired_claim(row: FieldNoteReport) -> ReportGenerationClaim:
        return ReportGenerationClaim(
            result=ReportClaimResult.ACQUIRED,
            report_id=row.id,
            business_id=BusinessId(row.business_id),
            case_id=row.case_id,
            status=ReportStatus.GENERATING,
            attempts=row.attempts,
            report_version=row.current_version + 1,
            source_fingerprint=row.source_fingerprint,
            lease_token=row.lease_token,
        )

    @staticmethod
    def _lease_filter(
        claim: ReportGenerationClaim,
    ) -> tuple[ColumnElement[bool], ...]:
        if claim.result is not ReportClaimResult.ACQUIRED or claim.lease_token is None:
            raise LostReportLeaseError("report claim does not hold an active lease")
        return (
            FieldNoteReport.id == claim.report_id,
            FieldNoteReport.business_id == claim.business_id,
            FieldNoteReport.lease_token == claim.lease_token,
            FieldNoteReport.status == ReportStatus.GENERATING.value,
            FieldNoteReport.source_fingerprint == claim.source_fingerprint,
        )

    async def complete(
        self,
        claim: ReportGenerationClaim,
        document: FieldNotesReportDocument,
        *,
        generated_at: datetime,
    ) -> FieldNotesReportVersion:
        _validate_aware(generated_at, "generated_at")
        result = await self.session.execute(
            update(FieldNoteReport)
            .where(*self._lease_filter(claim))
            .values(
                status=ReportStatus.COMPLETED.value,
                current_version=claim.report_version,
                finished_at=generated_at,
                error=None,
            )
        )
        if _rowcount(result) != 1:
            raise LostReportLeaseError("report claim is no longer active")

        version_id = field_notes_report_version_id(
            claim.report_id,
            claim.report_version,
            claim.source_fingerprint,
        )
        row = FieldNoteReportVersion(
            id=version_id,
            business_id=claim.business_id,
            report_id=claim.report_id,
            version=claim.report_version,
            schema_version=document.schema_version,
            source_fingerprint=claim.source_fingerprint,
            document=document.model_dump(mode="json"),
            generated_at=generated_at,
        )
        self.session.add(row)
        await self.session.flush()
        return FieldNotesReportVersion(
            report_id=claim.report_id,
            report_version_id=version_id,
            business_id=claim.business_id,
            case_id=claim.case_id,
            version=claim.report_version,
            source_fingerprint=claim.source_fingerprint,
            generated_at=generated_at,
            document=document,
        )

    async def fail(
        self,
        claim: ReportGenerationClaim,
        error: str,
        *,
        failed_at: datetime,
    ) -> None:
        _validate_aware(failed_at, "failed_at")
        result = await self.session.execute(
            update(FieldNoteReport)
            .where(*self._lease_filter(claim))
            .values(
                status=ReportStatus.FAILED.value,
                finished_at=failed_at,
                error=error,
            )
        )
        if _rowcount(result) != 1:
            raise LostReportLeaseError("report claim is no longer active")

    async def get_completed(
        self,
        business_id: BusinessId,
        report_id: UUID,
    ) -> FieldNotesReportVersion | None:
        result = await self.session.execute(
            select(FieldNoteReport, FieldNoteReportVersion)
            .join(
                FieldNoteReportVersion,
                (FieldNoteReportVersion.report_id == FieldNoteReport.id)
                & (FieldNoteReportVersion.version == FieldNoteReport.current_version),
            )
            .where(
                FieldNoteReport.business_id == business_id,
                FieldNoteReport.id == report_id,
                FieldNoteReport.status == ReportStatus.COMPLETED.value,
            )
        )
        row = result.one_or_none()
        if row is None:
            return None
        report, version = row
        return self._version_record(report, version)

    async def get_version(
        self,
        business_id: BusinessId,
        report_version_id: UUID,
    ) -> FieldNotesReportVersion | None:
        result = await self.session.execute(
            select(FieldNoteReport, FieldNoteReportVersion)
            .join(FieldNoteReportVersion, FieldNoteReportVersion.report_id == FieldNoteReport.id)
            .where(
                FieldNoteReport.business_id == business_id,
                FieldNoteReportVersion.id == report_version_id,
            )
        )
        row = result.one_or_none()
        if row is None:
            return None
        report, version = row
        return self._version_record(report, version)
