from datetime import datetime
from typing import Protocol

from pydantic import ValidationError

from gvas.domain.reporting import (
    FieldNoteCaseSnapshot,
    FieldNoteCaseStatus,
    FieldNotesReportDocument,
    FieldNotesReportVersion,
    IncompleteFieldNoteCaseError,
    MalformedGeneratedReportError,
    ReportClaimResult,
    ReportGenerationBusyError,
    ReportGenerationClaim,
    ReportGenerationFailedError,
    ReportGenerationPort,
    ReportGenerationRequest,
    ReportUnitOfWork,
    field_note_source_fingerprint,
)
from gvas.domain.templates import (
    ReportTemplateDefinition,
    ReportTemplateRef,
    TemplateResolutionPort,
    UnknownReportTemplateError,
)


class ReportUnitOfWorkFactory(Protocol):
    def __call__(self) -> ReportUnitOfWork: ...


class GenerateFieldNotesReportService:
    def __init__(
        self,
        unit_of_work_factory: ReportUnitOfWorkFactory,
        generator: ReportGenerationPort,
        templates: TemplateResolutionPort,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._generator = generator
        self._templates = templates

    async def generate(
        self,
        source: FieldNoteCaseSnapshot,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> FieldNotesReportVersion:
        if source.status is not FieldNoteCaseStatus.COMPLETED:
            raise IncompleteFieldNoteCaseError("only completed field-note cases are eligible")

        report_template = await self._pinned_report_template(source)
        source_fingerprint = field_note_source_fingerprint(source)
        async with self._unit_of_work_factory() as unit_of_work:
            claim = await unit_of_work.reports.claim(
                source,
                source_fingerprint,
                now=now,
                stale_before=stale_before,
            )
            await unit_of_work.commit()

        if claim.result is ReportClaimResult.TERMINAL:
            completed = claim.completed_version
            if completed is None:
                raise RuntimeError("terminal report claim is missing its completed version")
            return completed
        if claim.result is ReportClaimResult.BUSY:
            raise ReportGenerationBusyError("field-notes report generation is already leased")

        request = ReportGenerationRequest(
            report_id=claim.report_id,
            report_version=claim.report_version,
            source_fingerprint=claim.source_fingerprint,
            source=source,
            report_template=report_template,
        )
        try:
            generated = await self._generator.generate(request)
        except Exception as error:
            await self._record_failure(claim, str(error), failed_at=now)
            raise ReportGenerationFailedError("field-notes report generation failed") from error

        try:
            document = FieldNotesReportDocument.model_validate(generated)
            document.validate_evidence_against(source)
            document.validate_against_template(report_template)
        except (ValidationError, ValueError) as error:
            await self._record_failure(claim, str(error), failed_at=now)
            raise MalformedGeneratedReportError("generated report content is invalid") from error

        async with self._unit_of_work_factory() as unit_of_work:
            completed = await unit_of_work.reports.complete(
                claim,
                document,
                generated_at=now,
            )
            await unit_of_work.commit()
        return completed

    async def _pinned_report_template(
        self, source: FieldNoteCaseSnapshot
    ) -> ReportTemplateDefinition:
        """The section schema pinned to the case, so revisions keep their structure."""
        key = source.report_template_key
        version = source.report_template_version
        if key is None or version is None:
            raise UnknownReportTemplateError("field-note case has no pinned report template")
        return await self._templates.load_report_template(
            ReportTemplateRef(
                business_id=source.business_id,
                report_template_key=key,
                report_template_version=version,
            )
        )

    async def _record_failure(
        self, claim: ReportGenerationClaim, error: str, *, failed_at: datetime
    ) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.reports.fail(claim, error, failed_at=failed_at)
            await unit_of_work.commit()
