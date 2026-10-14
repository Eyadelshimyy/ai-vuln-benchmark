from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gvas.domain.completeness import ChecklistKey
from gvas.domain.identifiers import BusinessId, JsonValue
from gvas.domain.templates import (
    BusinessTemplateProfile,
    IndustryKey,
    ReportTemplateDefinition,
    ReportTemplateRef,
    ReportTemplateSection,
    ReportTemplateVersionConflictError,
    TemplateSet,
    TemplateSetKey,
    TemplateSetRef,
    TemplateSetStatus,
    TemplateSetVersionConflictError,
)
from gvas.infrastructure.template_models import (
    BusinessTemplateProfileRow,
    FieldNoteReportTemplate,
    FieldNoteTemplateSet,
)


def _template_set(row: FieldNoteTemplateSet) -> TemplateSet:
    return TemplateSet(
        business_id=BusinessId(row.business_id),
        template_set_key=TemplateSetKey(row.template_set_key),
        version=row.version,
        industry_key=IndustryKey(row.industry_key),
        status=TemplateSetStatus(row.status),
        checklist_key=ChecklistKey(row.checklist_key),
        checklist_version=row.checklist_version,
        report_template_key=row.report_template_key,
        report_template_version=row.report_template_version,
    )


def _report_template(row: FieldNoteReportTemplate) -> ReportTemplateDefinition:
    return ReportTemplateDefinition(
        business_id=BusinessId(row.business_id),
        report_template_key=row.report_template_key,
        version=row.version,
        title=row.title,
        sections=tuple(ReportTemplateSection.model_validate(section) for section in row.sections),
    )


def _profile(row: BusinessTemplateProfileRow) -> BusinessTemplateProfile:
    return BusinessTemplateProfile(
        business_id=BusinessId(row.business_id),
        industry_key=IndustryKey(row.industry_key),
        template_set_key=TemplateSetKey(row.template_set_key),
        default_template_set_key=TemplateSetKey(row.default_template_set_key),
    )


class SqlTemplateSetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, template_set: TemplateSet) -> None:
        row = await self._row(template_set.ref)
        if row is None:
            self.session.add(
                FieldNoteTemplateSet(
                    business_id=template_set.business_id,
                    template_set_key=template_set.template_set_key,
                    version=template_set.version,
                    industry_key=template_set.industry_key,
                    status=template_set.status.value,
                    checklist_key=template_set.checklist_key,
                    checklist_version=template_set.checklist_version,
                    report_template_key=template_set.report_template_key,
                    report_template_version=template_set.report_template_version,
                )
            )
            await self.session.flush()
            return
        stored = _template_set(row)
        if stored.model_copy(update={"status": template_set.status}) != template_set:
            raise TemplateSetVersionConflictError(
                f"{template_set.template_set_key} version {template_set.version} is immutable"
            )

    async def get(self, ref: TemplateSetRef) -> TemplateSet | None:
        row = await self._row(ref)
        return None if row is None else _template_set(row)

    async def get_active(
        self, business_id: BusinessId, template_set_key: TemplateSetKey
    ) -> TemplateSet | None:
        row = await self.session.scalar(
            select(FieldNoteTemplateSet).where(
                FieldNoteTemplateSet.business_id == business_id,
                FieldNoteTemplateSet.template_set_key == template_set_key,
                FieldNoteTemplateSet.status == TemplateSetStatus.ACTIVE.value,
            )
        )
        return None if row is None else _template_set(row)

    async def set_status(self, ref: TemplateSetRef, status: TemplateSetStatus) -> TemplateSet:
        await self.session.execute(
            update(FieldNoteTemplateSet)
            .where(
                FieldNoteTemplateSet.business_id == ref.business_id,
                FieldNoteTemplateSet.template_set_key == ref.template_set_key,
                FieldNoteTemplateSet.version == ref.version,
            )
            .values(status=status.value, updated_at=datetime.now(UTC))
        )
        row = await self._row(ref)
        if row is None:
            raise LookupError(
                f"template set {ref.template_set_key} version {ref.version} not found"
            )
        return _template_set(row)

    async def _row(self, ref: TemplateSetRef) -> FieldNoteTemplateSet | None:
        row: FieldNoteTemplateSet | None = await self.session.scalar(
            select(FieldNoteTemplateSet).where(
                FieldNoteTemplateSet.business_id == ref.business_id,
                FieldNoteTemplateSet.template_set_key == ref.template_set_key,
                FieldNoteTemplateSet.version == ref.version,
            )
        )
        return row


class SqlReportTemplateDefinitionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, definition: ReportTemplateDefinition) -> None:
        row = await self._row(definition.ref)
        if row is None:
            sections: list[JsonValue] = [
                section.model_dump(mode="json") for section in definition.sections
            ]
            self.session.add(
                FieldNoteReportTemplate(
                    business_id=definition.business_id,
                    report_template_key=definition.report_template_key,
                    version=definition.version,
                    title=definition.title,
                    sections=sections,
                )
            )
            await self.session.flush()
            return
        if _report_template(row) != definition:
            raise ReportTemplateVersionConflictError(
                f"report template {definition.report_template_key} version "
                f"{definition.version} is immutable"
            )

    async def get(self, ref: ReportTemplateRef) -> ReportTemplateDefinition | None:
        row = await self._row(ref)
        return None if row is None else _report_template(row)

    async def _row(self, ref: ReportTemplateRef) -> FieldNoteReportTemplate | None:
        row: FieldNoteReportTemplate | None = await self.session.scalar(
            select(FieldNoteReportTemplate).where(
                FieldNoteReportTemplate.business_id == ref.business_id,
                FieldNoteReportTemplate.report_template_key == ref.report_template_key,
                FieldNoteReportTemplate.version == ref.report_template_version,
            )
        )
        return row


class SqlBusinessTemplateProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, business_id: BusinessId) -> BusinessTemplateProfile | None:
        row = await self._row(business_id)
        return None if row is None else _profile(row)

    async def upsert(self, profile: BusinessTemplateProfile) -> BusinessTemplateProfile:
        row = await self._row(profile.business_id)
        now = datetime.now(UTC)
        if row is None:
            self.session.add(
                BusinessTemplateProfileRow(
                    business_id=profile.business_id,
                    industry_key=profile.industry_key,
                    template_set_key=profile.template_set_key,
                    default_template_set_key=profile.default_template_set_key,
                    created_at=now,
                    updated_at=now,
                )
            )
            await self.session.flush()
            return profile
        if _profile(row) != profile:
            await self.session.execute(
                update(BusinessTemplateProfileRow)
                .where(BusinessTemplateProfileRow.business_id == profile.business_id)
                .values(
                    industry_key=profile.industry_key,
                    template_set_key=profile.template_set_key,
                    default_template_set_key=profile.default_template_set_key,
                    updated_at=now,
                )
            )
        return profile

    async def _row(self, business_id: BusinessId) -> BusinessTemplateProfileRow | None:
        row: BusinessTemplateProfileRow | None = await self.session.scalar(
            select(BusinessTemplateProfileRow).where(
                BusinessTemplateProfileRow.business_id == business_id
            )
        )
        return row
