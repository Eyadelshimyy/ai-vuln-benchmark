from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.completeness import (
    ChecklistItem,
    ChecklistKey,
    CompletenessChecklist,
    UnknownChecklistError,
)
from gvas.domain.completeness_repositories import CompletenessUnitOfWork
from gvas.domain.identifiers import BusinessId, ConversationId
from gvas.domain.templates import (
    BusinessTemplateProfile,
    IndustryKey,
    ReportTemplateDefinition,
    ReportTemplateRef,
    ReportTemplateSection,
    TemplateSet,
    TemplateSetKey,
    TemplateSetRef,
    TemplateSetStatus,
    UnknownChecklistBindingError,
    UnknownReportTemplateError,
    UnknownTemplateSetError,
)


class TemplateUnitOfWorkFactory(Protocol):
    def __call__(self) -> CompletenessUnitOfWork: ...


class IndustryTemplateDefinition(BaseModel):
    """A checked-in industry seed: checklist content plus the report template pin.

    Industry policy lives in this data, never in resolution code, so onboarding an
    industry is a seed change rather than a code change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    industry_key: IndustryKey
    template_set_key: TemplateSetKey
    checklist_key: ChecklistKey
    version: int = Field(default=1, ge=1)
    items: tuple[ChecklistItem, ...] = Field(min_length=1)
    report_template_key: str = Field(min_length=1)
    report_template_version: int = Field(default=1, ge=1)
    report_title: str = Field(min_length=1)
    report_sections: tuple[ReportTemplateSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def report_sections_bind_known_items(self) -> "IndustryTemplateDefinition":
        known = {item.key for item in self.items}
        unknown = sorted(
            item_key
            for section in self.report_sections
            for item_key in section.checklist_item_keys
            if item_key not in known
        )
        if unknown:
            raise UnknownChecklistBindingError(
                f"industry {self.industry_key} report sections bind unknown "
                f"checklist items: {unknown}"
            )
        return self

    def checklist(self, business_id: BusinessId) -> CompletenessChecklist:
        return CompletenessChecklist(
            business_id=business_id,
            checklist_key=self.checklist_key,
            version=self.version,
            items=self.items,
        )

    def report_template(self, business_id: BusinessId) -> ReportTemplateDefinition:
        return ReportTemplateDefinition(
            business_id=business_id,
            report_template_key=self.report_template_key,
            version=self.report_template_version,
            title=self.report_title,
            sections=self.report_sections,
        )

    def template_set(self, business_id: BusinessId) -> TemplateSet:
        return TemplateSet(
            business_id=business_id,
            template_set_key=self.template_set_key,
            version=self.version,
            industry_key=self.industry_key,
            status=TemplateSetStatus.ACTIVE,
            checklist_key=self.checklist_key,
            checklist_version=self.version,
            report_template_key=self.report_template_key,
            report_template_version=self.report_template_version,
        )


class TemplateResolver:
    """Resolves the template set a business's next case is reviewed against.

    Resolution order follows the business's assigned key, then the key seeded for
    its industry, then a hard error: silently reviewing a case against some other
    industry's checklist produces a plausible but wrong report.
    """

    def __init__(self, unit_of_work_factory: TemplateUnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def resolve_for_new_case(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> TemplateSetRef:
        async with self._unit_of_work_factory() as unit_of_work:
            profile = await unit_of_work.business_template_profiles.get(business_id)
            if profile is None:
                await unit_of_work.rollback()
                raise UnknownTemplateSetError(
                    f"business {business_id} has no template profile; seed its industry first"
                )
            active = await unit_of_work.template_sets.get_active(
                business_id, profile.template_set_key
            )
            if active is None and profile.default_template_set_key != profile.template_set_key:
                active = await unit_of_work.template_sets.get_active(
                    business_id, profile.default_template_set_key
                )
            await unit_of_work.commit()
        if active is None:
            raise UnknownTemplateSetError(
                f"business {business_id} has no active template set for "
                f"{profile.template_set_key} or {profile.default_template_set_key}"
            )
        return active.ref

    async def load(self, ref: TemplateSetRef) -> TemplateSet:
        async with self._unit_of_work_factory() as unit_of_work:
            template_set = await unit_of_work.template_sets.get(ref)
            await unit_of_work.commit()
        if template_set is None:
            raise UnknownTemplateSetError(
                f"template set {ref.template_set_key} version {ref.version} is not configured"
            )
        return template_set

    async def load_report_template(self, ref: ReportTemplateRef) -> ReportTemplateDefinition:
        async with self._unit_of_work_factory() as unit_of_work:
            definition = await unit_of_work.report_templates.get(ref)
            await unit_of_work.commit()
        if definition is None:
            raise UnknownReportTemplateError(
                f"report template {ref.report_template_key} version "
                f"{ref.report_template_version} is not configured"
            )
        return definition


class PublishTemplateSetService:
    """Publishes template-set versions and seeds industry defaults.

    Publishing inserts the new version and flips statuses in one transaction, so
    at most one version per key is ever ACTIVE, and never rewrites an existing
    version: cases already pinned to it keep resolving it.
    """

    def __init__(self, unit_of_work_factory: TemplateUnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def publish(
        self,
        template_set: TemplateSet,
        checklist: CompletenessChecklist | None = None,
        report_template: ReportTemplateDefinition | None = None,
    ) -> TemplateSet:
        async with self._unit_of_work_factory() as unit_of_work:
            published = await self._write(unit_of_work, template_set, checklist, report_template)
            await unit_of_work.commit()
        return published

    async def seed_industry(
        self, business_id: BusinessId, definition: IndustryTemplateDefinition
    ) -> TemplateSetRef:
        """Copy an industry's seed rows into a business; safe to re-run."""
        async with self._unit_of_work_factory() as unit_of_work:
            published = await self._write(
                unit_of_work,
                definition.template_set(business_id),
                definition.checklist(business_id),
                definition.report_template(business_id),
            )
            profile = await unit_of_work.business_template_profiles.get(business_id)
            assigned = (
                profile.template_set_key
                if profile is not None
                and profile.template_set_key != profile.default_template_set_key
                else definition.template_set_key
            )
            await unit_of_work.business_template_profiles.upsert(
                BusinessTemplateProfile(
                    business_id=business_id,
                    industry_key=definition.industry_key,
                    template_set_key=assigned,
                    default_template_set_key=definition.template_set_key,
                )
            )
            await unit_of_work.commit()
        return published.ref

    async def _write(
        self,
        unit_of_work: CompletenessUnitOfWork,
        template_set: TemplateSet,
        checklist: CompletenessChecklist | None,
        report_template: ReportTemplateDefinition | None,
    ) -> TemplateSet:
        if checklist is not None:
            await unit_of_work.checklists.upsert(checklist)
        if report_template is not None:
            await unit_of_work.report_templates.upsert(report_template)
        await self._validate_bindings(unit_of_work, template_set)
        await unit_of_work.template_sets.upsert(
            template_set.model_copy(update={"status": TemplateSetStatus.DRAFT})
        )
        if template_set.status is not TemplateSetStatus.ACTIVE:
            return await unit_of_work.template_sets.set_status(
                template_set.ref, template_set.status
            )
        current = await unit_of_work.template_sets.get_active(
            template_set.business_id, template_set.template_set_key
        )
        if current is not None and current.version != template_set.version:
            await unit_of_work.template_sets.set_status(current.ref, TemplateSetStatus.DEPRECATED)
        return await unit_of_work.template_sets.set_status(
            template_set.ref, TemplateSetStatus.ACTIVE
        )

    async def _validate_bindings(
        self, unit_of_work: CompletenessUnitOfWork, template_set: TemplateSet
    ) -> None:
        """Check the versions this set pins, not the ones this call happened to write.

        A publish may reuse an already-persisted checklist or report template, and
        the foreign keys only prove both rows exist: they cannot prove the report
        sections bind items of the checklist version this set pins.
        """
        checklist = await unit_of_work.checklists.get(
            template_set.business_id,
            template_set.checklist_key,
            template_set.checklist_version,
        )
        if checklist is None:
            raise UnknownChecklistError(
                f"checklist {template_set.checklist_key} version "
                f"{template_set.checklist_version} is not configured"
            )
        report_template = await unit_of_work.report_templates.get(template_set.report_template)
        if report_template is None:
            raise UnknownReportTemplateError(
                f"report template {template_set.report_template_key} version "
                f"{template_set.report_template_version} is not configured"
            )
        report_template.validate_against(checklist)
