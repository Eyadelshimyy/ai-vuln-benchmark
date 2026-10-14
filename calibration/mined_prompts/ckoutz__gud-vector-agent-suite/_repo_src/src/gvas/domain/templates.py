from enum import StrEnum
from typing import NewType, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.completeness import ChecklistItemKey, ChecklistKey, CompletenessChecklist
from gvas.domain.identifiers import BusinessId, ConversationId

TemplateSetKey = NewType("TemplateSetKey", str)
IndustryKey = NewType("IndustryKey", str)


class TemplateSetStatus(StrEnum):
    """DRAFT resolves only by explicit version; DEPRECATED stays readable."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class TemplateModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TemplateSetRef(TemplateModel):
    """Pin recorded on an in-flight case."""

    business_id: BusinessId
    template_set_key: TemplateSetKey
    version: int = Field(ge=1)


class ReportTemplateRef(TemplateModel):
    """Report template a case was pinned to, carried into report generation."""

    business_id: BusinessId
    report_template_key: str = Field(min_length=1)
    report_template_version: int = Field(ge=1)


class ReportTemplateSection(TemplateModel):
    """One output section of a business's report, bound to checklist items.

    The bindings are what keep report structure out of the generator: a section
    names the checklist items whose evidence belongs under its heading.
    """

    section_key: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    heading: str = Field(min_length=1)
    checklist_item_keys: tuple[ChecklistItemKey, ...] = ()

    @model_validator(mode="after")
    def item_keys_are_unique(self) -> "ReportTemplateSection":
        if len(set(self.checklist_item_keys)) != len(self.checklist_item_keys):
            raise ValueError(f"section {self.section_key} binds a checklist item twice")
        return self


class ReportTemplateDefinition(TemplateModel):
    """Immutable versioned report structure for one business.

    Report generation receives this definition, so the section schema of a report
    is tenant configuration rather than industry logic inside a generator.
    """

    business_id: BusinessId
    report_template_key: str = Field(min_length=1)
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    sections: tuple[ReportTemplateSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def section_keys_are_unique(self) -> "ReportTemplateDefinition":
        section_keys = [section.section_key for section in self.sections]
        if len(set(section_keys)) != len(section_keys):
            raise ValueError("report template section keys must be unique")
        return self

    @property
    def ref(self) -> "ReportTemplateRef":
        return ReportTemplateRef(
            business_id=self.business_id,
            report_template_key=self.report_template_key,
            report_template_version=self.version,
        )

    def validate_against(self, checklist: CompletenessChecklist) -> None:
        """Reject bindings to checklist items the pinned checklist does not define."""
        known = {item.key for item in checklist.items}
        unknown = sorted(
            item_key
            for section in self.sections
            for item_key in section.checklist_item_keys
            if item_key not in known
        )
        if unknown:
            raise UnknownChecklistBindingError(
                f"report template {self.report_template_key} version {self.version} "
                f"binds unknown checklist items: {unknown}"
            )


class TemplateSet(TemplateModel):
    """Per-business unit of versioning for a note schema and its report template.

    A template set version pins one checklist version, so publishing a new
    checklist version requires publishing a new template-set version. The
    industry is an attribute rather than a resolution axis: a business working in
    two industries has two template-set keys.
    """

    business_id: BusinessId
    template_set_key: TemplateSetKey
    version: int = Field(ge=1)
    industry_key: IndustryKey
    status: TemplateSetStatus
    checklist_key: ChecklistKey
    checklist_version: int = Field(ge=1)
    report_template_key: str = Field(min_length=1)
    report_template_version: int = Field(ge=1)

    @property
    def ref(self) -> TemplateSetRef:
        return TemplateSetRef(
            business_id=self.business_id,
            template_set_key=self.template_set_key,
            version=self.version,
        )

    @property
    def report_template(self) -> ReportTemplateRef:
        return ReportTemplateRef(
            business_id=self.business_id,
            report_template_key=self.report_template_key,
            report_template_version=self.report_template_version,
        )


class BusinessTemplateProfile(TemplateModel):
    """Which template-set key a business resolves, and its seeded fallback.

    `template_set_key` is what onboarding or a later reassignment points at;
    `default_template_set_key` is the key seeded for the business's industry, so
    a business that never published a set of its own still resolves.
    """

    business_id: BusinessId
    industry_key: IndustryKey
    template_set_key: TemplateSetKey
    default_template_set_key: TemplateSetKey


class TemplateResolutionPort(Protocol):
    """Selects the template set for a new case. No provider, no channel."""

    async def resolve_for_new_case(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> TemplateSetRef: ...

    async def load(self, ref: TemplateSetRef) -> TemplateSet: ...

    async def load_report_template(self, ref: ReportTemplateRef) -> ReportTemplateDefinition: ...


class UnknownTemplateSetError(LookupError):
    """No template set resolves for the business; reviewing would be a guess."""


class TemplateSetVersionConflictError(ValueError):
    """A published template-set version is immutable apart from its status."""


class UnknownReportTemplateError(LookupError):
    """The pinned report template definition is missing; nothing can be rendered."""


class ReportTemplateVersionConflictError(ValueError):
    """A published report template version is immutable."""


class UnknownChecklistBindingError(ValueError):
    """A report template section binds a checklist item that does not exist."""
