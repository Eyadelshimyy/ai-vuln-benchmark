from typing import Protocol

from gvas.domain.identifiers import BusinessId
from gvas.domain.templates import (
    BusinessTemplateProfile,
    ReportTemplateDefinition,
    ReportTemplateRef,
    TemplateSet,
    TemplateSetKey,
    TemplateSetRef,
    TemplateSetStatus,
)


class TemplateSetRepository(Protocol):
    """Tenant-scoped template-set rows; versions are immutable once written.

    Rows are never deleted or mutated in place, so a case pinned to a version
    keeps resolving it after a newer version is published.
    """

    async def upsert(self, template_set: TemplateSet) -> None: ...

    async def get(self, ref: TemplateSetRef) -> TemplateSet | None: ...

    async def get_active(
        self, business_id: BusinessId, template_set_key: TemplateSetKey
    ) -> TemplateSet | None: ...

    async def set_status(self, ref: TemplateSetRef, status: TemplateSetStatus) -> TemplateSet: ...


class ReportTemplateDefinitionRepository(Protocol):
    """Versioned report structure per business; versions are immutable."""

    async def upsert(self, definition: ReportTemplateDefinition) -> None: ...

    async def get(self, ref: ReportTemplateRef) -> ReportTemplateDefinition | None: ...


class BusinessTemplateProfileRepository(Protocol):
    """One row per business recording its industry and resolvable keys."""

    async def get(self, business_id: BusinessId) -> BusinessTemplateProfile | None: ...

    async def upsert(self, profile: BusinessTemplateProfile) -> BusinessTemplateProfile: ...
