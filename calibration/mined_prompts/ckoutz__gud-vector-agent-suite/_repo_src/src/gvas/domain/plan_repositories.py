from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentReference
from gvas.domain.object_storage import StoredObject
from gvas.domain.plans import (
    PlanSetUpload,
    PlanSetUploadId,
    Site,
    SiteId,
    SitePlanSet,
    SitePlanSetId,
    SitePlanSetVersion,
)
from gvas.domain.repositories import OutboxRepository


class PlanRepositoryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlanSetUploadRegistration(PlanRepositoryModel):
    upload: PlanSetUpload
    created: bool


class PlanSetCopyClaimResult(StrEnum):
    ACQUIRED = "acquired"
    TERMINAL = "terminal"
    BUSY = "busy"
    MISSING = "missing"


class PlanSetCopyClaim(PlanRepositoryModel):
    """A fenced lease on one upload's copy into custody."""

    result: PlanSetCopyClaimResult
    upload_id: PlanSetUploadId | None = None
    business_id: BusinessId | None = None
    site_id: SiteId | None = None
    plan_set_id: SitePlanSetId | None = None
    source: AttachmentReference | None = None
    attempts: int = Field(default=0, ge=0)
    lease_token: UUID | None = None
    stored_version: SitePlanSetVersion | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "PlanSetCopyClaim":
        if self.result is PlanSetCopyClaimResult.MISSING:
            if any(
                value is not None
                for value in (
                    self.upload_id,
                    self.business_id,
                    self.site_id,
                    self.plan_set_id,
                    self.source,
                    self.lease_token,
                    self.stored_version,
                )
            ):
                raise ValueError("missing plan-set copy claims carry no details")
            return self
        if self.upload_id is None or self.business_id is None:
            raise ValueError("plan-set copy claims require upload and business identifiers")
        if self.result is PlanSetCopyClaimResult.ACQUIRED:
            if self.lease_token is None or self.source is None or self.plan_set_id is None:
                raise ValueError("acquired plan-set copy claims require lease details")
            if self.stored_version is not None:
                raise ValueError("acquired plan-set copy claims have nothing stored yet")
        else:
            if self.lease_token is not None:
                raise ValueError("non-acquired plan-set copy claims cannot hold a lease")
            if (self.result is PlanSetCopyClaimResult.TERMINAL) != (
                self.stored_version is not None
            ):
                raise ValueError("terminal plan-set copy claims carry the stored version")
        return self


class LostPlanSetCopyLeaseError(ValueError):
    pass


class CrossBusinessPlanError(ValueError):
    pass


class SiteRepository(Protocol):
    async def get(self, business_id: BusinessId, site_id: SiteId) -> Site | None: ...

    async def get_or_create(
        self,
        business_id: BusinessId,
        *,
        label: str,
        external_ref: str | None = None,
        site_id: SiteId | None = None,
        now: datetime,
    ) -> Site: ...


class SitePlanSetRepository(Protocol):
    async def get_or_create(
        self,
        business_id: BusinessId,
        site_id: SiteId,
        plan_set_key: str,
        *,
        now: datetime,
    ) -> SitePlanSet: ...


class SitePlanSetVersionRepository(Protocol):
    async def get(self, business_id: BusinessId, version_id: UUID) -> SitePlanSetVersion | None: ...

    async def list_for_plan_set(
        self, business_id: BusinessId, plan_set_id: SitePlanSetId
    ) -> tuple[SitePlanSetVersion, ...]: ...


class PlanSetUploadRepository(Protocol):
    async def register(
        self,
        business_id: BusinessId,
        site_id: SiteId,
        plan_set_id: SitePlanSetId,
        source: AttachmentReference,
        *,
        now: datetime,
    ) -> PlanSetUploadRegistration: ...

    async def claim(
        self,
        business_id: BusinessId,
        upload_id: PlanSetUploadId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> PlanSetCopyClaim: ...

    async def record_stored(
        self,
        claim: PlanSetCopyClaim,
        stored: StoredObject,
        *,
        uploaded_at: datetime,
    ) -> SitePlanSetVersion: ...

    async def record_failure(self, claim: PlanSetCopyClaim, error: str) -> None: ...


class PlanCustodyUnitOfWork(Protocol):
    sites: SiteRepository
    plan_sets: SitePlanSetRepository
    plan_set_versions: SitePlanSetVersionRepository
    plan_set_uploads: PlanSetUploadRepository
    outbox: OutboxRepository

    async def __aenter__(self) -> "PlanCustodyUnitOfWork": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
