"""Sites and immutable plan-set versions (design §3.1, phase P4).

This module stops where custody stops: an uploaded plan set that is durably
ours, immutable and addressable. Page extraction, sheets and annotations are
later phases and are deliberately absent.
"""

from datetime import datetime
from enum import StrEnum
from typing import NewType
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.identifiers import BusinessId, JsonValue, OutboxCommandId
from gvas.domain.messages import AttachmentReference
from gvas.domain.object_storage import CONTENT_DIGEST_PATTERN
from gvas.domain.outbox import OutboxCommand

SiteId = NewType("SiteId", UUID)
SitePlanSetId = NewType("SitePlanSetId", UUID)
SitePlanSetVersionId = NewType("SitePlanSetVersionId", UUID)
PlanSetUploadId = NewType("PlanSetUploadId", UUID)

PLAN_SET_KEY_PATTERN = r"^[a-z0-9][a-z0-9_.-]*$"
DEFAULT_PLAN_SET_KEY = "default"
PLAN_SET_CUSTODY_SCOPE = "plan-sets"

PLAN_SET_COPY_COMMAND_TYPE = "plan_set.copy_into_custody"
PLAN_SET_COPY_COMMAND_NAMESPACE = UUID("0d1c6a9b-2f45-5a7c-9b31-8c6f2d4e5a71")
PLAN_SET_NAMESPACE = UUID("3a8f1d52-6c74-5b19-8e2a-4f9d0b7c6e35")
PLAN_SET_VERSION_NAMESPACE = UUID("9e4b70c1-58d3-5f26-a4b8-1c7e3d9f0a62")
PLAN_SET_UPLOAD_NAMESPACE = UUID("5c2d9e18-7b46-5d03-9f75-2a8b6c1e4d97")
FIELD_NOTE_CASE_SITE_NAMESPACE = UUID("7e1f4b3a-9c2d-5e60-8a47-3d5b9f1c2e84")


class PlanModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Site(PlanModel):
    site_id: SiteId
    business_id: BusinessId
    label: str = Field(min_length=1)
    external_ref: str | None = None


class SitePlanSet(PlanModel):
    """The append-only series a plan-set version belongs to."""

    plan_set_id: SitePlanSetId
    business_id: BusinessId
    site_id: SiteId
    plan_set_key: str = Field(min_length=1, pattern=PLAN_SET_KEY_PATTERN)


class SitePlanSetVersion(PlanModel):
    """One immutable uploaded revision of a plan set (the whole document).

    The record is immutable. ``page_count`` is ``None`` because it is simply
    unknown at custody time — establishing it means opening the document, which
    custody does not do; how a known page count is eventually represented is
    left to the paused sheet-extraction work and is not promised here.
    """

    version_id: SitePlanSetVersionId
    plan_set_id: SitePlanSetId
    business_id: BusinessId
    site_id: SiteId
    version: int = Field(ge=1)
    artifact: AttachmentReference
    page_count: int | None = Field(default=None, ge=1)
    content_digest: str = Field(min_length=64, max_length=64, pattern=CONTENT_DIGEST_PATTERN)
    byte_size: int = Field(ge=0)
    uploaded_at: datetime

    @field_validator("uploaded_at")
    @classmethod
    def uploaded_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("uploaded_at must be timezone-aware")
        return value


class PlanSetUploadStatus(StrEnum):
    PENDING = "pending"
    COPYING = "copying"
    STORED = "stored"
    FAILED = "failed"


class PlanSetUpload(PlanModel):
    """An owner upload awaiting, or having completed, copy into custody."""

    upload_id: PlanSetUploadId
    business_id: BusinessId
    site_id: SiteId
    plan_set_id: SitePlanSetId
    source: AttachmentReference
    status: PlanSetUploadStatus
    attempts: int = Field(default=0, ge=0)
    version_id: SitePlanSetVersionId | None = None


def site_plan_set_id(business_id: BusinessId, site_id: SiteId, plan_set_key: str) -> SitePlanSetId:
    return SitePlanSetId(uuid5(PLAN_SET_NAMESPACE, f"{business_id}:{site_id}:{plan_set_key}"))


def site_plan_set_version_id(
    plan_set_id: SitePlanSetId, version: int, content_digest: str
) -> SitePlanSetVersionId:
    return SitePlanSetVersionId(
        uuid5(PLAN_SET_VERSION_NAMESPACE, f"{plan_set_id}:{version}:{content_digest}")
    )


def plan_set_upload_id(
    business_id: BusinessId, plan_set_id: SitePlanSetId, source: AttachmentReference
) -> PlanSetUploadId:
    """One upload identity per source file per plan set, so a replayed channel
    event resolves to the same row instead of a second copy."""

    return PlanSetUploadId(
        uuid5(
            PLAN_SET_UPLOAD_NAMESPACE,
            f"{business_id}:{plan_set_id}:{source.attachment_id}:{source.locator}",
        )
    )


def plan_set_custody_name(plan_set_id: SitePlanSetId, content_digest: str) -> str:
    """Deterministic, content-addressed logical name inside the plan-set scope."""

    return f"{plan_set_id}/{content_digest}"


def field_note_case_site_id(business_id: BusinessId, case_id: UUID) -> SiteId:
    """The site a plan set uploaded into a field-note thread belongs to.

    Cases carry no site of their own yet, so each case is its own site;
    re-uploads into the same thread version the same plan set.
    """

    return SiteId(uuid5(FIELD_NOTE_CASE_SITE_NAMESPACE, f"{business_id}:{case_id}"))


def plan_set_copy_command(
    business_id: BusinessId,
    upload_id: PlanSetUploadId,
    *,
    field_note_case_id: UUID | None = None,
) -> OutboxCommand:
    payload: dict[str, JsonValue] = {"plan_set_upload_id": str(upload_id)}
    if field_note_case_id is not None:
        payload["field_note_case_id"] = str(field_note_case_id)
    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(PLAN_SET_COPY_COMMAND_NAMESPACE, str(upload_id))),
        business_id=business_id,
        command_type=PLAN_SET_COPY_COMMAND_TYPE,
        payload=payload,
        dedup_key=f"plan_set_copy:{upload_id}",
    )


class UnknownSiteError(LookupError):
    pass


class PlanSetCustodyFailedError(RuntimeError):
    pass
