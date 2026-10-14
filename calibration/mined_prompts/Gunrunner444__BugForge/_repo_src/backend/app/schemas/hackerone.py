"""HackerOne integration API schemas. Tokens never appear here."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class HackerOneSyncRequest(BaseModel):
    handle: str


class OpenScopeAckRequest(BaseModel):
    policy: str


class ActiveTestingRequest(BaseModel):
    pass


class CreateDraftRequest(BaseModel):
    program_handle: str
    finding_id: UUID
    project_id: UUID
    severity: str | None = None
    weakness_id: int | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_client_findings(cls, value: Any) -> Any:
        if isinstance(value, dict) and "finding" in value:
            raise ValueError(
                "Client-supplied finding payloads are rejected. Pass finding_id of a "
                "persisted BugForge finding. status=verified from the client is ignored."
            )
        return value


class DraftActionRequest(BaseModel):
    program_handle: str | None = None


class DraftEditRequest(BaseModel):
    title: str | None = None
    vulnerability_information: str | None = None
    impact: str | None = None
    severity: str | None = None
    weakness_id: int | None = None


class IntentCreateRequest(BaseModel):
    program_handle: str
    finding_id: UUID
    project_id: UUID
    severity: str | None = None
    weakness_id: int | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_raw_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "payload" in value:
            raise ValueError("Raw HackerOne report-intent payloads cannot bypass BugForge findings")
        return value


class IntentPatchRequest(BaseModel):
    project_id: UUID
    title: str | None = None
    impact: str | None = None
    severity: str | None = None


class IntentSubmitRequest(BaseModel):
    project_id: UUID
    program_handle: str | None = None


class AttachmentCreateRequest(BaseModel):
    filename: str
    content_base64: str = ""
    project_id: UUID
    # Ignored. Client-supplied reviewed=true is never a human review.
    reviewed: bool | None = None


class AttachmentReviewRequest(BaseModel):
    project_id: UUID
    reject: bool = False


class AttachmentAuthorizeRequest(BaseModel):
    project_id: UUID


class IntentActionRequest(BaseModel):
    project_id: UUID
    program_handle: str | None = None


class ReconcileRequest(BaseModel):
    program_handle: str | None = None


# Legacy aliases kept so older tests importing these names fail loudly if misused.
class FindingPayload(BaseModel):
    """Removed. Present only so mis-use is a validation error, not a verified finding."""

    title: str | None = None
    status: str | None = Field(default=None)

    @model_validator(mode="after")
    def blocked(self) -> FindingPayload:
        raise ValueError("FindingPayload is not accepted by the HackerOne API")
