from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.field_notes import (
    FieldNoteCaseId,
    FieldNoteCaseStatus,
    FieldNotePart,
    FieldNotePartId,
    FieldNotePartKind,
    TranscriptionStatus,
    validate_field_note_part_values,
)
from gvas.domain.identifiers import (
    BusinessId,
    ConversationId,
    EndpointId,
    MessageId,
    MessageKey,
)
from gvas.domain.messages import (
    AttachmentReference,
    AudioReference,
    ConversationRef,
    TranscriptResult,
)


class FieldNoteRepositoryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FieldNotePartDraft(FieldNoteRepositoryModel):
    kind: FieldNotePartKind
    text: str | None = None
    attachment: AttachmentReference | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "FieldNotePartDraft":
        status = (
            TranscriptionStatus.PENDING
            if self.kind is FieldNotePartKind.AUDIO
            else TranscriptionStatus.NOT_REQUIRED
        )
        validate_field_note_part_values(self.kind, self.text, self.attachment, status, None)
        return self


class FieldNoteCaseRecord(FieldNoteRepositoryModel):
    case_id: FieldNoteCaseId
    business_id: BusinessId
    conversation_id: ConversationId
    conversation_ref: ConversationRef
    origin_inbound_message_id: MessageId
    status: FieldNoteCaseStatus
    parts: tuple[FieldNotePart, ...]


class FieldNoteIntakeResult(FieldNoteRepositoryModel):
    case: FieldNoteCaseRecord
    created_case: bool
    created_part_ids: tuple[FieldNotePartId, ...] = Field(default_factory=tuple)
    audio_part_ids: tuple[FieldNotePartId, ...] = Field(default_factory=tuple)


class FieldNoteCaseClosureResult(StrEnum):
    CLOSED = "closed"
    ALREADY_CLOSED = "already_closed"
    MISSING = "missing"


class TranscriptionClaimResult(StrEnum):
    ACQUIRED = "acquired"
    TERMINAL = "terminal"
    BUSY = "busy"
    MISSING = "missing"


class TranscriptionClaim(FieldNoteRepositoryModel):
    result: TranscriptionClaimResult
    part_id: FieldNotePartId | None = None
    business_id: BusinessId | None = None
    audio: AudioReference | None = None
    attempts: int = Field(default=0, ge=0)
    lease_token: UUID | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "TranscriptionClaim":
        identifiers = (self.part_id, self.business_id, self.audio, self.lease_token)
        if self.result is TranscriptionClaimResult.ACQUIRED:
            if any(value is None for value in identifiers):
                raise ValueError("acquired transcription claims require lease details")
        elif self.result is TranscriptionClaimResult.MISSING:
            if any(value is not None for value in identifiers):
                raise ValueError("missing transcription claims carry no details")
        elif self.audio is not None or self.lease_token is not None:
            raise ValueError("non-acquired transcription claims cannot carry lease details")
        elif self.part_id is None or self.business_id is None:
            raise ValueError("terminal or busy claims require part and business identifiers")
        return self


class LostTranscriptionLeaseError(ValueError):
    pass


class CrossBusinessFieldNoteError(ValueError):
    pass


class FieldNoteMessageLocation(FieldNoteRepositoryModel):
    business_id: BusinessId
    endpoint_id: EndpointId
    conversation_id: ConversationId
    inbound_message_id: MessageId


class AmbiguousFieldNoteMessageError(LookupError):
    pass


class FieldNoteCaseRepository(Protocol):
    async def get(
        self, business_id: BusinessId, case_id: FieldNoteCaseId
    ) -> FieldNoteCaseRecord | None: ...

    async def record_intake(
        self,
        *,
        location: FieldNoteMessageLocation,
        parts: Sequence[FieldNotePartDraft],
        case_id: FieldNoteCaseId | None,
    ) -> FieldNoteIntakeResult: ...

    async def close(
        self, business_id: BusinessId, case_id: FieldNoteCaseId, *, now: datetime
    ) -> FieldNoteCaseClosureResult: ...


class FieldNoteMessageLocator(Protocol):
    async def locate(
        self, business_id: BusinessId, conversation_ref: ConversationRef, message_key: MessageKey
    ) -> FieldNoteMessageLocation | None: ...


class FieldNoteConversationStateRepository(Protocol):
    async def get_active_case_id(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> FieldNoteCaseId | None: ...

    async def set_active_case(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        case_id: FieldNoteCaseId,
        *,
        now: datetime,
    ) -> None: ...

    async def clear_active_case(
        self, business_id: BusinessId, conversation_id: ConversationId, *, now: datetime
    ) -> None: ...


class FieldNoteTranscriptionRepository(Protocol):
    async def claim(
        self,
        business_id: BusinessId,
        part_id: FieldNotePartId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> TranscriptionClaim: ...

    async def record_success(
        self, claim: TranscriptionClaim, transcript: TranscriptResult
    ) -> None: ...

    async def record_failure(self, claim: TranscriptionClaim, error: str) -> None: ...


class FieldNoteUnitOfWork(Protocol):
    field_note_cases: FieldNoteCaseRepository
    field_note_messages: FieldNoteMessageLocator
    field_note_conversation_states: FieldNoteConversationStateRepository
    field_note_transcriptions: FieldNoteTranscriptionRepository

    async def __aenter__(self) -> "FieldNoteUnitOfWork": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
