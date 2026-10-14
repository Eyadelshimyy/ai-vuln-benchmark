from __future__ import annotations

from enum import StrEnum
from typing import NewType
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.enums import MediaKind
from gvas.domain.identifiers import (
    BusinessId,
    JsonValue,
    MessageId,
    OutboxCommandId,
    WorkflowIntent,
)
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentReference,
    AudioReference,
    ContentPart,
    ConversationRef,
    NormalizedOwnerMessage,
    TextPart,
    TranscriptResult,
)
from gvas.domain.outbox import OutboxCommand

FieldNoteCaseId = NewType("FieldNoteCaseId", UUID)
FieldNotePartId = NewType("FieldNotePartId", UUID)

FIELD_NOTE_INTENT = WorkflowIntent("field_note.capture")
FIELD_NOTE_TRIGGER_PREFIX = "field notes:"
FIELD_NOTE_CLOSE_TRIGGER = "close notes"
FIELD_NOTE_REPORT_APPROVE_TRIGGER = "approve report"
FIELD_NOTE_REPORT_SEND_TRIGGER = "send report to"
FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE = "field_note.transcribe"
FIELD_NOTE_TRANSCRIBE_COMMAND_NAMESPACE = UUID("b4a7fb38-8c21-4cb9-9de1-4ec9f0c2c7e6")
FIELD_NOTE_REVIEW_COMMAND_TYPE = "field_note.review"
FIELD_NOTE_REVIEW_COMMAND_NAMESPACE = UUID("6f2c9d0e-1a4b-5c68-9f3d-7b8e2c5a1d40")


class FieldNoteModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FieldNoteCaseStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class FieldNotePartKind(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    UNSUPPORTED = "unsupported"


class TranscriptionStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FieldNoteTriggerMatch(FieldNoteModel):
    is_new_case: bool
    parts: tuple[ContentPart, ...]


def match_field_note_trigger(message: NormalizedOwnerMessage) -> FieldNoteTriggerMatch | None:
    if all(
        isinstance(part, AttachmentPart) and part.attachment.media_kind is MediaKind.AUDIO
        for part in message.parts
    ):
        return FieldNoteTriggerMatch(is_new_case=False, parts=message.parts)
    if not message.parts or not isinstance(message.parts[0], TextPart):
        return None
    text = message.parts[0].text
    if not text.lstrip().lower().startswith(FIELD_NOTE_TRIGGER_PREFIX):
        return None
    remainder = text.lstrip()[len(FIELD_NOTE_TRIGGER_PREFIX) :].strip()
    parts: list[ContentPart] = []
    if remainder:
        parts.append(TextPart(text=remainder))
    parts.extend(message.parts[1:])
    return FieldNoteTriggerMatch(is_new_case=True, parts=tuple(parts))


def has_field_note_trigger(message: NormalizedOwnerMessage) -> bool:
    return match_field_note_trigger(message) is not None


def has_field_note_close_trigger(message: NormalizedOwnerMessage) -> bool:
    """Matches the explicit close command that ends a persisted field-note case."""

    return _has_command(message, FIELD_NOTE_CLOSE_TRIGGER)


def has_field_note_report_approve_trigger(message: NormalizedOwnerMessage) -> bool:
    """Matches the owner's sign-off that publishes the current report as a document."""

    return _has_command(message, FIELD_NOTE_REPORT_APPROVE_TRIGGER)


def match_field_note_report_send_trigger(message: NormalizedOwnerMessage) -> str | None:
    """Returns the typed recipient text of ``send report to <address>``, unvalidated.

    Only the leading text part is read; validation of the address belongs to
    the handler so the owner can be told what was wrong with it.
    """

    if not message.parts or not isinstance(message.parts[0], TextPart):
        return None
    text = message.parts[0].text.strip()
    if not text.lower().startswith(FIELD_NOTE_REPORT_SEND_TRIGGER):
        return None
    remainder = text[len(FIELD_NOTE_REPORT_SEND_TRIGGER) :]
    if remainder and not remainder[0].isspace():
        return None
    return remainder.strip()


def has_field_note_report_send_trigger(message: NormalizedOwnerMessage) -> bool:
    return match_field_note_report_send_trigger(message) is not None


def has_field_note_command_trigger(message: NormalizedOwnerMessage) -> bool:
    """Any bare field-note command that acts on the conversation's existing case."""

    return (
        has_field_note_close_trigger(message)
        or has_field_note_report_approve_trigger(message)
        or has_field_note_report_send_trigger(message)
    )


def _has_command(message: NormalizedOwnerMessage, command: str) -> bool:
    if not message.parts or not isinstance(message.parts[0], TextPart):
        return False
    return message.parts[0].text.strip().lower() == command


def validate_field_note_part_values(
    kind: FieldNotePartKind,
    text: str | None,
    attachment: AttachmentReference | None,
    status: TranscriptionStatus,
    transcript: TranscriptResult | None,
) -> None:
    if kind is FieldNotePartKind.TEXT:
        if not text or not text.strip():
            raise ValueError("text field-note parts require non-empty text")
        if attachment is not None:
            raise ValueError("text field-note parts cannot have an attachment")
        if status is not TranscriptionStatus.NOT_REQUIRED or transcript is not None:
            raise ValueError("text field-note parts do not require transcription")
    elif kind is FieldNotePartKind.AUDIO:
        if attachment is None or attachment.media_kind is not MediaKind.AUDIO:
            raise ValueError("audio field-note parts require an audio attachment")
        if text is not None:
            raise ValueError("audio field-note parts cannot have text")
        if (status is TranscriptionStatus.SUCCEEDED) != (transcript is not None):
            raise ValueError("audio transcript must match succeeded status")
        if status is TranscriptionStatus.NOT_REQUIRED:
            raise ValueError("audio field-note parts require transcription lifecycle status")
    else:
        if attachment is None or attachment.media_kind is MediaKind.AUDIO:
            raise ValueError("unsupported field-note parts require non-audio attachments")
        if text is not None:
            raise ValueError("unsupported field-note parts cannot have text")
        if status is not TranscriptionStatus.NOT_REQUIRED or transcript is not None:
            raise ValueError("unsupported field-note parts do not require transcription")


class FieldNotePart(FieldNoteModel):
    part_id: FieldNotePartId
    case_id: FieldNoteCaseId
    business_id: BusinessId
    sequence: int = Field(ge=0)
    kind: FieldNotePartKind
    text: str | None = None
    attachment: AttachmentReference | None = None
    transcription_status: TranscriptionStatus
    transcript: TranscriptResult | None = None
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> FieldNotePart:
        validate_field_note_part_values(
            self.kind,
            self.text,
            self.attachment,
            self.transcription_status,
            self.transcript,
        )
        return self

    def audio_reference(self) -> AudioReference:
        if self.kind is not FieldNotePartKind.AUDIO or self.attachment is None:
            raise ValueError("field-note part is not audio")
        return AudioReference(attachment=self.attachment, business_id=self.business_id)


class FieldNoteCase(FieldNoteModel):
    case_id: FieldNoteCaseId
    business_id: BusinessId
    conversation_ref: ConversationRef
    origin_inbound_message_id: MessageId
    status: FieldNoteCaseStatus
    parts: tuple[FieldNotePart, ...]

    @model_validator(mode="after")
    def validate_parts(self) -> FieldNoteCase:
        if any(
            part.business_id != self.business_id or part.case_id != self.case_id
            for part in self.parts
        ):
            raise ValueError("field-note parts must belong to their case and business")
        sequences = [part.sequence for part in self.parts]
        if sequences != sorted(set(sequences)):
            raise ValueError("field-note part sequences must be strictly increasing")
        return self


class TranscriptSegmentSource(StrEnum):
    TEXT = "text"
    TRANSCRIPTION = "transcription"


class TranscriptSegment(FieldNoteModel):
    sequence: int
    source: TranscriptSegmentSource
    text: str


class CanonicalFieldNoteTranscript(FieldNoteModel):
    case_id: FieldNoteCaseId
    business_id: BusinessId
    segments: tuple[TranscriptSegment, ...]
    pending_parts: int = Field(ge=0)
    failed_parts: int = Field(ge=0)
    unsupported_parts: int = Field(ge=0)

    @property
    def is_complete(self) -> bool:
        return self.pending_parts == 0 and self.failed_parts == 0

    @property
    def text(self) -> str:
        return "\n\n".join(segment.text for segment in self.segments)


def build_canonical_transcript(case: FieldNoteCase) -> CanonicalFieldNoteTranscript:
    segments: list[TranscriptSegment] = []
    pending = 0
    failed = 0
    unsupported = 0
    for part in sorted(case.parts, key=lambda item: item.sequence):
        if part.kind is FieldNotePartKind.TEXT:
            segments.append(
                TranscriptSegment(
                    sequence=part.sequence,
                    source=TranscriptSegmentSource.TEXT,
                    text=part.text or "",
                )
            )
        elif part.kind is FieldNotePartKind.AUDIO:
            if part.transcription_status is TranscriptionStatus.SUCCEEDED:
                text = (part.transcript.text if part.transcript else "").strip()
                if text:
                    segments.append(
                        TranscriptSegment(
                            sequence=part.sequence,
                            source=TranscriptSegmentSource.TRANSCRIPTION,
                            text=text,
                        )
                    )
            elif part.transcription_status is TranscriptionStatus.FAILED:
                failed += 1
            else:
                pending += 1
        else:
            unsupported += 1
    return CanonicalFieldNoteTranscript(
        case_id=case.case_id,
        business_id=case.business_id,
        segments=tuple(segments),
        pending_parts=pending,
        failed_parts=failed,
        unsupported_parts=unsupported,
    )


def field_note_transcribe_command(
    business_id: BusinessId, part_id: FieldNotePartId, case_id: FieldNoteCaseId
) -> OutboxCommand:
    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(FIELD_NOTE_TRANSCRIBE_COMMAND_NAMESPACE, str(part_id))),
        business_id=business_id,
        command_type=FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE,
        payload={"field_note_part_id": str(part_id), "field_note_case_id": str(case_id)},
        dedup_key=f"field_note_transcribe:{part_id}",
    )


class FieldNoteReviewTrigger(StrEnum):
    INTAKE = "intake"
    REPLY = "reply"
    TRANSCRIPTION = "transcription"


def field_note_review_command(
    business_id: BusinessId,
    case_id: FieldNoteCaseId,
    trigger: FieldNoteReviewTrigger,
    trigger_key: str,
    owner_reply_message_id: MessageId | None = None,
) -> OutboxCommand:
    """Hands a field-note case to completeness review once, per distinct trigger.

    ``owner_reply_message_id`` carries the persisted inbound message whose text
    answers the outstanding follow-up question; it is payload data rather than a
    framework linkage because the command is dispatched to the review path.
    """

    if not trigger_key:
        raise ValueError("field-note review commands require a trigger key")
    if (trigger is FieldNoteReviewTrigger.REPLY) != (owner_reply_message_id is not None):
        raise ValueError("only reply-triggered review commands carry an owner reply message")
    key = f"{case_id}:{trigger.value}:{trigger_key}"
    payload: dict[str, JsonValue] = {
        "field_note_case_id": str(case_id),
        "trigger": trigger.value,
        "trigger_key": trigger_key,
    }
    if owner_reply_message_id is not None:
        payload["owner_reply_message_id"] = str(owner_reply_message_id)
    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(FIELD_NOTE_REVIEW_COMMAND_NAMESPACE, key)),
        business_id=business_id,
        command_type=FIELD_NOTE_REVIEW_COMMAND_TYPE,
        payload=payload,
        dedup_key=f"field_note_review:{key}",
    )


class UnsupportedFieldNoteMediaError(ValueError):
    pass


class FieldNoteCaseNotFoundError(LookupError):
    pass


class NoFieldNoteContentError(ValueError):
    pass
