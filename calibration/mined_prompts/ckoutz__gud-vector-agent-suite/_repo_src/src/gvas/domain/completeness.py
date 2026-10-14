import hashlib
from datetime import datetime
from enum import StrEnum
from typing import NewType, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gvas.domain.identifiers import BusinessId, MessageId

ChecklistKey = NewType("ChecklistKey", str)
ChecklistItemKey = NewType("ChecklistItemKey", str)
FieldNoteReviewId = NewType("FieldNoteReviewId", UUID)
FollowUpQuestionId = NewType("FollowUpQuestionId", UUID)

FIELD_NOTE_THREAD_PREFIX = "field_note"
FOLLOW_UP_CORRELATION_PREFIX = "field_note_question"


class ChecklistItemRequirement(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class FieldNoteReviewStatus(StrEnum):
    AWAITING_REVIEW = "awaiting_review"
    AWAITING_ANSWERS = "awaiting_answers"
    COMPLETE = "complete"


class FollowUpQuestionStatus(StrEnum):
    PENDING = "pending"
    ASKED = "asked"
    ANSWERED = "answered"


class MissingItemReason(StrEnum):
    MISSING = "missing"
    CONTRADICTION = "contradiction"


class CompletenessModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChecklistItem(CompletenessModel):
    """One reviewable requirement. Prompts and markers are supplied by configuration."""

    key: ChecklistItemKey
    prompt: str = Field(min_length=1)
    requirement: ChecklistItemRequirement = ChecklistItemRequirement.REQUIRED
    evidence_markers: tuple[str, ...] = Field(default_factory=tuple)


class CompletenessChecklist(CompletenessModel):
    business_id: BusinessId
    checklist_key: ChecklistKey
    version: int = Field(ge=1)
    items: tuple[ChecklistItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def item_keys_are_unique(self) -> "CompletenessChecklist":
        keys = [item.key for item in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("checklist item keys must be unique")
        return self

    def item(self, key: ChecklistItemKey) -> ChecklistItem | None:
        for item in self.items:
            if item.key == key:
                return item
        return None

    @property
    def required_item_keys(self) -> frozenset[ChecklistItemKey]:
        return frozenset(
            item.key for item in self.items if item.requirement is ChecklistItemRequirement.REQUIRED
        )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class CorrelatedAnswer(CompletenessModel):
    item_key: ChecklistItemKey
    text: str = Field(min_length=1)
    received_at: datetime

    _received_at_aware = field_validator("received_at")(_aware)


class CompletenessReviewRequest(CompletenessModel):
    business_id: BusinessId
    checklist: CompletenessChecklist
    transcript_text: str
    answers: tuple[CorrelatedAnswer, ...] = Field(default_factory=tuple)
    round_index: int = Field(ge=0)

    @model_validator(mode="after")
    def checklist_belongs_to_business(self) -> "CompletenessReviewRequest":
        if self.checklist.business_id != self.business_id:
            raise ValueError("checklist business must match request business")
        return self


class MissingChecklistItem(CompletenessModel):
    """One follow-up the owner must answer before the review completes.

    A ``MISSING`` item repeats the checklist's configured prompt. A
    ``CONTRADICTION`` item names the checklist item the transcript conflicts on
    and carries the question that resolves the conflict.
    """

    item_key: ChecklistItemKey
    prompt: str = Field(min_length=1)
    detail: str | None = None
    reason: MissingItemReason = MissingItemReason.MISSING


class CompletenessReviewOutcome(CompletenessModel):
    missing_items: tuple[MissingChecklistItem, ...] = Field(default_factory=tuple)
    detail: str | None = None

    @model_validator(mode="after")
    def missing_items_are_unique(self) -> "CompletenessReviewOutcome":
        keys = [item.item_key for item in self.missing_items]
        if len(set(keys)) != len(keys):
            raise ValueError("missing checklist items must be unique")
        return self

    @property
    def is_complete(self) -> bool:
        return not self.missing_items


class CompletenessReviewPort(Protocol):
    """Provider-neutral review of a transcript plus correlated owner answers."""

    async def review(self, request: CompletenessReviewRequest) -> CompletenessReviewOutcome: ...


class DetectedContradiction(CompletenessModel):
    """A hard conflict in the transcript that would leave the report wrong as-is."""

    item_key: ChecklistItemKey
    question: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class ContradictionGuardOutcome(CompletenessModel):
    contradictions: tuple[DetectedContradiction, ...] = Field(default_factory=tuple)
    detail: str | None = None

    @property
    def is_clear(self) -> bool:
        return not self.contradictions


class ContradictionGuardPort(Protocol):
    """Provider-neutral second pass that only looks for hard contradictions."""

    async def detect(self, request: CompletenessReviewRequest) -> ContradictionGuardOutcome: ...


class UnknownChecklistError(LookupError):
    pass


class ChecklistVersionConflictError(ValueError):
    pass


class UnknownChecklistItemError(ValueError):
    pass


class InvalidCompletenessReviewOutcomeError(ValueError):
    pass


def field_note_thread_correlation_id(inbound_message_id: MessageId) -> str:
    """Channel-neutral thread identity for a field-note conversation."""

    return f"{FIELD_NOTE_THREAD_PREFIX}:{inbound_message_id}"


def transcript_fingerprint(transcript_text: str) -> str:
    """Stable identity of the reviewed transcript revision."""

    return hashlib.sha256(transcript_text.encode()).hexdigest()


def follow_up_correlation_id(
    review_id: FieldNoteReviewId, round_index: int, item_key: ChecklistItemKey
) -> str:
    if round_index < 1:
        raise ValueError("follow-up rounds are one-based")
    return f"{FOLLOW_UP_CORRELATION_PREFIX}:{review_id}:{round_index}:{item_key}"


class ActiveFieldNoteReviewExistsError(ValueError):
    pass
