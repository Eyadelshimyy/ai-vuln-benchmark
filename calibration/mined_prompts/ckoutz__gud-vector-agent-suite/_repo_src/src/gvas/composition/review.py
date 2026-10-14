from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from gvas.application.completeness import (
    CompletenessOutcome,
    CompletenessStatus,
    FieldNoteCompletenessService,
)
from gvas.application.field_note_transcription import FieldNoteTranscriptService
from gvas.domain.completeness import FieldNoteReviewId
from gvas.domain.field_note_repositories import FieldNoteUnitOfWork
from gvas.domain.field_notes import (
    FieldNoteCaseId,
    FieldNoteCaseNotFoundError,
    FieldNoteReviewTrigger,
)
from gvas.domain.identifiers import BusinessId, MessageId
from gvas.domain.messages import NormalizedOwnerMessage
from gvas.domain.outbox import OutboxCommand
from gvas.domain.reporting import field_notes_report_command
from gvas.domain.repositories import UnitOfWork
from gvas.domain.usage import UsageCeilingGuard, UsageKind


class ReviewCoordinationStatus(StrEnum):
    TRANSCRIPT_INCOMPLETE = "transcript_incomplete"
    QUESTIONS_SENT = "questions_sent"
    AWAITING_ANSWERS = "awaiting_answers"
    COMPLETE = "complete"
    ALREADY_COMPLETE = "already_complete"
    IGNORED_REPLY = "ignored_reply"
    CEILING_REACHED = "ceiling_reached"


@dataclass(frozen=True)
class ReviewCoordinationOutcome:
    status: ReviewCoordinationStatus
    review_id: FieldNoteReviewId | None = None
    questions_sent: int = 0
    report_requested: bool = False
    detail: str | None = None


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class MessageUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class MissingOwnerReplyError(LookupError):
    pass


_COORDINATION_STATUSES = {
    CompletenessStatus.QUESTIONS_SENT: ReviewCoordinationStatus.QUESTIONS_SENT,
    CompletenessStatus.AWAITING_ANSWERS: ReviewCoordinationStatus.AWAITING_ANSWERS,
    CompletenessStatus.COMPLETE: ReviewCoordinationStatus.COMPLETE,
    CompletenessStatus.ALREADY_COMPLETE: ReviewCoordinationStatus.ALREADY_COMPLETE,
    CompletenessStatus.DUPLICATE_REPLY: ReviewCoordinationStatus.IGNORED_REPLY,
    CompletenessStatus.UNCORRELATED_REPLY: ReviewCoordinationStatus.IGNORED_REPLY,
    CompletenessStatus.NO_ACTIVE_REVIEW: ReviewCoordinationStatus.IGNORED_REPLY,
}

_COMPLETED_STATUSES = frozenset({CompletenessStatus.COMPLETE, CompletenessStatus.ALREADY_COMPLETE})


class CoordinateFieldNoteReviewService:
    """Joins a field-note case to its completeness review and report request.

    Review only starts once the canonical transcript has no pending or failed
    audio. Each transcript revision of a case is reviewed once and enqueues
    exactly one report command; an already-complete review re-requests the report
    because the review commit and the report enqueue are separate transactions,
    and the command is keyed on the reviewed revision, so recovery cannot produce
    a second report while notes added to an open case still produce the next
    report version.
    """

    def __init__(
        self,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        message_unit_of_work_factory: MessageUnitOfWorkFactory,
        transcripts: FieldNoteTranscriptService,
        completeness: FieldNoteCompletenessService,
        ceilings: UsageCeilingGuard | None = None,
    ) -> None:
        self._field_notes = field_note_unit_of_work_factory
        self._messages = message_unit_of_work_factory
        self._transcripts = transcripts
        self._completeness = completeness
        self._ceilings = ceilings or UsageCeilingGuard()

    async def coordinate(
        self,
        business_id: BusinessId,
        case_id: FieldNoteCaseId,
        trigger: FieldNoteReviewTrigger,
        *,
        now: datetime,
        owner_reply_message_id: MessageId | None = None,
    ) -> ReviewCoordinationOutcome:
        async with self._field_notes() as unit_of_work:
            case = await unit_of_work.field_note_cases.get(business_id, case_id)
            await unit_of_work.commit()
        if case is None:
            raise FieldNoteCaseNotFoundError("field-note case was not found")

        transcript = await self._transcripts.canonical_transcript(business_id, case_id)
        if not transcript.is_complete:
            return ReviewCoordinationOutcome(
                ReviewCoordinationStatus.TRANSCRIPT_INCOMPLETE,
                detail="canonical transcript still has pending or failed audio",
            )
        # Nothing of the round is recorded: a primary pass that found the note
        # complete would go straight to the review model.
        if await self._ceilings.is_reached(business_id, UsageKind.REVIEW_TOKENS, now=now):
            return ReviewCoordinationOutcome(
                ReviewCoordinationStatus.CEILING_REACHED,
                detail="monthly review ceiling reached",
            )

        outcome: CompletenessOutcome | None = None
        if trigger is FieldNoteReviewTrigger.REPLY and owner_reply_message_id is not None:
            outcome = await self._completeness.record_owner_reply(
                business_id,
                case.conversation_id,
                owner_reply_message_id,
                await self._owner_reply(owner_reply_message_id),
            )
            if outcome.status is CompletenessStatus.NO_ACTIVE_REVIEW:
                outcome = None
        if outcome is None:
            outcome = await self._completeness.start_review(
                business_id,
                case.conversation_id,
                case.conversation_ref.external_conversation_id,
                case.origin_inbound_message_id,
                transcript.text,
            )

        report_requested = False
        if outcome.status in _COMPLETED_STATUSES and outcome.review_id is not None:
            await self._enqueue(
                field_notes_report_command(business_id, case_id, outcome.review_id, now)
            )
            report_requested = True
        return ReviewCoordinationOutcome(
            _COORDINATION_STATUSES[outcome.status],
            review_id=outcome.review_id,
            questions_sent=outcome.questions_sent,
            report_requested=report_requested,
            detail=outcome.detail,
        )

    async def _owner_reply(self, inbound_message_id: MessageId) -> NormalizedOwnerMessage:
        async with self._messages() as unit_of_work:
            record = await unit_of_work.inbound_messages.get_for_processing(inbound_message_id)
            await unit_of_work.commit()
        if record is None:
            raise MissingOwnerReplyError("owner reply message is not persisted")
        return record.message

    async def _enqueue(self, command: OutboxCommand) -> None:
        async with self._messages() as unit_of_work:
            await unit_of_work.outbox.enqueue(command)
            await unit_of_work.commit()
