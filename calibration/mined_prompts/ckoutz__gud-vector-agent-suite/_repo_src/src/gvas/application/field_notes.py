from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from gvas.domain.enums import MediaKind, WorkflowRunStatus
from gvas.domain.field_note_repositories import (
    AmbiguousFieldNoteMessageError,
    FieldNoteCaseClosureResult,
    FieldNotePartDraft,
    FieldNoteUnitOfWork,
)
from gvas.domain.field_notes import (
    FIELD_NOTE_INTENT,
    FieldNotePartKind,
    NoFieldNoteContentError,
    field_note_transcribe_command,
    match_field_note_trigger,
)
from gvas.domain.identifiers import ConversationId, WorkflowIntent
from gvas.domain.messages import (
    AttachmentPart,
    ContentPart,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.workflows import WorkflowContext, WorkflowResult

CLOSED_CASE_REPLY = (
    "Closed the field notes case. Send a new 'field notes:' message to start another one."
)
NO_OPEN_CASE_REPLY = (
    "There is no open field notes case in this conversation. "
    "Send a 'field notes:' message to start one."
)


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class FieldNoteIntentContribution:
    def __init__(self, unit_of_work_factory: FieldNoteUnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def contribute(self, message: NormalizedOwnerMessage) -> WorkflowIntent | None:
        match = match_field_note_trigger(message)
        if match is not None:
            return FIELD_NOTE_INTENT
        async with self._unit_of_work_factory() as unit_of_work:
            try:
                location = await unit_of_work.field_note_messages.locate(
                    message.business_id,
                    message.conversation_ref,
                    message.message_key,
                )
                if location is None:
                    await unit_of_work.commit()
                    return None
                case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
                    location.business_id, location.conversation_id
                )
                await unit_of_work.commit()
                return FIELD_NOTE_INTENT if case_id is not None else None
            except AmbiguousFieldNoteMessageError:
                await unit_of_work.rollback()
                return None


class CloseFieldNoteCaseHandler:
    """Closes the active field-note case on an explicit owner close command.

    Reviews and reports never close a case; only this command does, and it stays
    idempotent so replays and repeated commands produce a single stored reply.
    """

    def __init__(
        self, unit_of_work_factory: FieldNoteUnitOfWorkFactory, *, now: Callable[[], datetime]
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._now = now

    async def close(
        self, message: NormalizedOwnerMessage, conversation_id: ConversationId
    ) -> WorkflowResult:
        now = self._now()
        async with self._unit_of_work_factory() as unit_of_work:
            case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
                message.business_id, conversation_id
            )
            closure = (
                None
                if case_id is None
                else await unit_of_work.field_note_cases.close(
                    message.business_id, case_id, now=now
                )
            )
            if case_id is not None:
                await unit_of_work.field_note_conversation_states.clear_active_case(
                    message.business_id, conversation_id, now=now
                )
            await unit_of_work.commit()
        if closure is FieldNoteCaseClosureResult.CLOSED:
            body = CLOSED_CASE_REPLY
            detail = "field-note case closed"
        else:
            body = NO_OPEN_CASE_REPLY
            detail = "no open field-note case to close"
        reply = OutboundOwnerMessage(
            business_id=message.business_id,
            conversation_ref=message.conversation_ref,
            parts=(TextPart(text=body),),
            correlation_id=f"field_note.close:{message.message_key}",
        )
        return WorkflowResult(status=WorkflowRunStatus.SUCCEEDED, replies=(reply,), detail=detail)


class FieldNoteIntakeHandler:
    intent = FIELD_NOTE_INTENT

    def __init__(
        self, unit_of_work_factory: FieldNoteUnitOfWorkFactory, *, now: Callable[[], datetime]
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._now = now

    @staticmethod
    def _drafts(parts: tuple[ContentPart, ...]) -> tuple[FieldNotePartDraft, ...]:
        drafts: list[FieldNotePartDraft] = []
        for part in parts:
            if isinstance(part, TextPart):
                drafts.append(FieldNotePartDraft(kind=FieldNotePartKind.TEXT, text=part.text))
            elif isinstance(part, AttachmentPart):
                kind = (
                    FieldNotePartKind.AUDIO
                    if part.attachment.media_kind is MediaKind.AUDIO
                    else FieldNotePartKind.UNSUPPORTED
                )
                drafts.append(FieldNotePartDraft(kind=kind, attachment=part.attachment))
        return tuple(drafts)

    @staticmethod
    def _acknowledgement(drafts: tuple[FieldNotePartDraft, ...]) -> TextPart:
        text_count = sum(part.kind is FieldNotePartKind.TEXT for part in drafts)
        audio_count = sum(part.kind is FieldNotePartKind.AUDIO for part in drafts)
        unsupported_count = sum(part.kind is FieldNotePartKind.UNSUPPORTED for part in drafts)
        body = (
            f"Recorded {text_count} text part(s); queued {audio_count} audio part(s) "
            "for transcription."
        )
        if unsupported_count:
            body += f" Skipped {unsupported_count} unsupported attachment(s)."
        return TextPart(text=body)

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        message = context.message
        match = match_field_note_trigger(message)
        drafts = self._drafts(match.parts if match is not None else message.parts)
        if not drafts:
            return WorkflowResult(
                status=WorkflowRunStatus.FAILED,
                detail=str(NoFieldNoteContentError("field note contains no supported content")),
            )
        async with self._unit_of_work_factory() as unit_of_work:
            try:
                location = await unit_of_work.field_note_messages.locate(
                    message.business_id, message.conversation_ref, message.message_key
                )
                if location is None:
                    await unit_of_work.rollback()
                    return WorkflowResult(
                        status=WorkflowRunStatus.FAILED,
                        detail="field note message is not persisted",
                    )
                case_id = None
                if match is None or not match.is_new_case:
                    case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
                        location.business_id, location.conversation_id
                    )
                    if case_id is None and match is None:
                        await unit_of_work.rollback()
                        return WorkflowResult(
                            status=WorkflowRunStatus.FAILED,
                            detail="field note message has no active case",
                        )
                result = await unit_of_work.field_note_cases.record_intake(
                    location=location, parts=drafts, case_id=case_id
                )
                await unit_of_work.field_note_conversation_states.set_active_case(
                    location.business_id,
                    location.conversation_id,
                    result.case.case_id,
                    now=self._now(),
                )
                await unit_of_work.commit()
            except AmbiguousFieldNoteMessageError as error:
                await unit_of_work.rollback()
                return WorkflowResult(status=WorkflowRunStatus.FAILED, detail=str(error))
        reply = OutboundOwnerMessage(
            business_id=message.business_id,
            conversation_ref=message.conversation_ref,
            parts=(self._acknowledgement(drafts),),
            correlation_id=f"field_note.ack:{message.message_key}",
        )
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(reply,),
            commands=tuple(
                field_note_transcribe_command(message.business_id, part_id, result.case.case_id)
                for part_id in result.audio_part_ids
            ),
        )
