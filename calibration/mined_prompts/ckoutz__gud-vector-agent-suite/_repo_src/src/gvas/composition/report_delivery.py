from typing import Protocol

from gvas.application.deterministic_report import render_report_text
from gvas.domain.field_note_repositories import FieldNoteUnitOfWork
from gvas.domain.field_notes import FieldNoteCaseId, FieldNoteCaseNotFoundError
from gvas.domain.identifiers import MessageId
from gvas.domain.messages import OutboundOwnerMessage, TextPart
from gvas.domain.outbox import owner_reply_command
from gvas.domain.reporting import FieldNotesReportVersion
from gvas.domain.repositories import UnitOfWork


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class MessageUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class DeliverFieldNotesReportService:
    """Posts a completed report back into the conversation the case came from.

    Delivery goes through the owner reply outbox like any other reply, so the
    channel adapter stays out of reporting. The outbound message is keyed on the
    report version, and outbound creation is idempotent on that key, so a
    replayed report command reuses the one message instead of posting the report
    twice. The case is left open; only ``close notes`` closes it.
    """

    def __init__(
        self,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        message_unit_of_work_factory: MessageUnitOfWorkFactory,
    ) -> None:
        self._field_notes = field_note_unit_of_work_factory
        self._messages = message_unit_of_work_factory

    async def deliver(self, version: FieldNotesReportVersion) -> MessageId:
        case_id = FieldNoteCaseId(version.case_id)
        async with self._field_notes() as unit_of_work:
            case = await unit_of_work.field_note_cases.get(version.business_id, case_id)
            await unit_of_work.commit()
        if case is None:
            raise FieldNoteCaseNotFoundError("field-note case was not found")

        message = OutboundOwnerMessage(
            business_id=version.business_id,
            conversation_ref=case.conversation_ref,
            parts=(TextPart(text=render_report_text(version)),),
            correlation_id=f"field_notes_report:{version.report_id}:{version.version}",
        )
        async with self._messages() as unit_of_work:
            outbound_message_id = await unit_of_work.outbound_messages.create(
                message, case.conversation_id, case.origin_inbound_message_id
            )
            await unit_of_work.outbox.enqueue(
                owner_reply_command(version.business_id, outbound_message_id)
            )
            await unit_of_work.commit()
        return outbound_message_id
