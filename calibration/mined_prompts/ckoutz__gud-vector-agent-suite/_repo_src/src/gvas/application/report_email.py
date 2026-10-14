from typing import Protocol

from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.field_note_repositories import FieldNoteUnitOfWork
from gvas.domain.field_notes import (
    FIELD_NOTE_REPORT_APPROVE_TRIGGER,
    FIELD_NOTE_REPORT_SEND_TRIGGER,
    match_field_note_report_send_trigger,
)
from gvas.domain.identifiers import ConversationId
from gvas.domain.messages import NormalizedOwnerMessage, OutboundOwnerMessage, TextPart
from gvas.domain.outbox import OutboxCommand
from gvas.domain.reporting import (
    ReportUnitOfWork,
    field_notes_report_email_command,
    field_notes_report_id,
    normalize_email_address,
    report_publication_correlation_id,
)
from gvas.domain.repositories import UnitOfWork
from gvas.domain.workflows import WorkflowResult

NO_OPEN_CASE_REPLY = "There is no open field-note case in this conversation to send a report from."
NOT_PUBLISHED_REPLY = (
    "This case's report has not been published yet. Send "
    f"'{FIELD_NOTE_REPORT_APPROVE_TRIGGER}' first, then ask again once the Word "
    "document is in this thread."
)
INVALID_ADDRESS_REPLY = (
    "I need exactly one email address to send the report to, for example "
    f"'{FIELD_NOTE_REPORT_SEND_TRIGGER} name@example.com'. Nothing was sent."
)


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class ReportUnitOfWorkFactory(Protocol):
    def __call__(self) -> ReportUnitOfWork: ...


class MessageUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


def report_email_queued_reply(version: int, recipient_address: str) -> str:
    return f"Emailing report version {version} to {recipient_address}."


def report_email_sent_reply(version: int, recipient_address: str) -> str:
    return f"Report version {version} was emailed to {recipient_address}."


class SendFieldNoteReportHandler:
    """Turns ``send report to <address>`` into an email command for the published DOCX.

    Email is opt-in and never automatic: the thread stays the system of record
    and only a version that ``approve report`` has already published may be
    emailed. The version emailed is the case's current completed report, so
    notes added after a publish must be approved again before they can be
    sent. Each distinct request message is its own attempt, so asking again
    after a dead-lettered email retries it.
    """

    def __init__(
        self,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        report_unit_of_work_factory: ReportUnitOfWorkFactory,
        message_unit_of_work_factory: MessageUnitOfWorkFactory,
    ) -> None:
        self._field_notes = field_note_unit_of_work_factory
        self._reports = report_unit_of_work_factory
        self._messages = message_unit_of_work_factory

    async def send(
        self, message: NormalizedOwnerMessage, conversation_id: ConversationId
    ) -> WorkflowResult:
        typed = match_field_note_report_send_trigger(message)
        recipient = None if typed is None else normalize_email_address(typed)

        async with self._field_notes() as unit_of_work:
            case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
                message.business_id, conversation_id
            )
            await unit_of_work.commit()
        if case_id is None:
            return self._result(message, NO_OPEN_CASE_REPLY, "no open field-note case to send")
        if recipient is None:
            return self._result(message, INVALID_ADDRESS_REPLY, "recipient address rejected")

        async with self._reports() as report_unit_of_work:
            version = await report_unit_of_work.reports.get_completed(
                message.business_id, field_notes_report_id(message.business_id, case_id)
            )
            await report_unit_of_work.commit()
        if version is None:
            return self._result(message, NOT_PUBLISHED_REPLY, "no completed report to send")

        async with self._messages() as unit_of_work:
            published = await unit_of_work.outbound_messages.find_by_correlation(
                message.business_id,
                conversation_id,
                report_publication_correlation_id(version.report_version_id),
            )
            await unit_of_work.commit()
        if published is None:
            return self._result(
                message, NOT_PUBLISHED_REPLY, f"report version {version.version} not published"
            )

        return self._result(
            message,
            report_email_queued_reply(version.version, recipient),
            f"report version {version.version} queued for email",
            commands=(
                field_notes_report_email_command(
                    message.business_id,
                    case_id,
                    version.report_version_id,
                    recipient,
                    str(message.message_key),
                ),
            ),
        )

    @staticmethod
    def _result(
        message: NormalizedOwnerMessage,
        body: str,
        detail: str,
        commands: tuple[OutboxCommand, ...] = (),
    ) -> WorkflowResult:
        reply = OutboundOwnerMessage(
            business_id=message.business_id,
            conversation_ref=message.conversation_ref,
            parts=(TextPart(text=body),),
            correlation_id=f"field_notes_report.email:{message.message_key}",
        )
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(reply,),
            commands=commands,
            detail=detail,
        )
