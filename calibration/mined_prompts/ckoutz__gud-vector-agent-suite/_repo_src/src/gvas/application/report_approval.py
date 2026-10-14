from typing import Protocol

from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.field_note_repositories import FieldNoteUnitOfWork
from gvas.domain.identifiers import ConversationId
from gvas.domain.messages import NormalizedOwnerMessage, OutboundOwnerMessage, TextPart
from gvas.domain.outbox import OutboxCommand
from gvas.domain.reporting import (
    ReportUnitOfWork,
    field_notes_report_id,
    field_notes_report_publish_command,
)
from gvas.domain.workflows import WorkflowResult

NO_OPEN_CASE_REPLY = (
    "There is no open field-note case in this conversation to approve a report for."
)
NO_REPORT_REPLY = (
    "This case does not have a completed report yet. "
    "Approve once the report text has been posted here."
)


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class ReportUnitOfWorkFactory(Protocol):
    def __call__(self) -> ReportUnitOfWork: ...


def approved_report_reply(version: int) -> str:
    return f"Report version {version} approved. Publishing the Word document to this thread."


class ApproveFieldNoteReportHandler:
    """Turns the owner's in-thread sign-off into a publish command.

    The text report already posted in the thread is the review copy; approval
    pins the case's current completed version and asks the outbox to publish it
    as a document. Nothing is regenerated here and the case stays open, so a
    later note still produces a new version that can be approved in turn.
    Each distinct approval message is its own publish attempt, so approving
    again after a failed publish retries it; the published document is still
    deduplicated on the version.
    """

    def __init__(
        self,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        report_unit_of_work_factory: ReportUnitOfWorkFactory,
    ) -> None:
        self._field_notes = field_note_unit_of_work_factory
        self._reports = report_unit_of_work_factory

    async def approve(
        self, message: NormalizedOwnerMessage, conversation_id: ConversationId
    ) -> WorkflowResult:
        async with self._field_notes() as unit_of_work:
            case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
                message.business_id, conversation_id
            )
            await unit_of_work.commit()
        if case_id is None:
            return self._result(message, NO_OPEN_CASE_REPLY, "no open field-note case to approve")

        async with self._reports() as report_unit_of_work:
            version = await report_unit_of_work.reports.get_completed(
                message.business_id, field_notes_report_id(message.business_id, case_id)
            )
            await report_unit_of_work.commit()
        if version is None:
            return self._result(message, NO_REPORT_REPLY, "no completed report to approve")

        return self._result(
            message,
            approved_report_reply(version.version),
            f"report version {version.version} approved for publication",
            commands=(
                field_notes_report_publish_command(
                    message.business_id,
                    case_id,
                    version.report_version_id,
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
            correlation_id=f"field_notes_report.approve:{message.message_key}",
        )
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(reply,),
            commands=commands,
            detail=detail,
        )
