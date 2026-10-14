from uuid import UUID

from gvas.application.report_email import report_email_sent_reply
from gvas.composition.report_delivery import FieldNoteUnitOfWorkFactory, MessageUnitOfWorkFactory
from gvas.composition.report_publication import (
    ReportUnitOfWorkFactory,
    ReportVersionNotFoundError,
)
from gvas.domain.field_notes import FieldNoteCaseId, FieldNoteCaseNotFoundError
from gvas.domain.identifiers import BusinessId, MessageId
from gvas.domain.messages import OutboundOwnerMessage, TextPart
from gvas.domain.outbox import owner_reply_command
from gvas.domain.reporting import (
    ReportArtifactRendererPort,
    ReportEmailPort,
    ReportEmailRequest,
)


class ReportEmailUnavailableError(RuntimeError):
    """No email port is wired; the command stays retryable and dead-letters normally."""


class EmailFieldNotesReportService:
    """Emails a published report version's DOCX to the recipient the owner typed.

    The artifact is re-rendered from the pinned version, so the attachment is
    byte-for-byte the document already in the thread. The provider call is
    keyed on the command's dedup key so a retried command cannot send twice,
    and the thread confirmation is an outbound message correlated on the same
    key so a replay reuses one confirmation. Any provider failure propagates and
    leaves the command retryable.
    """

    def __init__(
        self,
        renderer: ReportArtifactRendererPort,
        email: ReportEmailPort | None,
        report_unit_of_work_factory: ReportUnitOfWorkFactory,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        message_unit_of_work_factory: MessageUnitOfWorkFactory,
    ) -> None:
        self._renderer = renderer
        self._email = email
        self._reports = report_unit_of_work_factory
        self._field_notes = field_note_unit_of_work_factory
        self._messages = message_unit_of_work_factory

    async def send(
        self,
        business_id: BusinessId,
        report_version_id: UUID,
        recipient_address: str,
        idempotency_key: str,
    ) -> MessageId:
        if self._email is None:
            raise ReportEmailUnavailableError("report email is not configured")
        async with self._reports() as report_unit_of_work:
            version = await report_unit_of_work.reports.get_version(business_id, report_version_id)
            await report_unit_of_work.commit()
        if version is None:
            raise ReportVersionNotFoundError("report version was not found")
        async with self._field_notes() as unit_of_work:
            case = await unit_of_work.field_note_cases.get(
                business_id, FieldNoteCaseId(version.case_id)
            )
            await unit_of_work.commit()
        if case is None:
            raise FieldNoteCaseNotFoundError("field-note case was not found")

        artifact = self._renderer.render(version)
        await self._email.deliver(
            ReportEmailRequest(
                business_id=business_id,
                recipient_address=recipient_address,
                idempotency_key=idempotency_key,
                subject=f"{version.document.title} — report version {version.version}",
                body_text=(
                    f"{version.document.title}, approved report version {version.version}, "
                    "is attached as a Word document."
                ),
                artifact=artifact,
            )
        )

        message = OutboundOwnerMessage(
            business_id=business_id,
            conversation_ref=case.conversation_ref,
            parts=(TextPart(text=report_email_sent_reply(version.version, recipient_address)),),
            correlation_id=idempotency_key,
        )
        async with self._messages() as unit_of_work:
            outbound_message_id = await unit_of_work.outbound_messages.create(
                message, case.conversation_id, case.origin_inbound_message_id
            )
            await unit_of_work.outbox.enqueue(owner_reply_command(business_id, outbound_message_id))
            await unit_of_work.commit()
        return outbound_message_id
