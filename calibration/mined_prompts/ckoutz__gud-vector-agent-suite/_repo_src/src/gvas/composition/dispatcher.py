import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from gvas.application.field_note_transcription import (
    TranscribeFieldNoteAudioService,
    TranscriptionOutcome,
)
from gvas.application.intake import (
    ArrangeIntakeBookingService,
    CancelIntakeBookingService,
    SendIntakeCustomerEmailService,
    SendIntakeCustomerTextService,
    SendOwnerEmailService,
)
from gvas.application.manual_payments import ManualPaymentEffectsService
from gvas.application.outbox_service import OutboxService
from gvas.application.owner_reply_delivery import (
    DeliverOwnerReplyService,
    OwnerReplyDeliveryStatus,
)
from gvas.application.plan_custody import (
    CopyPlanSetIntoCustodyService,
    PlanSetCustodyOutcome,
)
from gvas.application.processing import ProcessingStatus, ProcessOwnerMessageService
from gvas.application.quotes import (
    DeliverApprovedQuoteService,
    QuoteDeliveryStatus,
    QuoteTextStatus,
    TextDeliveredQuoteService,
)
from gvas.application.report_generation import GenerateFieldNotesReportService
from gvas.composition.failure_notices import NotifyExhaustedCommandService
from gvas.composition.report_delivery import DeliverFieldNotesReportService
from gvas.composition.report_email import EmailFieldNotesReportService
from gvas.composition.report_publication import PublishFieldNotesReportService
from gvas.composition.review import (
    CoordinateFieldNoteReviewService,
    ReviewCoordinationStatus,
)
from gvas.composition.snapshots import BuildFieldNoteCaseSnapshotService
from gvas.domain.completeness import FieldNoteReviewId
from gvas.domain.customers import (
    PORTAL_LOGIN_EMAIL_COMMAND_TYPE,
    portal_login_email_request,
)
from gvas.domain.enums import DeliveryStatus, OutboxStatus
from gvas.domain.field_notes import (
    FIELD_NOTE_REVIEW_COMMAND_TYPE,
    FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE,
    FieldNoteCaseId,
    FieldNotePartId,
    FieldNoteReviewTrigger,
    field_note_review_command,
)
from gvas.domain.identifiers import IntakeConversationId, MessageId, QuoteId
from gvas.domain.intake import (
    INTAKE_BOOKING_ARRANGE_COMMAND_TYPE,
    INTAKE_BOOKING_CANCEL_COMMAND_TYPE,
    INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
    INTAKE_CUSTOMER_TEXT_COMMAND_TYPE,
    OWNER_NOTICE_EMAIL_COMMAND_TYPE,
)
from gvas.domain.outbox import (
    OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
    OWNER_REPLY_COMMAND_TYPE,
    OutboxCommand,
    OutboxRecord,
)
from gvas.domain.payments import (
    CHECKOUT_EXPIRE_COMMAND_TYPE,
    MANUAL_RECEIPT_COMMAND_TYPE,
    PLAN_NUDGE_COMMAND_TYPE,
)
from gvas.domain.plans import PLAN_SET_COPY_COMMAND_TYPE, PlanSetUploadId
from gvas.domain.ports import PortalLoginEmailPort
from gvas.domain.quotes import QUOTE_DELIVERY_COMMAND_TYPE, QUOTE_TEXT_COMMAND_TYPE
from gvas.domain.reporting import (
    FIELD_NOTES_REPORT_COMMAND_TYPE,
    FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE,
    FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE,
    report_email_idempotency_key,
)


class UnknownCommandTypeError(RuntimeError):
    """Raised for commands with no registered handler; the record stays retryable."""


class TransientCommandError(RuntimeError):
    """Raised when a command cannot make progress yet and must be retried."""


class MalformedCommandPayloadError(ValueError):
    pass


logger = logging.getLogger(__name__)

# Commands of the retired reply-by-e-mail channel that may still sit in the
# outbox; they are acknowledged and dropped instead of dead-lettered.
RETIRED_COMMAND_TYPES = frozenset({"owner_email.reply"})


@dataclass(frozen=True)
class DispatchOutcome:
    command_type: str
    detail: str


def _text(command: OutboxCommand, key: str) -> str:
    value = command.payload.get(key)
    if not isinstance(value, str) or not value:
        raise MalformedCommandPayloadError(f"{command.command_type} requires payload {key!r}")
    return value


def _uuid(command: OutboxCommand, key: str) -> UUID:
    try:
        return UUID(_text(command, key))
    except ValueError as error:
        raise MalformedCommandPayloadError(
            f"{command.command_type} payload {key!r} is not a UUID"
        ) from error


class OutboxCommandDispatcher:
    """Routes claimed outbox commands to channel-neutral application services.

    Every provider call happens inside the dispatched service, outside the
    ingress transaction. Commands without a handler fail explicitly so the
    existing outbox retry and dead-letter behaviour applies unchanged.
    """

    def __init__(
        self,
        *,
        processing: ProcessOwnerMessageService,
        owner_replies: DeliverOwnerReplyService,
        quote_delivery: DeliverApprovedQuoteService,
        transcription: TranscribeFieldNoteAudioService,
        review: CoordinateFieldNoteReviewService,
        snapshots: BuildFieldNoteCaseSnapshotService,
        reports: GenerateFieldNotesReportService,
        report_delivery: DeliverFieldNotesReportService,
        report_publication: PublishFieldNotesReportService,
        report_email: EmailFieldNotesReportService,
        outbox: OutboxService,
        now: Callable[[], datetime],
        plan_custody: CopyPlanSetIntoCustodyService | None = None,
        lease_ttl: timedelta = timedelta(minutes=5),
        ceiling_notices: NotifyExhaustedCommandService | None = None,
        quote_text: TextDeliveredQuoteService | None = None,
        portal_login_email: PortalLoginEmailPort | None = None,
        intake_booking: ArrangeIntakeBookingService | None = None,
        intake_email: SendIntakeCustomerEmailService | None = None,
        intake_text: SendIntakeCustomerTextService | None = None,
        intake_booking_cancel: CancelIntakeBookingService | None = None,
        owner_email: SendOwnerEmailService | None = None,
        manual_payments: ManualPaymentEffectsService | None = None,
    ) -> None:
        self._owner_email = owner_email
        self._manual_payments = manual_payments
        self._quote_text = quote_text
        self._portal_login_email = portal_login_email
        self._intake_booking = intake_booking
        self._intake_email = intake_email
        self._intake_text = intake_text
        self._intake_booking_cancel = intake_booking_cancel
        self._processing = processing
        self._owner_replies = owner_replies
        self._quote_delivery = quote_delivery
        self._transcription = transcription
        self._review = review
        self._snapshots = snapshots
        self._reports = reports
        self._report_delivery = report_delivery
        self._report_publication = report_publication
        self._report_email = report_email
        self._outbox = outbox
        self._plan_custody = plan_custody
        self._ceiling_notices = ceiling_notices
        self._now = now
        self._lease_ttl = lease_ttl

    async def dispatch(self, record: OutboxRecord) -> DispatchOutcome:
        command = record.command
        if command.command_type == OWNER_MESSAGE_PROCESS_COMMAND_TYPE:
            return await self._process_owner_message(command)
        if command.command_type == OWNER_REPLY_COMMAND_TYPE:
            return await self._deliver_owner_reply(command)
        if command.command_type == QUOTE_DELIVERY_COMMAND_TYPE:
            return await self._deliver_quote(command)
        if command.command_type == QUOTE_TEXT_COMMAND_TYPE:
            return await self._text_quote(command)
        if command.command_type == FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE:
            return await self._transcribe(command)
        if command.command_type == FIELD_NOTE_REVIEW_COMMAND_TYPE:
            return await self._coordinate_review(command)
        if command.command_type == FIELD_NOTES_REPORT_COMMAND_TYPE:
            return await self._generate_report(command)
        if command.command_type == FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE:
            return await self._publish_report(command)
        if command.command_type == FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE:
            return await self._email_report(command)
        if command.command_type == PLAN_SET_COPY_COMMAND_TYPE:
            return await self._copy_plan_set(command)
        if command.command_type == PORTAL_LOGIN_EMAIL_COMMAND_TYPE:
            return await self._send_portal_login(command)
        if command.command_type == INTAKE_BOOKING_ARRANGE_COMMAND_TYPE:
            return await self._arrange_intake_booking(command)
        if command.command_type == INTAKE_BOOKING_CANCEL_COMMAND_TYPE:
            return await self._cancel_intake_booking(command)
        if command.command_type == OWNER_NOTICE_EMAIL_COMMAND_TYPE and self._owner_email:
            return await self._send_owner_email(command)
        if command.command_type in (
            INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
            OWNER_NOTICE_EMAIL_COMMAND_TYPE,
        ):
            return await self._send_intake_email(command)
        if command.command_type == INTAKE_CUSTOMER_TEXT_COMMAND_TYPE:
            return await self._send_intake_text(command)
        if command.command_type in (
            CHECKOUT_EXPIRE_COMMAND_TYPE,
            MANUAL_RECEIPT_COMMAND_TYPE,
            PLAN_NUDGE_COMMAND_TYPE,
        ):
            return await self._manual_payment_effect(command)
        if command.command_type in RETIRED_COMMAND_TYPES:
            logger.info("dropping retired %s command %s", command.command_type, command.command_id)
            return DispatchOutcome(command.command_type, "retired")
        raise UnknownCommandTypeError(f"no handler is registered for {command.command_type}")

    def _window(self) -> tuple[datetime, datetime]:
        now = self._now()
        return now, now - self._lease_ttl

    async def _process_owner_message(self, command: OutboxCommand) -> DispatchOutcome:
        inbound_message_id = command.inbound_message_id
        if inbound_message_id is None:
            raise MalformedCommandPayloadError("process command is missing its inbound message")
        now, stale_before = self._window()
        outcome = await self._processing.process(
            inbound_message_id, now=now, stale_before=stale_before
        )
        if outcome.status in {ProcessingStatus.BUSY, ProcessingStatus.LEASE_LOST}:
            raise TransientCommandError(f"owner message processing is {outcome.status.value}")
        if outcome.status in {
            ProcessingStatus.MISSING,
            ProcessingStatus.INTENT_UNRESOLVED,
            ProcessingStatus.UNKNOWN_INTENT,
            ProcessingStatus.HANDLER_FAILED,
        }:
            raise RuntimeError(f"owner message processing failed: {outcome.status.value}")
        return DispatchOutcome(command.command_type, outcome.status.value)

    async def _deliver_owner_reply(self, command: OutboxCommand) -> DispatchOutcome:
        outbound_message_id = command.outbound_message_id
        if outbound_message_id is None:
            raise MalformedCommandPayloadError("reply command is missing its outbound message")
        outcome = await self._owner_replies.deliver(outbound_message_id)
        if outcome.status is OwnerReplyDeliveryStatus.MISSING:
            raise RuntimeError("owner reply message is not persisted")
        return DispatchOutcome(command.command_type, outcome.status.value)

    async def _deliver_quote(self, command: OutboxCommand) -> DispatchOutcome:
        quote_id = QuoteId(_uuid(command, "quote_id"))
        outcome = await self._quote_delivery.deliver(command.business_id, quote_id)
        if outcome.status in {QuoteDeliveryStatus.MISSING, QuoteDeliveryStatus.NOT_APPROVED}:
            raise RuntimeError(f"quote delivery is {outcome.status.value}")
        return DispatchOutcome(command.command_type, outcome.status.value)

    async def _text_quote(self, command: OutboxCommand) -> DispatchOutcome:
        if self._quote_text is None:
            raise UnknownCommandTypeError("customer texting is not wired")
        quote_id = QuoteId(_uuid(command, "quote_id"))
        outcome = await self._quote_text.text(command.business_id, quote_id)
        if outcome.status in {QuoteTextStatus.MISSING, QuoteTextStatus.NOT_DELIVERED}:
            raise RuntimeError(f"quote text is {outcome.status.value}")
        return DispatchOutcome(command.command_type, outcome.status.value)

    async def _send_portal_login(self, command: OutboxCommand) -> DispatchOutcome:
        if self._portal_login_email is None:
            raise UnknownCommandTypeError("portal login e-mail is not wired")
        try:
            request = portal_login_email_request(command.business_id, command.payload)
        except ValueError as error:
            raise MalformedCommandPayloadError(str(error)) from error
        if request.is_expired(self._now()):
            return DispatchOutcome(command.command_type, "expired")
        receipt = await self._portal_login_email.send_login_link(request)
        if receipt.status is DeliveryStatus.FAILED:
            raise RuntimeError(receipt.detail or "portal login e-mail failed")
        return DispatchOutcome(command.command_type, receipt.status.value)

    async def _arrange_intake_booking(self, command: OutboxCommand) -> DispatchOutcome:
        if self._intake_booking is None:
            raise UnknownCommandTypeError("intake booking is not wired")
        conversation_id = IntakeConversationId(_uuid(command, "conversation_id"))
        await self._intake_booking.arrange(command.business_id, conversation_id)
        return DispatchOutcome(command.command_type, f"arranged {conversation_id}")

    async def _cancel_intake_booking(self, command: OutboxCommand) -> DispatchOutcome:
        if self._intake_booking_cancel is None:
            raise UnknownCommandTypeError("intake booking cancel is not wired")
        await self._intake_booking_cancel.cancel(command.business_id, _text(command, "event_uri"))
        return DispatchOutcome(command.command_type, "canceled")

    async def _send_owner_email(self, command: OutboxCommand) -> DispatchOutcome:
        assert self._owner_email is not None
        await self._owner_email.send(command.business_id, command.payload)
        return DispatchOutcome(command.command_type, "sent")

    async def _send_intake_email(self, command: OutboxCommand) -> DispatchOutcome:
        if self._intake_email is None:
            raise UnknownCommandTypeError("intake customer e-mail is not wired")
        await self._intake_email.send(command.business_id, command.payload)
        return DispatchOutcome(command.command_type, "sent")

    async def _manual_payment_effect(self, command: OutboxCommand) -> DispatchOutcome:
        if self._manual_payments is None:
            raise UnknownCommandTypeError("manual payments are not wired")
        if command.command_type == CHECKOUT_EXPIRE_COMMAND_TYPE:
            detail = await self._manual_payments.expire_checkout(command.payload)
        elif command.command_type == PLAN_NUDGE_COMMAND_TYPE:
            detail = await self._manual_payments.nudge_plan(command.business_id, command.payload)
        else:
            detail = await self._manual_payments.send_receipt(command.business_id, command.payload)
        return DispatchOutcome(command.command_type, detail)

    async def _send_intake_text(self, command: OutboxCommand) -> DispatchOutcome:
        if self._intake_text is None:
            raise UnknownCommandTypeError("intake customer texting is not wired")
        status = await self._intake_text.send(command.business_id, command.payload)
        return DispatchOutcome(command.command_type, status.value)

    async def _transcribe(self, command: OutboxCommand) -> DispatchOutcome:
        part_id = FieldNotePartId(_uuid(command, "field_note_part_id"))
        case_id = FieldNoteCaseId(_uuid(command, "field_note_case_id"))
        now, stale_before = self._window()
        report = await self._transcription.transcribe(
            command.business_id, part_id, now=now, stale_before=stale_before
        )
        if report.outcome in {
            TranscriptionOutcome.BUSY,
            TranscriptionOutcome.LEASE_LOST,
            TranscriptionOutcome.MISSING,
        }:
            raise TransientCommandError(f"field-note transcription is {report.outcome.value}")
        if report.outcome is TranscriptionOutcome.FAILED:
            raise RuntimeError(f"field-note transcription failed: {report.detail}")
        if report.outcome is TranscriptionOutcome.CEILING_REACHED:
            await self._notify_ceiling(command)
            return DispatchOutcome(command.command_type, report.outcome.value)
        await self._outbox.enqueue(
            field_note_review_command(
                command.business_id,
                case_id,
                FieldNoteReviewTrigger.TRANSCRIPTION,
                str(part_id),
            )
        )
        return DispatchOutcome(command.command_type, report.outcome.value)

    async def _coordinate_review(self, command: OutboxCommand) -> DispatchOutcome:
        case_id = FieldNoteCaseId(_uuid(command, "field_note_case_id"))
        trigger = FieldNoteReviewTrigger(_text(command, "trigger"))
        owner_reply_message_id = (
            MessageId(_uuid(command, "owner_reply_message_id"))
            if "owner_reply_message_id" in command.payload
            else None
        )
        outcome = await self._review.coordinate(
            command.business_id,
            case_id,
            trigger,
            now=self._now(),
            owner_reply_message_id=owner_reply_message_id,
        )
        if outcome.status is ReviewCoordinationStatus.CEILING_REACHED:
            await self._notify_ceiling(command)
        return DispatchOutcome(command.command_type, outcome.status.value)

    async def _notify_ceiling(self, command: OutboxCommand) -> None:
        """The command completes either way; the owner is told once, in the thread."""

        if self._ceiling_notices is not None:
            await self._ceiling_notices.notify_ceiling_reached(command)

    async def _generate_report(self, command: OutboxCommand) -> DispatchOutcome:
        case_id = FieldNoteCaseId(_uuid(command, "field_note_case_id"))
        review_id = FieldNoteReviewId(_uuid(command, "field_note_review_id"))
        completed_at = datetime.fromisoformat(_text(command, "completed_at"))
        snapshot = await self._snapshots.build(
            command.business_id, case_id, review_id, completed_at=completed_at
        )
        now, stale_before = self._window()
        version = await self._reports.generate(snapshot, now=now, stale_before=stale_before)
        await self._report_delivery.deliver(version)
        return DispatchOutcome(command.command_type, f"report version {version.version}")

    async def _publish_report(self, command: OutboxCommand) -> DispatchOutcome:
        report_version_id = _uuid(command, "report_version_id")
        await self._report_publication.publish(command.business_id, report_version_id)
        return DispatchOutcome(command.command_type, f"published {report_version_id}")

    async def _email_report(self, command: OutboxCommand) -> DispatchOutcome:
        report_version_id = _uuid(command, "report_version_id")
        recipient_address = _text(command, "recipient_address")
        await self._report_email.send(
            command.business_id,
            report_version_id,
            recipient_address,
            report_email_idempotency_key(
                report_version_id, recipient_address, _text(command, "request_key")
            ),
        )
        return DispatchOutcome(command.command_type, f"emailed {report_version_id}")

    async def _copy_plan_set(self, command: OutboxCommand) -> DispatchOutcome:
        if self._plan_custody is None:
            raise UnknownCommandTypeError("no object storage adapter is wired for plan custody")
        upload_id = PlanSetUploadId(_uuid(command, "plan_set_upload_id"))
        now, stale_before = self._window()
        report = await self._plan_custody.copy(
            command.business_id, upload_id, now=now, stale_before=stale_before
        )
        if report.outcome in {
            PlanSetCustodyOutcome.BUSY,
            PlanSetCustodyOutcome.LEASE_LOST,
            PlanSetCustodyOutcome.MISSING,
        }:
            raise TransientCommandError(f"plan-set custody is {report.outcome.value}")
        if report.outcome is PlanSetCustodyOutcome.FAILED:
            raise RuntimeError(f"plan-set custody failed: {report.detail}")
        return DispatchOutcome(command.command_type, report.outcome.value)


@dataclass(frozen=True)
class WorkerBatchReport:
    claimed: int
    succeeded: int
    failed: int
    details: tuple[str, ...] = ()


class OutboxWorker:
    """Claims outbox batches and dispatches them outside the ingress request."""

    def __init__(
        self,
        outbox: OutboxService,
        dispatcher: OutboxCommandDispatcher,
        *,
        now: Callable[[], datetime],
        worker_id: str = "outbox-worker",
        batch_size: int = 10,
        lease_ttl: timedelta = timedelta(minutes=5),
        retry_in: timedelta = timedelta(seconds=30),
        failure_notices: NotifyExhaustedCommandService | None = None,
    ) -> None:
        self._outbox = outbox
        self._dispatcher = dispatcher
        self._now = now
        self._failure_notices = failure_notices
        self._worker_id = worker_id
        self._batch_size = batch_size
        self._lease_ttl = lease_ttl
        self._retry_in = retry_in

    async def run_once(self) -> WorkerBatchReport:
        now = self._now()
        records = await self._outbox.claim_batch(
            self._batch_size, now, self._worker_id, stale_before=now - self._lease_ttl
        )
        succeeded = 0
        failed = 0
        details: list[str] = []
        for record in records:
            try:
                outcome = await self._dispatcher.dispatch(record)
            except Exception as error:
                failed += 1
                details.append(f"{record.command.command_type}: {error!r}")
                updated = await self._outbox.mark_failed(
                    record, self._retry_in, repr(error), self._now()
                )
                if updated.status is OutboxStatus.DEAD:
                    _log_record(logging.ERROR, updated, "dead-lettered", error=updated.last_error)
                    await self._notify_exhausted(updated, details)
                else:
                    _log_record(
                        logging.WARNING, updated, "failed, will retry", error=updated.last_error
                    )
                continue
            succeeded += 1
            details.append(f"{outcome.command_type}: {outcome.detail}")
            await self._outbox.mark_succeeded(record)
            _log_record(logging.INFO, record, outcome.detail)
        if records:
            logger.info(
                "batch done worker=%s claimed=%d succeeded=%d failed=%d",
                self._worker_id,
                len(records),
                succeeded,
                failed,
            )
        return WorkerBatchReport(
            claimed=len(records),
            succeeded=succeeded,
            failed=failed,
            details=tuple(details),
        )

    async def _notify_exhausted(self, record: OutboxRecord, details: list[str]) -> None:
        """A notice that cannot be enqueued must not fail the rest of the batch."""

        if self._failure_notices is None:
            return
        try:
            await self._failure_notices.notify(record)
        except Exception as error:
            details.append(f"failure notice for {record.command.command_type}: {error!r}")
            _log_record(logging.ERROR, record, "failure notice not enqueued", error=repr(error))

    async def drain(self, max_batches: int = 50) -> tuple[WorkerBatchReport, ...]:
        reports: list[WorkerBatchReport] = []
        for _ in range(max_batches):
            report = await self.run_once()
            reports.append(report)
            if report.claimed == 0:
                break
        return tuple(reports)


def _log_record(
    level: int, record: OutboxRecord, message: str, *, error: str | None = None
) -> None:
    """One line per command outcome; errors are the adapters' sanitized text."""

    logger.log(
        level,
        "%s command=%s id=%s business=%s attempt=%d/%d%s",
        message,
        record.command.command_type,
        record.command.command_id,
        record.command.business_id,
        record.attempts,
        record.max_attempts,
        f" error={error}" if error else "",
    )
