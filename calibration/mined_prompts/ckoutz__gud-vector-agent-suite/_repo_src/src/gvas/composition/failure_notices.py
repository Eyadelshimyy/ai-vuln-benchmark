"""Owner notification for outbox commands that exhausted their retries.

A dead command is otherwise invisible to the owner: the work simply never
happens. One sanitized notice is enqueued into the conversation the command
belongs to, keyed on the command, so replays reuse the same outbound message.
Only what the owner can act on is said; the provider response, exception and
attempt history stay in the outbox record and the logs.

Owner reply delivery is deliberately excluded. If replies cannot reach the
conversation, a notice about the reply would fail the same way and enqueue
another notice.

The same anchoring serves the monthly cost ceiling: a transcription or review
command held back by the ceiling completes normally and the owner is told once,
in the case thread, that the limit is reached. Only units are ever mentioned,
never money.
"""

from dataclasses import dataclass
from typing import Final, Protocol
from uuid import UUID

from gvas.domain.field_note_repositories import FieldNoteUnitOfWork
from gvas.domain.field_notes import (
    FIELD_NOTE_REVIEW_COMMAND_TYPE,
    FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE,
    FieldNoteCaseId,
)
from gvas.domain.identifiers import BusinessId, ConversationId, MessageId, QuoteId
from gvas.domain.intake import (
    INTAKE_BOOKING_ARRANGE_COMMAND_TYPE,
    INTAKE_BOOKING_CANCEL_COMMAND_TYPE,
    INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
    INTAKE_CUSTOMER_TEXT_COMMAND_TYPE,
    OWNER_NOTICE_EMAIL_COMMAND_TYPE,
)
from gvas.domain.messages import ConversationRef, OutboundOwnerMessage, TextPart
from gvas.domain.outbox import (
    OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
    OWNER_REPLY_COMMAND_TYPE,
    OutboxCommand,
    OutboxRecord,
    owner_reply_command,
)
from gvas.domain.plans import PLAN_SET_COPY_COMMAND_TYPE
from gvas.domain.quotes import QUOTE_DELIVERY_COMMAND_TYPE, QUOTE_TEXT_COMMAND_TYPE, Quote
from gvas.domain.reporting import (
    FIELD_NOTES_REPORT_COMMAND_TYPE,
    FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE,
    FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE,
)
from gvas.domain.repositories import UnitOfWork

# Guidance is per command type because the recoverable path differs, and a
# notice that advertises an unrecoverable one is worse than no notice. An
# approved quote does not re-enqueue its delivery, and a case whose audio never
# transcribed keeps an incomplete canonical transcript, so neither is retryable
# by repeating the message; both need a fresh case in a new thread. Intake,
# review and report work, by contrast, is re-enqueued by the next note in the
# same thread, a publish is re-enqueued by approving again, and a plan-set copy
# is re-enqueued by uploading the file again (a fresh upload is a new file).
# same thread, a publish is re-enqueued by approving again, and a report email
# by asking for it again.
RESEND_IN_THREAD: Final = "Send the message again in this thread."
NEW_NOTE_IN_THREAD: Final = (
    "Add the note again in this thread; that starts the review over. "
    "Send 'close notes' first if you would rather start the case fresh."
)
NEW_QUOTE_IN_NEW_THREAD: Final = (
    "This quote will not send itself again. Start a new quote in a new thread."
)
APPROVE_AGAIN_IN_THREAD: Final = (
    "The report text above is unchanged. Send 'approve report' again in this "
    "thread to retry posting the document."
)
UPLOAD_PLAN_SET_AGAIN_IN_THREAD: Final = (
    "The notes are unaffected. Upload the plan set file again in this thread to retry storing it."
)
SEND_REPORT_AGAIN_IN_THREAD: Final = (
    "The document in this thread is unaffected. Send 'send report to <address>' "
    "again in this thread to retry the email."
)
RESTART_NOTES_IN_NEW_THREAD: Final = (
    "That recording is not part of these notes and the case cannot complete "
    "without it. Send 'close notes' here, then start the notes again in a new "
    "thread and re-upload the recording."
)

CEILING_GUIDANCE: Final[dict[str, str]] = {
    FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE: (
        "This month's transcription limit for your business is reached, so this "
        "voice note was not transcribed and these notes cannot complete with it "
        "pending. The case stays open; when you are ready, send 'close notes' here "
        "and start the notes again in a new thread, typed, or with the recording "
        "again next month."
    ),
    FIELD_NOTE_REVIEW_COMMAND_TYPE: (
        "This month's review limit for your business is reached, so these notes "
        "were not reviewed. The case stays open and the transcript is kept; "
        "add a note in this thread next month to start the review."
    ),
}

FAILURE_GUIDANCE: Final[dict[str, tuple[str, str]]] = {
    OWNER_MESSAGE_PROCESS_COMMAND_TYPE: (
        "That message could not be processed.",
        RESEND_IN_THREAD,
    ),
    QUOTE_DELIVERY_COMMAND_TYPE: (
        "The approved quote could not be emailed to the customer.",
        NEW_QUOTE_IN_NEW_THREAD,
    ),
    # The text notice is composed from the quote (see ``quote_text_failure_notice``);
    # this entry only marks the command type as one that gets a notice.
    QUOTE_TEXT_COMMAND_TYPE: (
        "The quote was created, but the text to the customer could not be sent.",
        "Forward the quote link to the customer yourself.",
    ),
    FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE: (
        "A voice note could not be transcribed.",
        RESTART_NOTES_IN_NEW_THREAD,
    ),
    FIELD_NOTE_REVIEW_COMMAND_TYPE: (
        "These field notes could not be reviewed.",
        NEW_NOTE_IN_THREAD,
    ),
    FIELD_NOTES_REPORT_COMMAND_TYPE: (
        "The field-notes report could not be generated.",
        NEW_NOTE_IN_THREAD,
    ),
    FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE: (
        "The approved report document could not be posted to this thread.",
        APPROVE_AGAIN_IN_THREAD,
    ),
    PLAN_SET_COPY_COMMAND_TYPE: (
        "A plan set uploaded to this thread could not be stored.",
        UPLOAD_PLAN_SET_AGAIN_IN_THREAD,
    ),
    FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE: (
        "The published report could not be emailed to the requested address.",
        SEND_REPORT_AGAIN_IN_THREAD,
    ),
    INTAKE_BOOKING_ARRANGE_COMMAND_TYPE: (
        "An approved website booking could not be arranged with the calendar.",
        "Schedule the customer yourself; the request stays approved.",
    ),
    INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE: (
        "An email to a website-booking customer could not be sent.",
        "Email the customer yourself.",
    ),
    INTAKE_CUSTOMER_TEXT_COMMAND_TYPE: (
        "A text to a website-booking customer could not be sent.",
        "Text the customer yourself.",
    ),
    OWNER_NOTICE_EMAIL_COMMAND_TYPE: (
        "The e-mail copy of a website notice could not be sent to the notification inbox.",
        "The notice above in this channel is still the one to act on.",
    ),
    INTAKE_BOOKING_CANCEL_COMMAND_TYPE: (
        "The calendar event on a declined website booking could not be canceled.",
        "The event may still be on the calendar — cancel it in Calendly yourself.",
    ),
}


INTAKE_COMMAND_TYPES: Final = frozenset(
    {
        INTAKE_BOOKING_ARRANGE_COMMAND_TYPE,
        INTAKE_BOOKING_CANCEL_COMMAND_TYPE,
        INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
        INTAKE_CUSTOMER_TEXT_COMMAND_TYPE,
        OWNER_NOTICE_EMAIL_COMMAND_TYPE,
    }
)


@dataclass(frozen=True)
class ConversationAnchor:
    business_id: BusinessId
    conversation_id: ConversationId
    conversation_ref: ConversationRef
    inbound_message_id: MessageId


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class MessageUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class NotifyExhaustedCommandService:
    """Enqueues one sanitized owner notice per dead command."""

    def __init__(
        self,
        message_unit_of_work_factory: MessageUnitOfWorkFactory,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
    ) -> None:
        self._messages = message_unit_of_work_factory
        self._field_notes = field_note_unit_of_work_factory

    async def notify(self, record: OutboxRecord) -> MessageId | None:
        command = record.command
        guidance = FAILURE_GUIDANCE.get(command.command_type)
        if guidance is None or command.command_type == OWNER_REPLY_COMMAND_TYPE:
            return None
        summary, recovery = guidance
        text = f"{summary}\n{recovery}"
        if command.command_type == QUOTE_TEXT_COMMAND_TYPE:
            quote = await self._quote(command)
            if quote is not None:
                text = quote_text_failure_notice(quote)
        return await self._post(command, text, "failure_notice")

    async def notify_ceiling_reached(self, command: OutboxCommand) -> MessageId | None:
        """One notice per held-back command; replays reuse the same outbound message."""

        text = CEILING_GUIDANCE.get(command.command_type)
        if text is None:
            return None
        return await self._post(command, text, "ceiling_notice")

    async def _post(self, command: OutboxCommand, text: str, prefix: str) -> MessageId | None:
        anchor = await self._anchor(command)
        if anchor is None:
            return None
        message = OutboundOwnerMessage(
            business_id=anchor.business_id,
            conversation_ref=anchor.conversation_ref,
            parts=(TextPart(text=text),),
            correlation_id=f"{prefix}:{command.command_id}",
        )
        async with self._messages() as unit_of_work:
            outbound_message_id = await unit_of_work.outbound_messages.create(
                message, anchor.conversation_id, anchor.inbound_message_id
            )
            await unit_of_work.outbox.enqueue(
                owner_reply_command(anchor.business_id, outbound_message_id)
            )
            await unit_of_work.commit()
        return outbound_message_id

    async def _anchor(self, command: OutboxCommand) -> ConversationAnchor | None:
        if command.command_type == OWNER_MESSAGE_PROCESS_COMMAND_TYPE:
            return await self._inbound_anchor(command)
        if command.command_type in {QUOTE_DELIVERY_COMMAND_TYPE, QUOTE_TEXT_COMMAND_TYPE}:
            return await self._quote_anchor(command)
        if command.command_type in INTAKE_COMMAND_TYPES:
            return await self._intake_anchor(command)
        return await self._field_note_anchor(command)

    async def _quote(self, command: OutboxCommand) -> Quote | None:
        quote_id = _uuid(command, "quote_id")
        if quote_id is None:
            return None
        async with self._messages() as unit_of_work:
            quote = await unit_of_work.quotes.get(command.business_id, QuoteId(quote_id))
            await unit_of_work.commit()
        return quote

    async def _inbound_anchor(self, command: OutboxCommand) -> ConversationAnchor | None:
        inbound_message_id = command.inbound_message_id
        if inbound_message_id is None:
            return None
        async with self._messages() as unit_of_work:
            record = await unit_of_work.inbound_messages.get_for_processing(inbound_message_id)
            await unit_of_work.commit()
        if record is None:
            return None
        return ConversationAnchor(
            business_id=record.business_id,
            conversation_id=record.conversation_id,
            conversation_ref=record.message.conversation_ref,
            inbound_message_id=record.inbound_message_id,
        )

    async def _quote_anchor(self, command: OutboxCommand) -> ConversationAnchor | None:
        quote_id = _uuid(command, "quote_id")
        if quote_id is None:
            return None
        async with self._messages() as unit_of_work:
            quote = await unit_of_work.quotes.get(command.business_id, QuoteId(quote_id))
            if quote is None:
                await unit_of_work.commit()
                return None
            source = await unit_of_work.inbound_messages.find_by_key(
                quote.business_id, quote.conversation_id, quote.source_message_key
            )
            await unit_of_work.commit()
        if source is None:
            return None
        return ConversationAnchor(
            business_id=quote.business_id,
            conversation_id=quote.conversation_id,
            conversation_ref=quote.conversation_ref,
            inbound_message_id=source.inbound_message_id,
        )

    async def _intake_anchor(self, command: OutboxCommand) -> ConversationAnchor | None:
        """Intake commands carry no owner conversation; the notice goes to the
        business's most recent owner thread, like the booking notices."""

        async with self._messages() as unit_of_work:
            record = await unit_of_work.inbound_messages.find_latest_for_business(
                command.business_id
            )
            await unit_of_work.commit()
        if record is None:
            return None
        return ConversationAnchor(
            business_id=record.business_id,
            conversation_id=record.conversation_id,
            conversation_ref=record.message.conversation_ref,
            inbound_message_id=record.inbound_message_id,
        )

    async def _field_note_anchor(self, command: OutboxCommand) -> ConversationAnchor | None:
        case_id = _uuid(command, "field_note_case_id")
        if case_id is None:
            return None
        async with self._field_notes() as unit_of_work:
            case = await unit_of_work.field_note_cases.get(
                command.business_id, FieldNoteCaseId(case_id)
            )
            await unit_of_work.commit()
        if case is None:
            return None
        return ConversationAnchor(
            business_id=case.business_id,
            conversation_id=case.conversation_id,
            conversation_ref=case.conversation_ref,
            inbound_message_id=case.origin_inbound_message_id,
        )


def quote_text_failure_notice(quote: Quote) -> str:
    """The quote exists and may already have been emailed; the owner needs the
    link and whether the customer heard anything at all."""

    summary, recovery = FAILURE_GUIDANCE[QUOTE_TEXT_COMMAND_TYPE]
    receipt = quote.delivery_receipt
    recipient = quote.draft.recipient if quote.draft is not None else None
    lines = [summary]
    email = recipient.email_address if recipient is not None else None
    if receipt is not None and receipt.emailed and email:
        lines.append(f"The customer was emailed the link at {email}.")
    else:
        lines.append("The customer was not emailed either, so nothing has reached them.")
    if receipt is not None and receipt.customer_link:
        lines.append(f"{recovery} {receipt.customer_link}")
    else:
        lines.append(recovery)
    return "\n".join(lines)


def _uuid(command: OutboxCommand, key: str) -> UUID | None:
    value = command.payload.get(key)
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None
