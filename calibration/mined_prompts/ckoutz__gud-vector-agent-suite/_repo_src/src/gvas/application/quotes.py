import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from gvas.domain.appointments import (
    Appointment,
    AppointmentLookupError,
    AppointmentLookupPort,
    surrounding_days_window,
)
from gvas.domain.customer_linking import link_quote_customer
from gvas.domain.enums import (
    BillingInterval,
    DeliveryStatus,
    QuoteSendAction,
    QuoteStatus,
    RecipientAddressKind,
    WorkflowRunStatus,
)
from gvas.domain.identifiers import BusinessId, ConversationId, MessageKey, QuoteId
from gvas.domain.intents import IntentResolution
from gvas.domain.messages import (
    ConversationRef,
    CustomerDeliveryRequest,
    CustomerRecipient,
    CustomerTextRequest,
    DeliveryReceipt,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.money import format_money
from gvas.domain.outbox import owner_reply_command
from gvas.domain.ports import (
    CustomerQuoteDeliveryPort,
    CustomerTextDeliveryPort,
    QuoteDraftingPort,
)
from gvas.domain.quotes import (
    QUOTE_INTENT,
    OwnerApprovalRequiredPolicy,
    Quote,
    QuoteAppointmentContext,
    QuoteConcurrencyError,
    QuoteDraftProposal,
    QuoteDraftRejectedError,
    QuoteDraftRequest,
    QuoteSendAssessment,
    QuoteSendPolicy,
    customer_quote_text,
    delivery_line_items,
    has_customer_line,
    new_claim_token,
    new_quote,
    quote_delivery_command,
    quote_text_command,
    quote_trigger_request_text,
    requested_customer_name,
)
from gvas.domain.repositories import BusinessRecord, UnitOfWork
from gvas.domain.workflows import WorkflowContext, WorkflowResult

logger = logging.getLogger(__name__)

FREE_TEXT_DRAFT_NOTICE = "Drafted from your message — check items before approving."
BILLING_LABELS = {BillingInterval.MONTH: "monthly", BillingInterval.YEAR: "yearly"}
CUSTOMER_LOOKUP_UNAVAILABLE = (
    "The appointment calendar could not be reached. "
    "Please send the quote again with a customer: <email> line this time."
)


class QuoteIntakeError(ValueError):
    pass


class QuoteDeliveryError(RuntimeError):
    pass


class QuoteTextError(RuntimeError):
    """The customer text did not go; the command stays retryable."""


class QuoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class QuoteIntentSelector:
    def select(
        self,
        message: NormalizedOwnerMessage,
        conversation_id: ConversationId,
        active_quote: Quote | None,
    ) -> IntentResolution | None:
        if active_quote is not None:
            if (
                active_quote.business_id != message.business_id
                or active_quote.conversation_id != conversation_id
                or not active_quote.is_active
            ):
                raise ValueError("active quote does not match the normalized conversation")
            return IntentResolution(intent=QUOTE_INTENT, confidence=1)
        if quote_trigger_request_text(normalized_text(message)) is not None:
            quote_request_text(message)
            return IntentResolution(intent=QUOTE_INTENT, confidence=1)
        return None


class QuoteWorkflowHandler:
    intent = QUOTE_INTENT

    def __init__(
        self,
        unit_of_work_factory: QuoteUnitOfWorkFactory,
        drafting_port: QuoteDraftingPort,
        send_policy: QuoteSendPolicy | None = None,
        appointment_lookup: AppointmentLookupPort | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._drafting_port = drafting_port
        self._send_policy = send_policy or OwnerApprovalRequiredPolicy()
        self._appointment_lookup = appointment_lookup

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        conversation_id = context.conversation_id
        if conversation_id is None:
            raise ValueError("quote workflow requires a persisted conversation identity")
        message = context.message
        quote = await self._get_by_message(message, conversation_id)
        if quote is not None:
            if quote.status is QuoteStatus.DRAFTING:
                return await self._complete_draft(quote, message)
            if quote.status is QuoteStatus.AWAITING_CUSTOMER_SELECTION:
                return _customer_selection_reply(quote, message.message_key)
            if quote.status is QuoteStatus.AWAITING_APPROVAL and quote.draft is not None:
                return _workflow_reply(
                    quote,
                    message.message_key,
                    detail=self._send_decision_detail(quote.draft),
                )
            return _workflow_reply(quote, message.message_key)

        quote = await self._get_active(message.business_id, conversation_id)
        if quote is None:
            if owner_command(message) is not None:
                return _no_awaiting_approval_reply(message)
            request_text = quote_request_text(message)
            quote = new_quote(
                business_id=message.business_id,
                conversation_id=conversation_id,
                conversation_ref=message.conversation_ref,
                message_key=message.message_key,
                request_text=request_text,
                now=message.received_at,
            )
            async with self._unit_of_work_factory() as unit_of_work:
                await unit_of_work.quotes.add(quote)
                await unit_of_work.commit()
            return await self._complete_draft(quote, message)

        if quote.status is QuoteStatus.DRAFTING:
            return WorkflowResult(
                status=WorkflowRunStatus.SUCCEEDED,
                replies=(
                    _owner_reply(
                        quote,
                        message.message_key,
                        "A quote draft is already in progress.",
                    ),
                ),
            )
        if quote.status is QuoteStatus.AWAITING_CUSTOMER_SELECTION:
            return await self._handle_customer_selection(quote, message)
        return await self._handle_owner_reply(quote, message)

    async def _handle_customer_selection(
        self, quote: Quote, message: NormalizedOwnerMessage
    ) -> WorkflowResult:
        text = normalized_text(message).strip()
        if text.casefold() == "reject":
            return await self._abandon_draft(quote, message, "Quote cancelled.")
        candidates = quote.customer_candidates or ()
        choice = text.rstrip(".")
        if not choice.isdigit() or not 1 <= int(choice) <= len(candidates):
            return _customer_selection_reply(
                quote,
                message.message_key,
                prefix=f"Reply with a number from 1 to {len(candidates)}, or reject.",
            )
        drafting = quote.select_customer(int(choice), message.message_key, message.received_at)
        await self._save(drafting, expected_version=quote.version)
        return await self._complete_draft(drafting, message)

    async def _handle_owner_reply(
        self, quote: Quote, message: NormalizedOwnerMessage
    ) -> WorkflowResult:
        text = normalized_text(message).strip()
        command = owner_command(message)
        if command is not None and quote.status is not QuoteStatus.AWAITING_APPROVAL:
            return _no_awaiting_approval_reply(message)
        if command == "approve":
            approved = quote.approve(message.message_key, message.received_at)
            async with self._unit_of_work_factory() as unit_of_work:
                approved, _ = await link_quote_customer(unit_of_work, approved, message.received_at)
                await unit_of_work.quotes.save(approved, expected_version=quote.version)
                await unit_of_work.outbox.enqueue(quote_delivery_command(approved))
                await unit_of_work.commit()
            return _workflow_reply(approved, message.message_key)
        if command == "reject":
            rejected = quote.reject(message.message_key, message.received_at)
            await self._save(rejected, expected_version=quote.version)
            return _workflow_reply(rejected, message.message_key)
        if command == "correct":
            correction = text.split(":", 1)[1].strip()
            if not correction:
                raise QuoteIntakeError("correction text must not be empty")
            drafting = quote.begin_correction(
                message.message_key,
                correction,
                message.received_at,
            )
            await self._save(drafting, expected_version=quote.version)
            return await self._complete_draft(drafting, message)
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(
                _owner_reply(
                    quote,
                    message.message_key,
                    "Reply with approve, reject, or correct: <changes>.",
                ),
            ),
        )

    async def _complete_draft(
        self, quote: Quote, message: NormalizedOwnerMessage
    ) -> WorkflowResult:
        if (
            self._appointment_lookup is not None
            and quote.customer_appointment is None
            and not has_customer_line(quote.pending_request_text)
        ):
            try:
                candidates = await self._find_appointments(quote, message)
            except AppointmentLookupError as error:
                logger.warning("appointment lookup failed for quote %s: %s", quote.quote_id, error)
                return await self._abandon_draft(quote, message, CUSTOMER_LOOKUP_UNAVAILABLE)
            if len(candidates) == 1:
                quote = quote.with_customer_appointment(candidates[0])
            elif len(candidates) > 1:
                awaiting = quote.await_customer_selection(
                    candidates, message.message_key, message.received_at
                )
                await self._save(awaiting, expected_version=quote.version)
                return _customer_selection_reply(awaiting, message.message_key)
        appointment = quote.customer_appointment
        request = QuoteDraftRequest(
            quote_id=quote.quote_id,
            business_id=quote.business_id,
            conversation_id=quote.conversation_id,
            request_text=quote.pending_request_text,
            revision=quote.revision,
            idempotency_key=f"quote-draft:{quote.quote_id}:{quote.revision}",
            recipient=(None if appointment is None else _appointment_recipient(appointment)),
            appointment=(None if appointment is None else _appointment_context(appointment)),
        )
        try:
            proposal = await self._drafting_port.draft(request)
        except QuoteDraftRejectedError as error:
            return await self._abandon_draft(quote, message, str(error))
        drafted = quote.apply_draft(proposal, message.message_key, message.received_at)
        detail = self._send_decision_detail(proposal)
        try:
            await self._save(drafted, expected_version=quote.version)
        except QuoteConcurrencyError:
            current = await self._get_by_message(message, quote.conversation_id)
            if current is None:
                raise
            drafted = current
        return _workflow_reply(drafted, message.message_key, detail=detail)

    async def _abandon_draft(
        self, quote: Quote, message: NormalizedOwnerMessage, reason: str
    ) -> WorkflowResult:
        abandoned = quote.abandon_draft(message.message_key, message.received_at)
        await self._save(abandoned, expected_version=quote.version)
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(
                _owner_reply(
                    abandoned,
                    message.message_key,
                    f"{reason}\nSend the corrected request as a new quote.",
                ),
            ),
        )

    async def _find_appointments(
        self, quote: Quote, message: NormalizedOwnerMessage
    ) -> tuple[Appointment, ...]:
        if self._appointment_lookup is None:
            return ()
        window = surrounding_days_window(quote.business_id, message.received_at)
        found = await self._appointment_lookup.find(window)
        name_filter = requested_customer_name(quote.pending_request_text)
        if name_filter is not None:
            needle = name_filter.casefold()
            found = tuple(
                appointment
                for appointment in found
                if needle in appointment.invitee_name.casefold()
            )
        return tuple(sorted(found, key=lambda appointment: appointment.start_time))

    def _send_decision_detail(self, proposal: QuoteDraftProposal) -> str | None:
        decision = self._send_policy.decide(
            QuoteSendAssessment(
                confidence=proposal.confidence,
                risk_flags=proposal.risk_flags,
                suspected_mistake=bool(proposal.risk_flags),
            )
        )
        detail = decision.detail
        if decision.action is QuoteSendAction.AUTO_SEND:
            detail = "automatic sending is configured but disabled until a later phase"
        return detail

    async def _get_active(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> Quote | None:
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await unit_of_work.quotes.get_active(business_id, conversation_id)
            await unit_of_work.commit()
        return quote

    async def _get_by_message(
        self, message: NormalizedOwnerMessage, conversation_id: ConversationId
    ) -> Quote | None:
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await unit_of_work.quotes.get_by_message(
                message.business_id, conversation_id, message.message_key
            )
            await unit_of_work.commit()
        return quote

    async def _save(self, quote: Quote, *, expected_version: int) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.quotes.save(quote, expected_version=expected_version)
            await unit_of_work.commit()


class QuoteDeliveryStatus(StrEnum):
    COMPLETED = "completed"
    ALREADY_COMPLETED = "already_completed"
    NOT_APPROVED = "not_approved"
    MISSING = "missing"


@dataclass(frozen=True)
class QuoteDeliveryOutcome:
    status: QuoteDeliveryStatus
    quote_id: QuoteId


class DeliverApprovedQuoteService:
    """Hands the approved draft to the customer delivery port, then tells the
    owner what happened.

    When the receipt carries a ``customer_link`` and the recipient has a phone
    number, and a text port is wired, a follow-up command texts the link; its
    failure never undoes the delivery. The owner confirmation is written in
    the same transaction as the delivered quote, so a replay neither repeats
    the delivery nor the confirmation.
    """

    def __init__(
        self,
        unit_of_work_factory: QuoteUnitOfWorkFactory,
        delivery_port: CustomerQuoteDeliveryPort,
        text_port: CustomerTextDeliveryPort | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._delivery_port = delivery_port
        self._texts_customers = text_port is not None

    async def deliver(self, business_id: BusinessId, quote_id: QuoteId) -> QuoteDeliveryOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await unit_of_work.quotes.get(business_id, quote_id)
            await unit_of_work.commit()
        if quote is None:
            return QuoteDeliveryOutcome(QuoteDeliveryStatus.MISSING, quote_id)
        if quote.status in {QuoteStatus.DELIVERY_PENDING, QuoteStatus.DELIVERED}:
            return QuoteDeliveryOutcome(QuoteDeliveryStatus.ALREADY_COMPLETED, quote_id)
        if quote.status is not QuoteStatus.APPROVED or quote.draft is None:
            return QuoteDeliveryOutcome(QuoteDeliveryStatus.NOT_APPROVED, quote_id)

        business = await self._business(quote.business_id)
        if business is not None and business.site_url and quote.claim_token is None:
            # Quotes approved before claim tokens existed get one on delivery.
            quote = await self._issue_claim_token(quote)
        if quote.customer_id is None:
            quote = await self._link_customer(quote)
        quote_url = None
        business_name = None
        if business is not None and business.site_url:
            quote_url = quote.quote_url(business.site_url)
            business_name = business.display_name or business.name
        subject = f"Your quote from {business_name}" if business_name is not None else "Your quote"
        draft = quote.draft
        if draft is None:
            return QuoteDeliveryOutcome(QuoteDeliveryStatus.NOT_APPROVED, quote_id)
        receipt = await self._delivery_port.deliver(
            CustomerDeliveryRequest(
                business_id=quote.business_id,
                recipient=draft.recipient,
                idempotency_key=f"quote-delivery:{quote.quote_id}",
                subject=subject,
                body_text=_customer_quote_body(draft),
                links=tuple(link.reference for link in draft.hosted_links),
                line_items=delivery_line_items(draft),
                currency=draft.currency,
                quote_url=quote_url,
                business_name=business_name,
                note=draft.owner_note,
            )
        )
        if receipt.status is DeliveryStatus.FAILED:
            raise QuoteDeliveryError(receipt.detail or "quote delivery failed")
        delivered = quote.record_delivery(receipt)
        try:
            async with self._unit_of_work_factory() as unit_of_work:
                await unit_of_work.quotes.save(delivered, expected_version=quote.version)
                consented = await customer_consented_to_texts(unit_of_work, delivered)
                texting = self._will_text(delivered) and consented
                if texting:
                    await unit_of_work.outbox.enqueue(quote_text_command(delivered))
                await self._confirm_to_owner(
                    unit_of_work, delivered, texting=texting, consented=consented
                )
                await unit_of_work.commit()
        except QuoteConcurrencyError:
            async with self._unit_of_work_factory() as unit_of_work:
                current = await unit_of_work.quotes.get(business_id, quote_id)
                await unit_of_work.commit()
            if current is None or current.status not in {
                QuoteStatus.DELIVERY_PENDING,
                QuoteStatus.DELIVERED,
            }:
                raise
        return QuoteDeliveryOutcome(QuoteDeliveryStatus.COMPLETED, quote_id)

    async def _business(self, business_id: BusinessId) -> BusinessRecord | None:
        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get(business_id)
            await unit_of_work.commit()
        return business

    async def _issue_claim_token(self, quote: Quote) -> Quote:
        issued = quote.issue_claim_token(new_claim_token(), quote.updated_at)
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.quotes.save(issued, expected_version=quote.version)
            await unit_of_work.commit()
        return issued

    async def _link_customer(self, quote: Quote) -> Quote:
        async with self._unit_of_work_factory() as unit_of_work:
            linked, _ = await link_quote_customer(unit_of_work, quote, quote.updated_at)
            if linked is not quote:
                await unit_of_work.quotes.save(linked, expected_version=quote.version)
            await unit_of_work.commit()
        return linked

    def _will_text(self, quote: Quote) -> bool:
        if not self._texts_customers or quote.draft is None or quote.delivery_receipt is None:
            return False
        return (
            quote.delivery_receipt.customer_link is not None
            and quote.draft.recipient.phone_number is not None
        )

    async def _confirm_to_owner(
        self, unit_of_work: UnitOfWork, quote: Quote, *, texting: bool, consented: bool
    ) -> None:
        """Anchored on the message that started the quote; without one (nothing
        was ingested through a channel) there is nowhere to reply."""

        source = await unit_of_work.inbound_messages.find_by_key(
            quote.business_id, quote.conversation_id, quote.source_message_key
        )
        if source is None or quote.draft is None or quote.delivery_receipt is None:
            return
        if quote.delivery_receipt.customer_link is None:
            return
        message = OutboundOwnerMessage(
            business_id=quote.business_id,
            conversation_ref=quote.conversation_ref,
            parts=(
                TextPart(
                    text=quote_delivered_reply(
                        quote.draft, quote.delivery_receipt, texting=texting, consented=consented
                    )
                ),
            ),
            correlation_id=f"quote:{quote.quote_id}:delivered",
        )
        outbound_message_id = await unit_of_work.outbound_messages.create(
            message, quote.conversation_id, source.inbound_message_id
        )
        await unit_of_work.outbox.enqueue(
            owner_reply_command(quote.business_id, outbound_message_id)
        )


class SiteAwareQuoteDelivery:
    """Picks the delivery backend per request.

    A request carrying a hosted ``quote_url`` goes to the default adapter,
    which renders the hosted link into the customer email. A request without
    one goes to the external portal when it is wired; with neither in play the
    default adapter's generic email is sent.
    """

    def __init__(
        self,
        default: CustomerQuoteDeliveryPort,
        portal: CustomerQuoteDeliveryPort | None = None,
    ) -> None:
        self._default = default
        self._portal = portal

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        if request.quote_url is None and self._portal is not None:
            return await self._portal.deliver(request)
        return await self._default.deliver(request)


class QuoteTextStatus(StrEnum):
    SENT = "sent"
    NOTHING_TO_TEXT = "nothing_to_text"
    NO_CONSENT = "no_consent"
    NOT_DELIVERED = "not_delivered"
    MISSING = "missing"


@dataclass(frozen=True)
class QuoteTextOutcome:
    status: QuoteTextStatus
    quote_id: QuoteId


class TextDeliveredQuoteService:
    """Texts the customer the link a completed delivery produced.

    Runs as its own command after delivery so that a text that keeps failing
    dead-letters on its own, with the quote already delivered and the owner
    told; the text port keys the send on the quote so a retry cannot text twice.
    """

    def __init__(
        self,
        unit_of_work_factory: QuoteUnitOfWorkFactory,
        text_port: CustomerTextDeliveryPort,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._text_port = text_port

    async def text(self, business_id: BusinessId, quote_id: QuoteId) -> QuoteTextOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await unit_of_work.quotes.get(business_id, quote_id)
            consented = quote is not None and await customer_consented_to_texts(unit_of_work, quote)
            await unit_of_work.commit()
        if quote is None:
            return QuoteTextOutcome(QuoteTextStatus.MISSING, quote_id)
        if quote.status not in {QuoteStatus.DELIVERY_PENDING, QuoteStatus.DELIVERED}:
            return QuoteTextOutcome(QuoteTextStatus.NOT_DELIVERED, quote_id)
        receipt = quote.delivery_receipt
        draft = quote.draft
        if draft is None or receipt is None or receipt.customer_link is None:
            return QuoteTextOutcome(QuoteTextStatus.NOTHING_TO_TEXT, quote_id)
        phone_number = draft.recipient.phone_number
        if phone_number is None:
            return QuoteTextOutcome(QuoteTextStatus.NOTHING_TO_TEXT, quote_id)
        if not consented:
            return QuoteTextOutcome(QuoteTextStatus.NO_CONSENT, quote_id)
        sent = await self._text_port.send_text(
            CustomerTextRequest(
                business_id=quote.business_id,
                phone_number=phone_number,
                text=customer_quote_text(draft, receipt.customer_link),
                idempotency_key=f"quote-text:{quote.quote_id}",
            )
        )
        if sent.status is DeliveryStatus.FAILED:
            raise QuoteTextError(sent.detail or "customer text failed")
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.quotes.record_texted(
                quote.business_id, quote.quote_id, sent.occurred_at
            )
            await unit_of_work.commit()
        return QuoteTextOutcome(QuoteTextStatus.SENT, quote_id)


async def customer_consented_to_texts(unit_of_work: UnitOfWork, quote: Quote) -> bool:
    """Only a linked customer who said yes to texts may be texted."""

    if quote.customer_id is None:
        return False
    customer = await unit_of_work.customers.get(quote.business_id, quote.customer_id)
    return customer is not None and customer.sms_consent is True


def normalized_text(message: NormalizedOwnerMessage) -> str:
    return "\n".join(part.text for part in message.parts if isinstance(part, TextPart))


def owner_command(message: NormalizedOwnerMessage) -> str | None:
    text = normalized_text(message).strip().casefold()
    if text in {"approve", "reject"}:
        return text
    if text.startswith("correct:"):
        return "correct"
    return None


def quote_request_text(message: NormalizedOwnerMessage) -> str:
    request_text = quote_trigger_request_text(normalized_text(message))
    if request_text is None:
        raise QuoteIntakeError("new quote workflows require an explicit quote: prefix")
    if not request_text:
        raise QuoteIntakeError("quote request text must not be empty")
    return request_text


def _no_awaiting_approval_reply(message: NormalizedOwnerMessage) -> WorkflowResult:
    reply = OutboundOwnerMessage(
        business_id=message.business_id,
        conversation_ref=message.conversation_ref,
        parts=(TextPart(text="No quote is awaiting approval in this conversation."),),
        correlation_id=f"quote:message:{message.message_key}",
    )
    return WorkflowResult(
        status=WorkflowRunStatus.SUCCEEDED,
        replies=(reply,),
    )


def _appointment_recipient(appointment: Appointment) -> CustomerRecipient:
    return CustomerRecipient(
        address=appointment.invitee_email,
        address_kind=RecipientAddressKind.EMAIL,
        display_name=appointment.invitee_name,
        phone=appointment.invitee_phone,
        service_address=appointment.address,
    )


def _appointment_context(appointment: Appointment) -> QuoteAppointmentContext:
    return QuoteAppointmentContext(
        event_name=appointment.event_name,
        start_time=appointment.start_time,
        address=appointment.address,
        invitee_name=appointment.invitee_name,
        notes=appointment.notes,
    )


def _customer_selection_reply(
    quote: Quote, message_key: MessageKey, prefix: str | None = None
) -> WorkflowResult:
    candidates = quote.customer_candidates or ()
    listing = "  ".join(
        f"{index}. {candidate.choice_label}" for index, candidate in enumerate(candidates, 1)
    )
    body = f"Which appointment is this quote for? {listing} — reply with the number"
    if prefix:
        body = f"{prefix}\n{body}"
    return WorkflowResult(
        status=WorkflowRunStatus.SUCCEEDED,
        replies=(_owner_reply(quote, message_key, body),),
    )


def _workflow_reply(
    quote: Quote, message_key: MessageKey, detail: str | None = None
) -> WorkflowResult:
    if quote.status is QuoteStatus.AWAITING_APPROVAL:
        if quote.draft is None or quote.approval_correlation_id is None:
            raise ValueError("approval reply requires a complete quote draft")
        body = _owner_quote_body(quote)
        if detail:
            body = f"{body}\n\n{detail}"
        reply = OutboundOwnerMessage(
            business_id=quote.business_id,
            conversation_ref=_conversation_ref(quote),
            parts=(TextPart(text=body),),
            correlation_id=quote.approval_correlation_id,
        )
    elif quote.status is QuoteStatus.APPROVED:
        reply = _owner_reply(quote, message_key, "Quote approved and queued for customer delivery.")
    elif quote.status is QuoteStatus.REJECTED:
        reply = _owner_reply(quote, message_key, "Quote rejected.")
    elif quote.status in {QuoteStatus.DELIVERY_PENDING, QuoteStatus.DELIVERED}:
        reply = _owner_reply(quote, message_key, "Quote delivery is complete.")
    else:
        reply = _owner_reply(quote, message_key, "Quote drafting is in progress.")
    return WorkflowResult(
        status=WorkflowRunStatus.SUCCEEDED,
        replies=(reply,),
    )


def _owner_reply(quote: Quote, message_key: MessageKey, text: str) -> OutboundOwnerMessage:
    return OutboundOwnerMessage(
        business_id=quote.business_id,
        conversation_ref=_conversation_ref(quote),
        parts=(TextPart(text=text),),
        correlation_id=f"quote:{quote.quote_id}:message:{message_key}",
    )


def _conversation_ref(quote: Quote) -> ConversationRef:
    return quote.conversation_ref


def _owner_quote_body(quote: Quote) -> str:
    draft = quote.draft
    if draft is None:
        raise ValueError("quote draft is missing")
    lines: list[str] = []
    appointment = quote.customer_appointment
    if appointment is not None and draft.recipient.address == appointment.invitee_email:
        lines.append(
            f"Customer: {appointment.invitee_name} ({appointment.invitee_email})"
            f" — {appointment.summary}"
        )
    lines.extend(
        f"{item.quantity} × {item.description}: {format_money(item.total_minor, draft.currency)}"
        for item in draft.line_items
    )
    lines.append(f"Total: {format_money(draft.total_minor, draft.currency)}")
    if draft.interval is not None:
        lines.append(f"Billing: {BILLING_LABELS[draft.interval]}")
    if draft.owner_note:
        lines.append(f"Note: {draft.owner_note}")
    if draft.drafted_from_free_text:
        lines.append(FREE_TEXT_DRAFT_NOTICE)
    lines.append("Reply with approve, reject, or correct: <changes>.")
    return "\n".join(lines)


def quote_delivered_reply(
    draft: QuoteDraftProposal,
    receipt: DeliveryReceipt,
    *,
    texting: bool,
    consented: bool = True,
) -> str:
    """What the owner is told once the customer holds a hosted quote: the
    link, and which channels carried it."""

    recipient = draft.recipient
    customer = recipient.display_name or recipient.address
    lines = [f"Quote for {customer} is ready: {receipt.customer_link}"]
    email = recipient.email_address if receipt.emailed else None
    phone = recipient.phone_number if texting else None
    if email and phone:
        lines.append(f"Emailed to {email}; texting {phone}.")
    elif email and recipient.phone_number and not consented:
        lines.append(f"Emailed to {email}. No SMS consent on file, so no text.")
    elif email:
        lines.append(f"Emailed to {email}. No phone on file, so no text.")
    elif phone:
        lines.append(f"Texting {phone}. No email was sent.")
    else:
        lines.append("Not emailed or texted; forward the link to the customer yourself.")
    return "\n".join(lines)


def _customer_quote_body(draft: QuoteDraftProposal) -> str:
    lines = [
        f"{item.quantity} × {item.description}: {format_money(item.total_minor, draft.currency)}"
        for item in draft.line_items
    ]
    lines.append(f"Total: {format_money(draft.total_minor, draft.currency)}")
    if draft.interval is not None:
        lines.append(f"Billed {BILLING_LABELS[draft.interval]}")
    if draft.owner_note:
        lines.append(draft.owner_note)
    return "\n".join(lines)
