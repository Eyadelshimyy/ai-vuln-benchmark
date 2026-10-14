"""Website booking intake: start a chat, collect the request, offer real
availability and route the pick to the owner.

The hard rule: nothing is booked until the owner replies ``approve booking
<ref>``. The service collects, proposes, and notifies — the booking itself is
an outbox command so provider calls stay retryable and never run inside the
request that saved the approval.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from enum import StrEnum
from uuid import UUID, uuid4

from gvas.domain.customer_linking import (
    enqueue_intake_owner_notice,
)
from gvas.domain.customers import CustomerRecord, ServiceRequest
from gvas.domain.enums import DeliveryStatus, RecipientAddressKind, WorkflowRunStatus
from gvas.domain.identifiers import (
    BusinessId,
    CustomerId,
    IntakeConversationId,
    IntakeMessageId,
    ServiceRequestId,
)
from gvas.domain.intake import (
    BOOKING_DECISION_PATH,
    BOOKING_INTENT,
    INTAKE_CHANNEL_PORTAL,
    INTAKE_CHANNEL_WEB,
    INTAKE_CONVERSATION_TTL,
    INTAKE_MAX_USER_MESSAGES,
    INTAKE_MESSAGE_MAX_CHARS,
    SERVICE_REQUEST_SOURCE_INTAKE,
    VISIT_NOTE_LABEL,
    AvailabilityError,
    AvailableSlot,
    BookingDecision,
    BookingDecisionAction,
    BookingDecisionLink,
    BookingEventKind,
    BookingKind,
    BookingRequest,
    ExistingBooking,
    ExistingBookingStatus,
    ExpiredDecisionLinkError,
    IntakeAgentError,
    IntakeBookingEvent,
    IntakeCollected,
    IntakeConversation,
    IntakeCustomerEmail,
    IntakeMessage,
    IntakeMessageRole,
    IntakeProfile,
    IntakeState,
    IntakeTurnRequest,
    InvalidDecisionLinkError,
    booking_decision,
    booking_decision_link_token,
    booking_request_email,
    booking_request_notice,
    cancel_request_notice,
    conversation_token_hash,
    escalation_notice,
    format_slot_label,
    intake_booking_cancel_command,
    intake_customer_email_command,
    intake_customer_email_request,
    intake_customer_text_command,
    intake_owner_email_request,
    new_conversation_token,
    new_reference,
    parse_booking_decision_link_token,
    pick_offer_slots,
    scrub_agent_reply,
    slot_confirmed_reply,
    slot_held_reply,
    slot_message_start,
    unverified_booking_change_notice,
    visit_note,
    visit_note_notice,
    with_visit_note,
)
from gvas.domain.messages import (
    CustomerDeliveryRequest,
    CustomerRecipient,
    CustomerTextRequest,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.owner_actions import decide_booking
from gvas.domain.owner_email import (
    OWNER_EMAIL_SOURCE_NAMESPACE,
    OwnerEmailAction,
    OwnerEmailRequest,
)
from gvas.domain.ports import (
    AvailabilityPort,
    CustomerQuoteDeliveryPort,
    CustomerTextDeliveryPort,
    IntakeAgentPort,
    OwnerEmailPort,
)
from gvas.domain.quotes import normalize_customer_email
from gvas.domain.repositories import BusinessRecord, UnitOfWork
from gvas.domain.time_zones import business_zone, zone_key
from gvas.domain.usage import UsageCeilingGuard, UsageKind
from gvas.domain.workflows import WorkflowContext, WorkflowResult

logger = logging.getLogger(__name__)

UnitOfWorkFactory = Callable[[], UnitOfWork]
SLOT_LOOKAHEAD_DAYS = 14
# Calendly rejects a start_time that is not strictly in the future by the time
# the request lands; a lead also keeps slots the customer cannot make out.
SLOT_LEAD = timedelta(minutes=5)
REPLY_OFFSET = timedelta(microseconds=1)

MESSAGE_LIMIT_REPLY = (
    "This conversation has reached its message limit — the owner will follow up with you directly."
)
UNAVAILABLE_REPLY = "I'm having trouble right now — the owner will follow up with you shortly."
NO_AVAILABILITY_REPLY = (
    "I don't have an opening to offer just now — the owner will confirm a "
    "time when they review your request."
)
SLOTS_OFFER_REPLY = "Here are the next openings I can offer — pick whichever works for you:"
VISIT_NOTED_REPLY = "Thanks — I've added that to your request for the crew."
SLOT_NOT_OFFERED_REPLY = "That time isn't one I can offer — please pick one of the listed times."
OWNER_UNREACHABLE_REPLY = (
    "I couldn't reach the owner to confirm that time just now — "
    "please pick it again in a little while."
)
ESCALATION_REPLY = "Let me bring the owner in on this — they'll follow up with you directly."
OPENING_REPLY = (
    "Hi! I can help you book an inspection or estimate. What's going on, and where is the property?"
)
PORTAL_OPENING_REPLY = "Welcome back! What do you need this time, and where is the property?"
BOOKING_ABOUT_MAX_CHARS = 120
RESCHEDULE_OFFER_REPLY = (
    "Here are the next openings — pick one and I'll send it over for approval. "
    "Your current time stays until the new one is approved."
)
RESCHEDULE_UNAVAILABLE_REPLY = (
    "I don't have another opening to offer just now — the team will follow up to "
    "find a new time with you."
)
CANCEL_CONFIRMED_REPLY = "Done — your call is canceled and the team has been told."
UNVERIFIED_CHANGE_REPLY = (
    "I've passed that to the owner. I can't confirm who you are from this chat, "
    "so they'll check with the contact on the booking before anything changes."
)
UNVERIFIED_CHANGE_UNREACHABLE_REPLY = (
    "I couldn't reach the owner just now — please try again in a little while."
)


class IntakeError(ValueError):
    """Base for intake failures already safe to surface."""


class IntakeNotFoundError(IntakeError):
    """Unknown business or conversation."""


class IntakeAuthenticationError(IntakeError):
    """Missing, wrong or expired conversation token."""


class IntakeClosedError(IntakeError):
    """The conversation is decided, expired or over its message cap."""


class IntakeLimitError(IntakeError):
    """The business is over its daily intake cap."""


class IntakeAvailabilityError(IntakeError):
    """Availability could not be arranged; commands that see this retry."""


class IntakeDeliveryError(RuntimeError):
    """A customer email or text was rejected; the outbox retries."""


@dataclass(frozen=True)
class IntakeStart:
    """What a create-conversation response projects from."""

    conversation: IntakeConversation
    token: str
    reply: str


@dataclass(frozen=True)
class IntakeReply:
    conversation: IntakeConversation
    reply: str
    slots: tuple[AvailableSlot, ...]


class OwnerDecisionEmail:
    """The signed one-click Approve/Decline buttons for one booking request.
    Off until the secret and origin are configured; minted only once
    ``owner_notified_at`` is set on the conversation passed in."""

    def __init__(
        self,
        *,
        secret: str = "",
        origin: str = "",
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._secret = secret
        self._origin = origin.rstrip("/")
        self._now = now

    def actions(self, conversation: IntakeConversation) -> tuple[OwnerEmailAction, ...]:
        if not self._secret or not self._origin or conversation.owner_notified_at is None:
            return ()
        actions = []
        for action, verb in (
            (BookingDecisionAction.APPROVE, "Approve"),
            (BookingDecisionAction.DECLINE, "Decline"),
        ):
            token = booking_decision_link_token(
                self._secret, conversation=conversation, action=action, now=self._now()
            )
            actions.append(
                OwnerEmailAction(
                    label=verb,
                    url=f"{self._origin}{BOOKING_DECISION_PATH}{token}",
                    primary=action is BookingDecisionAction.APPROVE,
                )
            )
        return tuple(actions)


@dataclass(frozen=True)
class IntakeVisitor:
    """A visitor whose details are known before they type: the chat opens
    with them collected and ``opening`` as its first reply (``{business}``
    becomes the business's name)."""

    collected: IntakeCollected
    opening: str


class IntakeService:
    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        *,
        agent: IntakeAgentPort,
        availability: AvailabilityPort | None = None,
        ceiling: UsageCeilingGuard | None = None,
        max_conversations_per_day: int = 50,
        max_user_messages: int = INTAKE_MAX_USER_MESSAGES,
        message_budget: Callable[[UUID, datetime], Awaitable[str | None]] | None = None,
        visitor: Callable[[UUID], Awaitable[IntakeVisitor | None]] | None = None,
        decision_link_secret: str = "",
        decision_link_base_url: str = "",
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._agent = agent
        self._decision_email = OwnerDecisionEmail(
            secret=decision_link_secret,
            origin=decision_link_base_url,
            now=now,
        )
        self._availability = availability
        self._ceiling = ceiling or UsageCeilingGuard()
        self._max_conversations_per_day = max_conversations_per_day
        self._max_user_messages = max_user_messages
        # Returns the reply to give instead of a model turn once a budget
        # beyond this conversation (a demo sandbox's, a demo's day) is spent.
        self._message_budget = message_budget
        self._visitor = visitor
        self._now = now

    async def start_conversation(
        self, public_key: str, *, sms_consent: bool | None = None
    ) -> IntakeStart:
        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get_by_public_key(public_key)
            if business is None:
                raise IntakeNotFoundError("unknown business")
            await self._check_daily_cap(unit_of_work, business)
            return await self._open(unit_of_work, business, customer=None, sms_consent=sms_consent)

    async def start_portal_conversation(
        self,
        business: BusinessRecord,
        customer: CustomerRecord,
        *,
        sms_consent: bool | None = None,
    ) -> IntakeStart:
        """Portal-authenticated start: the customer is already identified, so
        the collected record comes pre-filled and identity questions are
        skipped."""

        async with self._unit_of_work_factory() as unit_of_work:
            await self._check_daily_cap(unit_of_work, business)
            return await self._open(
                unit_of_work, business, customer=customer, sms_consent=sms_consent
            )

    async def _check_daily_cap(self, unit_of_work: UnitOfWork, business: BusinessRecord) -> None:
        if self._max_conversations_per_day <= 0:
            return
        # Locking the business row serializes concurrent starts so two
        # requests cannot both pass the count before either inserts.
        await unit_of_work.businesses.lock(business.business_id)
        now = self._now()
        day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        count = await unit_of_work.intake_conversations.count_created_since(
            business.business_id, day_start
        )
        if count >= self._max_conversations_per_day:
            raise IntakeLimitError("this business is not taking new requests today")

    async def _open(
        self,
        unit_of_work: UnitOfWork,
        business: BusinessRecord,
        *,
        customer: CustomerRecord | None,
        sms_consent: bool | None,
    ) -> IntakeStart:
        customer_record = customer
        now = self._now()
        answered = sms_consent is not None
        consent_at = now if answered else None
        if not answered and customer_record is not None:
            sms_consent = customer_record.sms_consent
            consent_at = customer_record.sms_consent_at
        token = new_conversation_token()
        collected = IntakeCollected()
        if customer_record is not None:
            collected = IntakeCollected(
                name=customer_record.display_name,
                email=customer_record.email,
                phone=customer_record.phone,
            )
        if customer_record is not None:
            reply = PORTAL_OPENING_REPLY
        else:
            reply = business.intake_profile.opening or OPENING_REPLY
            visitor = None if self._visitor is None else await self._visitor(business.business_id)
            if visitor is not None:
                collected = visitor.collected
                name = business.display_name or business.name
                reply = visitor.opening.replace("{business}", name)
        conversation = IntakeConversation(
            conversation_id=IntakeConversationId(uuid4()),
            business_id=business.business_id,
            customer_id=None if customer_record is None else customer_record.customer_id,
            reference=new_reference(),
            token_hash=conversation_token_hash(token),
            channel=INTAKE_CHANNEL_WEB if customer_record is None else INTAKE_CHANNEL_PORTAL,
            collected=collected,
            sms_consent=sms_consent,
            sms_consent_at=consent_at,
            expires_at=now + INTAKE_CONVERSATION_TTL,
            created_at=now,
            updated_at=now,
        )
        await unit_of_work.intake_conversations.add(conversation)
        if customer_record is not None and answered and sms_consent is not None:
            await unit_of_work.customers.set_sms_consent(
                business.business_id, customer_record.customer_id, sms_consent, now
            )
        await self._append(unit_of_work, conversation, IntakeMessageRole.AGENT, reply, now)
        await unit_of_work.commit()
        return IntakeStart(conversation=conversation, token=token, reply=reply)

    async def authenticate(
        self, conversation_id: IntakeConversationId, raw_token: str
    ) -> IntakeConversation:
        """The bearer token is the credential: a conversation answers only to
        the token minted for it, so a token from one business can never read
        another's chat."""

        token_hash = conversation_token_hash(raw_token)
        async with self._unit_of_work_factory() as unit_of_work:
            conversation = await unit_of_work.intake_conversations.find_by_token(
                conversation_id, token_hash
            )
        if conversation is None or self._now() >= conversation.expires_at:
            raise IntakeAuthenticationError("invalid or expired conversation token")
        return conversation

    async def record_sms_consent(
        self, conversation: IntakeConversation, consent: bool
    ) -> IntakeConversation:
        """Stores the visitor's SMS consent answer; once a booking request
        made them a customer, the customer record follows it too. A finished
        chat still accepts a withdrawal, never a new yes."""

        now = self._now()
        async with self._unit_of_work_factory() as unit_of_work:
            locked = await unit_of_work.intake_conversations.lock(
                conversation.business_id, conversation.conversation_id
            )
            if (
                locked is None
                or locked.sms_consent is consent
                or (consent and not locked.is_live(now))
            ):
                await unit_of_work.commit()
                return locked or conversation
            updated = locked.with_updates(now, sms_consent=consent, sms_consent_at=now)
            await unit_of_work.intake_conversations.save(updated)
            if updated.customer_id is not None:
                await unit_of_work.customers.set_sms_consent(
                    updated.business_id, updated.customer_id, consent, now
                )
            await unit_of_work.commit()
            return updated

    async def summary(self, conversation: IntakeConversation) -> dict[str, object] | None:
        """The public summary, complete by the business's own questions."""

        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get(conversation.business_id)
        address_required = business is None or business.intake_profile.requires_address
        return conversation.collected.summary(
            address_required=address_required,
            phone_required=conversation.customer_id is None,
        )

    async def get_view(self, conversation: IntakeConversation) -> tuple[IntakeMessage, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.intake_messages.list_for(
                conversation.business_id, conversation.conversation_id
            )

    async def post_message(self, conversation: IntakeConversation, text: str) -> IntakeReply:
        """One customer turn: persist, decide, persist the agent reply.

        The deterministic checks (closed, cap, slot picks) all run before the
        model is asked anything; the model never sees a request it cannot
        answer and its reply is scrubbed before it is stored.
        """

        content = text.strip()
        if not content or len(content) > INTAKE_MESSAGE_MAX_CHARS:
            raise IntakeError("message must be between 1 and 2000 characters")
        now = self._now()
        if not conversation.is_live(now):
            raise IntakeClosedError("this conversation is closed")

        async with self._unit_of_work_factory() as unit_of_work:
            # The row lock makes the whole turn read-modify-write on one
            # snapshot: concurrent posts serialize instead of losing each
            # other's collected fields or state transitions.
            locked = await unit_of_work.intake_conversations.lock(
                conversation.business_id, conversation.conversation_id
            )
            if locked is None or not locked.is_live(now):
                raise IntakeClosedError("this conversation is closed")
            conversation = locked
            await self._append(unit_of_work, conversation, IntakeMessageRole.USER, content, now)
            user_count = await unit_of_work.intake_messages.count_user(
                conversation.business_id, conversation.conversation_id
            )
            if 0 < self._max_user_messages < user_count:
                # Terminal: without a closed state every further post would
                # still append transcript rows forever.
                closed = conversation.with_updates(now, state=IntakeState.CLOSED)
                await unit_of_work.intake_conversations.save(closed)
                reply = await self._reply(unit_of_work, closed, MESSAGE_LIMIT_REPLY, now)
                await unit_of_work.commit()
                return IntakeReply(closed, reply, closed.proposed_slots)

            result = await self._step(unit_of_work, conversation, content, now)
            await unit_of_work.commit()
            return result

    async def _append(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        role: IntakeMessageRole,
        content: str,
        now: datetime,
    ) -> None:
        await unit_of_work.intake_messages.add(
            IntakeMessage(
                message_id=IntakeMessageId(uuid4()),
                conversation_id=conversation.conversation_id,
                business_id=conversation.business_id,
                role=role,
                content=content,
                created_at=now,
            )
        )

    async def _reply(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        reply: str,
        now: datetime,
    ) -> str:
        # A reply shares the turn's clock with the message it answers; one
        # microsecond later keeps it after that message when ordered by time.
        await self._append(
            unit_of_work, conversation, IntakeMessageRole.AGENT, reply, now + REPLY_OFFSET
        )
        return reply

    async def _step(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        content: str,
        now: datetime,
    ) -> IntakeReply:
        picked = slot_message_start(content)
        if picked is not None:
            return await self._handle_slot_pick(
                unit_of_work, conversation, picked, now, explicit=True
            )
        # Read before the model turn so a cancel or reschedule still reaches
        # the booking-change handling; when the model can't run, the answer
        # is saved as is rather than lost.
        answering = await self._answers_visit_question(unit_of_work, conversation)

        if await self._ceiling.is_reached(
            conversation.business_id, UsageKind.REVIEW_TOKENS, now=now
        ):
            if answering:
                return await self._note_visit_answer(unit_of_work, conversation, content, now)
            reply = await self._reply(unit_of_work, conversation, UNAVAILABLE_REPLY, now)
            return IntakeReply(conversation, reply, ())
        if self._message_budget is not None:
            refusal = await self._message_budget(conversation.business_id, now)
            if refusal is not None:
                if answering:
                    return await self._note_visit_answer(unit_of_work, conversation, content, now)
                reply = await self._reply(unit_of_work, conversation, refusal, now)
                return IntakeReply(conversation, reply, conversation.proposed_slots)

        transcript = await unit_of_work.intake_messages.list_for(
            conversation.business_id, conversation.conversation_id
        )
        business = await self._business(unit_of_work, conversation.business_id)
        holder = await self._live_booking_holder(unit_of_work, conversation)
        verified = await self._holder_verified(unit_of_work, conversation, holder)
        try:
            turn = await self._agent.turn(
                IntakeTurnRequest(
                    business_id=conversation.business_id,
                    conversation_id=conversation.conversation_id,
                    business_name=business.display_name or business.name,
                    transcript=transcript,
                    collected=conversation.collected,
                    offered_slots=conversation.proposed_slots,
                    known_customer=conversation.customer_id is not None,
                    existing_booking=self._existing_booking(
                        holder, verified=verified, zone=_zone_of(business)
                    ),
                    brief=business.intake_profile.brief,
                    questions=business.intake_profile.questions,
                )
            )
        except (IntakeAgentError, AvailabilityError) as error:
            # A provider outage must not 500 the chat: the customer's message
            # is already persisted, so answer with the sanitized fallback and
            # leave the conversation live for the next message.
            logger.warning("intake agent unavailable for %s: %s", conversation.reference, error)
            if answering:
                return await self._note_visit_answer(unit_of_work, conversation, content, now)
            reply = await self._reply(unit_of_work, conversation, UNAVAILABLE_REPLY, now)
            return IntakeReply(conversation, reply, conversation.proposed_slots)
        known_customer = conversation.customer_id is not None
        # The model often never sets ``ready_for_slots``. Details complete
        # since the last turn mean Gus has had his turn for follow-ups, so
        # the next message offers times whatever the model says.
        was_ready = _ready_for_slots(
            conversation.collected, business.intake_profile, known_customer=known_customer
        )
        if answering and not (turn.wants_cancel or turn.wants_reschedule or turn.needs_human):
            # The answer is saved word for word: the model's notes from this
            # turn would repeat it. Other fields only fill blanks.
            collected = conversation.collected.merge(
                turn.collected.model_copy(update={"notes": None})
            )
            current = conversation.with_updates(now, collected=collected)
            return await self._note_visit_answer(unit_of_work, current, content, now)

        collected = conversation.collected.merge(turn.collected)
        current = conversation.with_updates(now, collected=collected)
        await unit_of_work.intake_conversations.save(current)

        if holder is not None:
            if (turn.wants_cancel or turn.wants_reschedule) and not verified:
                return await self._refer_booking_change(
                    unit_of_work, current, holder, business, now, cancel=turn.wants_cancel
                )
            if turn.wants_cancel:
                return await self._cancel_booking(unit_of_work, current, holder, business, now)
            if turn.wants_reschedule:
                return await self._offer_reschedule(unit_of_work, current, holder, now)

        if turn.needs_human:
            reply_text = f"{scrub_agent_reply(turn.reply)} {ESCALATION_REPLY}"
            await self._reply(unit_of_work, current, reply_text, now)
            if current.escalation_notified_at is None:
                notified = await enqueue_intake_owner_notice(
                    unit_of_work,
                    current.business_id,
                    correlation_id=f"intake_escalation:{current.conversation_id}",
                    text=escalation_notice(current, turn.summary),
                )
                if notified:
                    current = current.with_updates(now, escalation_notified_at=now)
                    await unit_of_work.intake_conversations.save(current)
            return IntakeReply(current, reply_text, current.proposed_slots)

        if turn.chosen_slot is not None and (
            current.state is IntakeState.PROPOSING_SLOTS or current.is_choosing_new_time
        ):
            return await self._handle_slot_pick(
                unit_of_work, current, turn.chosen_slot, now, explicit=False
            )

        if (
            holder is None
            and (turn.ready_for_slots or was_ready)
            and _ready_for_slots(
                current.collected, business.intake_profile, known_customer=known_customer
            )
        ):
            if current.state is IntakeState.PROPOSING_SLOTS:
                # A re-offer while slots are on the table: the customer can
                # still pick, just show them again.
                pass
            else:
                offered = await self._offer_slots(unit_of_work, current, now)
                if offered is None:
                    reply = await self._reply(unit_of_work, current, NO_AVAILABILITY_REPLY, now)
                    return IntakeReply(current, reply, ())
                current = offered
                offer = business.intake_profile.offer_line or SLOTS_OFFER_REPLY
                await self._reply(unit_of_work, current, offer, now)
                return IntakeReply(current, offer, current.proposed_slots)

        reply_text = scrub_agent_reply(turn.reply)
        await self._reply(unit_of_work, current, reply_text, now)
        return IntakeReply(current, reply_text, current.proposed_slots)

    async def _live_booking_holder(
        self, unit_of_work: UnitOfWork, conversation: IntakeConversation
    ) -> IntakeConversation | None:
        """The conversation holding this visitor's live booking, if any.

        The visitor's own conversation counts; otherwise a live booking under
        the same collected e-mail — the identifier the owner approved against
        — does, which is how a returning visitor is recognised without their
        stored conversation id.
        """

        if conversation.has_live_booking:
            return conversation
        if self._visitor is not None and await self._visitor(conversation.business_id) is not None:
            # Every chat there shares the visitor's e-mail, so it names no one.
            return None
        email = conversation.collected.email
        if not email:
            return None
        other = await unit_of_work.intake_conversations.lock_latest_by_invitee_email(
            conversation.business_id, email
        )
        if (
            other is None
            or other.conversation_id == conversation.conversation_id
            or not other.has_live_booking
        ):
            return None
        return other

    @staticmethod
    async def _holder_verified(
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        holder: IntakeConversation | None,
    ) -> bool:
        """Whether this chat may act on ``holder``'s booking itself.

        Its own booking, yes. Another chat's only from a verified portal
        session for the customer who owns it: an e-mail typed into an
        anonymous chat proves nothing.
        """

        if holder is None:
            return False
        if holder.conversation_id == conversation.conversation_id:
            return True
        if conversation.channel != INTAKE_CHANNEL_PORTAL or conversation.customer_id is None:
            return False
        if holder.customer_id is not None:
            return holder.customer_id == conversation.customer_id
        # An anonymous request typed with this customer's address was left
        # unlinked; the portal sign-in proved the address, so it is theirs.
        customer = await unit_of_work.customers.get(
            conversation.business_id, conversation.customer_id
        )
        email = holder.collected.email
        return (
            customer is not None
            and email is not None
            and normalize_customer_email(email) == customer.email
        )

    @staticmethod
    def _existing_booking(
        holder: IntakeConversation | None, *, verified: bool, zone: tzinfo | None
    ) -> ExistingBooking | None:
        if holder is None or holder.requested_slot_start is None:
            return None
        if not verified:
            return ExistingBooking(verified=False)
        status = (
            ExistingBookingStatus.REQUESTED
            if holder.state is IntakeState.AWAITING_OWNER
            else ExistingBookingStatus.CONFIRMED
        )
        return ExistingBooking(
            status=status, slot_label=format_slot_label(holder.requested_slot_start, zone)
        )

    async def _refer_booking_change(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        holder: IntakeConversation,
        business: BusinessRecord,
        now: datetime,
        *,
        cancel: bool,
    ) -> IntakeReply:
        """A chat linked to another chat's booking only by a typed e-mail
        asked to cancel or move it: the owner is told and nothing changes —
        the booking, its calendar event and its chat stay as they are."""

        kind = "cancel" if cancel else "move"
        notified = await enqueue_intake_owner_notice(
            unit_of_work,
            holder.business_id,
            correlation_id=(
                f"intake_unverified_{kind}:{conversation.conversation_id}:{holder.reference}"
            ),
            text=unverified_booking_change_notice(
                holder,
                conversation,
                cancel=cancel,
                business_name=business.display_name or business.name,
                zone=_zone_of(business),
            ),
        )
        text = UNVERIFIED_CHANGE_REPLY if notified else UNVERIFIED_CHANGE_UNREACHABLE_REPLY
        reply = await self._reply(unit_of_work, conversation, text, now)
        return IntakeReply(conversation, reply, ())

    async def _cancel_booking(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        holder: IntakeConversation,
        business: BusinessRecord,
        now: datetime,
    ) -> IntakeReply:
        """The customer drops their own call (this chat's, or a verified portal
        customer's) — no owner approval needed; the owner hears about it and
        any calendar event is cancelled."""

        notified = await enqueue_intake_owner_notice(
            unit_of_work,
            holder.business_id,
            correlation_id=f"intake_cancel:{holder.conversation_id}",
            text=cancel_request_notice(
                holder,
                business_name=business.display_name or business.name,
                zone=_zone_of(business),
            ),
        )
        if not notified:
            logger.warning(
                "customer cancel for booking %s stored without an owner notice",
                holder.reference,
            )
        if holder.booked_event_uri:
            await unit_of_work.outbox.enqueue(intake_booking_cancel_command(holder))
        # The caller may have already copied the conversation (``with_updates``
        # merge), so same-chat is decided by id, not object identity.
        same_chat = holder.conversation_id == conversation.conversation_id
        closed = (conversation if same_chat else holder).with_updates(now, state=IntakeState.CLOSED)
        await unit_of_work.intake_conversations.save(closed)
        reply = await self._reply(unit_of_work, conversation, CANCEL_CONFIRMED_REPLY, now)
        return IntakeReply(closed if same_chat else conversation, reply, ())

    async def _offer_reschedule(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        holder: IntakeConversation,
        now: datetime,
    ) -> IntakeReply:
        """Offer new times while the current booking stays in force.

        A pending request is simply re-picked inside its own conversation; an
        approved booking in another conversation is adopted here first, so the
        new request carries custody of the old event.
        """

        if holder.conversation_id != conversation.conversation_id:
            if holder.state is IntakeState.APPROVED:
                conversation = await self._adopt_booking(unit_of_work, conversation, holder, now)
            else:
                reply = await self._reply(
                    unit_of_work,
                    conversation,
                    "Your requested time is still waiting on approval — once it's "
                    "confirmed you can move it, or I can cancel it now.",
                    now,
                )
                return IntakeReply(conversation, reply, ())
        offered = await self._offer_slots(unit_of_work, conversation, now, keep_booked=True)
        if offered is None:
            reply = await self._reply(unit_of_work, conversation, RESCHEDULE_UNAVAILABLE_REPLY, now)
            return IntakeReply(conversation, reply, ())
        reply = await self._reply(unit_of_work, offered, RESCHEDULE_OFFER_REPLY, now)
        return IntakeReply(offered, reply, offered.proposed_slots)

    async def _adopt_booking(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        holder: IntakeConversation,
        now: datetime,
    ) -> IntakeConversation:
        """Move an approved booking from a previous conversation into this one
        (same verified portal customer); the old chat closes."""

        adopted = conversation.with_updates(
            now,
            state=IntakeState.APPROVED,
            requested_slot_start=holder.requested_slot_start,
            requested_slot_end=holder.requested_slot_end,
            booking_kind=holder.booking_kind,
            booking_link=holder.booking_link,
            booking_attempted_at=holder.booking_attempted_at,
            booked_event_uri=holder.booked_event_uri,
            booking_event_type_uri=holder.booking_event_type_uri,
            decision_at=holder.decision_at,
        )
        closed_holder = holder.with_updates(
            now,
            state=IntakeState.CLOSED,
            # The event now belongs to the new conversation; clearing the uri
            # keeps a late cancellation webhook from re-closing this row into
            # a misleading state.
            booked_event_uri=None,
            superseded_booking=None,
            reschedule_offered_at=None,
        )
        await unit_of_work.intake_conversations.save(closed_holder)
        await unit_of_work.intake_conversations.save(adopted)
        return adopted

    async def _offer_slots(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        now: datetime,
        *,
        keep_booked: bool = False,
    ) -> IntakeConversation | None:
        if self._availability is None:
            return None
        start = now + SLOT_LEAD
        end = start + timedelta(days=SLOT_LOOKAHEAD_DAYS)
        try:
            openings = await self._availability.available_slots(
                conversation.business_id, start, end
            )
        except AvailabilityError as error:
            logger.warning("availability lookup failed for %s: %s", conversation.reference, error)
            return None
        await self._learn_business_zone(unit_of_work, conversation.business_id, openings, now)
        offered = pick_offer_slots(tuple(openings), now=now)
        if not offered:
            return None
        if keep_booked:
            # Reschedule: the booking stands while new times are on the table.
            updated = conversation.with_updates(
                now, proposed_slots=offered, reschedule_offered_at=now
            )
        else:
            updated = conversation.with_updates(
                now, state=IntakeState.PROPOSING_SLOTS, proposed_slots=offered
            )
        await unit_of_work.intake_conversations.save(updated)
        return updated

    async def _handle_slot_pick(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        start: datetime,
        now: datetime,
        *,
        explicit: bool,
    ) -> IntakeReply:
        choosing = (
            conversation.state is IntakeState.PROPOSING_SLOTS or conversation.is_choosing_new_time
        )
        if not choosing:
            reply = await self._reply(unit_of_work, conversation, SLOT_NOT_OFFERED_REPLY, now)
            return IntakeReply(conversation, reply, conversation.proposed_slots)
        slot = conversation.find_proposed_slot(start)
        if slot is None:
            reply = await self._reply(unit_of_work, conversation, SLOT_NOT_OFFERED_REPLY, now)
            return IntakeReply(conversation, reply, conversation.proposed_slots)
        if conversation.is_choosing_new_time and conversation.requested_slot_start == slot.start:
            texts = await self._confirms_by_text(
                unit_of_work, conversation, conversation.customer_id
            )
            reply = await self._reply(unit_of_work, conversation, slot_held_reply(texts=texts), now)
            return IntakeReply(conversation, reply, conversation.proposed_slots)

        collected = conversation.collected
        customer_id = conversation.customer_id
        unverified_email = False
        if customer_id is None and collected.email:
            # A typed e-mail proves nothing: it may start a new customer
            # record, but never links to (or fills in) an existing one.
            customer = await unit_of_work.customers.create(
                conversation.business_id,
                collected.email,
                display_name=collected.name,
                phone=collected.phone,
                now=now,
            )
            if customer is None:
                unverified_email = True
            else:
                customer_id = customer.customer_id
        if customer_id is not None and conversation.sms_consent is not None:
            await unit_of_work.customers.set_sms_consent(
                conversation.business_id,
                customer_id,
                conversation.sms_consent,
                conversation.sms_consent_at or now,
            )
        # A pick while a reschedule is already pending re-requests under the
        # same reference — the approved event already in custody stays in
        # custody (``booking_snapshot`` only fires for an approved state).
        superseded = conversation.booking_snapshot() or conversation.superseded_booking
        business = await self._business(unit_of_work, conversation.business_id)
        zone = _zone_of(business) or slot.start.tzinfo
        previous_label: str | None = None
        if superseded is not None:
            previous_label = superseded.slot_label(zone)
        elif (
            conversation.state is IntakeState.AWAITING_OWNER
            and conversation.requested_slot_start is not None
        ):
            previous_label = format_slot_label(conversation.requested_slot_start, zone)
        updated = conversation.with_updates(
            now,
            state=IntakeState.AWAITING_OWNER,
            customer_id=customer_id,
            requested_slot_start=slot.start,
            requested_slot_end=slot.end,
            # Custody of the previous booking moves into ``superseded`` — it
            # stays on the calendar until the owner approves the new time.
            superseded_booking=superseded,
            reschedule_offered_at=None,
            booking_kind=None,
            booking_link=None,
            booking_attempted_at=None,
            booked_event_uri=None,
            booking_event_type_uri=None,
            decision_at=None,
            decision_reason=None,
            owner_notified_at=now,
        )
        correlation_id = f"intake_request:{conversation.conversation_id}"
        if conversation.owner_notified_at is not None:
            # A second request from one conversation needs a fresh id or the
            # notice/email copies dedup against the first request's send.
            correlation_id = f"{correlation_id}:{int(now.timestamp())}"
        notified = await enqueue_intake_owner_notice(
            unit_of_work,
            conversation.business_id,
            correlation_id=correlation_id,
            text=booking_request_notice(
                updated,
                business_name=business.display_name or business.name,
                profile=business.intake_profile,
                previous_label=previous_label,
                previous_booked=superseded is not None,
                unverified_email=unverified_email,
                zone=zone,
            ),
            email=booking_request_email(
                updated,
                profile=business.intake_profile,
                previous_label=previous_label,
                previous_booked=superseded is not None,
                unverified_email=unverified_email,
                zone=zone,
            ),
            email_actions=self._decision_email.actions(updated),
        )
        if not notified:
            # There is no owner thread to deliver the decision request to:
            # entering awaiting_owner would strand the customer waiting on an
            # approval that can never arrive, so the pick does not land and
            # the slots stay on the table for a retry.
            logger.warning(
                "booking request %s has no owner thread to notify",
                updated.reference,
            )
            reply = await self._reply(unit_of_work, conversation, OWNER_UNREACHABLE_REPLY, now)
            return IntakeReply(conversation, reply, conversation.proposed_slots)
        if customer_id is not None:
            preferred = format_slot_label(slot.start, zone)
            await unit_of_work.service_requests.add(
                ServiceRequest(
                    request_id=ServiceRequestId(uuid4()),
                    business_id=conversation.business_id,
                    customer_id=customer_id,
                    message=collected.details or "Booking request",
                    preferred_dates=preferred,
                    source=SERVICE_REQUEST_SOURCE_INTAKE,
                    created_at=now,
                )
            )
        await unit_of_work.intake_conversations.save(updated)
        texts = await self._confirms_by_text(unit_of_work, updated, customer_id)
        reply = slot_confirmed_reply(slot, zone, texts=texts)
        question = business.intake_profile.visit_question
        if question and not _visit_noted(updated.collected):
            reply = f"{reply} {question}"
        await self._reply(unit_of_work, updated, reply, now)
        return IntakeReply(updated, reply, ())

    async def _answers_visit_question(
        self, unit_of_work: UnitOfWork, conversation: IntakeConversation
    ) -> bool:
        """The message right after the business's visit question answers it."""

        if conversation.state not in (IntakeState.AWAITING_OWNER, IntakeState.APPROVED):
            return False
        business = await self._business(unit_of_work, conversation.business_id)
        question = business.intake_profile.visit_question
        if not question:
            return False
        transcript = await unit_of_work.intake_messages.list_for(
            conversation.business_id, conversation.conversation_id
        )
        last_agent = next(
            (m for m in reversed(transcript) if m.role is IntakeMessageRole.AGENT), None
        )
        return last_agent is not None and last_agent.content.endswith(question)

    async def _note_visit_answer(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        content: str,
        now: datetime,
    ) -> IntakeReply:
        """Saves the visit answer word for word on the request and tells the
        owner, whose request notice went out before the answer came."""

        note = visit_note(content)
        current = conversation.with_updates(
            now, collected=with_visit_note(conversation.collected, note)
        )
        await unit_of_work.intake_conversations.save(current)
        await enqueue_intake_owner_notice(
            unit_of_work,
            current.business_id,
            correlation_id=f"intake_visit_note:{current.conversation_id}",
            text=visit_note_notice(current, note),
        )
        reply = await self._reply(unit_of_work, current, VISIT_NOTED_REPLY, now)
        return IntakeReply(current, reply, current.proposed_slots)

    @staticmethod
    async def _confirms_by_text(
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        customer_id: CustomerId | None,
    ) -> bool:
        """The approval is texted only to a linked customer who said yes to
        texts (the same rule the delivery uses), so the reply promises no more."""
        if customer_id is None or not conversation.collected.phone:
            return False
        customer = await unit_of_work.customers.get(conversation.business_id, customer_id)
        return customer is not None and customer.sms_consent is True

    @staticmethod
    async def _learn_business_zone(
        unit_of_work: UnitOfWork,
        business_id: BusinessId,
        openings: Sequence[AvailableSlot],
        now: datetime,
    ) -> None:
        """The first availability read fills in an unset business zone from
        the calendar's own (the owner can change it in Settings)."""

        key = next((key for slot in openings if (key := zone_key(slot.start))), None)
        if key is None:
            return
        await unit_of_work.businesses.adopt_timezone(business_id, key, now)

    async def _business(self, unit_of_work: UnitOfWork, business_id: BusinessId) -> BusinessRecord:
        business = await unit_of_work.businesses.get(business_id)
        if business is None:
            raise IntakeNotFoundError("unknown business")
        return business


class ArrangeIntakeBookingService:
    """Turns an approval into a calendar booking via the availability port.

    The approval is already persisted (state ``approved``); this runs from the
    outbox so a provider hiccup retries instead of losing the booking, and the
    customer notifications are enqueued only after the booking result is
    known.
    """

    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        *,
        availability: AvailabilityPort | None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._availability = availability
        self._now = now

    async def arrange(self, business_id: BusinessId, conversation_id: IntakeConversationId) -> None:
        if self._availability is None:
            raise IntakeAvailabilityError("no availability provider is configured")
        async with self._unit_of_work_factory() as unit_of_work:
            conversation = await unit_of_work.intake_conversations.get(business_id, conversation_id)
            if conversation is None or conversation.state is not IntakeState.APPROVED:
                return
            if conversation.requested_slot_start is None or conversation.requested_slot_end is None:
                return
            if conversation.booking_kind is not None:
                return  # already arranged — an enqueued retry
            collected = conversation.collected
            if not collected.name or not collected.email:
                raise IntakeAvailabilityError(
                    "approved booking is missing the customer's name or email"
                )
            request = BookingRequest(
                business_id=business_id,
                slot_start=conversation.requested_slot_start,
                slot_end=conversation.requested_slot_end,
                invitee_name=collected.name,
                invitee_email=collected.email,
                invitee_phone=collected.phone,
                address=collected.address,
                details=collected.details,
                reference=conversation.reference,
                event_type_uri=conversation.booking_event_type_uri,
            )
            # A webhook-recorded event also counts as attempted: the provider
            # lookup below finds it, so a re-approved re-route reconciles
            # instead of booking a second event.
            attempted = (
                conversation.booking_attempted_at is not None
                or conversation.booked_event_uri is not None
            )
            if not attempted:
                # Persist the attempt and the event type book() will use
                # before the provider call: if the process dies after ``book``
                # succeeded, the retried command sees the marker and
                # reconciles against the same type instead of booking twice.
                event_type_uri = await self._availability.booking_event_type_uri(business_id)
                request = request.model_copy(update={"event_type_uri": event_type_uri})
                await unit_of_work.intake_conversations.save(
                    conversation.with_updates(
                        self._now(),
                        booking_attempted_at=self._now(),
                        booking_event_type_uri=event_type_uri,
                    )
                )
                await unit_of_work.commit()

        result = None
        if attempted:
            result = await self._availability.find_booking(request)
        if result is None:
            result = await self._availability.book(request)

        async with self._unit_of_work_factory() as unit_of_work:
            conversation = await unit_of_work.intake_conversations.get(business_id, conversation_id)
            if (
                conversation is None
                or conversation.state is not IntakeState.APPROVED
                or conversation.requested_slot_start is None
                or conversation.booking_kind is not None
            ):
                return
            business = await unit_of_work.businesses.get(business_id)
            now = self._now()
            business_name = (
                "" if business is None else (business.display_name or business.name)
            ) or "the business"
            slot_label = format_slot_label(conversation.requested_slot_start, _zone_of(business))
            about = _booking_about(conversation.collected, business)
            if result.kind is BookingKind.BOOKED:
                body = (
                    f"Good news — your {business_name} appointment{about} for "
                    f"{slot_label} is booked. You'll get the calendar invite "
                    "by email shortly."
                )
                subject = "Your appointment is confirmed"
            else:
                link = result.link or ""
                body = f"{business_name} approved {slot_label}{about}. Confirm your spot: {link}"
                subject = "Confirm your appointment"
            # The decision stamp keeps re-bookings (reschedule approved
            # again) from deduping against the first request's e-mail/text.
            stamp = conversation.decision_at or conversation.booking_attempted_at or now
            key_base = f"intake_booking:{conversation.conversation_id}:{int(stamp.timestamp())}"
            await unit_of_work.outbox.enqueue(
                intake_customer_email_command(
                    IntakeCustomerEmail(
                        business_id=business_id,
                        to=collected.email,
                        subject=subject,
                        body=body,
                        idempotency_key=f"{key_base}:email",
                    )
                )
            )
            customer = (
                None
                if conversation.customer_id is None
                else await unit_of_work.customers.get(business_id, conversation.customer_id)
            )
            if collected.phone and customer is not None and customer.sms_consent is True:
                await unit_of_work.outbox.enqueue(
                    intake_customer_text_command(
                        business_id,
                        customer_id=customer.customer_id,
                        phone=collected.phone,
                        text=body,
                        idempotency_key=f"{key_base}:text",
                    )
                )
            updated = conversation.with_updates(
                now,
                booking_kind=result.kind.value,
                booking_link=result.link,
                booking_event_type_uri=(
                    result.event_type_uri or conversation.booking_event_type_uri
                ),
            )
            await unit_of_work.intake_conversations.save(updated)
            await unit_of_work.commit()


def _ready_for_slots(
    collected: IntakeCollected, profile: IntakeProfile, *, known_customer: bool
) -> bool:
    """Contact details and what is needed; the generic questions also need
    the service address, and a known customer's record may lack a phone."""

    return collected.is_complete(
        address_required=profile.requires_address, phone_required=not known_customer
    )


def _visit_noted(collected: IntakeCollected) -> bool:
    return VISIT_NOTE_LABEL.lower() in (collected.notes or "").lower()


def _booking_about(collected: IntakeCollected, business: BusinessRecord | None) -> str:
    """`` (re: <details>)`` for profiled businesses, so the customer copy
    names what they booked rather than a generic appointment."""

    if business is None or not business.intake_profile.is_configured or not collected.details:
        return ""
    details = collected.details.strip().rstrip(".")
    if len(details) > BOOKING_ABOUT_MAX_CHARS:
        details = details[: BOOKING_ABOUT_MAX_CHARS - 1].rstrip() + "…"
    return f" (re: {details})"


def _decision_reply(message: NormalizedOwnerMessage, text: str) -> OutboundOwnerMessage:
    return OutboundOwnerMessage(
        business_id=message.business_id,
        conversation_ref=message.conversation_ref,
        parts=(TextPart(text=text),),
        correlation_id=f"booking:{message.message_key}",
    )


def _booking_text(message: NormalizedOwnerMessage) -> str:
    return "\n".join(part.text for part in message.parts if isinstance(part, TextPart))


class BookingDecisionHandler:
    """``approve booking <ref>`` / ``decline booking <ref> [reason]``.

    Tenant-scoped by the business of the owner's message; the transition
    itself is shared with the dashboard and the e-mail links
    (``decide_booking``).
    """

    intent = BOOKING_INTENT

    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._now = now

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        message = context.message
        decision = booking_decision(_booking_text(message))
        if decision is None:
            return self._result(
                message,
                "Reply `approve booking <id>`, `decline booking <id> <reason>` "
                "or `cancel booking <id>`.",
            )
        async with self._unit_of_work_factory() as unit_of_work:
            endpoint = await unit_of_work.conversations.find_endpoint(message.conversation_ref)
            if endpoint is not None and endpoint.source_namespace == OWNER_EMAIL_SOURCE_NAMESPACE:
                # A reply-by-e-mail queued before the channel was retired: it
                # can no longer prove which request it answered, and there is
                # nowhere to send a reply, so it decides nothing.
                return WorkflowResult(
                    status=WorkflowRunStatus.SUCCEEDED,
                    detail="e-mail replies no longer decide bookings",
                )
            outcome = await decide_booking(unit_of_work, message.business_id, decision, self._now())
        return self._result(message, outcome.text)

    @staticmethod
    def _result(message: NormalizedOwnerMessage, text: str) -> WorkflowResult:
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(_decision_reply(message, text),),
        )


class SendIntakeCustomerEmailService:
    """Delivers one customer email verbatim — decline notices and booking
    confirmations are composed before they are queued."""

    def __init__(self, delivery: CustomerQuoteDeliveryPort) -> None:
        self._delivery = delivery

    async def send(self, business_id: BusinessId, payload: Mapping[str, object]) -> None:
        email = intake_customer_email_request(business_id, payload)
        receipt = await self._delivery.deliver(
            CustomerDeliveryRequest(
                business_id=business_id,
                recipient=CustomerRecipient(
                    address=email.to, address_kind=RecipientAddressKind.EMAIL
                ),
                idempotency_key=email.idempotency_key,
                subject=email.subject,
                body_text=email.body,
            )
        )
        if receipt.status is DeliveryStatus.FAILED:
            raise IntakeDeliveryError(receipt.detail or "customer email failed")


class SendOwnerEmailService:
    """Delivers one owner notice e-mail as text + HTML; the layout is
    rendered before the command is queued."""

    def __init__(self, delivery: OwnerEmailPort) -> None:
        self._delivery = delivery

    async def send(self, business_id: BusinessId, payload: Mapping[str, object]) -> None:
        await self.deliver(intake_owner_email_request(business_id, payload))

    async def deliver(self, request: OwnerEmailRequest) -> None:
        receipt = await self._delivery.send(request)
        if receipt.status is DeliveryStatus.FAILED:
            raise IntakeDeliveryError(receipt.detail or "owner email failed")


class IntakeTextStatus(StrEnum):
    SENT = "sent"
    NO_CONSENT = "no_consent"


class SendIntakeCustomerTextService:
    """Delivers one customer SMS verbatim, only to a customer whose record
    says they consented to texts at the time of sending."""

    def __init__(
        self, texts: CustomerTextDeliveryPort, unit_of_work_factory: UnitOfWorkFactory
    ) -> None:
        self._texts = texts
        self._unit_of_work_factory = unit_of_work_factory

    async def send(
        self, business_id: BusinessId, payload: Mapping[str, object]
    ) -> IntakeTextStatus:
        phone = payload.get("phone")
        text = payload.get("text")
        key = payload.get("idempotency_key")
        if not all(isinstance(value, str) and value for value in (phone, text, key)):
            raise ValueError("intake text command payload is incomplete")
        if not await self._consented(business_id, payload.get("customer_id")):
            return IntakeTextStatus.NO_CONSENT
        receipt = await self._texts.send_text(
            CustomerTextRequest(
                business_id=business_id,
                phone_number=str(phone),
                text=str(text),
                idempotency_key=str(key),
            )
        )
        if receipt.status is DeliveryStatus.FAILED:
            raise IntakeDeliveryError(receipt.detail or "customer text failed")
        return IntakeTextStatus.SENT

    async def _consented(self, business_id: BusinessId, raw_customer_id: object) -> bool:
        if not isinstance(raw_customer_id, str):
            return False
        try:
            customer_id = CustomerId(UUID(raw_customer_id))
        except ValueError:
            return False
        async with self._unit_of_work_factory() as unit_of_work:
            customer = await unit_of_work.customers.get(business_id, customer_id)
            await unit_of_work.commit()
        return customer is not None and customer.sms_consent is True


class IntakeBookingEventResult(StrEnum):
    CONFIRMED = "confirmed"
    REROUTED = "rerouted"
    CANCELED = "canceled"
    IGNORED = "ignored"


@dataclass(frozen=True)
class IntakeBookingEventOutcome:
    result: IntakeBookingEventResult


class IntakeBookingEventService:
    """Applies a provider-confirmed booking event to a pending request.

    The scheduling-link path cannot pin an exact time, so the webhook is the
    authority on what actually landed on the calendar: an event at the
    approved time confirms the booking, an event at a different time puts the
    request back in front of the owner for a fresh decision, and a
    cancellation closes the request. ``booked_event_uri`` is the processed
    marker — a redelivered event is ignored, and a later different event is
    applied on its own.
    """

    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        *,
        decision_link_secret: str = "",
        decision_link_origin: str = "",
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._decision_email = OwnerDecisionEmail(
            secret=decision_link_secret,
            origin=decision_link_origin,
            now=now,
        )
        self._now = now

    async def handle(self, event: IntakeBookingEvent) -> IntakeBookingEventOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            conversation = await self._match(unit_of_work, event)
            if conversation is None:
                return IntakeBookingEventOutcome(IntakeBookingEventResult.IGNORED)
            recorded = conversation.booked_event_uri
            superseded = conversation.superseded_booking
            superseded_uri = None if superseded is None else superseded.event_uri
            now = self._now()
            if event.kind is BookingEventKind.CREATED:
                if recorded is not None or event.event_uri == superseded_uri:
                    # Redelivery or a second event for the same request:
                    # overwriting the recorded URI could orphan the event a
                    # decline is meant to cancel. A superseded event's
                    # redelivery is likewise not this request's booking.
                    return IntakeBookingEventOutcome(IntakeBookingEventResult.IGNORED)
            elif recorded != event.event_uri:
                if event.event_uri == superseded_uri:
                    # The customer (or Calendly) cancelled the old event
                    # while the reschedule is pending — drop the snapshot's
                    # event so an approve doesn't cancel it a second time.
                    result = await self._superseded_canceled(unit_of_work, conversation, now)
                    await unit_of_work.commit()
                    return IntakeBookingEventOutcome(result)
                # A cancellation only closes the request when it names the
                # event we already recorded — others are not ours to act on.
                return IntakeBookingEventOutcome(IntakeBookingEventResult.IGNORED)
            if event.kind is BookingEventKind.CANCELED:
                result = await self._canceled(unit_of_work, conversation, event, now)
            else:
                result = await self._created(unit_of_work, conversation, event, now)
            await unit_of_work.commit()
            return IntakeBookingEventOutcome(result)

    async def _match(
        self, unit_of_work: UnitOfWork, event: IntakeBookingEvent
    ) -> IntakeConversation | None:
        """Bind a webhook event to the request it belongs to — or reject it.

        Events carrying the reference our scheduling link embedded
        (utm_content) bind directly. Without one, an invitee e-mail match is
        only accepted when nothing rules the event out: a recorded booking
        event type that differs, or a start more than a day away from the
        requested slot (the link preselects the day, so a legitimate
        re-booked time always lands within it). E-mail alone cannot prove an
        unrelated appointment is this request's, and acting on one could
        later cancel an event we never created.
        """

        if event.reference is not None:
            return await unit_of_work.intake_conversations.lock_by_reference(
                event.business_id, event.reference
            )
        conversation = await unit_of_work.intake_conversations.lock_latest_by_invitee_email(
            event.business_id, event.invitee_email
        )
        if conversation is None or event.kind is not BookingEventKind.CREATED:
            return conversation
        expected_type = conversation.booking_event_type_uri
        if (
            expected_type is not None
            and event.event_type_uri is not None
            and event.event_type_uri != expected_type
        ):
            logger.info(
                "ignoring calendly event %s for %s: event type is not the booked kind",
                event.event_uri,
                conversation.reference,
            )
            return None
        requested = conversation.requested_slot_start
        if requested is not None and abs(event.start - requested) > timedelta(days=1):
            logger.info(
                "ignoring calendly event %s for %s: start is not on the requested day",
                event.event_uri,
                conversation.reference,
            )
            return None
        return conversation

    async def _created(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        event: IntakeBookingEvent,
        now: datetime,
    ) -> IntakeBookingEventResult:
        zone = _zone_of(await unit_of_work.businesses.get(conversation.business_id))
        if event.start == conversation.requested_slot_start:
            already_booked = conversation.booking_kind == BookingKind.BOOKED.value
            updated = conversation.with_updates(
                now,
                booked_event_uri=event.event_uri,
                booking_kind=BookingKind.BOOKED.value,
                reschedule_offered_at=None,
            )
            if conversation.state is IntakeState.APPROVED and not already_booked:
                await self._notify(
                    unit_of_work,
                    conversation,
                    event,
                    text=(
                        f"Booking {conversation.reference} is on the calendar: "
                        f"{conversation.collected.name or 'the customer'} confirmed "
                        f"{format_slot_label(event.start, zone)}."
                    ),
                )
            await unit_of_work.intake_conversations.save(updated)
            return IntakeBookingEventResult.CONFIRMED
        # The customer picked a different time than the approved/requested
        # slot: put the request back in front of the owner with the time that
        # actually landed. Approving keeps the event (arrange reconciles
        # through find_booking); declining cancels it.
        original = (
            format_slot_label(conversation.requested_slot_start, zone)
            if conversation.requested_slot_start is not None
            else "their requested time"
        )
        updated = conversation.with_updates(
            now,
            state=IntakeState.AWAITING_OWNER,
            requested_slot_start=event.start,
            requested_slot_end=event.end,
            booking_kind=None,
            booked_event_uri=event.event_uri,
            decision_at=None,
            decision_reason=None,
            # The event that landed becomes the request; anything it replaced
            # is still cancelled only on approve. Re-stamping ``owner_notified_at``
            # makes decision links minted for the earlier request stale.
            reschedule_offered_at=None,
            owner_notified_at=now,
        )
        await self._notify(
            unit_of_work,
            conversation,
            event,
            text=(
                f"Booking {conversation.reference} — "
                f"{conversation.collected.name or 'the customer'} booked "
                f"{format_slot_label(event.start, zone)} instead of the requested "
                f"{original}. Reply `approve booking {conversation.reference}` "
                f"to keep it or `decline booking {conversation.reference} <reason>` to cancel."
            ),
            email_actions=self._decision_email.actions(updated),
        )
        await unit_of_work.intake_conversations.save(updated)
        return IntakeBookingEventResult.REROUTED

    async def _superseded_canceled(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        now: datetime,
    ) -> IntakeBookingEventResult:
        superseded = conversation.superseded_booking
        if superseded is not None and superseded.event_uri:
            updated = conversation.with_updates(
                now,
                superseded_booking=superseded.model_copy(update={"event_uri": None}),
            )
            await unit_of_work.intake_conversations.save(updated)
        return IntakeBookingEventResult.CANCELED

    async def _canceled(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        event: IntakeBookingEvent,
        now: datetime,
    ) -> IntakeBookingEventResult:
        superseded = conversation.superseded_booking
        if superseded is not None and superseded.event_uri:
            # The new event died on the calendar while the owner was still
            # deciding: the old booking goes back in force.
            zone = _zone_of(await unit_of_work.businesses.get(conversation.business_id))
            updated = conversation.with_updates(
                now,
                state=IntakeState.APPROVED,
                requested_slot_start=superseded.slot_start,
                requested_slot_end=superseded.slot_end,
                booked_event_uri=superseded.event_uri,
                booking_kind=superseded.booking_kind,
                booking_link=superseded.booking_link,
                booking_attempted_at=superseded.booking_attempted_at,
                booking_event_type_uri=superseded.booking_event_type_uri,
                superseded_booking=None,
                decision_at=None,
                decision_reason=None,
            )
            await self._notify(
                unit_of_work,
                conversation,
                event,
                text=(
                    f"Booking {conversation.reference} — the new "
                    f"{format_slot_label(event.start, zone)} event was canceled on "
                    f"Calendly; the original {superseded.slot_label(zone)} booking "
                    "still stands."
                ),
            )
            await unit_of_work.intake_conversations.save(updated)
            return IntakeBookingEventResult.CANCELED
        updated = conversation.with_updates(
            now, state=IntakeState.CLOSED, booked_event_uri=event.event_uri
        )
        await self._notify(
            unit_of_work,
            conversation,
            event,
            text=(
                f"Booking {conversation.reference} was canceled on Calendly — "
                "the event is off the calendar."
            ),
        )
        await unit_of_work.intake_conversations.save(updated)
        return IntakeBookingEventResult.CANCELED

    async def _notify(
        self,
        unit_of_work: UnitOfWork,
        conversation: IntakeConversation,
        event: IntakeBookingEvent,
        *,
        text: str,
        email_actions: tuple[OwnerEmailAction, ...] = (),
    ) -> None:
        notified = await enqueue_intake_owner_notice(
            unit_of_work,
            conversation.business_id,
            correlation_id=f"intake_booking_event:{event.kind.value}:{event.event_uri}",
            text=text,
            email_actions=email_actions,
        )
        if not notified:
            logger.warning(
                "booking event for request %s stored without an owner notice",
                conversation.reference,
            )


class DecisionLinkPage(StrEnum):
    """The page a booking-decision link renders."""

    CONFIRM = "confirm"
    APPLIED = "applied"
    ALREADY_DECIDED = "already_decided"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass(frozen=True)
class DecisionLinkView:
    """What the public decision endpoint renders: a confirmation form, or a
    result line after the decision ran (or could not)."""

    page: DecisionLinkPage
    title: str
    body: str
    action: BookingDecisionAction | None = None


class IntakeDecisionLinkService:
    """The signed approve/decline links in the owner notification e-mail.

    A GET only previews (``CONFIRM``) so a mail client's prefetch never acts;
    the POST runs the shared ``decide_booking`` transition, so the customer
    e-mail/text and any follow-up are identical to the channel command. The
    token carries the stamp of the request it was minted for: a link for a
    request that was rescheduled, decided or superseded can never act again.
    """

    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        *,
        secret: str,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._secret = secret
        self._now = now

    async def preview(self, token: str) -> DecisionLinkView:
        resolved = await self._resolve(token)
        if isinstance(resolved, DecisionLinkView):
            return resolved
        link, conversation = resolved
        verb = "Approve" if link.action is BookingDecisionAction.APPROVE else "Decline"
        name = conversation.collected.name or "the customer"
        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get(conversation.business_id)
            await unit_of_work.commit()
        slot = (
            format_slot_label(conversation.requested_slot_start, _zone_of(business))
            if conversation.requested_slot_start is not None
            else "the requested time"
        )
        return DecisionLinkView(
            page=DecisionLinkPage.CONFIRM,
            title=f"{verb} booking #{conversation.reference}?",
            body=(
                f"{name} is asking for {slot}. This link works once; the "
                "customer is told the outcome the same way as a channel reply."
            ),
            action=link.action,
        )

    async def decide(self, token: str) -> DecisionLinkView:
        resolved = await self._resolve(token)
        if isinstance(resolved, DecisionLinkView):
            return resolved
        link, _conversation = resolved
        decision = BookingDecision(reference=link.reference, action=link.action, reason=None)
        async with self._unit_of_work_factory() as unit_of_work:
            outcome = await decide_booking(
                unit_of_work,
                link.business_id,
                decision,
                self._now(),
                request_epoch=link.request_epoch,
            )
            if outcome.applied:
                # Parity with the channel command, which replies in the owner
                # thread: record the decision there too (and on the
                # notification e-mail), so the e-mail link leaves a trail.
                await enqueue_intake_owner_notice(
                    unit_of_work,
                    link.business_id,
                    correlation_id=(f"intake_decided:{link.conversation_id}:{link.request_epoch}"),
                    text=outcome.text,
                )
                await unit_of_work.commit()
        if outcome.applied:
            done = (
                "approved — the customer is being notified"
                if link.action is BookingDecisionAction.APPROVE
                else "declined — the customer is being notified"
            )
            return DecisionLinkView(
                page=DecisionLinkPage.APPLIED,
                title=f"Booking {done}",
                body=outcome.text,
            )
        return DecisionLinkView(
            page=DecisionLinkPage.ALREADY_DECIDED,
            title="This request was already decided",
            body=outcome.text,
        )

    async def _resolve(
        self, token: str
    ) -> tuple[BookingDecisionLink, IntakeConversation] | DecisionLinkView:
        try:
            link = parse_booking_decision_link_token(self._secret, token, now=self._now())
        except ExpiredDecisionLinkError:
            return DecisionLinkView(
                page=DecisionLinkPage.EXPIRED,
                title="This link has expired",
                body="Use your owner channel to approve or decline the booking.",
            )
        except (InvalidDecisionLinkError, ValueError):
            return DecisionLinkView(
                page=DecisionLinkPage.INVALID,
                title="This link isn't valid",
                body="Check the link was copied in full, or decide in your owner channel.",
            )
        async with self._unit_of_work_factory() as unit_of_work:
            conversation = await unit_of_work.intake_conversations.lock_by_reference(
                link.business_id, link.reference
            )
            if (
                conversation is None
                or conversation.conversation_id != link.conversation_id
                or not link.matches_request(conversation)
            ):
                return DecisionLinkView(
                    page=DecisionLinkPage.STALE,
                    title="This link is for an earlier request",
                    body=(
                        "The request was updated since this e-mail — use the "
                        "newest booking e-mail's link."
                    ),
                )
            if conversation.state is not IntakeState.AWAITING_OWNER:
                return DecisionLinkView(
                    page=DecisionLinkPage.ALREADY_DECIDED,
                    title="This request was already decided",
                    body=f"Booking {link.reference} is no longer waiting on a decision.",
                )
            return link, conversation


class CancelIntakeBookingService:
    """Cancels the calendar event recorded on a declined request."""

    def __init__(self, *, availability: AvailabilityPort | None) -> None:
        self._availability = availability

    async def cancel(self, business_id: BusinessId, event_uri: str) -> None:
        if self._availability is None:
            raise IntakeAvailabilityError("no availability provider is configured")
        await self._availability.cancel_booking(business_id, event_uri)


def _zone_of(business: BusinessRecord | None) -> tzinfo | None:
    return business_zone(business.timezone if business is not None else None)
