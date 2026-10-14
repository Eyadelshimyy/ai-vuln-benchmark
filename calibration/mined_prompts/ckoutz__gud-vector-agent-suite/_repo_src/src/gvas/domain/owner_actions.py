"""Owner decisions shared by every surface the owner acts from.

``approve booking <ref>`` in the owner channel, the Approve button on the
owner dashboard, and the one-click links in the notification e-mail all run
the same transition here, so the customer sees the same outcome whichever
one the owner used.
"""

from datetime import datetime, tzinfo

from gvas.domain.customer_linking import link_quote_customer
from gvas.domain.customers import (
    PortalLoginEmailRequest,
    hash_portal_token,
    new_portal_token,
    portal_login_email_command,
    portal_login_url,
)
from gvas.domain.enums import QuoteStatus
from gvas.domain.identifiers import BusinessId, MessageKey
from gvas.domain.intake import (
    BookingDecision,
    BookingDecisionAction,
    IntakeConversation,
    IntakeCustomerEmail,
    IntakeState,
    SupersededBooking,
    format_slot_label,
    intake_booking_arrange_command,
    intake_booking_cancel_command,
    intake_customer_email_command,
)
from gvas.domain.owner import OWNER_LOGIN_TOKEN_TTL, OwnerLoginToken
from gvas.domain.quotes import InvalidQuoteTransitionError, Quote, quote_delivery_command
from gvas.domain.repositories import BusinessRecord, UnitOfWork
from gvas.domain.time_zones import business_zone


class BookingDecisionOutcome:
    """The reply text and whether the decision changed anything."""

    def __init__(self, text: str, *, applied: bool) -> None:
        self.text = text
        self.applied = applied


async def decide_booking(
    unit_of_work: UnitOfWork,
    business_id: BusinessId,
    decision: BookingDecision,
    now: datetime,
    *,
    request_epoch: int | None = None,
) -> BookingDecisionOutcome:
    """Approve or decline one website booking request of ``business_id``.

    Deciding the same reference twice is a no-op with a clear reply; an
    unknown reference gets one clear reply. ``request_epoch`` is set by the
    e-mail decision links: it must equal the stamp the current request was
    notified at, so a link written for a superseded request cannot act on
    its replacement. Commits only when applied.
    """

    reference = decision.reference
    conversation = await unit_of_work.intake_conversations.lock_by_reference(business_id, reference)
    if conversation is None:
        return BookingDecisionOutcome(f"I can't find booking {reference}.", applied=False)
    if request_epoch is not None and not _request_stamp_matches(conversation, request_epoch):
        return BookingDecisionOutcome(
            f"Booking {reference} was updated since this link was sent — "
            "use the newest booking e-mail's link.",
            applied=False,
        )
    if decision.action is BookingDecisionAction.CANCEL:
        if request_epoch is not None:
            return BookingDecisionOutcome("Links can't cancel a booking.", applied=False)
        return await _cancel_booking(unit_of_work, business_id, conversation, now)
    if conversation.state is IntakeState.APPROVED:
        return BookingDecisionOutcome(f"Booking {reference} is already approved.", applied=False)
    if conversation.state is IntakeState.DECLINED:
        return BookingDecisionOutcome(f"Booking {reference} was already declined.", applied=False)
    if conversation.state is not IntakeState.AWAITING_OWNER:
        return BookingDecisionOutcome(
            f"Booking {reference} isn't waiting on a decision.", applied=False
        )
    if decision.action is BookingDecisionAction.APPROVE:
        return await _approve_booking(
            unit_of_work, conversation, now, await _business_zone(unit_of_work, business_id)
        )
    return await _decline_booking(unit_of_work, business_id, conversation, decision, now)


async def _business_zone(unit_of_work: UnitOfWork, business_id: BusinessId) -> tzinfo | None:
    business = await unit_of_work.businesses.get(business_id)
    return business_zone(business.timezone if business is not None else None)


def _request_stamp_matches(conversation: IntakeConversation, request_epoch: int) -> bool:
    stamp = conversation.owner_notified_at
    return stamp is not None and int(stamp.timestamp()) == request_epoch


async def _approve_booking(
    unit_of_work: UnitOfWork,
    conversation: IntakeConversation,
    now: datetime,
    zone: tzinfo | None,
) -> BookingDecisionOutcome:
    if conversation.requested_slot_start is not None and conversation.requested_slot_start <= now:
        return BookingDecisionOutcome(
            f"Booking {conversation.reference}'s requested time has already "
            "passed — decline it or line up a new time with the customer.",
            applied=False,
        )
    superseded = conversation.superseded_booking
    updated = conversation.with_updates(
        now,
        state=IntakeState.APPROVED,
        decision_at=now,
        superseded_booking=None,
        reschedule_offered_at=None,
    )
    await unit_of_work.intake_conversations.save(updated)
    if superseded is not None and superseded.event_uri:
        # The new time wins: only now is the booking it replaced cancelled.
        await unit_of_work.outbox.enqueue(
            intake_booking_cancel_command(updated, superseded.event_uri)
        )
    await unit_of_work.outbox.enqueue(intake_booking_arrange_command(updated))
    await unit_of_work.commit()
    name = updated.collected.name or "the customer"
    slot = (
        format_slot_label(updated.requested_slot_start, zone)
        if updated.requested_slot_start is not None
        else "their requested time"
    )
    text = f"Approved booking {updated.reference} — arranging {slot} for {name}."
    if superseded is not None:
        previous = f" The old booking for {superseded.slot_label(zone)} is being canceled."
        text = f"{text}{previous}" if superseded.event_uri else f"{text} The old request is closed."
    return BookingDecisionOutcome(text, applied=True)


async def _decline_booking(
    unit_of_work: UnitOfWork,
    business_id: BusinessId,
    conversation: IntakeConversation,
    decision: BookingDecision,
    now: datetime,
) -> BookingDecisionOutcome:
    superseded = conversation.superseded_booking
    if superseded is not None:
        return await _decline_reschedule(
            unit_of_work, business_id, conversation, decision, superseded, now
        )
    updated = conversation.with_updates(
        now,
        state=IntakeState.DECLINED,
        decision_at=now,
        decision_reason=decision.reason,
    )
    await unit_of_work.intake_conversations.save(updated)
    business = await unit_of_work.businesses.get(business_id)
    email = updated.collected.email
    notified = "the customer has been notified"
    if email:
        booking_link = business.calendly_url if business is not None else None
        await unit_of_work.outbox.enqueue(
            intake_customer_email_command(
                IntakeCustomerEmail(
                    business_id=business_id,
                    to=email,
                    subject="About your requested appointment",
                    body=decline_body(updated, booking_link),
                    idempotency_key=f"intake_decline:{conversation.conversation_id}",
                )
            )
        )
    else:
        notified = "no customer email was collected, so nothing was sent"
    if updated.booked_event_uri:
        await unit_of_work.outbox.enqueue(intake_booking_cancel_command(updated))
        notified = f"{notified}, and the Calendly event is being canceled"
    await unit_of_work.commit()
    return BookingDecisionOutcome(
        f"Declined booking {updated.reference}; {notified}.", applied=True
    )


async def _decline_reschedule(
    unit_of_work: UnitOfWork,
    business_id: BusinessId,
    conversation: IntakeConversation,
    decision: BookingDecision,
    superseded: SupersededBooking,
    now: datetime,
) -> BookingDecisionOutcome:
    """Declining a request to *move* a booking keeps the original one: the
    snapshot is put back in force and the customer hears their time stands."""

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
        reschedule_offered_at=None,
        decision_at=now,
        decision_reason=decision.reason,
    )
    await unit_of_work.intake_conversations.save(updated)
    business = await unit_of_work.businesses.get(business_id)
    email = updated.collected.email
    notified = "the customer has been notified"
    kept = superseded.slot_label(business_zone(business.timezone if business else None))
    if email:
        business_name = (
            "" if business is None else (business.display_name or business.name)
        ) or "the business"
        await unit_of_work.outbox.enqueue(
            intake_customer_email_command(
                IntakeCustomerEmail(
                    business_id=business_id,
                    to=email,
                    subject="Your appointment stands",
                    body=(
                        f"Thanks for your patience — the owner couldn't take the "
                        f"new time. Your {business_name} appointment stays at "
                        f"{kept}. If you need a different time, reply here and "
                        "we'll sort it out."
                    ),
                    idempotency_key=(
                        f"intake_restore:{conversation.conversation_id}:{int(now.timestamp())}"
                    ),
                )
            )
        )
    else:
        notified = "no customer email was collected, so nothing was sent"
    await unit_of_work.commit()
    return BookingDecisionOutcome(
        f"Declined booking {updated.reference} — the original time {kept} stands; {notified}.",
        applied=True,
    )


async def _cancel_booking(
    unit_of_work: UnitOfWork,
    business_id: BusinessId,
    conversation: IntakeConversation,
    now: datetime,
) -> BookingDecisionOutcome:
    """``cancel booking <ref>``: the owner drops a live booking — the way a
    cancel asked for from an unverified chat goes ahead once the owner has
    checked with the customer. Every calendar event it holds is cancelled."""

    reference = conversation.reference
    if not conversation.has_live_booking:
        return BookingDecisionOutcome(
            f"Booking {reference} isn't active, so there's nothing to cancel.", applied=False
        )
    superseded = conversation.superseded_booking
    zone = await _business_zone(unit_of_work, business_id)
    if superseded is not None:
        standing: str | None = superseded.slot_label(zone)
    elif conversation.requested_slot_start is not None:
        standing = format_slot_label(conversation.requested_slot_start, zone)
    else:
        standing = None
    updated = conversation.with_updates(
        now,
        state=IntakeState.CLOSED,
        decision_at=now,
        superseded_booking=None,
        reschedule_offered_at=None,
    )
    await unit_of_work.intake_conversations.save(updated)
    events = {conversation.booked_event_uri, superseded.event_uri if superseded else None}
    for event_uri in sorted(uri for uri in events if uri):
        await unit_of_work.outbox.enqueue(intake_booking_cancel_command(updated, event_uri))
    business = await unit_of_work.businesses.get(business_id)
    email = updated.collected.email
    notified = "the customer has been notified"
    if email:
        business_name = (
            "" if business is None else (business.display_name or business.name)
        ) or "the business"
        when = f" for {standing}" if standing else ""
        await unit_of_work.outbox.enqueue(
            intake_customer_email_command(
                IntakeCustomerEmail(
                    business_id=business_id,
                    to=email,
                    subject="Your appointment is canceled",
                    body=(
                        f"Your {business_name} appointment{when} has been canceled. "
                        "If that's a mistake or you'd like a new time, reply here."
                    ),
                    idempotency_key=f"intake_owner_cancel:{conversation.conversation_id}",
                )
            )
        )
    else:
        notified = "no customer email was collected, so nothing was sent"
    if events - {None}:
        notified = f"{notified}, and the calendar event is being canceled"
    await unit_of_work.commit()
    return BookingDecisionOutcome(f"Canceled booking {reference}; {notified}.", applied=True)


def decline_body(conversation: IntakeConversation, booking_link: str | None) -> str:
    lines = [
        "Thanks for reaching out. The owner reviewed your request and can't "
        "take it on at the requested time.",
    ]
    if conversation.decision_reason:
        lines.append(f"Reason: {conversation.decision_reason}")
    if booking_link:
        lines.append(f"If another time works, you can pick one directly: {booking_link}")
    else:
        lines.append("If another time works, reply here and we'll sort it out.")
    return "\n".join(lines)


async def approve_quote(
    unit_of_work: UnitOfWork, quote: Quote, message_key: MessageKey, now: datetime
) -> Quote:
    """The owner's OK: link the recipient as a customer and queue delivery —
    exactly what replying ``approve`` does. The caller commits."""

    approved = quote.approve(message_key, now)
    approved, _ = await link_quote_customer(unit_of_work, approved, now)
    await unit_of_work.quotes.save(approved, expected_version=quote.version)
    await unit_of_work.outbox.enqueue(quote_delivery_command(approved))
    return approved


async def reject_quote(
    unit_of_work: UnitOfWork, quote: Quote, message_key: MessageKey, now: datetime
) -> Quote:
    if quote.status is not QuoteStatus.AWAITING_APPROVAL:
        raise InvalidQuoteTransitionError(f"cannot reject quote in {quote.status}")
    rejected = quote.reject(message_key, now)
    await unit_of_work.quotes.save(rejected, expected_version=quote.version)
    return rejected


async def issue_owner_login(
    unit_of_work: UnitOfWork, business: BusinessRecord, email: str, now: datetime
) -> None:
    """Mint one owner magic link and queue it to ``email``. The link has the
    same shape as a customer's so one sign-in page serves both."""

    if not business.site_url:
        return
    raw = new_portal_token()
    token_hash = hash_portal_token(raw)
    expires_at = now + OWNER_LOGIN_TOKEN_TTL
    await unit_of_work.owner_login_tokens.add(
        OwnerLoginToken(
            token_hash=token_hash,
            business_id=business.business_id,
            email=email,
            expires_at=expires_at,
            created_at=now,
        )
    )
    request = PortalLoginEmailRequest(
        business_id=business.business_id,
        to=email,
        business_display_name=business.display_name or business.name,
        login_url=portal_login_url(business.site_url, raw),
        idempotency_key=f"owner-login:{token_hash}",
        expires_at=expires_at,
    )
    await unit_of_work.outbox.enqueue(portal_login_email_command(request, token_hash=token_hash))


__all__ = [
    "BookingDecisionOutcome",
    "approve_quote",
    "decide_booking",
    "decline_body",
    "issue_owner_login",
    "reject_quote",
]
