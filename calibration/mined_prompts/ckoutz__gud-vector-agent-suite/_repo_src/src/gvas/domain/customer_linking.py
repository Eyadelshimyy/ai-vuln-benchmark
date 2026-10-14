"""Customer identity bookkeeping shared by the quote workflow and the portal.

A customer is ``(business, lowercased e-mail)``. Quotes addressed to an e-mail
are linked to that customer whenever they pass through approval or delivery,
and the portal links any older ones the first time the customer signs in — so
no historical backfill is ever required.
"""

from datetime import datetime

from gvas.domain.customers import CustomerRecord
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import IntakeCustomerEmail, owner_notice_email_command
from gvas.domain.messages import OutboundOwnerMessage, TextPart
from gvas.domain.outbox import owner_reply_command
from gvas.domain.owner_email import (
    OWNER_CHANNEL_FOOTER,
    OwnerEmailAction,
    OwnerEmailContent,
    owner_notice_content,
    render_owner_email_html,
    render_owner_email_text,
)
from gvas.domain.quotes import Quote
from gvas.domain.repositories import UnitOfWork


async def enqueue_quote_owner_notice(
    unit_of_work: UnitOfWork, quote: Quote, *, correlation_id: str, text: str
) -> bool:
    """Queue ``text`` for the owner in the conversation where ``quote`` was
    requested, anchored to the message that started it — the same routing an
    owner reply takes, so it lands in the same owner conversation.

    Returns False when the anchor is gone; True when queued or when the same
    ``correlation_id`` was queued before (the notice is idempotent).
    """

    source = await unit_of_work.inbound_messages.find_by_key(
        quote.business_id, quote.conversation_id, quote.source_message_key
    )
    if source is None:
        return False
    await enqueue_owner_email_copy(
        unit_of_work, quote.business_id, correlation_id=correlation_id, text=text
    )
    existing = await unit_of_work.outbound_messages.find_by_correlation(
        quote.business_id, quote.conversation_id, correlation_id
    )
    if existing is not None:
        return True
    message = OutboundOwnerMessage(
        business_id=quote.business_id,
        conversation_ref=quote.conversation_ref,
        parts=(TextPart(text=text),),
        correlation_id=correlation_id,
    )
    outbound_message_id = await unit_of_work.outbound_messages.create(
        message, quote.conversation_id, source.inbound_message_id
    )
    await unit_of_work.outbox.enqueue(owner_reply_command(quote.business_id, outbound_message_id))
    return True


async def enqueue_intake_owner_notice(
    unit_of_work: UnitOfWork,
    business_id: BusinessId,
    *,
    correlation_id: str,
    text: str,
    email: OwnerEmailContent | None = None,
    email_actions: tuple[OwnerEmailAction, ...] = (),
) -> bool:
    """Queue ``text`` for the owner in their latest channel conversation,
    plus the notification e-mail copy.

    Intake conversations start on the web, so there is no owner conversation
    to inherit — see ``enqueue_owner_thread_notice``. Idempotent on
    ``correlation_id``; False when nothing reached the owner — no owner
    thread to anchor to *and* no notification e-mail. ``email`` (the
    structured layout) and ``email_actions`` (the one-click decision buttons)
    shape the e-mail copy only.
    """

    emailed = await enqueue_owner_email_copy(
        unit_of_work,
        business_id,
        correlation_id=correlation_id,
        text=text,
        content=email,
        actions=email_actions,
    )
    threaded = await enqueue_owner_thread_notice(
        unit_of_work, business_id, correlation_id=correlation_id, text=text
    )
    # The e-mail copy alone still lets the owner act (decision links), so it
    # counts as notified.
    return threaded or emailed


async def enqueue_owner_thread_notice(
    unit_of_work: UnitOfWork, business_id: BusinessId, *, correlation_id: str, text: str
) -> bool:
    """Queue ``text`` in the newest owner thread across the connected chat
    channels, anchored to the business's most recent inbound message there.
    Reply-only channels (e-mail) never become the anchor. Idempotent on
    ``correlation_id``; False when there is no owner thread."""

    source = await unit_of_work.inbound_messages.find_latest_for_business(business_id)
    if source is None:
        return False
    existing = await unit_of_work.outbound_messages.find_by_correlation(
        business_id, source.conversation_id, correlation_id
    )
    if existing is not None:
        return True
    message = OutboundOwnerMessage(
        business_id=business_id,
        conversation_ref=source.message.conversation_ref,
        parts=(TextPart(text=text),),
        correlation_id=correlation_id,
    )
    outbound_message_id = await unit_of_work.outbound_messages.create(
        message, source.conversation_id, source.inbound_message_id
    )
    await unit_of_work.outbox.enqueue(owner_reply_command(business_id, outbound_message_id))
    return True


async def enqueue_owner_email_copy(
    unit_of_work: UnitOfWork,
    business_id: BusinessId,
    *,
    correlation_id: str,
    text: str,
    content: OwnerEmailContent | None = None,
    actions: tuple[OwnerEmailAction, ...] = (),
) -> bool:
    """Queue a copy of an owner notice to the business's ``notification_email``.

    Every notice renders through the one owner e-mail layout (text + HTML);
    ``content`` is the notice's structured form when it has one, otherwise
    ``text`` is laid out generically. Idempotent on ``correlation_id``; False when no notification
    e-mail is configured.
    """

    business = await unit_of_work.businesses.get(business_id)
    if business is None or business.notification_email is None:
        return False
    name = business.display_name or business.name
    layout = (content or owner_notice_content(text)).model_copy(update={"business_name": name})
    layout = layout.with_actions(actions, "Each button works once.")
    if layout.footer is None:
        layout = layout.with_footer(OWNER_CHANNEL_FOOTER)
    email = IntakeCustomerEmail(
        business_id=business_id,
        to=business.notification_email,
        subject=layout.email_subject(),
        body=render_owner_email_text(layout),
        html=render_owner_email_html(layout),
        idempotency_key=f"owner_copy:{business_id}:{correlation_id}",
    )
    await unit_of_work.outbox.enqueue(owner_notice_email_command(email))
    return True


async def link_quote_customer(
    unit_of_work: UnitOfWork, quote: Quote, now: datetime
) -> tuple[Quote, CustomerRecord | None]:
    """Upsert the recipient as a customer and return the quote linked to it.

    Quotes going to a phone number only have no customer identity and come
    back unchanged. The returned quote carries a bumped version when it
    changed; the caller decides whether and how to persist it.
    """

    email = quote.recipient_email
    draft = quote.draft
    if email is None or draft is None:
        return quote, None
    customer = await unit_of_work.customers.upsert(
        quote.business_id,
        email,
        display_name=draft.recipient.display_name,
        phone=draft.recipient.phone_number,
        now=now,
    )
    return quote.link_customer(customer.customer_id, now), customer


__all__ = [
    "enqueue_intake_owner_notice",
    "enqueue_owner_email_copy",
    "enqueue_owner_thread_notice",
    "enqueue_quote_owner_notice",
    "link_quote_customer",
]
