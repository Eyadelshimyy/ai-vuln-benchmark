"""Owner assistant: plain-language answers to an owner's questions, read only.

The business is the one the owner's message came from; the model never picks
it. The model may only call the read tools below, each runs here against that
one business, and none of them can change anything. Text customers wrote
(names, notes, chat details) goes back marked as ``customer_wrote`` data.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gvas.domain.enums import CustomerQuoteStatus, QuoteStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import IntakeConversation, IntakeState
from gvas.domain.invoices import InvoiceStatus, invoice_status
from gvas.domain.owner import CalendarEventSource
from gvas.domain.owner_assistant import (
    OWNER_ASSISTANT_REPLY_MAX_CHARS,
    BusinessCalendarPort,
    OwnerAssistantError,
    OwnerAssistantPort,
    OwnerAssistantRequest,
    OwnerAssistantSetupError,
    OwnerAssistantTool,
)
from gvas.domain.payments import month_totals
from gvas.domain.quotes import Quote
from gvas.domain.repositories import BusinessRecord, UnitOfWork
from gvas.domain.usage import UsageCeilingGuard, UsageKind

logger = logging.getLogger(__name__)

DEFAULT_ZONE = "America/Los_Angeles"
# The dashboard's own list limits, so the answers match what it shows.
QUOTE_LIST_LIMIT = 500
BOOKING_LIST_LIMIT = 200
SCHEDULE_MAX_DAYS = 7
LIST_MAX_ITEMS = 15
CUSTOMER_MATCH_MAX = 5
QUOTE_FILTERS = ("all", "waiting_for_owner", "sent", "accepted", "paid", "declined")

UNAVAILABLE_REPLY = "Sorry, I can't answer right now. The dashboard has everything in the meantime."
SETUP_REPLY = (
    "Sorry, I can't answer texts right now: the AI service is refusing my requests, "
    "a setup problem on Güd Vector's side. The dashboard has everything in the meantime."
)
LIMIT_REPLY = (
    "I've reached this month's limit for plain-language answers. "
    "The dashboard still has everything."
)

TOOLS: tuple[OwnerAssistantTool, ...] = (
    OwnerAssistantTool(
        "schedule",
        "Bookings, booking requests and calendar events from start_day for `days` days "
        f"(1-{SCHEDULE_MAX_DAYS}), in the business's local time.",
        {
            "start_day": {"type": "string", "description": "Local date, YYYY-MM-DD."},
            "days": {"type": "integer", "minimum": 1, "maximum": SCHEDULE_MAX_DAYS},
        },
    ),
    OwnerAssistantTool(
        "needs_you",
        "What is waiting for the owner: quotes to approve and booking requests to approve or "
        "decline.",
    ),
    OwnerAssistantTool(
        "money",
        "Paid this month and what is outstanding (open invoices, or accepted quotes not paid "
        "yet when the business doesn't use invoices).",
    ),
    OwnerAssistantTool(
        "find_customer",
        "Look up customers by name, e-mail or phone: contact details, quotes and bookings.",
        {"query": {"type": "string", "description": "Part of a name, e-mail or phone."}},
    ),
    OwnerAssistantTool(
        "quotes",
        f"The newest quotes (up to {LIST_MAX_ITEMS}) with their status.",
        {"status": {"type": "string", "enum": list(QUOTE_FILTERS)}},
    ),
)


def money_text(minor: int, currency: str = "usd") -> str:
    amount = f"${minor / 100:,.2f}".replace(".00", "")
    return amount if currency.lower() == "usd" else f"{amount} {currency.upper()}"


def totals_text(amounts: Iterable[tuple[int, str]]) -> str:
    """One total per currency, never added across currencies."""

    sums: dict[str, int] = {}
    for minor, currency in amounts:
        sums[currency.lower()] = sums.get(currency.lower(), 0) + minor
    return " + ".join(money_text(minor, currency) for currency, minor in sums.items()) or (
        money_text(0)
    )


def business_zone(business: BusinessRecord) -> tzinfo:
    try:
        return ZoneInfo(business.timezone or DEFAULT_ZONE)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_ZONE)


def _when(moment: datetime | None, zone: tzinfo) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(zone).strftime("%a %b %-d, %-I:%M %p %Z")


def _wrote(text: str | None) -> dict[str, str] | None:
    return {"customer_wrote": text} if text else None


def _quote_state(quote: Quote) -> str:
    if quote.customer_status is CustomerQuoteStatus.PAID:
        return "paid"
    if quote.customer_status is CustomerQuoteStatus.ACCEPTED:
        return "accepted, not paid yet"
    if quote.customer_status is CustomerQuoteStatus.DECLINED:
        return "declined by the customer"
    if quote.status is QuoteStatus.AWAITING_APPROVAL:
        return "waiting for the owner's approval"
    if quote.status is QuoteStatus.REJECTED:
        return "rejected by the owner"
    if quote.status is QuoteStatus.DELIVERED:
        return "viewed by the customer" if quote.customer_status else "sent to the customer"
    return "being prepared"


def _quote_filter(quote: Quote, status: str) -> bool:
    state = _quote_state(quote)
    return {
        "all": True,
        "waiting_for_owner": state.startswith("waiting"),
        "sent": state.startswith(("sent", "viewed")),
        "accepted": state.startswith("accepted"),
        "paid": state == "paid",
        "declined": state.startswith(("declined", "rejected")),
    }.get(status, True)


def _quote_row(quote: Quote, zone: tzinfo) -> dict[str, Any]:
    draft = quote.draft
    assert draft is not None
    return {
        "customer": _wrote(draft.recipient.display_name or draft.recipient.address),
        "total": money_text(draft.total_minor, draft.currency),
        "billing": draft.billing.value
        + (f" every {draft.interval.value}" if draft.interval is not None else ""),
        "items": [_wrote(item.description) for item in draft.line_items[:3]],
        "status": _quote_state(quote),
        "created": _when(quote.created_at, zone),
    }


def _booking_row(conversation: IntakeConversation, zone: tzinfo) -> dict[str, Any]:
    state = {
        IntakeState.AWAITING_OWNER: "waiting for the owner to approve or decline",
        IntakeState.APPROVED: "booked",
        IntakeState.DECLINED: "declined",
    }.get(conversation.state, conversation.state.value)
    collected = conversation.collected
    return {
        "reference": conversation.reference,
        "customer": _wrote(collected.name),
        "phone": collected.phone,
        "email": collected.email,
        "address": _wrote(collected.address),
        "requested_time": _when(conversation.requested_slot_start, zone),
        "status": state,
        "job": _wrote(collected.details),
        "notes": _wrote(collected.notes),
    }


_EVENT_STATUS = {
    CalendarEventSource.BOOKING: "booked",
    CalendarEventSource.REQUEST: "requested, waiting for the owner to approve or decline",
    CalendarEventSource.CALENDAR: "on the owner's own calendar",
}


class OwnerReadTools:
    """The assistant's fixed read tools, each bound to one business."""

    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        owner: BusinessCalendarPort,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._owner = owner
        self._now = now

    async def run(self, business: BusinessRecord, name: str, args: Mapping[str, Any]) -> Any:
        zone = business_zone(business)
        if name == "schedule":
            return await self._schedule(business, zone, args)
        if name == "needs_you":
            return await self._needs_you(business, zone)
        if name == "money":
            return await self._money(business, zone)
        if name == "find_customer":
            return await self._find_customer(business, zone, str(args.get("query") or ""))
        if name == "quotes":
            return await self._quotes(business, zone, str(args.get("status") or "all"))
        return {"error": f"unknown tool {name!r}"}

    async def _schedule(
        self, business: BusinessRecord, zone: tzinfo, args: Mapping[str, Any]
    ) -> Any:
        today = self._now().astimezone(zone).date()
        try:
            start_day = date.fromisoformat(str(args.get("start_day") or today.isoformat()))
        except ValueError:
            return {"error": "start_day must be YYYY-MM-DD"}
        try:
            days = min(max(int(args.get("days") or 1), 1), SCHEDULE_MAX_DAYS)
        except (TypeError, ValueError):
            days = 1
        try:
            start = datetime.combine(start_day, time(), zone)
            end = datetime.combine(start_day + timedelta(days=days), time(), zone)
        except OverflowError:
            return {"error": "that date is out of range"}
        view = await self._owner.business_calendar(business, start, end)
        return {
            "from": start_day.isoformat(),
            "days": days,
            "events": [
                {
                    "status": _EVENT_STATUS.get(event.source, event.source.value),
                    "start": _when(event.start, zone),
                    "end": _when(event.end, zone),
                    "title": _wrote(event.title),
                    "customer": _wrote(event.invitee_name),
                    "address": _wrote(event.location),
                }
                for event in view.events[: LIST_MAX_ITEMS * 2]
            ],
            "problems": list(view.problems),
        }

    async def _needs_you(self, business: BusinessRecord, zone: tzinfo) -> Any:
        async with self._unit_of_work_factory() as unit_of_work:
            quotes = await unit_of_work.quotes.list_for_business(
                business.business_id, limit=QUOTE_LIST_LIMIT, status=QuoteStatus.AWAITING_APPROVAL
            )
            bookings = await unit_of_work.intake_conversations.list_awaiting_owner(
                business.business_id, limit=BOOKING_LIST_LIMIT
            )
        waiting = [
            quote
            for quote in quotes
            if quote.draft is not None and quote.status is QuoteStatus.AWAITING_APPROVAL
        ]
        return {
            "quotes_to_approve": [_quote_row(quote, zone) for quote in waiting[:LIST_MAX_ITEMS]],
            "booking_requests": [_booking_row(item, zone) for item in bookings[:LIST_MAX_ITEMS]],
        }

    async def _money(self, business: BusinessRecord, zone: tzinfo) -> Any:
        now = self._now()
        async with self._unit_of_work_factory() as unit_of_work:
            payments = await unit_of_work.payments.list_for_business(business.business_id)
            invoices = (
                await unit_of_work.invoices.list_for_business(business.business_id)
                if business.invoice_prefix
                else ()
            )
            quotes = (
                ()
                if business.invoice_prefix
                else await unit_of_work.quotes.list_for_business(
                    business.business_id, limit=QUOTE_LIST_LIMIT
                )
            )
        paid = {
            currency: money_text(minor, currency)
            for currency, minor in month_totals(payments, zone, now).items()
        }
        result: dict[str, Any] = {
            "month": now.astimezone(zone).strftime("%B %Y"),
            "paid_this_month": paid or {"usd": money_text(0)},
        }
        if business.invoice_prefix:
            counted = {payment.quote_id: payment for payment in payments if payment.counts}
            today = now.astimezone(zone).date()
            open_invoices = []
            for invoice in invoices:
                status = invoice_status(invoice, counted.get(invoice.quote_id), today=today)
                if status is not InvoiceStatus.PAID:
                    open_invoices.append((invoice, status))
            result["outstanding"] = totals_text(
                (i.total_minor, i.currency) for i, _ in open_invoices
            )
            result["open_invoices"] = [
                {
                    "number": invoice.number,
                    "customer": _wrote(invoice.customer_name),
                    "total": money_text(invoice.total_minor, invoice.currency),
                    "due": invoice.due_on.strftime("%a %b %-d"),
                    "status": status.value,
                }
                for invoice, status in open_invoices[:LIST_MAX_ITEMS]
            ]
        else:
            unpaid = [
                quote
                for quote in quotes
                if quote.draft is not None and quote.customer_status is CustomerQuoteStatus.ACCEPTED
            ]
            result["outstanding"] = totals_text(
                (quote.draft.total_minor, quote.draft.currency)
                for quote in unpaid
                if quote.draft is not None
            )
            result["accepted_not_paid"] = [
                _quote_row(quote, zone) for quote in unpaid[:LIST_MAX_ITEMS]
            ]
        return result

    async def _find_customer(self, business: BusinessRecord, zone: tzinfo, query: str) -> Any:
        needle = query.strip().casefold()
        if len(needle) < 2:
            return {"error": "give at least 2 letters of a name, e-mail or phone"}
        digits = "".join(ch for ch in needle if ch.isdigit())

        def hit(*values: str | None) -> bool:
            for value in values:
                if not value:
                    continue
                if needle in value.casefold():
                    return True
                if len(digits) >= 4 and digits in "".join(ch for ch in value if ch.isdigit()):
                    return True
            return False

        async with self._unit_of_work_factory() as unit_of_work:
            customers = await unit_of_work.customers.list_for_business(business.business_id)
            bookings = await unit_of_work.intake_conversations.list_booking_requests(
                business.business_id, limit=BOOKING_LIST_LIMIT
            )
            quotes = await unit_of_work.quotes.list_for_business(
                business.business_id, limit=QUOTE_LIST_LIMIT
            )
        matches = [c for c in customers if hit(c.display_name, c.email, c.phone)]
        matched_ids = {customer.customer_id for customer in matches}
        matched_emails = {customer.email.casefold() for customer in matches}

        def booking_matches(item: IntakeConversation) -> bool:
            collected = item.collected
            return (item.customer_id is not None and item.customer_id in matched_ids) or hit(
                collected.name, collected.email, collected.phone
            )

        def quote_matches(quote: Quote) -> bool:
            if quote.draft is None:
                return False
            recipient = quote.draft.recipient
            return (
                (quote.customer_id is not None and quote.customer_id in matched_ids)
                or recipient.address.casefold() in matched_emails
                or hit(recipient.display_name, recipient.address, recipient.phone)
            )

        return {
            "customers": [
                {
                    "name": _wrote(customer.display_name),
                    "email": customer.email,
                    "phone": customer.phone,
                }
                for customer in matches[:CUSTOMER_MATCH_MAX]
            ],
            "bookings": [_booking_row(item, zone) for item in bookings if booking_matches(item)][
                :LIST_MAX_ITEMS
            ],
            "quotes": [_quote_row(quote, zone) for quote in quotes if quote_matches(quote)][
                :LIST_MAX_ITEMS
            ],
        }

    async def _quotes(self, business: BusinessRecord, zone: tzinfo, status: str) -> Any:
        async with self._unit_of_work_factory() as unit_of_work:
            quotes = await unit_of_work.quotes.list_for_business(
                business.business_id, limit=QUOTE_LIST_LIMIT
            )
        rows = [
            _quote_row(quote, zone)
            for quote in quotes
            if quote.draft is not None and _quote_filter(quote, status)
        ]
        return {"status": status, "count": len(rows), "quotes": rows[:LIST_MAX_ITEMS]}


class OwnerAssistant:
    """Answers one owner message for the business it came from."""

    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        model: OwnerAssistantPort,
        tools: OwnerReadTools,
        *,
        ceilings: UsageCeilingGuard | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._model = model
        self._tools = tools
        self._ceilings = ceilings
        self._now = now

    async def answer(
        self,
        business_id: BusinessId,
        text: str,
        history: tuple[tuple[str, str], ...] = (),
    ) -> str:
        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get(business_id)
        if business is None:
            return UNAVAILABLE_REPLY
        now = self._now()
        if self._ceilings is not None and await self._ceilings.is_reached(
            business_id, UsageKind.REVIEW_TOKENS, now=now
        ):
            return LIMIT_REPLY
        request = OwnerAssistantRequest(
            business_id=business_id,
            business_name=business.display_name or business.name,
            local_now=now.astimezone(business_zone(business)).strftime("%A %Y-%m-%d, %-I:%M %p %Z"),
            text=text,
            tools=TOOLS,
            history=history,
        )

        async def run_tool(name: str, args: Mapping[str, Any]) -> Any:
            return await self._tools.run(business, name, args)

        try:
            answer = await self._model.answer(request, run_tool)
        except OwnerAssistantSetupError as error:
            logger.error(
                "ALERT owner assistant setup problem: %s",
                error,
                extra={"business_id": str(business_id), "alert": "owner_assistant_setup"},
            )
            return SETUP_REPLY
        except OwnerAssistantError:
            logger.warning("owner assistant failed", exc_info=True)
            return UNAVAILABLE_REPLY
        reply = answer.text.strip()
        if not reply:
            return UNAVAILABLE_REPLY
        logger.info(
            "owner assistant answered",
            extra={"business_id": str(business_id), "tools": list(answer.tools_called)},
        )
        if len(reply) > OWNER_ASSISTANT_REPLY_MAX_CHARS:
            reply = reply[: OWNER_ASSISTANT_REPLY_MAX_CHARS - 1].rstrip() + "…"
        return reply
