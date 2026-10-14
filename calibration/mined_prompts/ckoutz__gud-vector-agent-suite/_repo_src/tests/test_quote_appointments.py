"""Quotes without ``customer:`` resolve the customer from the owner's appointments."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.quotes import CUSTOMER_LOOKUP_UNAVAILABLE, QuoteWorkflowHandler
from gvas.domain.appointments import (
    Appointment,
    AppointmentLookupError,
    AppointmentWindow,
    surrounding_days_window,
)
from gvas.domain.enums import QuoteStatus
from gvas.domain.identifiers import BusinessId, ConversationId, QuoteId
from gvas.domain.messages import TextPart
from gvas.domain.quotes import (
    QuoteDraftProposal,
    QuoteDraftRejectedError,
    QuoteDraftRequest,
    has_customer_line,
    requested_customer_name,
)
from gvas.domain.workflows import WorkflowResult
from gvas.infrastructure.quote_drafting import DeterministicQuoteDrafter
from test_quotes import (
    NOW,
    TrackedUnitOfWorkFactory,
    UnitOfWorkTracker,
    active_quote,
    owner_message,
    seed_conversation,
    workflow_context,
)

LEAK_MARKER = "provider-detail-never-shown"
REQUEST = "quote:\ncurrency: USD\nitem: 1 | Air sampling | 125.00"


def appointment(
    name: str,
    email: str,
    *,
    address: str | None,
    hour: int = 14,
    day_offset: int = 0,
) -> Appointment:
    return Appointment(
        appointment_id=f"appt:{email}",
        start_time=NOW.replace(hour=hour, minute=0) + timedelta(days=day_offset),
        invitee_name=name,
        invitee_email=email,
        event_name="Site visit",
        address=address,
        source_label="Calendly",
    )


class AppointmentsFake:
    def __init__(
        self,
        appointments: tuple[Appointment, ...] = (),
        *,
        failing: bool = False,
    ) -> None:
        self.appointments = appointments
        self.failing = failing
        self.windows: list[AppointmentWindow] = []

    async def find(self, window: AppointmentWindow) -> tuple[Appointment, ...]:
        self.windows.append(window)
        if self.failing:
            raise AppointmentLookupError(f"calendly returned http 401 ({LEAK_MARKER})")
        return self.appointments


def reply_text(result: WorkflowResult) -> str:
    assert len(result.replies) == 1
    part = result.replies[0].parts[0]
    assert isinstance(part, TextPart)
    return part.text


async def build(
    session_factory: async_sessionmaker[AsyncSession],
    lookup: AppointmentsFake | None,
) -> tuple[QuoteWorkflowHandler, BusinessId, ConversationId]:
    business_id, conversation_id = await seed_conversation(session_factory)
    handler = QuoteWorkflowHandler(
        TrackedUnitOfWorkFactory(session_factory, UnitOfWorkTracker()),
        DeterministicQuoteDrafter(),
        appointment_lookup=lookup,
    )
    return handler, business_id, conversation_id


def test_window_is_three_utc_days_around_now() -> None:
    window = surrounding_days_window(BusinessId(uuid4()), datetime(2026, 3, 10, 23, 30, tzinfo=UTC))
    assert window.start == datetime(2026, 3, 9, tzinfo=UTC)
    assert window.end == datetime(2026, 3, 12, tzinfo=UTC)


def test_request_line_helpers() -> None:
    assert has_customer_line("currency: USD\nCustomer: a@b.co")
    assert has_customer_line("email: a@b.co")
    assert not has_customer_line(REQUEST)
    assert requested_customer_name("for: Jane\ncurrency: USD") == "Jane"
    assert requested_customer_name(REQUEST) is None


async def test_single_match_drafts_and_names_the_customer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    lookup = AppointmentsFake((appointment("Jane Doe", "jane@example.test", address="234 Del Rd"),))
    handler, business_id, conversation_id = await build(session_factory, lookup)
    message = owner_message(business_id, conversation_id, "quote-1", REQUEST)

    result = await handler.handle(workflow_context(message, conversation_id))
    retry = await handler.handle(workflow_context(message, conversation_id))

    text = reply_text(result)
    assert text.startswith(
        "Customer: Jane Doe (jane@example.test) — Calendly, Fri 2:00pm, 234 Del Rd"
    )
    assert result == retry
    assert lookup.windows[0].business_id == business_id
    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is not None and quote.draft is not None
    assert quote.status is QuoteStatus.AWAITING_APPROVAL
    assert quote.draft.recipient.address == "jane@example.test"
    assert quote.draft.recipient.display_name == "Jane Doe"


async def test_explicit_customer_skips_the_lookup(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    lookup = AppointmentsFake((appointment("Jane Doe", "jane@example.test", address=None),))
    handler, business_id, conversation_id = await build(session_factory, lookup)
    message = owner_message(
        business_id, conversation_id, "quote-1", f"{REQUEST}\ncustomer: bob@example.test"
    )

    text = reply_text(await handler.handle(workflow_context(message, conversation_id)))

    assert lookup.windows == []
    assert "Customer:" not in text
    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is not None and quote.draft is not None
    assert quote.draft.recipient.address == "bob@example.test"


async def test_no_match_or_disabled_lookup_keeps_customer_required(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    for lookup in (AppointmentsFake(()), None):
        handler, business_id, conversation_id = await build(session_factory, lookup)
        message = owner_message(business_id, conversation_id, "quote-1", REQUEST)

        text = reply_text(await handler.handle(workflow_context(message, conversation_id)))

        assert "The quote is missing: customer." in text
        assert await active_quote(session_factory, business_id, conversation_id) is None


async def test_multiple_matches_ask_for_a_number_and_selection_proceeds(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    lookup = AppointmentsFake(
        (
            appointment("Bo Lee", "bo@example.test", address="343 Thing Ave", hour=16),
            appointment("Jane Doe", "jane@example.test", address="234 Del Rd", hour=9),
            appointment("Al Ng", "al@example.test", address=None, hour=11, day_offset=1),
        )
    )
    handler, business_id, conversation_id = await build(session_factory, lookup)
    message = owner_message(business_id, conversation_id, "quote-1", REQUEST)

    prompt = await handler.handle(workflow_context(message, conversation_id))
    prompt_retry = await handler.handle(workflow_context(message, conversation_id))
    assert prompt == prompt_retry
    assert reply_text(prompt) == (
        "Which appointment is this quote for? "
        "1. 234 Del Rd  2. 343 Thing Ave  3. Al Ng, Sat 11:00am — reply with the number"
    )
    pending = await active_quote(session_factory, business_id, conversation_id)
    assert pending is not None
    assert pending.status is QuoteStatus.AWAITING_CUSTOMER_SELECTION
    assert pending.customer_candidates is not None and len(pending.customer_candidates) == 3

    bad = owner_message(business_id, conversation_id, "pick-0", "7")
    assert reply_text(await handler.handle(workflow_context(bad, conversation_id))).startswith(
        "Reply with a number from 1 to 3, or reject."
    )

    pick = owner_message(business_id, conversation_id, "pick-1", "2")
    drafted = await handler.handle(workflow_context(pick, conversation_id))
    drafted_retry = await handler.handle(workflow_context(pick, conversation_id))
    assert drafted == drafted_retry
    assert reply_text(drafted).startswith(
        "Customer: Bo Lee (bo@example.test) — Calendly, Fri 4:00pm, 343 Thing Ave"
    )
    assert len(lookup.windows) == 1
    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is not None and quote.draft is not None
    assert quote.status is QuoteStatus.AWAITING_APPROVAL
    assert quote.customer_candidates is None
    assert quote.draft.recipient.address == "bo@example.test"

    approve = owner_message(business_id, conversation_id, "approve-1", "approve")
    assert reply_text(await handler.handle(workflow_context(approve, conversation_id))) == (
        "Quote approved and queued for customer delivery."
    )


async def test_reject_while_awaiting_selection_cancels_the_quote(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    lookup = AppointmentsFake(
        (
            appointment("Bo Lee", "bo@example.test", address="343 Thing Ave"),
            appointment("Jane Doe", "jane@example.test", address="234 Del Rd"),
        )
    )
    handler, business_id, conversation_id = await build(session_factory, lookup)
    await handler.handle(
        workflow_context(
            owner_message(business_id, conversation_id, "quote-1", REQUEST), conversation_id
        )
    )
    cancel = owner_message(business_id, conversation_id, "cancel", "reject")

    text = reply_text(await handler.handle(workflow_context(cancel, conversation_id)))

    assert text.startswith("Quote cancelled.")
    assert await active_quote(session_factory, business_id, conversation_id) is None


async def test_for_line_filters_candidates_by_name(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    lookup = AppointmentsFake(
        (
            appointment("Bo Lee", "bo@example.test", address="343 Thing Ave"),
            appointment("Jane Doe", "jane@example.test", address="234 Del Rd"),
        )
    )
    handler, business_id, conversation_id = await build(session_factory, lookup)
    message = owner_message(business_id, conversation_id, "quote-1", f"{REQUEST}\nfor: jane")

    text = reply_text(await handler.handle(workflow_context(message, conversation_id)))

    assert text.startswith("Customer: Jane Doe (jane@example.test)")


async def test_lookup_failure_asks_for_customer_once_without_leaking(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    lookup = AppointmentsFake(failing=True)
    handler, business_id, conversation_id = await build(session_factory, lookup)
    message = owner_message(business_id, conversation_id, "quote-1", REQUEST)

    result = await handler.handle(workflow_context(message, conversation_id))
    retry = await handler.handle(workflow_context(message, conversation_id))

    text = reply_text(result)
    assert text.startswith(CUSTOMER_LOOKUP_UNAVAILABLE)
    assert LEAK_MARKER not in text
    assert "401" not in text
    assert reply_text(retry) == "Quote rejected."
    assert len(lookup.windows) == 1
    assert await active_quote(session_factory, business_id, conversation_id) is None

    resent = owner_message(
        business_id, conversation_id, "quote-2", f"{REQUEST}\ncustomer: jane@example.test"
    )
    assert "Total:" in reply_text(await handler.handle(workflow_context(resent, conversation_id)))


async def test_drafter_prefers_the_explicit_customer_over_the_supplied_recipient() -> None:
    drafter = DeterministicQuoteDrafter()
    supplied = appointment("Jane Doe", "jane@example.test", address=None)
    base = QuoteDraftRequest(
        quote_id=QuoteId(uuid4()),
        business_id=BusinessId(uuid4()),
        conversation_id=ConversationId(uuid4()),
        request_text=f"{REQUEST}\nfor: Jane\ncustomer: bob@example.test",
        revision=1,
        idempotency_key="k",
        recipient=None,
    )
    explicit: QuoteDraftProposal = await drafter.draft(base)
    assert explicit.recipient.address == "bob@example.test"

    with pytest.raises(QuoteDraftRejectedError, match="missing: customer"):
        await drafter.draft(base.model_copy(update={"request_text": REQUEST}))
    assert supplied.choice_label == "Jane Doe, Fri 2:00pm"
