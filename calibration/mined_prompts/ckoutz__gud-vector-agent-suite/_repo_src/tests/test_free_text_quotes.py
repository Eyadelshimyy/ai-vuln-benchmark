"""``quote: inspection 250`` drafts through a model; GVAS still owns every price."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.quotes import FREE_TEXT_DRAFT_NOTICE, QuoteWorkflowHandler
from gvas.config import OpenAISettings
from gvas.domain.enums import QuoteStatus
from gvas.domain.identifiers import BusinessId, ConversationId, QuoteId
from gvas.domain.quotes import (
    FreeTextQuoteDraft,
    FreeTextQuoteDraftingError,
    FreeTextQuoteItem,
    QuoteAppointmentContext,
    QuoteDraftRejectedError,
    QuoteDraftRequest,
    has_quote_trigger,
    quote_trigger_request_text,
)
from gvas.domain.usage import UsageCeilingGuard, UsageCeilings, UsageKind
from gvas.infrastructure.calendly.api import (
    CalendlyInvitee,
    CalendlyQuestionAnswer,
    booking_notes,
)
from gvas.infrastructure.openai_quote_drafting import OpenAIFreeTextQuoteDrafter
from gvas.infrastructure.quote_drafting import (
    FORMAT_HELP,
    FREE_TEXT_CEILING_REACHED,
    FREE_TEXT_UNAVAILABLE,
    DeterministicQuoteDrafter,
    ModelAssistedQuoteDrafter,
    is_structured_request,
    quantity_is_written,
    written_amount_minor,
    written_amounts_minor,
)
from test_quote_appointments import AppointmentsFake, appointment, reply_text
from test_quotes import (
    NOW,
    TrackedUnitOfWorkFactory,
    UnitOfWorkTracker,
    active_quote,
    owner_message,
    seed_conversation,
    workflow_context,
)
from test_usage_ceilings import MemoryLedger

OPENAI_KEY = "sk-test-quotes"
LEAK_MARKER = "provider-detail-never-shown"
STRUCTURED = "quote:\ncustomer: bob@example.test\ncurrency: USD\nitem: 1 | Air sampling | 125.00"


def item(
    description: str, price: str | None, quantity: int = 1, quantity_text: str | None = None
) -> FreeTextQuoteItem:
    return FreeTextQuoteItem(
        description=description,
        quantity=quantity,
        quantity_text=quantity_text,
        unit_price=price,
    )


class ModelFake:
    def __init__(self, *drafts: FreeTextQuoteDraft, failing: bool = False) -> None:
        self._drafts = list(drafts)
        self.failing = failing
        self.requests: list[QuoteDraftRequest] = []

    async def draft(self, request: QuoteDraftRequest) -> FreeTextQuoteDraft:
        self.requests.append(request)
        if self.failing:
            raise FreeTextQuoteDraftingError(f"openai returned http 500 ({LEAK_MARKER})")
        return self._drafts.pop(0) if self._drafts else FreeTextQuoteDraft()


def drafter(
    model: ModelFake, *, ledger: MemoryLedger | None = None, review_tokens: int = 0
) -> ModelAssistedQuoteDrafter:
    return ModelAssistedQuoteDrafter(
        DeterministicQuoteDrafter(),
        model,
        ceilings=UsageCeilingGuard(ledger, UsageCeilings(review_tokens=review_tokens)),
        now=lambda: NOW,
    )


async def build(
    session_factory: async_sessionmaker[AsyncSession],
    model: ModelFake,
    lookup: AppointmentsFake | None = None,
) -> tuple[QuoteWorkflowHandler, BusinessId, ConversationId]:
    business_id, conversation_id = await seed_conversation(session_factory)
    handler = QuoteWorkflowHandler(
        TrackedUnitOfWorkFactory(session_factory, UnitOfWorkTracker()),
        drafter(model),
        appointment_lookup=lookup,
    )
    return handler, business_id, conversation_id


def request(
    text: str,
    appointment_context: QuoteAppointmentContext | None = None,
    business_id: BusinessId | None = None,
) -> QuoteDraftRequest:
    return QuoteDraftRequest(
        quote_id=QuoteId(uuid4()),
        business_id=business_id or BusinessId(uuid4()),
        conversation_id=ConversationId(uuid4()),
        request_text=text,
        revision=1,
        idempotency_key="k",
        appointment=appointment_context,
    )


def test_trigger_accepts_quote_for_name() -> None:
    assert quote_trigger_request_text("Quote: inspection 250") == "inspection 250"
    assert quote_trigger_request_text("quote for Jane: mold inspection 350") == (
        "for: Jane\nmold inspection 350"
    )
    assert quote_trigger_request_text("quote for: x") is None
    assert quote_trigger_request_text("quotes: 1") is None
    message = owner_message(BusinessId(uuid4()), ConversationId(uuid4()), "m", "quote for bo: x 1")
    assert has_quote_trigger(message)


def test_written_amounts_accept_owner_spellings() -> None:
    text = "2 air samples at $125 each plus the report 1,250.50, cert 300.5 and 3 hours"
    assert written_amounts_minor(text) == {12500, 125050, 30050, 200, 300}
    assert written_amount_minor("$1,250.00") == 125000
    assert written_amount_minor("250") == 25000
    assert written_amount_minor("250.5") == 25050
    assert written_amount_minor("12,50") is None
    assert written_amount_minor("250 each") is None
    assert written_amount_minor("") is None


def test_structured_request_detection() -> None:
    assert is_structured_request("customer: a@b.co\nitem: 1 | x | 2")
    assert not is_structured_request("inspection 250")
    assert not is_structured_request("for: jane\nmold inspection 350")
    assert not is_structured_request("mold inspection: 350")


def test_format_help_says_customer_is_optional_with_a_booking() -> None:
    assert "customer: is optional when a booking matches" in FORMAT_HELP


async def test_structured_input_never_reaches_the_model(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    model = ModelFake()
    handler, business_id, conversation_id = await build(session_factory, model)
    message = owner_message(business_id, conversation_id, "quote-1", STRUCTURED)

    text = reply_text(await handler.handle(workflow_context(message, conversation_id)))

    assert "Total: USD 125.00" in text
    assert FREE_TEXT_DRAFT_NOTICE not in text
    assert model.requests == []


async def test_structured_mistakes_keep_the_parser_message() -> None:
    model = ModelFake()
    with pytest.raises(QuoteDraftRejectedError, match="missing: currency"):
        await drafter(model).draft(request("customer: bob@example.test\nitem: 1 | x | 2"))
    assert model.requests == []


async def test_free_text_with_literal_prices_drafts_and_flags_the_reply(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    model = ModelFake(
        FreeTextQuoteDraft(
            line_items=(item("Air sample", "125", 2, "2 air samples"), item("Report", "200")),
            owner_note="We'll be there Tuesday",
            ambiguities=("'200' read as the report's price",),
        )
    )
    handler, business_id, conversation_id = await build(session_factory, model)
    message = owner_message(
        business_id,
        conversation_id,
        "quote-1",
        "quote: 2 air samples at 125 each plus the report 200, note we'll be there tuesday"
        "\ncustomer: bob@example.test",
    )

    text = reply_text(await handler.handle(workflow_context(message, conversation_id)))

    assert "2 × Air sample" in text
    assert "Total: USD 450.00" in text
    assert "Note: We'll be there Tuesday" in text
    assert FREE_TEXT_DRAFT_NOTICE in text
    assert text.index(FREE_TEXT_DRAFT_NOTICE) < text.index("Reply with approve")
    assert model.requests[0].request_text == (
        "2 air samples at 125 each plus the report 200, note we'll be there tuesday"
    )
    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is not None and quote.draft is not None
    assert quote.status is QuoteStatus.AWAITING_APPROVAL
    assert quote.draft.drafted_from_free_text
    assert quote.draft.owner_note == "We'll be there Tuesday"
    assert quote.draft.recipient.address == "bob@example.test"
    assert [(li.quantity, li.unit_price_minor) for li in quote.draft.line_items] == [
        (2, 12500),
        (1, 20000),
    ]


async def test_dollar_and_comma_amounts_and_default_quantity() -> None:
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Remediation", "$1,250"), item("Cert", "300.00")))
    )
    proposal = await drafter(model).draft(
        request("remediation $1,250 and the cert 300\ncustomer: bob@example.test")
    )
    assert [(li.quantity, li.unit_price_minor) for li in proposal.line_items] == [
        (1, 125000),
        (1, 30000),
    ]
    assert proposal.currency == "USD"
    assert proposal.drafted_from_free_text


async def test_quantity_not_in_text_is_refused() -> None:
    model = ModelFake(FreeTextQuoteDraft(line_items=(item("Air sample", "125", 3),)))
    with pytest.raises(QuoteDraftRejectedError, match="does not say 3 × 'Air sample'"):
        await drafter(model).draft(request("air samples at 125\ncustomer: bob@example.test"))
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Air sample", "125", 3, "3 air samples"),))
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 3 × 'Air sample'"):
        await drafter(model).draft(request("air samples at 125\ncustomer: bob@example.test"))


async def test_quantity_is_not_read_from_a_job_attribute() -> None:
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Inspection for 2 bedrooms", "250", 2, "2 bedrooms"),))
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 2 × 'Inspection"):
        await drafter(model).draft(
            request("inspection for 2 bedrooms 250\ncustomer: bob@example.test")
        )


async def test_quantity_is_not_read_from_a_price() -> None:
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Air sample", "1,250", 250, "250 air sample"),))
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 250"):
        await drafter(model).draft(request("air sample $1,250\ncustomer: bob@example.test"))
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Air sample", "125", 125, "air sample 125"),))
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 125"):
        await drafter(model).draft(request("air sample 125\ncustomer: bob@example.test"))


async def test_quantity_is_not_borrowed_from_another_item() -> None:
    model = ModelFake(
        FreeTextQuoteDraft(
            line_items=(
                item("Air sample", "125", 3, "3 air samples"),
                item("Report", "200", 3, "3 air samples"),
            )
        )
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 3 × 'Report'"):
        await drafter(model).draft(
            request("3 air samples at 125 and the report 200\ncustomer: bob@example.test")
        )


async def test_quantity_source_must_name_its_own_item() -> None:
    model = ModelFake(
        FreeTextQuoteDraft(
            line_items=(
                item("Mold inspection", "250", 2, "2 mold inspections"),
                item("Mold remediation", "900", 2, "2 mold inspections"),
            )
        )
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 2 × 'Mold remediation'"):
        await drafter(model).draft(
            request("2 mold inspections 250, mold remediation 900\ncustomer: bob@example.test")
        )
    model = ModelFake(
        FreeTextQuoteDraft(
            line_items=(
                item("Mold inspection", "250", 2, "2 inspections"),
                item("Radon inspection", "300", 2, "2 inspections"),
            )
        )
    )
    with pytest.raises(QuoteDraftRejectedError, match="does not say 2 × 'Radon inspection'"):
        await drafter(model).draft(
            request("2 inspections: mold 250 and radon 300\ncustomer: bob@example.test")
        )


async def test_quantity_written_twice_counts_two_items() -> None:
    model = ModelFake(
        FreeTextQuoteDraft(
            line_items=(
                item("Filter (furnace)", "40", 2, "2 filters"),
                item("Filter (return)", "60", 2, "2 filters"),
            )
        )
    )
    proposal = await drafter(model).draft(
        request("2 filters for the furnace 40 and 2 filters at the return 60\ncustomer: b@x.test")
    )
    assert [li.quantity for li in proposal.line_items] == [2, 2]


def test_quantity_is_written_checks_the_drafters_source_words() -> None:
    assert quantity_is_written("3 x Air Samples at 125", 3, "Air sample", "3 x air samples")
    assert quantity_is_written("3 x Air Samples at 125", 3, "Air sample", None) is False
    assert quantity_is_written("3 x Air Samples at 125", 3, "Air sample", "3 x reports") is False
    assert quantity_is_written("3 x Air Samples at 125", 3, "Air sample", "3 x") is False
    assert quantity_is_written("3 x Air Samples at 125", 3, "Air sample", "3 x at") is False
    assert quantity_is_written("reports: 2 and 1 inspection", 2, "Report", "reports: 2") is False
    assert quantity_is_written("on 2 bedrooms, 2 samples", 2, "Mold sample", "2 samples")
    assert (
        quantity_is_written("on 2 bedrooms, 2 samples", 2, "Sample, 2 bedrooms", "2 bedrooms")
        is False
    )
    assert (
        quantity_is_written("2 mold inspections", 2, "Mold remediation", "2 mold inspections")
        is False
    )
    assert quantity_is_written(
        "2 mold inspections", 2, "Mold inspection - attic", "2 mold inspections"
    )
    assert quantity_is_written("$2,500 samples", 2, "Sample", "2,500 samples") is False
    assert quantity_is_written("$2,500 samples", 2, "Sample", "500 samples") is False


async def test_price_not_in_text_asks_once_and_drafts_nothing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Mold inspection", None), item("Report", "999")))
    )
    handler, business_id, conversation_id = await build(session_factory, model)
    message = owner_message(
        business_id,
        conversation_id,
        "quote-1",
        "quote: mold inspection\ncustomer: bob@example.test",
    )

    result = await handler.handle(workflow_context(message, conversation_id))
    retry = await handler.handle(workflow_context(message, conversation_id))

    assert reply_text(result).startswith(
        "What's the prices for 'Mold inspection', 'Report'? "
        "Send the quote again with every price included."
    )
    assert reply_text(retry) == "Quote rejected."
    assert len(model.requests) == 1
    assert await active_quote(session_factory, business_id, conversation_id) is None


async def test_no_items_rejects_with_help() -> None:
    model = ModelFake(FreeTextQuoteDraft())
    with pytest.raises(QuoteDraftRejectedError, match="No priced items") as info:
        await drafter(model).draft(request("hello there 250\ncustomer: bob@example.test"))
    assert FORMAT_HELP in str(info.value)


async def test_appointment_context_reaches_the_model_but_never_prices(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    booked = appointment("Jane Doe", "jane@example.test", address="234 Del Rd").model_copy(
        update={"notes": ("Anything we should know?: attic mold, 2 bedrooms",)}
    )
    model = ModelFake(
        FreeTextQuoteDraft(line_items=(item("Mold inspection — attic and 2 bedrooms", "350"),))
    )
    handler, business_id, conversation_id = await build(
        session_factory, model, AppointmentsFake((booked,))
    )
    message = owner_message(
        business_id, conversation_id, "quote-1", "quote for jane: mold inspection 350"
    )

    text = reply_text(await handler.handle(workflow_context(message, conversation_id)))

    sent = model.requests[0]
    assert sent.request_text == "mold inspection 350"
    assert sent.appointment == QuoteAppointmentContext(
        event_name="Site visit",
        start_time=booked.start_time,
        address="234 Del Rd",
        invitee_name="Jane Doe",
        notes=("Anything we should know?: attic mold, 2 bedrooms",),
    )
    assert text.startswith("Customer: Jane Doe (jane@example.test)")
    assert "Mold inspection — attic and 2 bedrooms" in text
    assert "Total: USD 350.00" in text

    # A price the model lifted from the booking rather than the owner's text is refused.
    priced_from_booking = ModelFake(FreeTextQuoteDraft(line_items=(item("Mold inspection", "2"),)))
    handler, business_id, conversation_id = await build(
        session_factory, priced_from_booking, AppointmentsFake((booked,))
    )
    message = owner_message(business_id, conversation_id, "quote-1", "quote: mold inspection")
    assert reply_text(await handler.handle(workflow_context(message, conversation_id))).startswith(
        "What's the price for 'Mold inspection'?"
    )
    assert await active_quote(session_factory, business_id, conversation_id) is None


async def test_model_failure_gives_one_sanitized_help_reply(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    model = ModelFake(failing=True)
    handler, business_id, conversation_id = await build(session_factory, model)
    message = owner_message(
        business_id, conversation_id, "quote-1", "quote: inspection 250\ncustomer: bob@example.test"
    )

    result = await handler.handle(workflow_context(message, conversation_id))
    retry = await handler.handle(workflow_context(message, conversation_id))

    text = reply_text(result)
    assert text.startswith(FREE_TEXT_UNAVAILABLE)
    assert FORMAT_HELP in text
    assert LEAK_MARKER not in text and "500" not in text
    assert reply_text(retry) == "Quote rejected."
    assert len(model.requests) == 1
    assert await active_quote(session_factory, business_id, conversation_id) is None


async def test_review_ceiling_refuses_with_structured_help() -> None:
    ledger = MemoryLedger()
    business_id = BusinessId(uuid4())
    await ledger.record(business_id, UsageKind.REVIEW_TOKENS, 1000, at=NOW)
    model = ModelFake(FreeTextQuoteDraft(line_items=(item("Inspection", "250"),)))
    base = request("inspection 250\ncustomer: bob@example.test", business_id=business_id)

    with pytest.raises(QuoteDraftRejectedError) as info:
        await drafter(model, ledger=ledger, review_tokens=1000).draft(base)
    assert str(info.value) == FREE_TEXT_CEILING_REACHED
    assert FORMAT_HELP in str(info.value)
    assert model.requests == []

    proposal = await drafter(model, ledger=ledger, review_tokens=1001).draft(base)
    assert proposal.line_items[0].unit_price_minor == 25000


async def test_free_text_is_channel_neutral(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    replies: list[str] = []
    for _ in ("slack", "sms"):
        model = ModelFake(FreeTextQuoteDraft(line_items=(item("Inspection", "250"),)))
        handler, business_id, conversation_id = await build(session_factory, model)
        message = owner_message(
            business_id,
            conversation_id,
            "quote-1",
            "quote: inspection 250\ncustomer: bob@example.test",
        )
        replies.append(reply_text(await handler.handle(workflow_context(message, conversation_id))))
    assert replies[0] == replies[1]
    assert FREE_TEXT_DRAFT_NOTICE in replies[0]


def completion(content: object) -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
    }


@pytest.mark.asyncio
async def test_openai_drafter_sends_text_and_appointment_as_schema_request() -> None:
    seen: list[httpx.Request] = []
    ledger = MemoryLedger()

    def handle(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return httpx.Response(
            200,
            json=completion(
                {
                    "items": [
                        {
                            "description": " Mold  inspection ",
                            "quantity": 0,
                            "quantity_text": "",
                            "unit_price": "350",
                        },
                        {
                            "description": "Report",
                            "quantity": 2,
                            "quantity_text": " 2  reports ",
                            "unit_price": "",
                        },
                        {"description": "   ", "quantity": 1, "unit_price": "1"},
                    ],
                    "owner_note": "",
                    "ambiguities": ["report has no price"],
                }
            ),
        )

    context = QuoteAppointmentContext(
        event_name="Site visit",
        start_time=datetime(2026, 3, 10, 14, tzinfo=UTC),
        address="234 Del Rd",
        invitee_name="Jane Doe",
        notes=("Anything we should know?: attic mold, 2 bedrooms",),
    )
    business_id = BusinessId(uuid4())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = OpenAIFreeTextQuoteDrafter(
            OpenAISettings(api_key=OPENAI_KEY, review_model="review-model"),
            client,
            usage_ledger=ledger,
        )
        draft = await adapter.draft(
            request("mold inspection 350 and the report", context, business_id)
        )

    assert draft == FreeTextQuoteDraft(
        line_items=(item("Mold inspection", "350"), item("Report", None, 2, "2 reports")),
        owner_note=None,
        ambiguities=("report has no price",),
    )
    http_request = seen[0]
    assert http_request.url.path.endswith("/chat/completions")
    assert http_request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    body = json.loads(http_request.read())
    assert body["model"] == "review-model"
    assert "temperature" not in body
    assert "seed" in body
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    item_schema = body["response_format"]["json_schema"]["schema"]["properties"]["items"]["items"]
    assert "quantity_text" in item_schema["required"]
    user_content = json.loads(body["messages"][1]["content"])
    assert user_content["request_text"] == "mold inspection 350 and the report"
    assert user_content["appointment"] == {
        "event_name": "Site visit",
        "start_time": "2026-03-10T14:00:00+00:00",
        "address": "234 Del Rd",
        "customer_name": "Jane Doe",
        "booking_notes": ["Anything we should know?: attic mold, 2 bedrooms"],
    }
    assert ledger.records == [(business_id, UsageKind.REVIEW_TOKENS, 52)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"message": f"bad key {OPENAI_KEY}"}}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=completion({"items": "not a list"})),
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
    ],
)
async def test_openai_drafter_sanitizes_failures(response: httpx.Response) -> None:
    ledger = MemoryLedger()

    def handle(_: httpx.Request) -> httpx.Response:
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = OpenAIFreeTextQuoteDrafter(
            OpenAISettings(api_key=OPENAI_KEY), client, usage_ledger=ledger
        )
        with pytest.raises(FreeTextQuoteDraftingError) as info:
            await adapter.draft(request("inspection 250"))

    assert OPENAI_KEY not in str(info.value)
    assert "bad key" not in str(info.value)
    assert ledger.records == []


@pytest.mark.asyncio
async def test_openai_drafter_reports_transport_errors_without_retrying() -> None:
    calls = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("dns failure")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = OpenAIFreeTextQuoteDrafter(OpenAISettings(api_key=OPENAI_KEY), client)
        with pytest.raises(FreeTextQuoteDraftingError):
            await adapter.draft(request("inspection 250"))
    assert calls == 1


def test_calendly_booking_answers_become_appointment_notes() -> None:
    invitee = CalendlyInvitee(
        name="Jane Doe",
        email="jane@example.test",
        status="active",
        questions_and_answers=(
            CalendlyQuestionAnswer(question="Address", answer="234 Del Rd", position=1),
            CalendlyQuestionAnswer(question="Anything else?", answer="attic mold", position=0),
            CalendlyQuestionAnswer(question="Blank", answer="  ", position=2),
        ),
    )
    assert booking_notes(invitee) == ("Anything else?: attic mold", "Address: 234 Del Rd")
