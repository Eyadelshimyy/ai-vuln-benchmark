"""Approved quotes handed to the customer portal, texted via Telnyx, confirmed to the owner."""

import json
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import FAKE_NOW, OwnerReplyFake, TranscriptionFake
from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.application.quotes import SiteAwareQuoteDelivery, _appointment_recipient
from gvas.composition import ApplicationPorts, build_application
from gvas.composition.production import (
    ProductionConfigurationError,
    build_production_runtime,
    load_production_settings,
)
from gvas.domain.enums import DeliveryStatus, OutboxStatus, RecipientAddressKind
from gvas.domain.identifiers import BusinessId, QuoteId
from gvas.domain.messages import (
    CustomerDeliveryLineItem,
    CustomerDeliveryRequest,
    CustomerRecipient,
    CustomerTextRequest,
    DeliveryReceipt,
    TextPart,
)
from gvas.domain.quotes import (
    SMS_SEGMENT_LIMIT,
    QuoteDraftProposal,
    QuoteDraftRequest,
    QuoteLineItem,
    customer_quote_text,
)
from gvas.infrastructure.calendly.api import CalendlyAppointmentLookup
from gvas.infrastructure.customer_repositories import SqlCustomerRepository
from gvas.infrastructure.models import OutboxMessage, QuoteRecord
from gvas.infrastructure.portal import (
    InMemoryPortalHandoffLedger,
    PortalDeliveryError,
    PortalQuoteDelivery,
    PortalSettings,
    SqlPortalHandoffLedger,
)
from gvas.infrastructure.repositories import SqlQuoteRepository
from gvas.infrastructure.resend import ResendQuoteDeliveryAdapter
from gvas.infrastructure.telnyx.customer_text import TelnyxCustomerTextAdapter
from gvas.infrastructure.telnyx.delivery import (
    InMemoryTelnyxDeliveryLedger,
    TelnyxDeliveryError,
    TelnyxRoutingError,
    TelnyxSendRequest,
    TelnyxSendResult,
)
from gvas.infrastructure.telnyx.installations import TelnyxInstallation
from gvas.interfaces.http.owner import owner_quote_payload
from test_calendly_lookup import EVENT_A, Recorder, event, invitee
from test_calendly_lookup import WINDOW as CALENDLY_WINDOW
from test_calendly_lookup import client as calendly_client
from test_calendly_lookup import settings as calendly_settings
from test_composition import Clock, inbound, seed_business
from test_pilot_runtime import immediate_worker, texts_of
from test_production_runtime import ENVIRONMENT, TELNYX_ENVIRONMENT

BUSINESS_ID = BusinessId(uuid4())
PORTAL_TOKEN = "portal-not-a-real-token"  # noqa: S105
QUOTE_URL = "https://gudvector.com/q/claim-token-1"
PORTAL_ENVIRONMENT = {
    "GVAS_PORTAL_BASE_URL": "https://gudvector.com/",
    "GVAS_PORTAL_API_TOKEN": PORTAL_TOKEN,
}


def portal_settings() -> PortalSettings:
    return PortalSettings(base_url="https://gudvector.com/", api_token=PORTAL_TOKEN)


def recipient(
    *,
    email: str | None = "jane@example.test",
    phone: str | None = "+19255551234",
    service_address: str | None = "123 Main St, Walnut Creek, CA",
) -> CustomerRecipient:
    if email is None:
        assert phone is not None
        return CustomerRecipient(
            address=phone,
            address_kind=RecipientAddressKind.PHONE,
            display_name="Jane Doe",
            service_address=service_address,
        )
    return CustomerRecipient(
        address=email,
        address_kind=RecipientAddressKind.EMAIL,
        display_name="Jane Doe",
        phone=phone,
        service_address=service_address,
    )


def delivery_request(
    customer: CustomerRecipient | None = None, note: str | None = None
) -> CustomerDeliveryRequest:
    return CustomerDeliveryRequest(
        business_id=BUSINESS_ID,
        recipient=customer or recipient(),
        idempotency_key="quote-delivery:quote-1",
        subject="Your quote",
        body_text="1 × Inspection: $250.00\nTotal: $250.00",
        line_items=(
            CustomerDeliveryLineItem(description="Inspection", quantity=1, unit_price_minor=25_000),
            CustomerDeliveryLineItem(description="Air sample", quantity=2, unit_price_minor=12_500),
        ),
        currency="USD",
        note=note,
    )


def portal_ok(emailed: bool = True) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "ok": True,
            "id": "quo_1",
            "claimToken": "claim-token-1",
            "quoteUrl": QUOTE_URL,
            "emailed": emailed,
            "smsOk": False,
        },
    )


@pytest.mark.asyncio
async def test_portal_create_posts_the_draft_with_bearer_auth_and_keeps_the_link(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return portal_ok()

    ledger = SqlPortalHandoffLedger(session_factory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = PortalQuoteDelivery(portal_settings(), client, ledger)
        receipt = await adapter.deliver(delivery_request())

    request = seen[0]
    assert str(request.url) == "https://gudvector.com/api/quotes"
    assert request.headers["Authorization"] == f"Bearer {PORTAL_TOKEN}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.read()) == {
        "customerName": "Jane Doe",
        "customerEmail": "jane@example.test",
        "customerPhone": "+19255551234",
        "serviceAddress": "123 Main St, Walnut Creek, CA",
        "items": [
            {"description": "Inspection", "quantity": 1, "amountCents": 25_000},
            {"description": "Air sample", "quantity": 2, "amountCents": 12_500},
        ],
        "billing": "one_time",
        "sendEmail": True,
    }
    assert receipt.status is DeliveryStatus.ACCEPTED
    assert receipt.provider_message_id == "quo_1"
    assert receipt.customer_link == QUOTE_URL
    assert receipt.emailed is True
    recorded = await ledger.find("quote-delivery:quote-1")
    assert recorded is not None
    assert recorded.claim_token == "claim-token-1"  # noqa: S105
    assert recorded.quote_url == QUOTE_URL


@pytest.mark.asyncio
async def test_portal_replay_returns_the_recorded_link_without_a_second_create() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return portal_ok(emailed=False)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = PortalQuoteDelivery(portal_settings(), client, InMemoryPortalHandoffLedger())
        first = await adapter.deliver(delivery_request())
        replay = await adapter.deliver(delivery_request())

    assert calls == 1
    assert replay == first
    assert replay.customer_link == QUOTE_URL
    assert replay.emailed is False


@pytest.mark.asyncio
async def test_portal_phone_only_customer_does_not_ask_for_email() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return portal_ok(emailed=False)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = PortalQuoteDelivery(portal_settings(), client, InMemoryPortalHandoffLedger())
        await adapter.deliver(delivery_request(recipient(email=None, service_address=None)))

    body = json.loads(seen[0].read())
    assert body["customerPhone"] == "+19255551234"
    assert "customerEmail" not in body
    assert "serviceAddress" not in body
    assert "note" not in body
    assert body["sendEmail"] is False


@pytest.mark.asyncio
async def test_portal_create_carries_the_customer_note() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return portal_ok()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = PortalQuoteDelivery(portal_settings(), client, InMemoryPortalHandoffLedger())
        await adapter.deliver(delivery_request(note="We'll be there Tuesday"))

    assert json.loads(seen[0].read())["note"] == "We'll be there Tuesday"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": f"bad token {PORTAL_TOKEN}"}),
        httpx.Response(401, json={"error": "Unauthorized"}),
        httpx.Response(500, text=f"boom {PORTAL_TOKEN}"),
        httpx.Response(200, json={"ok": False}),
        httpx.Response(200, text="not json"),
    ],
)
async def test_portal_errors_are_sanitized_and_leave_the_command_retryable(
    response: httpx.Response,
) -> None:
    ledger = InMemoryPortalHandoffLedger()

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        adapter = PortalQuoteDelivery(portal_settings(), client, ledger)
        with pytest.raises(PortalDeliveryError) as error:
            await adapter.deliver(delivery_request())

    assert PORTAL_TOKEN not in str(error.value)
    assert "Unauthorized" not in str(error.value)
    assert await ledger.find("quote-delivery:quote-1") is None


@pytest.mark.asyncio
async def test_portal_refuses_what_it_cannot_represent_before_calling() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = PortalQuoteDelivery(portal_settings(), client, InMemoryPortalHandoffLedger())
        euros = delivery_request().model_copy(update={"currency": "EUR"})
        with pytest.raises(PortalDeliveryError, match="USD"):
            await adapter.deliver(euros)
        unstructured = delivery_request().model_copy(update={"line_items": ()})
        with pytest.raises(PortalDeliveryError, match="line items"):
            await adapter.deliver(unstructured)


def draft(description: str) -> QuoteDraftProposal:
    return QuoteDraftProposal(
        quote_id=uuid4(),
        business_id=BUSINESS_ID,
        recipient=recipient(),
        currency="USD",
        line_items=(QuoteLineItem(description=description, quantity=1, unit_price_minor=25_000),),
    )


def test_customer_text_fits_one_segment_and_never_cuts_the_link() -> None:
    short = customer_quote_text(draft("Mold inspection"), QUOTE_URL)
    assert short == f"Your Güd Vector quote for Mold inspection is ready: {QUOTE_URL}"
    assert len(short) <= SMS_SEGMENT_LIMIT

    long_description = "Full   attic and crawlspace mold inspection " * 6
    long = customer_quote_text(draft(long_description), QUOTE_URL)
    assert len(long) == SMS_SEGMENT_LIMIT
    assert long.endswith(f" is ready: {QUOTE_URL}")
    assert long.startswith("Your Güd Vector quote for Full attic and crawlspace")

    huge_link = "https://gudvector.com/q/" + "t" * 150
    only_link = customer_quote_text(draft("Inspection"), huge_link)
    assert only_link.endswith(huge_link)
    assert only_link.startswith("Your Güd Vector quote is ready: ")


class SenderFake:
    def __init__(self, *, fail: bool = False) -> None:
        self.requests: list[TelnyxSendRequest] = []
        self.fail = fail

    async def send_message(self, request: TelnyxSendRequest) -> TelnyxSendResult:
        self.requests.append(request)
        if self.fail:
            return TelnyxSendResult(detail="telnyx returned http 422")
        return TelnyxSendResult(message_id=f"msg-{len(self.requests)}")


def installation(business_id: BusinessId = BUSINESS_ID) -> TelnyxInstallation:
    return TelnyxInstallation(
        business_id=business_id,
        telnyx_number="+19255550000",
        owner_numbers=frozenset({"+19255550001"}),
    )


def text_request(business_id: BusinessId = BUSINESS_ID) -> CustomerTextRequest:
    return CustomerTextRequest(
        business_id=business_id,
        phone_number="+19255551234",
        text=f"Your Güd Vector quote for Inspection is ready: {QUOTE_URL}",
        idempotency_key="quote-text:quote-1",
    )


@pytest.mark.asyncio
async def test_customer_text_goes_from_the_business_number_once() -> None:
    sender = SenderFake()
    adapter = TelnyxCustomerTextAdapter(
        sender, (installation(),), InMemoryTelnyxDeliveryLedger(), clock=lambda: FAKE_NOW
    )

    receipt = await adapter.send_text(text_request())
    replay = await adapter.send_text(text_request())

    assert receipt == replay
    assert receipt.status is DeliveryStatus.ACCEPTED
    assert len(sender.requests) == 1
    sent = sender.requests[0]
    assert sent.from_number == "+19255550000"
    assert sent.to_number == "+19255551234"
    assert sent.text.endswith(QUOTE_URL)
    assert sent.idempotency_key == f"customer-text:{BUSINESS_ID}:quote-text:quote-1"


@pytest.mark.asyncio
async def test_customer_text_failures_raise_without_recording() -> None:
    ledger = InMemoryTelnyxDeliveryLedger()
    failing = TelnyxCustomerTextAdapter(SenderFake(fail=True), (installation(),), ledger)
    with pytest.raises(TelnyxDeliveryError):
        await failing.send_text(text_request())
    assert await ledger.find(f"customer-text:{BUSINESS_ID}:quote-text:quote-1") is None

    unrouted = TelnyxCustomerTextAdapter(SenderFake(), (installation(),), ledger)
    with pytest.raises(TelnyxRoutingError):
        await unrouted.send_text(text_request(BusinessId(uuid4())))


class PortalLikeDelivery:
    """A delivery port that, like the portal, hosts the quote and reports emailing."""

    def __init__(self, *, emailed: bool) -> None:
        self.emailed = emailed
        self.requests: list[CustomerDeliveryRequest] = []

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        self.requests.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id="quo_1",
            occurred_at=FAKE_NOW,
            customer_link=QUOTE_URL,
            emailed=self.emailed,
        )


class CustomerTextFake:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[CustomerTextRequest] = []

    async def send_text(self, request: CustomerTextRequest) -> DeliveryReceipt:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("telnyx key 'KEY-secret' rejected")
        return DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED, provider_message_id="msg-1", occurred_at=FAKE_NOW
        )


class PhoneAwareDrafting:
    def __init__(self, customer: CustomerRecipient) -> None:
        self.customer = customer

    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal:
        return QuoteDraftProposal(
            quote_id=request.quote_id,
            business_id=request.business_id,
            recipient=self.customer,
            currency="USD",
            line_items=(
                QuoteLineItem(description="Mold inspection", quantity=1, unit_price_minor=25_000),
            ),
            confidence=1,
        )


async def consent_to_texts(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    email: str,
    consent: bool = True,
) -> None:
    async with session_factory() as session:
        customers = SqlCustomerRepository(session)
        customer = await customers.upsert(
            business_id, email, display_name=None, phone=None, now=FAKE_NOW
        )
        await customers.set_sms_consent(business_id, customer.customer_id, consent, FAKE_NOW)
        await session.commit()


async def approved_quote(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    customer: CustomerRecipient,
    emailed: bool,
    text: CustomerTextFake | None,
    sms_consent: bool | None = None,
) -> tuple[OwnerReplyFake, PortalLikeDelivery]:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    if sms_consent is not None and customer.email_address is not None:
        await consent_to_texts(session_factory, business_id, customer.email_address, sms_consent)
    owner_replies = OwnerReplyFake()
    delivery = PortalLikeDelivery(emailed=emailed)
    application = build_application(
        ApplicationPorts(
            owner_replies=owner_replies,
            quote_drafting=PhoneAwareDrafting(customer),
            quote_delivery=delivery,
            customer_text=text,
            transcription=TranscriptionFake({}),
            completeness_review=MarkerCompletenessReviewer(),
            checklist_evidence=MarkerChecklistEvidenceAttributor(),
            report_generation=DeterministicReportGenerator(),
        ),
        session_factory=session_factory,
        now=Clock(),
    )
    worker = immediate_worker(application)
    await application.ingest_service.ingest(
        inbound(business_id, "quote: mold inspection 250", message_key="quote-1")
    )
    await worker.drain()
    approve = inbound(business_id, "approve", message_key="quote-1-approve")
    await application.ingest_service.ingest(approve)
    await application.ingest_service.ingest(approve)
    for _ in range(4):
        await worker.drain()
    return owner_replies, delivery


async def command_statuses(
    session_factory: async_sessionmaker[AsyncSession], command_type: str
) -> list[str]:
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(OutboxMessage).where(OutboxMessage.command_type == command_type)
            )
        ).all()
    return [row.status for row in rows]


async def owner_sent_at(session_factory: async_sessionmaker[AsyncSession]) -> object:
    async with session_factory() as session:
        (row,) = (await session.scalars(select(QuoteRecord))).all()
        quote = await SqlQuoteRepository(session).get(BusinessId(row.business_id), QuoteId(row.id))
    assert quote is not None
    return owner_quote_payload(quote)["sentAt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("email", "emailed", "sms_consent", "text_fails", "sent"),
    [
        ("jane@example.test", True, True, False, True),
        ("jane@example.test", True, None, False, True),
        ("jane@example.test", False, True, False, True),
        ("jane@example.test", False, None, False, False),
        ("jane@example.test", False, True, True, False),
        (None, False, None, False, False),
    ],
    ids=[
        "emailed-and-texted",
        "emailed-only",
        "texted-only",
        "link-to-forward",
        "text-failed",
        "phone-only",
    ],
)
async def test_a_quote_counts_as_sent_only_once_it_reached_the_customer(
    session_factory: async_sessionmaker[AsyncSession],
    email: str | None,
    emailed: bool,
    sms_consent: bool | None,
    text_fails: bool,
    sent: bool,
) -> None:
    await approved_quote(
        session_factory,
        customer=recipient(email=email),
        emailed=emailed,
        text=CustomerTextFake(fail=text_fails),
        sms_consent=sms_consent,
    )

    assert await owner_sent_at(session_factory) == (FAKE_NOW.isoformat() if sent else None)


@pytest.mark.asyncio
async def test_emailed_and_texted_quote_confirms_both_channels_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    text = CustomerTextFake()
    owner_replies, delivery = await approved_quote(
        session_factory, customer=recipient(), emailed=True, text=text, sms_consent=True
    )

    assert len(delivery.requests) == 1
    assert delivery.requests[0].recipient.phone_number == "+19255551234"
    assert delivery.requests[0].line_items[0].unit_price_minor == 25_000
    assert len(text.requests) == 1
    assert text.requests[0].phone_number == "+19255551234"
    assert text.requests[0].text == (
        f"Your Güd Vector quote for Mold inspection is ready: {QUOTE_URL}"
    )
    assert text.requests[0].idempotency_key.startswith("quote-text:")
    confirmations = texts_of(owner_replies, "Quote for Jane Doe is ready")
    assert confirmations == [
        f"Quote for Jane Doe is ready: {QUOTE_URL}\n"
        "Emailed to jane@example.test; texting +19255551234."
    ]
    assert await command_statuses(session_factory, "customer_quote.text") == [
        OutboxStatus.SUCCEEDED.value
    ]


@pytest.mark.asyncio
async def test_failed_text_dead_letters_with_the_emailed_flag_and_the_link(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner_replies, delivery = await approved_quote(
        session_factory,
        customer=recipient(),
        emailed=True,
        text=CustomerTextFake(fail=True),
        sms_consent=True,
    )

    assert len(delivery.requests) == 1
    assert await command_statuses(session_factory, "customer_quote.deliver") == [
        OutboxStatus.SUCCEEDED.value
    ]
    assert await command_statuses(session_factory, "customer_quote.text") == [
        OutboxStatus.DEAD.value
    ]
    notices = texts_of(owner_replies, "The quote was created, but the text")
    assert notices == [
        "The quote was created, but the text to the customer could not be sent.\n"
        "The customer was emailed the link at jane@example.test.\n"
        f"Forward the quote link to the customer yourself. {QUOTE_URL}"
    ]
    assert not any(
        "KEY-secret" in part.text
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, TextPart)
    )


@pytest.mark.asyncio
async def test_failed_text_after_an_unemailed_quote_says_nothing_reached_the_customer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner_replies, _ = await approved_quote(
        session_factory,
        customer=recipient(),
        emailed=False,
        text=CustomerTextFake(fail=True),
        sms_consent=True,
    )

    notices = texts_of(owner_replies, "The quote was created, but the text")
    assert len(notices) == 1
    assert "was not emailed either, so nothing has reached them" in notices[0]
    assert notices[0].endswith(QUOTE_URL)
    confirmations = texts_of(owner_replies, "Quote for Jane Doe is ready")
    assert confirmations == [
        f"Quote for Jane Doe is ready: {QUOTE_URL}\nTexting +19255551234. No email was sent."
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("sms_consent", [None, False])
async def test_quote_is_never_texted_without_sms_consent(
    session_factory: async_sessionmaker[AsyncSession], sms_consent: bool | None
) -> None:
    text = CustomerTextFake()
    owner_replies, delivery = await approved_quote(
        session_factory, customer=recipient(), emailed=True, text=text, sms_consent=sms_consent
    )

    assert len(delivery.requests) == 1
    assert text.requests == []
    assert await command_statuses(session_factory, "customer_quote.text") == []
    assert texts_of(owner_replies, "Quote for Jane Doe is ready") == [
        f"Quote for Jane Doe is ready: {QUOTE_URL}\n"
        "Emailed to jane@example.test. No SMS consent on file, so no text."
    ]


@pytest.mark.asyncio
async def test_phone_only_quote_has_no_consent_so_it_is_not_texted(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    text = CustomerTextFake()
    owner_replies, _ = await approved_quote(
        session_factory, customer=recipient(email=None), emailed=False, text=text
    )

    assert text.requests == []
    assert await command_statuses(session_factory, "customer_quote.text") == []
    assert texts_of(owner_replies, "Quote for Jane Doe is ready") == [
        f"Quote for Jane Doe is ready: {QUOTE_URL}\n"
        "Not emailed or texted; forward the link to the customer yourself."
    ]


@pytest.mark.asyncio
async def test_unemailed_quote_without_a_phone_tells_the_owner_to_forward_the_link(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    text = CustomerTextFake()
    owner_replies, _ = await approved_quote(
        session_factory, customer=recipient(phone=None), emailed=False, text=text
    )

    assert text.requests == []
    assert await command_statuses(session_factory, "customer_quote.text") == []
    confirmations = texts_of(owner_replies, "Quote for Jane Doe is ready")
    assert confirmations == [
        f"Quote for Jane Doe is ready: {QUOTE_URL}\n"
        "Not emailed or texted; forward the link to the customer yourself."
    ]


@pytest.mark.asyncio
async def test_emailed_quote_without_a_phone_or_text_port_confirms_email_only(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner_replies, _ = await approved_quote(
        session_factory, customer=recipient(phone=None), emailed=True, text=None
    )

    confirmations = texts_of(owner_replies, "Quote for Jane Doe is ready")
    assert confirmations == [
        f"Quote for Jane Doe is ready: {QUOTE_URL}\n"
        "Emailed to jane@example.test. No phone on file, so no text."
    ]


@pytest.fixture
def production_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


@pytest.mark.usefixtures("production_environment")
def test_without_portal_settings_quotes_are_emailed_and_never_texted() -> None:
    runtime = build_production_runtime(load_production_settings())
    service = runtime.application.quote_delivery_service
    # With no portal the composite degenerates to the email adapter for every
    # quote, hosted-site links included.
    assert isinstance(service._delivery_port, SiteAwareQuoteDelivery)  # noqa: SLF001
    assert isinstance(service._delivery_port._default, ResendQuoteDeliveryAdapter)  # noqa: SLF001
    assert service._delivery_port._portal is None  # noqa: SLF001
    assert service._texts_customers is False  # noqa: SLF001
    assert runtime.application.dispatcher._quote_text is None  # noqa: SLF001


@pytest.mark.usefixtures("production_environment")
def test_portal_settings_wire_the_portal_and_telnyx_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PORTAL_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    runtime = build_production_runtime(load_production_settings())
    service = runtime.application.quote_delivery_service
    # The portal still takes quotes without a hosted link; the composite holds it.
    assert isinstance(service._delivery_port, SiteAwareQuoteDelivery)  # noqa: SLF001
    assert isinstance(service._delivery_port._portal, PortalQuoteDelivery)  # noqa: SLF001
    assert service._texts_customers is False  # noqa: SLF001

    for name, value in TELNYX_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    runtime = build_production_runtime(load_production_settings())
    service = runtime.application.quote_delivery_service
    assert isinstance(service._delivery_port, SiteAwareQuoteDelivery)  # noqa: SLF001
    assert isinstance(service._delivery_port._portal, PortalQuoteDelivery)  # noqa: SLF001
    assert service._texts_customers is True  # noqa: SLF001
    assert runtime.application.dispatcher._quote_text is not None  # noqa: SLF001


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_partial_portal_settings_without_leaking_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GVAS_PORTAL_API_TOKEN", PORTAL_TOKEN)
    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()
    message = str(error.value)
    assert "GVAS_PORTAL_BASE_URL" in message
    assert "GVAS_PORTAL_API_TOKEN" not in message
    assert PORTAL_TOKEN not in message

    monkeypatch.delenv("GVAS_PORTAL_API_TOKEN")
    monkeypatch.setenv("GVAS_PORTAL_BASE_URL", "https://gudvector.com")
    with pytest.raises(ProductionConfigurationError, match="GVAS_PORTAL_API_TOKEN"):
        load_production_settings()


def test_portal_settings_are_optional_as_a_set() -> None:
    assert PortalSettings(base_url="", api_token="").is_configured is False
    assert PortalSettings(base_url="", api_token="").is_partially_configured is False
    assert PortalSettings(base_url="https://x", api_token="").is_partially_configured is True
    assert PortalSettings(base_url="https://x", api_token="t").is_configured is True  # noqa: S106


@pytest.mark.asyncio
async def test_calendly_invitee_phone_and_address_reach_the_recipient() -> None:
    reminder = {
        **invitee("Jane Doe", "jane@example.test", []),
        "text_reminder_number": "+19255551234",
    }
    answered = invitee(
        "Bo Lee",
        "bo@example.test",
        [
            {"question": "Best phone number?", "answer": "(925) 555-0199", "position": 0},
            {"question": "Service address", "answer": "343 Thing Ave", "position": 1},
        ],
    )
    unusable = invitee(
        "No Phone",
        "none@example.test",
        [{"question": "Phone", "answer": "call me", "position": 0}],
    )
    recorder = Recorder(
        {
            "/scheduled_events": [
                httpx.Response(
                    200,
                    json={
                        "collection": [event(EVENT_A, {"type": "physical", "location": "1 A St"})],
                        "pagination": {"count": 1, "next_page_token": None},
                    },
                )
            ],
            "/scheduled_events/EVENT_A/invitees": [
                httpx.Response(
                    200,
                    json={
                        "collection": [reminder, answered, unusable],
                        "pagination": {"count": 3, "next_page_token": None},
                    },
                )
            ],
        }
    )
    lookup = CalendlyAppointmentLookup(calendly_settings(), calendly_client(recorder))

    found = await lookup.find(CALENDLY_WINDOW)

    assert [(a.invitee_name, a.invitee_phone, a.address) for a in found] == [
        ("Jane Doe", "+19255551234", "1 A St"),
        ("Bo Lee", "+19255550199", "1 A St"),
        ("No Phone", None, "1 A St"),
    ]

    mapped = _appointment_recipient(found[1])
    assert mapped.email_address == "bo@example.test"
    assert mapped.phone_number == "+19255550199"
    assert mapped.service_address == "1 A St"
    assert mapped.display_name == "Bo Lee"
