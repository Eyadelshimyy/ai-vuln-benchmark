"""Hosted-quote backend: claim tokens, the public quote API, Stripe checkout
handoff and webhook processing, CORS and the approve→delivery precedence."""

import hashlib
import hmac
import json
import re
import time
from argparse import Namespace
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import FAKE_NOW, CustomerDeliveryFake, OwnerReplyFake, TranscriptionFake
from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.application.public_quotes import (
    UnknownPaymentSessionError,
    checkout_line_items,
)
from gvas.application.quotes import SiteAwareQuoteDelivery
from gvas.composition import Application, ApplicationPorts, build_application
from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import BusinessId, QuoteId
from gvas.domain.messages import (
    CustomerDeliveryRequest,
    CustomerTextRequest,
    DeliveryReceipt,
)
from gvas.domain.payments import (
    PaymentCheckoutRequest,
    PaymentCheckoutResult,
    PaymentLineItem,
    QuotePaymentConflictError,
)
from gvas.domain.quotes import (
    QuoteConcurrencyError,
    QuoteDraftProposal,
    QuoteLineItem,
    claim_token_matches,
    hash_claim_token,
    new_claim_token,
    public_quote_id,
)
from gvas.domain.repositories import normalize_site_url
from gvas.infrastructure.models import OutboundMessage, QuoteRecord
from gvas.infrastructure.payment_models import QuotePayment
from gvas.infrastructure.payment_repositories import SqlQuotePaymentRepository
from gvas.infrastructure.repositories import (
    SqlBusinessRepository,
    SqlQuoteRepository,
)
from gvas.infrastructure.stripe import StripeWebhookVerifier
from gvas.infrastructure.stripe.api import StripeCheckout, checkout_form
from gvas.infrastructure.stripe.config import StripeSettings
from gvas.infrastructure.stripe.events import parse_checkout_event
from gvas.infrastructure.stripe.signature import SIGNATURE_HEADER, StripeSignatureError
from gvas.interfaces.configure_business import (
    ConfigureBusinessInputError,
    build_request,
)
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.public import PerIpRateLimiter, create_public_router
from test_composition import Clock, inbound, seed_business
from test_pilot_runtime import immediate_worker, texts_of
from test_portal_quote_handoff import PhoneAwareDrafting, consent_to_texts, recipient

SITE_URL = "https://gudvector.com"
CALENDLY_URL = "https://calendly.com/gudvector"
PUBLIC_KEY = "gvb_publishable_test"
DISPLAY_NAME = "Güd Vector"
WEBHOOK_SECRET = "whsec_test"  # noqa: S105
CHECKOUT_URL = "https://checkout.stripe.com/c/pay/cs_test_1"
SESSION_ID = "cs_test_1"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def sign(body: bytes, secret: str = WEBHOOK_SECRET, timestamp: int | None = None) -> str:
    """Independently compute the ``t=...,v1=...`` header the endpoint verifies."""

    ts = timestamp if timestamp is not None else int(time.time())
    digest = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={digest}"


def checkout_event(
    *,
    event_id: str = "evt_1",
    event_type: str = "checkout.session.completed",
    session_id: str = SESSION_ID,
    payment_intent: str | None = "pi_1",
    payment_status: str = "paid",
    metadata: dict[str, str] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": event_id,
            "type": event_type,
            "data": {
                "object": {
                    "id": session_id,
                    "payment_intent": payment_intent,
                    "payment_status": payment_status,
                    "metadata": {} if metadata is None else metadata,
                }
            },
        }
    ).encode()


class HostedEmailDelivery:
    """What the email adapter does with a hosted quote: report the link it sent."""

    def __init__(self) -> None:
        self.requests: list[CustomerDeliveryRequest] = []

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        self.requests.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id="email-1",
            occurred_at=FAKE_NOW,
            customer_link=request.quote_url,
            emailed=True,
        )


CheckoutHook = Callable[[], Awaitable[None]]


class CheckoutFake:
    """Hands out canned sessions in order; ``hook`` runs inside the provider
    call so a test can fire the webhook before the payment row exists."""

    def __init__(
        self,
        sessions: list[tuple[str, str, datetime | None]] | None = None,
        hook: CheckoutHook | None = None,
    ) -> None:
        self.sessions = list(sessions or [(SESSION_ID, CHECKOUT_URL, None)])
        self.hook = hook
        self.requests: list[PaymentCheckoutRequest] = []
        self.expired: list[str] = []

    async def expire_checkout(self, session_id: str) -> None:
        self.expired.append(session_id)

    async def create_checkout(self, request: PaymentCheckoutRequest) -> PaymentCheckoutResult:
        self.requests.append(request)
        session_id, url, expires_at = self.sessions[
            min(len(self.requests) - 1, len(self.sessions) - 1)
        ]
        if self.hook is not None:
            await self.hook()
        return PaymentCheckoutResult(
            session_id=session_id,
            checkout_url=url,
            payment_intent_id=None,
            expires_at=expires_at,
        )


class CustomerTextFake:
    def __init__(self) -> None:
        self.requests: list[CustomerTextRequest] = []

    async def send_text(self, request: CustomerTextRequest) -> DeliveryReceipt:
        self.requests.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED, provider_message_id="msg-1", occurred_at=FAKE_NOW
        )


async def hosted_quote(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    checkout: CheckoutFake | None = None,
    customer_text: CustomerTextFake | None = None,
    configure_site: bool = True,
    sms_consent: bool | None = None,
    customer_email: CustomerDeliveryFake | None = None,
) -> tuple[Application, OwnerReplyFake, HostedEmailDelivery, str]:
    """A business with a site URL whose owner approves one quote end to end."""

    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    if sms_consent is not None:
        await consent_to_texts(session_factory, business_id, "jane@example.test", sms_consent)
    if configure_site:
        async with session_factory() as session:
            await SqlBusinessRepository(session).configure_site(
                business_id,
                site_url=SITE_URL,
                display_name=DISPLAY_NAME,
                calendly_url=CALENDLY_URL,
                public_key=PUBLIC_KEY,
                now=NOW,
            )
            await session.commit()
    owner_replies = OwnerReplyFake()
    delivery = HostedEmailDelivery()
    application = build_application(
        ApplicationPorts(
            owner_replies=owner_replies,
            quote_drafting=PhoneAwareDrafting(recipient()),
            quote_delivery=delivery,
            customer_text=customer_text,
            customer_email=customer_email,
            payment_checkout=checkout,
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
    await application.ingest_service.ingest(
        inbound(business_id, "approve", message_key="approve-1")
    )
    for _ in range(4):
        await worker.drain()
    async with session_factory() as session:
        row = await session.scalar(select(QuoteRecord))
    assert row is not None and row.claim_token is not None
    return application, owner_replies, delivery, row.claim_token


def public_client(
    application: Application,
    *,
    verifier: StripeWebhookVerifier | None = None,
    cors_origins: frozenset[str] | None = None,
) -> httpx.AsyncClient:
    async def origins() -> frozenset[str]:
        return frozenset() if cors_origins is None else cors_origins

    app = create_app(
        routers=(
            create_public_router(
                application.public_quotes,
                webhook_verifier=verifier,
            ),
        ),
        cors_origins=origins,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_claim_tokens_are_url_safe_and_verified_by_hash() -> None:
    token = new_claim_token()
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token), token
    stored = hashlib.sha256(token.encode()).hexdigest()
    assert hash_claim_token(token) == stored
    assert claim_token_matches(token, stored)
    assert not claim_token_matches(new_claim_token(), stored)


def test_public_quote_id_is_opaque_and_stable() -> None:
    quote_id = QuoteId(uuid4())
    assert public_quote_id(quote_id).startswith("gvq_")
    assert public_quote_id(quote_id) == public_quote_id(quote_id)
    assert str(quote_id) not in public_quote_id(quote_id)


def test_checkout_line_items_collapse_when_a_discount_is_in_play() -> None:
    draft = QuoteDraftProposal(
        quote_id=uuid4(),
        business_id=BusinessId(uuid4()),
        recipient=recipient(),
        currency="USD",
        line_items=(
            QuoteLineItem(description="Inspection", quantity=1, unit_price_minor=25_000),
            QuoteLineItem(description="Air sample", quantity=2, unit_price_minor=12_500),
        ),
        tax_minor=1_000,
        discount_minor=5_000,
    )
    items = checkout_line_items(draft)
    assert items == (PaymentLineItem(description="Quote total", quantity=1, amount_minor=46_000),)
    no_discount = QuoteDraftProposal(
        quote_id=uuid4(),
        business_id=BusinessId(uuid4()),
        recipient=recipient(),
        currency="USD",
        line_items=(QuoteLineItem(description="Inspection", quantity=2, unit_price_minor=25_000),),
        tax_minor=1_000,
    )
    assert [item.description for item in checkout_line_items(no_discount)] == [
        "Inspection",
        "Tax",
    ]


def test_checkout_form_encodes_items_urls_and_metadata() -> None:
    request = PaymentCheckoutRequest(
        business_id=BusinessId(uuid4()),
        quote_id=uuid4(),
        client_reference="gvq_abc",
        currency="USD",
        line_items=(PaymentLineItem(description="Inspection", quantity=2, amount_minor=25_000),),
        success_url=f"{SITE_URL}/q/tok?paid=1",
        cancel_url=f"{SITE_URL}/q/tok",
        idempotency_key="quote-delivery:1",
        metadata={"gvas_quote_id": "gvq_abc", "business_id": "b-1"},
    )
    form = checkout_form(request)
    assert form["mode"] == "payment"
    assert form["line_items[0][price_data][currency]"] == "usd"
    assert form["line_items[0][price_data][unit_amount]"] == "25000"
    assert form["line_items[0][price_data][product_data][name]"] == "Inspection"
    assert form["line_items[0][quantity]"] == "2"
    assert form["success_url"] == f"{SITE_URL}/q/tok?paid=1"
    assert form["cancel_url"] == f"{SITE_URL}/q/tok"
    assert form["client_reference_id"] == "gvq_abc"
    assert form["metadata[gvas_quote_id]"] == "gvq_abc"


@pytest.mark.asyncio
async def test_stripe_checkout_posts_form_with_auth_and_idempotency() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["idempotency_key"] = request.headers.get("idempotency-key")
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"id": "cs_1", "url": CHECKOUT_URL})

    settings = StripeSettings(secret_key="sk_test_secret", webhook_secret="whsec_x")  # noqa: S106
    adapter = StripeCheckout(settings, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    result = await adapter.create_checkout(
        PaymentCheckoutRequest(
            business_id=BusinessId(uuid4()),
            quote_id=uuid4(),
            client_reference="gvq_1",
            currency="USD",
            line_items=(PaymentLineItem(description="Inspection", quantity=1, amount_minor=1),),
            success_url="https://x.test/ok",
            cancel_url="https://x.test/cancel",
            idempotency_key="quote-delivery:1",
        )
    )
    assert result.session_id == "cs_1"
    assert result.checkout_url == CHECKOUT_URL
    assert seen["authorization"] == "Bearer sk_test_secret"
    assert seen["idempotency_key"] == "quote-delivery:1"
    assert "mode=payment" in str(seen["body"])
    # The key itself must never land in what the adapter logs or raises.
    assert "sk_test_secret" not in str(seen["body"])


def test_webhook_signature_scheme() -> None:
    body = b'{"id":"evt_1"}'
    verifier = StripeWebhookVerifier(WEBHOOK_SECRET)
    verifier.verify(body, sign(body))
    with pytest.raises(StripeSignatureError):
        verifier.verify(body, sign(body + b"x"))
    with pytest.raises(StripeSignatureError):
        verifier.verify(body, None)
    with pytest.raises(StripeSignatureError):
        verifier.verify(body, "t=123,v1=bad")
    with pytest.raises(StripeSignatureError):
        verifier.verify(body, sign(body, timestamp=int(time.time()) - 10 * 60))
    with pytest.raises(StripeSignatureError):
        StripeWebhookVerifier("")


def test_rate_limiter_allows_burst_then_blocks() -> None:
    limiter = PerIpRateLimiter(per_minute=60, burst=2)
    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    assert limiter.allow("5.6.7.8")


@pytest.mark.asyncio
async def test_rate_limit_keys_on_the_rightmost_forwarded_hop(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, claim_token = await hosted_quote(session_factory)

    async def origins() -> frozenset[str]:
        return frozenset()

    app = create_app(
        routers=(
            create_public_router(
                application.public_quotes,
                rate_limiter=PerIpRateLimiter(per_minute=60, burst=1),
            ),
        ),
        cors_origins=origins,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Rotating the caller-supplied leftmost entries cannot dodge the
        # bucket: only the edge-appended rightmost hop counts.
        first = await client.get(
            f"/v1/quotes/{claim_token}", headers={"X-Forwarded-For": "spoof-1, 9.9.9.9"}
        )
        assert first.status_code == 200
        spoofed = await client.get(
            f"/v1/quotes/{claim_token}", headers={"X-Forwarded-For": "spoof-2, 9.9.9.9"}
        )
        assert spoofed.status_code == 429
        other = await client.get(
            f"/v1/quotes/{claim_token}", headers={"X-Forwarded-For": "spoof-3, 8.8.8.8"}
        )
        assert other.status_code == 200


def test_site_urls_must_be_bare_origins() -> None:
    assert normalize_site_url("HTTPS://GUDVECTOR.COM/") == "https://gudvector.com"
    assert normalize_site_url("https://gudvector.com:8443") == "https://gudvector.com:8443"
    # Claim tokens ride in these links, so cleartext only passes for local dev.
    assert normalize_site_url("http://localhost:3000") == "http://localhost:3000"
    assert normalize_site_url("http://127.0.0.1:3000") == "http://127.0.0.1:3000"
    for bad in (
        "gudvector.com",
        "ftp://gudvector.com",
        "http://gudvector.com",
        "https://gudvector.com/path",
        "https://gudvector.com?q=1",
        "https://gudvector.com#frag",
        "https://user:pass@gudvector.com",
        "https://:badport",
        "https:///nohost",
    ):
        with pytest.raises(ValueError):
            normalize_site_url(bad)


@pytest.mark.asyncio
async def test_hosted_approve_emails_the_quote_url_and_texts_it(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    text = CustomerTextFake()
    application, owner_replies, delivery, claim_token = await hosted_quote(
        session_factory, customer_text=text, sms_consent=True
    )
    assert delivery.requests[0].quote_url == f"{SITE_URL}/q/{claim_token}"
    assert delivery.requests[0].business_name == DISPLAY_NAME
    assert delivery.requests[0].subject == f"Your quote from {DISPLAY_NAME}"
    assert text.requests, "the hosted link should have been texted"
    assert text.requests[0].text.endswith(f"{SITE_URL}/q/{claim_token}")
    confirmations = texts_of(owner_replies, "Quote for Jane Doe is ready")
    assert confirmations and f"{SITE_URL}/q/{claim_token}" in confirmations[0]


@pytest.mark.asyncio
async def test_business_without_site_url_uses_the_generic_email(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, delivery, claim_token = await hosted_quote(
        session_factory, configure_site=False, checkout=CheckoutFake()
    )
    assert delivery.requests[0].quote_url is None
    # A claim token still exists so the site can be turned on later.
    assert claim_token
    # But until it is, the public API must not serve the quote at all.
    async with public_client(application) as client:
        assert (await client.get(f"/v1/quotes/{claim_token}")).status_code == 404
        for suffix in ("accept", "decline"):
            declined = await client.post(f"/v1/quotes/{claim_token}/{suffix}")
            assert declined.status_code == 404


@pytest.mark.asyncio
async def test_get_quote_returns_the_public_projection_and_marks_viewed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, claim_token = await hosted_quote(session_factory)
    async with public_client(application) as client:
        response = await client.get(f"/v1/quotes/{claim_token}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["business"] == {"displayName": DISPLAY_NAME, "siteUrl": SITE_URL}
    quote = payload["quote"]
    assert quote["id"].startswith("gvq_")
    assert quote["status"] == "viewed"
    assert quote["customerName"] == "Jane Doe"
    assert quote["serviceAddress"] == "123 Main St, Walnut Creek, CA"
    assert quote["subtotalCents"] == 25_000
    assert quote["totalCents"] == 25_000
    assert quote["currency"] == "USD"
    assert quote["items"] == [
        {"description": "Mold inspection", "quantity": 1, "amountCents": 25_000}
    ]
    assert quote["createdAt"] and quote["approvedAt"]
    # Nothing internal leaks: no business id, no row id, no owner details.
    serialized = json.dumps(payload)
    async with session_factory() as session:
        row = await session.scalar(select(QuoteRecord))
    assert row is not None
    assert str(row.id) not in serialized
    assert str(row.business_id) not in serialized
    assert "+19255551234" not in serialized

    async with public_client(application) as client:
        again = await client.get(f"/v1/quotes/{claim_token}")
    assert again.status_code == 200
    assert again.json()["quote"]["status"] == "viewed"


@pytest.mark.asyncio
async def test_unknown_and_garbage_tokens_get_the_same_404(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, _ = await hosted_quote(session_factory)
    async with public_client(application) as client:
        for path in ("missing-token", "../healthz", "a" * 200):
            response = await client.get(f"/v1/quotes/{path}")
            assert response.status_code == 404
        response = await client.get("/v1/quotes/definitely-not-a-claim-token")
        assert response.json() == {"detail": "not found"}


@pytest.mark.asyncio
async def test_accept_opens_one_checkout_session_and_reuses_it(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application) as client:
        first = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert first.status_code == 200
        assert first.json() == {"checkoutUrl": CHECKOUT_URL}
        second = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert second.status_code == 200
        assert second.json() == {"checkoutUrl": CHECKOUT_URL}
    assert len(checkout.requests) == 1
    request = checkout.requests[0]
    assert request.client_reference.startswith("gvq_")
    assert request.success_url == f"{SITE_URL}/q/{claim_token}?paid=1"
    assert request.cancel_url == f"{SITE_URL}/q/{claim_token}"
    assert request.metadata["gvas_quote_id"].startswith("gvq_")
    assert request.idempotency_key.startswith("quote-delivery:")
    async with session_factory() as session:
        payments = (await session.scalars(select(QuotePayment))).all()
        quote = await session.scalar(select(QuoteRecord))
    assert len(payments) == 1
    assert payments[0].checkout_session_id == SESSION_ID
    assert payments[0].status == "open"
    assert quote is not None and quote.customer_status == "accepted"


@pytest.mark.asyncio
async def test_accept_without_checkout_configured_is_503(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=None)
    async with public_client(application) as client:
        response = await client.post(f"/v1/quotes/{claim_token}/accept")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_decline_blocks_later_accept_and_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application) as client:
        declined = await client.post(f"/v1/quotes/{claim_token}/decline")
        assert declined.status_code == 200
        assert declined.json() == {"status": "declined"}
        again = await client.post(f"/v1/quotes/{claim_token}/decline")
        assert again.status_code == 200
        accept = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert accept.status_code == 409


@pytest.mark.asyncio
async def test_webhook_marks_paid_notifies_owner_and_replays_safely(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, owner_replies, _, claim_token = await hosted_quote(
        session_factory, checkout=checkout
    )
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        accepted = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert accepted.status_code == 200
        body = checkout_event()
        delivered = await client.post(
            "/webhooks/stripe",
            content=body,
            headers={SIGNATURE_HEADER: sign(body)},
        )
        assert delivered.status_code == 200
        assert delivered.json() == {"status": "recorded"}
        # A second delivery of the same event id is answered without redoing work.
        replayed = await client.post(
            "/webhooks/stripe",
            content=body,
            headers={SIGNATURE_HEADER: sign(body)},
        )
        assert replayed.status_code == 200
        assert replayed.json() == {"status": "ignored"}
        # A paid quote can no longer be accepted or declined.
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 409
        assert (await client.post(f"/v1/quotes/{claim_token}/decline")).status_code == 409
        got = await client.get(f"/v1/quotes/{claim_token}")
        assert got.json()["quote"]["status"] == "paid"

    async with session_factory() as session:
        payment = await session.scalar(select(QuotePayment))
        quote = await session.scalar(select(QuoteRecord))
        notices = (
            await session.scalars(
                select(OutboundMessage).where(OutboundMessage.correlation_id.like("%:paid"))
            )
        ).all()
    assert payment is not None and payment.status == "paid"
    assert payment.payment_intent_id == "pi_1"
    assert quote is not None and quote.customer_status == "paid"
    assert len(notices) == 1

    worker = immediate_worker(application)
    await worker.drain()
    paid_notices = texts_of(owner_replies, "Quote gvq_")
    assert paid_notices == [
        f"Quote {public_quote_id(QuoteId(quote.id))} for Jane Doe — USD 250.00 paid"
    ]


@pytest.mark.asyncio
async def test_webhook_rejects_unsigned_and_unknown_events(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, _ = await hosted_quote(session_factory, checkout=CheckoutFake())
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        body = checkout_event()
        assert (await client.post("/webhooks/stripe", content=body)).status_code == 401
        wrong = await client.post(
            "/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body, "whsec_wrong")}
        )
        assert wrong.status_code == 401
        unknown = checkout_event(event_type="charge.refunded")
        signed = await client.post(
            "/webhooks/stripe",
            content=unknown,
            headers={SIGNATURE_HEADER: sign(unknown)},
        )
        assert signed.status_code == 200
        assert signed.json() == {"status": "ignored"}


@pytest.mark.asyncio
async def test_booking_link_returns_the_publishable_details(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, _ = await hosted_quote(session_factory)
    async with public_client(application) as client:
        ok = await client.get(f"/v1/businesses/{PUBLIC_KEY}/booking-link")
        assert ok.status_code == 200
        assert ok.json() == {"calendlyUrl": CALENDLY_URL, "displayName": DISPLAY_NAME}
        missing = await client.get("/v1/businesses/gvb_nope/booking-link")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_cors_answers_only_for_configured_site_origins(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, claim_token = await hosted_quote(session_factory)
    async with public_client(application, cors_origins=frozenset({SITE_URL})) as client:
        preflight = await client.options(
            f"/v1/quotes/{claim_token}",
            headers={
                "Origin": SITE_URL,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == SITE_URL
        assert "GET" in preflight.headers["access-control-allow-methods"]
        denied = await client.options(
            f"/v1/quotes/{claim_token}",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in denied.headers
        ok = await client.get(f"/v1/quotes/{claim_token}", headers={"Origin": SITE_URL})
        assert ok.headers["access-control-allow-origin"] == SITE_URL
        foreign = await client.get(
            f"/v1/quotes/{claim_token}", headers={"Origin": "https://evil.example"}
        )
        assert "access-control-allow-origin" not in foreign.headers


@pytest.mark.asyncio
async def test_site_aware_delivery_sends_hosted_quotes_to_email_and_others_to_portal() -> None:
    default = HostedEmailDelivery()

    class Portalish:
        def __init__(self) -> None:
            self.requests: list[CustomerDeliveryRequest] = []

        async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
            self.requests.append(request)
            return DeliveryReceipt(status=DeliveryStatus.ACCEPTED, occurred_at=FAKE_NOW)

    portal = Portalish()
    composite = SiteAwareQuoteDelivery(default, portal)
    request = CustomerDeliveryRequest(
        business_id=BusinessId(uuid4()),
        recipient=recipient(),
        idempotency_key="quote-delivery:1",
        body_text="body",
    )
    await composite.deliver(request)
    assert portal.requests and not default.requests
    hosted = request.model_copy(update={"quote_url": f"{SITE_URL}/q/tok"})
    await composite.deliver(hosted)
    assert default.requests and len(portal.requests) == 1


@pytest.mark.asyncio
async def test_unpaid_completion_waits_for_the_async_outcome(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A delayed-method checkout reports completed before the money moves, and
    the pending attempt can no longer expire out from under the settlement."""

    pending_session = (SESSION_ID, CHECKOUT_URL, NOW - timedelta(hours=1))
    checkout = CheckoutFake(sessions=[pending_session])
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        early = checkout_event(event_id="evt_unpaid", payment_status="unpaid")
        answered = await client.post(
            "/webhooks/stripe", content=early, headers={SIGNATURE_HEADER: sign(early)}
        )
        assert answered.status_code == 200
        assert answered.json() == {"status": "recorded"}
        got = await client.get(f"/v1/quotes/{claim_token}")
        assert got.json()["quote"]["status"] == "accepted"
        # The session's own expiry passed, but a completed delayed payment is
        # pinned: re-accepting returns it instead of opening a second charge.
        again = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert again.status_code == 200
        assert again.json() == {"checkoutUrl": CHECKOUT_URL}
        assert len(checkout.requests) == 1
        settled = checkout_event(
            event_id="evt_settled",
            event_type="checkout.session.async_payment_succeeded",
        )
        done = await client.post(
            "/webhooks/stripe", content=settled, headers={SIGNATURE_HEADER: sign(settled)}
        )
        assert done.status_code == 200
        got = await client.get(f"/v1/quotes/{claim_token}")
        assert got.json()["quote"]["status"] == "paid"
    async with session_factory() as session:
        payments = (await session.scalars(select(QuotePayment))).all()
    assert [p.status for p in payments] == ["paid"]


@pytest.mark.asyncio
async def test_webhook_before_the_payment_record_retries_later(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Stripe can answer the webhook before accept commits the session row."""

    checkout = CheckoutFake()
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    early = checkout_event(event_id="evt_early")

    async def fire_webhook_early() -> None:
        with pytest.raises(UnknownPaymentSessionError):
            await application.public_quotes.record_payment_event(parse_checkout_event(early))

    checkout.hook = fire_webhook_early
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        # The premature delivery was refused, not recorded: a redelivery works.
        delivered = await client.post(
            "/webhooks/stripe", content=early, headers={SIGNATURE_HEADER: sign(early)}
        )
        assert delivered.status_code == 200
        assert delivered.json() == {"status": "recorded"}
        got = await client.get(f"/v1/quotes/{claim_token}")
        assert got.json()["quote"]["status"] == "paid"


@pytest.mark.asyncio
async def test_webhook_for_an_unknown_session_is_a_503_retry(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, _ = await hosted_quote(session_factory, checkout=CheckoutFake())
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        body = checkout_event(session_id="cs_never_created")
        response = await client.post(
            "/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body)}
        )
        assert response.status_code == 503
        # Nothing was recorded, so a redelivery is free to keep retrying.
        again = await client.post(
            "/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body)}
        )
        assert again.status_code == 503


@pytest.mark.asyncio
async def test_a_checkout_tagged_for_another_deployment_is_ignored_not_retried(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        assert checkout.requests[0].metadata["gvas_deployment"] == "production"
        demo = checkout_event(
            event_id="evt_demo", session_id="cs_demo_1", metadata={"gvas_deployment": "demo"}
        )
        response = await client.post(
            "/webhooks/stripe", content=demo, headers={SIGNATURE_HEADER: sign(demo)}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ignored"}
        # Even one naming production's own session is not acted on.
        stray = checkout_event(event_id="evt_stray", metadata={"gvas_deployment": "demo"})
        response = await client.post(
            "/webhooks/stripe", content=stray, headers={SIGNATURE_HEADER: sign(stray)}
        )
        assert response.json() == {"status": "ignored"}
        got = await client.get(f"/v1/quotes/{claim_token}")
        assert got.json()["quote"]["status"] != "paid"


@pytest.mark.asyncio
async def test_expired_session_is_retired_and_a_fresh_one_opens(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    expired_session = (
        "cs_test_expired",
        "https://checkout.stripe.com/c/pay/cs_test_expired",
        NOW - timedelta(hours=1),
    )
    fresh_session = (
        "cs_test_fresh",
        "https://checkout.stripe.com/c/pay/cs_test_fresh",
        None,
    )
    checkout = CheckoutFake(sessions=[expired_session, fresh_session])
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application) as client:
        first = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert first.status_code == 200
        assert first.json() == {"checkoutUrl": expired_session[1]}
        second = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert second.status_code == 200
        assert second.json() == {"checkoutUrl": fresh_session[1]}
    assert len(checkout.requests) == 2
    assert checkout.requests[0].idempotency_key != checkout.requests[1].idempotency_key
    assert checkout.requests[1].idempotency_key.endswith(":attempt-2")
    async with session_factory() as session:
        payments = (
            await session.scalars(select(QuotePayment).order_by(QuotePayment.created_at))
        ).all()
    assert [p.status for p in payments] == ["expired", "open"]


@pytest.mark.asyncio
async def test_failed_payment_lets_the_customer_try_again(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake(
        sessions=[(SESSION_ID, CHECKOUT_URL, None), ("cs_test_2", CHECKOUT_URL + "2", None)]
    )
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        failed = checkout_event(
            event_id="evt_failed", event_type="checkout.session.async_payment_failed"
        )
        answered = await client.post(
            "/webhooks/stripe", content=failed, headers={SIGNATURE_HEADER: sign(failed)}
        )
        assert answered.status_code == 200
        assert answered.json() == {"status": "recorded"}
        retry = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert retry.status_code == 200
        assert retry.json() == {"checkoutUrl": CHECKOUT_URL + "2"}
    async with session_factory() as session:
        payments = (
            await session.scalars(select(QuotePayment).order_by(QuotePayment.created_at))
        ).all()
    assert [p.status for p in payments] == ["failed", "open"]


@pytest.mark.asyncio
async def test_accept_losing_a_concurrent_write_answers_409(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A racing decline or accept must not produce a payable session from stale
    state: the loser answers 409 and a retry converges on the persisted row."""

    application, _, _, claim_token = await hosted_quote(session_factory, checkout=CheckoutFake())
    real_save = SqlQuoteRepository.save
    calls = 0

    async def fail_once(self: SqlQuoteRepository, quote: object, *, expected_version: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise QuoteConcurrencyError("a racing write won")
        await real_save(self, quote, expected_version=expected_version)  # type: ignore[arg-type]

    monkeypatch.setattr(SqlQuoteRepository, "save", fail_once)
    async with public_client(application) as client:
        conflicted = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert conflicted.status_code == 409
        # The retry lands on the committed state and opens checkout normally.
        retried = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert retried.status_code == 200
        assert retried.json() == {"checkoutUrl": CHECKOUT_URL}


def test_configure_rejects_unusable_public_keys_and_booking_links() -> None:
    def arguments(**overrides: str | bool | None) -> Namespace:
        base: dict[str, str | bool | None] = {
            "business_id": str(uuid4()),
            "site_url": None,
            "display_name": None,
            "calendly_url": CALENDLY_URL,
            "stripe_account_id": None,
            "public_key": "gvb_ok-1.~_x",
            "intake_brief": None,
            "intake_questions": None,
            "intake_opening": None,
            "notification_email": None,
            "clear_notification_email": False,
        }
        base.update(overrides)
        return Namespace(**base)

    request = build_request(arguments())
    assert request.public_key == "gvb_ok-1.~_x"
    assert request.calendly_url == CALENDLY_URL
    assert build_request(arguments(notification_email=" Info@Example.com ")).notification_email == (
        "info@example.com"
    )
    with pytest.raises(ConfigureBusinessInputError):
        build_request(arguments(notification_email="not-an-address"))
    assert build_request(arguments(clear_notification_email=True)).notification_email == ""
    # An empty --invoice-prefix switches invoices off; a prefix switches them on.
    assert build_request(arguments(invoice_prefix="")).invoice_prefix == ""
    assert build_request(arguments(invoice_prefix=" LL ")).invoice_prefix == "LL"
    # A booking link is a URL with a path, not an origin.
    assert build_request(arguments(calendly_url="https://calendly.com/x?foo=1"))
    for bad_key in ("has/slash", "has space", "x" * 300, "-leading-dash"):
        with pytest.raises(ConfigureBusinessInputError):
            build_request(arguments(public_key=bad_key))
    for bad_link in (
        "calendly.com/x",
        "https://",
        "ftp://calendly.com/x",
        "https://:xx/x",
        "http://calendly.com/x",
    ):
        with pytest.raises(ConfigureBusinessInputError):
            build_request(arguments(calendly_url=bad_link))
    # Cleartext is only tolerated for local development hosts.
    assert build_request(arguments(calendly_url="http://localhost:3000/x"))


@pytest.mark.asyncio
async def test_success_settles_an_expired_attempt_and_supersedes_its_replacement(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A session can complete just before its deadline while its webhook lands
    after we already regenerated a checkout: the money still settles the quote
    and the payable replacement is closed out."""

    expired_session = (
        "cs_expired",
        "https://checkout.stripe.com/c/pay/cs_expired",
        NOW - timedelta(hours=1),
    )
    fresh_session = (
        "cs_fresh",
        "https://checkout.stripe.com/c/pay/cs_fresh",
        None,
    )
    checkout = CheckoutFake(sessions=[expired_session, fresh_session])
    application, owner_replies, _, claim_token = await hosted_quote(
        session_factory, checkout=checkout
    )
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        regenerated = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert regenerated.json() == {"checkoutUrl": fresh_session[1]}
        # The late success names the original, locally expired session.
        late = checkout_event(event_id="evt_late", session_id="cs_expired")
        settled = await client.post(
            "/webhooks/stripe", content=late, headers={SIGNATURE_HEADER: sign(late)}
        )
        assert settled.status_code == 200
        assert settled.json() == {"status": "recorded"}
        got = await client.get(f"/v1/quotes/{claim_token}")
        assert got.json()["quote"]["status"] == "paid"
        again = await client.post(f"/v1/quotes/{claim_token}/accept")
        assert again.status_code == 409
    async with session_factory() as session:
        payments = dict(
            (p.checkout_session_id, p.status)
            for p in (await session.scalars(select(QuotePayment))).all()
        )
    assert payments == {"cs_expired": "paid", "cs_fresh": "expired"}
    worker = immediate_worker(application)
    await worker.drain()
    assert texts_of(owner_replies, "Quote gvq_")[0].endswith("paid")


@pytest.mark.asyncio
async def test_payment_save_guard_rejects_a_regressing_transition(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A stale event that processed after the winner cannot overwrite the
    settled row: the guarded update raises and the event retries later."""

    application, _, _, claim_token = await hosted_quote(session_factory, checkout=CheckoutFake())
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
    async with session_factory() as session:
        repo = SqlQuotePaymentRepository(session)
        payment = await repo.find_by_checkout_session(SESSION_ID)
        assert payment is not None
        paid = payment.mark_paid("pi_1", NOW)
        await repo.save(paid, expected_from=payment.status)
        with pytest.raises(QuotePaymentConflictError):
            # A second processor holding the still-open snapshot loses.
            await repo.save(payment.mark_pending(NOW), expected_from=payment.status)
        persisted = await repo.find_by_checkout_session(SESSION_ID)
        assert persisted is not None and persisted.status.value == "paid"


@pytest.mark.asyncio
async def test_fetch_conflict_returns_the_committed_state(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch that loses its view-marker write re-reads the committed row
    rather than reporting a stale status."""

    application, _, _, claim_token = await hosted_quote(session_factory, checkout=CheckoutFake())
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200

        # Force the fetch onto the conflict path once by raising before the
        # write: the committed row (already accepted) is what answers.
        async def raise_once(
            self: SqlQuoteRepository, quote: object, *, expected_version: int
        ) -> None:
            raise QuoteConcurrencyError("a racing write won")

        monkeypatch.setattr(SqlQuoteRepository, "save", raise_once)
        response = await client.get(f"/v1/quotes/{claim_token}")
        assert response.status_code == 200
        assert response.json()["quote"]["status"] == "accepted"
