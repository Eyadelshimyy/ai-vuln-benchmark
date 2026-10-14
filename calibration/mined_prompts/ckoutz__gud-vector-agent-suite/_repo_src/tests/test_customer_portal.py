"""Customer portal: magic-link login, sessions, tenant isolation, recurring
quotes (subscription checkout + lifecycle webhooks) and service requests."""

import json
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import FAKE_NOW, OwnerReplyFake, TranscriptionFake
from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.composition import Application, ApplicationPorts, build_application
from gvas.config import ResendSettings
from gvas.domain.customers import (
    LOGIN_TOKEN_TTL,
    PORTAL_LOGIN_EMAIL_COMMAND_TYPE,
    PortalLoginEmailRequest,
    hash_portal_token,
    new_portal_token,
    portal_token_matches,
)
from gvas.domain.enums import BillingInterval, DeliveryStatus, QuoteBilling
from gvas.domain.identifiers import BusinessId, ConversationId, QuoteId
from gvas.domain.messages import CustomerRecipient, DeliveryReceipt
from gvas.domain.payments import (
    BillingCustomerRequest,
    BillingCustomerResult,
    BillingPortalRequest,
    BillingPortalResult,
    PaymentCheckoutRequest,
    PaymentEventOutcome,
    PaymentLineItem,
    QuoteSubscriptionRecord,
)
from gvas.domain.quotes import (
    QuoteDraftProposal,
    QuoteDraftRejectedError,
    QuoteDraftRequest,
    QuoteLineItem,
)
from gvas.infrastructure.models import (
    Customer,
    OutboundMessage,
    OutboxMessage,
    PortalLoginTokenRecord,
    PortalSessionRecord,
    QuoteRecord,
    ServiceRequestRecord,
)
from gvas.infrastructure.payment_models import PaymentProviderEvent, QuoteSubscription
from gvas.infrastructure.quote_drafting import (
    DeterministicQuoteDrafter,
    written_recurrence,
)
from gvas.infrastructure.repositories import SqlBusinessRepository
from gvas.infrastructure.resend import ResendPortalLoginEmailAdapter
from gvas.infrastructure.stripe import StripeWebhookVerifier
from gvas.infrastructure.stripe.api import (
    StripeCheckout,
    billing_portal_form,
    checkout_form,
    customer_form,
)
from gvas.infrastructure.stripe.config import StripeSettings
from gvas.infrastructure.stripe.events import HANDLED_EVENTS, parse_checkout_event
from gvas.infrastructure.stripe.signature import SIGNATURE_HEADER
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.portal import create_portal_router
from gvas.interfaces.http.public import PerIpRateLimiter, create_public_router
from test_composition import Clock, inbound, seed_business
from test_hosted_quotes import (
    CALENDLY_URL,
    CHECKOUT_URL,
    DISPLAY_NAME,
    SESSION_ID,
    SITE_URL,
    WEBHOOK_SECRET,
    CheckoutFake,
    HostedEmailDelivery,
    sign,
)
from test_pilot_runtime import immediate_worker, texts_of
from test_portal_quote_handoff import PhoneAwareDrafting, recipient

PUBLIC_KEY = "gvb_portal_test"
EMAIL = "jane@example.test"
STRIPE_CUSTOMER = "cus_test_1"
STRIPE_SUBSCRIPTION = "sub_test_1"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


class RecurringDrafting:
    """Drafts a monthly quote for the given recipient."""

    def __init__(self, customer: CustomerRecipient) -> None:
        self.customer = customer

    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal:
        return QuoteDraftProposal(
            quote_id=request.quote_id,
            business_id=request.business_id,
            recipient=self.customer,
            currency="USD",
            line_items=(
                QuoteLineItem(description="Monthly monitoring", quantity=1, unit_price_minor=9_900),
            ),
            billing=QuoteBilling.RECURRING,
            interval=BillingInterval.MONTH,
            confidence=1,
        )


class BillingFake:
    """Stripe customers + billing portal, in memory."""

    def __init__(self) -> None:
        self.customers: list[BillingCustomerRequest] = []
        self.portals: list[BillingPortalRequest] = []

    async def create_customer(self, request: BillingCustomerRequest) -> BillingCustomerResult:
        self.customers.append(request)
        return BillingCustomerResult(customer_ref=STRIPE_CUSTOMER)

    async def create_billing_portal_session(
        self, request: BillingPortalRequest
    ) -> BillingPortalResult:
        self.portals.append(request)
        return BillingPortalResult(url="https://billing.stripe.com/p/session/test")


class Portal:
    def __init__(
        self,
        application: Application,
        owner_replies: OwnerReplyFake,
        delivery: HostedEmailDelivery,
        business_id: BusinessId,
        claim_token: str,
    ) -> None:
        self.application = application
        self.owner_replies = owner_replies
        self.delivery = delivery
        self.business_id = business_id
        self.claim_token = claim_token


async def portal_business(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    public_key: str = PUBLIC_KEY,
    email: str = EMAIL,
    recurring: bool = False,
    checkout: CheckoutFake | None = None,
    billing: BillingFake | None = None,
    site_url: str = SITE_URL,
    deployment: str = "production",
) -> Portal:
    """A hosted business whose owner approves one quote addressed to ``email``."""

    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id,
            site_url=site_url,
            display_name=DISPLAY_NAME,
            calendly_url=CALENDLY_URL,
            public_key=public_key,
            now=NOW,
        )
        await session.commit()
    owner_replies = OwnerReplyFake()
    delivery = HostedEmailDelivery()
    who = recipient(email=email)
    application = build_application(
        ApplicationPorts(
            owner_replies=owner_replies,
            quote_drafting=RecurringDrafting(who) if recurring else PhoneAwareDrafting(who),
            quote_delivery=delivery,
            payment_checkout=checkout,
            billing_accounts=billing,
            transcription=TranscriptionFake({}),
            completeness_review=MarkerCompletenessReviewer(),
            checklist_evidence=MarkerChecklistEvidenceAttributor(),
            report_generation=DeterministicReportGenerator(),
        ),
        session_factory=session_factory,
        now=Clock(),
        payment_deployment=deployment,
    )
    worker = immediate_worker(application)
    await application.ingest_service.ingest(
        inbound(business_id, "quote: mold inspection 250", message_key=f"quote-{business_id}")
    )
    await worker.drain()
    await application.ingest_service.ingest(
        inbound(business_id, "approve", message_key=f"approve-{business_id}")
    )
    for _ in range(4):
        await worker.drain()
    async with session_factory() as session:
        row = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == business_id)
        )
    assert row is not None and row.claim_token is not None
    return Portal(application, owner_replies, delivery, business_id, row.claim_token)


def client(
    portal: Portal,
    *,
    verifier: StripeWebhookVerifier | None = None,
    login_rate_limiter: PerIpRateLimiter | None = None,
) -> httpx.AsyncClient:
    async def origins() -> frozenset[str]:
        return frozenset({SITE_URL})

    app = create_app(
        routers=(
            create_public_router(portal.application.public_quotes, webhook_verifier=verifier),
            create_portal_router(portal.application.portal, login_rate_limiter=login_rate_limiter),
        ),
        cors_origins=origins,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def login_commands(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> list[OutboxMessage]:
    async with session_factory() as session:
        rows = await session.scalars(
            select(OutboxMessage)
            .where(
                OutboxMessage.business_id == business_id,
                OutboxMessage.command_type == PORTAL_LOGIN_EMAIL_COMMAND_TYPE,
            )
            .order_by(OutboxMessage.available_at, OutboxMessage.id)
        )
        return list(rows.all())


def token_from(command: OutboxMessage) -> str:
    login_url = command.payload["login_url"]
    assert isinstance(login_url, str)
    parsed = urlparse(login_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == f"{SITE_URL}/portal/login"
    return parse_qs(parsed.query)["token"][0]


async def sign_in(
    session_factory: async_sessionmaker[AsyncSession],
    portal: Portal,
    http: httpx.AsyncClient,
    *,
    public_key: str = PUBLIC_KEY,
    email: str = EMAIL,
) -> tuple[str, dict[str, object]]:
    response = await http.post(f"/v1/businesses/{public_key}/portal/login", json={"email": email})
    assert response.status_code == 202 and response.json() == {}
    commands = [
        command
        for command in await login_commands(session_factory, portal.business_id)
        if command.payload["to"] == email
    ]
    assert commands, "a known customer should have been e-mailed a link"
    exchanged = await http.post("/v1/portal/sessions", json={"token": token_from(commands[-1])})
    assert exchanged.status_code == 200, exchanged.text
    payload = exchanged.json()
    return payload["sessionToken"], payload


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite hands timestamps back naive; they were stored as UTC."""

    return value if value is None or value.tzinfo is not None else value.replace(tzinfo=UTC)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# -- tokens -----------------------------------------------------------------


def test_portal_tokens_are_url_safe_hashed_and_compared_in_constant_time() -> None:
    token = new_portal_token()
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token), token
    stored = hash_portal_token(token)
    assert len(stored) == 64
    assert portal_token_matches(token, stored)
    assert not portal_token_matches(new_portal_token(), stored)


# -- login flow -------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_links_a_customer_and_login_issues_one_link(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    async with session_factory() as session:
        customer = await session.scalar(
            select(Customer).where(Customer.business_id == portal.business_id)
        )
        quote = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == portal.business_id)
        )
    assert customer is not None and customer.email == EMAIL
    assert customer.display_name == "Jane Doe"
    assert quote is not None and quote.customer_id == customer.id

    async with client(portal) as http:
        # Mixed case and whitespace resolve to the same customer.
        response = await http.post(
            f"/v1/businesses/{PUBLIC_KEY}/portal/login", json={"email": "  Jane@Example.TEST "}
        )
        assert response.status_code == 202 and response.json() == {}
    commands = await login_commands(session_factory, portal.business_id)
    assert len(commands) == 1
    payload = commands[0].payload
    assert payload["to"] == EMAIL
    assert payload["business_display_name"] == DISPLAY_NAME
    raw = token_from(commands[0])
    async with session_factory() as session:
        token_row = await session.scalar(select(PortalLoginTokenRecord))
    assert token_row is not None
    assert token_row.token_hash == hash_portal_token(raw)
    assert raw not in json.dumps({"hash": token_row.token_hash})
    assert token_row.expires_at - token_row.created_at == LOGIN_TOKEN_TTL


@pytest.mark.asyncio
async def test_login_never_reveals_whether_an_account_exists(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    async with client(portal) as http:
        for public_key, email in (
            (PUBLIC_KEY, "nobody@example.test"),
            ("gvb_unknown", EMAIL),
            (PUBLIC_KEY, "not-an-email"),
        ):
            response = await http.post(
                f"/v1/businesses/{public_key}/portal/login", json={"email": email}
            )
            assert response.status_code == 202
            assert response.json() == {}
    assert await login_commands(session_factory, portal.business_id) == []
    async with session_factory() as session:
        assert (await session.scalars(select(PortalLoginTokenRecord))).all() == []


@pytest.mark.asyncio
async def test_login_is_rate_limited_per_ip_and_per_email(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    limiter = PerIpRateLimiter(per_minute=1, burst=2)
    async with client(portal, login_rate_limiter=limiter) as http:
        path = f"/v1/businesses/{PUBLIC_KEY}/portal/login"
        # Same address from rotating ips: the e-mail bucket blocks.
        for hop in ("1.1.1.1", "2.2.2.2"):
            ok = await http.post(path, json={"email": EMAIL}, headers={"X-Forwarded-For": hop})
            assert ok.status_code == 202
        blocked = await http.post(
            path, json={"email": EMAIL}, headers={"X-Forwarded-For": "3.3.3.3"}
        )
        assert blocked.status_code == 429
        # A fresh address from an exhausted ip: the ip bucket blocks.
        again = await http.post(
            path, json={"email": "other@example.test"}, headers={"X-Forwarded-For": "1.1.1.1"}
        )
        assert again.status_code == 202
        third = await http.post(
            path, json={"email": "third@example.test"}, headers={"X-Forwarded-For": "1.1.1.1"}
        )
        assert third.status_code == 429


@pytest.mark.asyncio
async def test_login_token_is_single_use_and_expires(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    async with client(portal) as http:
        assert (
            await http.post(f"/v1/businesses/{PUBLIC_KEY}/portal/login", json={"email": EMAIL})
        ).status_code == 202
        commands = await login_commands(session_factory, portal.business_id)
        raw = token_from(commands[0])
        first = await http.post("/v1/portal/sessions", json={"token": raw})
        assert first.status_code == 200
        body = first.json()
        assert set(body) == {"sessionToken", "role", "customer", "business"}
        assert body["role"] == "customer"
        assert body["customer"] == {"displayName": "Jane Doe", "email": EMAIL}
        assert body["business"] == {"displayName": DISPLAY_NAME, "siteUrl": SITE_URL}
        second = await http.post("/v1/portal/sessions", json={"token": raw})
        assert second.status_code == 401
        assert second.json() == {"detail": "unauthorized"}
        garbage = await http.post("/v1/portal/sessions", json={"token": "nope"})
        assert garbage.status_code == 401
        assert garbage.json() == second.json()

        # A second link, aged past its TTL, is refused the same way.
        assert (
            await http.post(f"/v1/businesses/{PUBLIC_KEY}/portal/login", json={"email": EMAIL})
        ).status_code == 202
        commands = await login_commands(session_factory, portal.business_id)
        assert len(commands) == 2
        stale = token_from(commands[-1])
        async with session_factory() as session:
            row = await session.scalar(
                select(PortalLoginTokenRecord).where(
                    PortalLoginTokenRecord.token_hash == hash_portal_token(stale)
                )
            )
            assert row is not None
            row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
            await session.commit()
        expired = await http.post("/v1/portal/sessions", json={"token": stale})
        assert expired.status_code == 401
        assert expired.json() == {"detail": "unauthorized"}
    async with session_factory() as session:
        sessions = (await session.scalars(select(PortalSessionRecord))).all()
    assert len(sessions) == 1
    assert sessions[0].expires_at - sessions[0].created_at == timedelta(days=30)


# -- sessions ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_authenticates_portal_routes_and_can_be_revoked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    async with client(portal) as http:
        token, _ = await sign_in(session_factory, portal, http)
        me = await http.get("/v1/portal/me", headers=bearer(token))
        assert me.status_code == 200
        assert me.json() == {
            "customer": {"displayName": "Jane Doe", "email": EMAIL, "phone": "+19255551234"},
            "business": {
                "displayName": DISPLAY_NAME,
                "siteUrl": SITE_URL,
                "calendlyUrl": CALENDLY_URL,
            },
        }
        quotes = await http.get("/v1/portal/quotes", headers=bearer(token))
        assert quotes.status_code == 200
        listed = quotes.json()["quotes"]
        assert len(listed) == 1
        quote = listed[0]
        assert set(quote) == {
            "id",
            "status",
            "totalCents",
            "currency",
            "createdAt",
            "approvedAt",
            "claimToken",
            "billing",
            "interval",
        }
        assert quote["claimToken"] == portal.claim_token
        assert quote["totalCents"] == 25_000
        assert quote["currency"] == "USD"
        assert quote["billing"] == "one_time"
        assert quote["interval"] is None
        assert quote["approvedAt"]
        assert str(portal.business_id) not in quotes.text

        for headers in ({}, {"Authorization": "Basic abc"}, bearer("not-a-session")):
            denied = await http.get("/v1/portal/me", headers=headers)
            assert denied.status_code == 401
            assert denied.json() == {"detail": "unauthorized"}

        revoked = await http.delete("/v1/portal/sessions", headers=bearer(token))
        assert revoked.status_code == 204
        after = await http.get("/v1/portal/me", headers=bearer(token))
        assert after.status_code == 401
        # Revoking twice, revoking garbage, or without a token is the same 401.
        assert (await http.delete("/v1/portal/sessions", headers=bearer(token))).status_code == 401
        assert (await http.delete("/v1/portal/sessions", headers=bearer("x"))).status_code == 401
        assert (await http.delete("/v1/portal/sessions")).status_code == 401


@pytest.mark.asyncio
async def test_cors_preflight_allows_the_portal_delete(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    async with client(portal) as http:
        preflight = await http.options(
            "/v1/portal/sessions",
            headers={
                "Origin": SITE_URL,
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
    assert preflight.status_code == 200
    assert "DELETE" in preflight.headers["access-control-allow-methods"]
    assert "authorization" in preflight.headers["access-control-allow-headers"]


# -- tenant isolation -------------------------------------------------------


@pytest.mark.asyncio
async def test_same_email_at_two_businesses_is_two_customers(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    a = await portal_business(session_factory, public_key="gvb_a")
    b = await portal_business(session_factory, public_key="gvb_b", site_url="https://b.example")
    async with session_factory() as session:
        customers = (await session.scalars(select(Customer))).all()
    assert {c.business_id for c in customers} == {a.business_id, b.business_id}
    assert {c.email for c in customers} == {EMAIL}
    assert len({c.id for c in customers}) == 2

    async with client(a) as http:
        token_a, _ = await sign_in(session_factory, a, http, public_key="gvb_a")
        quotes_a = (await http.get("/v1/portal/quotes", headers=bearer(token_a))).json()
        assert [q["claimToken"] for q in quotes_a["quotes"]] == [a.claim_token]
        assert b.claim_token not in json.dumps(quotes_a)
        me_a = (await http.get("/v1/portal/me", headers=bearer(token_a))).json()
        assert me_a["business"]["siteUrl"] == SITE_URL
    # Logging in at B mails B's link (to B's site) and shows only B's quote.
    async with client(b) as http:
        response = await http.post("/v1/businesses/gvb_b/portal/login", json={"email": EMAIL})
        assert response.status_code == 202
        commands = await login_commands(session_factory, b.business_id)
        assert len(commands) == 1
        login_url = commands[0].payload["login_url"]
        assert isinstance(login_url, str) and login_url.startswith("https://b.example/portal/login")
        raw = parse_qs(urlparse(login_url).query)["token"][0]
        exchanged = await http.post("/v1/portal/sessions", json={"token": raw})
        token_b = exchanged.json()["sessionToken"]
        quotes_b = (await http.get("/v1/portal/quotes", headers=bearer(token_b))).json()
        assert [q["claimToken"] for q in quotes_b["quotes"]] == [b.claim_token]
        me_b = (await http.get("/v1/portal/me", headers=bearer(token_b))).json()
        assert me_b["business"]["siteUrl"] == "https://b.example"
        # A's session is worthless against B's data and vice versa: the
        # session names the tenant, so the same bearer answers the same view.
        assert (await http.get("/v1/portal/quotes", headers=bearer(token_a))).json() == quotes_a


@pytest.mark.asyncio
async def test_another_customer_of_the_same_business_sees_only_their_quotes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    other_id = portal.business_id
    # A second customer of the same business, approved through the same flow.
    worker = immediate_worker(portal.application)
    second = build_application(
        ApplicationPorts(
            owner_replies=portal.owner_replies,
            quote_drafting=PhoneAwareDrafting(recipient(email="bob@example.test")),
            quote_delivery=portal.delivery,
            transcription=TranscriptionFake({}),
            completeness_review=MarkerCompletenessReviewer(),
            checklist_evidence=MarkerChecklistEvidenceAttributor(),
            report_generation=DeterministicReportGenerator(),
        ),
        session_factory=session_factory,
        now=Clock(),
    )
    second_worker = immediate_worker(second)
    await second.ingest_service.ingest(
        inbound(other_id, "quote: attic check 100", message_key="quote-bob", conversation="bob")
    )
    await second_worker.drain()
    await second.ingest_service.ingest(
        inbound(other_id, "approve", message_key="approve-bob", conversation="bob")
    )
    for _ in range(4):
        await second_worker.drain()
    await worker.drain()
    async with session_factory() as session:
        quotes = (await session.scalars(select(QuoteRecord))).all()
        customers = (await session.scalars(select(Customer))).all()
    assert len(quotes) == 2 and len(customers) == 2

    async with client(portal) as http:
        jane, _ = await sign_in(session_factory, portal, http)
        bob, _ = await sign_in(session_factory, portal, http, email="bob@example.test")
        jane_quotes = (await http.get("/v1/portal/quotes", headers=bearer(jane))).json()["quotes"]
        bob_quotes = (await http.get("/v1/portal/quotes", headers=bearer(bob))).json()["quotes"]
    assert [q["claimToken"] for q in jane_quotes] == [portal.claim_token]
    assert len(bob_quotes) == 1 and bob_quotes[0]["claimToken"] != portal.claim_token
    assert bob_quotes[0]["totalCents"] == 25_000  # the fake drafter's price


@pytest.mark.asyncio
async def test_older_quotes_are_linked_lazily_on_first_login(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    # Simulate a quote that predates the customers table: unlink it and drop
    # the customer row, as an upgraded deployment would find them.
    async with session_factory() as session:
        quote = await session.scalar(select(QuoteRecord))
        assert quote is not None
        quote.customer_id = None
        customer = await session.scalar(select(Customer))
        assert customer is not None
        await session.delete(customer)
        await session.commit()
    async with client(portal) as http:
        token, _ = await sign_in(session_factory, portal, http)
        quotes = (await http.get("/v1/portal/quotes", headers=bearer(token))).json()["quotes"]
    assert [q["claimToken"] for q in quotes] == [portal.claim_token]
    async with session_factory() as session:
        quote = await session.scalar(select(QuoteRecord))
        customer = await session.scalar(select(Customer))
    assert quote is not None and customer is not None
    assert quote.customer_id == customer.id


# -- service requests -------------------------------------------------------


@pytest.mark.asyncio
async def test_service_request_is_stored_and_the_owner_is_told(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory)
    async with client(portal) as http:
        token, _ = await sign_in(session_factory, portal, http)
        response = await http.post(
            "/v1/portal/requests",
            headers=bearer(token),
            json={"message": "Can you come back for the crawlspace?", "preferredDates": "Any Tue"},
        )
        assert response.status_code == 202 and response.json() == {}
        too_long = await http.post(
            "/v1/portal/requests", headers=bearer(token), json={"message": "x" * 2001}
        )
        assert too_long.status_code == 422
        anonymous = await http.post("/v1/portal/requests", json={"message": "hi"})
        assert anonymous.status_code == 401
    async with session_factory() as session:
        stored = (await session.scalars(select(ServiceRequestRecord))).all()
        customer = await session.scalar(select(Customer))
    assert customer is not None and len(stored) == 1
    assert stored[0].business_id == portal.business_id
    assert stored[0].customer_id == customer.id
    assert stored[0].message == "Can you come back for the crawlspace?"
    assert stored[0].preferred_dates == "Any Tue"
    assert stored[0].status == "new"
    assert stored[0].source == "portal"

    await immediate_worker(portal.application).drain()
    notices = texts_of(portal.owner_replies, "New service request from")
    assert len(notices) == 1
    assert notices[0].startswith(
        f"New service request from Jane Doe ({EMAIL}): Can you come back for the crawlspace?"
    )
    assert "Any Tue" in notices[0]


# -- recurring quotes: parser -----------------------------------------------


@pytest.mark.asyncio
async def test_structured_billing_line_makes_a_recurring_quote() -> None:
    drafter = DeterministicQuoteDrafter()
    text = (
        "quote:\ncustomer: person@example.com\ncurrency: USD\n"
        "item: 1 | Monitoring | 99.00\nbilling: monthly"
    )

    def request_of(request_text: str) -> QuoteDraftRequest:
        return QuoteDraftRequest(
            quote_id=QuoteId(uuid4()),
            business_id=BusinessId(uuid4()),
            conversation_id=ConversationId(uuid4()),
            request_text=request_text,
            revision=1,
            idempotency_key="quote:1",
        )

    draft = await drafter.draft(request_of(text))
    assert draft.billing is QuoteBilling.RECURRING
    assert draft.interval is BillingInterval.MONTH
    yearly = await drafter.draft(request_of(text.replace("monthly", "yearly")))
    assert yearly.interval is BillingInterval.YEAR
    one_time = await drafter.draft(request_of(text.rsplit("\n", 1)[0]))
    assert one_time.billing is QuoteBilling.ONE_TIME and one_time.interval is None
    with pytest.raises(QuoteDraftRejectedError, match="billing"):
        await drafter.draft(request_of(text.replace("monthly", "fortnightly")))


def test_free_text_recurrence_comes_only_from_the_owner_wording() -> None:
    assert written_recurrence("quote Jane 99 per month for monitoring") is BillingInterval.MONTH
    assert written_recurrence("quote Jane $99/mo monitoring") is BillingInterval.MONTH
    assert written_recurrence("monthly monitoring 99") is BillingInterval.MONTH
    assert written_recurrence("annual service plan 990") is BillingInterval.YEAR
    assert written_recurrence("yearly plan 990") is BillingInterval.YEAR
    # A recurring-sounding service with no recurrence word stays one-time.
    assert written_recurrence("quote Jane 99 for a monitoring subscription") is None
    assert written_recurrence("quote Jane 250 mold inspection") is None
    with pytest.raises(QuoteDraftRejectedError):
        written_recurrence("99 per month or 990 yearly")


@pytest.mark.asyncio
async def test_approval_preview_shows_billing_for_recurring_quotes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory, recurring=True, checkout=CheckoutFake())
    previews = [text for text in texts_of(portal.owner_replies, "") if "Billing: monthly" in text]
    assert previews and "Reply with approve" in previews[0]
    assert portal.delivery.requests and "Billed monthly" in portal.delivery.requests[0].body_text


# -- recurring quotes: checkout ---------------------------------------------


def test_subscription_checkout_form_uses_subscription_mode_and_recurring_prices() -> None:
    request = PaymentCheckoutRequest(
        business_id=BusinessId(uuid4()),
        quote_id=uuid4(),
        client_reference="gvq_abc",
        currency="USD",
        line_items=(PaymentLineItem(description="Monitoring", quantity=1, amount_minor=9_900),),
        success_url=f"{SITE_URL}/q/tok?paid=1",
        cancel_url=f"{SITE_URL}/q/tok",
        idempotency_key="quote-delivery:1",
        metadata={"gvas_quote_id": "gvq_abc", "business_id": "b-1"},
        recurring_interval=BillingInterval.MONTH,
        customer_ref=STRIPE_CUSTOMER,
    )
    form = checkout_form(request)
    assert form["mode"] == "subscription"
    assert form["customer"] == STRIPE_CUSTOMER
    assert form["line_items[0][price_data][currency]"] == "usd"
    assert form["line_items[0][price_data][unit_amount]"] == "9900"
    assert form["line_items[0][price_data][recurring][interval]"] == "month"
    assert form["line_items[0][price_data][product_data][name]"] == "Monitoring"
    assert form["metadata[gvas_quote_id]"] == "gvq_abc"
    assert form["subscription_data[metadata][gvas_quote_id]"] == "gvq_abc"
    # One-time requests are untouched: no customer, no recurring, payment mode.
    one_time = checkout_form(
        request.model_copy(update={"recurring_interval": None, "customer_ref": None})
    )
    assert one_time["mode"] == "payment"
    assert "customer" not in one_time
    assert not any("recurring" in key or "subscription_data" in key for key in one_time)


def test_customer_and_billing_portal_forms() -> None:
    business_id = BusinessId(uuid4())
    customer = customer_form(
        BillingCustomerRequest(
            business_id=business_id,
            customer_id=uuid4(),
            email=EMAIL,
            name="Jane Doe",
            phone="+19255551234",
            idempotency_key="billing-customer:1",
            metadata={"business_id": str(business_id)},
        )
    )
    assert customer == {
        "email": EMAIL,
        "name": "Jane Doe",
        "phone": "+19255551234",
        "metadata[business_id]": str(business_id),
    }
    assert billing_portal_form(
        BillingPortalRequest(
            business_id=business_id, customer_ref=STRIPE_CUSTOMER, return_url=f"{SITE_URL}/portal"
        )
    ) == {"customer": STRIPE_CUSTOMER, "return_url": f"{SITE_URL}/portal"}


@pytest.mark.asyncio
async def test_stripe_adapter_posts_customers_and_billing_portal_sessions() -> None:
    seen: list[tuple[str, str, str | None]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(
            (request.url.path, request.content.decode(), request.headers.get("idempotency-key"))
        )
        if request.url.path.endswith("/customers"):
            return httpx.Response(200, json={"id": STRIPE_CUSTOMER, "object": "customer"})
        return httpx.Response(200, json={"url": "https://billing.stripe.com/p/session/x"})

    settings = StripeSettings(secret_key="sk_test_secret", webhook_secret="whsec_x")  # noqa: S106
    adapter = StripeCheckout(settings, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    created = await adapter.create_customer(
        BillingCustomerRequest(
            business_id=BusinessId(uuid4()),
            customer_id=uuid4(),
            email=EMAIL,
            idempotency_key="billing-customer:1",
        )
    )
    assert created.customer_ref == STRIPE_CUSTOMER
    session = await adapter.create_billing_portal_session(
        BillingPortalRequest(
            business_id=BusinessId(uuid4()),
            customer_ref=STRIPE_CUSTOMER,
            return_url=f"{SITE_URL}/portal",
        )
    )
    assert session.url == "https://billing.stripe.com/p/session/x"
    assert seen[0][0] == "/v1/customers" and seen[0][2] == "billing-customer:1"
    assert f"email={EMAIL.replace('@', '%40')}" in seen[0][1]
    assert seen[1][0] == "/v1/billing_portal/sessions" and seen[1][2] is None
    assert f"customer={STRIPE_CUSTOMER}" in seen[1][1]


def subscription_checkout_event(
    *,
    event_id: str = "evt_sub_1",
    session_id: str = SESSION_ID,
    metadata: dict[str, str] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": event_id,
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": session_id,
                    "object": "checkout.session",
                    "mode": "subscription",
                    "payment_intent": None,
                    "payment_status": "paid",
                    "customer": STRIPE_CUSTOMER,
                    "subscription": STRIPE_SUBSCRIPTION,
                    "metadata": {"gvas_quote_id": "gvq_x"} if metadata is None else metadata,
                }
            },
        }
    ).encode()


def invoice_event(
    *,
    event_id: str,
    event_type: str,
    billing_reason: str = "subscription_cycle",
    amount_paid: int = 9_900,
    subscription: str | None = STRIPE_SUBSCRIPTION,
    period_end: int = 1_800_000_000,
    metadata: dict[str, str] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": event_id,
            "type": event_type,
            "data": {
                "object": {
                    "id": "in_1",
                    "object": "invoice",
                    "customer": STRIPE_CUSTOMER,
                    "subscription": subscription,
                    "billing_reason": billing_reason,
                    "amount_paid": amount_paid,
                    "amount_due": amount_paid,
                    "currency": "usd",
                    "lines": {"data": [{"period": {"start": 1, "end": period_end}}]},
                    "subscription_details": {
                        "metadata": {"gvas_quote_id": "gvq_x"} if metadata is None else metadata
                    },
                }
            },
        }
    ).encode()


def subscription_event(
    *,
    event_id: str,
    event_type: str,
    status: str = "active",
    cancel_at_period_end: bool = False,
    subscription: str = STRIPE_SUBSCRIPTION,
    metadata: dict[str, str] | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": event_id,
            "type": event_type,
            "data": {
                "object": {
                    "id": subscription,
                    "object": "subscription",
                    "customer": STRIPE_CUSTOMER,
                    "status": status,
                    "cancel_at_period_end": cancel_at_period_end,
                    "current_period_end": 1_800_000_000,
                    "metadata": {"gvas_quote_id": "gvq_x"} if metadata is None else metadata,
                    "items": {
                        "data": [
                            {
                                "quantity": 1,
                                "price": {
                                    "unit_amount": 9_900,
                                    "currency": "usd",
                                    "recurring": {"interval": "month"},
                                },
                            }
                        ]
                    },
                }
            },
        }
    ).encode()


def test_lifecycle_events_normalize_to_subscription_outcomes() -> None:
    assert HANDLED_EVENTS >= {
        "checkout.session.completed",
        "invoice.paid",
        "invoice.payment_failed",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }
    completed = parse_checkout_event(subscription_checkout_event())
    assert completed.outcome is PaymentEventOutcome.SUCCEEDED
    assert completed.checkout_session_id == SESSION_ID
    assert completed.subscription is not None
    assert completed.subscription.subscription_ref == STRIPE_SUBSCRIPTION
    assert completed.subscription.customer_ref == STRIPE_CUSTOMER

    renewed = parse_checkout_event(invoice_event(event_id="e", event_type="invoice.paid"))
    assert renewed.outcome is PaymentEventOutcome.SUBSCRIPTION_RENEWED
    assert renewed.subscription is not None
    # The invoice total is what was collected, not the recurring price.
    assert renewed.subscription.paid_minor == 9_900
    assert renewed.subscription.amount_minor is None
    assert renewed.subscription.currency == "usd"
    assert renewed.subscription.current_period_end == datetime.fromtimestamp(1_800_000_000, UTC)
    assert renewed.metadata == {"gvas_quote_id": "gvq_x"}
    initial = parse_checkout_event(
        invoice_event(event_id="e", event_type="invoice.paid", billing_reason="subscription_create")
    )
    assert initial.outcome is PaymentEventOutcome.SUBSCRIPTION_UPDATED
    failed = parse_checkout_event(invoice_event(event_id="e", event_type="invoice.payment_failed"))
    assert failed.outcome is PaymentEventOutcome.SUBSCRIPTION_PAYMENT_FAILED
    unrelated = parse_checkout_event(
        invoice_event(event_id="e", event_type="invoice.paid", subscription=None)
    )
    assert unrelated.outcome is PaymentEventOutcome.OTHER

    updated = parse_checkout_event(
        subscription_event(
            event_id="e", event_type="customer.subscription.updated", cancel_at_period_end=True
        )
    )
    assert updated.outcome is PaymentEventOutcome.SUBSCRIPTION_UPDATED
    assert updated.subscription is not None
    assert updated.subscription.cancel_at_period_end is True
    assert updated.subscription.interval is BillingInterval.MONTH
    assert updated.subscription.amount_minor == 9_900
    deleted = parse_checkout_event(
        subscription_event(event_id="e", event_type="customer.subscription.deleted")
    )
    assert deleted.outcome is PaymentEventOutcome.SUBSCRIPTION_CANCELLED
    assert deleted.subscription is not None and deleted.subscription.status == "canceled"


async def post_event(http: httpx.AsyncClient, body: bytes) -> httpx.Response:
    return await http.post("/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body)})


async def subscription_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[QuoteSubscription]:
    async with session_factory() as session:
        return list((await session.scalars(select(QuoteSubscription))).all())


@pytest.mark.asyncio
async def test_recurring_accept_creates_a_customer_and_a_subscription_checkout(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    billing = BillingFake()
    portal = await portal_business(
        session_factory, recurring=True, checkout=checkout, billing=billing
    )
    async with client(portal) as http:
        first = await http.post(f"/v1/quotes/{portal.claim_token}/accept")
        assert first.status_code == 200 and first.json() == {"checkoutUrl": CHECKOUT_URL}
        again = await http.post(f"/v1/quotes/{portal.claim_token}/accept")
        assert again.status_code == 200
        public = (await http.get(f"/v1/quotes/{portal.claim_token}")).json()["quote"]
        assert public["billing"] == "recurring" and public["interval"] == "month"
    assert len(billing.customers) == 1
    created = billing.customers[0]
    assert created.email == EMAIL and created.name == "Jane Doe"
    assert created.idempotency_key.startswith("billing-customer:")
    assert created.metadata["business_id"] == str(portal.business_id)
    assert created.metadata["gvas_deployment"] == "production"
    assert len(checkout.requests) == 1
    request = checkout.requests[0]
    assert request.is_subscription
    assert request.recurring_interval is BillingInterval.MONTH
    assert request.customer_ref == STRIPE_CUSTOMER
    assert request.currency == "usd"
    assert request.metadata["gvas_quote_id"].startswith("gvq_")
    assert request.metadata["gvas_deployment"] == "production"
    assert request.idempotency_key.startswith("quote-delivery:")
    async with session_factory() as session:
        customer = await session.scalar(select(Customer))
    assert customer is not None and customer.stripe_customer_id == STRIPE_CUSTOMER


# -- deployments sharing one Stripe account ----------------------------------

DEMO_TAG = {"gvas_quote_id": "gvq_x", "gvas_deployment": "demo"}
PRODUCTION_TAG = {"gvas_quote_id": "gvq_x", "gvas_deployment": "production"}


@pytest.mark.asyncio
async def test_production_answers_ok_to_the_demos_events_and_records_nothing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Stripe sends every event on the shared account to every endpoint. A
    demo event names a session or subscription production never made, which
    would otherwise be a 503 retried for days."""

    portal = await portal_business(
        session_factory, recurring=True, checkout=CheckoutFake(), billing=BillingFake()
    )
    async with client(portal, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as http:
        assert (await http.post(f"/v1/quotes/{portal.claim_token}/accept")).status_code == 200
        demo_events = [
            subscription_checkout_event(
                event_id="evt_demo_checkout", session_id="cs_demo", metadata=DEMO_TAG
            ),
            invoice_event(
                event_id="evt_demo_invoice", event_type="invoice.paid", metadata=DEMO_TAG
            ),
            subscription_event(
                event_id="evt_demo_sub",
                event_type="customer.subscription.updated",
                subscription="sub_demo",
                metadata=DEMO_TAG,
            ),
        ]
        for body in demo_events:
            response = await post_event(http, body)
            assert response.status_code == 200 and response.json() == {"status": "ignored"}
        # Production's own event, tagged or from before the tag, still lands.
        completed = await post_event(http, subscription_checkout_event())
        assert completed.json() == {"status": "recorded"}
        tagged = await post_event(
            http,
            invoice_event(
                event_id="evt_prod_renew", event_type="invoice.paid", metadata=PRODUCTION_TAG
            ),
        )
        assert tagged.json() == {"status": "recorded"}
    async with session_factory() as session:
        seen = set((await session.scalars(select(PaymentProviderEvent.event_id))).all())
    assert seen == {"evt_sub_1", "evt_prod_renew"}


@pytest.mark.asyncio
async def test_the_demo_tags_its_records_and_acts_only_on_its_own_events(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    billing = BillingFake()
    portal = await portal_business(
        session_factory, recurring=True, checkout=checkout, billing=billing, deployment="demo"
    )
    async with client(portal, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as http:
        assert (await http.post(f"/v1/quotes/{portal.claim_token}/accept")).status_code == 200
        assert checkout.requests[0].metadata["gvas_deployment"] == "demo"
        assert billing.customers[0].metadata["gvas_deployment"] == "demo"
        # Production's events, untagged or tagged, are not the demo's.
        for body in (
            subscription_checkout_event(event_id="evt_untagged"),
            subscription_checkout_event(event_id="evt_production", metadata=PRODUCTION_TAG),
        ):
            response = await post_event(http, body)
            assert response.status_code == 200 and response.json() == {"status": "ignored"}
        assert (await http.get(f"/v1/quotes/{portal.claim_token}")).json()["quote"][
            "status"
        ] != "paid"
        own = await post_event(
            http, subscription_checkout_event(event_id="evt_demo", metadata=DEMO_TAG)
        )
        assert own.json() == {"status": "recorded"}
        quote = (await http.get(f"/v1/quotes/{portal.claim_token}")).json()["quote"]
        assert quote["status"] == "paid"
    assert len(await subscription_rows(session_factory)) == 1


@pytest.mark.asyncio
async def test_recurring_accept_without_billing_accounts_is_503(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory, recurring=True, checkout=CheckoutFake())
    async with client(portal) as http:
        response = await http.post(f"/v1/quotes/{portal.claim_token}/accept")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_subscription_lifecycle_webhooks_update_the_row_and_notify_the_owner(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    billing = BillingFake()
    portal = await portal_business(
        session_factory, recurring=True, checkout=checkout, billing=billing
    )
    verifier = StripeWebhookVerifier(WEBHOOK_SECRET)
    async with client(portal, verifier=verifier) as http:
        assert (await http.post(f"/v1/quotes/{portal.claim_token}/accept")).status_code == 200
        # A lifecycle event that races ahead of the completion for one of our
        # subscriptions is retried, not swallowed.
        early = invoice_event(event_id="evt_early", event_type="invoice.paid")
        assert (await post_event(http, early)).status_code == 503
        # ...while a subscription that is not a quote is recorded and ignored.
        foreign = subscription_event(
            event_id="evt_foreign",
            event_type="customer.subscription.updated",
            subscription="sub_other",
            metadata={},
        )
        assert (await post_event(http, foreign)).json() == {"status": "recorded"}

        completed = await post_event(http, subscription_checkout_event())
        assert completed.status_code == 200 and completed.json() == {"status": "recorded"}
        assert (await post_event(http, subscription_checkout_event())).json() == {
            "status": "ignored"
        }
        rows = await subscription_rows(session_factory)
        assert len(rows) == 1
        row = rows[0]
        assert row.stripe_subscription_id == STRIPE_SUBSCRIPTION
        assert row.business_id == portal.business_id
        assert row.status == "active"
        assert row.interval == "month"
        assert row.amount_cents == 9_900
        assert row.currency == "USD"
        assert row.cancel_at_period_end is False
        quote = (await http.get(f"/v1/quotes/{portal.claim_token}")).json()["quote"]
        assert quote["status"] == "paid"

        # A prorated renewal collects less than the plan price; the plan price stays.
        renewed = await post_event(
            http, invoice_event(event_id="evt_renew", event_type="invoice.paid", amount_paid=7_425)
        )
        assert renewed.json() == {"status": "recorded"}
        rows = await subscription_rows(session_factory)
        assert as_utc(rows[0].current_period_end) == datetime.fromtimestamp(1_800_000_000, UTC)
        assert rows[0].amount_cents == 9_900

        failed = await post_event(
            http, invoice_event(event_id="evt_fail", event_type="invoice.payment_failed")
        )
        assert failed.json() == {"status": "recorded"}

        updated = await post_event(
            http,
            subscription_event(
                event_id="evt_upd",
                event_type="customer.subscription.updated",
                status="past_due",
                cancel_at_period_end=True,
            ),
        )
        assert updated.json() == {"status": "recorded"}
        rows = await subscription_rows(session_factory)
        assert rows[0].status == "past_due" and rows[0].cancel_at_period_end is True

        deleted = await post_event(
            http, subscription_event(event_id="evt_del", event_type="customer.subscription.deleted")
        )
        assert deleted.json() == {"status": "recorded"}
        assert (
            await post_event(
                http,
                subscription_event(event_id="evt_del", event_type="customer.subscription.deleted"),
            )
        ).json() == {"status": "ignored"}
        rows = await subscription_rows(session_factory)
        assert rows[0].status == "canceled"

        # The customer sees the subscription in the portal.
        token, _ = await sign_in(session_factory, portal, http)
        listed = (await http.get("/v1/portal/subscriptions", headers=bearer(token))).json()
        assert len(listed["subscriptions"]) == 1
        subscription = listed["subscriptions"][0]
        assert set(subscription) == {
            "id",
            "quoteId",
            "status",
            "interval",
            "amountCents",
            "currency",
            "currentPeriodEnd",
            "cancelAtPeriodEnd",
            "manual",
            "paidThrough",
        }
        assert subscription["manual"] is False and subscription["paidThrough"] is None
        assert subscription["status"] == "canceled"
        assert subscription["interval"] == "month"
        assert subscription["amountCents"] == 9_900
        assert subscription["currency"] == "USD"
        # The deletion event carried cancel_at_period_end=false and wins.
        assert subscription["cancelAtPeriodEnd"] is False
        assert STRIPE_SUBSCRIPTION not in json.dumps(listed)
        quotes = (await http.get("/v1/portal/quotes", headers=bearer(token))).json()["quotes"]
        assert quotes[0]["id"].startswith("gvq_") and quotes[0]["id"] == quote["id"]
        assert quotes[0]["billing"] == "recurring" and quotes[0]["interval"] == "month"
        assert quotes[0]["status"] == "paid"
        assert quotes[0]["id"] == subscription["quoteId"]

        # Billing portal opens against the stored Stripe customer.
        opened = await http.post("/v1/portal/billing-portal", headers=bearer(token))
        assert opened.status_code == 200
        assert opened.json() == {"url": "https://billing.stripe.com/p/session/test"}
        assert billing.portals[-1].customer_ref == STRIPE_CUSTOMER
        assert billing.portals[-1].return_url == f"{SITE_URL}/portal"

    await immediate_worker(portal.application).drain()
    notices = texts_of(portal.owner_replies, "Subscription for Jane Doe")
    assert notices == [
        "Subscription for Jane Doe renewed USD 74.25",
        "Subscription for Jane Doe payment failed (USD 99.00 monthly)",
        "Subscription for Jane Doe cancelled (USD 99.00 monthly)",
    ]
    paid = texts_of(portal.owner_replies, "Quote gvq_")
    assert len(paid) == 1 and paid[0].endswith("(monthly subscription started)")
    async with session_factory() as session:
        outbound = (await session.scalars(select(OutboundMessage))).all()
    assert len([m for m in outbound if m.correlation_id.startswith("subscription:")]) == 3


@pytest.mark.asyncio
async def test_billing_portal_is_404_without_a_stripe_customer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(session_factory, billing=BillingFake())
    async with client(portal) as http:
        token, _ = await sign_in(session_factory, portal, http)
        response = await http.post("/v1/portal/billing-portal", headers=bearer(token))
        assert response.status_code == 404
        empty = await http.get("/v1/portal/subscriptions", headers=bearer(token))
        assert empty.json() == {"subscriptions": []}


# -- e-mail adapter and worker ----------------------------------------------


@pytest.mark.asyncio
async def test_login_email_carries_the_link_and_subject() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        seen["idempotency"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"id": "email_1"})

    adapter = ResendPortalLoginEmailAdapter(
        ResendSettings(api_key="re_test", from_address="quotes@gudvector.com"),  # noqa: S106
        httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    receipt = await adapter.send_login_link(
        PortalLoginEmailRequest(
            business_id=BusinessId(uuid4()),
            to=EMAIL,
            business_display_name=DISPLAY_NAME,
            login_url=f"{SITE_URL}/portal/login?token=raw-token",
            idempotency_key="portal-login:abc",
            expires_at=NOW + LOGIN_TOKEN_TTL,
        )
    )
    assert receipt.status is DeliveryStatus.ACCEPTED
    sent = seen["json"]
    assert isinstance(sent, dict)
    assert sent["to"] == [EMAIL]
    assert sent["subject"] == f"Sign in to your {DISPLAY_NAME} account"
    assert f"{SITE_URL}/portal/login?token=raw-token" in str(sent["text"])
    assert seen["idempotency"] == "portal-login:abc"


@pytest.mark.asyncio
async def test_worker_sends_the_login_email_through_the_port(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    sent: list[PortalLoginEmailRequest] = []

    class LoginEmailFake:
        async def send_login_link(self, request: PortalLoginEmailRequest) -> DeliveryReceipt:
            sent.append(request)
            return DeliveryReceipt(status=DeliveryStatus.ACCEPTED, occurred_at=FAKE_NOW)

    portal = await portal_business(session_factory)
    wired = build_application(
        ApplicationPorts(
            owner_replies=portal.owner_replies,
            quote_drafting=PhoneAwareDrafting(recipient()),
            quote_delivery=portal.delivery,
            portal_login_email=LoginEmailFake(),
            transcription=TranscriptionFake({}),
            completeness_review=MarkerCompletenessReviewer(),
            checklist_evidence=MarkerChecklistEvidenceAttributor(),
            report_generation=DeterministicReportGenerator(),
        ),
        session_factory=session_factory,
        now=Clock(),
    )
    await wired.portal.request_login(PUBLIC_KEY, EMAIL)
    report = await immediate_worker(wired).drain()
    assert sum(r.succeeded for r in report) >= 1
    assert len(sent) == 1
    assert sent[0].to == EMAIL
    assert sent[0].subject == f"Sign in to your {DISPLAY_NAME} account"
    assert sent[0].login_url.startswith(f"{SITE_URL}/portal/login?token=")
    assert sent[0].business_id == portal.business_id


@pytest.mark.asyncio
async def test_worker_does_not_send_a_login_link_whose_token_has_expired(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    sent: list[PortalLoginEmailRequest] = []

    class LoginEmailFake:
        async def send_login_link(self, request: PortalLoginEmailRequest) -> DeliveryReceipt:
            sent.append(request)
            return DeliveryReceipt(status=DeliveryStatus.ACCEPTED, occurred_at=FAKE_NOW)

    class SkippableClock(Clock):
        def advance(self, delta: timedelta) -> None:
            self._now += delta

    portal = await portal_business(session_factory)
    clock = SkippableClock()
    wired = build_application(
        ApplicationPorts(
            owner_replies=portal.owner_replies,
            quote_drafting=PhoneAwareDrafting(recipient()),
            quote_delivery=portal.delivery,
            portal_login_email=LoginEmailFake(),
            transcription=TranscriptionFake({}),
            completeness_review=MarkerCompletenessReviewer(),
            checklist_evidence=MarkerChecklistEvidenceAttributor(),
            report_generation=DeterministicReportGenerator(),
        ),
        session_factory=session_factory,
        now=clock,
    )
    await wired.portal.request_login(PUBLIC_KEY, EMAIL)
    # The worker only gets to the command after the token's 15 minutes are up.
    clock.advance(LOGIN_TOKEN_TTL + timedelta(seconds=1))
    report = await immediate_worker(wired).drain()
    assert sum(r.succeeded for r in report) >= 1 and not any(r.failed for r in report)
    assert sent == []


def test_subscription_record_applies_only_the_fields_an_event_carries() -> None:
    record = QuoteSubscriptionRecord(
        subscription_id=uuid4(),
        business_id=BusinessId(uuid4()),
        quote_id=uuid4(),
        customer_id=uuid4(),
        provider="stripe",
        subscription_ref=STRIPE_SUBSCRIPTION,
        status="active",
        interval=BillingInterval.MONTH,
        amount_minor=9_900,
        currency="USD",
        created_at=NOW,
        updated_at=NOW,
    )
    event = parse_checkout_event(
        invoice_event(event_id="e", event_type="invoice.payment_failed")
    ).subscription
    assert event is not None
    later = NOW + timedelta(days=1)
    updated = record.apply(event, later)
    assert updated.status == "active"  # invoices carry no status
    assert updated.amount_minor == 9_900
    assert updated.current_period_end == datetime.fromtimestamp(1_800_000_000, UTC)
    assert updated.updated_at == later
