"""Owner dashboard: owner sign-in through the portal login, role separation,
tenant isolation, approvals, settings and the merged calendar."""

from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import OwnerReplyFake, TranscriptionFake
from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.composition import Application, ApplicationPorts, build_application
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import (
    INTAKE_BOOKING_ARRANGE_COMMAND_TYPE,
    INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
)
from gvas.domain.owner import (
    CalendarEvent,
    CalendarEventSource,
    OwnerCalendarError,
    calendar_feed_host,
    normalize_calendar_feed_url,
)
from gvas.domain.quotes import QUOTE_DELIVERY_COMMAND_TYPE
from gvas.infrastructure.calendar import IcsCalendarFeed, parse_calendar_feed
from gvas.infrastructure.models import OutboxMessage, QuoteRecord
from gvas.infrastructure.repositories import SqlBusinessRepository
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.owner import create_owner_router
from gvas.interfaces.http.portal import create_portal_router
from test_composition import Clock, inbound, seed_business
from test_customer_portal import login_commands
from test_hosted_quotes import CALENDLY_URL, DISPLAY_NAME, SITE_URL, HostedEmailDelivery
from test_intake_booking import PUBLIC_KEY as INTAKE_PUBLIC_KEY
from test_intake_booking import AvailabilityFake, commands_of, reach_awaiting_owner
from test_pilot_runtime import immediate_worker
from test_portal_quote_handoff import PhoneAwareDrafting, recipient

OWNER_EMAIL = "owner@example.test"
CUSTOMER_EMAIL = "jane@example.test"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_START = datetime(2026, 3, 2, tzinfo=UTC)
WINDOW_END = WINDOW_START + timedelta(days=7)
FEED_URL = "https://calendar.example.com/ical/private-abc123/basic.ics"


class BookedEventsFake:
    def __init__(self, events: tuple[CalendarEvent, ...] = (), *, error: bool = False) -> None:
        self.events = events
        self.error = error

    def serves(self, business_id: BusinessId) -> bool:
        return True

    async def upcoming(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[CalendarEvent, ...]:
        if self.error:
            raise OwnerCalendarError("Calendly bookings couldn't be loaded right now.")
        return self.events


class CalendarFeedFake:
    def __init__(self, events: tuple[CalendarEvent, ...] = (), *, error: bool = False) -> None:
        self.feed_events = events
        self.error = error
        self.urls: list[str] = []

    async def events(
        self, feed_url: str, start: datetime, end: datetime
    ) -> tuple[CalendarEvent, ...]:
        self.urls.append(feed_url)
        if self.error:
            raise OwnerCalendarError("Your calendar link couldn't be read. Check it in Settings.")
        return self.feed_events


async def owner_business(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    public_key: str = "gvb_owner_test",
    owner_email: str | None = OWNER_EMAIL,
    customer_email: str = CUSTOMER_EMAIL,
    approve: bool = False,
    booked_events: BookedEventsFake | None = None,
    calendar_feed: CalendarFeedFake | None = None,
) -> tuple[Application, BusinessId]:
    """A hosted business with one quote to ``customer_email``; approved by
    text when ``approve``, else waiting for the owner's OK."""

    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id,
            site_url=SITE_URL,
            display_name=DISPLAY_NAME,
            calendly_url=CALENDLY_URL,
            public_key=public_key,
            owner_email=owner_email,
            now=NOW,
        )
        await session.commit()
    application = build_application(
        ApplicationPorts(
            owner_replies=OwnerReplyFake(),
            quote_drafting=PhoneAwareDrafting(recipient(email=customer_email)),
            quote_delivery=HostedEmailDelivery(),
            transcription=TranscriptionFake({}),
            completeness_review=MarkerCompletenessReviewer(),
            checklist_evidence=MarkerChecklistEvidenceAttributor(),
            report_generation=DeterministicReportGenerator(),
            booked_events=booked_events,
            calendar_feed=calendar_feed,
        ),
        session_factory=session_factory,
        now=Clock(),
    )
    worker = immediate_worker(application)
    await application.ingest_service.ingest(
        inbound(business_id, "quote: mold inspection 250", message_key=f"quote-{business_id}")
    )
    await worker.drain()
    if approve:
        await application.ingest_service.ingest(
            inbound(business_id, "approve", message_key=f"approve-{business_id}")
        )
        for _ in range(4):
            await worker.drain()
    return application, business_id


def http_client(application: Application) -> httpx.AsyncClient:
    async def origins() -> frozenset[str]:
        return frozenset({SITE_URL})

    app = create_app(
        routers=(
            create_portal_router(application.portal, owner=application.owner),
            create_owner_router(application.owner),
        ),
        cors_origins=origins,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def raw_token(command: OutboxMessage) -> str:
    login_url = command.payload["login_url"]
    assert isinstance(login_url, str) and urlparse(login_url).path == "/portal/login"
    return parse_qs(urlparse(login_url).query)["token"][0]


async def sign_in(
    session_factory: async_sessionmaker[AsyncSession],
    http: httpx.AsyncClient,
    business_id: BusinessId,
    *,
    public_key: str = "gvb_owner_test",
    email: str = OWNER_EMAIL,
) -> dict[str, object]:
    response = await http.post(f"/v1/businesses/{public_key}/portal/login", json={"email": email})
    assert response.status_code == 202 and response.json() == {}
    commands = [
        command
        for command in await login_commands(session_factory, business_id)
        if command.payload["to"] == email.strip().lower()
    ]
    assert commands, f"{email} should have been e-mailed a link"
    exchanged = await http.post("/v1/portal/sessions", json={"token": raw_token(commands[-1])})
    assert exchanged.status_code == 200, exchanged.text
    payload: dict[str, object] = exchanged.json()
    return payload


def bearer(payload: dict[str, object]) -> dict[str, str]:
    return {"Authorization": f"Bearer {payload['sessionToken']}"}


# -- sign-in and roles -------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_email_signs_in_to_an_owner_session_only_owner_routes_accept(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory, approve=True)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id, email="  Owner@Example.TEST ")
        assert owner["role"] == "owner"
        assert owner["owner"] == {"email": OWNER_EMAIL}
        assert owner["business"] == {"displayName": DISPLAY_NAME, "siteUrl": SITE_URL}

        me = await http.get("/v1/owner/me", headers=bearer(owner))
        assert me.status_code == 200 and me.json()["role"] == "owner"
        # An owner session is not a customer session.
        assert (await http.get("/v1/portal/me", headers=bearer(owner))).status_code == 401

        customer = await sign_in(session_factory, http, business_id, email=CUSTOMER_EMAIL)
        assert customer["role"] == "customer"
        assert (await http.get("/v1/portal/me", headers=bearer(customer))).status_code == 200
        for path in ("/v1/owner/me", "/v1/owner/quotes", "/v1/owner/settings"):
            response = await http.get(path, headers=bearer(customer))
            assert response.status_code == 401 and response.json() == {"detail": "unauthorized"}
        assert (await http.get("/v1/owner/me")).status_code == 401

        revoked = await http.delete("/v1/owner/sessions", headers=bearer(owner))
        assert revoked.status_code == 204
        assert (await http.get("/v1/owner/me", headers=bearer(owner))).status_code == 401
        again = await http.delete("/v1/owner/sessions", headers=bearer(owner))
        assert again.status_code == 401


@pytest.mark.asyncio
async def test_owner_link_is_single_use_and_changing_the_owner_ends_sessions(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        commands = await login_commands(session_factory, business_id)
        reused = await http.post("/v1/portal/sessions", json={"token": raw_token(commands[-1])})
        assert reused.status_code == 401

        async with session_factory() as session:
            await SqlBusinessRepository(session).configure_site(
                business_id, owner_email="new-owner@example.test", now=NOW
            )
            await session.commit()
        assert (await http.get("/v1/owner/me", headers=bearer(owner))).status_code == 401


@pytest.mark.asyncio
async def test_a_business_without_an_owner_email_issues_no_owner_link(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory, owner_email=None)
    async with http_client(application) as http:
        response = await http.post(
            "/v1/businesses/gvb_owner_test/portal/login", json={"email": OWNER_EMAIL}
        )
        assert response.status_code == 202 and response.json() == {}
    assert await login_commands(session_factory, business_id) == []


# -- reads and tenant isolation ---------------------------------------------


@pytest.mark.asyncio
async def test_owner_sees_only_their_own_business(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app_a, business_a = await owner_business(session_factory, approve=True)
    app_b, business_b = await owner_business(
        session_factory,
        public_key="gvb_owner_b",
        owner_email="owner-b@example.test",
        customer_email="bob@example.test",
    )
    async with http_client(app_a) as http:
        owner_a = await sign_in(session_factory, http, business_a)
        owner_b = await sign_in(
            session_factory,
            http,
            business_b,
            public_key="gvb_owner_b",
            email="owner-b@example.test",
        )
        quotes_a = (await http.get("/v1/owner/quotes", headers=bearer(owner_a))).json()["quotes"]
        quotes_b = (await http.get("/v1/owner/quotes", headers=bearer(owner_b))).json()["quotes"]
        assert [quote["customer"]["email"] for quote in quotes_a] == [CUSTOMER_EMAIL]
        assert [quote["customer"]["email"] for quote in quotes_b] == ["bob@example.test"]
        assert quotes_a[0]["status"] in {"delivery_pending", "delivered"}
        assert not quotes_a[0]["needsApproval"]
        assert quotes_a[0]["totalCents"] == 25_000
        assert quotes_a[0]["sentAt"] is not None
        assert quotes_b[0]["sentAt"] is None

        customers = (await http.get("/v1/owner/customers", headers=bearer(owner_a))).json()
        assert [row["email"] for row in customers["customers"]] == [CUSTOMER_EMAIL]
        assert customers["customers"][0]["quoteIds"] == [quotes_a[0]["id"]]

        # Business A's owner cannot act on business B's quote.
        foreign = await http.post(
            f"/v1/owner/quotes/{quotes_b[0]['id']}/approve", headers=bearer(owner_a)
        )
        assert foreign.status_code == 404
        for path in ("/v1/owner/subscriptions", "/v1/owner/bookings", "/v1/owner/requests"):
            response = await http.get(path, headers=bearer(owner_a))
            assert response.status_code == 200, path
    del app_b


# -- quote approvals ---------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_approve_queues_delivery_like_a_text_approval(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        quotes = (await http.get("/v1/owner/quotes", headers=bearer(owner))).json()["quotes"]
        assert len(quotes) == 1 and quotes[0]["needsApproval"]
        assert quotes[0]["status"] == "awaiting_approval"

        approved = await http.post(
            f"/v1/owner/quotes/{quotes[0]['id']}/approve", headers=bearer(owner)
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["quote"]["status"] == "approved"
        delivery = await commands_of(session_factory, business_id, QUOTE_DELIVERY_COMMAND_TYPE)
        assert len(delivery) == 1

        again = await http.post(
            f"/v1/owner/quotes/{quotes[0]['id']}/approve", headers=bearer(owner)
        )
        assert again.status_code == 409
        for _ in range(4):
            await immediate_worker(application).drain()
        refreshed = (await http.get("/v1/owner/quotes", headers=bearer(owner))).json()["quotes"]
        assert refreshed[0]["status"] in {"delivery_pending", "delivered"}
    async with session_factory() as session:
        row = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == business_id)
        )
    assert row is not None and row.customer_id is not None


@pytest.mark.asyncio
async def test_dashboard_reject_sends_nothing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        quotes = (await http.get("/v1/owner/quotes", headers=bearer(owner))).json()["quotes"]
        rejected = await http.post(
            f"/v1/owner/quotes/{quotes[0]['id']}/reject", headers=bearer(owner)
        )
        assert rejected.status_code == 200
        assert rejected.json()["quote"]["status"] == "rejected"
        missing = await http.post("/v1/owner/quotes/gvq_nope/reject", headers=bearer(owner))
        assert missing.status_code == 404
    assert await commands_of(session_factory, business_id, QUOTE_DELIVERY_COMMAND_TYPE) == []


# -- booking approvals -------------------------------------------------------


async def booking_owner(
    session_factory: async_sessionmaker[AsyncSession], availability: AvailabilityFake
) -> tuple[Application, BusinessId, str]:
    application, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id, owner_email=OWNER_EMAIL, now=NOW
        )
        await session.commit()
    return application, business_id, reference


@pytest.mark.asyncio
async def test_dashboard_approves_a_booking_request(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, business_id, reference = await booking_owner(session_factory, availability)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id, public_key=INTAKE_PUBLIC_KEY)
        bookings = (await http.get("/v1/owner/bookings", headers=bearer(owner))).json()
        assert [row["reference"] for row in bookings["bookings"]] == [reference]
        assert bookings["bookings"][0]["needsDecision"]
        assert bookings["bookings"][0]["customer"]["name"] == "Jane Doe"
        assert "notes" in bookings["bookings"][0]

        approved = await http.post(
            f"/v1/owner/bookings/{reference.upper()}/approve", headers=bearer(owner)
        )
        assert approved.status_code == 200
        assert approved.json()["applied"] is True
        assert approved.json()["message"].startswith(f"Approved booking {reference}")
        repeat = await http.post(f"/v1/owner/bookings/{reference}/approve", headers=bearer(owner))
        assert repeat.json() == {
            "applied": False,
            "message": f"Booking {reference} is already approved.",
        }
    arrange = await commands_of(session_factory, business_id, INTAKE_BOOKING_ARRANGE_COMMAND_TYPE)
    assert len(arrange) == 1
    for _ in range(4):
        await immediate_worker(application).drain()
    assert availability.book_calls


@pytest.mark.asyncio
async def test_dashboard_declines_a_booking_with_a_reason(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, business_id, reference = await booking_owner(session_factory, availability)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id, public_key=INTAKE_PUBLIC_KEY)
        declined = await http.post(
            f"/v1/owner/bookings/{reference}/decline",
            json={"reason": "fully booked\nthat week"},
            headers=bearer(owner),
        )
        assert declined.status_code == 200 and declined.json()["applied"] is True
    emails = await commands_of(session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE)
    bodies = [command.payload["body"] for command in emails]
    assert any(isinstance(body, str) and "fully booked that week" in body for body in bodies)
    assert (
        await commands_of(session_factory, business_id, INTAKE_BOOKING_ARRANGE_COMMAND_TYPE) == []
    )
    assert availability.book_calls == []


# -- settings ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_update_and_the_calendar_link_is_never_returned(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        current = (await http.get("/v1/owner/settings", headers=bearer(owner))).json()["settings"]
        assert current["calendarFeed"] == {"connected": False, "host": None}
        assert current["ownerEmail"] == OWNER_EMAIL

        updated = await http.patch(
            "/v1/owner/settings",
            json={
                "displayName": "  Bay Area   Services ",
                "intakeBrief": "Plumbing repairs in the East Bay.",
                "notificationEmail": "Office@Example.test",
                "calendarFeedUrl": FEED_URL.replace("https://", "webcal://"),
                "timezone": " America/Chicago ",
            },
            headers=bearer(owner),
        )
        assert updated.status_code == 200, updated.text
        assert "private-abc123" not in updated.text
        settings = updated.json()["settings"]
        assert settings["displayName"] == "Bay Area Services"
        assert settings["intakeBrief"] == "Plumbing repairs in the East Bay."
        assert settings["notificationEmail"] == "office@example.test"
        assert settings["calendarFeed"] == {"connected": True, "host": "calendar.example.com"}
        assert settings["timezone"] == "America/Chicago"
        me = (await http.get("/v1/owner/me", headers=bearer(owner))).json()
        assert me["business"]["timezone"] == "America/Chicago"
        assert (
            "private-abc123"
            not in (await http.get("/v1/owner/settings", headers=bearer(owner))).text
        )
        async with session_factory() as session:
            stored = await SqlBusinessRepository(session).get(business_id)
        assert stored is not None and stored.calendar_feed_url == FEED_URL
        assert "private-abc123" not in repr(stored)

        for body in (
            {"calendarFeedUrl": "http://calendar.example.com/feed.ics"},
            {"calendarFeedUrl": "https://localhost/feed.ics"},
            {"calendarFeedUrl": "https://10.0.0.5/feed.ics"},
            {"calendlyUrl": "http://calendly.com/x"},
            {"displayName": "   "},
            {"notificationEmail": "not-an-email"},
            {"timezone": "Pacific"},
        ):
            rejected = await http.patch("/v1/owner/settings", json=body, headers=bearer(owner))
            assert rejected.status_code == 422, body

        cleared = await http.patch(
            "/v1/owner/settings",
            json={"calendarFeedUrl": "", "intakeBrief": "", "timezone": ""},
            headers=bearer(owner),
        )
        assert cleared.json()["settings"]["timezone"] is None
        assert cleared.json()["settings"]["calendarFeed"]["connected"] is False
        assert cleared.json()["settings"]["intakeBrief"] is None


# -- calendar ----------------------------------------------------------------


def event(source: CalendarEventSource, title: str, start: datetime) -> CalendarEvent:
    return CalendarEvent(source=source, title=title, start=start, end=start + timedelta(hours=1))


@pytest.mark.asyncio
async def test_calendar_merges_bookings_and_the_owner_calendar(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    estimate = WINDOW_START + timedelta(hours=17)
    booked = BookedEventsFake((event(CalendarEventSource.BOOKING, "Estimate", estimate),))
    feed = CalendarFeedFake(
        (
            # Calendly's copy of the booking in the owner's own calendar.
            event(CalendarEventSource.CALENDAR, "Estimate with Jordan", estimate),
            event(CalendarEventSource.CALENDAR, "Dentist", estimate + timedelta(days=1)),
        )
    )
    application, business_id = await owner_business(
        session_factory, booked_events=booked, calendar_feed=feed
    )
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        params = {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()}
        without_feed = await http.get("/v1/owner/calendar", params=params, headers=bearer(owner))
        assert [row["title"] for row in without_feed.json()["events"]] == ["Estimate"]
        assert feed.urls == [], "no feed is read before the owner adds one"

        await http.patch(
            "/v1/owner/settings", json={"calendarFeedUrl": FEED_URL}, headers=bearer(owner)
        )
        merged = (await http.get("/v1/owner/calendar", params=params, headers=bearer(owner))).json()
        assert [(row["source"], row["title"]) for row in merged["events"]] == [
            ("booking", "Estimate"),
            ("calendar", "Dentist"),
        ]
        assert merged["problems"] == []
        assert feed.urls == [FEED_URL]

        feed.error = True
        partial = (
            await http.get("/v1/owner/calendar", params=params, headers=bearer(owner))
        ).json()
        assert [row["title"] for row in partial["events"]] == ["Estimate"]
        assert partial["problems"] == ["Your calendar link couldn't be read. Check it in Settings."]

        too_long = await http.get(
            "/v1/owner/calendar",
            params={
                "start": WINDOW_START.isoformat(),
                "end": (WINDOW_START + timedelta(days=90)).isoformat(),
            },
            headers=bearer(owner),
        )
        assert too_long.status_code == 422


# -- calendar feed adapter ---------------------------------------------------

FEED_BODY = b"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//EN
BEGIN:VEVENT
UID:weekly@test
DTSTART:20260302T160000Z
DTEND:20260302T163000Z
RRULE:FREQ=WEEKLY;COUNT=4
SUMMARY:Crew standup
END:VEVENT
BEGIN:VEVENT
UID:allday@test
DTSTART;VALUE=DATE:20260304
DTEND;VALUE=DATE:20260305
SUMMARY:Supplier day
END:VEVENT
BEGIN:VEVENT
UID:cancelled@test
DTSTART:20260305T180000Z
DTEND:20260305T190000Z
STATUS:CANCELLED
SUMMARY:Cancelled thing
END:VEVENT
BEGIN:VEVENT
UID:outside@test
DTSTART:20260401T180000Z
DTEND:20260401T190000Z
SUMMARY:Next month
END:VEVENT
END:VCALENDAR
"""


def test_feed_parsing_expands_recurrences_and_skips_cancelled() -> None:
    events = parse_calendar_feed(FEED_BODY, WINDOW_START, WINDOW_START + timedelta(days=14))
    assert [(item.title, item.start.isoformat(), item.all_day) for item in events] == [
        ("Crew standup", "2026-03-02T16:00:00+00:00", False),
        ("Supplier day", "2026-03-04T00:00:00+00:00", True),
        ("Crew standup", "2026-03-09T16:00:00+00:00", False),
    ]
    assert all(item.source is CalendarEventSource.CALENDAR for item in events)


def test_calendar_links_are_normalized_and_private_networks_refused() -> None:
    assert normalize_calendar_feed_url("webcal://p01-caldav.icloud.com/published/2/x") == (
        "https://p01-caldav.icloud.com/published/2/x"
    )
    assert calendar_feed_host(FEED_URL) == "calendar.example.com"
    for bad in (
        "http://calendar.example.com/x.ics",
        "https://user:pw@calendar.example.com/x.ics",
        "https://calendar.example.com:8443/x.ics",
        "https://127.0.0.1/x.ics",
        "https://[::1]/x.ics",
        "https://169.254.169.254/latest",
        "https://printer.local/x.ics",
        "https://intranet/x.ics",
        "ftp://calendar.example.com/x.ics",
    ):
        with pytest.raises(ValueError):
            normalize_calendar_feed_url(bad)


@pytest.mark.asyncio
async def test_feed_adapter_follows_safe_redirects_and_refuses_unsafe_ones() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "https://p02.example.com/feed.ics"})
        if request.url.path == "/to-private":
            return httpx.Response(302, headers={"location": "https://192.168.1.10/feed.ics"})
        if request.url.path == "/big":
            return httpx.Response(200, content=b"x" * 6_000_000)
        if request.url.path == "/missing":
            return httpx.Response(404)
        return httpx.Response(200, content=FEED_BODY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        feed = IcsCalendarFeed(client, resolve_hosts=False)
        events = await feed.events("https://calendar.example.com/start", WINDOW_START, WINDOW_END)
        assert [item.title for item in events] == ["Crew standup", "Supplier day"]
        for path in ("/to-private", "/big", "/missing"):
            with pytest.raises(OwnerCalendarError) as raised:
                await feed.events(f"https://calendar.example.com{path}", WINDOW_START, WINDOW_END)
            assert "calendar.example.com" not in str(raised.value)
