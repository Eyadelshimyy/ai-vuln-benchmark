"""Calendly booking webhooks: signature check, then the booking event is
applied to the pending request — same-time confirms, a different time
re-routes to the owner, and a cancellation closes it.
"""

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.composition import Application
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import (
    INTAKE_BOOKING_CANCEL_COMMAND_TYPE,
    BookingKind,
    BookingResult,
)
from gvas.infrastructure.calendly.config import CalendlyInstallation
from gvas.infrastructure.calendly.events import CalendlyPayloadError, parse_calendly_event
from gvas.infrastructure.calendly.ingress import (
    CalendlyIngressResult,
    CalendlyWebhookIngress,
)
from gvas.infrastructure.calendly.signature import (
    SIGNATURE_HEADER,
    CalendlySignatureError,
    verify_calendly_signature,
)
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.calendly import create_calendly_router
from test_composition import inbound
from test_intake_booking import (
    EMAIL,
    AvailabilityFake,
    commands_of,
    conversation_row,
    intake_app,
    intake_business,
    reach_awaiting_owner,
)
from test_pilot_runtime import immediate_worker, texts_of

SIGNING_KEY = "whsec-test-key"
USER_URI = "https://api.calendly.com/users/TESTUSER"
EVENT_URI = "https://api.calendly.com/scheduled_events/EVENT1"
NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def _signature(body: bytes, key: str, ts: int) -> str:
    digest = hmac.new(key.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={digest}"


def _payload(
    *,
    event: str = "invitee.created",
    email: str = EMAIL,
    user_uri: str = USER_URI,
    event_uri: str = EVENT_URI,
    event_type: str | None = None,
    reference: str | None = None,
    start: datetime,
    end: datetime | None = None,
) -> bytes:
    # Stored sqlite datetimes come back naive — the provider sends UTC.
    start = start if start.tzinfo else start.replace(tzinfo=UTC)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    scheduled: dict[str, object] = {
        "uri": event_uri,
        "start_time": start.isoformat().replace("+00:00", "Z"),
        "event_memberships": [{"user": user_uri}],
    }
    if end is not None:
        scheduled["end_time"] = end.isoformat().replace("+00:00", "Z")
    if event_type is not None:
        scheduled["event_type"] = event_type
    payload: dict[str, object] = {"email": email, "scheduled_event": scheduled}
    if reference is not None:
        payload["tracking"] = {"utm_content": reference}
    return json.dumps({"event": event, "payload": payload}).encode()


async def _deliver(
    application: Application,
    business_id: BusinessId,
    body: bytes,
    *,
    key: str = SIGNING_KEY,
    clock_ts: int | None = None,
) -> CalendlyIngressResult:
    ts = clock_ts if clock_ts is not None else int(datetime.now(UTC).timestamp())
    ingress = CalendlyWebhookIngress(
        application.intake_booking_events,
        (CalendlyInstallation(business_id=business_id, user_uri=USER_URI),),
        signing_key=key,
        clock=lambda: datetime.fromtimestamp(ts + 1, UTC),
    )
    outcome = await ingress.handle(body=body, signature=_signature(body, key, ts))
    return outcome.result


def test_signature_verification_accepts_and_rejects() -> None:
    body = b'{"event":"invitee.created"}'
    ts = int(NOW.timestamp())
    good = _signature(body, SIGNING_KEY, ts)

    def clock() -> datetime:
        return datetime.fromtimestamp(ts + 60, UTC)

    verify_calendly_signature(body, good, SIGNING_KEY, now=clock)

    with pytest.raises(CalendlySignatureError):
        verify_calendly_signature(body, None, SIGNING_KEY, now=clock)
    with pytest.raises(CalendlySignatureError):
        verify_calendly_signature(body, good, "other-key", now=clock)
    with pytest.raises(CalendlySignatureError):
        verify_calendly_signature(body, f"t={ts},v1={'0' * 64}", SIGNING_KEY, now=clock)
    stale = _signature(body, SIGNING_KEY, ts - 400)
    with pytest.raises(CalendlySignatureError):
        verify_calendly_signature(body, stale, SIGNING_KEY, now=clock)


def test_parse_extracts_the_booking_fact() -> None:
    start = datetime(2026, 1, 9, 17, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    event = parse_calendly_event(_payload(start=start, end=end))
    assert event is not None
    assert event.user_uri == USER_URI
    assert event.invitee_email == EMAIL
    assert event.event_uri == EVENT_URI
    assert event.start == start
    assert event.end == end

    other = parse_calendly_event(_payload(event="invitee_no_show", start=start))
    assert other is None

    with pytest.raises(CalendlyPayloadError):
        parse_calendly_event(b"not json")
    with pytest.raises(CalendlyPayloadError):
        parse_calendly_event(json.dumps({"event": "invitee.created"}).encode())


@pytest.mark.asyncio
async def test_created_at_the_approved_time_confirms_the_booking(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link="https://x"))
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-link")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved" and row.booking_kind == "link"
    approved_start = row.requested_slot_start
    assert approved_start is not None

    body = _payload(start=approved_start, end=row.requested_slot_end)
    result = await _deliver(application, business_id, body)
    assert result is CalendlyIngressResult.CONFIRMED
    for _ in range(3):
        await immediate_worker(application).drain()

    row = await conversation_row(session_factory, business_id)
    assert row.booking_kind == "booked"
    assert row.booked_event_uri == EVENT_URI
    notices = [t for t in texts_of(owner, "Booking") if "on the calendar" in t]
    assert notices, "owner hears the customer confirmed the approved time"

    # A redelivery of the same event is a no-op.
    again = await _deliver(application, business_id, body)
    assert again is CalendlyIngressResult.IGNORED


@pytest.mark.asyncio
async def test_created_at_a_different_time_routes_back_to_the_owner(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link="https://x"))
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-mis")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved"
    assert row.requested_slot_start is not None

    picked = row.requested_slot_start + timedelta(hours=3)
    body = _payload(start=picked, end=picked + timedelta(hours=1))
    result = await _deliver(application, business_id, body)
    assert result is CalendlyIngressResult.REROUTED
    for _ in range(3):
        await immediate_worker(application).drain()

    row = await conversation_row(session_factory, business_id)
    assert row.state == "awaiting_owner"
    assert row.requested_slot_start == picked
    assert row.booked_event_uri == EVENT_URI
    assert row.booking_kind is None
    notices = [t for t in texts_of(owner, "Booking") if "instead of the requested" in t]
    assert notices, "the owner is asked to decide on the actual time"


@pytest.mark.asyncio
async def test_approve_after_reroute_reconciles_without_rebooking(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(
        result=BookingResult(kind=BookingKind.LINK, link="https://x"),
        found=BookingResult(kind=BookingKind.BOOKED),
    )
    application, _owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-re")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    picked = row.requested_slot_start + timedelta(hours=3)
    await _deliver(
        application,
        business_id,
        _payload(start=picked, end=picked + timedelta(hours=1)),
    )

    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-re-2")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    # The first approve minted the link (one book call); the re-approved
    # reroute finds the webhook-created event instead of booking again.
    assert len(availability.book_calls) == 1
    assert availability.find_calls, "arrange reconciles through find_booking"
    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved" and row.booking_kind == "booked"


@pytest.mark.asyncio
async def test_decline_after_reroute_cancels_the_calendly_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link="https://x"))
    application, _owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-de")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    picked = row.requested_slot_start + timedelta(hours=3)
    await _deliver(
        application,
        business_id,
        _payload(start=picked, end=picked + timedelta(hours=1)),
    )

    await application.ingest_service.ingest(
        inbound(
            business_id,
            f"decline booking {reference} wrong time",
            message_key="decline-re",
        )
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    row = await conversation_row(session_factory, business_id)
    assert row.state == "declined"
    cancels = await commands_of(session_factory, business_id, INTAKE_BOOKING_CANCEL_COMMAND_TYPE)
    assert cancels and cancels[0].payload["event_uri"] == EVENT_URI
    assert availability.cancel_calls == [(business_id, EVENT_URI)]


@pytest.mark.asyncio
async def test_canceled_event_closes_the_request(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    row = await conversation_row(session_factory, business_id)
    start = row.requested_slot_start
    assert start is not None

    created = await _deliver(
        application, business_id, _payload(start=start, end=start + timedelta(hours=1))
    )
    assert created is CalendlyIngressResult.CONFIRMED
    canceled = await _deliver(
        application,
        business_id,
        _payload(event="invitee.canceled", start=start, end=start + timedelta(hours=1)),
    )
    assert canceled is CalendlyIngressResult.CANCELED
    for _ in range(3):
        await immediate_worker(application).drain()

    row = await conversation_row(session_factory, business_id)
    assert row.state == "closed"
    notices = [t for t in texts_of(owner, "Booking") if "was canceled" in t]
    assert notices


@pytest.mark.asyncio
async def test_unknown_calendly_user_is_dropped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    application, _owner = intake_app(session_factory)
    body = _payload(
        start=datetime(2026, 1, 9, 17, 0, tzinfo=UTC),
        user_uri="https://api.calendly.com/users/SOMEONE_ELSE",
    )
    result = await _deliver(application, business_id, body)
    assert result is CalendlyIngressResult.UNKNOWN_INSTALLATION


@pytest.mark.asyncio
async def test_webhook_route_verifies_the_signature(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    application, _owner = intake_app(session_factory)
    ingress = CalendlyWebhookIngress(
        application.intake_booking_events,
        (CalendlyInstallation(business_id=business_id, user_uri=USER_URI),),
        signing_key=SIGNING_KEY,
    )
    app = create_app(routers=(create_calendly_router(ingress),))
    body = _payload(start=datetime(2026, 1, 9, 17, 0, tzinfo=UTC))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        missing = await client.post("/calendly/events", content=body)
        assert missing.status_code == 401

        bad = await client.post(
            "/calendly/events",
            content=body,
            headers={SIGNATURE_HEADER: "t=1,v1=bad"},
        )
        assert bad.status_code == 401

        malformed = await client.post(
            "/calendly/events",
            content=b"nope",
            headers={
                SIGNATURE_HEADER: _signature(
                    b"nope", SIGNING_KEY, int(datetime.now(UTC).timestamp())
                )
            },
        )
        assert malformed.status_code == 400

        ts = int(datetime.now(UTC).timestamp())
        ok = await client.post(
            "/calendly/events",
            content=body,
            headers={SIGNATURE_HEADER: _signature(body, SIGNING_KEY, ts)},
        )
        assert ok.status_code == 200
        assert ok.json()["status"] == "ignored"  # no conversation for that email


@pytest.mark.asyncio
async def test_unrelated_event_on_another_day_is_ignored(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An invitee e-mail match alone is not enough: an event days away from
    the requested slot is not this request's booking."""
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link="https://x"))
    application, _owner, business_id, _reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {_reference}", message_key="approve-day")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    unrelated = row.requested_slot_start + timedelta(days=3)
    result = await _deliver(
        application,
        business_id,
        _payload(
            event_uri="https://api.calendly.com/scheduled_events/OTHER",
            start=unrelated,
            end=unrelated + timedelta(hours=1),
        ),
    )
    assert result is CalendlyIngressResult.IGNORED
    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved" and row.booked_event_uri is None


@pytest.mark.asyncio
async def test_event_for_a_different_event_type_is_ignored(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The arrange step records the booked event type; an event for another
    kind is not this request."""
    availability = AvailabilityFake(
        result=BookingResult(
            kind=BookingKind.LINK,
            link="https://x",
            event_type_uri="https://api.calendly.com/event_types/OURS",
        )
    )
    application, _owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-et")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.booking_event_type_uri == "https://api.calendly.com/event_types/OURS"
    assert row.requested_slot_start is not None

    result = await _deliver(
        application,
        business_id,
        _payload(
            start=row.requested_slot_start,
            end=row.requested_slot_start + timedelta(hours=1),
            event_type="https://api.calendly.com/event_types/OTHER",
        ),
    )
    assert result is CalendlyIngressResult.IGNORED
    row = await conversation_row(session_factory, business_id)
    assert row.booked_event_uri is None


@pytest.mark.asyncio
async def test_reference_bound_event_matches_without_the_email(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The scheduling link carries the request reference as utm_content; an
    event echoing it binds to the request directly."""
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link="https://x"))
    application, _owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    result = await _deliver(
        application,
        business_id,
        _payload(
            email="different-invitee@example.com",
            reference=reference,
            start=row.requested_slot_start,
            end=row.requested_slot_start + timedelta(hours=1),
        ),
    )
    assert result is CalendlyIngressResult.CONFIRMED
    row = await conversation_row(session_factory, business_id)
    assert row.booked_event_uri == EVENT_URI and row.booking_kind == "booked"


@pytest.mark.asyncio
async def test_a_second_created_event_does_not_overwrite_the_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link="https://x"))
    application, _owner, business_id, _reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    picked = row.requested_slot_start + timedelta(hours=3)
    first = await _deliver(
        application,
        business_id,
        _payload(start=picked, end=picked + timedelta(hours=1)),
    )
    assert first is CalendlyIngressResult.REROUTED
    second = await _deliver(
        application,
        business_id,
        _payload(
            event_uri="https://api.calendly.com/scheduled_events/EVENT2",
            start=picked + timedelta(hours=2),
            end=picked + timedelta(hours=3),
        ),
    )
    assert second is CalendlyIngressResult.IGNORED
    row = await conversation_row(session_factory, business_id)
    assert row.booked_event_uri == EVENT_URI


@pytest.mark.asyncio
async def test_cancel_notice_reaches_the_owner_after_confirmation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Creation and cancellation notices dedupe separately: the owner who
    saw the confirmation also sees the cancellation."""
    availability = AvailabilityFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-cx")
    )
    for _ in range(5):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    created = await _deliver(
        application,
        business_id,
        _payload(start=row.requested_slot_start, end=row.requested_slot_start + timedelta(hours=1)),
    )
    assert created is CalendlyIngressResult.CONFIRMED
    canceled = await _deliver(
        application,
        business_id,
        _payload(
            event="invitee.canceled",
            start=row.requested_slot_start,
            end=row.requested_slot_start + timedelta(hours=1),
        ),
    )
    assert canceled is CalendlyIngressResult.CANCELED
    for _ in range(3):
        await immediate_worker(application).drain()
    notices = [t for t in texts_of(owner, "Booking") if "was canceled on Calendly" in t]
    assert notices, "the cancellation notice must not be swallowed by the creation notice"


@pytest.mark.asyncio
async def test_slot_pick_without_an_owner_thread_stays_proposing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """No owner inbound means no notice can land — the pick must not strand
    the customer in awaiting_owner."""
    from gvas.domain.intake import IntakeTurn
    from test_intake_booking import (
        PUBLIC_KEY,
        IntakeAgentFake,
        collected_turn,
        http_client,
        slot,
    )

    business_id = await intake_business(session_factory)
    availability = AvailabilityFake()
    offered = slot(datetime.now(UTC))
    availability.slots = (offered,)
    agent = IntakeAgentFake(
        [
            collected_turn(name="Jane Doe", email=EMAIL, phone="+15555550100", problem="roof leak"),
            collected_turn(address="1 Main St"),
            IntakeTurn(reply="Great, here are openings.", ready_for_slots=True),
        ]
    )
    application, _owner = intake_app(session_factory, agent=agent, availability=availability)

    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        assert created.status_code == 201
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        for message in ("roof leak at my place", "1 Main St", "what times do you have?"):
            response = await client.post(
                f"/v1/intake/conversations/{conversation_id}/messages",
                json={"message": message},
                headers=headers,
            )
        assert response.json()["state"] == "proposing_slots"
        picked = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": f"slot:{offered.start.isoformat()}"},
            headers=headers,
        )
        assert picked.status_code == 200
        assert picked.json()["state"] == "proposing_slots"
        assert "couldn't reach the owner" in picked.json()["reply"]

    row = await conversation_row(session_factory, business_id)
    assert row.state == "proposing_slots"
