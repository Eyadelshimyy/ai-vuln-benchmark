"""Owner time blocks by text: parse, confirm with ``yes``, apply, decline, undo.

The hard rule under test: nothing changes on the booking calendar and no
customer is told anything until the owner answers ``yes``.
"""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import CustomerDeliveryFake, OwnerReplyFake, TranscriptionFake
from gvas.application.calendar_blocks import BLOCK_DECLINE_REASON, BLOCK_NOT_CONFIGURED_REPLY
from gvas.composition import Application, build_application
from gvas.config import IntakeSettings
from gvas.domain.calendar_blocks import (
    DayHours,
    Interval,
    block_confirmation,
    resolve_day,
    subtract_interval,
    unavailable_request,
    unblock_request,
)
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE, AvailabilityError
from gvas.infrastructure.calendar_block_models import CalendarBlockRecord
from gvas.infrastructure.calendly.availability import CalendlyAvailability
from gvas.infrastructure.calendly.config import CalendlySettings
from test_composition import Clock, inbound
from test_intake_booking import (
    CALENDLY_LINK,
    AvailabilityFake,
    IntakeAgentFake,
    commands_of,
    conversation_row,
    reach_awaiting_owner,
)
from test_pilot_runtime import deterministic_ports, immediate_worker, texts_of

WORKDAY = DayHours(intervals=((9 * 60, 17 * 60),), overridden=False)


class ScheduleFake(AvailabilityFake):
    """Availability plus a one-calendar ``ScheduleBlockPort``."""

    def __init__(self) -> None:
        super().__init__()
        self.overrides: dict[date, tuple[Interval, ...]] = {}
        self.set_calls: list[tuple[date, tuple[Interval, ...] | None]] = []
        self.fail_after_write = False

    def serves(self, business_id: BusinessId) -> bool:
        return True

    async def schedule_timezone(self, business_id: BusinessId) -> str | None:
        return "UTC"

    async def day_hours(self, business_id: BusinessId, day: date) -> DayHours:
        if day in self.overrides:
            return DayHours(intervals=self.overrides[day], overridden=True)
        return WORKDAY

    async def set_day_hours(
        self, business_id: BusinessId, day: date, intervals: tuple[Interval, ...] | None
    ) -> None:
        self.set_calls.append((day, intervals))
        if intervals is None:
            self.overrides.pop(day, None)
        else:
            self.overrides[day] = intervals
        if self.fail_after_write:
            self.fail_after_write = False
            raise AvailabilityError("calendly returned http 502")


def block_app(
    session_factory: async_sessionmaker[AsyncSession], schedule: ScheduleFake | None
) -> tuple[Application, OwnerReplyFake]:
    owner = OwnerReplyFake()
    ports = deterministic_ports(owner, TranscriptionFake({}), CustomerDeliveryFake())
    ports = replace(
        ports,
        intake_agent=IntakeAgentFake(),
        availability=schedule,
        schedule_blocks=schedule,
        customer_email=CustomerDeliveryFake(),
    )
    application = build_application(
        ports,
        session_factory=session_factory,
        now=Clock(),
        intake_settings=IntakeSettings(max_conversations_per_day=0),
    )
    return application, owner


async def say(application: Application, business_id: BusinessId, text: str, key: str) -> None:
    await application.ingest_service.ingest(inbound(business_id, text, message_key=key))
    for _ in range(4):
        await immediate_worker(application).drain()


def test_unavailable_requests_read_bare_hours_as_the_working_day() -> None:
    request = unavailable_request("book another time, I'm unavailable 8 to 12")
    assert request is not None
    assert (request.start_minute, request.end_minute, request.day_hint) == (480, 720, None)

    afternoon = unavailable_request("busy tue 1-4")
    assert afternoon is not None
    assert (afternoon.start_minute, afternoon.end_minute, afternoon.day_hint) == (780, 960, "tue")

    named = unavailable_request("block 10/7 9:30am-11 for booking AB12CD")
    assert named is not None
    assert (named.start_minute, named.end_minute) == (570, 660)
    assert (named.day_hint, named.reference) == ("10/7", "ab12cd")

    evening = unavailable_request("I'm out 10-2")
    assert evening is not None
    assert (evening.start_minute, evening.end_minute) == (600, 840)


@pytest.mark.parametrize(
    "text",
    [
        "Quote $1,650 for the 50 gal plus $150 haul-away, 8-12 hours",
        "10-15% off if they're busy",
        "busy block wall 8-10 ft",
        "I'm unavailable",
        "approve booking ab12cd",
        "unavailable 8-12 or 2-4",
    ],
)
def test_prices_sizes_and_vague_texts_are_not_blocks(text: str) -> None:
    assert unavailable_request(text) is None


def test_confirmations_unblock_and_day_resolution() -> None:
    assert block_confirmation("Yes!") is True
    assert block_confirmation("no") is False
    assert block_confirmation("yes, but quote it at $500") is None
    assert unblock_request("unblock tue") == "tue"
    assert unblock_request("Unblock") == ""
    assert unblock_request("please unblock tue") is None

    monday = date(2026, 10, 5)
    assert resolve_day("tue", monday, None) == date(2026, 10, 6)
    assert resolve_day("mon", monday, None) == monday
    assert resolve_day("tomorrow", monday, None) == date(2026, 10, 6)
    assert resolve_day("1/2", monday, None) == date(2027, 1, 2)
    assert resolve_day(None, monday, date(2026, 10, 9)) == date(2026, 10, 9)


def test_subtracting_a_block_splits_or_trims_the_hours() -> None:
    assert subtract_interval(((540, 1020),), 720, 840) == ((540, 720), (840, 1020))
    assert subtract_interval(((540, 1020),), 480, 720) == ((720, 1020),)
    assert subtract_interval(((540, 1020),), 1080, 1140) == ((540, 1020),)
    assert subtract_interval(((540, 600), (660, 720)), 0, 1440) == ()


@pytest.mark.asyncio
async def test_yes_blocks_the_time_and_declines_the_request_it_answers(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    schedule = ScheduleFake()
    _, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=schedule
    )
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    day = row.requested_slot_start.astimezone(UTC).date()
    application, owner = block_app(session_factory, schedule)

    await say(application, business_id, "book another time, I'm unavailable 8 to 12", "busy-1")

    [question] = texts_of(owner, "Block ")
    assert "8:00 AM–12:00 PM" in question
    assert "Jane Doe" in question
    assert schedule.set_calls == []
    assert (await conversation_row(session_factory, business_id)).state == "awaiting_owner"
    assert await commands_of(session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE) == []

    await say(application, business_id, "yes", "busy-yes")

    assert schedule.set_calls == [(day, ((720, 1020),))]
    [done] = texts_of(owner, "Blocked ")
    assert f"Declined booking {reference}" in done
    assert (await conversation_row(session_factory, business_id)).state == "declined"
    [email] = await commands_of(session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE)
    body = email.payload["body"]
    assert isinstance(body, str)
    assert BLOCK_DECLINE_REASON in body
    assert CALENDLY_LINK in body
    assert schedule.book_calls == []

    await say(application, business_id, f"unblock {day.month}/{day.day}", "unblock-1")

    assert schedule.set_calls[-1] == (day, None)
    assert texts_of(owner, "Unblocked ")


@pytest.mark.asyncio
async def test_no_leaves_the_calendar_and_the_request_alone(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    schedule = ScheduleFake()
    _, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=schedule
    )
    application, owner = block_app(session_factory, schedule)

    await say(application, business_id, "busy 8-12", "busy-1")
    await say(application, business_id, "no", "busy-no")

    [reply] = texts_of(owner, "OK, nothing was blocked.")
    assert f"approve booking {reference}" in reply
    assert schedule.set_calls == []
    assert (await conversation_row(session_factory, business_id)).state == "awaiting_owner"


@pytest.mark.asyncio
async def test_a_block_away_from_the_requested_time_keeps_the_request(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    schedule = ScheduleFake()
    _, _, business_id, _ = await reach_awaiting_owner(session_factory, availability=schedule)
    application, owner = block_app(session_factory, schedule)

    await say(application, business_id, "busy 2-4", "busy-1")
    [question] = texts_of(owner, "Block ")
    assert "Jane Doe" not in question
    await say(application, business_id, "yes", "busy-yes")

    assert len(schedule.set_calls) == 1
    assert (await conversation_row(session_factory, business_id)).state == "awaiting_owner"


@pytest.mark.asyncio
async def test_a_block_covering_only_the_end_of_the_request_still_answers_it(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    schedule = ScheduleFake()
    _, _, business_id, _ = await reach_awaiting_owner(session_factory, availability=schedule)
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    assert row.requested_slot_end is not None
    end = row.requested_slot_end.astimezone(UTC)
    assert end - row.requested_slot_start.astimezone(UTC) > timedelta(minutes=1)
    application, owner = block_app(session_factory, schedule)
    first = (end - timedelta(minutes=1)).strftime("%H:%M")
    last = (end + timedelta(hours=1)).strftime("%H:%M")

    await say(application, business_id, f"busy {first} to {last}", "busy-1")

    [question] = texts_of(owner, "Block ")
    assert "Jane Doe" in question


@pytest.mark.asyncio
async def test_a_retry_after_a_lost_write_restores_the_original_hours(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    schedule = ScheduleFake()
    _, _, business_id, _ = await reach_awaiting_owner(session_factory, availability=schedule)
    row = await conversation_row(session_factory, business_id)
    assert row.requested_slot_start is not None
    day = row.requested_slot_start.astimezone(UTC).date()
    application, owner = block_app(session_factory, schedule)

    await say(application, business_id, "busy 8-12", "busy-1")
    schedule.fail_after_write = True
    await say(application, business_id, "yes", "busy-yes-1")
    assert texts_of(owner, "I couldn't reach the calendar")
    await say(application, business_id, "yes", "busy-yes-2")

    assert schedule.set_calls == [(day, ((720, 1020),)), (day, ((720, 1020),))]
    await say(application, business_id, f"unblock {day.month}/{day.day}", "unblock-1")
    assert schedule.set_calls[-1] == (day, None)


@pytest.mark.asyncio
async def test_an_expired_proposal_is_not_applied(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    schedule = ScheduleFake()
    _, _, business_id, _ = await reach_awaiting_owner(session_factory, availability=schedule)
    application, owner = block_app(session_factory, schedule)

    await say(application, business_id, "busy 8-12", "busy-1")
    async with session_factory() as session:
        await session.execute(
            update(CalendarBlockRecord).values(expires_at=datetime.now(UTC) - timedelta(minutes=1))
        )
        await session.commit()
    await say(application, business_id, "yes", "busy-yes")

    assert texts_of(owner, "That request expired")
    assert schedule.set_calls == []
    assert (await conversation_row(session_factory, business_id)).state == "awaiting_owner"


@pytest.mark.asyncio
async def test_blocking_without_a_schedule_port_says_so(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    _, _, business_id, _ = await reach_awaiting_owner(
        session_factory, availability=AvailabilityFake()
    )
    application, owner = block_app(session_factory, None)

    await say(application, business_id, "busy 8-12", "busy-1")

    assert texts_of(owner, BLOCK_NOT_CONFIGURED_REPLY)


USER_URI = "https://api.calendly.com/users/AAAAAAAAAAAAAAAA"
EVENT_TYPE = "https://api.calendly.com/event_types/ESTIMATE"


def schedule_payload(rules: list[dict[str, object]]) -> dict[str, object]:
    return {
        "collection": [
            {
                "event_type": EVENT_TYPE,
                "availability_setting": "host",
                "availability_rule": {"timezone": "America/Los_Angeles", "rules": rules},
            }
        ]
    }


WEEKLY: list[dict[str, object]] = [
    {"type": "wday", "wday": "tuesday", "intervals": [{"from": "09:00", "to": "17:00"}]},
    {"type": "date", "date": "2030-01-08", "intervals": []},
]


@pytest.mark.asyncio
async def test_calendly_pins_one_date_and_writes_every_other_rule_back() -> None:
    business_id = BusinessId(uuid4())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/event_types":
            return httpx.Response(200, json={"collection": [{"uri": EVENT_TYPE, "duration": 30}]})
        if request.url.path.startswith("/users/"):
            return httpx.Response(200, json={"resource": {"timezone": "America/Los_Angeles"}})
        if request.method == "PATCH":
            return httpx.Response(200, json={"resource": {}})
        return httpx.Response(200, json=schedule_payload(WEEKLY))

    calendly = CalendlyAvailability(
        CalendlySettings(token="not-a-token", installations=f"{business_id}={USER_URI}"),  # noqa: S106
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tuesday = date(2030, 1, 1)
    assert await calendly.schedule_timezone(business_id) == "America/Los_Angeles"
    assert await calendly.day_hours(business_id, tuesday) == WORKDAY
    assert await calendly.day_hours(business_id, date(2030, 1, 8)) == DayHours(
        intervals=(), overridden=True
    )

    await calendly.set_day_hours(business_id, tuesday, ((540, 720), (960, 1440)))

    patch = requests[-1]
    assert patch.method == "PATCH"
    assert patch.url.params["event_type"] == EVENT_TYPE
    body = json.loads(patch.content)
    assert body["availability_rule"]["timezone"] == "America/Los_Angeles"
    assert body["availability_rule"]["rules"] == [
        *WEEKLY,
        {
            "type": "date",
            "date": "2030-01-01",
            "intervals": [{"from": "09:00", "to": "12:00"}, {"from": "16:00", "to": "23:59"}],
        },
    ]

    await calendly.set_day_hours(business_id, date(2030, 1, 8), None)
    assert json.loads(requests[-1].content)["availability_rule"]["rules"] == [WEEKLY[0]]
