"""The owner blocks time by text: ``unavailable 8-12``, ``yes``/``no``, ``unblock tue``.

A request is only a proposal until the owner answers ``yes``. Applying it
narrows that date's hours on the booking calendar, so neither the website chat
nor the business's own booking link offers the time, and declines the booking
request it answered so the customer is sent a link to pick another time.
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, tzinfo
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gvas.domain.calendar_blocks import (
    BLOCK_PROPOSAL_TTL,
    CALENDAR_BLOCK_INTENT,
    CalendarBlock,
    CalendarBlockState,
    UnavailableRequest,
    block_confirmation,
    format_block,
    format_day,
    overlaps,
    resolve_day,
    subtract_interval,
    unavailable_request,
    unblock_request,
)
from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.intake import (
    AvailabilityError,
    BookingDecision,
    BookingDecisionAction,
    IntakeConversation,
    IntakeState,
)
from gvas.domain.messages import NormalizedOwnerMessage, OutboundOwnerMessage, TextPart
from gvas.domain.owner_actions import decide_booking
from gvas.domain.ports import ScheduleBlockPort
from gvas.domain.repositories import UnitOfWork
from gvas.domain.workflows import WorkflowContext, WorkflowResult

logger = logging.getLogger(__name__)

BLOCK_DECLINE_REASON = "That time is no longer open."
BLOCK_HELP_REPLY = (
    "To block time, reply like `unavailable tue 8-12`. To open it again, reply `unblock tue`."
)
BLOCK_NOT_CONFIGURED_REPLY = (
    "Blocking time isn't set up for this business's calendar yet, so nothing was blocked."
)
BLOCK_UNREACHABLE_REPLY = "I couldn't reach the calendar, so nothing changed. Try again shortly."


class CalendarBlockHandler:
    intent = CALENDAR_BLOCK_INTENT

    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        schedule: ScheduleBlockPort | None,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._schedule = schedule
        self._now = now

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        message = context.message
        business_id = message.business_id
        schedule = self._schedule
        if schedule is None or not schedule.serves(business_id):
            return _result(message, BLOCK_NOT_CONFIGURED_REPLY)
        text = "\n".join(part.text for part in message.parts if isinstance(part, TextPart))
        now = self._now()
        try:
            unblock = unblock_request(text)
            request = unavailable_request(text)
            answer = block_confirmation(text)
            if unblock is not None:
                reply = await self._unblock(schedule, message, unblock, now)
            elif request is not None:
                reply = await self._propose(schedule, message, request, now)
            elif answer is True:
                reply = await self._apply(schedule, message, now)
            elif answer is False:
                reply = await self._cancel(message, now)
            else:
                reply = BLOCK_HELP_REPLY
        except AvailabilityError as error:
            logger.warning("calendar block failed for business %s: %s", business_id, error)
            reply = BLOCK_UNREACHABLE_REPLY
        return _result(message, reply)

    async def _propose(
        self,
        schedule: ScheduleBlockPort,
        message: NormalizedOwnerMessage,
        request: UnavailableRequest,
        now: datetime,
    ) -> str:
        business_id = message.business_id
        zone = await _zone(schedule, message)
        today = now.astimezone(zone).date()
        async with self._unit_of_work_factory() as unit_of_work:
            booking = await _booking(unit_of_work, message, request.reference)
            if request.reference and booking is None:
                return f"I can't find booking {request.reference}."
            booked_at = (
                booking.requested_slot_start.astimezone(zone)
                if booking is not None and booking.requested_slot_start is not None
                else None
            )
            day = resolve_day(request.day_hint, today, booked_at.date() if booked_at else None)
            if day is None:
                return "Which day? Reply like `unavailable tue 8-12`."
            if day < today:
                return f"{format_day(day)} has already passed, so nothing was blocked."
            pending = await unit_of_work.calendar_blocks.latest_proposed(business_id)
            if pending is not None:
                await unit_of_work.calendar_blocks.save(
                    pending.with_updates(state=CalendarBlockState.CANCELLED, decided_at=now)
                )
            block = CalendarBlock(
                block_id=uuid4(),
                business_id=business_id,
                day=day,
                start_minute=request.start_minute,
                end_minute=request.end_minute,
                created_at=now,
                expires_at=now + BLOCK_PROPOSAL_TTL,
            )
            answered = (
                booking
                if booking is not None
                and booking.state is IntakeState.AWAITING_OWNER
                and booked_at is not None
                and overlaps(block, booked_at, booking.requested_slot_end)
                else None
            )
            if answered is not None:
                block = block.with_updates(reference=answered.reference)
            await unit_of_work.calendar_blocks.add(block)
            await unit_of_work.commit()
        question = f"Block {format_block(block)}"
        if answered is not None:
            name = answered.collected.name or "the customer"
            question += f" and send {name} a link to pick a new time"
        return f"{question}? Reply yes or no."

    async def _apply(
        self, schedule: ScheduleBlockPort, message: NormalizedOwnerMessage, now: datetime
    ) -> str:
        business_id = message.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            block = await unit_of_work.calendar_blocks.latest_proposed(business_id)
            if block is None:
                return f"Nothing is waiting on a yes. {BLOCK_HELP_REPLY}"
            if not block.is_live(now):
                await unit_of_work.calendar_blocks.save(
                    block.with_updates(state=CalendarBlockState.CANCELLED, decided_at=now)
                )
                await unit_of_work.commit()
                return (
                    "That request expired, so nothing was blocked. "
                    "Send it again if you still need the time."
                )
            hours = block.previous
            if hours is None:
                # Recorded before Calendly is touched: a retry after a write
                # that landed but never committed reuses the original hours
                # instead of reading back the already-narrowed ones.
                hours = await schedule.day_hours(business_id, block.day)
                block = block.with_updates(previous=hours)
                await unit_of_work.calendar_blocks.save(block)
                await unit_of_work.commit()
            remaining = subtract_interval(hours.intervals, block.start_minute, block.end_minute)
            await schedule.set_day_hours(business_id, block.day, remaining)
            await unit_of_work.calendar_blocks.save(
                block.with_updates(state=CalendarBlockState.APPLIED, decided_at=now)
            )
            reply = f"Blocked {format_block(block)}, so no one can book it."
            if block.reference is not None:
                outcome = await decide_booking(
                    unit_of_work,
                    business_id,
                    BookingDecision(
                        action=BookingDecisionAction.DECLINE,
                        reference=block.reference,
                        reason=BLOCK_DECLINE_REASON,
                    ),
                    now,
                )
                reply += f" {outcome.text}"
            await unit_of_work.commit()
        return f"{reply} Reply `unblock {_day_key(block.day)}` to undo."

    async def _cancel(self, message: NormalizedOwnerMessage, now: datetime) -> str:
        async with self._unit_of_work_factory() as unit_of_work:
            block = await unit_of_work.calendar_blocks.latest_proposed(message.business_id)
            if block is None:
                return "Nothing is waiting on a yes or no."
            await unit_of_work.calendar_blocks.save(
                block.with_updates(state=CalendarBlockState.CANCELLED, decided_at=now)
            )
            await unit_of_work.commit()
        reply = "OK, nothing was blocked."
        if block.reference is not None:
            ref = block.reference
            reply += (
                f" Booking {ref} is still waiting: reply `approve booking {ref}` "
                f"or `decline booking {ref} <reason>`."
            )
        return reply

    async def _unblock(
        self,
        schedule: ScheduleBlockPort,
        message: NormalizedOwnerMessage,
        hint: str,
        now: datetime,
    ) -> str:
        business_id = message.business_id
        zone = await _zone(schedule, message)
        day = resolve_day(hint or None, now.astimezone(zone).date(), None)
        if day is None:
            return "Which day? Reply like `unblock tue`."
        async with self._unit_of_work_factory() as unit_of_work:
            blocks = await unit_of_work.calendar_blocks.applied_on(business_id, day)
            if not blocks:
                return f"Nothing is blocked on {format_day(day)}."
            previous = blocks[0].previous
            await schedule.set_day_hours(
                business_id,
                day,
                previous.intervals if previous is not None and previous.overridden else None,
            )
            for block in blocks:
                await unit_of_work.calendar_blocks.save(
                    block.with_updates(state=CalendarBlockState.REMOVED, decided_at=now)
                )
            await unit_of_work.commit()
        return f"Unblocked {format_day(day)}. Your usual hours are open again."


async def _booking(
    unit_of_work: UnitOfWork, message: NormalizedOwnerMessage, reference: str | None
) -> IntakeConversation | None:
    """The named request, else the one request waiting on the owner (if only one)."""

    if reference is not None:
        return await unit_of_work.intake_conversations.find_by_reference(
            message.business_id, reference
        )
    waiting = await unit_of_work.intake_conversations.list_awaiting_owner(
        message.business_id, limit=2
    )
    return waiting[0] if len(waiting) == 1 else None


async def _zone(schedule: ScheduleBlockPort, message: NormalizedOwnerMessage) -> tzinfo:
    name = await schedule.schedule_timezone(message.business_id)
    if not name:
        return UTC
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return UTC


def _day_key(day: date) -> str:
    return f"{day.month}/{day.day}"


def _result(message: NormalizedOwnerMessage, text: str) -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowRunStatus.SUCCEEDED,
        replies=(
            OutboundOwnerMessage(
                business_id=message.business_id,
                conversation_ref=message.conversation_ref,
                parts=(TextPart(text=text),),
                correlation_id=f"calendar_block:{message.message_key}",
            ),
        ),
    )
