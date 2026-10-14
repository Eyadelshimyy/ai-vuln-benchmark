"""Owner-requested calendar blocks: "busy then, I'm unavailable 8 to 12".

Gus never blocks time on a parsed text alone. The request is stored as a
proposal, read back to the owner, and applied only after an explicit ``yes``.
Times are minutes after midnight on the booking calendar's own wall clock.
"""

import re
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.identifiers import BusinessId, WorkflowIntent

CALENDAR_BLOCK_INTENT = WorkflowIntent("calendar_block")
BLOCK_PROPOSAL_TTL = timedelta(minutes=30)
MINUTES_PER_DAY = 24 * 60

Interval = tuple[int, int]

_WEEKDAYS = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}
_TIME = r"(\d{1,2})(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?|a|p)?"
_RANGE = re.compile(
    rf"(?<![\d$:.]){_TIME}\s*(?:-|–|—|to|until|till|til|thru|through)\s*{_TIME}"
    r"(?![\d%:])(?!\s*(?:ft|feet|foot|in\b|inch|gal|lb|sq|yd|x\b|\"|'))",
    re.IGNORECASE,
)
_UNAVAILABLE = re.compile(
    r"\b(?:unavailable|not\s+available|busy|block(?:\s+off)?|"
    r"(?:i'?m|i\s+am|im)\s+(?:off|out|away|booked))\b",
    re.IGNORECASE,
)
_UNBLOCK = re.compile(r"^\s*unblock\b(?P<rest>.*)$", re.IGNORECASE | re.DOTALL)
_REFERENCE = re.compile(r"\bbooking\s+#?(?P<reference>[0-9a-z]{4,32})\b", re.IGNORECASE)
_DAY = re.compile(
    r"\b(?P<word>today|tomorrow|" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b"
    r"|\b(?P<month>1[0-2]|0?[1-9])/(?P<day>3[01]|[12]\d|0?[1-9])\b",
    re.IGNORECASE,
)
_YES = frozenset({"yes", "y", "yep", "yeah", "yup", "ok", "okay", "confirm", "do it", "sure"})
_NO = frozenset({"no", "n", "nope", "cancel", "dont", "don't", "never mind", "nevermind"})


class CalendarBlockError(ValueError):
    """The owner's request could not be read as one block of time."""


class CalendarBlockState(StrEnum):
    PROPOSED = "proposed"
    APPLIED = "applied"
    CANCELLED = "cancelled"
    REMOVED = "removed"


class DayHours(BaseModel):
    """Bookable hours of one date, and whether they come from a date rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervals: tuple[Interval, ...]
    overridden: bool


class CalendarBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: UUID
    business_id: BusinessId
    day: date
    start_minute: int = Field(ge=0, lt=MINUTES_PER_DAY)
    end_minute: int = Field(gt=0, le=MINUTES_PER_DAY)
    # The booking request this block answers; declined once the block lands.
    reference: str | None = None
    state: CalendarBlockState = CalendarBlockState.PROPOSED
    # The date's hours before the block, so ``unblock`` restores them exactly.
    previous: DayHours | None = None
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "CalendarBlock":
        if self.end_minute <= self.start_minute:
            raise ValueError("a block must end after it starts")
        return self

    def is_live(self, now: datetime) -> bool:
        return self.state is CalendarBlockState.PROPOSED and now < self.expires_at

    def with_updates(self, **changes: object) -> "CalendarBlock":
        return self.model_copy(update=changes)


class UnavailableRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_minute: int
    end_minute: int
    day_hint: str | None = None
    reference: str | None = None


def unavailable_request(text: str) -> UnavailableRequest | None:
    """``book another time, I'm unavailable 8 to 12`` -> 8:00-12:00.

    Needs an availability phrase and one time range; prices and percentages
    never read as times. Bare hours follow the working day: 1-6 are PM.
    """

    if "$" in text or "%" in text or _UNAVAILABLE.search(text) is None:
        return None
    ranges = list(_RANGE.finditer(text))
    if len(ranges) != 1:
        return None
    try:
        start, end = _range_minutes(ranges[0])
    except CalendarBlockError:
        return None
    reference = _REFERENCE.search(text)
    return UnavailableRequest(
        start_minute=start,
        end_minute=end,
        day_hint=day_hint(text[: ranges[0].start()] + " " + text[ranges[0].end() :]),
        reference=reference.group("reference").lower() if reference else None,
    )


def unblock_request(text: str) -> str | None:
    """``unblock tue`` -> the day hint (``""`` when no day was named)."""

    match = _UNBLOCK.match(text)
    if match is None:
        return None
    return day_hint(match.group("rest")) or ""


def block_confirmation(text: str) -> bool | None:
    cleaned = " ".join(re.sub(r"[^\w\s']", " ", text.casefold()).split())
    if cleaned in _YES:
        return True
    if cleaned in _NO:
        return False
    return None


def day_hint(text: str) -> str | None:
    match = _DAY.search(text)
    if match is None:
        return None
    if match.group("word"):
        return match.group("word").lower()
    return f"{int(match.group('month'))}/{int(match.group('day'))}"


def resolve_day(hint: str | None, today: date, fallback: date | None) -> date | None:
    """The named day on or after ``today``; no name falls back (the booking's day)."""

    if not hint:
        return fallback
    if hint == "today":
        return today
    if hint == "tomorrow":
        return today + timedelta(days=1)
    if hint in _WEEKDAYS:
        return today + timedelta(days=(_WEEKDAYS[hint] - today.weekday()) % 7)
    month, day = (int(part) for part in hint.split("/"))
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


def subtract_interval(
    intervals: tuple[Interval, ...], start: int, end: int
) -> tuple[Interval, ...]:
    remaining: list[Interval] = []
    for low, high in intervals:
        if high <= start or low >= end:
            remaining.append((low, high))
            continue
        if low < start:
            remaining.append((low, start))
        if high > end:
            remaining.append((end, high))
    return tuple(remaining)


def overlaps(block: CalendarBlock, start: datetime, end: datetime | None) -> bool:
    """Whether a booking from ``start`` to ``end`` (calendar wall clock) touches ``block``."""

    if start.date() != block.day:
        return False
    first = start.hour * 60 + start.minute
    last = first if end is None else first + max(int((end - start).total_seconds() // 60), 0)
    return first < block.end_minute and max(last, first + 1) > block.start_minute


def format_minute(minute: int) -> str:
    hour, rest = divmod(minute % MINUTES_PER_DAY, 60)
    meridiem = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{rest:02d} {meridiem}"


def format_block(block: CalendarBlock) -> str:
    """``Tue Oct 6, 8:00 AM-12:00 PM``."""

    end = "midnight" if block.end_minute == MINUTES_PER_DAY else format_minute(block.end_minute)
    return f"{format_day(block.day)}, {format_minute(block.start_minute)}–{end}"


def format_day(day: date) -> str:
    return f"{day:%a} {day:%b} {day.day}"


def _range_minutes(match: re.Match[str]) -> tuple[int, int]:
    start_hour, start_min, start_mer, end_hour, end_min, end_mer = match.groups()
    end_meridiem = _meridiem(end_mer)
    start_meridiem = _meridiem(start_mer)
    start = _to_minutes(int(start_hour), int(start_min or 0), start_meridiem)
    end = _to_minutes(int(end_hour), int(end_min or 0), end_meridiem)
    if start_meridiem is None and end_meridiem == "pm" and start > end:
        start -= 12 * 60
    if end_meridiem is None and end <= start and end < 12 * 60:
        end += 12 * 60
    if end_meridiem == "am" and end == 0:
        end = MINUTES_PER_DAY
    if not 0 <= start < end <= MINUTES_PER_DAY:
        raise CalendarBlockError("the range must end after it starts")
    return start, end


def _meridiem(value: str | None) -> str | None:
    if not value:
        return None
    return "am" if value.lower().startswith("a") else "pm"


def _to_minutes(hour: int, minute: int, meridiem: str | None) -> int:
    if not 0 <= hour <= 24:
        raise CalendarBlockError("not an hour")
    if meridiem is not None:
        if not 1 <= hour <= 12:
            raise CalendarBlockError("not a 12-hour time")
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif 1 <= hour <= 6:
        hour += 12
    return hour * 60 + minute


class CalendarBlockRepository(Protocol):
    async def add(self, block: CalendarBlock) -> None: ...

    async def save(self, block: CalendarBlock) -> None: ...

    async def latest_proposed(self, business_id: BusinessId) -> CalendarBlock | None:
        """The newest block still waiting on the owner's yes or no."""
        ...

    async def applied_on(self, business_id: BusinessId, day: date) -> tuple[CalendarBlock, ...]:
        """Blocks in force on ``day``, oldest first."""
        ...
