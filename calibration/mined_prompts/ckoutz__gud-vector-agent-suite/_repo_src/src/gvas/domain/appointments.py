"""Appointment lookup: the provider-neutral view of the owner's calendar.

The quote workflow asks this port who the owner is seeing around today so a
``quote:`` without ``customer:`` can be addressed to that person. The port
returns plain records; which calendar answers, and how, is an adapter concern.
"""

from datetime import UTC, datetime, time, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.identifiers import BusinessId


class AppointmentModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Appointment(AppointmentModel):
    appointment_id: str = Field(min_length=1)
    start_time: datetime
    invitee_name: str = Field(min_length=1)
    invitee_email: str = Field(min_length=1)
    # E.164 when the booking captured one.
    invitee_phone: str | None = None
    event_name: str = Field(min_length=1)
    address: str | None = None
    # What the invitee wrote when booking, as "question: answer" strings.
    notes: tuple[str, ...] = Field(default_factory=tuple)
    # Owner-facing name of the calendar the record came from, e.g. the
    # provider's product name; the adapter supplies it so no provider is named
    # in this layer.
    source_label: str = Field(min_length=1)

    @field_validator("start_time")
    @classmethod
    def start_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("appointment start time must be timezone-aware")
        return value

    @property
    def display_time(self) -> str:
        """``Tue 2:00pm`` in the start time's own zone (the invitee's when known)."""

        hour = self.start_time.hour % 12 or 12
        meridiem = "am" if self.start_time.hour < 12 else "pm"
        return f"{self.start_time:%a} {hour}:{self.start_time:%M}{meridiem}"

    @property
    def summary(self) -> str:
        parts = [self.source_label, self.display_time]
        if self.address:
            parts.append(self.address)
        return ", ".join(parts)

    @property
    def choice_label(self) -> str:
        """What the owner picks from when several appointments match."""

        return self.address or f"{self.invitee_name}, {self.display_time}"


class AppointmentWindow(AppointmentModel):
    business_id: BusinessId
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def bounds_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("appointment window bounds must be timezone-aware")
        return value


def surrounding_days_window(business_id: BusinessId, now: datetime) -> AppointmentWindow:
    """Yesterday, today and tomorrow as whole days.

    Businesses carry no timezone yet, so the day boundary is UTC: the window
    runs from 00:00 UTC of the previous day to 00:00 UTC two days ahead.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("window anchor must be timezone-aware")
    today = now.astimezone(UTC).date()
    start = datetime.combine(today - timedelta(days=1), time.min, tzinfo=UTC)
    return AppointmentWindow(business_id=business_id, start=start, end=start + timedelta(days=3))


class AppointmentLookupError(RuntimeError):
    """The calendar could not be consulted. The message is sanitized by the
    adapter: no credentials or raw provider responses."""


class AppointmentLookupPort(Protocol):
    async def find(self, window: AppointmentWindow) -> tuple[Appointment, ...]: ...
