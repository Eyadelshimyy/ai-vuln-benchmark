"""The owner's own calendar, read through its private iCal (.ics) link."""

from gvas.infrastructure.calendar.feed import IcsCalendarFeed, parse_calendar_feed

__all__ = ["IcsCalendarFeed", "parse_calendar_feed"]
