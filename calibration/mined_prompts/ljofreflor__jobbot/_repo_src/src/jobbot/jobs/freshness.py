"""How old a posting is, and whether that is still worth applying to."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from babel.dates import format_timedelta

DEFAULT_MAX_AGE_DAYS = 30


def _aware(moment: datetime) -> datetime:
    """Legacy rows and fixtures may carry naive datetimes."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def age_in_days(posted_at: datetime | None, *, now: datetime | None = None) -> int | None:
    if posted_at is None:
        return None
    reference = _aware(now or datetime.now(UTC))
    return max((reference - _aware(posted_at)).days, 0)


def is_fresh(
    posted_at: datetime | None,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: datetime | None = None,
) -> bool:
    """
    True when the posting is recent enough to bother applying.

    Postings without a known date are kept: JobBot does not hide work it failed
    to date. `max_age_days <= 0` disables the filter.
    """
    if max_age_days <= 0:
        return True
    days = age_in_days(posted_at, now=now)
    if days is None:
        return True
    return days <= max_age_days


def age_label(posted_at: datetime | None, *, now: datetime | None = None) -> str:
    """Age for the sweep table; units and plurals come from CLDR, not from us."""
    days = age_in_days(posted_at, now=now)
    if days is None:
        return "?"
    if days == 0:
        return "hoy"
    # threshold=1.0: never round 10 months up to "1 año"; the age must not flatter a post.
    return format_timedelta(
        timedelta(days=days),
        locale="es",
        granularity="day",
        threshold=1.0,
    )
