"""Provider-neutral usage accounting for metered model calls.

Usage is counted in provider units, never money: audio seconds for
transcription and input plus output tokens for review-model calls. Totals are
kept per business per calendar month (UTC), which is the granularity the
ceilings are set at. Recording is at-least-once like the commands that cause
it, so a replayed command may count twice; over-counting only makes a ceiling
bite earlier, which is the safe side.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Protocol

from gvas.domain.identifiers import BusinessId


class UsageKind(StrEnum):
    TRANSCRIPTION_AUDIO_SECONDS = "transcription_audio_seconds"
    REVIEW_TOKENS = "review_tokens"


def usage_month(at: datetime) -> date:
    """First day of the UTC calendar month a usage event belongs to."""

    if at.tzinfo is None:
        raise ValueError("usage timestamps must be timezone-aware")
    utc = at.astimezone(UTC)
    return date(utc.year, utc.month, 1)


@dataclass(frozen=True)
class UsageCeilings:
    """Monthly limits per business in provider units; 0 means unlimited."""

    transcription_seconds: int = 0
    review_tokens: int = 0

    def __post_init__(self) -> None:
        if self.transcription_seconds < 0 or self.review_tokens < 0:
            raise ValueError("usage ceilings must not be negative")

    def limit(self, kind: UsageKind) -> int:
        if kind is UsageKind.TRANSCRIPTION_AUDIO_SECONDS:
            return self.transcription_seconds
        return self.review_tokens


class UsageLedgerPort(Protocol):
    async def record(
        self, business_id: BusinessId, kind: UsageKind, units: int, *, at: datetime
    ) -> None: ...

    async def total(self, business_id: BusinessId, kind: UsageKind, *, month: date) -> int: ...


class UsageCeilingGuard:
    """Decides whether a metered call may still be made this month.

    The units a call will consume are only known from the provider response, so
    a ceiling counts as reached once the month's recorded total has met it: the
    call that crosses the line is allowed, the next one is not. Without a ledger
    or with a 0 ceiling nothing is ever held back.
    """

    def __init__(
        self, ledger: UsageLedgerPort | None = None, ceilings: UsageCeilings | None = None
    ) -> None:
        self._ledger = ledger
        self._ceilings = ceilings or UsageCeilings()

    async def is_reached(self, business_id: BusinessId, kind: UsageKind, *, now: datetime) -> bool:
        limit = self._ceilings.limit(kind)
        if limit == 0 or self._ledger is None:
            return False
        used = await self._ledger.total(business_id, kind, month=usage_month(now))
        return used >= limit
