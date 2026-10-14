"""Program-data freshness checks. Live submission requires current scope."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.adapters.hackerone.errors import HackerOneError
from app.adapters.hackerone.models import HackerOneProgram, ProgramSyncStatus
from app.core.config import Settings, get_settings


def program_is_fresh(
    program: HackerOneProgram,
    *,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> tuple[bool, str]:
    cfg = settings or get_settings()
    clock = now or datetime.now(UTC)
    if program.sync_status is not ProgramSyncStatus.OK:
        return False, f"Program sync status is {program.sync_status.value}"
    if not program.scope_sync_complete:
        return False, "Structured scope snapshot is incomplete"
    if not program.structured_scopes:
        return False, "Program has no structured scope"
    if program.fetched_at is None:
        return False, "Program has never been synchronized"
    max_age = timedelta(hours=max(1, cfg.hackerone_program_max_age_hours))
    fetched_at = _as_utc(program.fetched_at)
    if clock - fetched_at > max_age:
        return False, "Program scope snapshot is stale; re-sync before live submission"
    weakness_age = timedelta(hours=max(1, cfg.hackerone_weakness_max_age_hours))
    weakness_at = _as_utc(program.weaknesses_synced_at or program.fetched_at)
    if clock - weakness_at > weakness_age:
        return False, "Program weakness list is stale; re-sync before live submission"
    if program.requires_severity is None:
        return False, "Required program configuration is unknown"
    return True, "Program data is current"


def require_fresh_program(program: HackerOneProgram, *, settings: Settings | None = None) -> None:
    ok, reason = program_is_fresh(program, settings=settings)
    if not ok:
        raise HackerOneError(reason, code="stale_program")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
