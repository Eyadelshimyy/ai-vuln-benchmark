"""Durable storage for bounty campaigns (Phase 51 hardening).

Campaign state lives in the application database:

* the orchestrator's typed state and immutable decision history go through the
  existing Phase 49 :class:`~app.discovery.orchestration.store.SqlStore` (tables
  from migration ``027``);
* the operator inputs needed to rebuild a campaign after a restart (manifest,
  spec, operator, approvals) plus control state, background-job progress, and
  campaign artifacts (source selection, VFCS feedback ledger, stored compiler
  differential) live in ``discovery_bounty_campaigns`` (migration ``028``).

The application database is async (asyncpg / aiosqlite) while the orchestrator
and ``SqlStore`` are synchronous. :class:`AsyncBridgeRunner` runs the synchronous
ORM work on a dedicated event-loop thread through ``AsyncSession.run_sync``, so
the same ``SqlStore`` code serves both. There is no second evidence database and
no in-memory production fallback: when the database is unreachable the campaign
API reports that persistence is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, cast

from sqlalchemy import create_engine, inspect, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.discovery.orchestration.model import (
    DecisionRecord,
    PersistenceError,
    ResearchState,
)
from app.discovery.orchestration.store import SqlStore

T = TypeVar("T")
RUN_TIMEOUT_SECONDS = 60.0
MAX_RECORD_TEXT = 2_000_000
_TABLES = (
    "discovery_orchestration_campaigns",
    "discovery_orchestration_decisions",
    "discovery_bounty_campaigns",
)


class CampaignPersistenceError(PersistenceError):
    """Campaign storage is unreachable, misconfigured, or not migrated."""


class CampaignConflictError(PersistenceError):
    """Another writer changed the campaign since it was loaded."""


class SessionRunner(Protocol):
    def run(self, fn: Callable[[Session], T]) -> T: ...

    def describe(self) -> str: ...

    def dialect(self) -> str: ...


class SyncSessionRunner:
    """A synchronous engine (``sqlite:///...``, ``postgresql+psycopg://...``)."""

    def __init__(self, target: str | Engine) -> None:
        if isinstance(target, str):
            kwargs: dict[str, Any] = {}
            if target.startswith("sqlite"):
                # The runner is shared across worker threads; sqlite needs these.
                kwargs["connect_args"] = {"check_same_thread": False}
                if ":memory:" in target or "mode=memory" in target:
                    kwargs["poolclass"] = StaticPool
            self._engine = create_engine(target, **kwargs)
        else:
            self._engine = target

    def run(self, fn: Callable[[Session], T]) -> T:
        with Session(self._engine, expire_on_commit=False) as session:
            return fn(session)

    def describe(self) -> str:
        return f"sync:{self._engine.dialect.name}"

    def dialect(self) -> str:
        return str(self._engine.dialect.name)

    def close(self) -> None:
        self._engine.dispose()


class AsyncBridgeRunner:
    """Run synchronous ORM work against an async database from any thread.

    A dedicated daemon thread owns an event loop and an async engine. ``run``
    submits ``AsyncSession.run_sync(fn)`` to that loop and waits for it, so the
    caller may be a worker thread or a FastAPI threadpool thread. It must not be
    called from the bridge's own loop (that would deadlock); nothing does.
    """

    def __init__(self, url: str, *, engine: AsyncEngine | None = None) -> None:
        self._url = url
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="bugforge-campaign-db", daemon=True
        )
        self._thread.start()
        self._engine = engine or create_async_engine(url, pool_pre_ping=True)
        self._factory = async_sessionmaker(self._engine, expire_on_commit=False)

    def _submit(self, coro: Coroutine[Any, Any, T]) -> T:
        if threading.current_thread() is self._thread:
            raise CampaignPersistenceError("the campaign database bridge cannot call itself")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=RUN_TIMEOUT_SECONDS)

    def run(self, fn: Callable[[Session], T]) -> T:
        async def go() -> T:
            async with self._factory() as session:
                return await session.run_sync(fn)

        return self._submit(go())

    def describe(self) -> str:
        return f"async-bridge:{self._engine.dialect.name}"

    def dialect(self) -> str:
        return str(self._engine.dialect.name)

    def close(self) -> None:
        async def dispose() -> None:
            await self._engine.dispose()

        try:
            self._submit(dispose())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


def runner_for_url(url: str) -> SessionRunner:
    """Pick the runner for a database URL. Async drivers go through the bridge."""
    if not url:
        raise CampaignPersistenceError("no campaign database URL is configured")
    if "+asyncpg" in url or "+aiosqlite" in url or "+asyncmy" in url or "+aiomysql" in url:
        return AsyncBridgeRunner(url)
    return SyncSessionRunner(url)


def runner_from_settings() -> SessionRunner:
    from app.core.config import get_settings

    settings = get_settings()
    return runner_for_url(settings.bounty_campaign_database_url or settings.database_url)


def ensure_schema(runner: SessionRunner) -> None:
    """Verify the campaign tables exist.

    SQLite (development and tests) creates them on first use. Any other database
    must be migrated with ``alembic upgrade head``; a missing table is reported,
    never silently created next to the migrations.
    """
    from app.models.base import Base
    from app.models.discovery_orchestration import (  # noqa: F401  (register tables)
        DBBountyCampaign,
        DBOrchestrationCampaign,
        DBOrchestrationDecision,
    )

    def check(session: Session) -> list[str]:
        bind = session.connection()
        present = set(inspect(bind).get_table_names())
        missing = [name for name in _TABLES if name not in present]
        if missing and bind.dialect.name == "sqlite":
            tables = [Base.metadata.tables[name] for name in _TABLES]
            Base.metadata.create_all(bind, tables=tables, checkfirst=True)
            session.commit()
            return []
        return missing

    try:
        missing = runner.run(check)
    except (SQLAlchemyError, OSError, TimeoutError) as exc:
        raise CampaignPersistenceError(
            f"campaign database is unavailable ({type(exc).__name__})"
        ) from exc
    if missing:
        raise CampaignPersistenceError(
            "campaign tables are missing (" + ", ".join(missing) + "); run `alembic upgrade head`"
        )


class RunnerSqlStore:
    """The Phase 49 ``SqlStore`` run through a :class:`SessionRunner`.

    The optimistic revision map is kept across calls, so a writer that did not
    load the latest revision is refused exactly as ``SqlStore`` refuses it.
    """

    def __init__(self, runner: SessionRunner) -> None:
        self._runner = runner
        self._revisions: dict[str, int] = {}

    def _store(self, session: Session) -> SqlStore:
        store = SqlStore(session)
        store._revisions = self._revisions
        return store

    def _run(self, fn: Callable[[Session], T]) -> T:
        try:
            return self._runner.run(fn)
        except PersistenceError as exc:
            if "changed since it was loaded" in str(exc) or "was not loaded" in str(exc):
                raise CampaignConflictError(str(exc)) from exc
            raise
        except (SQLAlchemyError, OSError, TimeoutError) as exc:
            raise CampaignPersistenceError(
                f"campaign database is unavailable ({type(exc).__name__})"
            ) from exc

    def save(self, state: ResearchState) -> None:
        self._run(lambda session: self._store(session).save(state))

    def load(self, campaign_id: str) -> ResearchState | None:
        return self._run(lambda session: self._store(session).load(campaign_id))

    def decisions(self, campaign_id: str) -> tuple[DecisionRecord, ...]:
        return self._run(lambda session: self._store(session).decisions(campaign_id))

    def forget(self, campaign_id: str) -> None:
        self._revisions.pop(campaign_id, None)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class CampaignRecord:
    """Everything needed to rebuild a campaign, plus campaign-level state."""

    campaign_id: str
    operator_identity: str
    engine_version: str
    program_context: str
    manifest: dict[str, Any]
    spec: dict[str, Any]
    approvals: list[dict[str, str]] = field(default_factory=list)
    control: str = "active"  # active | paused | stopped
    control_reason: str = ""
    job: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    revision: int = 0

    def copy(self) -> CampaignRecord:
        return replace(
            self,
            manifest=json.loads(json.dumps(self.manifest)),
            spec=json.loads(json.dumps(self.spec)),
            approvals=[dict(item) for item in self.approvals],
            job=json.loads(json.dumps(self.job)),
            artifacts=json.loads(json.dumps(self.artifacts)),
        )


def _text(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(text) > MAX_RECORD_TEXT:
        raise PersistenceError("campaign record exceeds the stored document limit")
    return text


def _parse(text: str, kind: type) -> Any:
    try:
        value = json.loads(text or ("[]" if kind is list else "{}"))
    except ValueError as exc:
        raise PersistenceError("campaign record is corrupt") from exc
    if not isinstance(value, kind):
        raise PersistenceError("campaign record has the wrong shape")
    return value


class CampaignRecordStore:
    """Optimistic-concurrency CRUD over ``discovery_bounty_campaigns``."""

    def __init__(self, runner: SessionRunner) -> None:
        self._runner = runner

    def _run(self, fn: Callable[[Session], T]) -> T:
        try:
            return self._runner.run(fn)
        except PersistenceError:
            raise
        except (SQLAlchemyError, OSError, TimeoutError) as exc:
            raise CampaignPersistenceError(
                f"campaign database is unavailable ({type(exc).__name__})"
            ) from exc

    def insert(self, record: CampaignRecord) -> CampaignRecord:
        from app.models.discovery_orchestration import DBBountyCampaign

        def go(session: Session) -> CampaignRecord:
            if session.get(DBBountyCampaign, record.campaign_id) is not None:
                raise CampaignConflictError("campaign record already exists")
            now = _now()
            session.add(
                DBBountyCampaign(
                    campaign_id=record.campaign_id,
                    operator_identity=record.operator_identity[:128],
                    engine_version=record.engine_version[:32],
                    program_context=record.program_context[:64],
                    manifest=_text(record.manifest),
                    spec=_text(record.spec),
                    approvals=_text(record.approvals),
                    control=record.control,
                    control_reason=record.control_reason[:200],
                    job=_text(record.job),
                    artifacts=_text(record.artifacts),
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
            return replace(record, revision=1)

        return self._run(go)

    def update(self, record: CampaignRecord) -> CampaignRecord:
        """Write the record if nobody else wrote it since ``record.revision``."""
        from app.models.discovery_orchestration import DBBountyCampaign

        def go(session: Session) -> CampaignRecord:
            table = cast("Any", DBBountyCampaign.__table__)
            result = session.execute(
                update(table)
                .where(
                    table.c.campaign_id == record.campaign_id,
                    table.c.revision == record.revision,
                )
                .values(
                    approvals=_text(record.approvals),
                    control=record.control,
                    control_reason=record.control_reason[:200],
                    job=_text(record.job),
                    artifacts=_text(record.artifacts),
                    revision=record.revision + 1,
                    updated_at=_now(),
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                session.rollback()
                raise CampaignConflictError("campaign record changed since it was loaded")
            session.commit()
            return replace(record, revision=record.revision + 1)

        return self._run(go)

    def get(self, campaign_id: str) -> CampaignRecord | None:
        from app.models.discovery_orchestration import DBBountyCampaign

        def go(session: Session) -> CampaignRecord | None:
            row = session.get(DBBountyCampaign, campaign_id)
            if row is None:
                return None
            return CampaignRecord(
                campaign_id=row.campaign_id,
                operator_identity=row.operator_identity,
                engine_version=row.engine_version,
                program_context=row.program_context,
                manifest=_parse(row.manifest, dict),
                spec=_parse(row.spec, dict),
                approvals=_parse(row.approvals, list),
                control=row.control,
                control_reason=row.control_reason,
                job=_parse(row.job, dict),
                artifacts=_parse(row.artifacts, dict),
                revision=row.revision,
            )

        return self._run(go)

    def list_ids(self, operator_identity: str | None = None) -> list[str]:
        from app.models.discovery_orchestration import DBBountyCampaign

        def go(session: Session) -> list[str]:
            query = select(DBBountyCampaign.campaign_id, DBBountyCampaign.operator_identity)
            rows = session.execute(query).all()
            return sorted(
                str(cid)
                for cid, owner in rows
                if operator_identity is None or not owner or owner == operator_identity
            )

        return self._run(go)


__all__ = [
    "AsyncBridgeRunner",
    "CampaignConflictError",
    "CampaignPersistenceError",
    "CampaignRecord",
    "CampaignRecordStore",
    "RunnerSqlStore",
    "SessionRunner",
    "SyncSessionRunner",
    "ensure_schema",
    "runner_for_url",
    "runner_from_settings",
]
