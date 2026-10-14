"""The state store: one ``tracebi_runs`` table for every kind of run.

Pipeline layers, report builds, schedule ticks and background report runs
share this table. ``TRACEBI_STATE_URL`` selects the database (default
``data/tracebi.db`` beside the working directory). A pipeline runner that
was given its own URL keeps that database; the table shape is the same,
and ``upgrade`` is what both call.

HTML stays on disk. The row holds ``output_path``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import zlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional, Union

#: Which extras key a missing-store error names. Schedule and the runner
#: say ``pipeline``. Web routes set ``web`` for the call.
_install_extra: ContextVar[str] = ContextVar(
    "tracebi_state_install_extra", default="pipeline")

#: Namespace for advisory lock keys, so TraceBi's locks cannot collide with
#: an application's own use of pg_advisory_lock in the same database.
ADVISORY_NAMESPACE = 0x54424900  # "TBI\0"

_RUN_COLUMNS = (
    "layer_name", "layer_type", "started_at", "completed_at", "status",
    "rows_in", "rows_out", "upstream_run_id", "error_message",
    "actor", "actor_role", "kind", "target", "finished",
    "output_path", "verdict", "detail",
)
_UPDATABLE = {
    "completed_at", "finished", "status", "rows_in", "rows_out",
    "error_message", "output_path", "verdict", "detail",
}
_UNSET = object()
_engines: dict[str, Any] = {}
_ready: set[str] = set()
_guard = threading.Lock()
_migrations = Path(__file__).resolve().parent / "migrations"


def state_url() -> str:
    """The shared store. An explicit URL wins; otherwise a file under ``data/``."""
    url = os.environ.get("TRACEBI_STATE_URL", "").strip()
    if url:
        return url
    path = (Path.cwd() / "data" / "tracebi.db").resolve()
    return "sqlite:///" + path.as_posix()


def _sqlite_file(url: str) -> Optional[str]:
    if not url.startswith("sqlite:///"):
        return None
    path = url[len("sqlite:///"):]
    if not path or path.startswith(":memory:"):
        return None
    return path


@contextmanager
def install_extra(extra: str):
    """Name *extra* in the store's missing-dependency error for this block."""
    token = _install_extra.set(extra)
    try:
        yield
    finally:
        _install_extra.reset(token)


def _missing(dep: str) -> ImportError:
    return ImportError(
        f"The state store needs {dep}. "
        f"Install with: pip install 'tracebi[{_install_extra.get()}]'"
    )


def _engine(url: str):
    eng = _engines.get(url)
    if eng is not None:
        return eng
    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise _missing("SQLAlchemy") from None
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
    file_path = _sqlite_file(url)
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, **kwargs)
    _engines[url] = eng
    return eng


def upgrade(db_url: str, engine=None) -> None:
    """Apply Alembic migrations. Pass *engine* when the caller already has one
    (a ``:memory:`` database lives on that engine's connection)."""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        raise _missing("Alembic") from None
    file_path = _sqlite_file(db_url)
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    cfg = Config()
    cfg.set_main_option("script_location", str(_migrations))
    cfg.set_main_option("sqlalchemy.url", db_url)
    if engine is None:
        command.upgrade(cfg, "head")
        return
    with engine.connect() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")


def ensure(url: Optional[str] = None):
    """The engine for *url*, migrated."""
    url = url or state_url()
    eng = _engine(url)
    with _guard:
        if url not in _ready:
            upgrade(url, engine=eng)
            _ready.add(url)
    return eng


def advisory_key(name: str) -> int:
    """Stable key for *name*, safe for ``pg_try_advisory_lock``'s bigint.

    Shift by 32, not 31: crc32 fills a full 32 bits, so a narrower shift
    lets a checksum with its top bit set bleed into the namespace and two
    different names could share a key.
    """
    return (ADVISORY_NAMESPACE << 32) | zlib.crc32(name.encode("utf-8"))


@contextmanager
def advisory_lock(engine, name: str):
    """Hold a Postgres advisory lock on *name* for the block.

    Yields False when another session holds it. Other dialects yield True:
    SQLite is the single-process fallback, and the lock is a no-op there.
    """
    if engine.dialect.name != "postgresql":
        yield True
        return
    from sqlalchemy import text
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    key = advisory_key(name)
    acquired = False
    try:
        acquired = bool(
            conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()
        )
        yield acquired
    finally:
        if acquired:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
        conn.close()


def _json_text(value) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _returning_id(engine, conn, sql: str, params: dict) -> int:
    from sqlalchemy import text
    if engine.dialect.name == "sqlite":
        conn.execute(text(sql), params)
        return conn.execute(text("SELECT last_insert_rowid()")).scalar()
    return conn.execute(text(sql + " RETURNING id"), params).scalar()


def insert_run(engine, values: dict) -> int:
    """Insert one ``tracebi_runs`` row. Missing columns are NULL. Returns id."""
    params = {c: values.get(c) for c in _RUN_COLUMNS}
    if not isinstance(params.get("detail"), str):
        params["detail"] = _json_text(params.get("detail"))
    cols = ", ".join(_RUN_COLUMNS)
    marks = ", ".join(f":{c}" for c in _RUN_COLUMNS)
    with engine.begin() as conn:
        return _returning_id(
            engine, conn,
            f"INSERT INTO tracebi_runs ({cols}) VALUES ({marks})",
            params,
        )


def update_run(url: Optional[str], run_id: int, **fields) -> None:
    """Update one row. Unknown keys are ignored. ``detail`` dicts are stored as JSON."""
    fields = {k: v for k, v in fields.items() if k in _UPDATABLE}
    if not fields:
        return
    if "detail" in fields:
        fields["detail"] = _json_text(fields["detail"])
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    fields["id"] = run_id
    eng = ensure(url)
    from sqlalchemy import text
    with eng.begin() as conn:
        conn.execute(text(f"UPDATE tracebi_runs SET {sets} WHERE id = :id"), fields)


def record_run(
    *,
    kind: str,
    target: str,
    status: str,
    started: Optional[str] = None,
    finished: Optional[str] = None,
    output_path: Optional[str] = None,
    verdict: Optional[str] = None,
    detail=None,
    actor: Any = _UNSET,
    actor_role: Any = _UNSET,
    url: Optional[str] = None,
) -> int:
    """Insert a non-pipeline run. Actor comes from the audit context when omitted."""
    from datetime import datetime, timezone

    from tracebi.audit import get_actor

    if actor is _UNSET and actor_role is _UNSET:
        actor, actor_role = get_actor()
    else:
        if actor is _UNSET:
            actor = None
        if actor_role is _UNSET:
            actor_role = None
    now = datetime.now(timezone.utc).isoformat()
    eng = ensure(url)
    return insert_run(eng, {
        "layer_name": target,
        "layer_type": kind,
        "started_at": started or now,
        "completed_at": finished,
        "status": status,
        "actor": actor,
        "actor_role": actor_role,
        "kind": kind,
        "target": target,
        "finished": finished,
        "output_path": output_path,
        "verdict": verdict,
        "detail": detail,
    })


def _parse_detail(raw):
    if not isinstance(raw, str) or not raw:
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _public(row: dict) -> dict:
    detail = _parse_detail(row.get("detail"))
    if not isinstance(detail, dict):
        detail = {} if detail in (None, "") else {"note": detail}
    if row.get("error_message") and "error" not in detail:
        detail = {**detail, "error": row["error_message"]}
    if row.get("rows_in") is not None and "rows_in" not in detail:
        detail = {**detail, "rows_in": row["rows_in"], "rows_out": row["rows_out"]}
    return {
        "id": row["id"],
        "kind": row.get("kind"),
        "target": row.get("target") or row.get("layer_name"),
        "actor": row.get("actor"),
        "actor_role": row.get("actor_role"),
        "status": row.get("status"),
        "started": row.get("started_at"),
        "finished": row.get("finished") or row.get("completed_at"),
        "output_path": row.get("output_path"),
        "verdict": row.get("verdict"),
        "detail": detail or None,
    }


def record_report_build(
    target: str,
    output_path: Optional[str],
    *,
    verdict: Optional[str] = None,
    manifest_path: Optional[str] = None,
    url: Optional[str] = None,
) -> int:
    """A ``tracebi report build`` / ``build_report`` / web render.

    Verdict stays unset until a caller has actually verified the receipt.
    """
    detail = {"manifest_path": manifest_path} if manifest_path else None
    return record_run(
        kind="report_build",
        target=target,
        status="succeeded",
        output_path=output_path,
        verdict=verdict,
        detail=detail,
        url=url,
    )


def try_record_report_build(
    target: str,
    output_path: Optional[str],
    *,
    manifest_path: Optional[str] = None,
    url: Optional[str] = None,
) -> Optional[str]:
    """Record a report build. A store failure returns a short note; it does
    not raise. The artifact is already written."""
    try:
        record_report_build(
            target, output_path, manifest_path=manifest_path, url=url,
        )
    except ImportError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 — recording is not the build
        return f"{type(exc).__name__}: {exc}"
    return None


def list_runs(
    *,
    kind: Optional[str] = None,
    target: Optional[str] = None,
    limit: int = 50,
    url: Optional[str] = None,
) -> list[dict]:
    """Newest first. ``kind`` and ``target`` are optional filters."""
    limit = max(1, min(int(limit), 500))
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM tracebi_runs "
            "WHERE (:kind IS NULL OR kind = :kind) "
            "AND (:target IS NULL OR target = :target OR layer_name = :target) "
            "ORDER BY id DESC LIMIT :lim"
        ), {"kind": kind, "target": target, "lim": limit}).mappings().all()
    return [_public(dict(r)) for r in rows]


def newest_by_target(kind: str, url: Optional[str] = None) -> dict[str, dict]:
    """The newest row of *kind* for each target.

    Newest is the highest id, the same order :func:`list_runs` uses.
    """
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM tracebi_runs WHERE kind = :kind AND id IN ("
            "SELECT MAX(id) FROM tracebi_runs WHERE kind = :kind GROUP BY target)"
        ), {"kind": kind}).mappings().all()
    out = {}
    for row in rows:
        public = _public(dict(row))
        target = public.get("target")
        if target:
            out[target] = public
    return out


def count_by_target(kind: str, url: Optional[str] = None) -> dict[str, int]:
    """How many rows of *kind* each target has."""
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT target, COUNT(*) AS n FROM tracebi_runs "
            "WHERE kind = :kind AND target IS NOT NULL GROUP BY target"
        ), {"kind": kind}).fetchall()
    return {target: int(n) for target, n in rows if target}


def get_run(run_id: int, url: Optional[str] = None) -> Optional[dict]:
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM tracebi_runs WHERE id = :id"), {"id": run_id},
        ).mappings().first()
    return dict(row) if row else None


def latest_output(target: str, url: Optional[str] = None) -> Optional[dict]:
    """The newest page-producing run for *target* that names a file."""
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        row = conn.execute(text(
            "SELECT * FROM tracebi_runs "
            "WHERE (target = :t OR layer_name = :t) "
            "AND kind IN ('report_build', 'background_run') "
            "AND output_path IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ), {"t": target}).mappings().first()
    return _public(dict(row)) if row else None


def _strip_schedule(rec: dict) -> dict:
    out = dict(rec)
    out.pop("import_key", None)
    out.pop("_output_dir", None)
    return out


def _schedule_rows(output_dir: Union[str, Path], url: Optional[str] = None) -> list[dict]:
    """Stored schedule records for *output_dir*, oldest first, markers stripped."""
    root = str(Path(output_dir).resolve())
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT detail FROM tracebi_runs WHERE kind = 'schedule' ORDER BY id"
        )).fetchall()
    out = []
    for (raw,) in rows:
        rec = _parse_detail(raw)
        if not isinstance(rec, dict) or rec.get("_output_dir") != root:
            continue
        if rec.get("report"):
            out.append(_strip_schedule(rec))
    return out


def import_schedule_log(output_dir: Union[str, Path], url: Optional[str] = None) -> int:
    """Import ``schedule_runs.jsonl`` once. A second call adds nothing.

    Broken lines are skipped. Each imported row remembers the file it came
    from, so two directories do not share a history.
    """
    log = Path(output_dir) / "schedule_runs.jsonl"
    if not log.is_file():
        return 0
    root = str(Path(output_dir).resolve())
    eng = ensure(url)
    from sqlalchemy import text
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT detail FROM tracebi_runs WHERE kind = 'schedule'"
        )).fetchall()
    seen = set()
    for (raw,) in rows:
        rec = _parse_detail(raw)
        if isinstance(rec, dict) and rec.get("import_key"):
            seen.add(rec["import_key"])
    added = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        key = hashlib.sha256(f"{root}\n{line}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or not rec.get("report"):
            continue
        rec["import_key"] = key
        rec["_output_dir"] = root
        record_run(
            kind="schedule",
            target=rec["report"],
            status=rec.get("status") or "failed",
            started=rec.get("started_at"),
            finished=rec.get("finished_at"),
            output_path=rec.get("output"),
            verdict=rec.get("verdict"),
            detail=rec,
            actor=rec.get("actor"),
            actor_role=rec.get("actor_role"),
            url=url,
        )
        seen.add(key)
        added += 1
    return added


def record_schedule(record: dict, output_dir: Union[str, Path],
                    url: Optional[str] = None) -> None:
    """Append one schedule run. Imports a legacy log in this directory first."""
    import_schedule_log(output_dir, url=url)
    stored = dict(record)
    stored["_output_dir"] = str(Path(output_dir).resolve())
    record_run(
        kind="schedule",
        target=record.get("report") or "",
        status=record.get("status") or "failed",
        started=record.get("started_at"),
        finished=record.get("finished_at"),
        output_path=record.get("output"),
        verdict=record.get("verdict"),
        detail=stored,
        actor=record.get("actor"),
        actor_role=record.get("actor_role"),
        url=url,
    )


def schedule_records(output_dir: Union[str, Path], url: Optional[str] = None) -> list[dict]:
    """Schedule runs for *output_dir*, oldest first, after a one-time import."""
    import_schedule_log(output_dir, url=url)
    return _schedule_rows(output_dir, url=url)
