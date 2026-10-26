"""Persist local CLI/loop failures for issue → hotfix planning."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from jobbot.config import JobbotConfig, load_config
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.db.models import OpsFailureRow
from jobbot.exit_codes import SUCCESS, USER_CANCEL, VALIDATION_FAILURE
from jobbot.jobs.ids import next_failure_id
from jobbot.ops.redact import redact_context, redact_text

STATUSES = frozenset({"new", "triaged", "fixed", "wontfix"})
_OPS_SKIP_PREFIXES = ("ops", "version")


@dataclass(frozen=True)
class FailureRecord:
    id: str
    ts: datetime
    command: str
    component: str
    exit_code: int
    error_class: str
    message: str
    traceback: str
    context: dict[str, Any]
    fingerprint: str
    status: str
    issue_url: str | None


def normalize_message(message: str) -> str:
    """Stabilize messages for fingerprinting (drop digits/paths noise)."""
    text = redact_text(message or "", max_len=2000).lower().strip()
    text = re.sub(r"\b[jf]\d{4}\b", "<id>", text)
    text = re.sub(r"\b\d+\b", "<n>", text)
    text = re.sub(r"/[^\s]+", "<path>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def failure_fingerprint(
    *,
    component: str,
    error_class: str,
    message: str,
) -> str:
    payload = f"{component}|{error_class}|{normalize_message(message)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def infer_component(argv: Sequence[str]) -> str:
    """First subcommand after program name, or 'cli'."""
    parts = [a for a in argv if a and not a.startswith("-")]
    # Drop program name if present
    if parts and Path(parts[0]).name in {"jobbot", "python", "python3"}:
        parts = parts[1:]
    if parts and parts[0].endswith(".py"):
        parts = parts[1:]
    if not parts:
        return "cli"
    return parts[0]


def _stabilize_command_url(arg: str) -> str:
    """Keep identity for hard-link re-runs; drop tracking query noise."""
    if not arg.startswith(("http://", "https://")):
        return arg
    try:
        from jobbot.jobs.indeed_url import IndeedUrlError, canonical_indeed_job_url

        return canonical_indeed_job_url(arg)
    except IndeedUrlError:
        pass
    except Exception:  # noqa: BLE001 — never break failure recording on URL parse
        pass
    from jobbot.ops.redact import host_only_url

    return host_only_url(arg)


def normalize_command(argv: Sequence[str]) -> str:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            cleaned.append("<path>")
            continue
        if arg in {"--seed", "--fixture", "--cdp", "--file"}:
            cleaned.append(arg)
            skip_next = True
            continue
        if arg.startswith("--cdp="):
            cleaned.append("--cdp=<redacted>")
            continue
        cleaned.append(_stabilize_command_url(arg))
    # Drop absolute interpreter / script noise: keep jobbot-ish argv
    if cleaned and Path(cleaned[0]).name.startswith("python"):
        cleaned = cleaned[1:]
    if cleaned and cleaned[0].endswith("jobbot"):
        cleaned[0] = "jobbot"
    elif cleaned and "jobbot" not in cleaned[0]:
        cleaned = ["jobbot", *cleaned]
    return " ".join(cleaned)


def _row_to_record(row: OpsFailureRow) -> FailureRecord:
    try:
        ctx = json.loads(row.context_json or "{}")
    except json.JSONDecodeError:
        ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    return FailureRecord(
        id=row.id,
        ts=row.ts,
        command=row.command,
        component=row.component,
        exit_code=row.exit_code,
        error_class=row.error_class,
        message=row.message,
        traceback=row.traceback,
        context=ctx,
        fingerprint=row.fingerprint,
        status=row.status,
        issue_url=row.issue_url,
    )


def _mirror_json(config: JobbotConfig, record: FailureRecord) -> Path:
    out_dir = config.output_dir / "ops" / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.id}.json"
    payload = {
        "id": record.id,
        "ts": record.ts.isoformat(),
        "command": record.command,
        "component": record.component,
        "exit_code": record.exit_code,
        "error_class": record.error_class,
        "message": record.message,
        "traceback": record.traceback,
        "context": record.context,
        "fingerprint": record.fingerprint,
        "status": record.status,
        "issue_url": record.issue_url,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def record_failure(
    session: Session,
    *,
    config: JobbotConfig,
    exit_code: int,
    argv: Sequence[str] | None = None,
    exc: BaseException | None = None,
    component: str | None = None,
    error_class: str | None = None,
    message: str | None = None,
    tb: str | None = None,
    context: dict[str, Any] | None = None,
    write_mirror: bool = True,
) -> FailureRecord:
    """Insert a failure row and optional output/ops/failures/Fxxxx.json mirror."""
    argv_list = list(argv if argv is not None else sys.argv[1:])
    comp = component or infer_component(argv_list)
    cls = error_class or (type(exc).__name__ if exc is not None else "Exit")
    msg = message if message is not None else (str(exc) if exc is not None else "")
    if tb is None and exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    from jobbot.vault import needles_for_profile

    needles = needles_for_profile(config.profile_path)
    tb_text = redact_text(tb or "", max_len=12000, extra_secrets=needles)
    msg_text = redact_text(msg, max_len=2000, extra_secrets=needles)
    ctx = redact_context(context, extra_secrets=needles)
    fp = failure_fingerprint(component=comp, error_class=cls, message=msg_text)
    fid = next_failure_id(session)
    row = OpsFailureRow(
        id=fid,
        ts=datetime.now(UTC),
        command=normalize_command(argv_list),
        component=comp,
        exit_code=int(exit_code),
        error_class=cls,
        message=msg_text,
        traceback=tb_text,
        context_json=json.dumps(ctx, ensure_ascii=False),
        fingerprint=fp,
        status="new",
        issue_url=None,
    )
    session.add(row)
    session.commit()
    record = _row_to_record(row)
    if write_mirror:
        _mirror_json(config, record)
    return record


def should_record_cli_failure(argv: Sequence[str], exit_code: int) -> bool:
    """Skip success, user cancel, validation errors, help probes, and ops/version."""
    if exit_code in {SUCCESS, USER_CANCEL, VALIDATION_FAILURE}:
        return False
    if any(flag in {"--help", "-h"} for flag in argv):
        return False
    parts = [a for a in argv if a and not a.startswith("-")]
    if parts and Path(parts[0]).name in {"jobbot", "python", "python3"}:
        parts = parts[1:]
    if not parts:
        return True
    head = parts[0]
    return head not in _OPS_SKIP_PREFIXES


ABORT_MESSAGE = "Command aborted at a confirmation prompt (no answer given)"


def runtime_context() -> dict[str, Any]:
    """Facts a future issue needs: could anyone answer prompts, on what build."""
    from jobbot import __version__

    return {
        "stdin_tty": sys.stdin.isatty(),
        "version": __version__,
        "platform": sys.platform,
    }


def capture_cli_failure(
    exit_code: int,
    *,
    argv: Sequence[str] | None = None,
    exc: BaseException | None = None,
    config: JobbotConfig | None = None,
    error_class: str | None = None,
    message: str | None = None,
    context: dict[str, Any] | None = None,
) -> FailureRecord | None:
    """Record from CLI boundary; never raise (observability must not crash the crash)."""
    argv_list = list(argv if argv is not None else sys.argv)
    # Prefer argv without interpreter when called as python -m
    if not should_record_cli_failure(argv_list, exit_code):
        return None
    try:
        cfg = config or load_config()
        engine = make_engine(cfg.database_path)
        session = make_session_factory(engine)()
        try:
            return record_failure(
                session,
                config=cfg,
                exit_code=exit_code,
                argv=argv_list,
                exc=exc,
                error_class=error_class,
                message=message,
                context=context if context is not None else runtime_context(),
            )
        finally:
            session.close()
    except Exception:  # noqa: BLE001 — never mask the original CLI failure
        return None


def list_failures(
    session: Session,
    *,
    status: str | None = "new",
    limit: int = 50,
) -> list[FailureRecord]:
    stmt = select(OpsFailureRow).order_by(OpsFailureRow.id.desc())
    if status is not None:
        stmt = stmt.where(OpsFailureRow.status == status)
    rows = session.scalars(stmt.limit(limit)).all()
    return [_row_to_record(r) for r in rows]


def get_failure(session: Session, failure_id: str) -> FailureRecord | None:
    row = session.get(OpsFailureRow, failure_id)
    return _row_to_record(row) if row is not None else None


def mark_status(
    session: Session,
    failure_id: str,
    status: str,
    *,
    issue_url: str | None = None,
) -> FailureRecord | None:
    if status not in STATUSES:
        msg = f"status must be one of {sorted(STATUSES)}"
        raise ValueError(msg)
    row = session.get(OpsFailureRow, failure_id)
    if row is None:
        return None
    row.status = status
    if issue_url is not None:
        row.issue_url = issue_url
    session.commit()
    return _row_to_record(row)


def group_by_fingerprint(records: Sequence[FailureRecord]) -> list[tuple[str, list[FailureRecord]]]:
    """Newest-first groups for triage listing."""
    order: list[str] = []
    groups: dict[str, list[FailureRecord]] = {}
    for rec in records:
        if rec.fingerprint not in groups:
            order.append(rec.fingerprint)
            groups[rec.fingerprint] = []
        groups[rec.fingerprint].append(rec)
    return [(fp, groups[fp]) for fp in order]


def issue_title(record: FailureRecord) -> str:
    short = record.message.strip().splitlines()[0] if record.message.strip() else record.error_class
    if len(short) > 72:
        short = short[:69] + "…"
    return f"[jobbot:{record.component}] {record.error_class}: {short} ({record.fingerprint})"


def issue_body(record: FailureRecord) -> str:
    ctx = json.dumps(record.context, indent=2, ensure_ascii=False) if record.context else "{}"
    tb = record.traceback or "(none)"
    return (
        f"## Failure `{record.id}`\n\n"
        f"- **fingerprint:** `{record.fingerprint}`\n"
        f"- **component:** `{record.component}`\n"
        f"- **exit_code:** `{record.exit_code}`\n"
        f"- **command:** `{record.command}`\n"
        f"- **error:** `{record.error_class}`: {record.message}\n"
        f"- **ts:** `{record.ts.isoformat()}`\n\n"
        f"### Context\n\n```json\n{ctx}\n```\n\n"
        f"### Traceback\n\n```\n{tb}\n```\n\n"
        f"### Checklist\n\n"
        f"- [ ] Reproduce locally\n"
        f"- [ ] Regression unit test that fails first\n"
        f"- [ ] Hotfix / refactor\n"
        f"- [ ] `jobbot ops failure triage {record.id} --status fixed`\n"
    )
