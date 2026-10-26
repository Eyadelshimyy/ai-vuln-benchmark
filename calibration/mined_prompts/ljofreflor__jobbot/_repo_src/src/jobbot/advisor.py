"""Advisor view across workspaces: status, a client report, consent, deletion.

Each sandbox stays the client's. This module only reads one root at a time and
never prints the owner's name. Contact data that leaves a file goes through
``pii_guard.redact``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from sqlalchemy import select

from jobbot.db.engine import make_engine, make_session_factory
from jobbot.db.models import ApplicationRow, JobRow
from jobbot.ops.pii_guard import redact
from jobbot.workspace import STAMP_NAME, read_stamp, sandboxes_dir, workspace_root

CONSENT_NAME = "consent.yaml"
PROPOSAL_REL = Path("output") / "cv" / "improvement_proposal.yaml"


class WorkspaceDeleteRefused(Exception):
    """Deletion was not confirmed, or the path is not that sandbox."""


@dataclass(frozen=True)
class Consent:
    consented_at: str
    scope: str
    delete_after: str


@dataclass(frozen=True)
class WorkspaceActivity:
    name: str
    fingerprint: str
    jobs: int
    prepared: int
    proposal: bool
    last_activity: str
    retention_due: bool


def consent_path(root: Path) -> Path:
    return root / "data" / CONSENT_NAME


def read_consent(root: Path) -> Consent | None:
    path = consent_path(root)
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    consented = str(raw.get("consented_at") or "").strip()
    scope = str(raw.get("scope") or "").strip()
    delete_after = str(raw.get("delete_after") or "").strip()
    if not consented or not delete_after:
        return None
    return Consent(consented_at=consented, scope=scope, delete_after=delete_after)


def write_consent(root: Path, consent: Consent) -> Path:
    path = consent_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "consented_at": consent.consented_at,
                "scope": consent.scope,
                "delete_after": consent.delete_after,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


def retention_due(consent: Consent | None, *, today: datetime) -> bool:
    if consent is None:
        return False
    try:
        deadline = datetime.fromisoformat(consent.delete_after)
    except ValueError:
        return False
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return deadline.date() < today.astimezone(UTC).date()


def retention_warning(root: Path, *, today: datetime | None = None) -> str | None:
    now = today or datetime.now(UTC)
    if not retention_due(read_consent(root), today=now):
        return None
    return f"Retention date has passed for workspace {root.name}. Delete it when the client agrees."


def workspace_activity(
    name: str, root: Path, *, today: datetime | None = None
) -> WorkspaceActivity:
    now = today or datetime.now(UTC)
    stamp = read_stamp(root / "data" / STAMP_NAME)
    jobs, prepared, last = _counts(root / "data" / "jobbot.sqlite")
    return WorkspaceActivity(
        name=name,
        fingerprint=stamp.fingerprint if stamp else "unstamped",
        jobs=jobs,
        prepared=prepared,
        proposal=(root / PROPOSAL_REL).is_file(),
        last_activity=last or "none",
        retention_due=retention_due(read_consent(root), today=now),
    )


def status_lines(
    names_and_roots: list[tuple[str, Path]], *, today: datetime | None = None
) -> list[str]:
    """One plain line per workspace. No profile name, email, or phone."""
    lines: list[str] = []
    for name, root in names_and_roots:
        if not root.is_dir():
            continue
        row = workspace_activity(name, root, today=today)
        line = (
            f"{row.name} owner={row.fingerprint} jobs={row.jobs} "
            f"prepared={row.prepared} proposal={'yes' if row.proposal else 'no'} "
            f"last={row.last_activity}"
        )
        if row.retention_due:
            line += " retention=due"
        lines.append(line)
    return lines


def write_report(root: Path, *, since: datetime, now: datetime | None = None) -> Path:
    """Shareable summary of this workspace only. Other sandboxes are not opened."""
    moment = now or datetime.now(UTC)
    rows = _rows_since(root / "data" / "jobbot.sqlite", since=since)
    out = root / "output" / "advisor"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    if not rows:
        body = "No activity in this period.\n"
    else:
        lines = ["# Advisor report", "", f"Since {since.date().isoformat()}", ""]
        for job_id, title, company, status, updated, note in rows:
            extra = f" {note}" if note else ""
            lines.append(
                f"- {job_id} {title} @ {company} — {status} ({updated}){extra}"
            )
        body = "\n".join(lines) + "\n"
    path.write_text(redact(body), encoding="utf-8")
    _ = moment
    return path


def parse_since(raw: str, *, now: datetime) -> datetime:
    text = (raw or "7d").strip().casefold()
    if text.endswith("d") and text[:-1].isdigit():
        return now - timedelta(days=int(text[:-1]))
    return datetime.fromisoformat(text)


def delete_workspace(name: str, *, confirmed: bool) -> Path:
    """Remove one sandbox. Refuses the checkout root and any sibling path."""
    if not confirmed:
        raise WorkspaceDeleteRefused("confirmation required")
    root = workspace_root(name).resolve()
    base = sandboxes_dir().resolve()
    if root == base or base not in root.parents or root.name != name:
        raise WorkspaceDeleteRefused(f"refusing to delete {root}")
    if not root.is_dir():
        raise WorkspaceDeleteRefused(f"no workspace {name}")
    shutil.rmtree(root)
    return root


def _counts(db: Path) -> tuple[int, int, str | None]:
    if not db.is_file():
        return 0, 0, None
    session = make_session_factory(make_engine(db))()
    try:
        jobs = len(session.scalars(select(JobRow.id)).all())
        prepared = len(
            session.scalars(
                select(ApplicationRow.id).where(ApplicationRow.status == "prepared")
            ).all()
        )
        stamps = [row.updated_at for row in session.scalars(select(ApplicationRow)).all()]
        stamps.extend(row.discovered_at for row in session.scalars(select(JobRow)).all())
        last = max(stamps).isoformat(timespec="seconds") if stamps else None
        return jobs, prepared, last
    finally:
        session.close()


def _rows_since(
    db: Path, *, since: datetime
) -> list[tuple[str, str, str, str, str, str]]:
    if not db.is_file():
        return []
    session = make_session_factory(make_engine(db))()
    try:
        jobs = {row.id: row for row in session.scalars(select(JobRow)).all()}
        found: list[tuple[str, str, str, str, str, str]] = []
        for app in session.scalars(select(ApplicationRow)).all():
            updated = app.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if updated < since:
                continue
            job = jobs.get(app.job_id)
            title = job.title if job else ""
            company = job.company if job else ""
            note = (job.note or "") if job else ""
            found.append(
                (
                    app.job_id,
                    title,
                    company,
                    app.status,
                    updated.isoformat(timespec="seconds"),
                    note,
                )
            )
        return found
    finally:
        session.close()
