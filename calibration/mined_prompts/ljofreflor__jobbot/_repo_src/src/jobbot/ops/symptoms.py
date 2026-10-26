"""Symptoms: appearances that return — local, redacted; not a requirements backlog.

Phenomenology of software: we do not satisfy needs via tickets. A *symptom* is
an appearance in vibecode that *returns* (still paid with client tokens). We
capture it securely so we can analyze the **conditions of possibility** of that
need-for-a-need and encode them as System 1 code. ``rule_hypothesis`` is that
structural condition — not a wish to fulfill.

Nothing here phones home; SQLite stays gitignored; text is redacted before write
and before any HITL GitHub issue body.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from jobbot.db.models import OpsSymptomRow
from jobbot.jobs.ids import next_symptom_id
from jobbot.ops.compile import COMPRESS_CONTRACT, PROMOTION_STEPS
from jobbot.ops.redact import redact_context, redact_text

AREAS = frozenset(
    {"cv", "jobs", "portals", "matching", "ops", "nlp", "companies", "profile", "other"}
)
STATUSES = frozenset({"latent", "acknowledged", "compressing", "resolved", "wontfix"})

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_CL_PHONE_RE = re.compile(r"\+\s?56[\s\-.,]*9[\s\-.,]*\d{4}[\s\-.,]*\d{4}")
_RUT_RE = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}\s*-\s*[\dkK]\b")
_HOME_PATH_RE = re.compile(r"(?i)(/Users|/home)/[A-Za-z0-9._\-]+")


@dataclass(frozen=True)
class SymptomRecord:
    id: str
    created_at: datetime
    updated_at: datetime
    area: str
    title: str
    intent: str
    rule_hypothesis: str
    fingerprint: str
    sightings: int
    status: str
    feature_path: str | None
    issue_url: str | None
    context: dict[str, Any]


def sanitize_symptom_text(text: str, *, max_len: int = 2000) -> str:
    """Strip secrets and contact-shaped PII before local store or issue export."""
    out = redact_text(text or "", max_len=max_len)
    out = _EMAIL_RE.sub("<email>", out)
    out = _CL_PHONE_RE.sub("<phone>", out)
    out = _RUT_RE.sub("<rut>", out)
    out = _HOME_PATH_RE.sub("<home>", out)
    out = re.sub(r"\b[jf]\d{4}\b", "<id>", out, flags=re.I)
    return out.strip()


def normalize_for_fingerprint(text: str) -> str:
    text = sanitize_symptom_text(text, max_len=1500).casefold()
    text = re.sub(r"\b\d+\b", "<n>", text)
    text = re.sub(r"/[^\s]+", "<path>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def symptom_fingerprint(*, area: str, intent: str, rule_hypothesis: str = "") -> str:
    intent_n = normalize_for_fingerprint(intent)
    rule_n = normalize_for_fingerprint(rule_hypothesis)
    payload = f"{area}|{intent_n}|{rule_n}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _row_to_record(row: OpsSymptomRow) -> SymptomRecord:
    try:
        ctx = json.loads(row.context_json or "{}")
    except json.JSONDecodeError:
        ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    return SymptomRecord(
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        area=row.area,
        title=row.title,
        intent=row.intent,
        rule_hypothesis=row.rule_hypothesis,
        fingerprint=row.fingerprint,
        sightings=row.sightings,
        status=row.status,
        feature_path=row.feature_path,
        issue_url=row.issue_url,
        context=ctx,
    )


def note_symptom(
    session: Session,
    *,
    intent: str,
    area: str = "other",
    title: str | None = None,
    rule_hypothesis: str = "",
    context: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> SymptomRecord:
    """Record or reinforce a latent requirement. Same fingerprint → +sightings."""
    area_key = (area or "other").casefold().strip()
    if area_key not in AREAS:
        area_key = "other"
    clean_intent = sanitize_symptom_text(intent)
    if not clean_intent:
        msg = "symptom intent is empty after redaction"
        raise ValueError(msg)
    clean_rule = sanitize_symptom_text(rule_hypothesis, max_len=1000)
    clean_title = sanitize_symptom_text(
        title or clean_intent.split("\n", 1)[0][:120],
        max_len=200,
    )
    fp = symptom_fingerprint(area=area_key, intent=clean_intent, rule_hypothesis=clean_rule)
    now = datetime.now(UTC)
    existing = session.scalars(select(OpsSymptomRow).where(OpsSymptomRow.fingerprint == fp)).first()
    if existing is not None:
        existing.sightings = int(existing.sightings or 1) + 1
        existing.updated_at = now
        if clean_rule and not existing.rule_hypothesis:
            existing.rule_hypothesis = clean_rule
        if existing.status == "resolved":
            # Symptom returned after compression — reopen as latent signal.
            existing.status = "latent"
        session.commit()
        record = _row_to_record(existing)
    else:
        row = OpsSymptomRow(
            id=next_symptom_id(session),
            created_at=now,
            updated_at=now,
            area=area_key,
            title=clean_title,
            intent=clean_intent,
            rule_hypothesis=clean_rule,
            fingerprint=fp,
            sightings=1,
            status="latent",
            feature_path=None,
            issue_url=None,
            context_json=json.dumps(redact_context(context), ensure_ascii=False),
        )
        session.add(row)
        session.commit()
        record = _row_to_record(row)

    if output_dir is not None:
        _mirror_json(output_dir, record)
    return record


def list_symptoms(
    session: Session,
    *,
    status: str | None = None,
    area: str | None = None,
    limit: int = 50,
) -> list[SymptomRecord]:
    stmt = select(OpsSymptomRow).order_by(OpsSymptomRow.updated_at.desc())
    if status:
        stmt = stmt.where(OpsSymptomRow.status == status)
    if area:
        stmt = stmt.where(OpsSymptomRow.area == area.casefold())
    rows = session.scalars(stmt.limit(limit)).all()
    return [_row_to_record(r) for r in rows]


def get_symptom(session: Session, symptom_id: str) -> SymptomRecord | None:
    row = session.get(OpsSymptomRow, symptom_id)
    return _row_to_record(row) if row else None


def mark_symptom_status(
    session: Session,
    symptom_id: str,
    status: str,
    *,
    feature_path: str | None = None,
    issue_url: str | None = None,
) -> SymptomRecord | None:
    if status not in STATUSES:
        msg = f"unknown status {status!r}; expected one of {sorted(STATUSES)}"
        raise ValueError(msg)
    row = session.get(OpsSymptomRow, symptom_id)
    if row is None:
        return None
    row.status = status
    row.updated_at = datetime.now(UTC)
    if feature_path is not None:
        row.feature_path = sanitize_symptom_text(feature_path, max_len=400)
    if issue_url is not None:
        row.issue_url = issue_url
    session.commit()
    return _row_to_record(row)


def promote_plan(record: SymptomRecord) -> str:
    """Human-readable plan: conditions of possibility → System 1 (no code written)."""
    from jobbot.ops.compile import PHENOMENOLOGY_CONTRACT

    steps = "\n".join(f"  {i}. {step}" for i, step in enumerate(PROMOTION_STEPS, start=1))
    rule = record.rule_hypothesis or (
        "(name the structural condition of possibility — one falsifiable sentence)"
    )
    return (
        f"Symptom {record.id} — appearance still reconstituting on client tokens\n"
        f"Phenomenology: {PHENOMENOLOGY_CONTRACT}\n"
        f"Compress: {COMPRESS_CONTRACT}\n"
        f"Area: {record.area}  sightings: {record.sightings}  status: {record.status}\n"
        f"Title: {record.title}\n"
        f"Appearance (redacted): {record.intent}\n"
        f"Condition of possibility: {rule}\n"
        f"Fingerprint: {record.fingerprint}\n\n"
        f"Encode conditions as System 1:\n{steps}\n\n"
        "Do not treat this as a wish-list ticket. Do not paste raw chat or PII.\n"
        f"When done: jobbot ops symptom triage {record.id} --status resolved "
        f"--feature path/to/module.py"
    )


def issue_title(record: SymptomRecord) -> str:
    return f"[symptom:{record.area}] {record.title[:80]}"


def issue_body(record: SymptomRecord) -> str:
    """Redacted body only — safe enough for HITL ``gh issue create``."""
    plan = promote_plan(record)
    return (
        "## Symptom (appearance that returns)\n\n"
        "Not a requirements ticket to satisfy. Analyze the conditions of "
        "possibility of this need-for-a-need; encode them as System 1.\n\n"
        f"```\n{plan}\n```\n\n"
        "No candidate PII, no raw transcript, no secrets.\n"
    )


def _mirror_json(output_dir: Path, record: SymptomRecord) -> None:
    dest = output_dir / "ops" / "symptoms" / f"{record.id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "area": record.area,
        "title": record.title,
        "intent": record.intent,
        "rule_hypothesis": record.rule_hypothesis,
        "fingerprint": record.fingerprint,
        "sightings": record.sightings,
        "status": record.status,
        "feature_path": record.feature_path,
        "issue_url": record.issue_url,
        "context": record.context,
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
