"""Registry of public sources about hiring practice, with the same lifecycle as companies.

Candidate until you promote it, rejected forever once you reject it. The file is
local and gitignored: what it holds is a reading list plus extracted practices, and
that reflects the searches of whoever ran them.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from jobbot.companies.models import KnowledgeStatus, utc_now
from jobbot.companies.urls import PrivateRouteRejected, canonical_key
from jobbot.recruiters.playbook import Practice


class RecruiterSource(BaseModel):
    """One public page about how hiring reads CVs, and what it taught.

    There is deliberately no field for an author, an employer or a contact: the
    shape cannot hold a person, so no code path can start storing one.
    """

    url: str = Field(min_length=1)
    title: str = ""
    practices: list[Practice] = Field(default_factory=list)
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE
    first_seen: datetime = Field(default_factory=utc_now)
    last_checked: datetime = Field(default_factory=utc_now)
    evidence: str = ""


def default_recruiters_path(root: Path) -> Path:
    return root / "data" / "recruiters.yaml"


def load_sources(path: Path) -> list[RecruiterSource]:
    if not path.is_file():
        return []
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("sources", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [RecruiterSource.model_validate(item) for item in items]


def save_sources(sources: list[RecruiterSource], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "note": (
            "Public sources about hiring practice. Practices only, never people. "
            "Candidate until promoted; only active ones reach `cv advise`."
        ),
        "sources": [source.model_dump(mode="json") for source in sources],
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _key(url: str) -> str:
    try:
        return canonical_key(url)
    except PrivateRouteRejected:
        return url.casefold()


def upsert_source(
    sources: list[RecruiterSource],
    observed: RecruiterSource,
) -> list[RecruiterSource]:
    """Merge a new reading of a source, without undoing a decision you made."""
    key = _key(observed.url)
    out: list[RecruiterSource] = []
    merged = False
    for existing in sources:
        if _key(existing.url) != key or merged:
            out.append(existing)
            continue
        updated = observed.model_copy(
            update={
                "status": existing.status,
                "first_seen": existing.first_seen,
                "last_checked": utc_now(),
            }
        )
        out.append(updated)
        merged = True
    if not merged:
        out.append(observed)
    return out


def _set_status(
    sources: list[RecruiterSource],
    url: str,
    status: KnowledgeStatus,
) -> list[RecruiterSource]:
    key = _key(url)
    return [
        source.model_copy(update={"status": status}) if _key(source.url) == key else source
        for source in sources
    ]


def promote_source(sources: list[RecruiterSource], url: str) -> list[RecruiterSource]:
    """Make one source's practices usable by the advisor. Only you may call this."""
    return _set_status(sources, url, KnowledgeStatus.ACTIVE)


def reject_source(sources: list[RecruiterSource], url: str) -> list[RecruiterSource]:
    return _set_status(sources, url, KnowledgeStatus.REJECTED)


def active_sources(sources: list[RecruiterSource]) -> list[RecruiterSource]:
    return [source for source in sources if source.status is KnowledgeStatus.ACTIVE]


def export_payload(sources: list[RecruiterSource]) -> dict[str, Any]:
    """Shareable view: active sources and their practices, nothing else."""
    return {
        "version": 1,
        "note": "Public hiring practices. No people, no candidate data.",
        "sources": [
            {
                "url": source.url,
                "title": source.title,
                "practices": [
                    {"kind": practice.kind.value, "text": practice.text}
                    for practice in source.practices
                ],
            }
            for source in active_sources(sources)
        ],
    }
