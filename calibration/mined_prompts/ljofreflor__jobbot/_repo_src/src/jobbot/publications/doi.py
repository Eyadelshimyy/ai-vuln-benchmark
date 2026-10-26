"""Enrich publications with DOI metadata (Crossref) — facts only, no invention."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("jobbot.publications.doi")


@dataclass(frozen=True)
class DoiMetadata:
    doi: str
    title: str | None
    journal: str | None
    year: int | None
    authors: list[str]


def fetch_crossref_by_doi(doi: str) -> DoiMetadata:
    doi = doi.removeprefix("https://doi.org/").strip()
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    raw = _get_json(url)
    return _parse_work(raw["message"])


def search_crossref_by_title(title: str, *, rows: int = 3) -> list[DoiMetadata]:
    q = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows={rows}"
    raw = _get_json(url)
    return [_parse_work(item) for item in raw["message"]["items"]]


def _parse_work(message: dict[str, Any]) -> DoiMetadata:
    authors: list[str] = []
    for author in message.get("author") or []:
        name = " ".join(x for x in (author.get("given"), author.get("family")) if x)
        if name:
            authors.append(name)
    year = None
    for key in ("published-print", "published-online", "created"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            year = int(parts[0][0])
            break
    titles = message.get("title") or []
    containers = message.get("container-title") or []
    return DoiMetadata(
        doi=str(message.get("DOI") or ""),
        title=titles[0] if titles else None,
        journal=containers[0] if containers else None,
        year=year,
        authors=authors,
    )


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "jobbot/0.1 (mailto:local; Crossref enrichment)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — Crossref API
        payload: dict[str, Any] = json.load(resp)
        return payload
