"""Empleos Públicos (Chile) — public concurso board.

Live listing is JS-loaded and often refuses datacenter fetches (403). Search and
detail ingest are fixture-first: pass saved ficha HTML or a search JSON dump.
Nothing here invents experience or bypasses the portal login for apply.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from jobbot.config import JobbotConfig, load_config
from jobbot.jobs.normalization import fold_text
from jobbot.jobs.parsing import extract_skills_from_text
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind

SITE_ORIGIN = "https://www.empleospublicos.cl"
FICHA_PATH = "/pub/convocatorias/avisotrabajoficha.aspx"

_INSTITUTION_RE = re.compile(
    r"(?i)^\s*instituci[oó]n\s*:\s*(.+?)\s*$"
)
_REGION_RE = re.compile(r"(?i)^\s*regi[oó]n\s*:\s*(.+?)\s*$")


class EmpleosPublicosParseError(ValueError):
    """HTML is not a recognizable Empleos Públicos ficha."""


def canonical_ficha_url(url: str) -> str:
    """Keep host + ficha path + convocatoria id ``i``; drop tracking."""
    parsed = urlparse(url.strip())
    if not parsed.netloc:
        msg = f"not an Empleos Públicos URL: {url!r}"
        raise EmpleosPublicosParseError(msg)
    host = parsed.netloc.casefold().removeprefix("www.")
    if host != "empleospublicos.cl":
        msg = f"not an Empleos Públicos host: {parsed.netloc}"
        raise EmpleosPublicosParseError(msg)
    query = parse_qs(parsed.query)
    ids = query.get("i") or query.get("I")
    if not ids or not ids[0].isdigit():
        msg = f"Empleos Públicos ficha needs ?i=<id>: {url}"
        raise EmpleosPublicosParseError(msg)
    return f"{SITE_ORIGIN}{FICHA_PATH}?i={ids[0]}"


def job_from_ficha_html(html: str, *, url: str) -> JobPosting:
    """Parse a saved avisotrabajoficha page into a JobPosting."""
    soup = BeautifulSoup(html, "html.parser")
    title = _title(soup)
    company = _institution(soup)
    if not title or not company:
        msg = "Empleos Públicos ficha needs a visible cargo title and institución"
        raise EmpleosPublicosParseError(msg)

    description = _description(soup)
    location = _location(soup)
    requirements = _requirements(soup)
    try:
        canonical = canonical_ficha_url(url)
        source_id = parse_qs(urlparse(canonical).query)["i"][0]
    except EmpleosPublicosParseError:
        canonical = url.strip()
        source_id = None

    return JobPosting(
        id="PENDING",
        source="empleos_publicos",
        source_job_id=source_id,
        url=canonical or None,
        title=title,
        company=company,
        location=location,
        description=description,
        raw_description=description,
        requirements=requirements,
        skills=extract_skills_from_text(description + "\n" + "\n".join(requirements)),
        ats_url=canonical or None,
        ats_kind=AtsKind.EMPLEOS_PUBLICOS.value,
        note="empleos publicos ficha",
    )


def jobs_from_search_payload(payload: dict[str, Any] | list[Any]) -> list[JobPosting]:
    """Turn a saved search JSON dump into lead postings (not full JDs)."""
    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("results") or payload.get("items") or []
    jobs: list[JobPosting] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        company = str(row.get("institution") or row.get("company") or "").strip()
        if not title or not company:
            continue
        raw_url = str(row.get("url") or "").strip()
        try:
            url = canonical_ficha_url(raw_url) if raw_url else None
            source_id = (
                parse_qs(urlparse(url).query)["i"][0]
                if url
                else str(row.get("id") or "") or None
            )
        except EmpleosPublicosParseError:
            url = raw_url or None
            source_id = str(row.get("id") or "") or None
        region = str(row.get("region") or "").strip() or None
        closes = str(row.get("closes_on") or "").strip()
        snippet = f"Cierra: {closes}" if closes else "Convocatoria Empleos Públicos"
        jobs.append(
            JobPosting(
                id="PENDING",
                source="empleos_publicos",
                source_job_id=source_id,
                url=url,
                title=title,
                company=company,
                location=region,
                description=snippet,
                raw_description=snippet,
                skills=[],
                ats_url=url,
                ats_kind=AtsKind.EMPLEOS_PUBLICOS.value,
                note="empleos publicos search lead",
            )
        )
    return jobs


def load_search_fixture(path: Path) -> list[JobPosting]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, (dict, list)):
        msg = f"search fixture must be a JSON object or array: {path}"
        raise EmpleosPublicosParseError(msg)
    return jobs_from_search_payload(data)


class EmpleosPublicosJobSource:
    """Discover concursos from a saved search dump (offline / HITL export)."""

    def __init__(
        self,
        config: JobbotConfig | None = None,
        *,
        fixture: Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.fixture = fixture

    def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        if self.fixture is None:
            msg = (
                "Empleos Públicos search is fixture-first (live board is JS + often 403). "
                "Pass --fixture with a saved search JSON dump."
            )
            raise EmpleosPublicosParseError(msg)
        jobs = load_search_fixture(self.fixture)
        needle = fold_text(query.query or "").strip()
        if needle:
            jobs = [
                job
                for job in jobs
                if needle in fold_text(job.title)
                or needle in fold_text(job.company)
                or needle in fold_text(job.description or "")
            ]
        return jobs[: query.limit]

    def get_job(self, job_id: str) -> JobPosting:
        msg = "Use jobbot get URL --fixture PATH for an Empleos Públicos ficha"
        raise NotImplementedError(msg)


def _title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    if soup.title and soup.title.string:
        text = soup.title.string.strip()
        text = re.split(r"\s*[·|]\s*", text, maxsplit=1)[0].strip()
        text = re.sub(r"(?i)\s*ficha.*$", "", text).strip()
        if text:
            return text
    return ""


def _institution(soup: BeautifulSoup) -> str:
    for node in soup.find_all(["p", "div", "span", "li", "td"]):
        text = node.get_text(" ", strip=True)
        match = _INSTITUTION_RE.match(text)
        if match:
            return match.group(1).strip()
    return ""


def _location(soup: BeautifulSoup) -> str | None:
    region = ""
    for node in soup.find_all(["p", "div", "span", "li", "td"]):
        text = node.get_text(" ", strip=True)
        match = _REGION_RE.match(text)
        if match:
            region = match.group(1).strip()
            break
    return region or None


def _description(soup: BeautifulSoup) -> str:
    chunks: list[str] = []
    for heading in soup.find_all(["h2", "h3"]):
        label = heading.get_text(" ", strip=True).casefold()
        if any(
            key in label
            for key in (
                "objetivo",
                "perfil",
                "formación",
                "formacion",
                "experiencia",
                "requisitos",
                "condiciones",
            )
        ):
            parts = [heading.get_text(" ", strip=True)]
            for sibling in heading.find_next_siblings():
                if getattr(sibling, "name", None) in {"h2", "h3"}:
                    break
                text = sibling.get_text("\n", strip=True)
                if text:
                    parts.append(text)
            chunks.append("\n".join(parts))
    body = "\n\n".join(chunks).strip()
    if body:
        return body
    return soup.get_text("\n", strip=True)


def _requirements(soup: BeautifulSoup) -> list[str]:
    for heading in soup.find_all(["h2", "h3"]):
        label = heading.get_text(" ", strip=True).casefold()
        if "requisitos" not in label:
            continue
        for sibling in heading.find_next_siblings():
            if getattr(sibling, "name", None) in {"h2", "h3"}:
                break
            if getattr(sibling, "name", None) == "ul":
                items = [
                    li.get_text(" ", strip=True)
                    for li in sibling.find_all("li")
                    if li.get_text(strip=True)
                ]
                if items:
                    return items
    return []
