"""Get on Board job source (LATAM/ES) via public search API + hard-link pages."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from jobbot.config import JobbotConfig, load_config
from jobbot.jobs.parsing import extract_skills_from_text
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind
from jobbot.portals.registry import (
    default_portals_path,
    domain_from_url,
    load_registry,
    save_registry,
)

logger = logging.getLogger("jobbot.getonboard")

API_SEARCH = "https://www.getonbrd.com/api/v0/search/jobs"
SITE_ORIGIN = "https://www.getonbrd.com"

# /empleos/<cat>/<slug>, /jobs/<cat>/<slug>, or /jobs/<slug>
_JOB_PATH = re.compile(
    r"^/(?:empleos|jobs)/(?:[^/]+/)?([a-z0-9][a-z0-9-]*)/?$",
    re.I,
)

_SECTION_HEADINGS = (
    "descripción del trabajo",
    "descripcion del trabajo",
    "calificaciones clave",
    "nice to have",
    "beneficios",
    "responsabilidades",
    "requisitos",
)


class GetOnBoardJobSource:
    """Discover Spanish/LATAM jobs from Get on Board."""

    def __init__(self, config: JobbotConfig | None = None) -> None:
        self.config = config or load_config()

    def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        raw_items = search_jobs_api(query.query, per_page=min(query.limit, 50))
        jobs = [job_from_api_item(item) for item in raw_items]
        # Remember portal
        remember_portal(
            self.config,
            url=f"{SITE_ORIGIN}/",
            ats_kind=AtsKind.GETONBOARD,
            notes="Get on Board (LATAM job board)",
        )
        return jobs

    def get_job(self, job_id: str) -> JobPosting:
        msg = "Use JobRepository; GetOnBoardJobSource.get_job is not supported"
        raise NotImplementedError(msg)


def search_jobs_api(query: str, *, per_page: int = 20, page: int = 1) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"query": query, "per_page": per_page, "page": page}
    )
    url = f"{API_SEARCH}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "jobbot/0.1 (local; GetOnBoard search)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — public API
        payload = json.load(resp)
    data = payload.get("data") or []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def slug_from_url(url: str) -> str | None:
    """Extract the public job slug from a Get on Board hard link."""
    raw = url.strip()
    if not raw.lower().startswith(("http://", "https://")):
        raw = "https://" + raw
    path = urlparse(raw).path or ""
    match = _JOB_PATH.match(path)
    return match.group(1) if match else None


def fetch_job_html(
    url: str,
    *,
    timeout: float = 30.0,
    on_chunk: Callable[[int, int | None], None] | None = None,
) -> str:
    """Download a public job page (no login). Raises OSError on transport failure.

    `on_chunk(bytes_read, total_or_none)` fires as the body arrives so a caller
    can show progress instead of going silent.
    """
    raw = url.strip()
    if not raw.lower().startswith(("http://", "https://")):
        raw = "https://" + raw
    req = urllib.request.Request(
        raw,
        headers={
            "User-Agent": "jobbot/0.1 (local; GetOnBoard job page)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            charset = resp.headers.get_content_charset() or "utf-8"
            length = resp.headers.get("Content-Length")
            total = int(length) if length and str(length).isdigit() else None
            parts: list[bytes] = []
            read = 0
            while True:
                block = resp.read(65536)
                if not block:
                    break
                parts.append(block)
                read += len(block)
                if on_chunk is not None:
                    on_chunk(read, total)
            return b"".join(parts).decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        msg = f"Get on Board returned HTTP {exc.code} for {raw}"
        raise OSError(msg) from exc


def parse_job_html(html: str, *, url: str) -> JobPosting:
    """Build a JobPosting from a public Get on Board HTML page (no invention)."""
    soup = BeautifulSoup(html, "html.parser")
    slug = slug_from_url(url) or ""
    title, company_from_h1 = _title_and_company_from_h1(soup)
    company = _company_from_links(soup) or company_from_h1 or _company_from_slug(slug)
    location, remote_mod = _location_from_page(soup)
    description = _description_from_page(soup)
    if not title:
        title = slug.replace("-", " ").title() if slug else "Role"
    if not company:
        company = "GetOnBoard company"
    langs = ["Spanish"]
    if re.search(r"\bingl[eé]s\b|\benglish\b", description, re.I):
        langs.append("English")
    public = url.strip()
    if not public.lower().startswith(("http://", "https://")):
        public = "https://" + public
    return JobPosting(
        id="PENDING",
        source="getonboard",
        source_job_id=slug or None,
        url=public,
        title=title,
        company=company,
        location=location,
        description=description,
        raw_description=description,
        skills=extract_skills_from_text(description),
        ats_url=public,
        ats_kind=AtsKind.GETONBOARD.value,
        language_requirements=langs,
        remote_type=remote_mod,
        note="hard link",
    )


def job_from_hard_link(
    url: str,
    *,
    html: str | None = None,
    fetch: Any | None = None,
    on_chunk: Callable[[int, int | None], None] | None = None,
) -> JobPosting:
    """Resolve a public hard link into a JobPosting (fixture HTML or live fetch)."""
    if html is not None:
        body = html
    elif fetch is not None:
        body = fetch(url)
    else:
        body = fetch_job_html(url, on_chunk=on_chunk)
    return parse_job_html(body, url=url)


def job_from_api_item(item: dict[str, Any]) -> JobPosting:
    attrs = item.get("attributes") or {}
    links = item.get("links") or {}
    slug = str(item.get("id") or "")
    title = str(attrs.get("title") or "Role").strip() or "Role"
    company = _company_name(item, attrs)
    location = _location(attrs)
    parts = [
        str(attrs.get("description_headline") or ""),
        str(attrs.get("description") or ""),
        str(attrs.get("functions_headline") or ""),
        str(attrs.get("functions") or ""),
        str(attrs.get("desirable") or ""),
    ]
    text = _strip_html("\n".join(parts))
    url = str(links.get("public_url") or "")
    if not url and slug:
        url = f"{SITE_ORIGIN}/jobs/{slug}"
    remote_mod = str(attrs.get("remote_modality") or "") or None
    if not remote_mod:
        remote_mod = "remote" if attrs.get("remote") is True else None
    langs = ["Spanish"] if str(attrs.get("lang") or "es").lower().startswith("es") else []
    if not langs:
        langs = ["Spanish"]  # GetOnBoard is Spanish-first LATAM board
    # English often mentioned in JD
    if (
        re.search(r"\bingl[eé]s\b|\benglish\b", text, re.I)
        and "English" not in langs
    ):
        langs.append("English")
    countries = attrs.get("countries") or []
    if isinstance(countries, list) and countries and not location:
        location = ", ".join(str(c) for c in countries)
    return JobPosting(
        id="PENDING",
        source="getonboard",
        source_job_id=slug or None,
        url=url,
        title=title,
        company=company,
        location=location,
        description=text,
        raw_description=text,
        skills=extract_skills_from_text(text),
        ats_url=url,
        ats_kind=AtsKind.GETONBOARD.value,
        language_requirements=langs,
        remote_type=remote_mod,
        note="spanish/LATAM board",
    )


def remember_portal(
    config: JobbotConfig,
    *,
    url: str,
    ats_kind: AtsKind | None = None,
    notes: str | None = None,
    registered: bool | None = None,
) -> None:
    """Upsert an employment platform into data/portals.yaml."""
    path = default_portals_path(config.root)
    registry = load_registry(path)
    domain = domain_from_url(url)
    if not domain:
        return
    kind = ats_kind or AtsKind.UNKNOWN
    if kind == AtsKind.UNKNOWN:
        from jobbot.portals.detect import detect_ats

        kind = detect_ats(url)
    registry.upsert(
        domain=domain,
        ats_kind=kind,
        example_url=url if "://" in url else f"https://{domain}/",
        notes=notes,
        registered=registered,
        seen=True,
    )
    save_registry(registry, path)
    logger.info("Portal learned: %s (%s)", domain, kind.value)


def remember_portal_from_url(
    config: JobbotConfig, url: str, *, notes: str | None = None
) -> str:
    """Learn any non-LinkedIn employment host from a URL. Returns domain."""
    from jobbot.portals.detect import detect_ats

    domain = domain_from_url(url)
    if not domain:
        return ""
    if "linkedin.com" in domain or domain.endswith("lnkd.in"):
        return domain
    remember_portal(config, url=url, ats_kind=detect_ats(url), notes=notes)
    return domain


_company_name_cache: dict[int, str] = {}


def _company_name(item: dict[str, Any], attrs: dict[str, Any]) -> str:
    if isinstance(attrs.get("company"), str) and attrs["company"].strip():
        return str(attrs["company"]).strip()
    company_ref = attrs.get("company")
    if isinstance(company_ref, dict):
        data = company_ref.get("data") or {}
        cid = data.get("id")
        if isinstance(cid, int) or (isinstance(cid, str) and str(cid).isdigit()):
            name = _fetch_company_name(int(cid))
            if name:
                return name
    # Prefer /companies/<slug> mentioned in HTML blobs
    blob = " ".join(
        str(attrs.get(k) or "")
        for k in ("projects", "description", "functions", "benefits")
    )
    match = re.search(
        r'getonbrd\.com/companies/([a-z0-9-]+)[^>]*>\s*([^<]+)',
        blob,
        re.I,
    )
    if match:
        slug = match.group(1)
        label = match.group(2).strip()
        if label and not re.search(
            r"conoce|nosotros|more about|about us|ver empresa", label, re.I
        ):
            return label
        return slug.replace("-", " ").title()
    match = re.search(r"getonbrd\.com/companies/([a-z0-9-]+)", blob, re.I)
    if match:
        return match.group(1).replace("-", " ").title()
    # Fallback: job id often ends with <company>-<city>-<hash>
    slug = str(item.get("id") or "")
    parts = slug.split("-")
    if len(parts) >= 3:
        # drop trailing hash-like token
        if re.fullmatch(r"[a-f0-9]{3,8}", parts[-1]):
            parts = parts[:-1]
        # drop trailing city-ish token if present
        cities = {
            "santiago",
            "mexico",
            "lima",
            "bogota",
            "remote",
            "chile",
            "colombia",
            "peru",
            "argentina",
            "us",
            "latam",
            "america",
        }
        if parts and parts[-1].casefold() in cities:
            parts = parts[:-1]
        if parts:
            return parts[-1].replace("_", " ").title()
    return "GetOnBoard company"


def _fetch_company_name(company_id: int) -> str | None:
    if company_id in _company_name_cache:
        return _company_name_cache[company_id]
    url = f"{SITE_ORIGIN}/api/v0/companies/{company_id}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "jobbot/0.1 (local; GetOnBoard company)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 — public API
            payload = json.load(resp)
    except OSError as exc:
        logger.debug("Company fetch failed for %s: %s", company_id, exc)
        return None
    name = str(
        ((payload.get("data") or {}).get("attributes") or {}).get("name") or ""
    ).strip()
    if name:
        _company_name_cache[company_id] = name
    return name or None


def _location(attrs: dict[str, Any]) -> str | None:
    countries = attrs.get("countries") or []
    country_line = ", ".join(str(c) for c in countries) if isinstance(countries, list) else None
    modality = attrs.get("remote_modality")
    bits = [b for b in (country_line, str(modality) if modality else None) if b]
    return " / ".join(bits) if bits else None


def _strip_html(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    # Without the newline the items glue together ('Python- Español') and the list
    # structure — the only thing that says 'these are separate requirements' — is lost.
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.I)
    text = re.sub(r"</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _title_and_company_from_h1(soup: BeautifulSoup) -> tuple[str, str | None]:
    h1 = soup.find("h1")
    if not isinstance(h1, Tag):
        return "", None
    raw = h1.get_text("\n", strip=True)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return "", None
    title = lines[0]
    company: str | None = None
    if len(lines) >= 3 and lines[1].casefold() in {"in", "en", "at"}:
        company = lines[2]
    elif len(lines) == 2 and lines[1].casefold() not in {"in", "en", "at"}:
        # "Title in Company" collapsed
        match = re.match(r"^(.+?)\s+(?:in|en|at)\s+(.+)$", raw, re.I | re.S)
        if match:
            title = _clean_text(match.group(1))
            company = _clean_text(match.group(2))
    return _clean_text(title), company


def _company_from_links(soup: BeautifulSoup) -> str | None:
    for anchor in soup.select('a[href*="/companies/"]'):
        href = str(anchor.get("href") or "")
        if "/companies/signup" in href or "/follow" in href:
            continue
        label = _clean_text(anchor.get_text(" ", strip=True))
        if not label:
            continue
        if re.search(r"conoce|nosotros|more about|about us|follow|employers", label, re.I):
            continue
        if len(label) > 80:
            continue
        return label
    return None


def _company_from_slug(slug: str) -> str | None:
    parts = slug.split("-")
    if len(parts) < 3:
        return None
    if re.fullmatch(r"[a-f0-9]{3,8}", parts[-1]):
        parts = parts[:-1]
    cities = {
        "santiago",
        "mexico",
        "lima",
        "bogota",
        "remote",
        "chile",
        "colombia",
        "peru",
        "argentina",
        "us",
        "latam",
        "america",
    }
    if parts and parts[-1].casefold() in cities:
        parts = parts[:-1]
    if not parts:
        return None
    return parts[-1].replace("_", " ").title()


def _location_from_page(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    remote_mod: str | None = None
    location: str | None = None
    for heading in soup.find_all("h2"):
        text = heading.get_text(" ", strip=True)
        if not text:
            continue
        folded = text.casefold()
        if "hybrid" in folded or "híbrid" in folded or "hibrid" in folded:
            remote_mod = "hybrid"
        elif "remote" in folded or "remoto" in folded:
            remote_mod = "remote"
        if "santiago" in folded or "chile" in folded or "lima" in folded:
            # First token before modality prose is usually the city.
            city = text.split("This job", 1)[0].split("Este", 1)[0].strip(" |")
            if city and len(city) < 60:
                location = city
            elif location is None:
                location = "Santiago" if "santiago" in folded else None
        if remote_mod or location:
            break
    return location, remote_mod


def _description_from_page(soup: BeautifulSoup) -> str:
    chunks: list[str] = []
    for heading in soup.find_all(["h2", "h3"]):
        title = _clean_text(heading.get_text(" ", strip=True))
        if not title:
            continue
        folded = title.casefold()
        if not any(key in folded for key in _SECTION_HEADINGS):
            continue
        block = heading.find_next(class_="gb-rich-txt")
        body = ""
        if isinstance(block, Tag):
            body = _strip_html(str(block))
        if not body:
            sibling = heading.find_next_sibling()
            if isinstance(sibling, Tag):
                body = _strip_html(str(sibling))
        if body:
            chunks.append(f"{title}\n{body}")
    if chunks:
        return "\n\n".join(chunks)
    # Fallback: concatenate every rich-text block on the page.
    rich = [
        _strip_html(str(block))
        for block in soup.select(".gb-rich-txt")
        if _strip_html(str(block))
    ]
    return "\n\n".join(rich)
