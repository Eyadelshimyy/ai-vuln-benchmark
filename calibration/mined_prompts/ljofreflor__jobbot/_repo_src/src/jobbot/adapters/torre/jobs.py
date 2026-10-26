"""Torre job source (LATAM / remote-first) via its public opportunity search.

Torre answers with structured fields instead of a prose posting: the skills carry
their own required experience, so requirements come from the payload and not from
parsing free text. The full description lives behind a client-rendered page, so a
posting here is a lead to open, not a complete JD.

The payload also lists the people behind the opportunity. Those are third parties,
so nothing under `members` ever reaches a JobPosting.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Protocol

from jobbot.config import JobbotConfig, load_config
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind, detect_ats

logger = logging.getLogger("jobbot.torre")

API_SEARCH = "https://search.torre.co/opportunities/_search/"
SITE_ORIGIN = "https://torre.ai"
MAX_SIZE = 50

# Torre's ``remote: {term: True}`` still returns office roles; drop these locally.
_ONSITE_LOCATION_TYPES = frozenset(
    {
        "on_site",
        "onsite",
        "physical_location",
    }
)


class Fetcher(Protocol):
    """POST a JSON body, get JSON back. Injectable so tests stay offline."""

    def post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]: ...


class UrllibFetcher:
    def post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "User-Agent": "jobbot/0.1 (local; Torre search)",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 — public API
            payload = json.load(resp)
        return payload if isinstance(payload, dict) else {}


class TorreJobSource:
    """Discover LATAM / remote opportunities from Torre."""

    def __init__(
        self,
        config: JobbotConfig | None = None,
        *,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.config = config or load_config()
        self.fetcher = fetcher or UrllibFetcher()

    def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        items = search_opportunities(
            query.query,
            size=min(query.limit, MAX_SIZE),
            remote=query.remote or None,
            fetcher=self.fetcher,
        )
        if query.remote:
            items = [item for item in items if is_remote_opportunity(item)]
        jobs = [job_from_api_item(item) for item in items]
        self._remember_portal()
        return jobs

    def get_job(self, job_id: str) -> JobPosting:
        msg = "Use JobRepository; TorreJobSource.get_job is not supported"
        raise NotImplementedError(msg)

    def _remember_portal(self) -> None:
        from jobbot.adapters.getonboard.jobs import remember_portal

        remember_portal(
            self.config,
            url=f"{SITE_ORIGIN}/",
            ats_kind=AtsKind.TORRE,
            notes="Torre (LATAM / remote job board)",
        )


def search_opportunities(
    query: str,
    *,
    size: int = 20,
    offset: int = 0,
    remote: bool | None = None,
    fetcher: Fetcher | None = None,
) -> list[dict[str, Any]]:
    """Open opportunities matching a role or skill, newest ranking first."""
    params = urllib.parse.urlencode({"size": max(1, min(size, MAX_SIZE)), "offset": offset})
    body: dict[str, Any] = {
        "skill/role": {"text": query, "experience": "potential-to-develop"},
    }
    if remote:
        body["remote"] = {"term": True}
    client = fetcher or UrllibFetcher()
    payload = client.post_json(f"{API_SEARCH}?{params}", body)
    results = payload.get("results") or []
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def is_remote_opportunity(item: dict[str, Any]) -> bool:
    """True when the payload shows remote work, not an office / physical seat.

    Used after ``remote: {term: True}`` because Torre still returns
    ``physical_location`` / ``on_site`` rows that must not reach ``--remote``.
    """
    place: dict[str, Any] = {}
    raw_place = item.get("place")
    if isinstance(raw_place, dict):
        place = raw_place
    kind = str(place.get("locationType") or "").strip().casefold().replace("-", "_")
    if kind in _ONSITE_LOCATION_TYPES:
        return False
    if place.get("anywhere") is True or place.get("remote") is True:
        return True
    if item.get("remote") is True:
        return True
    return kind.startswith("remote") or kind in {
        "hybrid",
        "remote_anywhere",
        "remote_countries",
    }


def job_from_api_item(item: dict[str, Any]) -> JobPosting:
    """Map one opportunity. Never reads `members`: those are other people."""
    opportunity_id = str(item.get("id") or "").strip()
    title = str(item.get("objective") or "").strip() or "Role"
    skills = _skills(item)
    ats_kind = AtsKind.TORRE
    url = _public_url(opportunity_id, str(item.get("slug") or ""))
    external = _external_application_url(item)
    if external:
        url = external
        ats_kind = detect_ats(external)
    text = _description(item, title=title, skills=skills)
    return JobPosting(
        id="PENDING",
        source="torre",
        source_job_id=opportunity_id or None,
        url=url,
        title=title,
        company=_company(item),
        location=_location(item),
        description=text,
        raw_description=text,
        skills=skills,
        ats_url=url,
        ats_kind=ats_kind.value,
        employment_type=_employment_type(item),
        remote_type=_remote_type(item),
        posted_at=_posted_at(item),
        note="torre: summary only, open the post for the full description",
    )


def _external_application_url(item: dict[str, Any]) -> str | None:
    """Where the posting is applied to, when that is not Torre itself.

    The API has sent this both as a string ``external`` and as ``external: true``
    beside ``externalApplicationUrl``.
    """
    for key in ("externalApplicationUrl", "external"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return None


def _public_url(opportunity_id: str, slug: str) -> str:
    if not opportunity_id:
        return f"{SITE_ORIGIN}/"
    tail = f"{opportunity_id}-{slug}" if slug else opportunity_id
    return f"{SITE_ORIGIN}/post/{tail}"


def _company(item: dict[str, Any]) -> str:
    organizations = item.get("organizations")
    if isinstance(organizations, list):
        for org in organizations:
            if isinstance(org, dict):
                name = str(org.get("name") or "").strip()
                if name:
                    return name
    return "Torre opportunity"


def _skills(item: dict[str, Any]) -> list[str]:
    """The payload states each requirement, so no text mining is needed."""
    out: list[str] = []
    raw = item.get("skills")
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _description(item: dict[str, Any], *, title: str, skills: list[str]) -> str:
    """Compose what the payload actually says; do not invent a posting body."""
    lines = [title]
    tagline = str(item.get("tagline") or "").strip()
    if tagline:
        lines.append(tagline)
    if str(item.get("status") or "").strip().casefold() == "closed":
        lines.append("Estado: cerrada — no se reciben más postulantes")
    deadline = str(item.get("deadline") or "").strip()[:10]
    if deadline:
        lines.append(f"Fecha límite: {deadline}")
    place = _location(item)
    if place:
        lines.append(f"Modalidad: {place}")
    countries = _residence_countries(item)
    if countries:
        lines.append(f"Debe residir en: {countries}")
    lines.extend(f"Idioma: {entry}" for entry in _languages(item))
    if _compensation_hidden(item):
        lines.append("Compensación: no publicada")
    else:
        money = _compensation(item)
        if money:
            lines.append(f"Compensación: {money}")
    if skills:
        lines.append("")
        lines.extend(_skill_lines(item, skills))
    return "\n".join(lines).strip()


def _skill_lines(item: dict[str, Any], skills: list[str]) -> list[str]:
    experience_by_name: dict[str, str] = {}
    raw = item.get("skills")
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                name = str(entry.get("name") or "").strip()
                exp = str(entry.get("experience") or "").strip()
                if name and exp:
                    experience_by_name[name] = exp.replace("-", " ")
    lines = []
    for name in skills:
        needed = experience_by_name.get(name)
        lines.append(f"- {name} ({needed})" if needed else f"- {name}")
    return lines


def _residence_countries(item: dict[str, Any]) -> str:
    """`remote_countries` limits where the candidate may live, not where an office is."""
    place = item.get("place")
    if not isinstance(place, dict):
        return ""
    if str(place.get("locationType") or "").strip().casefold() != "remote_countries":
        return ""
    return _place_names(place.get("location"))


def _languages(item: dict[str, Any]) -> list[str]:
    """'English (fully fluent)': the language and the fluency the payload asks for."""
    raw = item.get("languages")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        language = entry.get("language")
        name = language.get("name") if isinstance(language, dict) else language
        name = str(name or "").strip()
        fluency = str(entry.get("fluency") or "").strip().replace("-", " ")
        if name:
            out.append(f"{name} ({fluency})" if fluency else name)
    return out


def _compensation_hidden(item: dict[str, Any]) -> bool:
    data = item.get("compensation")
    return isinstance(data, dict) and data.get("visible") is False


def _compensation(item: dict[str, Any]) -> str | None:
    data = item.get("compensation")
    if not isinstance(data, dict):
        return None
    inner = data.get("data")
    if not isinstance(inner, dict):
        return None
    currency = str(inner.get("currency") or "").strip()
    low = _amount(inner.get("minAmount"))
    high = _amount(inner.get("maxAmount"))
    periodicity = str(inner.get("periodicity") or "").strip()
    if low is None and high is None:
        return None
    if not low and not high:
        return "a convenir" if inner.get("negotiable") else None
    span = f"{low or high}" if low == high or not high else f"{low}–{high}"
    bits = [b for b in (currency, span, periodicity) if b]
    return " ".join(bits) or None


def _amount(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _location(item: dict[str, Any]) -> str | None:
    named = _place_names(item.get("locations")) or _place_names(
        (item.get("place") or {}).get("location") if isinstance(item.get("place"), dict) else None
    )
    if named:
        return named
    place = item.get("place")
    if isinstance(place, dict):
        if place.get("anywhere") is True:
            return "Remote / anywhere"
        kind = str(place.get("locationType") or "").strip()
        if kind:
            return kind.replace("_", " ")
    return None


def _place_names(raw: Any) -> str:
    """Places arrive either as plain names or as objects keyed by `id`."""
    if not isinstance(raw, list):
        return ""
    names: list[str] = []
    for entry in raw:
        name = str(entry.get("id") or "") if isinstance(entry, dict) else str(entry)
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names)


def _remote_type(item: dict[str, Any]) -> str | None:
    place = item.get("place")
    if isinstance(place, dict):
        if place.get("anywhere") is True:
            return "fully_remote"
        kind = str(place.get("locationType") or "").strip()
        if kind:
            return kind
    if item.get("remote") is True:
        return "remote"
    return None


def _employment_type(item: dict[str, Any]) -> str | None:
    for key in ("type", "commitment"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return None


def _posted_at(item: dict[str, Any]) -> datetime | None:
    """Torre dates its opportunities, so freshness filters work on this source."""
    raw = str(item.get("created") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparsable Torre date: %s", raw)
        return None
