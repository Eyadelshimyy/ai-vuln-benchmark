"""Parse LinkedIn recruiter posts into job-shaped records (no invention)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from jobbot.jobs.geo import country_allows, detect_country, normalize_countries
from jobbot.jobs.normalization import fold_text
from jobbot.jobs.parsing import extract_skills_from_text
from jobbot.models.job import JobPosting
from jobbot.portals.detect import (
    JOB_BOARD_KINDS,
    AtsKind,
    detect_ats,
    extract_http_urls,
    first_external_ats_url,
    sniff_ats,
)
from jobbot.portals.email_apply import first_apply_email, mailto_url
from jobbot.portals.message_apply import asks_for_linkedin_message
from jobbot.portals.redirect import expand_url_map

# Roles / themes relevant to this candidate's data career (title-agnostic filter)


# What makes a post a job post is hiring language and a way to apply — not the
# field being hired for. A list of roles only ever recognises one career.
_HIRING_HINTS = (
    "hiring",
    "we are hiring",
    "we're hiring",
    "contratando",
    "reclutando",
    "buscamos",
    "estoy buscando",
    "estamos buscando",
    "seguimos buscando",
    "sigo buscando",
    "se busca",
    "se necesita",
    "looking for",
    "join our team",
    "únete",
    "unete",
    "vacante",
    "vacantes",
    "vacancy",
    "vacancies",
    "oferta laboral",
    "ofertas laborales",
    "oportunidad laboral",
    "oportunidades laborales",
    "convocatoria",
    "búsqueda laboral",
    "busqueda laboral",
)
_APPLY_HINTS = (
    "postula",
    "postular",
    "apply",
    "enviar cv",
    "envía tu cv",
    "envia tu cv",
    "send your cv",
    "send your resume",
    "send your materials",
    "cargo:",
    "puesto:",
    "role:",
)


def looks_like_job_post(text: str) -> bool:
    """A post is a job post when it says it is hiring or how to apply."""
    lowered = text.casefold()
    if any(hint in lowered for hint in (*_HIRING_HINTS, *_APPLY_HINTS)):
        return True
    return asks_for_linkedin_message(text)


@dataclass(frozen=True)
class PostVacancy:
    """One labelled vacancy link inside a recruiter post ("Lead Data Scientist: <url>")."""

    title: str
    url: str
    ats_kind: AtsKind = AtsKind.UNKNOWN
    country: str | None = None


@dataclass(frozen=True)
class LinkedInPostCandidate:
    """One recruiter post before it becomes a JobPosting."""

    post_id: str
    text: str
    author: str | None = None
    post_url: str | None = None
    ats_url: str | None = None
    ats_kind: AtsKind = AtsKind.UNKNOWN
    posted_at: datetime | None = None
    vacancies: tuple[PostVacancy, ...] = ()


# LinkedIn activity ids carry their creation time in the high bits: ms = id >> 22.
_ACTIVITY_ID_RE = re.compile(r"(?:activity|share|ugcPost)[-:](\d{15,25})", re.IGNORECASE)
_LINKEDIN_EPOCH_START = datetime(2005, 1, 1, tzinfo=UTC)


def posted_at_from_url(url: str | None) -> datetime | None:
    """
    Publication date of a post, read from the activity id in its own URL.

    Works for `/posts/…-activity-<id>-xxxx`, `/posts/…-share-<id>-xxxx` and
    `feed/update/urn:li:{activity,share,ugcPost}:<id>`, which share the same id encoding.
    Returns None for anything else (e.g. the author-profile fallback), so a post is
    never given an invented date.
    """
    match = _ACTIVITY_ID_RE.search(url or "")
    if match is None:
        return None
    try:
        moment = datetime.fromtimestamp((int(match.group(1)) >> 22) / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if moment < _LINKEDIN_EPOCH_START:
        return None
    return moment


_FEED_HEADERS = ("publicación en el feed", "publicacion en el feed", "feed post")
_CHROME_LINE = re.compile(
    r"^(?:•\s*\d+\S*|seguir|follow|recomendar|editado.*"
    r"|\d+\s*(?:min|h|d|sem|mes(?:es)?|año(?:s)?)\b.*)$",
    re.IGNORECASE,
)
_CHROME_WINDOW = 5

# Engagement UI around the post: exact lines LinkedIn renders, never post content.
_ENGAGEMENT_LINE = re.compile(
    r"^(?:\d+[\d.,\s]*"
    r"(?:reacci(?:ón|on|ones)|comentario(?:s)?|comment(?:s)?|reaction(?:s)?|"
    r"veces compartido|compartido(?:s)?|repost(?:s)?)"
    r"|recomendar|comentar|compartir|enviar|like|comment|share|send|repost"
    r"|ver traducción|ver traduccion|see translation"
    r"|\d+[\d.,\s]*$"
    r"|.{0,40}\ben linkedin\.com$)$",
    re.IGNORECASE,
)


def strip_engagement_chrome(lines: list[str]) -> list[str]:
    """Drop reaction / comment / share lines that LinkedIn paints around a post."""
    return [line for line in lines if not _ENGAGEMENT_LINE.match(line.strip())]


def strip_feed_chrome(text: str) -> tuple[str | None, str]:
    """
    Split LinkedIn's card header from the actual post body.

    Returns (author, body). Author is the name LinkedIn shows above the post;
    only the first few lines are cleaned so bullet content in the post survives,
    while engagement lines are dropped wherever they appear.
    """
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0].casefold() not in _FEED_HEADERS:
        return None, "\n".join(strip_engagement_chrome(lines)).strip()
    rest = [line for line in lines[1:] if line]
    if not rest:
        return None, ""
    author = rest[0]
    head = [line for line in rest[1 : 1 + _CHROME_WINDOW] if not _CHROME_LINE.match(line)]
    body = strip_engagement_chrome(head + rest[1 + _CHROME_WINDOW :])
    return author, "\n".join(body).strip()


# A post offering several roles lists them as "🔹 Lead Data Scientist: <url>". Cards keep one
# bullet per line, but a collapsed paragraph puts them all on one, so split on both.
_VACANCY_SEGMENT_RE = re.compile(r"[\r\n]+|(?=[🔹🔸🔷🔶🟢🟣▪◾◽•·‧‣])")
_VACANCY_LINE_RE = re.compile(
    r"^(?P<label>[^\n]{4,90}?)\s*[:\-–—]\s*(?P<url>https?://\S+)",
)
_VACANCY_LABEL_TRIM_RE = re.compile(r"^[^\w(¿¡]+|[\s\-–—:]+$")

# Words that make a label a job title. Without one, a labelled link is just a link
# ("Learn more: …", "Our careers page: …") and never becomes a vacancy.
_ROLE_WORDS = (
    "scientist",
    "engineer",
    "consultant",
    "analyst",
    "developer",
    "architect",
    "specialist",
    "researcher",
    "manager",
    "administrator",
    "cient[íi]fic",
    "ingenier",
    "analista",
    "desarrollador",
    "arquitect",
    "especialista",
    "investigador",
    "jefe",
    "l[íi]der",
)
_ROLE_WORD_RE = re.compile("|".join(_ROLE_WORDS), re.IGNORECASE)


def labelled_vacancy_links(text: str) -> list[tuple[str, str]]:
    """(role label, link) for every vacancy the recruiter listed, before any resolution."""
    out: list[tuple[str, str]] = []
    for segment in _VACANCY_SEGMENT_RE.split(text or ""):
        match = _VACANCY_LINE_RE.match(segment.strip())
        if match is None:
            continue
        label = _VACANCY_LABEL_TRIM_RE.sub("", match.group("label")).strip()
        if not label or not _ROLE_WORD_RE.search(label):
            continue
        out.append((label, match.group("url").rstrip(".,;:)")))
    return out


def split_vacancy_links(text: str, url_map: dict[str, str] | None = None) -> list[PostVacancy]:
    """
    Labelled vacancy links in a post, in the order the recruiter listed them.

    ``url_map`` carries each link's resolved destination. A link that still points at
    LinkedIn is dropped: the post's own roles are only appliable through the external
    page, and storing a lnkd.in URL would teach the portal registry a shortener.
    """
    resolved_by_raw = url_map or {}
    out: list[PostVacancy] = []
    seen: set[str] = set()
    for label, raw_url in labelled_vacancy_links(text):
        url = resolved_by_raw.get(raw_url, raw_url)
        kind = detect_ats(url)
        if kind == AtsKind.LINKEDIN or _is_linkedin_shortener(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(PostVacancy(title=label, url=url, ats_kind=kind, country=detect_country(label)))
    return out


def _is_linkedin_shortener(url: str) -> bool:
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").casefold()
    return host == "lnkd.in" or host.endswith(".lnkd.in")


def post_offers_wanted_country(
    text: str,
    *,
    wanted: Sequence[str] = (),
    allow_remote: bool = True,
) -> bool:
    """
    Whether a post is worth keeping for the countries we want to work in.

    A weekly roundup advertises one country per role ("… – Colombia", "… – Chile"), so the
    post as a whole reads as foreign and ``country_allows`` rejects it — taking the wanted
    role down with it. One role explicitly labelled with a wanted country keeps the post;
    the roles themselves are filtered later, in ``post_to_jobs``.
    """
    if country_allows(text, wanted=wanted, allow_remote=allow_remote):
        return True
    codes = normalize_countries(wanted)
    return any(detect_country(label) in codes for label, _url in labelled_vacancy_links(text))


def _vacancy_wanted(vacancy: PostVacancy, codes: tuple[str, ...]) -> bool:
    """A role is out only when its own label places it in a country we did not ask for."""
    return not codes or vacancy.country is None or vacancy.country in codes


def parse_post_blob(
    blob: str,
    *,
    author: str | None = None,
    post_url: str | None = None,
    mailto_urls: Sequence[str] | None = None,
    extra_urls: Sequence[str] | None = None,
) -> LinkedInPostCandidate:
    """
    Parse a single post body (fixture or scraped text).

    `extra_urls` are the card's anchors: they feed ATS detection without entering
    the description, which must read like the advert the recruiter wrote.
    """
    header_author, text = strip_feed_chrome(blob.strip())
    author = author or header_author
    url_map = expand_url_map(extract_http_urls(text) + [u for u in (extra_urls or []) if u])
    urls = _both_ends(url_map)
    vacancies = split_vacancy_links(text, url_map)
    ats_url, ats_kind = first_external_ats_url(urls)
    if ats_url is not None and ats_kind == AtsKind.UNKNOWN:
        # careers.example.cl and friends: host says nothing, the page still names its ATS.
        ats_kind = sniff_ats(ats_url)
    if ats_url is None:
        email = first_apply_email(text)
        if email:
            ats_url = mailto_url(email)
            ats_kind = AtsKind.EMAIL
    if ats_url is None and mailto_urls:
        # Address only reachable through the post's mailto link, not its text.
        first = next((m for m in mailto_urls if "@" in m), None)
        if first:
            ats_url = first if first.startswith("mailto:") else mailto_url(first)
            ats_kind = AtsKind.EMAIL
    if ats_url is None and post_url and asks_for_linkedin_message(text):
        # The permalink (or author-profile fallback) is the apply surface.
        ats_url = post_url
        ats_kind = AtsKind.LINKEDIN
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]  # noqa: S324 — id only
    post_id = digest
    if post_url:
        post_id = hashlib.sha1(post_url.encode("utf-8")).hexdigest()[:12]  # noqa: S324
    return LinkedInPostCandidate(
        post_id=post_id,
        text=text,
        author=author,
        post_url=post_url,
        ats_url=ats_url,
        ats_kind=ats_kind,
        posted_at=posted_at_from_url(post_url),
        vacancies=tuple(vacancies),
    )


def _both_ends(url_map: dict[str, str]) -> list[str]:
    """Raw and resolved URLs, deduped, in the order they appeared."""
    out: list[str] = []
    seen: set[str] = set()
    for raw, resolved in url_map.items():
        for candidate in (raw, resolved):
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def parse_posts_fixture(text: str) -> list[LinkedInPostCandidate]:
    """Split a multi-post fixture separated by '---' lines."""
    chunks = re.split(r"\n-{3,}\n", text.strip())
    out: list[LinkedInPostCandidate] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        author = None
        post_url = None
        body_lines: list[str] = []
        for line in chunk.splitlines():
            if line.lower().startswith("author:"):
                author = line.split(":", 1)[1].strip()
            elif line.lower().startswith("url:"):
                post_url = line.split(":", 1)[1].strip()
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if body:
            out.append(parse_post_blob(body, author=author, post_url=post_url))
    return out


def _job_from_post_body(post: LinkedInPostCandidate, *, job_id: str = "PENDING") -> JobPosting:
    """Whole-post JobPosting when the advert has no labelled vacancy links."""
    title = _guess_title(post.text) or "Role from LinkedIn post"
    company = employer_from_post(post.text, post.ats_url, author=post.author)
    note = None
    if post.ats_url:
        note = f"ats={post.ats_kind.value}"
    return JobPosting(
        id=job_id,
        source="linkedin_post",
        source_job_id=post.post_id,
        url=post.post_url,
        title=title,
        company=company,
        description=post.text,
        raw_description=post.text,
        posted_at=post.posted_at,
        skills=extract_skills_from_text(post.text),
        ats_url=post.ats_url,
        ats_kind=post.ats_kind.value if post.ats_url else None,
        note=note,
    )


def _job_from_vacancy(
    post: LinkedInPostCandidate,
    vacancy: PostVacancy,
    *,
    company: str,
    skills: list[str],
) -> JobPosting:
    """One appliable role: vacancy title/url win; permalink stays in ``note``."""
    note = f"ats={vacancy.ats_kind.value}"
    if post.post_url:
        note = f"{note} post={post.post_url}"
    return JobPosting(
        id="PENDING",
        source="linkedin_post",
        source_job_id=_vacancy_source_id(vacancy.url),
        url=vacancy.url,
        title=vacancy.title,
        company=company,
        description=post.text,
        raw_description=post.text,
        posted_at=post.posted_at,
        skills=skills,
        ats_url=vacancy.url,
        ats_kind=vacancy.ats_kind.value,
        note=note,
    )


def post_to_jobs(
    post: LinkedInPostCandidate,
    *,
    countries: Sequence[str] = (),
) -> list[JobPosting]:
    """
    One JobPosting per labelled vacancy, or a single job for the whole post.

    A recruiter offering N roles behind N links is N applications: collapsing them
    leaves N-1 unreachable, since a job carries one apply URL. The same rule holds
    for exactly one labelled vacancy — its title/url/ats win over the permalink.

    ``countries`` drops the roles whose own label names a country we did not ask for:
    a Latam roundup lists one country per role, so the choice belongs to each role and
    not to the post. Posts with no labelled vacancies keep the whole-post mapping
    (country filtering for those stays at the post level, via ``post_offers_wanted_country``).
    """
    if not post.vacancies:
        return [_job_from_post_body(post)]
    wanted = [v for v in post.vacancies if _vacancy_wanted(v, normalize_countries(countries))]
    if not wanted:
        return []
    company = employer_from_post(post.text, wanted[0].url, author=post.author)
    skills = extract_skills_from_text(post.text)
    return [_job_from_vacancy(post, vacancy, company=company, skills=skills) for vacancy in wanted]


def post_to_job(post: LinkedInPostCandidate, *, job_id: str = "PENDING") -> JobPosting:
    """Map a post to one JobPosting (first vacancy, or the whole post when none)."""
    jobs = post_to_jobs(post)
    job = jobs[0]
    if job_id == job.id:
        return job
    return job.model_copy(update={"id": job_id})


def vacancy_dedupe_key(job: JobPosting) -> str | None:
    """Same apply target + title ⇒ same vacancy, even when reposted by others."""
    target = (job.ats_url or "").strip().casefold()
    if not target:
        return None
    title = fold_text(job.title or "")
    if not title:
        return None
    return f"{target}|{title}"


def dedupe_jobs_by_apply_target(jobs: list[JobPosting]) -> list[JobPosting]:
    """Keep the earliest (or first-seen) post per apply target + title."""
    winners: dict[str, JobPosting] = {}
    order: list[str] = []
    passthrough: list[JobPosting] = []
    for job in jobs:
        key = vacancy_dedupe_key(job)
        if key is None:
            passthrough.append(job)
            continue
        existing = winners.get(key)
        if existing is None:
            winners[key] = job
            order.append(key)
            continue
        # Prefer the older posting when both carry a real date.
        if (
            job.posted_at is not None
            and existing.posted_at is not None
            and job.posted_at < existing.posted_at
        ):
            winners[key] = job
        # else keep existing (first-seen / older)
    return [winners[key] for key in order] + passthrough


def _vacancy_source_id(url: str) -> str:
    """Keyed on the vacancy page, so the same role found twice stays one job."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]  # noqa: S324 — id only


_GENDERED_ARTICLE_RE = re.compile(r"^(?:un\(a\)|un/a)\s+", re.IGNORECASE)


def _guess_title(text: str) -> str | None:
    patterns = [
        (
            # Articles must be their own word. Otherwise "(?i)a" eats the A of "Applied".
            # "Estoy buscando" / "seguimos buscando" are the first-person forms recruiters
            # use instead of "buscamos". Stop before "para empresa …" so a length cap
            # cannot leave "para emp". Gendered "un(a)" is not a Latin article token.
            r"(?i)(?:hiring|buscamos|estoy buscando|estamos buscando|"
            r"seguimos buscando|sigo buscando|"
            r"looking for|we(?:'re| are) looking for)\s+"
            r"(?:(?:a|an|un|una|un\(a\)|un/a)\s+)?([^\n.!?;]{8,160}?)"
            r"(?=\s+para\s+empresa\b|[;.!?\n]|$)"
        ),
        r"(?i)(?:role|puesto|cargo|posici[oó]n|vacante)\s*[:\-]\s*([^\n]{5,80})",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip(" -:")
            title = _GENDERED_ARTICLE_RE.sub("", title)
            title = _TITLE_LEAD_IN_RE.sub("", title).strip(" -:,")
            title = _strip_location_bars(title)
            return title[:120] or None
    return None


def _strip_location_bars(title: str) -> str:
    """'Role – Specialty | Chile | Híbrido' keeps the role, drops the place bars."""
    parts = re.split(r"\s*\|\s*", title)
    if len(parts) < 2:
        return title
    if any(detect_country(part) for part in parts[1:]):
        return parts[0].strip(" -:")
    return title


# "Buscamos talento | Machine Learning & Customer Analytics": the word after the verb says
# whom the recruiter wants, not what the job is. The role is on the other side of the bar.
_TITLE_LEAD_IN_RE = re.compile(
    r"^(?:talento|talent|gente|personas?|profesionales?|professionals?|candidatos?)"
    r"\s*[|\-–—:]\s*",
    re.IGNORECASE,
)


# "En NTT DATA buscamos …", "Expert Analyst @ NTT Data": the employer follows the preposition.
# The word must start a word itself, or the "en" inside "when Chile" names a company.
_COMPANY_LEAD_RE = re.compile(r"(?:^|[\s(])(?:[Aa]t|[Ee]n|@)\s+(?P<tail>[^\n.!?,;]{1,80})")
_COMPANY_TOKEN_RE = re.compile(r"[A-ZÁÉÍÓÚÑ0-9][\w&.\-]*$")
_COMPANY_CONNECTORS: frozenset[str] = frozenset({"&", "de", "del", "la", "las", "los", "of", "y"})

# "Coinvestigador EN INTELIGENCIA ARTIFICIAL": the field after EN is not an employer.
_FIELD_NOT_EMPLOYER: frozenset[str] = frozenset(
    {
        "artificialintelligence",
        "cienciadedatos",
        "computacion",
        "computación",
        "computerscience",
        "datascience",
        "deeplearning",
        "inteligenciaartificial",
        "machinelearning",
        "softwareengineering",
    }
)


def _guess_company(text: str) -> str | None:
    """
    First name-shaped token run after at/en/@ that is not a field or a place.

    Titles like "BUSCAMOS COINVESTIGADOR(A) EN INTELIGENCIA ARTIFICIAL" hit the
    preposition before the real line ("En FST NEGOCIOS – Centro…"); those false
    leads are skipped so the employer is not the research field.
    """
    for match in _COMPANY_LEAD_RE.finditer(text or ""):
        name = _company_name_prefix(match.group("tail"))
        if name and not _looks_like_field(name) and not _looks_like_place(name):
            return name[:80]
    return None


def _company_name_prefix(tail: str) -> str | None:
    """
    Leading run of name-shaped words: "NTT DATA buscamos 4 profesionales" → "NTT DATA".

    A company name is written in capitals; the first lowercase word after it is the
    recruiter's prose, so the name ends there instead of eating the rest of the sentence.
    """
    words: list[str] = []
    for word in tail.split():
        if _COMPANY_TOKEN_RE.match(word) or (words and word.casefold() in _COMPANY_CONNECTORS):
            words.append(word)
            continue
        break
    while words and words[-1].casefold() in _COMPANY_CONNECTORS:
        words.pop()
    return " ".join(words) or None


# Legal forms and industry words: alone they never identify one employer.
_GENERIC_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "chile",
        "colombia",
        "consulting",
        "corp",
        "digital",
        "global",
        "group",
        "labs",
        "latam",
        "llc",
        "ltda",
        "services",
        "solutions",
        "technologies",
        "technology",
    }
)

_PLACE_NAME_FLAT: frozenset[str] = frozenset(
    {
        "barcelona",
        "bogota",
        "lima",
        "madrid",
        "santiago",
    }
)


def _flatten_name(name: str) -> str:
    from jobbot.companies.urls import slugify

    return slugify(name).replace("-", "")


def _looks_like_field(name: str) -> bool:
    return _flatten_name(name) in _FIELD_NOT_EMPLOYER


def _looks_like_place(name: str) -> bool:
    """'en Santiago' / 'en Perú' must not become the employer."""
    if detect_country(name) is not None:
        return True
    flattened = _flatten_name(name)
    return flattened in _GENERIC_NAME_TOKENS or flattened in _PLACE_NAME_FLAT


# Consumer / free-mail hosts: never treat these as the hiring company.
_FREE_MAIL_LABELS: frozenset[str] = frozenset(
    {
        "gmail",
        "googlemail",
        "outlook",
        "hotmail",
        "live",
        "yahoo",
        "ymail",
        "icloud",
        "me",
        "proton",
        "protonmail",
        "aol",
        "gmx",
        "mail",
    }
)

# Placeholder / generic second-level domains in fixtures and anonymized posts.
_GENERIC_DOMAIN_LABELS: frozenset[str] = frozenset(
    {
        "empresa",
        "company",
        "correo",
        "email",
        "ejemplo",
        "example",
        "test",
        "domain",
        "cliente",
        "client",
        "contacto",
        "info",
    }
)


def employer_from_post(text: str, apply_url: str | None, *, author: str | None) -> str:
    """
    Employer behind a recruiter post, preferring corroborated evidence.

    Whoever writes a hiring post is usually a recruiter, not the company, so the author
    is a poor employer name — but it is the only one available most of the time. When the
    post names a company *and* the apply URL carries that name, the two agree and the
    employer is a fact. A hiring mailbox on a company domain (postulaciones@empresa.cl)
    is stronger than the author's name when the body never says "En Empresa…".
    """
    named = _guess_company(text)
    if named and _post_names_employer(named, text, apply_url):
        return named
    from_url = _company_from_apply_url(apply_url)
    if from_url:
        return from_url
    if _unnamed_client(text):
        # Headhunter post: "una compañía líder" is not a name, and neither is the author.
        return "Unknown company"
    return author or named or "Unknown company"


_UNNAMED_CLIENT_RE = re.compile(
    r"(?i)(?:una?\s+)?(?:compa[ñn][ií]a|empresa|company|cliente?)\s+"
    r"l[ií]der(?:\s+de\s+la\s+regi[oó]n)?"
    r"|cliente\s+confidencial"
    r"|confidential\s+(?:client|company)"
    r"|unnamed\s+(?:company|client)",
)


def _unnamed_client(text: str) -> bool:
    """Whether the post says the employer is unnamed rather than naming one."""
    return bool(_UNNAMED_CLIENT_RE.search(text or ""))


def _company_from_apply_url(url: str | None) -> str | None:
    """Brand-shaped label from a mailto/HTTPS apply host, or None for free mail."""
    if not url:
        return None
    raw = url.strip()
    if raw.casefold().startswith("mailto:"):
        host = raw.partition("@")[2].split("?", 1)[0].strip().rstrip(">")
    else:
        # A LinkedIn permalink used as the DM apply surface is not the employer.
        kind = detect_ats(raw)
        if kind in JOB_BOARD_KINDS:
            return None
        host = urlparse(raw if "://" in raw else f"https://{raw}").hostname or ""
    host = host.casefold().removeprefix("www.")
    if not host or "." not in host:
        return None
    # careers.empresa.cl → empresa.cl; jobs.ejemplo.com → ejemplo.com
    labels = host.split(".")
    if labels[0] in {"careers", "career", "jobs", "empleo", "empleos", "mail", "www"}:
        labels = labels[1:]
    if len(labels) < 2:
        return None
    # empresa.com.ar → empresa; ejemplo.cl → ejemplo
    if len(labels) >= 3 and ".".join(labels[-2:]) in {
        "com.ar",
        "com.br",
        "com.co",
        "com.mx",
        "com.pe",
        "com.uy",
        "co.uk",
    }:
        sld = labels[-3]
    else:
        sld = labels[-2]
    if (
        sld in _FREE_MAIL_LABELS
        or sld in _GENERIC_DOMAIN_LABELS
        or sld in _GENERIC_NAME_TOKENS
        or len(sld) < 3
    ):
        return None
    # Drop common "inc/corp" packing in the label without inventing a prettier brand.
    brand = re.sub(r"(inc|corp|llc)$", "", sld, flags=re.I)
    brand = brand or sld
    return brand[0].upper() + brand[1:]


def _post_names_employer(company: str, text: str, apply_url: str | None) -> bool:
    """
    Whether the post backs up the name ``_guess_company`` read out of one sentence.

    The apply host is the strongest witness, but many posts keep the link in the first
    comment, out of reach. There a hashtag spelling the same name (#NTTDATA next to
    "En NTT DATA buscamos") is what tells a company apart from the "en Santiago" the
    preposition rule also matches — the post still has to say it. An org dash
    ("En FST NEGOCIOS – Centro de I+D+i") or a hiring verb right after the name
    is the same kind of witness when the post has neither link nor hashtag.
    """
    if apply_url and _apply_url_names(apply_url, company):
        return True
    if _hashtag_names(text, company):
        return True
    return _org_context_names(text, company)


def _org_context_names(text: str, company: str) -> bool:
    """Whether 'En COMPANY – Centro…' / 'En COMPANY buscamos…' names this employer."""
    if _looks_like_place(company) or _looks_like_field(company):
        return False
    pattern = re.compile(
        r"(?:^|[\s(])(?:[Aa]t|[Ee]n|@)\s+" + re.escape(company) + r"\s*(?:"
        r"[–—]\s+\S|"
        r"\b(?:buscamos|estamos|busca|hiring)\b"
        r")",
    )
    return pattern.search(text or "") is not None


_HASHTAG_RE = re.compile(r"#(\w{3,40})")


def _hashtag_names(text: str, company: str) -> bool:
    """Whether a hashtag spells out exactly this company: #NTTDATA for "NTT DATA"."""
    from jobbot.companies.urls import slugify

    flattened = slugify(company).replace("-", "")
    if len(flattened) < 4:
        return False
    return any(tag.casefold() == flattened for tag in _HASHTAG_RE.findall(text or ""))


def _apply_url_names(url: str, company: str) -> bool:
    """Whether the apply host (or mail domain) carries the company's name."""
    from jobbot.companies.urls import slugify

    raw = url.strip()
    if raw.casefold().startswith("mailto:"):
        host = raw.partition("@")[2]
    else:
        host = urlparse(raw if "://" in raw else f"https://{raw}").hostname or ""
    flattened = re.sub(r"[^a-z0-9]", "", host.casefold())
    slug = slugify(company)
    if len(slug) >= 4 and slug.replace("-", "") in flattened:
        return True
    tokens = [t for t in slug.split("-") if len(t) >= 5 and t not in _GENERIC_NAME_TOKENS]
    return any(token in flattened for token in tokens)
