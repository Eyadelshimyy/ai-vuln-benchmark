"""Indeed job discovery — small explicit volumes, no mass crawl."""

from __future__ import annotations

import html
import logging
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

from playwright.sync_api import Page
from rich.console import Console

from jobbot.adapters.indeed import selectors
from jobbot.browser.session import BrowserSession
from jobbot.config import JobbotConfig, load_config
from jobbot.jobs.closure import visible_soup
from jobbot.jobs.indeed_url import indeed_jk_rejection
from jobbot.jobs.parsing import parse_job_text
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting

logger = logging.getLogger("jobbot.indeed.jobs")
console = Console()

# Wording Indeed uses when a posting is closed. Not a list of job families.
_CLOSED_MARKERS: tuple[str, ...] = (
    "this job has expired",
    "no longer accepting",
    "ya no acepta",
    "este empleo ha expirado",
    "esta oferta ha caducado",
)


class IndeedJobClosed(ValueError):
    """The posting is no longer accepting applications."""


class IndeedJobSource:
    """JobSourceAdapter for Indeed (cl.indeed.com by default)."""

    def __init__(
        self,
        config: JobbotConfig | None = None,
        *,
        cdp_url: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base = selectors.indeed_jobs_base(self._country())
        self.profile_dir = self.config.browser_profile_dir("indeed")
        self.debug_root = self.config.output_dir / "debug"
        from jobbot.browser.cdp import resolve_cdp_url

        self.cdp_url = resolve_cdp_url(cdp_url)

    def _country(self) -> str:
        # Loaded from .jobbot.toml [indeed].country when present
        path = self.config.root / ".jobbot.toml"
        if path.is_file():
            import tomllib

            with path.open("rb") as fh:
                raw = tomllib.load(fh)
            return str(raw.get("indeed", {}).get("country", "cl"))
        return "cl"

    def _session(self) -> BrowserSession:
        return BrowserSession(
            profile_dir=self.profile_dir,
            headless=False,
            slow_mo=100,
            debug_root=self.debug_root,
            cdp_url=self.cdp_url,
        )

    def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        """Search Indeed and return up to query.limit postings (cards + detail)."""
        return self._search(query, enrich=True)

    def search_cards(self, query: JobSearchQuery) -> list[JobPosting]:
        """Search Indeed using result cards only (no per-job navigation)."""
        return self._search(query, enrich=False)

    def _search(self, query: JobSearchQuery, *, enrich: bool) -> list[JobPosting]:
        limit = min(query.limit, 50)
        url = self._search_url(query)
        logger.info("Indeed search: %s", url)
        with self._session() as browser:
            # Warm session on home first — reduces empty challenge landings.
            browser.page.goto(self.base + "/", wait_until="domcontentloaded")
            browser.page.goto(url, wait_until="domcontentloaded")
            if not self._clear_challenge(browser, context="search results"):
                return []
            html = browser.page.content()
            discards: list[tuple[str, str]] = []
            cards = parse_indeed_search_html(html, base_url=self.base, discards=discards)
            if not cards and _looks_like_challenge_html(html):
                if not self._clear_challenge(browser, context="search results"):
                    return []
                browser.page.goto(url, wait_until="domcontentloaded")
                html = browser.page.content()
                discards = []
                cards = parse_indeed_search_html(html, base_url=self.base, discards=discards)
            if discards:
                reasons: dict[str, int] = {}
                for _, reason in discards:
                    reasons[reason] = reasons.get(reason, 0) + 1
                summary = ", ".join(f"{count} {name}" for name, count in sorted(reasons.items()))
                console.print(
                    f"[yellow]{len(discards)} tarjeta(s) descartada(s) ({summary}).[/yellow]"
                )
            if not cards:
                browser.dump_debug("indeed", "jobs_search", selector_attempts=["job cards"])
                console.print(
                    "[yellow]No jobs found.[/yellow]"
                    if discards
                    else "[yellow]No job cards found. UI may have changed; "
                    "debug snapshot saved.[/yellow]"
                )
                return []

            results: list[JobPosting] = []
            for card in cards[:limit]:
                detail_html = None
                if enrich and card.get("url"):
                    try:
                        browser.page.goto(
                            card["url"],
                            wait_until="domcontentloaded",
                            timeout=20_000,
                        )
                        if not self._clear_challenge(browser, context="job detail"):
                            continue
                        detail_html = browser.page.content()
                        if detail_is_empty_or_search(detail_html):
                            logger.info(
                                "skipping Indeed card %s: detail is not a posting",
                                card.get("source_job_id"),
                            )
                            continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("detail fetch failed for %s: %s", card.get("url"), exc)
                job = card_to_job_posting(card, detail_html=detail_html, placeholder_id="TMP")
                results.append(job)
            return results

    def get_job(self, job_id: str) -> JobPosting:
        """Fetch a single Indeed job by jk / URL fragment."""
        url = job_id if job_id.startswith("http") else f"{self.base}/viewjob?jk={job_id}"
        with self._session() as browser:
            browser.page.goto(url, wait_until="domcontentloaded")
            self._clear_challenge(browser, context="job detail")
            html = browser.page.content()
            card = {
                "source_job_id": _extract_jk(url) or job_id,
                "url": url,
                "title": "",
                "company": "",
                "location": None,
                "snippet": "",
            }
            return card_to_job_posting(card, detail_html=html, placeholder_id="TMP")

    def _clear_challenge(self, browser: BrowserSession, *, context: str) -> bool:
        if not _looks_like_challenge(browser.page):
            return True
        cleared = browser.pause_for_manual(
            f"Indeed showed a challenge/CAPTCHA/consent on {context}. "
            "Complete it in the Chromium window, then press Enter "
            "(or wait — this run polls until the page clears).",
            is_clear=lambda: not _looks_like_challenge(browser.page),
            timeout_seconds=300,
        )
        if not cleared:
            browser.dump_debug("indeed", "challenge_timeout", context=context)
            console.print(
                f"[red]Challenge on {context} not cleared within 5 minutes.[/red]"
            )
            return False
        return True

    def _search_url(self, query: JobSearchQuery) -> str:
        q = quote_plus(query.query)
        loc = quote_plus(query.location or "")
        url = f"{self.base}/jobs?q={q}&l={loc}"
        if query.remote:
            url += "&sc=0kf%3Aattr(DSQF7)%3B"  # common remote filter; may vary by country
        return url


def parse_indeed_search_html(
    html: str,
    *,
    base_url: str,
    discards: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Parse search result cards from Indeed HTML (fixture-friendly).

    Decoy ``jk`` values and cards marked hidden in the DOM are skipped. When
    ``discards`` is provided, each skipped ``(jk_or_marker, reason)`` is appended.
    """
    # Drop nodes the page itself hides (decoy templates, aria-hidden cards).
    visible_html = str(visible_soup(html))
    cards: list[dict[str, Any]] = []
    # Prefer explicit fixture/test markers
    for block in re.finditer(
        r'<article[^>]*data-testid="job-card"[^>]*>.*?</article>',
        visible_html,
        flags=re.I | re.S,
    ):
        card = _parse_card_block(block.group(0), base_url, discards=discards)
        if card is not None:
            cards.append(card)

    if cards:
        return _unique_cards(cards)

    # Live Indeed often uses data-jk on result cards (mid-<a>); include lookbehind for title.
    for match in re.finditer(
        r'data-jk="([a-f0-9]+)"',
        visible_html,
        flags=re.I,
    ):
        jk = match.group(1)
        reason = indeed_jk_rejection(jk)
        if reason is not None:
            if discards is not None:
                discards.append((jk, reason))
            continue
        title_window = visible_html[max(0, match.start() - 400) : match.start() + 3000]
        # Company/location sit after the link — avoid previous card via lookbehind.
        after = visible_html[match.start() : match.start() + 3000]
        title = _first(
            title_window,
            [
                rf'id="jobTitle-{re.escape(jk)}"[^>]*>(.*?)</span>',
                rf'<span[^>]*title="([^"]+)"[^>]*id="jobTitle-{re.escape(jk)}"',
                r'<span[^>]*title="([^"]+)"[^>]*id="jobTitle-[^"]+"',
                r'aria-label="detalles completos de ([^"]+)"',
                r'aria-label="full details of ([^"]+)"',
                r'data-testid="job-title"[^>]*>(.*?)</',
                r'<h[23][^>]*class="[^"]*jobTitle[^"]*"[^>]*>.*?<span[^>]*>(.*?)</span>',
                r'<a[^>]*id="job_[^"]*"[^>]*>(.*?)</a>',
            ],
        )
        company = _first(
            after,
            [
                r'data-testid="company-name"[^>]*>(.*?)</',
                r'data-testid="companyName"[^>]*>(.*?)</',
                r'class="[^"]*companyName[^"]*"[^>]*>(.*?)</',
            ],
        )
        location = _first(
            after,
            [
                r'data-testid="text-location"[^>]*>(.*?)</',
                r'class="[^"]*companyLocation[^"]*"[^>]*>(.*?)</',
            ],
        )
        snippet = _first(
            after,
            [
                r'data-testid="job-snippet"[^>]*>(.*?)</',
                r'class="[^"]*job-snippet[^"]*"[^>]*>(.*?)</',
            ],
        )
        cards.append(
            {
                "source_job_id": jk.casefold(),
                "url": f"{base_url}/viewjob?jk={jk.casefold()}",
                "title": _clean(title) or "Untitled",
                "company": _clean(company) or "Unknown",
                "location": _clean(location) or None,
                "snippet": _clean(snippet) or "",
            }
        )
    return _unique_cards(cards)


def _unique_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for card in cards:
        jk = str(card["source_job_id"])
        if jk in seen:
            continue
        seen.add(jk)
        unique.append(card)
    return unique


def posting_is_closed(html: str) -> bool:
    """True when the page says the vacancy is no longer open."""
    folded = (html or "").casefold()
    return any(marker in folded for marker in _CLOSED_MARKERS)


def indeed_apply_url(viewjob_url: str | None, jk: str | None) -> str | None:
    """Indeed Apply handoff for one vacancy: same country host, only jk."""
    if not viewjob_url or not jk:
        return None
    parsed = urlparse(viewjob_url)
    if not parsed.hostname:
        return None
    scheme = parsed.scheme or "https"
    return f"{scheme}://{parsed.hostname}/applystart?jk={jk}"


def parse_indeed_job_detail_html(html: str) -> dict[str, Any]:
    """Extract description and metadata from a job detail page."""
    title = _first(
        html,
        [
            r'data-testid="jobsearch-JobInfoHeader-title"[^>]*>(.*?)</',
            r"<h1[^>]*>(.*?)</h1>",
        ],
    )
    company = _first(
        html,
        [
            r'data-testid="inlineHeader-companyName"[^>]*>(.*?)</',
            r'data-company-name="([^"]+)"',
        ],
    )
    location = _first(
        html,
        [
            r'data-testid="jobsearch-JobInfoHeader-locationText"[^>]*>(.*?)</',
            r'data-testid="text-location"[^>]*>(.*?)</',
        ],
    )
    description = _first(
        html,
        [
            r'data-testid="jobsearch-JobComponent-description"[^>]*>(.*?)</div>',
            r'class="[^"]*jobsearch-JobComponent-description[^"]*"[^>]*>(.*?)</div>',
            r'id="jobDescriptionText"[^>]*>(.*?)</div>',
            r'data-testid="job-description"[^>]*>(.*?)</',
        ],
    )
    
    # Detect external ATS URL (Apply on company site button)
    ats_url = None
    
    # Look for "Apply on company site" or similar external apply links
    apply_patterns = [
        r'href="([^"]+)"[^>]*>Apply on company site',
        r'href="([^"]+)"[^>]*>Apply on employer site',
        r'href="([^"]+)"[^>]*>Aplicar en el sitio de la empresa',
        r'data-tn-element="[^"]*externalApply[^"]*"[^>]*href="([^"]+)"',
    ]
    
    for pattern in apply_patterns:
        match = re.search(pattern, html, flags=re.I)
        if match:
            ats_url = match.group(1)
            # Clean up Indeed redirect wrapper if present
            if "indeed.com" in ats_url and ("rclk?jk=" in ats_url or "/rc/clk" in ats_url):
                # Extract actual URL from Indeed redirect
                redirect_match = re.search(r'[?&]dest=([^&]+)', ats_url)
                if redirect_match:
                    from urllib.parse import unquote
                    ats_url = unquote(redirect_match.group(1))
            break
    
    return {
        "title": _clean(title),
        "company": _clean(company),
        "location": _clean(location),
        "description": _clean_preserve_breaks(description or ""),
        "ats_url": ats_url,
    }


def card_to_job_posting(
    card: dict[str, Any],
    *,
    detail_html: str | None,
    placeholder_id: str,
) -> JobPosting:
    if detail_html and posting_is_closed(detail_html):
        raise IndeedJobClosed(
            "This Indeed job is no longer accepting applications (expired)."
        )
    detail = parse_indeed_job_detail_html(detail_html) if detail_html else {}
    title = detail.get("title") or card.get("title") or "Untitled"
    company = detail.get("company") or card.get("company") or "Unknown"
    location = detail.get("location") or card.get("location")
    description = detail.get("description") or card.get("snippet") or ""
    raw = description
    # Reuse text parser for skills/requirements heuristics
    stub = (
        f"Title: {title}\nCompany: {company}\n"
        f"Location: {location or ''}\n\n{description}"
    )
    parsed = parse_job_text(stub, job_id=placeholder_id, source="indeed", url=card.get("url"))
    
    ats_url = detail.get("ats_url") if detail else None
    if isinstance(ats_url, str) and "indeed.com" in ats_url.casefold():
        ats_url = None
    if detail_html and not ats_url:
        ats_url = indeed_apply_url(card.get("url"), card.get("source_job_id"))
    ats_kind = None
    if ats_url:
        from jobbot.portals.detect import detect_ats

        ats_kind = detect_ats(ats_url).value
    
    return JobPosting(
        id=placeholder_id,
        source="indeed",
        source_job_id=card.get("source_job_id"),
        url=card.get("url"),
        title=title,
        company=company,
        location=location,
        description=description,
        raw_description=raw,
        requirements=parsed.requirements,
        skills=parsed.skills,
        seniority=parsed.seniority,
        language_requirements=parsed.language_requirements,
        employment_type=parsed.employment_type,
        remote_type=parsed.remote_type,
        ats_url=ats_url,
        ats_kind=ats_kind,
        discovered_at=datetime.now(UTC),
    )


def _parse_card_block(
    block: str,
    base_url: str,
    *,
    discards: list[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    jk = _first(block, [r'data-jk="([^"]+)"', r'data-testid="jk-([^"]+)"']) or ""
    if jk:
        reason = indeed_jk_rejection(jk)
        if reason is not None:
            if discards is not None:
                discards.append((jk, reason))
            return None
        jk = jk.casefold()
    href = _first(block, [r'href="([^"]+)"'])
    url = urljoin(base_url + "/", href) if href else (f"{base_url}/viewjob?jk={jk}" if jk else None)
    title = _clean(
        _first(block, [r'data-testid="job-title"[^>]*>(.*?)</', r"<h2[^>]*>(.*?)</h2>"])
    )
    if not jk:
        if discards is not None:
            discards.append((href or "unknown", "formato"))
        return None
    return {
        "source_job_id": jk,
        "url": url,
        "title": title or "Untitled",
        "company": _clean(_first(block, [r'data-testid="company-name"[^>]*>(.*?)</'])) or "Unknown",
        "location": _clean(_first(block, [r'data-testid="text-location"[^>]*>(.*?)</'])) or None,
        "snippet": _clean(_first(block, [r'data-testid="job-snippet"[^>]*>(.*?)</'])) or "",
    }


def _looks_like_challenge(page: Page) -> bool:
    url = page.url.lower()
    if any(x in url for x in ("captcha", "challenge", "consent")):
        return True
    try:
        return _looks_like_challenge_html(page.content())
    except Exception:  # noqa: BLE001 — navigation mid-check
        return False


def _looks_like_challenge_html(html: str) -> bool:
    lowered = html.lower()
    markers = (
        "captcha",
        "unusual traffic",
        "additional verification",
        "verificación adicional",
        "indeed_cloudflare_static_page",
        "cf-turnstile",
        "challenge-platform",
    )
    return any(m in lowered for m in markers)


def _extract_jk(url: str) -> str | None:
    match = re.search(r"[?&]jk=([a-f0-9]+)", url, flags=re.I)
    if match is None:
        return None
    jk = match.group(1)
    if indeed_jk_rejection(jk) is not None:
        return None
    return jk.casefold()


def detail_is_empty_or_search(html: str) -> bool:
    """True when a viewjob page did not open a real posting (redirect/search shell)."""
    detail = parse_indeed_job_detail_html(html)
    title = (detail.get("title") or "").strip()
    description = (detail.get("description") or "").strip()
    if title or description:
        return False
    folded = (html or "").casefold()
    return "data-jk=" in folded or "/jobs?" in folded or 'name="q"' in folded


def _first(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match.group(1)
    return None


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _clean_preserve_breaks(text: str) -> str:
    text = re.sub(r"<(br|BR)\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
