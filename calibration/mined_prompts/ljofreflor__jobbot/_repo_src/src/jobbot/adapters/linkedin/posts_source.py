"""LinkedIn recruiter-post job source (MVP: posts → ATS URL / email / LinkedIn message)."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse, urlsplit, urlunsplit

from jobbot.adapters.linkedin.sweep import (
    dedupe_jobs_by_apply_target,
    looks_like_job_post,
    parse_post_blob,
    parse_posts_fixture,
    post_offers_wanted_country,
    post_to_jobs,
)
from jobbot.browser.session import BrowserSession
from jobbot.config import JobbotConfig, load_config
from jobbot.jobs.freshness import age_label, is_fresh
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind
from jobbot.portals.redirect import expand_urls, follow_redirect_url

logger = logging.getLogger("jobbot.linkedin.sweep")

# LinkedIn 2026 search results: hashed class names, stable `componentkey` hook.
MODERN_POST_CARD = 'div[componentkey^="update-card-focus"]'
LEGACY_POST_CARD = "div.feed-shared-update-v2"
TEXT_ONLY_CARD = "div.update-components-text, div.feed-shared-text, span.break-words"

SEE_MORE_TOGGLE = (
    "button.feed-shared-inline-show-more-text__see-more-less-toggle, "
    'button[aria-label*="see more"], '
    'button[aria-label*="ver más"], '
    'button:has-text("… más"), '
    'button:has-text("…more"), '
    'button:has-text("see more")'
)

_PERMALINK_RE = re.compile(r"linkedin\.com/(?:feed/update|posts)/", re.IGNORECASE)
_URN_RE = re.compile(r"urn:li:(?:activity|share|ugcPost):\d+")
_PROFILE_RE = re.compile(r"linkedin\.com/in/[^/?#]+", re.IGNORECASE)
_LINK_RE = re.compile(r"https?://", re.IGNORECASE)


def permalink_from_html(html: str) -> str | None:
    """Search cards rarely link themselves, but their HTML still carries the urn."""
    match = _URN_RE.search(html or "")
    if match is None:
        return None
    return f"https://www.linkedin.com/feed/update/{match.group(0)}/"


def first_post_permalink(hrefs: Sequence[str]) -> str | None:
    """The post's own URL, ignoring author profiles and hashtag searches."""
    for href in hrefs:
        if _PERMALINK_RE.search(href):
            return href
    return None


def author_profile_url(hrefs: Sequence[str]) -> str | None:
    """
    Fallback when the card exposes no permalink.

    Modern LinkedIn cards carry no urn:li:activity, so the recruiter's profile is
    the only durable handle: it leaves the post one click away instead of nothing.
    """
    for href in hrefs:
        if _PROFILE_RE.search(href):
            return href
    return None


def canonical_post_url(url: str) -> str:
    """
    Resolve lnkd.in and drop the query string.

    The copied link ends in ?utm_source=share&rcm=<your member id>: tracking that
    identifies the account doing the sweep, so it never reaches the database.
    """
    resolved = url
    host = (urlparse(url).hostname or "").casefold()
    if host.endswith("lnkd.in"):
        resolved = follow_redirect_url(url) or url
    split = urlsplit(resolved)
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))


def copy_post_permalink(page: Any, card: Any) -> str | None:
    """
    Read the post's own URL out of the card's "…" menu → "Copy link to post".

    Read-only: the menu is opened, the link copied and the menu closed with
    Escape. Nothing is shared, reported or saved on LinkedIn.
    """
    from jobbot.adapters.linkedin import selectors as sel

    try:
        card.locator(sel.POST_CONTROL_MENU).first.click(timeout=4_000)
    except Exception as exc:  # noqa: BLE001 — Playwright error types vary
        logger.debug("No control menu on this card: %s", exc)
        return None
    try:
        page.wait_for_timeout(800)
        link = _click_copy_link(page)
    finally:
        page.keyboard.press("Escape")
    if link and _LINK_RE.match(link.strip()):
        return link.strip()
    if link:
        logger.debug("Clipboard held no post link: %r", link[:60])
    return None


def _click_copy_link(page: Any) -> str | None:
    """Click the menu entry that copies the link, then read the clipboard."""
    from jobbot.adapters.linkedin import selectors as sel

    items = page.locator(sel.POST_MENU_ITEM)
    for index in range(items.count()):
        item = items.nth(index)
        try:
            label = item.inner_text(timeout=1_000).strip().casefold()
        except Exception:  # noqa: BLE001, S112 — menus repaint while we read them
            continue
        if not any(hint in label for hint in sel.COPY_LINK_LABELS):
            continue
        item.click(timeout=4_000)
        page.wait_for_timeout(800)
        clipboard = page.evaluate("() => navigator.clipboard.readText()")
        return str(clipboard) if clipboard else None
    return None


def allow_clipboard_read(context: Any) -> bool:
    """Copying the post link needs clipboard permission on linkedin.com."""
    try:
        context.grant_permissions(
            ["clipboard-read", "clipboard-write"],
            origin="https://www.linkedin.com",
        )
    except Exception as exc:  # noqa: BLE001 — unsupported on some CDP targets
        logger.warning("Clipboard permission denied; post links will fall back: %s", exc)
        return False
    return True


def wait_for_post_cards(page: Any, *, timeout_ms: int = 20_000) -> bool:
    """Search results render async; scrolling too early yields an empty sweep."""
    selector = f"{MODERN_POST_CARD}, {LEGACY_POST_CARD}"
    try:
        page.wait_for_selector(selector, timeout=timeout_ms)
    except Exception as exc:  # noqa: BLE001 — Playwright timeout types vary
        logger.warning("No post cards rendered within %dms: %s", timeout_ms, exc)
        return False
    return True


def autoscroll_feed(page: Any, *, rounds: int = 6, pause_ms: int = 800) -> None:
    """Load lazy feed posts without a human scrolling (read-only discovery)."""
    for _ in range(rounds):
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        page.wait_for_timeout(pause_ms)


def expand_truncated_posts(page: Any, *, limit: int = 40) -> int:
    """Click '…see more' toggles: the apply email is often inside the hidden tail."""
    toggles = page.locator(SEE_MORE_TOGGLE)
    try:
        total = min(int(toggles.count()), limit)
    except Exception:  # noqa: BLE001 — locator may vanish mid-scroll
        return 0
    expanded = 0
    for i in range(total):
        try:
            toggles.nth(i).click(timeout=1500)
            expanded += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("see-more toggle %d not clickable: %s", i, exc)
    return expanded


class LinkedInPostJobSource:
    """Discover jobs from LinkedIn recruiter posts (fixture or live HITL)."""

    def __init__(
        self,
        config: JobbotConfig | None = None,
        *,
        cdp_url: str | None = None,
    ) -> None:
        self.config = config or load_config()
        from jobbot.browser.cdp import resolve_cdp_url

        self.cdp_url = resolve_cdp_url(cdp_url)
        self.profile_dir = self.config.browser_profile_dir("linkedin")

    def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        """Live LinkedIn content search for posts (HITL / CDP)."""
        return self.search_live(query)

    def search_from_fixture(
        self,
        path: Path,
        *,
        query: str | None = None,
        countries: Sequence[str] = (),
        allow_remote: bool = True,
        max_age_days: int = 0,
    ) -> list[JobPosting]:
        text = path.read_text(encoding="utf-8")
        posts = parse_posts_fixture(text)
        jobs: list[JobPosting] = []
        q = (query or "").casefold()
        for post in posts:
            if not post.vacancies and not looks_like_job_post(post.text):
                continue
            if not post_offers_wanted_country(
                post.text, wanted=countries, allow_remote=allow_remote
            ):
                continue
            if not is_fresh(post.posted_at, max_age_days=max_age_days):
                continue
            if (
                q
                and q not in post.text.casefold()
                and q not in (post.author or "").casefold()
                and post.ats_kind == AtsKind.UNKNOWN
                and not post.vacancies
            ):
                continue
            jobs.extend(post_to_jobs(post, countries=countries))
        return dedupe_jobs_by_apply_target(jobs)

    def search_live(self, query: JobSearchQuery) -> list[JobPosting]:
        """Open LinkedIn content search; user assists; scrape visible post texts."""
        search_url = (
            "https://www.linkedin.com/search/results/content/"
            f"?keywords={quote_plus(query.query)}&origin=GLOBAL_SEARCH_HEADER"
        )
        with BrowserSession(
            profile_dir=self.profile_dir,
            headless=False,
            slow_mo=100,
            cdp_url=self.cdp_url,
            debug_root=self.config.output_dir / "debug",
        ) as browser:
            # Attached to the user's own Chrome: do not take over a tab in use.
            page = browser.context.new_page() if self.cdp_url else browser.page
            copy_links = query.copy_permalinks and allow_clipboard_read(browser.context)
            page.goto(search_url, wait_until="domcontentloaded", timeout=60_000)
            if "login" in page.url.lower() or "authwall" in page.url.lower():
                browser.pause_for_manual(
                    "Log in to LinkedIn, run the content search if needed, "
                    "then continue so JobBot can read visible posts."
                )
            wait_for_post_cards(page)
            autoscroll_feed(page)
            wait_for_post_cards(page, timeout_ms=10_000)
            expanded = expand_truncated_posts(page)
            logger.info("Expanded %d truncated posts", expanded)
            browser.pause_for_manual(
                "Scroll further if you want more LinkedIn posts included, then continue "
                "(JobBot already auto-scrolled and expanded '…see more')."
            )
            autoscroll_feed(page, rounds=2)
            expand_truncated_posts(page)
            return collect_jobs_from_feed_page(page, query, copy_permalinks=copy_links)

    def get_job(self, job_id: str) -> JobPosting:
        msg = "LinkedInPostJobSource.get_job is not supported; use JobRepository"
        raise NotImplementedError(msg)


def collect_jobs_from_feed_page(
    page: Any,
    query: JobSearchQuery,
    *,
    resolve_short_links: bool = True,
    copy_permalinks: bool = False,
    now: datetime | None = None,
) -> list[JobPosting]:
    """
    Scrape visible LinkedIn-like feed cards from an already-open page.

    Same path used by live sweep after HITL scroll — suitable for game-condition
    browser tests with a local DOM fixture.
    """
    jobs: list[JobPosting] = []
    cards = page.locator(MODERN_POST_CARD)
    if cards.count() == 0:
        cards = page.locator(LEGACY_POST_CARD)
    if cards.count() == 0:
        cards = page.locator(TEXT_ONLY_CARD)
    seen: set[str] = set()
    limit = query.limit
    for i in range(min(cards.count(), 80)):
        if len(jobs) >= limit:
            break
        card = cards.nth(i)
        try:
            text = card.inner_text(timeout=1000).strip()
        except Exception:  # noqa: BLE001
            continue
        if len(text) < 40 or text in seen:
            continue
        seen.add(text)
        # Vacancies are parsed after the card is accepted; hiring language or
        # an explicit apply hint is enough to keep the card for parsing.
        if not looks_like_job_post(text):
            continue
        if not post_offers_wanted_country(
            text,
            wanted=query.countries,
            allow_remote=query.allow_remote,
        ):
            logger.info("Skipped post outside %s", ", ".join(query.countries))
            continue
        q = query.query.casefold()
        if q not in text.casefold() and (
            "hiring" not in text.casefold()
            and "buscamos" not in text.casefold()
            and "postula" not in text.casefold()
            and "vacante" not in text.casefold()
            and "enviar cv" not in text.casefold()
        ):
            continue
        raw_hrefs = _hrefs_from_locator(card, base_url=page.url)
        mailtos = _mailto_from_locator(card)
        permalink = first_post_permalink(raw_hrefs) or _permalink_from_card(card)
        if permalink is None and copy_permalinks:
            copied = copy_post_permalink(page, card)
            permalink = canonical_post_url(copied) if copied else None
        post_url = permalink or author_profile_url(raw_hrefs)
        hrefs = expand_urls([h for h in raw_hrefs if h != permalink])
        if resolve_short_links:
            unresolved = [h for h in hrefs if "lnkd.in" in h or "linkedin.com" in h]
            if unresolved:
                hrefs = expand_urls(hrefs + _click_resolve_hrefs(card, unresolved))
        post = parse_post_blob(
            text,
            post_url=post_url,
            mailto_urls=mailtos,
            extra_urls=hrefs,
        )
        if not is_fresh(post.posted_at, max_age_days=query.max_age_days, now=now):
            logger.info(
                "Skipped post published %s ago (max %d days)",
                age_label(post.posted_at, now=now),
                query.max_age_days,
            )
            continue
        jobs.extend(post_to_jobs(post, countries=query.countries))
    return dedupe_jobs_by_apply_target(jobs)


def _permalink_from_card(card: Any) -> str | None:
    try:
        return permalink_from_html(card.inner_html(timeout=1000))
    except Exception:  # noqa: BLE001 — card may detach while scrolling
        return None


def _mailto_from_locator(card: object) -> list[str]:
    """Collect mailto: links: some posts only expose the address as a link."""
    out: list[str] = []
    try:
        anchors = card.locator('a[href^="mailto:"]')  # type: ignore[attr-defined]
        count = min(int(anchors.count()), 20)
    except Exception:  # noqa: BLE001
        return out
    for j in range(count):
        try:
            href = (anchors.nth(j).get_attribute("href") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if href and href not in out:
            out.append(href)
    return out


def _click_resolve_hrefs(card: object, hrefs: list[str]) -> list[str]:
    """Click short links in-card (HITL browser) and capture navigation targets."""
    resolved: list[str] = []
    try:
        anchors = card.locator("a[href]")  # type: ignore[attr-defined]
        count = min(int(anchors.count()), 50)
    except Exception:  # noqa: BLE001
        return resolved
    targets = set(hrefs)
    for j in range(count):
        try:
            anchor = anchors.nth(j)
            href = (anchor.get_attribute("href") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if href not in targets:
            continue
        try:
            with anchor.page.expect_navigation(timeout=8000):
                anchor.click(timeout=3000)
            final = anchor.page.url
            if final.startswith("http") and final not in resolved:
                resolved.append(final)
            anchor.page.go_back(timeout=8000)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Click-resolve failed for %s: %s", href, exc)
    return resolved


def _hrefs_from_locator(card: object, *, base_url: str) -> list[str]:
    """Collect absolute http(s) links from a post card (for ATS / board detection)."""
    out: list[str] = []
    seen: set[str] = set()
    try:
        anchors = card.locator("a[href]")  # type: ignore[attr-defined]
        count = min(int(anchors.count()), 50)
    except Exception:  # noqa: BLE001
        return out
    for j in range(count):
        try:
            href = anchors.nth(j).get_attribute("href") or ""
        except Exception:  # noqa: BLE001
            continue
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(base_url, href)
        if not abs_url.startswith("http"):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append(abs_url)
    return out
