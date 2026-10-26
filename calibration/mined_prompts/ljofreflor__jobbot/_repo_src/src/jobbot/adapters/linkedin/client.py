"""LinkedIn adapter — read-only audit + publications sync (HITL)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from jobbot.adapters.diff_engine import save_snapshot
from jobbot.adapters.html_parse import parse_linkedin_profile_html
from jobbot.adapters.linkedin import selectors
from jobbot.adapters.linkedin.package import (
    LinkedInPublicationItem,
    build_linkedin_publication_items,
    publications_missing_from_remote,
)
from jobbot.adapters.linkedin.publications import (
    fill_publication_form,
    list_remote_publication_titles,
    open_edit_publication_form,
    open_new_publication_form,
    save_publication,
    vanity_from_linkedin_url,
)
from jobbot.browser.debug import inspect_page
from jobbot.browser.session import BrowserSession
from jobbot.config import JobbotConfig, load_config
from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from jobbot.models.sync import SyncOperation, SyncOpType, SyncPlan, SyncResult
from jobbot.profile.loader import load_profile

logger = logging.getLogger("jobbot.linkedin")
console = Console()


class LinkedInAdapter:
    """LinkedIn: read-only by default; publications writable with --apply + confirm."""

    def __init__(self, config: JobbotConfig, *, cdp_url: str | None = None) -> None:
        self.config = config
        self.profile_dir = config.browser_profile_dir("linkedin")
        self.debug_root = config.output_dir / "debug"
        from jobbot.browser.cdp import resolve_cdp_url

        self.cdp_url = resolve_cdp_url(cdp_url)

    @classmethod
    def from_config(
        cls,
        config: JobbotConfig | None = None,
        *,
        cdp_url: str | None = None,
    ) -> LinkedInAdapter:
        return cls(config or load_config(), cdp_url=cdp_url)

    def _session(self) -> BrowserSession:
        return BrowserSession(
            profile_dir=self.profile_dir,
            headless=False,
            slow_mo=100,
            cdp_url=self.cdp_url,
            debug_root=self.debug_root,
        )

    def login(self) -> None:
        sentinel = self.debug_root / "linkedin-continue"
        with self._session() as browser:
            browser.page.goto(selectors.LINKEDIN_LOGIN, wait_until="domcontentloaded")

            def _logged_in() -> bool:
                url = browser.page.url.lower()
                return "login" not in url and "authwall" not in url and "checkpoint" not in url

            browser.pause_for_manual(
                "Log in to LinkedIn (including 2FA if prompted).\n"
                f"When the feed/profile loads, press Enter or: touch {sentinel}",
                is_clear=_logged_in,
                timeout_seconds=600,
                continue_sentinel=sentinel,
            )
            if _logged_in():
                console.print("[green]LinkedIn login OK. Session persisted.[/green]")
            else:
                console.print(
                    "[yellow]Login wait timed out — check browser-data/linkedin.[/yellow]"
                )

    def session_valid(self) -> bool:
        try:
            with self._session() as browser:
                browser.page.goto(selectors.LINKEDIN_FEED, wait_until="domcontentloaded")
                url = browser.page.url.lower()
                return "login" not in url and "authwall" not in url
        except Exception as exc:  # noqa: BLE001
            logger.warning("session check failed: %s", exc)
            return False

    def inspect(self) -> None:
        with self._session() as browser:
            browser.page.goto(selectors.LINKEDIN_ME, wait_until="domcontentloaded")
            browser.pause_for_manual("Open your LinkedIn profile if needed, then press Enter.")
            inspect_page(browser.page, console)

    def pull_profile(self) -> ExternalProfile:
        with self._session() as browser:
            browser.page.goto(selectors.LINKEDIN_ME, wait_until="domcontentloaded")
            if "login" in browser.page.url.lower() or "authwall" in browser.page.url.lower():
                browser.pause_for_manual("LinkedIn requires login. Complete it, then press Enter.")
            html = browser.page.content()
            profile = parse_linkedin_profile_html(html)
            if not profile.headline:
                console.print(
                    "[yellow]LinkedIn DOM lacked stable markers; sparse snapshot saved.[/yellow]"
                )
                profile = ExternalProfile(
                    source="linkedin",
                    headline=None,
                    summary=None,
                    captured_at=datetime.now(UTC),
                )
            return profile

    def pull_and_save(self) -> Path:
        return save_snapshot(self.pull_profile(), self.config.output_dir)

    def build_publications_plan(
        self,
        candidate: Candidate | None = None,
        *,
        remote_titles: list[str] | None = None,
        enrich_existing: bool = False,
    ) -> SyncPlan:
        cand = candidate or load_profile(self.config.profile_path)
        items = build_linkedin_publication_items(cand)
        if remote_titles is None:
            remote_titles = []
        missing = publications_missing_from_remote(items, remote_titles)
        ops: list[SyncOperation] = []
        for item in missing:
            ops.append(
                SyncOperation(
                    op=SyncOpType.ADD,
                    section="publications",
                    field=item.id,
                    before=None,
                    after={
                        "title": item.title,
                        "publisher": item.publisher,
                        "year": item.year,
                        "url": item.url,
                        "coauthors": list(item.coauthors),
                        "authors": list(item.authors),
                    },
                    label=item.title,
                )
            )
        for item in items:
            on_remote = any(_title_match(item.title, t) for t in remote_titles)
            if on_remote and enrich_existing:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.CHANGE,
                        section="publications",
                        field=item.id,
                        after={
                            "title": item.title,
                            "publisher": item.publisher,
                            "year": item.year,
                            "url": item.url,
                            "coauthors": list(item.coauthors),
                            "authors": list(item.authors),
                        },
                        label=f"enrich: {item.title}",
                    )
                )
            elif on_remote:
                ops.append(
                    SyncOperation(
                        op=SyncOpType.SAME,
                        section="publications",
                        field=item.id,
                        after=item.title,
                        label=item.title,
                    )
                )
        return SyncPlan(target="linkedin", operations=ops)

    def fetch_remote_publication_titles(self) -> list[str]:
        candidate = load_profile(self.config.profile_path)
        vanity = vanity_from_linkedin_url(candidate.personal.linkedin)
        with self._session() as browser:
            browser.page.goto(selectors.LINKEDIN_FEED, wait_until="domcontentloaded")
            if "login" in browser.page.url.lower() or "authwall" in browser.page.url.lower():
                browser.pause_for_manual("LinkedIn requires login. Complete it, then press Enter.")
            return list_remote_publication_titles(browser.page, vanity)

    def apply_publications(
        self,
        *,
        confirm_each: Callable[[LinkedInPublicationItem], bool] | None = None,
        remote_titles: list[str] | None = None,
        enrich_existing: bool = False,
    ) -> SyncResult:
        """Add missing publications; optionally enrich existing (date/authors/DOI)."""
        candidate = load_profile(self.config.profile_path)
        vanity = vanity_from_linkedin_url(candidate.personal.linkedin)
        items = build_linkedin_publication_items(candidate)
        applied: list[SyncOperation] = []
        errors: list[str] = []

        with self._session() as browser:
            page = browser.page
            page.goto(selectors.LINKEDIN_FEED, wait_until="domcontentloaded")
            if "login" in page.url.lower() or "authwall" in page.url.lower():
                browser.pause_for_manual("LinkedIn requires login. Complete it, then press Enter.")

            remote = remote_titles
            if remote is None:
                remote = list_remote_publication_titles(page, vanity)
                console.print(f"Remote publications detected: {len(remote)}")

            missing = publications_missing_from_remote(items, remote)
            to_enrich = [
                item
                for item in items
                if enrich_existing and any(_title_match(item.title, t) for t in remote)
            ]

            if not missing and not to_enrich:
                return SyncResult(
                    target="linkedin",
                    applied=[],
                    verified=True,
                    message="No missing publications to add.",
                )

            for item in missing:
                if confirm_each is not None and not confirm_each(item):
                    console.print(f"[yellow]Skipped[/yellow] {item.title}")
                    continue
                console.print(f"Adding: [bold]{item.title}[/bold]")
                try:
                    open_new_publication_form(page, vanity)
                    self._fill_and_save(browser, item)
                    applied.append(
                        SyncOperation(
                            op=SyncOpType.ADD,
                            section="publications",
                            field=item.id,
                            after=item.title,
                            label=item.title,
                        )
                    )
                    console.print(f"[green]Saved[/green] {item.title}")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to add publication %s", item.id)
                    errors.append(f"{item.id}: {exc}")
                    browser.dump_debug(
                        "linkedin", "publication-error", pub_id=item.id, error=str(exc)
                    )

            for item in to_enrich:
                if confirm_each is not None and not confirm_each(item):
                    console.print(f"[yellow]Skipped enrich[/yellow] {item.title}")
                    continue
                console.print(f"Enriching: [bold]{item.title}[/bold]")
                try:
                    if not open_edit_publication_form(page, vanity, item.title):
                        errors.append(f"{item.id}: edit link not found")
                        continue
                    self._fill_and_save(browser, item)
                    applied.append(
                        SyncOperation(
                            op=SyncOpType.CHANGE,
                            section="publications",
                            field=item.id,
                            after=item.title,
                            label=item.title,
                        )
                    )
                    console.print(f"[green]Updated[/green] {item.title}")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to enrich publication %s", item.id)
                    errors.append(f"{item.id}: {exc}")
                    browser.dump_debug(
                        "linkedin", "publication-enrich-error", pub_id=item.id, error=str(exc)
                    )

        msg = f"Applied {len(applied)} publication(s)."
        if errors:
            msg += f" Errors: {'; '.join(errors)}"
        return SyncResult(
            target="linkedin",
            applied=applied,
            verified=not errors,
            message=msg,
        )

    def _fill_and_save(self, browser: BrowserSession, item: LinkedInPublicationItem) -> None:
        page = browser.page
        missing_fields = fill_publication_form(page, item)
        # Coauthor LinkedIn matches are optional; description carries official authors.
        hard_missing = [f for f in missing_fields if not f.startswith("coauthor:")]
        if hard_missing:
            console.print(f"[yellow]Could not fill:[/yellow] {', '.join(hard_missing)}")
            browser.dump_debug("linkedin", "publication-fill-partial", pub_id=item.id)
            sentinel = self.debug_root / "linkedin-continue"
            browser.pause_for_manual(
                "Complete remaining fields in the LinkedIn form, then Save "
                f"(or touch {sentinel} / press Enter).",
                continue_sentinel=sentinel,
                timeout_seconds=300,
            )
        soft = [f for f in missing_fields if f.startswith("coauthor:")]
        if soft:
            console.print(
                f"[dim]Coauthors without LinkedIn match (kept in Descripción): "
                f"{len(soft)}[/dim]"
            )
        if not save_publication(page):
            sentinel = self.debug_root / "linkedin-continue"
            browser.pause_for_manual(
                f"Could not click Guardar. Save manually, then touch {sentinel}.",
                continue_sentinel=sentinel,
                timeout_seconds=300,
            )


def _title_match(a: str, b: str) -> bool:
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())
