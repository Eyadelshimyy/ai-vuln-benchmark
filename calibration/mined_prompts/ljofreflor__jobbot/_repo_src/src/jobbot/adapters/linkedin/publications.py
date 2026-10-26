"""Playwright helpers for LinkedIn Publications (Spanish UI; HITL; no CAPTCHA bypass)."""

from __future__ import annotations

import logging
import re

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from jobbot.adapters.linkedin.package import LinkedInPublicationItem

logger = logging.getLogger("jobbot.linkedin.publications")

PUBLICATION_NEW_PATH = "/edit/forms/publication/new/"
PUBLICATIONS_DETAILS_PATH = "/details/publications/"


def profile_publications_new_url(vanity_or_me: str = "me") -> str:
    slug = vanity_or_me.strip("/").removeprefix("in/")
    return f"https://www.linkedin.com/in/{slug}{PUBLICATION_NEW_PATH}"


def profile_publications_details_url(vanity_or_me: str = "me") -> str:
    slug = vanity_or_me.strip("/").removeprefix("in/")
    return f"https://www.linkedin.com/in/{slug}{PUBLICATIONS_DETAILS_PATH}"


def vanity_from_linkedin_url(url: str | None) -> str:
    if not url:
        return "me"
    match = re.search(r"linkedin\.com/in/([^/?#]+)", url)
    return match.group(1) if match else "me"


def list_remote_publication_titles(page: Page, vanity: str) -> list[str]:
    """Best-effort scrape of existing publication titles on details page."""
    page.goto(
        profile_publications_details_url(vanity),
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    page.wait_for_timeout(2000)
    titles: list[str] = []
    edits = page.get_by_role("link", name=re.compile(r"Editar publicación|Edit publication", re.I))
    for i in range(edits.count()):
        label = (edits.nth(i).get_attribute("aria-label") or edits.nth(i).inner_text()).strip()
        title = re.sub(
            r"^(Editar publicación|Edit publication)\s+",
            "",
            label,
            flags=re.I,
        ).strip()
        if title and title not in titles:
            titles.append(title)
    return titles


def open_new_publication_form(page: Page, vanity: str) -> None:
    page.goto(
        profile_publications_new_url(vanity),
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    page.wait_for_timeout(1500)


def open_edit_publication_form(page: Page, vanity: str, title: str) -> bool:
    page.goto(
        profile_publications_details_url(vanity),
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    page.wait_for_timeout(1500)
    esc = re.escape(title)
    link = page.get_by_role(
        "link", name=re.compile(rf"Editar publicación\s+{esc}", re.I)
    )
    if link.count() == 0:
        link = page.get_by_role(
            "link", name=re.compile(rf"Edit publication\s+{esc}", re.I)
        )
    if link.count() == 0:
        # Partial title match
        link = page.get_by_role(
            "link", name=re.compile(r"Editar publicación|Edit publication", re.I)
        ).filter(has_text=title[:40])
    if link.count() == 0:
        return False
    link.first.click()
    page.wait_for_timeout(1500)
    return True


def fill_publication_form(page: Page, item: LinkedInPublicationItem) -> list[str]:
    """
    Fill the LinkedIn publication modal/form (ES/EN labels).

    Returns field names that could not be filled. Does NOT click Save.
    """
    missing: list[str] = []
    if not _fill_by_placeholder(
        page,
        ("P. ej.: procesos de evaluación", "Ex: Evaluation processes", "Title"),
        item.title,
    ) and not _fill_labeled(page, ("Título", "Title"), item.title):
        missing.append("title")

    if item.publisher and not (
        _fill_by_placeholder(page, ("P. ej.: Espasa", "Ex: Acme"), item.publisher)
        or _fill_labeled(
            page,
            ("Publicación/Editorial", "Publication/Publisher", "Publisher", "Publicación"),
            item.publisher,
        )
    ):
        missing.append("publisher")

    if item.url and not _fill_labeled(
        page,
        ("URL de la publicación", "Publication URL", "URL"),
        item.url,
    ):
        missing.append("url")

    if item.year is not None and not _set_publication_date(page, item.year):
        missing.append("year")

    # Full author list always goes in Description (LinkedIn Autor is member-only typeahead).
    authors_line = _authors_description(item)
    if authors_line and not _fill_labeled(page, ("Descripción", "Description"), authors_line):
        missing.append("description")

    for coauthor in item.coauthors:
        if not _add_coauthor(page, coauthor):
            missing.append(f"coauthor:{coauthor}")

    return missing


def save_publication(page: Page) -> bool:
    for name in ("Guardar", "Save"):
        btn = page.get_by_role("button", name=name, exact=True)
        if btn.count() and btn.first.is_visible():
            btn.first.click()
            page.wait_for_timeout(2000)
            return True
    primary = page.locator("button.artdeco-button--primary:visible")
    if primary.count():
        primary.first.click()
        page.wait_for_timeout(2000)
        return True
    return False


def _authors_description(item: LinkedInPublicationItem) -> str:
    if item.authors:
        return "Autores: " + "; ".join(item.authors)
    if item.coauthors:
        return "Coautores: " + "; ".join(item.coauthors)
    return ""


def _fill_labeled(page: Page, labels: tuple[str, ...], value: str) -> bool:
    for label in labels:
        field = page.get_by_label(label, exact=False)
        for i in range(field.count()):
            el = field.nth(i)
            try:
                if el.is_visible() and el.is_editable():
                    el.click()
                    el.fill(value)
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


def _fill_by_placeholder(page: Page, placeholders: tuple[str, ...], value: str) -> bool:
    for ph in placeholders:
        loc = page.locator(f'input[placeholder="{ph}"], textarea[placeholder="{ph}"]')
        for i in range(loc.count()):
            el = loc.nth(i)
            try:
                if el.is_visible() and el.is_editable():
                    el.click()
                    el.fill(value)
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


def _set_publication_date(page: Page, year: int) -> bool:
    """LinkedIn ES uses date-picker-input with dd/mm/aaaa."""
    date = page.locator('[data-testid="date-picker-input"]')
    if date.count() and date.first.is_visible():
        try:
            # Mid-year default when only year is known from Crossref.
            value = f"01/06/{year}"
            date.first.click()
            date.first.fill(value)
            page.keyboard.press("Tab")
            page.wait_for_timeout(300)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("date-picker fill failed: %s", exc)
    for label in ("Fecha de publicación", "Publication date", "Year", "Año"):
        combo = page.get_by_label(label, exact=False)
        if combo.count() and combo.first.is_visible():
            try:
                combo.first.fill(f"01/06/{year}")
                return True
            except PlaywrightTimeout:
                continue
            except Exception:  # noqa: BLE001
                continue
    return False


def _add_coauthor(page: Page, name: str) -> bool:
    """Open Autor typeahead (member search). Returns False if no LinkedIn match."""
    dialog = page.get_by_role("dialog")
    root = dialog.first if dialog.count() else page

    for add_label in ("Añadir autor", "Agregar autor", "Add author", "Add another author"):
        add_btn = root.get_by_role("button", name=add_label, exact=False)
        if add_btn.count() and add_btn.first.is_visible():
            try:
                add_btn.first.click(timeout=3000)
                page.wait_for_timeout(500)
            except Exception:  # noqa: BLE001
                pass
            break

    # Prefer typeahead inside dialog (not global search).
    candidates = [
        root.locator('[data-testid="typeahead-input"]'),
        root.locator('input[placeholder="Buscar"]'),
        root.get_by_label(re.compile(r"Autor|Author|Buscar", re.I)),
    ]
    for loc in candidates:
        if loc.count() == 0:
            continue
        el = loc.last
        try:
            if not el.is_visible():
                continue
            el.click(timeout=3000)
            el.fill(name)
            page.wait_for_timeout(1200)
            first = re.escape(name.split()[0])
            opt = page.get_by_role("option").filter(
                has_text=re.compile(first, re.I)
            )
            if opt.count():
                opt.first.click()
                page.wait_for_timeout(400)
                return True
            # No member match — clear and report missing (description still has names).
            el.fill("")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("coauthor typeahead failed for %s: %s", name, exc)
            continue
    return False
