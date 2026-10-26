"""Playwright helpers to edit Indeed Resume UI (no PDF upload)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from jobbot.adapters.indeed.package import build_indeed_sync_package
from jobbot.models.candidate import Candidate
from jobbot.models.experience import Experience

logger = logging.getLogger("jobbot.indeed.resume_edit")

INDEED_RESUME = "https://profile.indeed.com/resume"
RESUME_SECTIONS = (
    '[data-testid="work-experience-section"],'
    '[data-testid="education-section"],'
    '[data-testid="skills-section"]'
)
INDEED_EXPERIENCE_ADD = "https://profile.indeed.com/resume/experience/add"
INDEED_CONTACT_EDIT = "https://profile.indeed.com/edit/contact"

_MONTHS_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


@dataclass(frozen=True)
class ResumeEditResult:
    summary: bool = False
    headline: bool = False
    experiences_added: int = 0
    skills_added: int = 0
    education_added: int = 0
    errors: tuple[str, ...] = ()


def open_resume(page: Page) -> None:
    page.goto(INDEED_RESUME, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(1500)


def wait_for_resume_render(page: Page, *, timeout_ms: int = 15_000) -> bool:
    """Resume sections are client-rendered: HTML read earlier is an empty shell."""
    try:
        page.wait_for_selector(RESUME_SECTIONS, timeout=timeout_ms)
    except PlaywrightTimeout:
        return False
    return True


def set_summary(page: Page, summary: str) -> None:
    open_resume(page)
    # Empty state or existing section edit
    empty = page.get_by_test_id("summary-empty-state")
    if empty.count() and empty.first.is_visible():
        empty.first.click()
    else:
        section = page.get_by_test_id("summary-section")
        edit = section.get_by_role("button").first
        if edit.count():
            edit.click()
        else:
            section.click()
    page.wait_for_timeout(800)
    editor = page.get_by_role("textbox", name="Resumen")
    if editor.count() == 0:
        editor = page.locator('[aria-label="Resumen"]')
    editor.first.click()
    editor.first.fill(summary)
    _click_guardar(page)
    page.wait_for_timeout(1200)


def set_headline_via_contact(page: Page, headline: str) -> None:
    page.goto(INDEED_CONTACT_EDIT, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(1000)
    # Indeed sometimes exposes Título as a button that reveals an input
    title_btn = page.get_by_role("button", name="Título")
    if title_btn.count() and title_btn.first.is_visible():
        title_btn.first.click()
        page.wait_for_timeout(500)
    filled = False
    for name in ("Título", "Title", "Headline"):
        field = page.get_by_label(name, exact=False)
        for i in range(field.count()):
            el = field.nth(i)
            try:
                if el.is_visible() and el.is_editable():
                    el.fill(headline)
                    filled = True
                    break
            except Exception:  # noqa: BLE001
                continue
        if filled:
            break
    if not filled:
        # Visible text inputs: Nombre, Apellido, then Título often 3rd
        inputs = page.locator('input[type="text"]:visible')
        if inputs.count() >= 3:
            inputs.nth(2).fill(headline)
            filled = True
    if not filled:
        msg = "Could not find editable Título/headline field on contact edit"
        raise RuntimeError(msg)
    _click_guardar(page)
    page.wait_for_timeout(1200)


def add_experience(page: Page, exp: Experience) -> None:
    page.goto(INDEED_EXPERIENCE_ADD, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(1000)
    page.get_by_test_id("job-title-input-autocomplete-input").fill(exp.title)
    page.get_by_test_id("company-input-autocomplete-input").fill(exp.company)
    if exp.location:
        page.get_by_test_id("location-input-autocomplete-input").fill(exp.location)
        page.keyboard.press("Escape")  # dismiss autocomplete

    start_y, start_m = _parse_ym(exp.start_date)
    _select_month_year(page, "from", start_m, start_y)

    if exp.current:
        toggle = page.get_by_test_id("is-current-toggle")
        if toggle.count():
            toggle.click()
    else:
        end = exp.end_date or exp.start_date
        end_y, end_m = _parse_ym(end)
        _select_month_year(page, "to", end_m, end_y)

    description = _experience_description(exp)
    desc = page.get_by_role("textbox", name="Descripción")
    if desc.count() == 0:
        desc = page.locator('[aria-label="Descripción"]')
    if desc.count():
        desc.first.click()
        desc.first.fill(description)

    _click_guardar(page)
    page.wait_for_timeout(1500)


def add_skills(page: Page, skills: list[str], *, existing: set[str] | None = None) -> int:
    existing_cf = {s.casefold() for s in (existing or set())}
    try:
        open_resume(page)
        section = page.get_by_test_id("skills-section")
        if section.count():
            chips = section.locator('[data-testid^="edit-chip-"]').all_text_contents()
            existing_cf |= {c.strip().casefold() for c in chips if c.strip()}
    except Exception:  # noqa: BLE001
        pass

    added = 0
    for skill in skills:
        if skill.casefold() in existing_cf:
            continue
        try:
            page.goto(
                "https://profile.indeed.com/resume/skills/add",
                wait_until="domcontentloaded",
                timeout=60_000,
            )
            page.wait_for_timeout(600)
            page.get_by_test_id("skill-name-input-autocomplete-input").fill(skill)
            page.wait_for_timeout(300)
            page.keyboard.press("Escape")
            _click_guardar(page)
            page.wait_for_timeout(800)
            existing_cf.add(skill.casefold())
            added += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("skill add failed for %s: %s", skill, exc)
    return added


def add_education(page: Page, edu: object) -> None:
    from jobbot.models.education import Education

    assert isinstance(edu, Education)
    # Direct URL often renders an empty shell; open via Resume CTA.
    open_resume(page)
    btn = page.get_by_role("button", name="Agregar información sobre escolaridad")
    if btn.count():
        btn.first.click()
        page.wait_for_timeout(1000)
    else:
        page.goto(
            "https://profile.indeed.com/resume/education/add",
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        page.wait_for_timeout(800)

    page.get_by_test_id("level-of-education-input-autocomplete-input").fill(edu.degree)
    page.keyboard.press("Escape")
    # Prefer full institution name when short acronyms confuse Indeed
    school = edu.institution
    if school.upper() == "PUC":
        school = "Pontificia Universidad Católica de Chile"
    page.get_by_test_id("school-input-autocomplete-input").fill(school)
    page.keyboard.press("Escape")
    if edu.start_date:
        y, m = _parse_ym(edu.start_date)
        _select_education_month_year(page, "from", m, y)
    if edu.end_date:
        y, m = _parse_ym(edu.end_date)
        _select_education_month_year(page, "to", m, y)
    else:
        toggle = page.get_by_text("Actualmente estudio aquí", exact=False)
        if toggle.count():
            toggle.first.click()
    if edu.details:
        desc = page.get_by_role("textbox", name="Descripción")
        if desc.count() == 0:
            desc = page.locator('[aria-label="Descripción"]')
        if desc.count():
            desc.first.fill(edu.details)
    _click_guardar(page)
    page.wait_for_timeout(1200)


def apply_full_resume_from_candidate(
    page: Page,
    candidate: Candidate,
    *,
    skip_existing_companies: bool = True,
) -> ResumeEditResult:
    """Push headline, summary, experiences, skills, education from Candidate."""
    package = build_indeed_sync_package(candidate)
    errors: list[str] = []
    summary_ok = headline_ok = False
    exp_added = 0
    skills_added = 0
    edu_added = 0

    try:
        set_summary(page, package.summary)
        summary_ok = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"summary: {exc}")

    try:
        set_headline_via_contact(page, package.headline)
        headline_ok = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"headline: {exc}")

    existing_companies: set[str] = set()
    if skip_existing_companies:
        try:
            open_resume(page)
            text = page.get_by_test_id("work-experience-section").inner_text().casefold()
            existing_companies = {
                e.company.casefold()
                for e in candidate.experience
                if e.company.casefold() in text
                or e.company.split(",")[0].casefold() in text
            }
            # Also match partial "Centro de modelamiento"
            for e in candidate.experience:
                key = e.company.casefold()
                short = key.split(",")[0].strip()
                if short and short in text:
                    existing_companies.add(key)
        except Exception:  # noqa: BLE001
            existing_companies = set()

    for exp in candidate.experience:
        if exp.company.casefold() in existing_companies:
            logger.info("skip existing company %s", exp.company)
            continue
        try:
            add_experience(page, exp)
            exp_added += 1
            existing_companies.add(exp.company.casefold())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"experience {exp.company}: {exc}")
            logger.warning("experience add failed: %s", exc)

    try:
        skills_added = add_skills(page, package.skills)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"skills: {exc}")

    try:
        open_resume(page)
        edu_text = page.get_by_test_id("education-section").inner_text().casefold()
    except Exception:  # noqa: BLE001
        edu_text = ""
    for edu in candidate.education:
        if edu.institution.casefold() in edu_text or edu.degree.casefold() in edu_text:
            continue
        try:
            add_education(page, edu)
            edu_added += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"education {edu.institution}: {exc}")

    return ResumeEditResult(
        summary=summary_ok,
        headline=headline_ok,
        experiences_added=exp_added,
        skills_added=skills_added,
        education_added=edu_added,
        errors=tuple(errors),
    )


def parse_resume_page_text(body: str) -> dict[str, object]:
    """Lightweight parse of /resume body text for snapshots/verification."""
    summary = None
    m = re.search(
        r"Resumen\s*(.*?)\s*Datos personales",
        body,
        flags=re.S | re.I,
    )
    if m:
        summary = re.sub(r"\s+", " ", m.group(1)).strip() or None
        if summary and "proporciona un resumen" in summary.casefold():
            summary = None
    skills: list[str] = []
    sm = re.search(
        r"Habilidades\s*(.*?)\s*(?:Certificaciones|Reconocimientos|Links|Agregar)",
        body,
        flags=re.S | re.I,
    )
    if sm:
        chunk = sm.group(1)
        for line in re.split(r"[\n•]", chunk):
            t = line.strip()
            if 0 < len(t) < 60 and "agregar" not in t.casefold():
                skills.append(t)
    return {"summary": summary, "skills": skills, "raw_len": len(body)}


def _experience_description(exp: Experience) -> str:
    lines: list[str] = []
    if exp.description:
        lines.append(exp.description.strip())
    for ach in exp.achievements:
        lines.append(f"• {ach.text}")
    return "\n".join(lines).strip()


def _parse_ym(ym: str) -> tuple[int, int]:
    year_s, month_s = ym.split("-", 1)
    return int(year_s), int(month_s)


def _select_month_year(page: Page, which: str, month: int, year: int) -> None:
    """which: 'from' or 'to' (work experience testids)."""
    month_id = f"work-experience-date-range-{which}-month"
    year_id = f"work-experience-date-range-{which}-year"
    _select_month_year_ids(page, month_id, year_id, month, year)


def _select_education_month_year(page: Page, which: str, month: int, year: int) -> None:
    month_id = f"education-date-range-{which}-month"
    year_id = f"education-date-range-{which}-year"
    _select_month_year_ids(page, month_id, year_id, month, year)


def _select_month_year_ids(
    page: Page, month_id: str, year_id: str, month: int, year: int
) -> None:
    month_label = _MONTHS_ES[month]
    year_label = str(year)

    page.get_by_test_id(month_id).locator('[data-testid="select-button"]').click()
    page.wait_for_timeout(400)
    if not _click_visible_text(page, month_label):
        logger.warning("month option %s not visible; continuing with year", month_label)

    page.get_by_test_id(year_id).locator('[data-testid="select-button"]').click()
    page.wait_for_timeout(400)
    if not _click_visible_text(page, year_label):
        msg = f"Year option not found: {year_label}"
        raise RuntimeError(msg)


def _click_visible_text(page: Page, name: str) -> bool:
    loc = page.get_by_text(name, exact=True)
    for i in range(loc.count()):
        el = loc.nth(i)
        try:
            if el.is_visible():
                el.click(timeout=5_000)
                page.wait_for_timeout(200)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _click_option(page: Page, name: str) -> None:
    if not _click_visible_text(page, name):
        msg = f"Option not found: {name}"
        raise RuntimeError(msg)


def _select_month_year_by_aria(page: Page, prefix: str, month: int, year: int) -> None:
    """Fallback when range testids are unavailable."""
    month_label = _MONTHS_ES[month]
    year_label = str(year)
    page.get_by_role("button", name=f"{prefix} month").click()
    page.wait_for_timeout(300)
    _click_option(page, month_label)
    page.get_by_role("button", name=f"{prefix} year").click()
    page.wait_for_timeout(300)
    _click_option(page, year_label)


def _click_guardar(page: Page) -> None:
    btn = page.get_by_role("button", name="Guardar")
    if not btn.count():
        btn = page.locator('button:has-text("Guardar")')
    if not btn.count():
        msg = "Guardar button not found"
        raise RuntimeError(msg)
    try:
        btn.first.click(timeout=10_000)
    except PlaywrightTimeout:
        btn.first.click(force=True, timeout=5_000)
