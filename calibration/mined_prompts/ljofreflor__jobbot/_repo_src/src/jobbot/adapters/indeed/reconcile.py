"""Reconcile Indeed Resume to fully mirror Candidate (profile.yaml / LaTeX baseline)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from playwright.sync_api import Page

from jobbot.adapters.indeed.package import build_indeed_sync_package
from jobbot.adapters.indeed.resume_edit import (
    _click_guardar,
    _click_visible_text,
    _experience_description,
    _parse_ym,
    _select_month_year,
    add_education,
    add_experience,
    add_skills,
    open_resume,
    set_headline_via_contact,
    set_summary,
)
from jobbot.models.candidate import Candidate
from jobbot.models.education import Education
from jobbot.models.experience import Experience

logger = logging.getLogger("jobbot.indeed.reconcile")


@dataclass
class ReconcileResult:
    summary: bool = False
    headline: bool = False
    experiences_updated: int = 0
    experiences_added: int = 0
    experiences_deleted: int = 0
    education_updated: int = 0
    education_added: int = 0
    education_deleted: int = 0
    skills_added: int = 0
    skills_removed: int = 0
    errors: list[str] = field(default_factory=list)


def reconcile_resume_to_candidate(page: Page, candidate: Candidate) -> ReconcileResult:
    """Make Indeed Resume a full mirror of Candidate facts (update/add/delete)."""
    package = build_indeed_sync_package(candidate)
    result = ReconcileResult()

    try:
        set_summary(page, package.summary)
        result.summary = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"summary: {exc}")

    try:
        set_headline_via_contact(page, package.headline)
        result.headline = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"headline: {exc}")

    try:
        _reconcile_experiences(page, candidate, result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"experiences: {exc}")
        logger.exception("experience reconcile failed")

    try:
        _reconcile_education(page, candidate, result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"education: {exc}")
        logger.exception("education reconcile failed")

    try:
        _reconcile_skills(page, package.skills, result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"skills: {exc}")

    return result


def match_experience(remote_title: str, remote_company: str, local: Experience) -> float:
    """Score 0..1 how well a remote Indeed row matches a local experience."""
    rt, rc = _norm(remote_title), _norm(remote_company)
    lt, lc = _norm(local.title), _norm(local.company)
    score = 0.0
    if rt and (rt == lt or rt in lt or lt in rt):
        score += 0.55
    if rc and (rc == lc or rc in lc or lc in rc or _company_overlap(rc, lc)):
        score += 0.45
    return score


def match_education(remote_degree: str, remote_school: str, local: Education) -> float:
    rd, rs = _norm(remote_degree), _norm(remote_school)
    ld, ls = _norm(local.degree), _norm(_expand_school(local.institution))
    score = 0.0
    if rd and (rd == ld or rd in ld or ld in rd):
        score += 0.6
    if rs and (rs == ls or rs in ls or ls in rs):
        score += 0.4
    # Incomplete engineering row often has empty school
    if "ingenier" in rd and "ingenier" in ld and not rs:
        score = max(score, 0.7)
    return score


def _reconcile_experiences(page: Page, candidate: Candidate, result: ReconcileResult) -> None:
    open_resume(page)
    labels = _experience_edit_labels(page)
    matched_local: set[str] = set()

    for label in labels:
        title_hint = label.removeprefix("Editar experiencia laboral de ").strip()
        page.get_by_role("button", name=label, exact=True).click()
        page.wait_for_timeout(1200)
        remote_title = page.get_by_test_id("job-title-input-autocomplete-input").input_value()
        remote_company = page.get_by_test_id("company-input-autocomplete-input").input_value()

        best: Experience | None = None
        best_score = 0.0
        for exp in candidate.experience:
            if exp.id in matched_local:
                continue
            score = match_experience(remote_title or title_hint, remote_company, exp)
            if score > best_score:
                best_score = score
                best = exp

        if best is not None and best_score >= 0.45:
            _fill_experience_form(page, best)
            _click_guardar(page)
            page.wait_for_timeout(1200)
            matched_local.add(best.id)
            result.experiences_updated += 1
            logger.info(
                "updated experience %s @ %s (score=%.2f)",
                best.title,
                best.company,
                best_score,
            )
        else:
            # Legacy Indeed-only row → delete
            if _delete_current_form(page):
                result.experiences_deleted += 1
                logger.info("deleted legacy experience %s @ %s", remote_title, remote_company)
            else:
                result.errors.append(f"could not delete experience {remote_title}")
                open_resume(page)
        open_resume(page)

    for exp in candidate.experience:
        if exp.id in matched_local:
            continue
        try:
            add_experience(page, exp)
            result.experiences_added += 1
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"add experience {exp.company}: {exc}")


def _reconcile_education(page: Page, candidate: Candidate, result: ReconcileResult) -> None:
    open_resume(page)
    labels = _education_edit_labels(page)
    # Also open incomplete alert entry if present
    alert_btn = page.get_by_role("button", name="Edita la información de tu escolaridad")
    matched_local: set[str] = set()

    edit_targets: list[str | None] = list(labels)
    if alert_btn.count() and alert_btn.first.is_visible():
        edit_targets.append(None)  # special: alert button

    for label in edit_targets:
        open_resume(page)
        if label is None:
            btn = page.get_by_role("button", name="Edita la información de tu escolaridad")
            if not btn.count() or not btn.first.is_visible():
                continue
            btn.first.click()
        else:
            page.get_by_role("button", name=label, exact=True).click()
        page.wait_for_timeout(1200)

        remote_degree = page.get_by_test_id(
            "level-of-education-input-autocomplete-input"
        ).input_value()
        remote_school = page.get_by_test_id("school-input-autocomplete-input").input_value()

        best: Education | None = None
        best_score = 0.0
        for edu in candidate.education:
            if edu.id in matched_local:
                continue
            score = match_education(remote_degree, remote_school, edu)
            if score > best_score:
                best_score = score
                best = edu

        if best is not None and best_score >= 0.5:
            _fill_education_form(page, best)
            _click_guardar(page)
            page.wait_for_timeout(1200)
            matched_local.add(best.id)
            result.education_updated += 1
        else:
            if _delete_current_form(page):
                result.education_deleted += 1
            else:
                result.errors.append(f"could not delete education {remote_degree}")
                open_resume(page)

    for edu in candidate.education:
        if edu.id in matched_local:
            continue
        try:
            add_education(page, edu)
            result.education_added += 1
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"add education {edu.institution}: {exc}")


def _reconcile_skills(page: Page, desired: list[str], result: ReconcileResult) -> None:
    open_resume(page)
    desired_cf = {s.casefold() for s in desired}
    section = page.get_by_test_id("skills-section")
    existing = [
        t.strip()
        for t in section.locator('[data-testid^="edit-chip-"]').all_text_contents()
        if t.strip()
    ]

    for skill in existing:
        if skill.casefold() in desired_cf:
            continue
        try:
            open_resume(page)
            chip = page.get_by_test_id("skills-section").locator(
                f'[data-testid^="edit-chip-"]:has-text("{skill}")'
            )
            if not chip.count():
                continue
            chip.first.click()
            page.wait_for_timeout(800)
            if _delete_current_form(page):
                result.skills_removed += 1
            else:
                page.keyboard.press("Escape")
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"remove skill {skill}: {exc}")

    open_resume(page)
    chips_now = {
        t.strip().casefold()
        for t in page.get_by_test_id("skills-section")
        .locator('[data-testid^="edit-chip-"]')
        .all_text_contents()
        if t.strip()
    }
    missing = [s for s in desired if s.casefold() not in chips_now]
    if missing:
        result.skills_added += add_skills(page, missing, existing=chips_now)


def _fill_experience_form(page: Page, exp: Experience) -> None:
    page.get_by_test_id("job-title-input-autocomplete-input").fill(exp.title)
    page.keyboard.press("Escape")
    page.get_by_test_id("company-input-autocomplete-input").fill(exp.company)
    page.keyboard.press("Escape")
    if exp.location:
        page.get_by_test_id("location-input-autocomplete-input").fill(exp.location)
        page.keyboard.press("Escape")

    start_y, start_m = _parse_ym(exp.start_date)
    _select_month_year(page, "from", start_m, start_y)

    current = page.get_by_test_id("is-current-toggle")
    # Toggle state is opaque; set end dates or current explicitly via UI clicks carefully
    if exp.current:
        # If end fields still enabled, click current
        if current.count():
            # Only click if "Trabajo aquí actualmente" appears unchecked — best-effort
            current.click()
            page.wait_for_timeout(200)
            # If we toggled off by mistake, click again when end date required — skip heuristics
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


def _fill_education_form(page: Page, edu: Education) -> None:
    page.get_by_test_id("level-of-education-input-autocomplete-input").fill(edu.degree)
    page.keyboard.press("Escape")
    page.get_by_test_id("school-input-autocomplete-input").fill(_expand_school(edu.institution))
    page.keyboard.press("Escape")
    if edu.start_date:
        y, m = _parse_ym(edu.start_date)
        _select_edu_dates(page, "from", m, y)
    if edu.end_date:
        y, m = _parse_ym(edu.end_date)
        _select_edu_dates(page, "to", m, y)
    else:
        toggle = page.get_by_text("Actualmente estudio aquí", exact=False)
        if toggle.count():
            toggle.first.click()
    if edu.details:
        desc = page.locator('[aria-label="Descripción"]')
        if desc.count():
            desc.first.fill(edu.details)


def _select_edu_dates(page: Page, which: str, month: int, year: int) -> None:
    from jobbot.adapters.indeed.resume_edit import _MONTHS_ES

    month_id = f"education-date-range-{which}-month"
    year_id = f"education-date-range-{which}-year"
    page.get_by_test_id(month_id).locator('[data-testid="select-button"]').click()
    page.wait_for_timeout(400)
    _click_visible_text(page, _MONTHS_ES[month])
    page.get_by_test_id(year_id).locator('[data-testid="select-button"]').click()
    page.wait_for_timeout(400)
    if not _click_visible_text(page, str(year)):
        msg = f"education year not found: {year}"
        raise RuntimeError(msg)


def _delete_current_form(page: Page) -> bool:
    # Prefer explicit "delete this section" over field clear buttons ("Borrar Grado…")
    for name in (
        "Eliminar esta escolaridad",
        "Eliminar esta experiencia laboral",
        "Eliminar esta habilidad",
        "Eliminar",
        "Borrar",
        "Delete",
    ):
        btn = page.get_by_role("button", name=name)
        if not btn.count():
            continue
        for i in range(btn.count()):
            el = btn.nth(i)
            try:
                if not el.is_visible():
                    continue
                el.click(timeout=8_000)
                page.wait_for_timeout(800)
                for conf in (
                    "Eliminar esta escolaridad",
                    "Eliminar esta experiencia laboral",
                    "Eliminar",
                    "Confirmar",
                    "Sí",
                    "Yes",
                    "Delete",
                ):
                    c = page.get_by_role("button", name=conf)
                    for j in range(c.count()):
                        ce = c.nth(j)
                        if ce.is_visible():
                            ce.click(timeout=5_000)
                            page.wait_for_timeout(800)
                            break
                return True
            except Exception:  # noqa: BLE001
                continue
    return False


def _click_eliminar_skill(page: Page) -> bool:
    return _delete_current_form(page)


def _experience_edit_labels(page: Page) -> list[str]:
    labels: list[str] = []
    for btn in page.get_by_test_id("work-experience-section").locator("button").all():
        aria = (btn.get_attribute("aria-label") or "").strip()
        if aria.startswith("Editar experiencia laboral de "):
            labels.append(aria)
    return labels


def _education_edit_labels(page: Page) -> list[str]:
    labels: list[str] = []
    for btn in page.get_by_test_id("education-section").locator("button").all():
        aria = (btn.get_attribute("aria-label") or "").strip()
        if aria.startswith("Editar la información de escolaridad de "):
            labels.append(aria)
    return labels


def _expand_school(institution: str) -> str:
    if institution.strip().upper() == "PUC":
        return "Pontificia Universidad Católica de Chile"
    return institution


def _norm(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _company_overlap(a: str, b: str) -> bool:
    tokens_a = {t for t in a.split() if len(t) > 3}
    tokens_b = {t for t in b.split() if len(t) > 3}
    if not tokens_a or not tokens_b:
        return False
    return len(tokens_a & tokens_b) >= 2
