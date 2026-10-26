"""Fill an external ATS form in your own Chrome (CDP) and stop before submit.

The plan is pure: which inputs of an observed form ``prefill_field_map`` already
answers, and which file input takes the CV. A field is filled only when its
autocomplete token, its label or its name says unambiguously what it is; any
other question is handed back to the human. Nothing here clicks: no submit, no
"next", no OAuth button, no terms checkbox. A password on the page means a login
wall, and then nothing is typed at all.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobbot.adapters.ats.apply import prefill_field_map
from jobbot.browser.background import new_background_page
from jobbot.models.candidate import Candidate
from jobbot.portals.form_learn import FieldKind, FormField, FormKnowledge, learn_form_html

logger = logging.getLogger("jobbot.ats.apply_fill")

ALWAYS_HUMAN: tuple[str, ...] = (
    "Final submit / apply button",
    "Screening questions, salary, visa and consent",
    "CAPTCHA / 2FA if shown",
)

_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "password",
    "contrasena",
    "passcode",
    "captcha",
    "recaptcha",
    "otp",
    "verification",
    "verificacion",
    "terms",
    "terminos",
    "privacy",
    "privacidad",
    "consent",
    "acepto",
    "agree",
    "refer",
    "emergency",
    "emergencia",
    "confirm",
)

_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "your",
        "enter",
        "tu",
        "su",
        "ingresa",
        "ingrese",
        "address",
        "url",
        "urls",
        "link",
        "profile",
        "perfil",
        "number",
        "numero",
        "nro",
        "current",
        "actual",
        "the",
        "of",
        "de",
        "del",
        "la",
        "el",
        "please",
    }
)

_LABEL_KEYS: dict[str, str] = {
    "name": "full_name",
    "full name": "full_name",
    "legal name": "full_name",
    "full legal name": "full_name",
    "nombre completo": "full_name",
    "nombre y apellido": "full_name",
    "nombre y apellidos": "full_name",
    "nombre": "full_name",
    "first name": "first_name",
    "firstname": "first_name",
    "fname": "first_name",
    "given name": "first_name",
    "nombres": "first_name",
    "primer nombre": "first_name",
    "last name": "last_name",
    "lastname": "last_name",
    "lname": "last_name",
    "surname": "last_name",
    "family name": "last_name",
    "apellido": "last_name",
    "apellidos": "last_name",
    "email": "email",
    "e mail": "email",
    "mail": "email",
    "correo": "email",
    "correo electronico": "email",
    "phone": "phone",
    "mobile": "phone",
    "mobile phone": "phone",
    "phone mobile": "phone",
    "cell phone": "phone",
    "telephone": "phone",
    "telefono": "phone",
    "celular": "phone",
    "movil": "phone",
    "linkedin": "linkedin",
    "city": "city",
    "ciudad": "city",
    "country": "country",
    "pais": "country",
    "country residence": "country",
    "pais residencia": "country",
    "location": "location",
    "ubicacion": "location",
}

_AUTOCOMPLETE_KEYS: dict[str, str] = {
    "name": "full_name",
    "given-name": "first_name",
    "family-name": "last_name",
    "email": "email",
    "tel": "phone",
    "tel-national": "phone",
    "address-level2": "city",
    "country-name": "country",
}

# What each value may be typed into. Textareas are questions, never identity.
_KINDS_FOR_KEY: dict[str, frozenset[FieldKind]] = {
    "email": frozenset({FieldKind.EMAIL, FieldKind.TEXT}),
    "phone": frozenset({FieldKind.PHONE, FieldKind.TEXT}),
    "linkedin": frozenset({FieldKind.URL, FieldKind.TEXT}),
}
_TEXT_ONLY: frozenset[FieldKind] = frozenset({FieldKind.TEXT})

_RESUME_WORDS: tuple[str, ...] = ("resume", "curriculum", "hoja de vida")
_NOT_A_CV_WORDS: tuple[str, ...] = (
    "cover",
    "carta",
    "motivation",
    "photo",
    "foto",
    "picture",
    "imagen",
    "avatar",
    "portfolio",
)

_LOGIN_URL_HINTS: tuple[str, ...] = ("/login", "/signin", "/sign-in", "/sign_in", "/auth")
_PASSWORD_INPUT = re.compile(r"<input\b[^>]*\btype\s*=\s*[\"']?password", re.IGNORECASE)
_CAPTCHA_MARKERS: tuple[str, ...] = ("g-recaptcha", "h-captcha", "cf-turnstile", "hcaptcha.com")


class ApplyFillError(RuntimeError):
    """The page could not be opened or read over CDP."""


@dataclass(frozen=True)
class FieldFill:
    """One input the profile answers."""

    label: str
    key: str
    selector: str
    value: str


@dataclass(frozen=True)
class ApplyFillPlan:
    """What will be typed and attached. ``will_submit`` is always false."""

    url: str
    readable: bool
    fills: tuple[FieldFill, ...]
    attach_selector: str | None
    cv_path: Path | None
    left_for_human: tuple[str, ...]
    will_submit: bool = False


@dataclass(frozen=True)
class ApplyFillResult:
    """What happened on the page. ``submitted`` is always false."""

    url: str
    readable: bool
    filled: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    attached: bool = False
    left_for_human: tuple[str, ...] = ALWAYS_HUMAN
    login_required: bool = False
    note: str = ""
    submitted: bool = False


def profile_key_for(field_: FormField) -> str | None:
    """The ``prefill_field_map`` key this input asks for, or None when unclear."""
    if field_.kind is FieldKind.FILE or _forbidden(field_):
        return None
    key = _AUTOCOMPLETE_KEYS.get(field_.autocomplete.strip().casefold())
    if key is None:
        key = _key_from_text(field_.label) or _key_from_text(field_.name)
    if key is None:
        return None
    allowed = _KINDS_FOR_KEY.get(key, _TEXT_ONLY)
    return key if field_.kind in allowed else None


def plan_apply_fill(
    form: FormKnowledge,
    candidate: Candidate,
    *,
    cv_path: Path | None = None,
) -> ApplyFillPlan:
    """Pure: which inputs to fill from the profile and where the CV goes."""
    if not form.readable:
        return ApplyFillPlan(
            url=form.url,
            readable=False,
            fills=(),
            attach_selector=None,
            cv_path=None,
            left_for_human=(form.evidence or "no application form on the page", *ALWAYS_HUMAN),
        )
    answers = prefill_field_map(candidate)
    keyed = [(f, profile_key_for(f)) for f in form.fields]
    split_name = any(key == "last_name" for _f, key in keyed)
    fills: list[FieldFill] = []
    used: set[str] = set()
    left: list[str] = []
    for form_field, key in keyed:
        if form_field.kind is FieldKind.FILE:
            continue
        if key == "full_name" and split_name:
            key = "first_name"
        value = answers.get(key or "", "")
        selector = _selector(form_field.name)
        if key is None or not value or selector is None or selector in used:
            left.append(form_field.label)
            continue
        used.add(selector)
        fills.append(FieldFill(label=form_field.label, key=key, selector=selector, value=value))
    attachable = _attachable_pdf(cv_path)
    cv_field = _cv_file_field(form.fields)
    attach_selector = _selector(cv_field.name) if cv_field is not None else None
    if attachable is None or attach_selector is None:
        attach_selector = None
        for form_field in form.fields:
            if form_field.kind is FieldKind.FILE:
                left.append(form_field.label)
    return ApplyFillPlan(
        url=form.url,
        readable=True,
        fills=tuple(fills),
        attach_selector=attach_selector,
        cv_path=attachable if attach_selector else None,
        left_for_human=(*left, *ALWAYS_HUMAN),
    )


def looks_like_login(url: str, html: str) -> bool:
    """A password input or a sign-in route: a wall only the human may cross."""
    if _PASSWORD_INPUT.search(html or ""):
        return True
    path = (url or "").casefold().split("?", 1)[0]
    return any(hint in path for hint in _LOGIN_URL_HINTS) or "//accounts." in path


def has_captcha(html: str) -> bool:
    folded = (html or "").casefold()
    return any(marker in folded for marker in _CAPTCHA_MARKERS)


def fill_open_page(
    page: Any,
    candidate: Candidate,
    *,
    cv_path: Path | None = None,
) -> ApplyFillResult:
    """Fill the first frame of an open page whose form the profile answers. Never clicks."""
    url = str(getattr(page, "url", "") or "")
    main_html = page.content()
    if looks_like_login(url, main_html):
        return ApplyFillResult(
            url=url,
            readable=False,
            login_required=True,
            left_for_human=("Sign in (password / SSO / 2FA are yours)", *ALWAYS_HUMAN),
            note="The page asks for a password (sign-in or account); nothing was typed.",
        )
    captcha = has_captcha(main_html)
    first_plan: ApplyFillPlan | None = None
    for frame, html in _frames_with_html(page, main_html):
        frame_url = str(getattr(frame, "url", "") or url)
        plan = plan_apply_fill(learn_form_html(html, url=frame_url), candidate, cv_path=cv_path)
        if first_plan is None:
            first_plan = plan
        if plan.fills or plan.attach_selector:
            return _execute(frame, plan, url=url, captcha=captcha)
    left = first_plan.left_for_human if first_plan is not None else ALWAYS_HUMAN
    readable = first_plan.readable if first_plan is not None else False
    return ApplyFillResult(
        url=url,
        readable=readable,
        left_for_human=left,
        note=(
            "No input on the page is one the profile answers"
            if readable
            else "Could not read an application form on the page"
        ),
    )


def open_and_fill_over_cdp(
    cdp_url: str,
    url: str,
    candidate: Candidate,
    *,
    cv_path: Path | None = None,
    timeout_ms: int = 30_000,
    playwright_factory: Callable[[], Any] | None = None,
) -> ApplyFillResult:
    """Open ``url`` in a new background tab of your Chrome, fill what is known, then disconnect.

    The tab stays open for review; the browser is never closed nor raised.
    """
    if playwright_factory is None:
        from playwright.sync_api import sync_playwright

        playwright_factory = sync_playwright
    manager = playwright_factory()
    playwright = manager.start()
    try:
        try:
            browser = playwright.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:  # noqa: BLE001 — Playwright error types vary
            msg = f"Could not attach to Chrome at {cdp_url}: {exc}"
            raise ApplyFillError(msg) from exc
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = new_background_page(context, browser=browser)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not load {url}: {exc}"
            raise ApplyFillError(msg) from exc
        _settle(page)
        try:
            return fill_open_page(page, candidate, cv_path=cv_path)
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not read the page at {url}: {exc}"
            raise ApplyFillError(msg) from exc
    finally:
        # Stopping the driver drops the CDP connection; Chrome and the tab stay open.
        playwright.stop()


def _execute(frame: Any, plan: ApplyFillPlan, *, url: str, captcha: bool) -> ApplyFillResult:
    filled: list[str] = []
    kept: list[str] = []
    left = list(plan.left_for_human)
    for item in plan.fills:
        target = _first(frame.locator(item.selector))
        if _current_value(target):
            kept.append(item.label)
            continue
        try:
            target.fill(item.value)
        except Exception as exc:  # noqa: BLE001 — a hidden or disabled input
            logger.debug("Could not fill %s: %s", item.selector, exc)
            left.insert(0, item.label)
            continue
        filled.append(item.label)
    attached = False
    if plan.attach_selector and plan.cv_path is not None:
        try:
            _first(frame.locator(plan.attach_selector)).set_input_files(str(plan.cv_path))
            attached = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not attach CV: %s", exc)
            left.insert(0, "Resume/CV upload")
    if captcha:
        left.insert(0, "CAPTCHA on the page")
    return ApplyFillResult(
        url=url,
        readable=True,
        filled=tuple(filled),
        kept=tuple(kept),
        attached=attached,
        left_for_human=tuple(left),
        note="Filled what the profile answers; review and submit yourself.",
    )


def _frames_with_html(page: Any, main_html: str) -> list[tuple[Any, str]]:
    out: list[tuple[Any, str]] = [(page, main_html)]
    main_frame = getattr(page, "main_frame", None)
    for frame in list(getattr(page, "frames", None) or []):
        if frame is main_frame:
            continue
        try:
            out.append((frame, frame.content()))
        except Exception as exc:  # noqa: BLE001 — detached or cross-origin frame
            logger.debug("Skipping frame: %s", exc)
    return out


def _settle(page: Any) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception as exc:  # noqa: BLE001 — a slow page is still a page
        logger.debug("Page kept loading: %s", exc)
    try:
        page.wait_for_selector("form, input, select, textarea", timeout=5_000)
    except Exception as exc:  # noqa: BLE001 — no form yet is reported, not raised
        logger.debug("No form element yet: %s", exc)


def _first(locator: Any) -> Any:
    return locator.first if hasattr(locator, "first") else locator


def _current_value(target: Any) -> str:
    getter = getattr(target, "input_value", None)
    if getter is None:
        return ""
    try:
        return str(getter() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _key_from_text(text: str) -> str | None:
    # "firstName" reads as two words, "LinkedIn" as one: try both spellings.
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    return _LABEL_KEYS.get(_normalize(text)) or _LABEL_KEYS.get(_normalize(split))


def _normalize(text: str) -> str:
    folded = "".join(
        char
        for char in unicodedata.normalize("NFD", (text or "").casefold())
        if unicodedata.category(char) != "Mn"
    )
    words = re.sub(r"[^a-z0-9]+", " ", folded).split()
    return " ".join(word for word in words if word not in _FILLER_WORDS)


def _forbidden(field_: FormField) -> bool:
    words = f"{_normalize(field_.label)} {_normalize(field_.name)}".split()
    return any(word.startswith(token) for word in words for token in _FORBIDDEN_TOKENS)


def _cv_file_field(fields: list[FormField]) -> FormField | None:
    files = [f for f in fields if f.kind is FieldKind.FILE and not _forbidden(f)]
    for form_field in files:
        blob = f"{_normalize(form_field.label)} {_normalize(form_field.name)}"
        if any(word in blob for word in _RESUME_WORDS) or "cv" in blob.split():
            return form_field
    if len(files) == 1:
        only = files[0]
        blob = f"{_normalize(only.label)} {_normalize(only.name)}"
        accepts_pdf = not only.accepts or any("pdf" in a.casefold() for a in only.accepts)
        if accepts_pdf and not any(word in blob for word in _NOT_A_CV_WORDS):
            return only
    return None


def _attachable_pdf(path: Path | None) -> Path | None:
    if path is None or not path.is_file() or path.suffix.casefold() != ".pdf":
        return None
    with path.open("rb") as handle:
        if not handle.read(5).startswith(b"%PDF"):
            return None
    return path


def _selector(name: str) -> str | None:
    if not name:
        return None
    quoted = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'[name="{quoted}"], [id="{quoted}"]'
