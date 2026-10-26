"""What an application form asks for, read without submitting anything.

This is the return path of the loop: the postings say what a company wants in
prose, but the form says it in fields, and that is what the CV has to satisfy.
Only the questions are kept. Values typed into the page, hidden tokens and
placeholder contact data are not knowledge about the company, so they never reach
the file — a form is read, never answered here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import yaml
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from jobbot.companies.models import KnowledgeStatus, utc_now
from jobbot.companies.urls import PrivateRouteRejected, canonical_key, public_url
from jobbot.ops.pii_guard import redact
from jobbot.portals.detect import AtsKind, detect_ats, detect_ats_in_html
from jobbot.portals.sso import detect_sso_providers


class FieldKind(StrEnum):
    """What the form expects in a field, as the markup declares it."""

    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    NUMBER = "number"
    DATE = "date"
    FILE = "file"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    LONG_TEXT = "long_text"
    UNKNOWN = "unknown"


# Input types that are page furniture, not a question to the candidate.
_IGNORED_TYPES: frozenset[str] = frozenset(
    {"hidden", "submit", "button", "reset", "image", "search", "password"}
)

_TYPE_KINDS: dict[str, FieldKind] = {
    "text": FieldKind.TEXT,
    "email": FieldKind.EMAIL,
    "tel": FieldKind.PHONE,
    "phone": FieldKind.PHONE,
    "url": FieldKind.URL,
    "number": FieldKind.NUMBER,
    "date": FieldKind.DATE,
    "month": FieldKind.DATE,
    "file": FieldKind.FILE,
    "checkbox": FieldKind.CHECKBOX,
    "radio": FieldKind.RADIO,
}

# Fields every form asks and no CV has to argue for.
_IDENTITY_HINTS: tuple[str, ...] = (
    "first name",
    "last name",
    "full name",
    "nombre",
    "apellido",
    "email",
    "correo",
    "e-mail",
    "phone",
    "teléfono",
    "telefono",
    "celular",
    "address",
    "dirección",
    "direccion",
    "city",
    "ciudad",
    "country",
    "país",
    "pais",
    "linkedin",
    "github",
    "website",
    "portfolio",
    "resume",
    "cv",
    "curriculum",
    "currículum",
    "cover letter",
    "carta",
    "privacy",
    "privacidad",
    "consent",
    "consentimiento",
    "terms",
    "términos",
)

# Prompts a select shows before a real choice is made.
_EMPTY_OPTIONS: frozenset[str] = frozenset(
    {
        "",
        "please select",
        "select",
        "select…",
        "select...",
        "selecciona",
        "selecciona…",
        "selecciona...",
        "seleccione",
        "choose",
        "choose one",
        "-- select --",
        "none",
    }
)


class FormField(BaseModel):
    """One question the form asks. Never its answer."""

    name: str
    label: str
    kind: FieldKind = FieldKind.UNKNOWN
    required: bool = False
    options: list[str] = Field(default_factory=list)
    max_length: int | None = None
    accepts: list[str] = Field(default_factory=list)
    is_screening: bool = False
    autocomplete: str = ""


class FormKnowledge(BaseModel):
    """What one application form asks, as observed once."""

    url: str
    company: str | None = None
    ats: AtsKind = AtsKind.UNKNOWN
    fields: list[FormField] = Field(default_factory=list)
    sso_providers: list[str] = Field(default_factory=list)
    readable: bool = False
    evidence: str = ""
    observed_at: datetime = Field(default_factory=utc_now)
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE

    def screening_questions(self) -> list[str]:
        return [field.label for field in self.fields if field.is_screening]

    def required_files(self) -> list[FormField]:
        return [f for f in self.fields if f.kind is FieldKind.FILE and f.required]


class PageLike(Protocol):
    def content(self) -> str: ...

    @property
    def url(self) -> str: ...


def learn_form_html(
    html: str,
    *,
    url: str,
    company: str | None = None,
    ats: AtsKind | None = None,
) -> FormKnowledge:
    """Read the questions out of an application page. Pure: no network, no writes."""
    try:
        stored_url = public_url(url)
    except PrivateRouteRejected:
        stored_url = url
    kind = ats if ats is not None else _ats_of(stored_url, html)
    sso = [p.value for p in detect_sso_providers(html)]
    soup = BeautifulSoup(html or "", "html.parser")
    form = _application_form(soup)
    if form is None:
        return FormKnowledge(
            url=stored_url,
            company=company,
            ats=kind,
            sso_providers=sso,
            readable=False,
            evidence=(
                "unknown: no application form in the served HTML "
                "(client-rendered or behind a login)"
            ),
        )
    fields = _fields(form)
    if not _looks_like_an_application(fields):
        return FormKnowledge(
            url=stored_url,
            company=company,
            ats=kind,
            sso_providers=sso,
            readable=False,
            evidence="unknown: a form was found but it does not ask for an application",
        )
    return FormKnowledge(
        url=stored_url,
        company=company,
        ats=kind,
        fields=fields,
        sso_providers=sso,
        readable=True,
        evidence=f"read {len(fields)} field(s) from the served HTML",
    )


def learn_form_page(
    page: PageLike,
    *,
    url: str | None = None,
    company: str | None = None,
) -> FormKnowledge:
    """Read an already-open form. Only reads the DOM; clicks nothing."""
    return learn_form_html(page.content(), url=url or page.url, company=company)


def _ats_of(url: str, html: str) -> AtsKind:
    """Host first, embedded marker second: both are technical evidence."""
    by_host = detect_ats(url)
    if by_host is not AtsKind.UNKNOWN:
        return by_host
    return detect_ats_in_html(html)[0]


def _application_form(soup: BeautifulSoup) -> Tag | None:
    """The form that asks for an application, chosen by how much it asks."""
    best: Tag | None = None
    best_score = 0
    for form in soup.find_all("form"):
        if not isinstance(form, Tag):
            continue
        score = len(_fields(form))
        if _fields_include_a_file(form):
            score += 3
        if score > best_score:
            best, best_score = form, score
    if best is not None:
        return best
    # Some ATS render the inputs without a <form> wrapper.
    body = soup.body
    return body if isinstance(body, Tag) and body.find(["input", "select", "textarea"]) else None


def _fields_include_a_file(form: Tag) -> bool:
    return any(
        isinstance(node, Tag) and str(node.get("type") or "").lower() == "file"
        for node in form.find_all("input")
    )


def _looks_like_an_application(fields: list[FormField]) -> bool:
    """A search box is a form too; an application asks for more than one thing."""
    if len(fields) < 2:
        return False
    return any(
        field.kind in {FieldKind.FILE, FieldKind.EMAIL} or field.is_screening
        for field in fields
    )


def _fields(form: Tag) -> list[FormField]:
    fields: list[FormField] = []
    seen: set[str] = set()
    for node in form.find_all(["input", "select", "textarea"]):
        if not isinstance(node, Tag):
            continue
        field = _field(node, form)
        if field is None:
            continue
        key = f"{field.name}|{field.label}".casefold()
        if key in seen:
            continue
        seen.add(key)
        fields.append(field)
    return fields


def _field(node: Tag, form: Tag) -> FormField | None:
    tag = node.name.lower()
    node_type = str(node.get("type") or "").lower()
    if tag == "input" and node_type in _IGNORED_TYPES:
        return None
    name = str(
        node.get("name") or node.get("data-automation-id") or node.get("id") or ""
    ).strip()
    if not name:
        return None
    label = _label_for(node, form)
    if not label:
        return None
    kind = _kind(tag, node_type, node=node, label=label)
    if kind is FieldKind.RADIO:
        # A radio group is one question; the first member carries it.
        siblings = [
            sib
            for sib in form.find_all("input")
            if isinstance(sib, Tag) and str(sib.get("name") or "") == name
        ]
        if siblings and siblings[0] is not node:
            return None
        label = _group_label(node, form) or label
        options = [_option_label(sib, form) for sib in siblings]
        return FormField(
            name=name,
            label=label,
            kind=kind,
            required=any(sib.has_attr("required") for sib in siblings),
            options=[opt for opt in options if opt],
            is_screening=_is_screening(label, kind),
        )
    options = _select_options(node) if tag == "select" else []
    required = (
        node.has_attr("required")
        or str(node.get("aria-required") or "").casefold() == "true"
        or "*" in _raw_label(node, form)
    )
    return FormField(
        name=name,
        label=label,
        kind=kind,
        required=required,
        options=options,
        max_length=_int_attr(node, "maxlength"),
        accepts=_accepts(node),
        is_screening=_is_screening(label, kind),
        autocomplete=_autocomplete(node),
    )


def _kind(
    tag: str,
    node_type: str,
    *,
    node: Tag | None = None,
    label: str = "",
) -> FieldKind:
    if tag == "textarea":
        return FieldKind.LONG_TEXT
    if tag == "select":
        return FieldKind.SELECT
    # Workday (and others) often render email as type=text with autocomplete /
    # data-automation-id / a label that says email (#90). Prefer that over the
    # generic "text" type.
    if node is not None and node_type in {"", "text"} and _looks_like_email_control(
        node, label
    ):
        return FieldKind.EMAIL
    by_type = _TYPE_KINDS.get(node_type)
    if by_type is not None:
        return by_type
    if node_type == "":
        return FieldKind.TEXT
    return FieldKind.UNKNOWN

def _looks_like_email_control(node: Tag, label: str) -> bool:
    hints = " ".join(
        [
            label,
            str(node.get("autocomplete") or ""),
            str(node.get("name") or ""),
            str(node.get("id") or ""),
            str(node.get("data-automation-id") or ""),
            str(node.get("aria-label") or ""),
        ]
    ).casefold()
    return "email" in hints or "correo" in hints or "e-mail" in hints


def _raw_label(node: Tag, form: Tag) -> str:
    node_id = str(node.get("id") or "")
    if node_id:
        for candidate in form.find_all("label"):
            if isinstance(candidate, Tag) and candidate.get("for") == node_id:
                return candidate.get_text(" ", strip=True)
    parent = node.find_parent("label")
    if isinstance(parent, Tag):
        return parent.get_text(" ", strip=True)
    aria = str(node.get("aria-label") or "").strip()
    return aria


def _label_for(node: Tag, form: Tag) -> str:
    """The question as the page words it, with contact data taken out.

    A placeholder often carries a real phone or address as the example, and that
    belongs to whoever wrote the page, not to our knowledge base.
    """
    text = _raw_label(node, form) or str(node.get("placeholder") or "")
    clean = " ".join(text.replace("*", " ").split()).strip(" :")
    return redact(clean)


def _group_label(node: Tag, form: Tag) -> str:
    fieldset = node.find_parent("fieldset")
    if isinstance(fieldset, Tag):
        legend = fieldset.find("legend")
        if isinstance(legend, Tag):
            return redact(legend.get_text(" ", strip=True))
    return ""


def _option_label(node: Tag, form: Tag) -> str:
    parent = node.find_parent("label")
    if isinstance(parent, Tag):
        return redact(parent.get_text(" ", strip=True))
    return ""


def _select_options(node: Tag) -> list[str]:
    options: list[str] = []
    for option in node.find_all("option"):
        if not isinstance(option, Tag):
            continue
        text = option.get_text(" ", strip=True)
        if text.casefold() in _EMPTY_OPTIONS:
            continue
        options.append(redact(text))
    return options


def _accepts(node: Tag) -> list[str]:
    raw = str(node.get("accept") or "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _autocomplete(node: Tag) -> str:
    """The field's declared purpose (``given-name``, ``email``), not a value."""
    tokens = str(node.get("autocomplete") or "").casefold().split()
    if not tokens or tokens[-1] in {"on", "off"}:
        return ""
    return tokens[-1]


def _int_attr(node: Tag, attr: str) -> int | None:
    raw = str(node.get(attr) or "").strip()
    return int(raw) if raw.isdigit() else None


def _is_screening(label: str, kind: FieldKind) -> bool:
    """A question the candidate has to answer, as opposed to who they are.

    Deliberately not a list of topics: anything that is not the usual identity or
    attachment field counts as a question, so a form that asks something we never
    anticipated still shows up.
    """
    if kind in {FieldKind.FILE, FieldKind.EMAIL, FieldKind.PHONE, FieldKind.URL}:
        return False
    folded = label.casefold()
    return not any(hint in folded for hint in _IDENTITY_HINTS)


def default_form_knowledge_path(root: Path) -> Path:
    """Local only: gitignored and blocked by the commit guard."""
    return root / "data" / "form_knowledge.yaml"


def load_form_knowledge(path: Path) -> list[FormKnowledge]:
    if not path.is_file():
        return []
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("forms", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [FormKnowledge.model_validate(item) for item in items]


def save_form_knowledge(forms: list[FormKnowledge], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "note": (
            "What application forms ask, observed locally. Questions only, never "
            "answers. Not shared: `companies export` does not read this file."
        ),
        "forms": [form.model_dump(mode="json") for form in forms],
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def upsert_form(forms: list[FormKnowledge], observed: FormKnowledge) -> list[FormKnowledge]:
    """Replace the entry for the same form, keyed by URL without tracking noise."""
    try:
        key = canonical_key(observed.url)
    except PrivateRouteRejected:
        key = observed.url.casefold()
    out: list[FormKnowledge] = []
    replaced = False
    for existing in forms:
        try:
            existing_key = canonical_key(existing.url)
        except PrivateRouteRejected:
            existing_key = existing.url.casefold()
        if existing_key == key:
            if not replaced:
                out.append(_merge(existing, observed))
                replaced = True
            continue
        out.append(existing)
    if not replaced:
        out.append(observed)
    return out


def _merge(existing: FormKnowledge, observed: FormKnowledge) -> FormKnowledge:
    """A new reading wins, except that a promoted form stays promoted.

    An unreadable reading never erases fields we did read: a page that failed to
    render tells us about the run, not about the form.
    """
    if not observed.readable and existing.readable:
        return existing
    merged = observed.model_copy()
    if existing.status is KnowledgeStatus.ACTIVE:
        merged.status = KnowledgeStatus.ACTIVE
    if merged.company is None:
        merged.company = existing.company
    return merged
