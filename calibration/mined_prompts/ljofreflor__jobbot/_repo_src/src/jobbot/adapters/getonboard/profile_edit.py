"""Write the permanent Get on Board profile through its own edit form.

The long blocks are Trix rich-text editors, so the value lives in a hidden input
driven by `editor.loadHTML` — a plain fill() would leave the form unchanged. Reads
come from the portal itself, so the plan can show what each field holds today.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html import escape
from typing import Any, Protocol

logger = logging.getLogger("jobbot.getonboard.profile_edit")

GOB_EDIT_URL = "https://www.getonbrd.com/webpros/edit"
SAVE_BUTTON = 'input[type="submit"][value*="Guardar"]'


@dataclass(frozen=True)
class GobField:
    """One editable field of the permanent profile."""

    key: str
    label: str
    selector: str
    trix: bool


FIELDS: tuple[GobField, ...] = (
    GobField(
        key="description_es",
        label="Descripción profesional",
        selector='input[name="webpro[description_es]"]',
        trix=False,
    ),
    GobField(
        key="professional_es",
        label="Perfil profesional y experiencia laboral",
        selector="trix-editor#trix-professional_es",
        trix=True,
    ),
    GobField(
        key="academic_background_es",
        label="Formación académica y estudios",
        selector="trix-editor#trix-academic_background_es",
        trix=True,
    ),
)


@dataclass(frozen=True)
class FieldWrite:
    """A pending change on one field, with the text the portal holds today."""

    field: GobField
    current: str
    new: str

    @property
    def verb(self) -> str:
        return "fill" if not self.current else "replace"

    def describe(self) -> str:
        if self.verb == "fill":
            return f"fill {self.field.label} ({len(self.new)} chars, empty today)"
        return (
            f"replace {self.field.label} "
            f"({len(self.current)} → {len(self.new)} chars)"
        )


class ElementLike(Protocol):
    def inner_text(self) -> str: ...
    def input_value(self) -> str: ...
    def evaluate(self, expression: str, arg: Any = None) -> Any: ...
    def fill(self, value: str) -> None: ...


class PageLike(Protocol):
    def goto(self, url: str, **kwargs: Any) -> Any: ...
    def query_selector(self, selector: str) -> ElementLike | None: ...
    def click(self, selector: str, **kwargs: Any) -> None: ...
    def wait_for_timeout(self, timeout: float) -> None: ...


def read_current(page: PageLike) -> dict[str, str]:
    """What the portal shows right now, per field key ('' when absent)."""
    out: dict[str, str] = {}
    for field in FIELDS:
        element = page.query_selector(field.selector)
        if element is None:
            logger.debug("Field %s not present on the edit form", field.key)
            continue
        text = element.inner_text() if field.trix else element.input_value()
        out[field.key] = (text or "").strip()
    return out


def plan_writes(current: dict[str, str], desired: dict[str, str]) -> list[FieldWrite]:
    """Only real changes, and never an empty text over something the portal holds."""
    writes: list[FieldWrite] = []
    for field in FIELDS:
        if field.key not in current:
            continue
        new = (desired.get(field.key) or "").strip()
        old = current[field.key].strip()
        if not new or _same_text(new, old):
            continue
        writes.append(FieldWrite(field=field, current=old, new=new))
    return writes


def apply_writes(page: PageLike, writes: list[FieldWrite]) -> list[FieldWrite]:
    """Set each field, then save the form. Returns the writes that went through."""
    done: list[FieldWrite] = []
    for write in writes:
        element = page.query_selector(write.field.selector)
        if element is None:
            logger.warning("Field %s vanished before writing", write.field.key)
            continue
        if write.field.trix:
            element.evaluate("(node, html) => node.editor.loadHTML(html)", _as_html(write.new))
        else:
            element.fill(write.new)
        done.append(write)
    if done:
        page.click(SAVE_BUTTON)
        page.wait_for_timeout(2500)
    return done


def desired_from_fields(headline: str, professional: str, academic: str) -> dict[str, str]:
    """Map the permanent profile blocks onto the portal's Spanish fields."""
    return {
        "description_es": headline.strip(),
        "professional_es": professional.strip(),
        "academic_background_es": academic.strip(),
    }


def _as_html(text: str) -> str:
    """Trix stores HTML: keep paragraph breaks, escape everything else."""
    paragraphs = [escape(p.strip()) for p in text.split("\n\n") if p.strip()]
    return "".join(f"<div>{p}</div>" for p in paragraphs)


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.split()) == " ".join(right.split())
