"""Portal field labels → profile facts (#94).

A homologation table lists, for each fact in ``profile.yaml``, the portal
labels that mean that fact (ES/EN). Lookup is exact after folding — never by
substring — so \"Middle Name\" is not confused with \"name\", and new aliases
are added to the table instead of inventing matches at fill time.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from jobbot.models.candidate import Candidate


class ProfileFact(StrEnum):
    """Canonical slot in the candidate profile a portal field may ask for."""

    FULL_NAME = "full_name"
    GIVEN_NAME = "given_name"
    FAMILY_NAME = "family_name"
    EMAIL = "email"
    PHONE = "phone"
    CITY = "city"
    COUNTRY = "country"
    LOCATION = "location"
    HEADLINE = "headline"
    LINKEDIN = "linkedin"
    GITHUB = "github"
    RESUME = "resume"


# One fact → many accepted portal labels. Unknown labels stay unmapped (HITL).
# Do not add "middle name", "father's family name", or "mother's family name"
# unless profile.yaml gains those facts.
_HOMOLOGATION: dict[ProfileFact, tuple[str, ...]] = {
    ProfileFact.FULL_NAME: (
        "full name",
        "name",
        "legal name",
        "nombre completo",
    ),
    ProfileFact.GIVEN_NAME: (
        "first name",
        "given name",
        "given name(s)",
        "nombre",
        "nombres",
    ),
    ProfileFact.FAMILY_NAME: (
        "last name",
        "family name",
        "surname",
        "apellido",
        "apellidos",
    ),
    ProfileFact.EMAIL: (
        "email",
        "email address",
        "e-mail",
        "correo",
        "correo electrónico",
        "correo electronico",
    ),
    ProfileFact.PHONE: (
        "phone",
        "phone number",
        "mobile",
        "móvil",
        "movil",
        "celular",
        "teléfono",
        "telefono",
    ),
    ProfileFact.CITY: ("city", "ciudad"),
    ProfileFact.COUNTRY: ("country", "país", "pais"),
    ProfileFact.LOCATION: ("location", "ubicación", "ubicacion"),
    ProfileFact.HEADLINE: (
        "headline",
        "current role",
        "current title",
        "cargo actual",
    ),
    ProfileFact.LINKEDIN: (
        "linkedin",
        "linkedin profile",
        "linkedin url",
        "perfil linkedin",
    ),
    ProfileFact.GITHUB: ("github", "github profile", "github url"),
    ProfileFact.RESUME: (
        "resume",
        "resume/cv",
        "cv",
        "curriculum",
        "currículum",
        "currículum vitae",
        "curriculum vitae",
    ),
}

_ALIAS_TO_FACT: dict[str, ProfileFact] = {}


def _fold(label: str) -> str:
    """Stable key for a portal label: casefold, strip *, collapse space."""
    text = label.casefold().replace("*", " ")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_ish = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_ish).strip()


def _ensure_index() -> None:
    if _ALIAS_TO_FACT:
        return
    for fact, aliases in _HOMOLOGATION.items():
        for alias in aliases:
            key = _fold(alias)
            if key in _ALIAS_TO_FACT and _ALIAS_TO_FACT[key] is not fact:
                msg = f"homologation alias {alias!r} maps to two facts"
                raise RuntimeError(msg)
            _ALIAS_TO_FACT[key] = fact


def resolve_fact(label: str) -> ProfileFact | None:
    """Which profile fact this portal label asks for, or None if unknown."""
    _ensure_index()
    return _ALIAS_TO_FACT.get(_fold(label))


def aliases_for(fact: ProfileFact) -> tuple[str, ...]:
    """Accepted portal wordings for one fact (for docs/tests)."""
    return _HOMOLOGATION[fact]


def value_for_fact(candidate: Candidate, fact: ProfileFact) -> tuple[str, str]:
    """(value, source) from profile.yaml for this fact. Empty when unknown."""
    personal = candidate.personal
    if fact is ProfileFact.FULL_NAME:
        return _pair(personal.name, "personal.name")
    if fact is ProfileFact.GIVEN_NAME:
        given, _ = _split_name(personal.name)
        return _pair(given, "personal.name")
    if fact is ProfileFact.FAMILY_NAME:
        _, family = _split_name(personal.name)
        return _pair(family, "personal.name")
    if fact is ProfileFact.EMAIL:
        return _pair(personal.email, "personal.email")
    if fact is ProfileFact.PHONE:
        return _pair(personal.phone, "personal.phone")
    if fact is ProfileFact.CITY:
        return _pair(personal.city, "personal.city")
    if fact is ProfileFact.COUNTRY:
        return _pair(personal.country, "personal.country")
    if fact is ProfileFact.LOCATION:
        return _pair(personal.location_line(), "personal.city + country")
    if fact is ProfileFact.HEADLINE:
        return _pair(personal.headline, "personal.headline")
    if fact is ProfileFact.LINKEDIN:
        return _pair(personal.linkedin, "personal.linkedin")
    if fact is ProfileFact.GITHUB:
        return _pair(personal.github, "personal.github")
    if fact is ProfileFact.RESUME:
        return ("output/base/cv.pdf", "cv build")
    return ("", "you decide: not a fact in the profile")


def answer_for_label(candidate: Candidate, label: str) -> tuple[str, str]:
    """Resolve a portal label via the homologation table only."""
    fact = resolve_fact(label)
    if fact is None:
        return ("", "you decide: not a fact in the profile")
    return value_for_fact(candidate, fact)


def _pair(value: object, source: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return ("", "you decide: not a fact in the profile")
    return (text, source)


def _split_name(full: str | None) -> tuple[str, str]:
    """Given / family split for Spanish two-of-each when four tokens."""
    parts = (full or "").split()
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "")
    if len(parts) >= 4:
        return (" ".join(parts[:2]), " ".join(parts[2:]))
    return (parts[0], " ".join(parts[1:]))
