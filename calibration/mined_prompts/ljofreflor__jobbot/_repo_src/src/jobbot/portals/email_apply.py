"""Extract apply-to emails from free text (LinkedIn posts, JDs). Never invent addresses."""

from __future__ import annotations

import re

from email_validator import EmailNotValidError, validate_email

_EMAIL_RE = re.compile(
    r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b"
)

_APPLY_HINTS = (
    "enviar cv",
    "envia cv",
    "envía cv",
    "envie cv",
    "envíe cv",
    "enviar curriculum",
    "enviar currículum",
    "send cv",
    "send your cv",
    "send resume",
    "send your resume",
    "email your cv",
    "email your resume",
    "postula",
    "postular",
    "aplicar a",
    "apply to",
    "apply at",
    "cv a ",
    "cv:",
    "curriculum a",
    "currículum a",
    "mandar cv",
    "manda tu cv",
)

# Same invitations with a pronoun in between ("send me your CV", "envíame tu currículum"),
# which substring hints cannot cover.
_APPLY_HINT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:send|email|share|forward)\s+(?:me|us|it)?\s*(?:your|me|us)?\s*"
        r"(?:\w+\s+){0,2}?(?:cv|resume|r[ée]sum[ée])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:env[ií]a|env[ií]ar|env[ií][ae]me|env[ií]enme|manda|mandar|mand[áa]me|"
        r"comparte|comp[áa]rteme)\s*(?:me|nos)?\s*(?:tu|su|el|un)?\s*"
        r"(?:\w+\s+){0,2}?(?:cv|curr[ií]culum|curriculum|hoja de vida)\b",
        re.IGNORECASE,
    ),
)


def _has_apply_hint(text: str) -> bool:
    lowered = text.casefold()
    if any(hint in lowered for hint in _APPLY_HINTS):
        return True
    return any(pattern.search(text) for pattern in _APPLY_HINT_RES)

# Local-part tokens that look like hiring inboxes
_MAILBOX_LOCAL_TOKENS = (
    "seleccion",
    "selección",
    "rrhh",
    "jobs",
    "careers",
    "talent",
    "recruit",
    "recluta",
    "postula",
    "empleo",
    "trabajo",
    "people",
    "hiring",
    "hr",
    "rh",
)


def is_valid_email(addr: str) -> bool:
    """
    RFC-aware check, no DNS lookup (JobBot must work offline).

    The extraction regex is deliberately loose so it can find addresses in prose;
    this is what stops a loose match from becoming an apply route.
    """
    try:
        validate_email(addr, check_deliverability=False)
    except EmailNotValidError:
        return False
    return True


def extract_emails(text: str) -> list[str]:
    """Return unique, valid emails in appearance order."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _EMAIL_RE.finditer(text or ""):
        addr = match.group(1).rstrip(".,;:)")
        key = addr.casefold()
        if key in seen or not is_valid_email(addr):
            continue
        seen.add(key)
        out.append(addr)
    return out


def _is_hiring_mailbox(addr: str) -> bool:
    local = addr.casefold().split("@", 1)[0]
    return any(tok in local for tok in _MAILBOX_LOCAL_TOKENS)


def first_apply_email(text: str) -> str | None:
    """
    Best apply mailbox from text, or None.

    Prefers hiring-style mailbox prefixes, then emails near apply hints.
    Does not invent addresses — only returns emails present in ``text``.
    """
    body = text or ""
    emails = extract_emails(body)
    if not emails:
        return None
    for addr in emails:
        if _is_hiring_mailbox(addr):
            return addr
    for match in _EMAIL_RE.finditer(body):
        addr = match.group(1).rstrip(".,;:)")
        if not is_valid_email(addr):
            continue
        start = max(0, match.start() - 80)
        end = min(len(body), match.end() + 40)
        if _has_apply_hint(body[start:end]):
            return addr
    if _has_apply_hint(body) and len(emails) == 1:
        return emails[0]
    return None


def mailto_url(email: str) -> str:
    addr = email.strip()
    if addr.lower().startswith("mailto:"):
        return addr
    return f"mailto:{addr}"
