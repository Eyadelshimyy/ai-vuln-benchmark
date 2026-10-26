"""Detect apply-by-LinkedIn-message (DM / InMail). Never send."""

from __future__ import annotations

import re

# Imperative "message me" / "mandame un mensaje" — not past-tense chatter.
_MESSAGE_APPLY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"mensaje interno", re.IGNORECASE),
    re.compile(
        r"\b(?:m[aá]ndame|env[ií]ame|escribime|escr[ií]beme)\s+"
        r"(?:un\s+)?(?:mensaje|dm|inmail)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:enviar|env[ií]a|manda(?:r)?)\s+por\s+mensaje"
        r"(?:\s+(?:interno|privado))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:send|drop)\s+me\s+(?:a\s+)?(?:dm|message|inmail|pm)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:message|dm|inmail)\s+me\b", re.IGNORECASE),
    re.compile(r"\bvia\s+(?:a\s+)?(?:linkedin\s+)?(?:dm|inmail|message)\b", re.IGNORECASE),
)


def asks_for_linkedin_message(text: str) -> bool:
    """Whether the posting's only told apply route is a LinkedIn message.

    Matches how to apply (imperative / «mensaje interno»), never a field's
    vocabulary and never past-tense chatter («te mandé un mensaje»).
    """
    body = text or ""
    return any(pattern.search(body) for pattern in _MESSAGE_APPLY_RES)
