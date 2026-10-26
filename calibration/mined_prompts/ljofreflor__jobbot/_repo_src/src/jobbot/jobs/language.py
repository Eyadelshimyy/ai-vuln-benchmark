"""Spanish or English? Deterministic, from function words only — never a guess on a tie."""

from __future__ import annotations

from typing import Literal

from jobbot.jobs.normalization import fold_text
from jobbot.models.job import JobPosting

Language = Literal["es", "en"]

# Function words only (folded, no accents). Words both languages spell alike
# ('a', 'no', 'me') count for neither.
_ES = frozenset(
    {
        "de", "la", "que", "el", "en", "y", "los", "del", "las", "por", "un", "una",
        "para", "con", "se", "su", "al", "lo", "como", "mas", "es", "sus", "nuestro",
        "nuestra", "nuestros", "somos", "buscamos", "tu", "seras", "estamos", "o",
    }
)  # fmt: skip
_EN = frozenset(
    {
        "the", "and", "of", "to", "in", "you", "will", "with", "for", "is", "are",
        "we", "our", "your", "be", "as", "on", "at", "by", "this", "that", "have",
        "from", "an", "or", "who", "looking", "join",
    }
)  # fmt: skip

_MIN_HITS = 4
_MIN_SHARE = 0.7


def detect_language(text: str) -> Language | None:
    """'es' or 'en' when one side holds a clear majority of function words, else None."""
    words = fold_text(text).split()
    es = sum(1 for word in words if word in _ES)
    en = sum(1 for word in words if word in _EN)
    total = es + en
    if total < _MIN_HITS:
        return None
    if es / total >= _MIN_SHARE:
        return "es"
    if en / total >= _MIN_SHARE:
        return "en"
    return None


def posting_language(job: JobPosting) -> Language | None:
    """The language the posting itself is written in, from its stored text."""
    text = "\n".join(
        part
        for part in (job.title, job.description or job.raw_description, *job.requirements)
        if part
    )
    return detect_language(text)
