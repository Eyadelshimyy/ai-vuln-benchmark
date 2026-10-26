"""Scoring helpers (kept thin; core logic in analyzer)."""

from collections.abc import Sequence

from jobbot.models.match import JobMatch

_BLIND_MIN_JOBS = 2


def blind_matcher_warning(scores: Sequence[float]) -> str | None:
    """Every job at 0% says more about the matcher than about the candidate.

    A real search always turns up something partly related; when nothing scores at
    all, the matcher could not read this profile against these postings (language,
    vocabulary, page noise), and a ranked list of zeros would read as 'nothing fits'.
    """
    if len(scores) < _BLIND_MIN_JOBS or any(score > 0 for score in scores):
        return None
    return (
        f"Matcher alert: all {len(scores)} jobs scored 0%. That usually means the matcher "
        "could not read this profile against these postings, not that nothing fits. "
        "Open one with `jobbot jobs match Jxxxx` before discarding them."
    )


def format_match_report(match: JobMatch) -> str:
    lines = [f"MATCH: {match.score:.0f}%"]
    groups = [
        ("Strong", "strong_match", "✓"),
        ("Partial", "partial_match", "~"),
        ("Missing", "missing", "✗"),
        ("Unknown", "unknown", "?"),
    ]
    data = match.to_dict()
    for title, key, mark in groups:
        values = data[key]
        assert isinstance(values, list)
        if not values:
            continue
        lines.append(title)
        for label in values:
            lines.append(f"{mark} {label}")
    return "\n".join(lines)
