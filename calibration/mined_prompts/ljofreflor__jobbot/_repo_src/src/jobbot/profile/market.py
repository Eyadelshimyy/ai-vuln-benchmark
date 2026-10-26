"""Market-language suggestions for baseline CV (no invention; no deletions)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jobbot.jobs.normalization import normalize_skill
from jobbot.jobs.parsing import extract_skills_from_text
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting

# Market language is whatever the stored JDs repeat, so the terms are read from the
# corpus: a fixed list would only ever return the field it was written for.
_MAX_TERMS = 40
_NEUTRAL_SKILL_GROUP = "other"
_WORD_RE = re.compile(r"[a-záéíóúñü][a-záéíóúñü0-9+#.-]{2,}")

_CORPUS_STOPWORDS = frozenset(
    {
        "para",
        "con",
        "como",
        "los",
        "las",
        "del",
        "que",
        "por",
        "una",
        "uno",
        "the",
        "and",
        "for",
        "with",
        "you",
        "our",
        "will",
        "have",
        "need",
        "needs",
        "must",
        "your",
        "are",
        "this",
        "that",
        "from",
        "experiencia",
        "experience",
        "conocimiento",
        "conocimientos",
        "requisitos",
        "requirements",
        "required",
        "deseable",
        "excluyente",
        "manejo",
        "años",
        "anos",
        "years",
        "equipo",
        "equipos",
        "trabajo",
        "empresa",
        "company",
        "cargo",
        "puesto",
        "role",
        "senior",
        "junior",
        "semi",
        "title",
        "location",
        "seniority",
    }
)


@dataclass
class MarketGapQuestion:
    term: str
    reason: str
    evidence_jobs: list[str] = field(default_factory=list)


@dataclass
class MarketSuggestion:
    """Proposed baseline wording changes + questions (never auto-applied)."""

    market_terms: list[tuple[str, int]]  # term, count
    present: list[str]
    missing_suspected: list[MarketGapQuestion]
    rephrase_hints: list[str]
    summary_suggestion: str | None = None


def suggest_from_market(
    candidate: Candidate,
    jobs: list[JobPosting],
    *,
    min_count: int = 2,
) -> MarketSuggestion:
    """Extract frequent JD terms; suggest rephrases; flag gaps as questions only."""
    counts = _count_market_terms(jobs)
    frequent = _rank_terms(counts, min_count=min_count)

    cand_tokens = _candidate_token_set(candidate)
    present: list[str] = []
    missing: list[MarketGapQuestion] = []
    for term, count in frequent:
        if _term_present_in_profile(term, cand_tokens):
            present.append(term)
        else:
            job_ids = [
                j.id
                for j in jobs
                if term.casefold() in ((j.description or "") + (j.raw_description or "")).casefold()
            ][:5]
            missing.append(
                MarketGapQuestion(
                    term=term,
                    reason=(
                        f"Appears in {count} job description(s). "
                        "Confirm only if you have used it; otherwise leave out."
                    ),
                    evidence_jobs=job_ids,
                )
            )

    rephrase_hints = _rephrase_hints(candidate, present)
    summary_suggestion = None
    if present:
        top = ", ".join(present[:8])
        base = (candidate.summary or "").strip()
        if base and top:
            summary_suggestion = (
                f"{base}\n\n# Suggested market-aligned emphasis (review before promote):\n"
                f"# Highlight where already true: {top}."
            )
        elif top:
            summary_suggestion = (
                f"# Suggested emphasis (only keep facts you already have):\n# {top}."
            )

    return MarketSuggestion(
        market_terms=frequent,
        present=present,
        missing_suspected=missing,
        rephrase_hints=rephrase_hints,
        summary_suggestion=summary_suggestion,
    )


def render_suggestion_markdown(suggestion: MarketSuggestion) -> str:
    lines = [
        "# Market feedback (from stored job descriptions)",
        "",
        "Rules: do not invent experience; do not delete baseline facts;",
        "confirm gaps interactively before adding anything to profile.yaml.",
        "",
        "## Frequent market terms",
    ]
    if not suggestion.market_terms:
        lines.append("(none above threshold)")
    else:
        for term, count in suggestion.market_terms:
            lines.append(f"- {term} ({count})")
    lines += ["", "## Already reflected in profile"]
    lines += [f"- {t}" for t in suggestion.present] or ["- (none)"]
    lines += ["", "## Ask before adding (suspected gaps)"]
    if not suggestion.missing_suspected:
        lines.append("- (none)")
    else:
        for gap in suggestion.missing_suspected:
            jobs = ", ".join(gap.evidence_jobs) or "—"
            lines.append(f"- **{gap.term}**: {gap.reason} (jobs: {jobs})")
    lines += ["", "## Rephrase hints (presentation only)"]
    lines += [f"- {h}" for h in suggestion.rephrase_hints] or ["- (none)"]
    if suggestion.summary_suggestion:
        lines += [
            "",
            "## Summary draft (commented / review)",
            "```",
            suggestion.summary_suggestion,
            "```",
        ]
    lines.append("")
    return "\n".join(lines)


def apply_confirmed_skills(
    profile: Candidate | Mapping[str, Any],
    confirmed: list[str],
    *,
    group: str = _NEUTRAL_SKILL_GROUP,
) -> Candidate | dict[str, Any]:
    """Add user-confirmed skills only; never remove existing ones.

    Writers persist the raw mapping so unknown YAML keys survive. Passing a
    ``Candidate`` still returns a ``Candidate`` (schema fields only).
    """
    if isinstance(profile, Candidate):
        merged = merge_confirmed_skills_into_raw(profile.model_dump(), confirmed, group=group)
        return Candidate.model_validate(merged)
    return merge_confirmed_skills_into_raw(dict(profile), confirmed, group=group)


def merge_confirmed_skills_into_raw(
    raw: dict[str, Any],
    confirmed: list[str],
    *,
    group: str = _NEUTRAL_SKILL_GROUP,
) -> dict[str, Any]:
    """Merge confirmed skills into a raw profile mapping (no deletions)."""
    import copy

    out = copy.deepcopy(raw)
    skills = out.setdefault("skills", {})
    if not isinstance(skills, dict):
        skills = {}
        out["skills"] = skills
    bucket = list(skills.get(group) or [])
    existing = {str(s).casefold() for s in bucket}
    for term in confirmed:
        if term.casefold() not in existing:
            bucket.append(term)
            existing.add(term.casefold())
    skills[group] = bucket
    return out


def _count_market_terms(jobs: list[JobPosting]) -> Counter[str]:
    """How many stored jobs name each term, in the words the jobs themselves use."""
    phrases = {term for job in jobs for term in _named_terms(job) if len(term.split()) > 1}
    counts: Counter[str] = Counter()
    for job in jobs:
        blob = _job_blob(job)
        terms = {term for term in _named_terms(job) if _is_term_like(term)}
        terms |= {phrase for phrase in phrases if _mentions(blob, phrase)}
        terms |= {word for word in _WORD_RE.findall(blob) if _is_term_like(word)}
        for term in terms:
            counts[term] += 1
    return counts


def _rank_terms(counts: Counter[str], *, min_count: int) -> list[tuple[str, int]]:
    """Frequent first, longest phrase first; a word inside a kept phrase is noise."""
    frequent = [(term, count) for term, count in counts.items() if count >= min_count]
    frequent.sort(key=lambda item: (-item[1], -len(item[0].split()), item[0]))

    kept: list[tuple[str, int]] = []
    for term, count in frequent:
        if any(count <= other_count and _inside(term, other) for other, other_count in kept):
            continue
        kept.append((term, count))
        if len(kept) >= _MAX_TERMS:
            break
    return kept


def _inside(term: str, phrase: str) -> bool:
    return term != phrase and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", phrase) is not None


def _named_terms(job: JobPosting) -> set[str]:
    """The skills the JD names, read from its own structure (no fixed vocabulary)."""
    terms = {term.casefold() for term in job.skills if term.strip()}
    for text in (job.description or "", job.raw_description or ""):
        terms |= {term.casefold() for term in extract_skills_from_text(text)}
    return {term for term in terms if term}


def _job_blob(job: JobPosting) -> str:
    parts = (job.description or "", job.raw_description or "", *job.requirements, *job.skills)
    return "\n".join(parts).casefold()


def _mentions(blob: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", blob) is not None


def _is_term_like(term: str) -> bool:
    words = term.split()
    if any(word in _CORPUS_STOPWORDS for word in words):
        return False
    return all(len(word) >= 2 for word in words) and any(len(word) >= 3 for word in words)


def _candidate_token_set(candidate: Candidate) -> set[str]:
    tokens: set[str] = set()
    for group in candidate.skills.as_dict():
        # The group the candidate filed a skill under is a claim too: 'machine_learning'
        # for someone who lists XGBoost, 'clinical' for someone who lists ventilation.
        tokens.add(group.casefold())
        tokens.add(group.replace("_", " ").casefold())
    for skill in candidate.skills.all_skills():
        tokens.add(normalize_skill(skill))
        tokens.add(skill.casefold())
    for specialty in candidate.specialties:
        tokens.add(normalize_skill(specialty))
        tokens.add(specialty.casefold())
    if candidate.summary:
        tokens.add(candidate.summary.casefold())
    for exp in candidate.experience:
        if exp.description:
            tokens.add(exp.description.casefold())
        for ach in exp.achievements:
            tokens.add(ach.text.casefold())
    return tokens


def _term_present_in_profile(term: str, cand_tokens: set[str]) -> bool:
    """Present when the profile says it, in its own words — never by implication."""
    canon = normalize_skill(term)
    if canon in cand_tokens or term.casefold() in cand_tokens:
        return True
    folded = term.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", token)
        for token in cand_tokens
        if len(token) > len(folded)
    )


def _rephrase_hints(candidate: Candidate, present_market_terms: list[str]) -> list[str]:
    hints: list[str] = []
    if not present_market_terms:
        return hints
    # Suggest weaving known market terms into existing bullets when already present
    for exp in candidate.experience[:3]:
        for ach in exp.achievements[:2]:
            text = ach.text.casefold()
            hits = [t for t in present_market_terms if t.casefold() in text]
            if hits:
                hints.append(
                    f"Keep fact at {exp.company}: emphasize wording around {', '.join(hits[:3])} "
                    f"(bullet already contains the concept)."
                )
    if candidate.summary:
        hints.append(
            "Keep summary facts; optionally lead with market-aligned terms you already cover: "
            + ", ".join(present_market_terms[:5])
        )
    return hints[:10]
