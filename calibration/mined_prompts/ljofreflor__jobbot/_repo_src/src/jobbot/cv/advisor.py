"""Small, non-destructive suggestions for how the CV presents existing facts.

The advisor never changes what the candidate did; it changes how it reads. Three
axes: whether a machine can parse it, whether the words match the market's, and
whether the page puts the important part where it is seen.

Two invariants make it safe to run often, and both are checked here rather than
trusted: a suggestion may not assert anything the profile does not already back,
and it may not lose a fact, a figure or a name that the current text has. Anything
that fails either check is dropped before the candidate ever sees it, which is also
how the LLM tiers stay honest.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.nlp.refine import _named_entities
from jobbot.portals.form_learn import FormKnowledge


class Axis(StrEnum):
    """What kind of improvement this is."""

    MACHINE = "machine"
    LANGUAGE = "language"
    LAYOUT = "layout"


class TargetKind(StrEnum):
    """Which piece of the profile a suggestion is about."""

    SUMMARY = "summary"
    ACHIEVEMENT = "achievement"
    DESCRIPTION = "description"
    PROFILE = "profile"


@dataclass(frozen=True)
class Target:
    """Where the suggestion applies, by id — never by position."""

    kind: TargetKind
    experience_id: str = ""
    achievement_id: str = ""

    def describe(self) -> str:
        if self.kind is TargetKind.ACHIEVEMENT:
            return f"{self.experience_id} · {self.achievement_id}"
        if self.kind is TargetKind.DESCRIPTION:
            return f"{self.experience_id} · description"
        return self.kind.value


@dataclass(frozen=True)
class Advice:
    """One suggestion: what to change, where, and the text it proposes."""

    axis: Axis
    target: Target
    what: str
    before: str = ""
    after: str = ""
    why: str = ""

    @property
    def is_rewrite(self) -> bool:
        """A rewrite proposes replacement text; a note only describes the problem."""
        return bool(self.after) and self.after != self.before

    @property
    def id(self) -> str:
        """Stable across runs, so the log can remember what you already answered."""
        seed = f"{self.axis.value}|{self.target.describe()}|{self.what}|{self.before[:120]}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


# Limits that describe a page, not a profession.
BULLET_LIMIT = 6
BULLET_CHARS = 240
SUMMARY_CHARS = 700

# Openers that spend words before saying what was done, in both languages we see.
_WEAK_OPENERS: tuple[tuple[str, str], ...] = (
    ("fui responsable de ", ""),
    ("responsable de ", ""),
    ("encargado de ", ""),
    ("encargada de ", ""),
    ("participé en ", ""),
    ("ayudé a ", ""),
    ("me tocó ", ""),
    ("responsible for ", ""),
    ("worked on ", ""),
    ("helped to ", ""),
    ("helped ", ""),
)

# A PDF export often loses the space glyph, gluing a word to the next one. The `\b`
# is what separates that accident from a product's own spelling: in `experienciaClínica`
# the lowercase run starts a word, while in `BigQuery` it sits inside one, right after
# a capital. Without it the advisor proposed "Google Big Query", which is wrong.
_GLUED_RE = re.compile(r"\b([a-záéíóúñ]+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})")
_ACRONYM_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,6})\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?")

# Acronyms that are units or common shorthand, not something to expand.
_KNOWN_ACRONYMS: frozenset[str] = frozenset(
    {"CV", "PDF", "SQL", "API", "URL", "USD", "CLP", "EUR", "KPI", "OK", "TI", "IT", "HR", "RRHH"}
)

# Intensifiers and filler: lengthening with only these is fluff, not clarity (#54).
_FLUFF_TOKENS: frozenset[str] = frozenset(
    {
        "realmente",
        "altamente",
        "verdaderamente",
        "absolutamente",
        "completamente",
        "totalmente",
        "excepcional",
        "excepcionalmente",
        "desafiante",
        "comprehensive",
        "passionate",
        "highly",
        "really",
        "truly",
        "extremely",
        "very",
        "manera",
        "forma",
        "effective",
        "efectiva",
        "efectivo",
        "entorno",
    }
)

# Redundant openers → denser wording (same facts, fewer words).
_COMPRESSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcon el fin de\b", re.IGNORECASE), "para"),
    (re.compile(r"\bcon el objetivo de\b", re.IGNORECASE), "para"),
    (re.compile(r"\ba efectos de\b", re.IGNORECASE), "para"),
    (re.compile(r"\ben orden de\b", re.IGNORECASE), "por"),
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bfor the purpose of\b", re.IGNORECASE), "to"),
)


def char_delta(before: str, after: str) -> int:
    """How many characters the rewrite adds (negative = denser)."""
    return len(after) - len(before)


def compress_phrase(text: str) -> str | None:
    """Replace stock circumlocutions with shorter equivalents. None if unchanged."""
    out = text
    for pattern, replacement in _COMPRESSIONS:
        out = pattern.sub(replacement, out)
    if out == text:
        return None
    return out


def advise(
    candidate: Candidate,
    *,
    jobs: Sequence[JobPosting] = (),
    forms: Sequence[FormKnowledge] = (),
    playbook: Sequence[str] = (),
    limit: int = 3,
    log_path: Path | None = None,
) -> list[Advice]:
    """The few most useful suggestions for this profile right now.

    Deterministic: the same profile yields the same suggestions in the same order,
    so a run is reproducible and the log can suppress what you already decided.
    Density first: among rewrites that pass the gate, shorter ones surface before
    longer ones (#54).
    """
    decided = load_advice_log(log_path) if log_path is not None else {}
    market_terms = _market_terms_from_jobs(candidate, jobs)
    found: list[Advice] = []
    found.extend(_machine_advice(candidate))
    found.extend(_language_advice(candidate, jobs, playbook))
    found.extend(_layout_advice(candidate))
    found.extend(_form_advice(candidate, forms))

    eligible: list[Advice] = []
    seen: set[str] = set()
    for item in found:
        if item.id in decided or item.id in seen:
            continue
        if item.is_rewrite and validate_advice(
            item, candidate, allowed_market_terms=market_terms
        ) is not None:
            continue
        seen.add(item.id)
        eligible.append(item)

    eligible.sort(key=_density_sort_key)
    return eligible[: max(0, limit)]


def _density_sort_key(item: Advice) -> tuple[int, int, str]:
    """Shorter rewrites first; notes after rewrites; stable by id."""
    if item.is_rewrite:
        return (0, char_delta(item.before, item.after), item.id)
    return (1, 0, item.id)


def _market_terms_from_jobs(
    candidate: Candidate, jobs: Sequence[JobPosting]
) -> frozenset[str]:
    """Terms the postings use for something the profile already backs."""
    if not jobs:
        return frozenset()
    from jobbot.profile.market import suggest_from_market

    suggestion = suggest_from_market(candidate, list(jobs))
    return frozenset(term.casefold() for term in suggestion.present)


def validate_advice(
    advice: Advice,
    candidate: Candidate,
    *,
    allowed_market_terms: Iterable[str] = (),
) -> str | None:
    """Why this rewrite must not be shown, or None when it is safe.

    Checks: no invented names, no dropped figures/names, no fluff lengthening.
    A longer rewrite is allowed only when the new tokens are market terms the
    profile and postings already back — never intensifiers alone (#54).
    """
    if not advice.is_rewrite:
        return None
    after, before = advice.after, advice.before
    if not after.strip():
        return "it would leave the text empty"

    # Whitespace is not content: splitting a glued word asserts nothing new, and
    # merging two words loses nothing. Both checks read through the spaces.
    backed = _profile_vocabulary(candidate) | {token.casefold() for token in _tokens(before)}
    before_squashed, after_squashed = _squashed(before), _squashed(after)
    for entity in _named_entities(after):
        folded = entity.casefold()
        if folded not in backed and folded not in before_squashed:
            return f"it introduces {entity!r}, which the profile does not back"

    for number in _NUMBER_RE.findall(before):
        if number not in after:
            return f"it drops the figure {number!r}"

    for entity in _named_entities(before):
        if entity.casefold() not in after_squashed:
            return f"it drops {entity!r}"

    if before_squashed == after_squashed:
        return None

    before_tokens = {token.casefold() for token in _tokens(before)}
    after_tokens = {token.casefold() for token in _tokens(after)}
    new_tokens = after_tokens - before_tokens
    market = {term.casefold() for term in allowed_market_terms}

    if new_tokens and new_tokens <= _FLUFF_TOKENS:
        return "it only adds fluff intensifiers"

    if len(after) > len(before):
        if not new_tokens:
            return "it is longer than what it replaces"
        if not new_tokens <= (market | backed):
            return "it is longer without backed market terms"
        if new_tokens <= _FLUFF_TOKENS:
            return "it only adds fluff intensifiers"
    return None

def apply_advice(raw: dict[str, Any], advice: Advice) -> bool:
    """Replace the text this advice targets, in a raw profile mapping.

    Returns False when the current text is not what the advice was written
    against: a proposal from an earlier state must never overwrite a newer one.
    """
    if not advice.is_rewrite:
        return False
    target = advice.target
    if target.kind is TargetKind.SUMMARY:
        if str(raw.get("summary") or "").strip() != advice.before.strip():
            return False
        raw["summary"] = advice.after
        return True

    experience = raw.get("experience")
    if not isinstance(experience, list):
        return False
    for exp in experience:
        if not isinstance(exp, dict) or exp.get("id") != target.experience_id:
            continue
        if target.kind is TargetKind.DESCRIPTION:
            if str(exp.get("description") or "").strip() != advice.before.strip():
                return False
            exp["description"] = advice.after
            return True
        for achievement in exp.get("achievements") or []:
            if not isinstance(achievement, dict):
                continue
            if achievement.get("id") != target.achievement_id:
                continue
            if str(achievement.get("text") or "").strip() != advice.before.strip():
                return False
            achievement["text"] = advice.after
            return True
    return False


def default_advice_log_path(output_dir: Path) -> Path:
    return output_dir / "cv" / "advice_log.yaml"


def load_advice_log(path: Path | None) -> dict[str, str]:
    """advice id → 'applied' | 'rejected'."""
    if path is None or not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    decisions = raw.get("decisions", {}) if isinstance(raw, dict) else {}
    if not isinstance(decisions, dict):
        return {}
    return {str(key): str(value) for key, value in decisions.items()}


def record_decision(path: Path, advice: Advice, decision: str) -> Path:
    """Remember what you said, so the next run proposes something else."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict):
        raw = {}
    decisions = raw.get("decisions")
    if not isinstance(decisions, dict):
        decisions = {}
    decisions[advice.id] = decision
    notes = raw.get("notes")
    if not isinstance(notes, dict):
        notes = {}
    notes[advice.id] = f"{advice.axis.value}: {advice.what} ({advice.target.describe()})"
    path.write_text(
        yaml.safe_dump(
            {"version": 1, "decisions": decisions, "notes": notes},
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def render_advice_markdown(advice: Sequence[Advice]) -> str:
    lines = [
        "# CV advice",
        "",
        "Presentation only: nothing here adds a fact you do not have.",
        "",
    ]
    for item in advice:
        lines.append(f"## [{item.axis.value}] {item.what}")
        lines.append(f"- where: {item.target.describe()}")
        lines.append(f"- why: {item.why}")
        if item.before:
            lines.append(f"- now: {item.before}")
        if item.is_rewrite:
            lines.append(f"- proposed: {item.after}")
        lines.append(f"- id: {item.id}")
        lines.append("")
    return "\n".join(lines)


# ── the three axes ───────────────────────────────────────────────────────────


@dataclass
class _Text:
    """One editable piece of the profile, with the target that points back at it."""

    value: str
    target: Target
    label: str
    achievement_ids: list[str] = field(default_factory=list)


def _texts(candidate: Candidate) -> list[_Text]:
    out: list[_Text] = []
    if candidate.summary:
        out.append(_Text(candidate.summary, Target(kind=TargetKind.SUMMARY), "summary"))
    for exp in candidate.experience:
        if exp.description:
            out.append(
                _Text(
                    exp.description,
                    Target(kind=TargetKind.DESCRIPTION, experience_id=exp.id),
                    f"{exp.title} @ {exp.company}",
                )
            )
        for achievement in exp.achievements:
            out.append(
                _Text(
                    achievement.text,
                    Target(
                        kind=TargetKind.ACHIEVEMENT,
                        experience_id=exp.id,
                        achievement_id=achievement.id,
                    ),
                    f"{exp.title} @ {exp.company}",
                )
            )
    return out


def _machine_advice(candidate: Candidate) -> list[Advice]:
    """What a parser trips on: glued words, unexpanded acronyms, missing dates."""
    out: list[Advice] = []
    for text in _texts(candidate):
        for match in _GLUED_RE.finditer(text.value):
            glued = match.group(0)
            out.append(
                Advice(
                    axis=Axis.MACHINE,
                    target=text.target,
                    what=f"separate {glued!r} into two words",
                    before=text.value,
                    after=text.value.replace(glued, f"{match.group(1)} {match.group(2)}", 1),
                    why=(
                        "two words with no space between them are read as one unknown "
                        "token by any parser, and by a person skimming"
                    ),
                )
            )
    out.extend(_acronym_advice(candidate))
    for exp in candidate.experience:
        if exp.end_date is None and not exp.current:
            out.append(
                Advice(
                    axis=Axis.MACHINE,
                    target=Target(kind=TargetKind.DESCRIPTION, experience_id=exp.id),
                    what=f"give {exp.title} @ {exp.company} an end date, or mark it current",
                    why=(
                        "a role with no end and not marked current reads as an open range, "
                        "and filters that sort by recency skip it"
                    ),
                )
            )
    return out


def _acronym_advice(candidate: Candidate) -> list[Advice]:
    """An acronym used but never written out: typography, not a field's jargon."""
    blob = " ".join(text.value for text in _texts(candidate))
    out: list[Advice] = []
    reported: set[str] = set()
    for text in _texts(candidate):
        for acronym in _ACRONYM_RE.findall(text.value):
            if acronym in _KNOWN_ACRONYMS or acronym in reported:
                continue
            if _looks_expanded(acronym, blob):
                continue
            reported.add(acronym)
            out.append(
                Advice(
                    axis=Axis.MACHINE,
                    target=text.target,
                    what=f"write {acronym} out once, then keep the acronym",
                    before=text.value,
                    why=(
                        f"a search for the full name never matches {acronym} alone, "
                        "and a reader outside your last employer may not know it"
                    ),
                )
            )
    return out


def _looks_expanded(acronym: str, blob: str) -> bool:
    """True when the text also spells it out, e.g. 'CAE (Crédito con Aval del Estado)'."""
    if f"{acronym} (" in blob or f"({acronym})" in blob:
        return True
    initials = [word[0].casefold() for word in blob.split() if word[:1].isalpha()]
    letters = [letter.casefold() for letter in acronym]
    joined = "".join(initials)
    return "".join(letters) in joined and len(letters) > 2


def _language_advice(
    candidate: Candidate,
    jobs: Sequence[JobPosting],
    playbook: Sequence[str],
) -> list[Advice]:
    """Say the same thing in fewer words, or in the market's words."""
    out: list[Advice] = []
    for text in _texts(candidate):
        rewritten = _drop_weak_opener(text.value)
        if rewritten is not None:
            out.append(
                Advice(
                    axis=Axis.LANGUAGE,
                    target=text.target,
                    what="start with the action, not with the role you held",
                    before=text.value,
                    after=rewritten,
                    why=(
                        "the reader gives each line a second: the verb says what you did, "
                        "'responsible for' only says you were there"
                    ),
                )
            )
        compressed = compress_phrase(text.value)
        if compressed is not None:
            out.append(
                Advice(
                    axis=Axis.LANGUAGE,
                    target=text.target,
                    what="same facts, fewer words",
                    before=text.value,
                    after=compressed,
                    why=(
                        "stock phrases spend characters before the fact; "
                        "a denser line keeps every number and name"
                    ),
                )
            )
    out.extend(_market_wording_advice(candidate, jobs))
    for note in playbook[:2]:
        out.append(
            Advice(
                axis=Axis.LANGUAGE,
                target=Target(kind=TargetKind.PROFILE),
                what=note,
                why="from the recruiter knowledge you promoted (recruiters list)",
            )
        )
    return out


def _market_wording_advice(
    candidate: Candidate,
    jobs: Sequence[JobPosting],
) -> list[Advice]:
    """A term the postings use for something you already do, worded their way."""
    if not jobs:
        return []
    from jobbot.profile.market import suggest_from_market

    suggestion = suggest_from_market(candidate, list(jobs))
    out: list[Advice] = []
    for term in suggestion.present[:3]:
        where = _first_text_missing(candidate, term)
        if where is None:
            continue
        out.append(
            Advice(
                axis=Axis.LANGUAGE,
                target=where.target,
                what=f"name {term!r} where you already do it",
                before=where.value,
                why=(
                    f"the stored postings say {term!r} for this, and your profile backs it; "
                    "a filter matches the word, not the idea"
                ),
            )
        )
    return out


def _first_text_missing(candidate: Candidate, term: str) -> _Text | None:
    """The first experience text that does not yet use this term."""
    folded = term.casefold()
    for text in _texts(candidate):
        if text.target.kind is TargetKind.SUMMARY:
            continue
        if folded not in text.value.casefold():
            return text
    return None


def _layout_advice(candidate: Candidate) -> list[Advice]:
    """How much is on the page, and what the first screen shows."""
    out: list[Advice] = []
    summary = (candidate.summary or "").strip()
    if len(summary) > SUMMARY_CHARS:
        out.append(
            Advice(
                axis=Axis.LAYOUT,
                target=Target(kind=TargetKind.SUMMARY),
                what=f"shorten the summary to about {SUMMARY_CHARS} characters",
                before=summary,
                why=(
                    f"it is {len(summary)} characters, so the first role starts below the "
                    "first screen and the reader scrolls to find where you worked"
                ),
            )
        )
    for exp in candidate.experience:
        if not exp.achievements:
            empty = not exp.description
            out.append(
                Advice(
                    axis=Axis.LAYOUT,
                    target=Target(kind=TargetKind.DESCRIPTION, experience_id=exp.id),
                    what=f"add at least one bullet to {exp.title} @ {exp.company}",
                    why=(
                        "a role with a title and dates but no content reads as filler, "
                        "and it is often the role being asked about"
                        if empty
                        else "the description says what the role covered; a bullet says "
                        "what came out of it, and that is the line that gets read"
                    ),
                )
            )
        if len(exp.achievements) > BULLET_LIMIT:
            out.append(
                Advice(
                    axis=Axis.LAYOUT,
                    target=Target(kind=TargetKind.DESCRIPTION, experience_id=exp.id),
                    what=(
                        f"keep the {BULLET_LIMIT} strongest bullets in {exp.title} "
                        f"@ {exp.company} and move the rest to older roles' level of detail"
                    ),
                    why=(
                        f"{len(exp.achievements)} bullets in one role flatten each other; "
                        "nothing is deleted from the profile by deciding what to show"
                    ),
                )
            )
        for achievement in exp.achievements:
            if len(achievement.text) > BULLET_CHARS:
                out.append(
                    Advice(
                        axis=Axis.LAYOUT,
                        target=Target(
                            kind=TargetKind.ACHIEVEMENT,
                            experience_id=exp.id,
                            achievement_id=achievement.id,
                        ),
                        what="split this bullet in two",
                        before=achievement.text,
                        why=(
                            f"{len(achievement.text)} characters is a paragraph pretending "
                            "to be a bullet; two lines each land, one line is skipped"
                        ),
                    )
                )
    return out


def _form_advice(candidate: Candidate, forms: Sequence[FormKnowledge]) -> list[Advice]:
    """What employers ask on their own forms, and the CV does not answer yet."""
    out: list[Advice] = []
    blob = " ".join(text.value for text in _texts(candidate)).casefold()
    asked: dict[str, int] = {}
    for form in forms:
        for question in form.screening_questions():
            asked[question] = asked.get(question, 0) + 1
    for question, times in sorted(asked.items(), key=lambda pair: (-pair[1], pair[0]))[:2]:
        if _question_answered(question, blob):
            continue
        out.append(
            Advice(
                axis=Axis.LANGUAGE,
                target=Target(kind=TargetKind.PROFILE),
                what=f"decide how your CV answers: {question}",
                why=(
                    f"the application form asks it ({times} form(s) observed) and nothing in "
                    "the profile answers it — only you can, JobBot will not guess"
                ),
            )
        )
    return out


def _question_answered(question: str, blob: str) -> bool:
    """Whether the profile already contains the words the question is about."""
    words = [word for word in _tokens(question) if len(word) > 4]
    if not words:
        return True
    hits = sum(1 for word in words if word.casefold() in blob)
    return hits >= max(2, len(words) // 2)


def _drop_weak_opener(text: str) -> str | None:
    """Same sentence, starting at the verb. None when there is nothing to drop."""
    stripped = text.lstrip()
    folded = stripped.casefold()
    for opener, _ in _WEAK_OPENERS:
        if folded.startswith(opener):
            rest = stripped[len(opener) :]
            if not rest:
                return None
            return rest[:1].upper() + rest[1:]
    return None


def _profile_vocabulary(candidate: Candidate) -> set[str]:
    """Every word the profile already contains: the ceiling for any rewrite."""
    parts: list[str] = [
        candidate.personal.name or "",
        candidate.personal.headline or "",
        candidate.summary or "",
        *candidate.specialties,
        *candidate.skills.all_skills(),
    ]
    for exp in candidate.experience:
        parts.extend([exp.company, exp.title, exp.location or "", exp.description or ""])
        parts.extend(achievement.text for achievement in exp.achievements)
    for edu in candidate.education:
        parts.extend([edu.institution, edu.degree])
    return {token.casefold() for token in _tokens(" ".join(parts))}


def _squashed(text: str) -> str:
    """The text with whitespace gone, for comparisons where spacing is the change."""
    return re.sub(r"\s+", "", text.casefold())


def _tokens(text: str) -> Iterable[str]:
    return re.findall(r"[\wÁÉÍÓÚÑáéíóúñ+#.-]{2,}", text or "")
