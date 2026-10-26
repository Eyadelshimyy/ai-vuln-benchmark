"""Cumulative text refinement for portal profile blurbs.

Previous drafts are computational capital: each iteration should improve them
using profile.yaml as the fact base — not discard and regenerate from scratch.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from typing import Protocol

from jobbot.adapters.getonboard.draft import (
    EDUCATION_MAX,
    EXPERIENCE_MAX,
    PermanentProfileFields,
    build_permanent_profile_fields,
    recent_roles,
    role_head,
    role_label,
)
from jobbot.branding import stamp_description, strip_mark
from jobbot.models.candidate import Candidate
from jobbot.models.experience import Experience

logger = logging.getLogger("jobbot.nlp.refine")


class ProfileTextRefiner(Protocol):
    def refine(
        self,
        candidate: Candidate,
        previous: PermanentProfileFields | None,
    ) -> PermanentProfileFields: ...


@dataclass(frozen=True)
class RefineResult:
    fields: PermanentProfileFields
    mode: str  # cold | cumulative | llm
    kept_paragraphs: int
    added_paragraphs: int
    dropped_paragraphs: int


def refine_permanent_profile(
    candidate: Candidate,
    previous: PermanentProfileFields | None,
    *,
    use_llm: bool = False,
) -> RefineResult:
    """Prefer cumulative refine; optional LangChain when use_llm=True.

    NL work lives here (not in chat): heuristics by default, LLM only if requested.
    """
    from jobbot.nlp.gateway import run_optional_llm

    def _deterministic() -> RefineResult:
        if previous is None:
            cold = build_permanent_profile_fields(candidate)
            return RefineResult(
                fields=cold,
                mode="cold",
                kept_paragraphs=0,
                added_paragraphs=0,
                dropped_paragraphs=0,
            )
        return CumulativeProfileRefiner().refine_with_stats(candidate, previous)

    def _llm() -> RefineResult:
        from jobbot.nlp.langchain_refine import LangChainProfileRefiner

        fields = LangChainProfileRefiner().refine(candidate, previous)
        return RefineResult(
            fields=fields,
            mode="llm",
            kept_paragraphs=0,
            added_paragraphs=0,
            dropped_paragraphs=0,
        )

    outcome = run_optional_llm(
        deterministic=_deterministic,
        llm=_llm,
        use_llm=use_llm,
        task_name="permanent_profile_refine",
    )
    if outcome.mode == "llm_fallback":
        logger.warning("LLM refine fell back: %s", outcome.detail)
    return outcome.value


class CumulativeProfileRefiner:
    """Improve previous GoB texts using Candidate facts — keep what still holds."""

    def refine(
        self,
        candidate: Candidate,
        previous: PermanentProfileFields | None,
    ) -> PermanentProfileFields:
        return self.refine_with_stats(candidate, previous).fields

    def refine_with_stats(
        self,
        candidate: Candidate,
        previous: PermanentProfileFields | None,
    ) -> RefineResult:
        if previous is None:
            cold = build_permanent_profile_fields(candidate)
            return RefineResult(
                fields=cold,
                mode="cold",
                kept_paragraphs=0,
                added_paragraphs=0,
                dropped_paragraphs=0,
            )

        cold = build_permanent_profile_fields(candidate)
        companies = _company_tokens(candidate)
        exp_text, kept, added, dropped = _merge_roles(
            strip_mark(previous.experiencia_y_perfil),
            strip_mark(cold.experiencia_y_perfil),
            candidate,
            companies,
            EXPERIENCE_MAX,
        )
        edu_text, _, _, _ = _merge_experience(
            previous.formacion_academica,
            cold.formacion_academica,
            _education_tokens(candidate),
            EDUCATION_MAX,
        )
        exp_text = stamp_description(exp_text, max_len=EXPERIENCE_MAX)
        fields = PermanentProfileFields(
            experiencia_y_perfil=exp_text,
            formacion_academica=edu_text,
            headline=candidate.personal.headline or previous.headline,
            skills=list(candidate.skills.all_skills())[:10] or previous.skills,
            signature=previous.signature,
        )
        return RefineResult(
            fields=fields,
            mode="cumulative",
            kept_paragraphs=kept,
            added_paragraphs=added,
            dropped_paragraphs=dropped,
        )


def _company_tokens(candidate: Candidate) -> set[str]:
    """What the profile backs in an experience paragraph: employers, roles and tools.

    A paragraph that only lists the candidate's own stack names no employer, so
    without the skills it read as an unbacked claim and was deleted — and the cold
    draft added it again on the next run, eroding the text back and forth.
    """
    tokens: set[str] = set()
    for exp in candidate.experience:
        for part in re.split(r"[/|,]", exp.company):
            t = part.strip().casefold()
            if len(t) >= 3:
                tokens.add(t)
        tokens.add(exp.title.casefold())
    for skill in candidate.skills.all_skills():
        tokens.add(skill.casefold())
    if candidate.summary:
        tokens.add(candidate.summary.casefold())
    return tokens


def _education_tokens(candidate: Candidate) -> set[str]:
    tokens: set[str] = set()
    for edu in candidate.education:
        tokens.add(edu.institution.casefold())
        tokens.add(edu.degree.casefold())
    for pub in candidate.publications:
        if pub.journal:
            tokens.add(pub.journal.casefold())
        tokens.add(pub.title.casefold())
    for exp in candidate.experience:
        title_l = exp.title.casefold()
        if any(k in title_l for k in ("profesor", "docente", "ayudante", "investigador")):
            tokens.add(exp.company.casefold())
            tokens.add(title_l)
    return tokens


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _paragraph_supported(paragraph: str, anchors: set[str]) -> bool:
    """Keep a paragraph unless it claims an employer or tool the profile dropped.

    The old rule kept a paragraph only when it used a fixed set of words, which
    meant a candidate outside that field lost their own text. What actually makes
    a stored paragraph stale is naming an entity the profile no longer backs, and
    that is checkable: the profile is the fact base.
    """
    lower = paragraph.casefold()
    if any(a in lower for a in anchors if len(a) >= 4):
        return True
    return not any(
        entity.casefold() not in " ".join(anchors)
        for entity in _named_entities(paragraph)
    )


def _named_entities(paragraph: str) -> list[str]:
    """Proper names as typography marks them: 'Adexus', 'MATLAB', 'Qt-Python'.

    Sentence-initial words are skipped: capitalisation there says nothing.
    """
    entities: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
        words = sentence.split()
        for word in words[1:]:
            token = word.strip(".,;:()[]\"'")
            if len(token) < 3:
                continue
            if token.isupper() or (token[:1].isupper() and not token.isdigit()):
                entities.append(token)
    return entities


def _merge_experience(
    previous: str,
    cold: str,
    anchors: set[str],
    maximum: int,
) -> tuple[str, int, int, int]:
    prev_parts = _paragraphs(previous)
    cold_parts = _paragraphs(cold)
    kept: list[str] = []
    dropped = 0
    for part in prev_parts:
        if _paragraph_supported(part, anchors):
            kept.append(_completed_by(part, cold_parts))
        else:
            dropped += 1

    # Ensure cold facts for recent companies appear if missing
    added = 0
    cold_missing: list[str] = []
    kept_blob = "\n".join(kept).casefold()
    for part in cold_parts:
        # If cold paragraph introduces a company/title not covered, append
        tokens = re.findall(r"[a-záéíóúñ]{4,}", part.casefold())
        if not tokens:
            continue
        # Prefer appending role paragraphs that mention companies in anchors
        if (
            any(a in part.casefold() for a in anchors)
            and part.casefold() not in kept_blob
            and not any(_similar(part, k) for k in kept)
        ):
            cold_missing.append(part)
            added += 1

    merged = kept + cold_missing
    # Prefer order: summary-like first (from kept or cold), then roles
    text = "\n\n".join(merged).strip()
    if len(text) > maximum:
        # Drop from the end of cold additions first, then oldest kept
        while merged and len("\n\n".join(merged)) > maximum:
            merged.pop()
        text = "\n\n".join(merged).strip()
    if not text:
        text = cold[:maximum]
        added = len(cold_parts)
        kept = []
        dropped = len(prev_parts)
    return text, len(kept), added, dropped


_HEAD_TAIL = re.compile(r"\s*(?:\([^)]*\))?\s*\.")


def _role_of(paragraph: str, candidate: Candidate) -> tuple[Experience, int] | None:
    """The role a paragraph opens with, and where its head sentence ends."""
    found: list[tuple[Experience, re.Match[str]]] = []
    for exp in candidate.experience:
        label = role_label(exp)
        if paragraph[: len(label)].casefold() != label.casefold():
            continue
        tail = _HEAD_TAIL.match(paragraph, len(label))
        if tail is not None:
            found.append((exp, tail))
    if not found:
        return None
    exp, tail = next(
        ((e, t) for e, t in found if e.start_date and e.start_date in t.group(0)),
        found[0],
    )
    return exp, tail.end()


def _with_profile_head(paragraph: str, candidate: Candidate) -> str:
    """A stored role keeps its body; its period is whatever profile.yaml says today."""
    role = _role_of(paragraph, candidate)
    if role is None:
        return paragraph
    exp, end = role
    body = paragraph[end:].strip()
    head = role_head(exp)
    return f"{head} {body}" if body else head


def _merge_roles(
    previous: str,
    cold: str,
    candidate: Candidate,
    anchors: set[str],
    maximum: int,
) -> tuple[str, int, int, int]:
    """Cumulative merge where profile.yaml owns which roles are current and when they ended.

    A stored paragraph keeps its wording, but a role that has ended can no longer be
    current, and the most recent roles outrank older ones when the cap forces a cut.
    """
    prev_parts = [_with_profile_head(p, candidate) for p in _paragraphs(previous)]
    text, kept, added, dropped = _merge_experience(
        "\n\n".join(prev_parts), cold, anchors, sys.maxsize
    )
    rank = {exp.id: i for i, exp in enumerate(recent_roles(candidate, len(candidate.experience)))}
    recent = [exp.id for exp in recent_roles(candidate)]

    pairs: list[tuple[str | None, str]] = []
    for index, part in enumerate(_paragraphs(text)):
        role = _role_of(part, candidate)
        role_id = role[0].id if role else None
        if role_id is not None and any(rid == role_id for rid, _ in pairs):
            if index < kept:
                kept, dropped = kept - 1, dropped + 1
            else:
                added -= 1
            continue
        pairs.append((role_id, part))

    present = {rid for rid, _ in pairs}
    slot = next((i for i, (rid, _) in enumerate(pairs) if rid is not None), len(pairs))
    for part in _paragraphs(cold):
        role = _role_of(part, candidate)
        if role is not None and role[0].id in recent and role[0].id not in present:
            pairs.insert(slot, (role[0].id, part))
            present.add(role[0].id)
            added += 1

    role_pairs = [p for p in pairs if p[0] is not None]
    by_recency = iter(sorted(role_pairs, key=lambda p: rank[p[0] or ""]))
    pairs = [next(by_recency) if rid is not None else (rid, part) for rid, part in pairs]

    def victim() -> int:
        older = [i for i, (rid, _) in enumerate(pairs) if rid is not None and rid not in recent]
        others = [i for i, (rid, _) in enumerate(pairs) if rid is None]
        if older:
            return older[-1]
        if others:
            return others[-1]
        return len(pairs) - 1

    while pairs and len("\n\n".join(part for _, part in pairs)) > maximum:
        pairs.pop(victim())
    return "\n\n".join(part for _, part in pairs).strip(), kept, added, dropped


def _completed_by(paragraph: str, cold_parts: list[str]) -> str:
    """Restore a stored paragraph that is only a prefix of the current text.

    Old length caps cut paragraphs mid-sentence; keeping the mutilated version is
    losing information, which is the opposite of a cumulative refine.
    """
    stem = paragraph.rstrip(" ,;:.").casefold()
    if not stem:
        return paragraph
    for cold in cold_parts:
        if len(cold) > len(paragraph) and cold.casefold().startswith(stem):
            return cold
    return paragraph


def _similar(a: str, b: str, *, threshold: float = 0.55) -> bool:
    ta = set(re.findall(r"[a-záéíóúñ0-9]{4,}", a.casefold()))
    tb = set(re.findall(r"[a-záéíóúñ0-9]{4,}", b.casefold()))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold
