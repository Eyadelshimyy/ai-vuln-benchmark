"""Check a free-text application answer against the profile and the posting.

A cover letter or a "why do you want to work here" box is pasted into a portal as
written, so every claim in it has to hold before it leaves the machine. Two failures
survive a fact-by-fact read: a sentence that pairs a degree, role or figure with the
wrong institution or employer, and a motivation answer that lists the candidate's
achievements and says nothing about the company. Three layers catch them:

1. Candidate claims: figures, proper names, organisations and the skills the posting
   asks for must be in ``profile.yaml``, and a degree, role or figure named next to an
   institution or employer must belong to that same entry.
2. Company claims: what a sentence says about the employer must be in the posting.
3. Question type: motivation is the candidate's own. It is never approved here; the
   check only shows what the posting and the profile offer to write it from.

Deterministic and offline: words are compared tolerant of accents, gender and plural,
never by a model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

from jobbot.jobs.normalization import WordIndex, fold_text
from jobbot.models.candidate import Candidate
from jobbot.models.experience import format_metric
from jobbot.models.job import JobPosting
from jobbot.nlp.refine import _named_entities


class QuestionKind(StrEnum):
    """Whether the profile can answer the question, or only the candidate can."""

    FACTUAL = "factual"
    MOTIVATION = "motivation"


class Verdict(StrEnum):
    OK = "ok"
    REJECTED = "rejected"
    NEEDS_CANDIDATE = "needs_candidate"


@dataclass(frozen=True)
class Evidence:
    """What backs a sentence: a profile entry or a phrase of the posting."""

    source: str  # profile | posting
    where: str
    quote: str


@dataclass(frozen=True)
class SentenceTrace:
    text: str
    about_company: bool = False
    evidence: tuple[Evidence, ...] = ()
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerCheck:
    question: str
    kind: QuestionKind
    verdict: Verdict
    sentences: tuple[SentenceTrace, ...]
    warnings: tuple[str, ...] = ()
    posting_hooks: tuple[str, ...] = ()
    profile_hooks: tuple[str, ...] = ()


def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Function words of both languages; content words are what a claim is made of.
_STOP: frozenset[str] = _wordset(
    """a al an and are as at be but by con como cada de del desde donde e el en entre es
    esta estas este estos for from fue ha han has have he i in is la las le les lo los
    mas me mi mis muy my no nos nuestra nuestras nuestro nuestros o of on or our para
    pero por que se ser sin sobre son soy su sus tambien that the these this those to
    u un una unas uno unos was we were with y ya also very more"""
)

# Words any answer about any job uses; they say nothing specific about a company.
_GENERIC: frozenset[str] = _wordset(
    """equipo team empresa company compania organizacion organization trabajo trabajar
    work working cargo puesto role position rol oportunidad opportunity parte part
    unirme unirte join joining aportar contribute contribuir sumarme crecer grow growth
    experiencia experience postular apply applying ustedes vuestro vuestra your"""
)

# Opinion and motivation verbs: the candidate's stance, not a claim to check.
_OPINION: frozenset[str] = _wordset(
    """interesa interesaria interes interest interested interesting motiva motivan
    motivacion motivation motivates motivated gusta gustaria like love encanta
    encantaria admiro admire quiero quisiera want would deseo wish excited exciting
    emociona entusiasma apasiona passionate passion creo believe think pienso siento
    feel valoro value atrae attracts attract llama atencion hope espero porque because
    since"""
)

# Kinds of organisation, across sectors: part of a name, never what tells two apart.
_ORG_KINDS: frozenset[str] = _wordset(
    """universidad university instituto institute escuela school college colegio
    facultad faculty fundacion foundation centro center centre hospital clinica clinic
    ministerio ministry servicio service municipalidad banco bank empresa company grupo
    group inc ltd ltda spa corp llc"""
)

# Seniority modifiers a role title carries and an answer often drops.
_SENIORITY: frozenset[str] = _wordset(
    "senior junior semi sr jr lead principal trainee intern practicante"
)

# A sentence that addresses the employer by deixis rather than by name.
_DEIXIS: tuple[str, ...] = (
    " ustedes ",
    " su equipo ",
    " sus equipos ",
    " su empresa ",
    " su compania ",
    " vuestro ",
    " vuestra ",
    " tu equipo ",
    " la empresa ",
    " la compania ",
    " esta empresa ",
    " esta compania ",
    " el cargo ",
    " este cargo ",
    " el puesto ",
    " este puesto ",
    " este rol ",
    " you ",
    " your ",
    " the company ",
    " this company ",
    " the team ",
    " the role ",
    " this role ",
    " this position ",
)

# Questions only the candidate can answer: why here, what moves them. Matched on
# folded text (no accents, no punctuation), by phrasing — never by a trade's words.
_MOTIVATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bpor ?que (?:te |le |les )?(?:interesa\w*|gustaria\w*|quier\w*|quer\w*|dese\w*"
        r"|postul\w*|motiv\w*|atrae\w*|llama\w*)\b",
        r"\bpor ?que (?:deberiamos|te elegiriamos|nosotros|esta empresa|este cargo"
        r"|este puesto|este rol|esta posicion)\b",
        r"\bque (?:te|le|les) (?:motiva|atrae|interesa|entusiasma|llama la atencion)\b",
        r"\bmotivacion\w*\b",
        r"\bcarta de (?:presentacion|motivacion)\b",
        r"\bwhy (?:do|would|are|did|should) (?:you|we)\b",
        r"\bwhy (?:us|this company|this role|this position|this job|this team|join\w*)\b",
        r"\bwhat (?:motivates|attracts|excites|draws|interests) you\b",
        r"\bmotivat\w*\b",
        r"\b(?:cover|motivation) letter\b",
        r"\binterest\w* in (?:working|joining)\b",
    )
)

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
_FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_LABEL_LINE_RE = re.compile(r"^[^\W\d_][\w ]{0,24}:\s*\S.{0,60}$")
_ORG_NAME_RE = re.compile(
    r"\b(?:Universidad|University|Instituto|Institute|Escuela|School|College|Colegio"
    r"|Facultad|Faculty|Fundación|Foundation|Centro|Center|Centre|Hospital|Clínica"
    r"|Clinic|Ministerio|Ministry|Servicio|Municipalidad|Banco|Bank)"
    r"(?:\s+(?:(?:de|del|la|los|las|of|the|for)\s+)*[A-ZÁÉÍÓÚÑ][\w'’-]*)+"
)
_QUOTE_CHARS = 160
_HOOKS = 3


@dataclass(frozen=True)
class _Token:
    text: str
    word: str
    is_figure: bool


@dataclass(frozen=True, eq=False)
class _Fact:
    where: str
    text: str
    index: WordIndex
    figures: frozenset[str]
    experience_id: str = ""


@dataclass(frozen=True)
class _Org:
    name: str
    words: tuple[str, ...]
    distinctive: tuple[str, ...] = ()


@dataclass
class _Context:
    candidate: Candidate
    job: JobPosting
    facts: list[_Fact]
    institutions: list[_Org]
    employers: list[_Org]
    company: _Org
    known_orgs: list[_Org]
    segments: list[str]
    posting_index: WordIndex
    posting_figures: frozenset[str]
    profile_index: WordIndex
    company_by_experience: dict[str, str] = field(default_factory=dict)


def classify_question(question: str, *, company: str = "") -> QuestionKind:
    """Motivation when the question asks why, or what moves the candidate."""
    folded = fold_text(question)
    if any(pattern.search(folded) for pattern in _MOTIVATION_PATTERNS):
        return QuestionKind.MOTIVATION
    asks_why = re.search(r"\b(?:por ?que|why)\b", folded) is not None
    names = {w for w in _content_words(company, minimum=3) if w not in _ORG_KINDS}
    if asks_why and names & set(folded.split()):
        return QuestionKind.MOTIVATION
    return QuestionKind.FACTUAL


def check_answer(
    question: str, answer: str, candidate: Candidate, job: JobPosting
) -> AnswerCheck:
    """Trace each sentence of an answer to the profile and the posting, then rule on it."""
    kind = classify_question(question, company=job.company)
    context = _context(candidate, job)
    sentences = tuple(_check_sentence(text, context) for text in _sentences(answer))
    warnings: list[str] = []
    if not sentences:
        warnings.append("the answer is empty")

    posting_hooks: tuple[str, ...] = ()
    profile_hooks: tuple[str, ...] = ()
    if kind is QuestionKind.MOTIVATION:
        posting_hooks = _posting_hooks(context)
        profile_hooks = _profile_hooks(context)
        if sentences and not _addresses_company(sentences, context):
            warnings.append(
                f"does not address the company: it names nothing the posting says about "
                f"{job.company}, its team or its work"
            )

    if not sentences or any(s.problems for s in sentences):
        verdict = Verdict.REJECTED
    elif kind is QuestionKind.MOTIVATION:
        verdict = Verdict.NEEDS_CANDIDATE
    else:
        verdict = Verdict.OK
    return AnswerCheck(
        question=question,
        kind=kind,
        verdict=verdict,
        sentences=sentences,
        warnings=tuple(warnings),
        posting_hooks=posting_hooks,
        profile_hooks=profile_hooks,
    )


_VERDICT_NOTES: dict[Verdict, str] = {
    Verdict.OK: "every claim is backed; read it once more before pasting",
    Verdict.REJECTED: "fix the flagged sentences before pasting",
    Verdict.NEEDS_CANDIDATE: (
        "motivation is yours: write or confirm it yourself; JobBot never approves it"
    ),
}


def render_answer_check(check: AnswerCheck, job: JobPosting | None = None) -> list[str]:
    """Plain lines: each sentence with its backing or its problem, then the verdict."""
    lines: list[str] = []
    if job is not None:
        lines.append(f"{job.id} · {job.company} — {job.title}")
    lines.append(f"Question ({check.kind.value}): {check.question}")
    for number, sentence in enumerate(check.sentences, start=1):
        mark = "✗" if sentence.problems else "✓"
        lines.append(f"{number}. {mark} {sentence.text}")
        for problem in sentence.problems:
            lines.append(f"     ✗ {problem}")
        for item in sentence.evidence:
            lines.append(f"     {item.source}: {item.where} — {item.quote}")
        if not sentence.problems and not sentence.evidence:
            lines.append("     (no checkable claim: opinion or connective text)")
    for warning in check.warnings:
        lines.append(f"! {warning}")
    if check.posting_hooks:
        lines.append("Posting phrases a draft could rely on:")
        lines.extend(f"  - {hook}" for hook in check.posting_hooks)
    if check.profile_hooks:
        lines.append("Profile facts that meet the posting:")
        lines.extend(f"  - {hook}" for hook in check.profile_hooks)
    lines.append(f"Verdict: {check.verdict.value} — {_VERDICT_NOTES[check.verdict]}")
    return lines


def _check_sentence(text: str, context: _Context) -> SentenceTrace:
    tokens = _tokens(text)
    folded = f" {fold_text(text)} "
    about = bool(_org_positions(tokens, context.company)) or any(d in folded for d in _DEIXIS)
    evidence: list[Evidence] = []
    problems: list[str] = []

    _check_organisations(text, about, context, problems)
    _check_pairings(tokens, context, evidence, problems)
    _check_figures(tokens, about, context, evidence, problems)
    if not about:
        _check_posting_skills(tokens, context, evidence, problems)
    _check_entities(text, about, context, evidence, problems)
    if about:
        _check_company_claim(text, tokens, context, evidence, problems)

    return SentenceTrace(
        text=text,
        about_company=about,
        evidence=tuple(_dedupe(evidence)),
        problems=tuple(dict.fromkeys(problems)),
    )


def _check_organisations(
    text: str, about: bool, context: _Context, problems: list[str]
) -> None:
    """An institution named by its kind ('Universidad de …') must be one the profile has."""
    for match in _ORG_NAME_RE.finditer(text):
        name = match.group(0)
        words = [w for w in _content_words(name, minimum=2) if w not in _ORG_KINDS]
        if not words:
            continue
        if any(all(WordIndex(frozenset(org.words)).has(w) for w in words)
               for org in context.known_orgs):
            continue
        if about and all(context.posting_index.has(w) for w in words):
            continue
        problems.append(f"names {name}, which is not an institution or employer in the profile")


def _check_entities(
    text: str,
    about: bool,
    context: _Context,
    evidence: list[Evidence],
    problems: list[str],
) -> None:
    """Proper names must be the profile's; a sentence about the employer may cite the posting.

    A name already inside a traced degree, role or figure adds no evidence of its own.
    """
    company_words = context.company.words
    seen = [item.quote for item in evidence] + problems
    covered = WordIndex(frozenset(w for t in seen for w in _content_words(t, minimum=2)))
    for entity in _named_entities(text):
        for word in fold_text(entity).split():
            if word in _STOP or word.isdigit():
                continue
            fact = _fact_with_word(word, context.facts)
            if fact is not None:
                if not covered.has(word):
                    evidence.append(Evidence("profile", fact.where, _short(fact.text)))
                    covered = WordIndex(covered.words | fact.index.words)
                continue
            if any(_same(word, own) for own in company_words):
                continue
            if about and context.posting_index.has(word):
                continue
            problems.append(f"names {entity!r}, which the profile does not back")
            break


def _check_figures(
    tokens: list[_Token],
    about: bool,
    context: _Context,
    evidence: list[Evidence],
    problems: list[str],
) -> None:
    for position, token in enumerate(tokens):
        if not token.is_figure:
            continue
        key = _figure_key(token.text)
        backing = [fact for fact in context.facts if key in fact.figures]
        if backing:
            evidence.append(Evidence("profile", backing[0].where, _short(backing[0].text)))
            owners = {fact.experience_id for fact in backing}
            if "" not in owners:
                valid = {context.company_by_experience[o] for o in owners}
                neighbour = _misplaced(position, valid, _mentions(tokens, context.employers))
                if neighbour is not None:
                    holders = _join(
                        f"{exp.title} at {exp.company}"
                        for exp in context.candidate.experience
                        if exp.id in owners
                    )
                    problems.append(
                        f"misattributed: the figure {token.text} belongs to {holders} in the "
                        f"profile, not {neighbour}"
                    )
            continue
        if about and key in context.posting_figures:
            continue
        problems.append(f"the figure {token.text} is not in the profile")


def _check_posting_skills(
    tokens: list[_Token],
    context: _Context,
    evidence: list[Evidence],
    problems: list[str],
) -> None:
    """A skill the posting asks for is the claim most likely to be echoed, not held."""
    posting = context.job.description or context.job.raw_description
    for skill in context.job.skills:
        if re.search(rf"(?mi)^\s*{re.escape(skill)}\s*:", posting):
            continue
        words = tuple(_content_words(skill, minimum=2))
        if not words or not _find(tokens, words):
            continue
        fact = next(
            (f for f in context.facts if all(f.index.has(w) for w in words)),
            None,
        )
        if fact is None:
            problems.append(
                f"claims {skill!r}, which the posting asks for but the profile does not back"
            )
        else:
            evidence.append(Evidence("profile", fact.where, _short(fact.text)))


def _check_pairings(
    tokens: list[_Token],
    context: _Context,
    evidence: list[Evidence],
    problems: list[str],
) -> None:
    """A degree or role must sit next to the institution or employer of its own entry."""
    candidate = context.candidate
    institutions = _mentions(tokens, context.institutions)
    for degree in dict.fromkeys(edu.degree for edu in candidate.education):
        entries = [edu for edu in candidate.education if edu.degree == degree]
        for position in _find(tokens, tuple(_content_words(degree, minimum=2))):
            valid = {edu.institution for edu in entries}
            neighbour = _misplaced(position, valid, institutions)
            if neighbour is not None:
                problems.append(
                    f"misattributed: {degree!r} is from {_join(sorted(valid))} in the "
                    f"profile, not {neighbour}"
                )
                continue
            for edu in entries:
                evidence.append(
                    Evidence("profile", f"education {edu.id}", f"{edu.degree} · {edu.institution}")
                )

    employers = _mentions(tokens, context.employers)
    for title in dict.fromkeys(exp.title for exp in candidate.experience):
        held = [exp for exp in candidate.experience if exp.title == title]
        words = tuple(w for w in _content_words(title, minimum=2) if w not in _SENIORITY)
        for position in _find(tokens, words):
            valid = {exp.company for exp in held}
            neighbour = _misplaced(position, valid, employers)
            if neighbour is not None:
                problems.append(
                    f"misattributed: {title!r} was held at {_join(sorted(valid))} in the "
                    f"profile, not {neighbour}"
                )
                continue
            for exp in held:
                evidence.append(
                    Evidence("profile", f"experience {exp.id}", f"{exp.title} · {exp.company}")
                )


def _check_company_claim(
    text: str,
    tokens: list[_Token],
    context: _Context,
    evidence: list[Evidence],
    problems: list[str],
) -> None:
    content = [
        token.word
        for token in tokens
        if not token.is_figure and _claim_word(token.word, context.company)
    ]
    if not content:
        return
    best, hits = "", 0
    for segment in context.segments:
        index = WordIndex(frozenset(_content_words(segment)))
        count = sum(1 for word in content if index.has(word))
        if count > hits:
            best, hits = segment, count
    if hits >= min(2, len(content)) and hits * 2 >= len(content):
        evidence.append(Evidence("posting", "posting", _short(best)))
        return
    problems.append(
        f"says something about {context.job.company} that the posting does not: "
        f"{_short(text, 90)!r}"
    )


def _addresses_company(sentences: Sequence[SentenceTrace], context: _Context) -> bool:
    """Named something the posting says and the candidate's own profile does not."""
    if any(e.source == "posting" for s in sentences for e in s.evidence):
        return True
    distinctive: set[str] = set()
    for sentence in sentences:
        for word in _content_words(sentence.text):
            if (
                _claim_word(word, context.company)
                and context.posting_index.has(word)
                and not context.profile_index.has(word)
            ):
                distinctive.add(word)
    return len(distinctive) >= 2


def _posting_hooks(context: _Context) -> tuple[str, ...]:
    scored: list[tuple[int, int, str]] = []
    for order, segment in enumerate(context.segments):
        if len(segment) < 30 or _LABEL_LINE_RE.match(segment):
            continue
        score = sum(
            1
            for word in set(_content_words(segment))
            if _claim_word(word, context.company) and not context.profile_index.has(word)
        )
        if score >= 2:
            scored.append((-score, order, segment))
    return tuple(_short(segment) for _, _, segment in sorted(scored)[:_HOOKS])


def _profile_hooks(context: _Context) -> tuple[str, ...]:
    hooks: list[str] = []
    for fact in context.facts:
        if not fact.where.startswith("experience ") or fact.where.count("·") != 1:
            continue
        shared = {
            word
            for word in _content_words(fact.text)
            if _claim_word(word, context.company) and context.posting_index.has(word)
        }
        if len(shared) >= 2:
            hooks.append(f"{fact.where}: {_short(fact.text)}")
    for skill in context.job.skills:
        words = _content_words(skill, minimum=2)
        if words and context.profile_index.has(words[0]) and all(
            context.profile_index.has(w) for w in words
        ):
            hooks.append(f"skill: {skill}")
    return tuple(dict.fromkeys(hooks))[: _HOOKS * 2]


def _context(candidate: Candidate, job: JobPosting) -> _Context:
    facts = _profile_facts(candidate)
    institution_names = list(dict.fromkeys(edu.institution for edu in candidate.education))
    employer_names = list(dict.fromkeys(exp.company for exp in candidate.experience))
    pool = [*institution_names, *employer_names, job.company]
    institutions = [_org(name, pool) for name in institution_names]
    employers = [_org(name, pool) for name in employer_names]
    company = _org(job.company, pool)

    posting_text = "\n".join(
        part
        for part in (job.title, job.description, job.raw_description, *job.requirements)
        if part
    )
    segments = list(dict.fromkeys(_sentences(posting_text)))
    profile_words: set[str] = set()
    for fact in facts:
        profile_words |= fact.index.words
    return _Context(
        candidate=candidate,
        job=job,
        facts=facts,
        institutions=institutions,
        employers=employers,
        company=company,
        known_orgs=[*institutions, *employers, company],
        segments=segments,
        posting_index=WordIndex(frozenset(_content_words(posting_text, minimum=2))),
        posting_figures=_figures(posting_text),
        profile_index=WordIndex(frozenset(profile_words)),
        company_by_experience={exp.id: exp.company for exp in candidate.experience},
    )


def _profile_facts(candidate: Candidate) -> list[_Fact]:
    """Every entry of the profile that can back a claim, most specific first."""
    facts: list[_Fact] = []

    def add(
        where: str, text: str | None, figures: Iterable[str] = (), experience_id: str = ""
    ) -> None:
        if not text or not text.strip():
            return
        words = frozenset(_content_words(text, minimum=2))
        keys = _figures(text) | {_figure_key(f) for f in figures if f}
        facts.append(_Fact(where, text.strip(), WordIndex(words), frozenset(keys), experience_id))

    for exp in candidate.experience:
        years = [exp.start_date[:4], (exp.end_date or "")[:4]]
        head = " · ".join(p for p in (exp.title, exp.company, exp.location) if p)
        add(f"experience {exp.id}", head, years, exp.id)
        add(f"experience {exp.id} · description", exp.description, (), exp.id)
        for achievement in exp.achievements:
            add(
                f"experience {exp.id} · {achievement.id}",
                achievement.text,
                _metric_figures(achievement.metrics),
                exp.id,
            )
    for edu in candidate.education:
        years = [(edu.start_date or "")[:4], (edu.end_date or "")[:4]]
        head = " · ".join(p for p in (edu.degree, edu.institution, edu.details) if p)
        add(f"education {edu.id}", head, years)
    for publication in candidate.publications:
        head = " · ".join(p for p in (publication.title, publication.journal) if p)
        add(f"publication {publication.id}", head, [str(publication.year or "")])
    for project in candidate.projects:
        add(f"project {project.id}", f"{project.name}: {project.description}")
    for group, values in candidate.skills.as_dict().items():
        add(f"skills.{group}", ", ".join(values))
    add("specialties", ", ".join(candidate.specialties))
    add("summary", candidate.summary)
    personal = candidate.personal
    add(
        "personal",
        " · ".join(p for p in (personal.name, personal.headline, personal.city, personal.country)
                   if p),
    )
    return facts


def _metric_figures(metrics: dict[str, int | float | str]) -> list[str]:
    """A metric backs the number as stored and as the CV displays it (18, 6M → 6)."""
    out: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        out.append(str(value))
        out.extend(_FIGURE_RE.findall(format_metric(key, value)))
    return out


def _org(name: str, pool: Sequence[str]) -> _Org:
    words = tuple(_content_words(name, minimum=2))
    others: set[str] = set()
    for other in pool:
        if fold_text(other) != fold_text(name):
            others |= set(_content_words(other, minimum=2))
    distinctive = tuple(
        w for w in words if w not in _ORG_KINDS and len(w) >= 3 and w not in others
    )
    return _Org(name=name, words=words, distinctive=distinctive)


def _org_positions(tokens: list[_Token], org: _Org) -> list[int]:
    """Where an organisation is named: in full, or by a capitalised word only it has."""
    hits = _find(tokens, org.words)
    if hits:
        return hits
    return [
        i
        for i, token in enumerate(tokens)
        if token.text[:1].isupper() and any(_same(token.word, w) for w in org.distinctive)
    ]


def _mentions(tokens: list[_Token], orgs: Sequence[_Org]) -> list[tuple[int, str]]:
    return sorted((pos, org.name) for org in orgs for pos in _org_positions(tokens, org))


def _misplaced(position: int, valid: set[str], partners: list[tuple[int, str]]) -> str | None:
    """The organisation a claim sits next to, when it is not the claim's own.

    Both languages put the institution after what it qualifies ('Magíster de X',
    'Magíster (X)') or open the clause with it ('En X lideré …'), so the nearest
    mention on each side is what the sentence pairs the claim with.
    """
    before = [name for pos, name in partners if pos < position]
    after = [name for pos, name in partners if pos > position]
    neighbours = [n for n in (before[-1] if before else None, after[0] if after else None) if n]
    if not neighbours:
        return None
    folded_valid = {fold_text(v) for v in valid}
    if any(fold_text(n) in folded_valid for n in neighbours):
        return None
    return after[0] if after else before[-1]


def _find(tokens: list[_Token], words: tuple[str, ...]) -> list[int]:
    """Positions where all the words of a name appear together, in order of the first."""
    if not words:
        return []
    hits: list[int] = []
    for i, token in enumerate(tokens):
        if token.is_figure or not _same(token.word, words[0]):
            continue
        window = tokens[i : i + len(words) + 3]
        if all(any(_same(t.word, w) for t in window) for w in words[1:]):
            hits.append(i)
    return hits


def _fact_with_word(word: str, facts: Sequence[_Fact]) -> _Fact | None:
    return next((fact for fact in facts if fact.index.has(word)), None)


@lru_cache(maxsize=4096)
def _same(left: str, right: str) -> bool:
    """One word, tolerant of gender, plural and cognates ('ingeniera' ~ 'ingeniería')."""
    return left == right or WordIndex(frozenset({right})).has(left)


def _claim_word(word: str, company: _Org) -> bool:
    if len(word) < 4 or word.isdigit() or word in _STOP:
        return False
    if any(_same(word, own) for own in company.words):
        return False
    return not (_GENERIC_INDEX.has(word) or _OPINION_INDEX.has(word))


_GENERIC_INDEX = WordIndex(_GENERIC)
_OPINION_INDEX = WordIndex(_OPINION)


def _tokens(text: str) -> list[_Token]:
    out: list[_Token] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        is_figure = raw[0].isdigit()
        out.append(_Token(raw, raw if is_figure else fold_text(raw), is_figure))
    return out


def _content_words(text: str, *, minimum: int = 4) -> list[str]:
    return [
        word
        for word in fold_text(text).split()
        if len(word) >= minimum and word not in _STOP and not word.isdigit()
    ]


def _figures(text: str) -> frozenset[str]:
    return frozenset(_figure_key(raw) for raw in _FIGURE_RE.findall(text))


def _figure_key(raw: str) -> str:
    """'6.000.000', '6,000,000' and '6000000' are one figure; '1,5' is '1.5'."""
    raw = raw.strip(".,")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):
        return re.sub(r"[.,]", "", raw)
    return raw.replace(",", ".")


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_RE.split(text or ""):
        cleaned = _BULLET_RE.sub("", part).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _dedupe(items: Iterable[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    out: list[Evidence] = []
    for item in items:
        if item.where not in seen:
            seen.add(item.where)
            out.append(item)
    return out


def _join(values: Iterable[str]) -> str:
    return " / ".join(dict.fromkeys(values))


def _short(text: str, limit: int = _QUOTE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
