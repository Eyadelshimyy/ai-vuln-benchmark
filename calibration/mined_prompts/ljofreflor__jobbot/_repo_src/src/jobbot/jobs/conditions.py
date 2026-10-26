"""A stored posting's special conditions: closed, residency, language, contract, pay…

Read from the posting's own wording and stored fields, offline. The cues are
structural — "must reside in", a CEFR level, "excluyente", "prestación de
servicios", a closed banner — never the vocabulary of one profession. A posting
that does not state a condition yields nothing: silence is not a condition.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from functools import lru_cache

from babel import Locale

from jobbot.jobs.closure import closure_evidence
from jobbot.jobs.geo import countries_in_text
from jobbot.models.job import JobPosting


class ConditionKind(StrEnum):
    CLOSED = "closed"
    DEADLINE = "deadline"
    RESIDENCY = "residency"
    WORK_AUTHORIZATION = "work_authorization"
    LANGUAGE = "language"
    REQUIREMENT = "requirement"
    CONTRACT = "contract"
    SALARY = "salary"
    MODALITY = "modality"
    SENIORITY = "seniority"
    AVAILABILITY = "availability"
    INSTRUCTION = "instruction"


@dataclass(frozen=True)
class Condition:
    """One condition, with the posting's own phrase and what was parsed from it."""

    kind: ConditionKind
    source: str
    mandatory: bool | None = None
    countries: tuple[str, ...] = ()
    anywhere: bool = False
    language: str | None = None  # ISO 639-1
    level: str | None = None  # CEFR A1–C2, read as a minimum
    contract: str | None = None  # contractor | indefinite | fixed_term | employee
    modality: str | None = None  # remote | hybrid | onsite
    shown: bool | None = None  # salary published or not
    when: date | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"kind": self.kind.value, "source": self.source}
        for key in ("mandatory", "language", "level", "contract", "modality", "shown", "detail"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.countries:
            data["countries"] = list(self.countries)
        if self.anywhere:
            data["anywhere"] = True
        if self.when is not None:
            data["when"] = self.when.isoformat()
        return data


_CEFR = ("A1", "A2", "B1", "B2", "C1", "C2")

# Ordered: the longer phrase must win ('upper intermediate' before 'intermediate').
_LEVEL_WORDS: tuple[tuple[str, str], ...] = (
    (r"lengua materna|mother tongue|nativ[oa]s?|native|bilingue|bilingual", "C2"),
    (r"fully fluent|full professional", "C1"),
    (r"upper intermediate|intermedio alto|intermedio avanzado|professional working", "B2"),
    (r"fluent|fluid[oa]|fluidez|fluency|advanced|avanzad[oa]|proficient", "C1"),
    (r"intermediate|intermedi[oa]|conversational|conversacional", "B1"),
    (r"basic|basic[oa]|elementary|reading|lectura|technical|tecnic[oa]", "A2"),
)
_CEFR_RE = re.compile(r"(?<![a-z0-9])([abc][12])(?![a-z0-9])")
_LEVEL_CUE = re.compile(r"\b(level|nivel|proficiency|dominio|manejo)\b")

_MANDATORY = re.compile(
    r"\b(excluyentes?|obligatori[oa]s?|indispensables?|imprescindibles?|required|"
    r"requerid[oa]s?|mandatory|must have|necesari[oa]s?)\b"
)
_DESIRABLE = re.compile(
    r"\b(deseables?|desirable|nice to have|valorables?|se valora|preferred|preferably|"
    r"preferible|ideally|idealmente|a plus|un plus|bonus)\b"
)
_MARKER_STRIP = re.compile(
    r"\(?\s*\b(excluyentes?|obligatori[oa]s?|indispensables?|imprescindibles?|required|"
    r"requerid[oa]s?|mandatory|must[ -]have|necesari[oa]s?|deseables?|desirable|"
    r"nice[ -]to[ -]have|valorables?|se valora|preferred|a plus|un plus)\b\s*\)?\s*:?",
    re.I,
)
_GENERIC_HEADING = re.compile(
    r"^(requisitos?|requirements?|requerimientos|conocimientos|competencias|skills|"
    r"habilidades|qualifications|calificaciones)\b"
)

_STRONG_RESIDE = re.compile(
    r"\b(?:reside|residing|resident|residents|live|living|residir|resida|residan|residentes?|"
    r"residencia|radicad[oa]s?|vivir|viva|vivan)\s+(?:in|within|en|dentro de)\b\s*:?\s*(?P<tail>.+)"
)
_WEAK_RESIDE = re.compile(
    r"\b(?:based|located|ubicad[oa]s?)\s+(?:in|within|en)\b\s*:?\s*(?P<tail>.+)"
)
_CANDIDATE_CUE = re.compile(
    r"\b(candidates?|applicants?|you|must|should|only|postulantes?|candidat[oa]s?|debes|debe|"
    r"deben|solo|unicamente|exclusivamente|personas?)\b"
)
_ANYWHERE = re.compile(
    r"\banywhere\b|cualquier (?:parte|lugar|pais)|todo el mundo|desde donde quieras"
)
_TAIL_STOP = re.compile(r"[.;(]|\bbut\b|\bpero\b")

_WORK_AUTH = re.compile(
    r"\bcitizen(ship)?s?\b|\bciudadan[oa]s?\b|\bvisa\b|work (permit|authori[sz]ation)|"
    r"authori[sz]ed to work|permiso de trabajo|sponsorship|patrocinio|right to work|green card"
)

_CONTRACTOR = re.compile(
    r"\bcontractor\b|\bcontract based\b|\bfreelancer?\b|non employment contract|"
    r"not an employment contract|prestacion de servicios|honorarios|contrato de servicios"
)
_INDEFINITE = re.compile(r"\bindefinid[oa]\b|\bindefinite\b|\bpermanent (contract|position|role)\b")
_FIXED_TERM = re.compile(r"plazo fijo|fixed term|temporary contract|contrato temporal")
_EMPLOYEE = re.compile(r"employment contract|contrato de trabajo|\bplanilla\b")

_SALARY_LABEL = re.compile(
    r"^(salary range|salary|salario|sueldo|renta|remuneracion|compensacion|compensation|pay|"
    r"rango salarial)\b\s*:"
)
_SALARY_ASK = re.compile(r"pretensi(on|ones) (de )?(renta|salarial)|salary expectations?")
_SALARY_HIDDEN = re.compile(
    r"not disclosed|undisclosed|confidencial|confidential|a convenir|no publicad|no visible|"
    r"competitive|acorde|segun experiencia|to be discussed"
)
_AMOUNT = re.compile(r"(usd|clp|mxn|cop|pen|ars|brl|eur|uf|us\$|\$|€)\s?\d|\d\s?(usd|clp|eur|uf)\b")

_HYBRID = re.compile(r"\bhibrid[oa]\b|\bhybrid\b")
_ONSITE = re.compile(r"\bpresencial\b|\bon ?site\b|\bin office\b")
_REMOTE = re.compile(r"\bremot[oa]\b|\bremote\b|\bteletrabajo\b|\bhome office\b|work from home")

_AVAILABILITY = re.compile(
    r"disponibilidad inmediata|incorporacion inmediata|immediate(ly)? (start|availability|"
    r"available)|start immediately|notice period|preaviso|available to start"
)
_INSTRUCTION = re.compile(
    r"send (us )?(your )?(cv|resume)|envia(r|nos)? (tu |su )?(cv|curriculum)|"
    r"apply (via|through|by|at)|postula(r)? (a traves|via|por|en el|en la)|cover letter|"
    r"carta de presentacion|portfolio|portafolio|\bsubject\b|\basunto\b|"
    r"appl(y|ying) in|postular en|(cv|resume) (in|en) "
)
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:anos|years?|yrs)\b")
_EXPERIENCE = re.compile(r"experien")

_DEADLINE = re.compile(
    r"(fecha limite|deadline|postula hasta|postulaciones hasta|apply by|applications close|"
    r"cierre de postulaciones|closing date)\s*:?\s*"
    r"(?P<date>\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})"
)

_HEADER_LINE = re.compile(r"^(title|company|location|seniority|url|empresa|ubicacion|cargo)\s*:")
_BULLET = re.compile(r"^\s*(?:[-*•·▪◦–]|\d+[.)])\s+")
_MAX_ITEM_WORDS = 12


def _fold(text: str, *, keep_hyphens: bool = False) -> str:
    """Case- and accent-free, hyphens as spaces; '+' and punctuation survive."""
    stripped = unicodedata.normalize("NFKD", text or "")
    plain = "".join(ch for ch in stripped if not unicodedata.combining(ch)).casefold()
    if not keep_hyphens:
        plain = plain.replace("-", " ")
    return re.sub(r"\s+", " ", plain).strip()


def cefr_rank(level: str | None) -> int | None:
    """A1 → 1 … C2 → 6; None for anything that is not a CEFR level."""
    key = (level or "").strip().upper()
    return _CEFR.index(key) + 1 if key in _CEFR else None


def level_from_text(text: str) -> str | None:
    """'B2+' → B2, 'fully-fluent' → C1, 'conversacional' → B1; None when it says no level."""
    folded = _fold(text)
    found = _CEFR_RE.search(folded)
    if found:
        return found.group(1).upper()
    for pattern, level in _LEVEL_WORDS:
        if re.search(rf"\b(?:{pattern})\b", folded):
            return level
    return None


_MIN_LANGUAGE_NAME = 4


@lru_cache(maxsize=1)
def _language_names() -> dict[str, str]:
    """Language names in Spanish and English from CLDR (folded name → ISO 639-1)."""
    names: dict[str, str] = {}
    for locale in ("es", "en"):
        for code, name in Locale(locale).languages.items():
            folded = _fold(str(name))
            if len(code) == 2 and len(folded) >= _MIN_LANGUAGE_NAME:
                names.setdefault(folded, code)
    return names


def language_code(name: str) -> str | None:
    """'Inglés' / 'english' / 'en' → 'en'; None when it is not a language name."""
    folded = _fold(name)
    if folded in _language_names():
        return _language_names()[folded]
    if len(folded) == 2 and folded in set(_language_names().values()):
        return folded
    return None


@lru_cache(maxsize=1)
def _language_pattern() -> re.Pattern[str]:
    alternatives = sorted(_language_names(), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(a) for a in alternatives) + r")(?![a-z])")


def language_mentions(text: str) -> list[tuple[str, str | None]]:
    """(ISO code, CEFR level or None) for each language a line names, levels kept apart."""
    folded = _fold(text)
    matches = list(_language_pattern().finditer(folded))
    out: list[tuple[str, str | None]] = []
    if len(matches) == 1:
        code = _language_names()[matches[0].group(1)]
        return [(code, level_from_text(folded))]
    previous_end = 0
    for index, match in enumerate(matches):
        code = _language_names()[match.group(1)]
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(folded)
        level = level_from_text(folded[match.end() : next_start]) or level_from_text(
            folded[previous_end : match.start()]
        )
        previous_end = match.end()
        if all(code != seen for seen, _ in out):
            out.append((code, level))
    return out


def _flag(folded: str) -> bool | None:
    mandatory = bool(_MANDATORY.search(folded))
    desirable = bool(_DESIRABLE.search(folded))
    if mandatory == desirable:
        return None
    return mandatory


def _strip_markers(line: str) -> str:
    cleaned = _MARKER_STRIP.sub(" ", line)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -–:,.;()")


def _parse_date(raw: str) -> date | None:
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return date.fromisoformat(raw)
        day, month, year = (int(part) for part in re.split(r"[/-]", raw))
        return date(year, month, day)
    except ValueError:
        return None


def _residency(line: str, folded: str) -> Condition | None:
    match = _STRONG_RESIDE.search(folded)
    if match is None:
        weak = _WEAK_RESIDE.search(folded)
        if weak is not None and _CANDIDATE_CUE.search(folded[: weak.start()]):
            match = weak
    if match is None:
        if _ANYWHERE.search(folded):
            return Condition(kind=ConditionKind.RESIDENCY, source=line, anywhere=True)
        return None
    tail = _TAIL_STOP.split(match.group("tail"), maxsplit=1)[0].strip(" :,")
    if _ANYWHERE.search(tail):
        return Condition(kind=ConditionKind.RESIDENCY, source=line, anywhere=True)
    countries = countries_in_text(tail)
    if not countries and not tail:
        return None
    return Condition(
        kind=ConditionKind.RESIDENCY,
        source=line,
        mandatory=True,
        countries=countries,
        detail=None if countries else tail,
    )


def _languages(line: str, folded: str, flag: bool | None) -> list[Condition]:
    mentions = language_mentions(line)
    if not mentions:
        return []
    instruction = bool(_INSTRUCTION.search(folded))
    has_cue = bool(_LEVEL_CUE.search(folded)) or _flag(folded) is not None
    out = []
    for code, level in mentions:
        if level is None and (instruction or not has_cue):
            continue
        out.append(
            Condition(
                kind=ConditionKind.LANGUAGE,
                source=line,
                language=code,
                level=level,
                mandatory=_flag(folded) if _flag(folded) is not None else flag,
            )
        )
    return out


def _contract_value(folded: str) -> str | None:
    if _CONTRACTOR.search(folded):
        return "contractor"
    if _INDEFINITE.search(folded):
        return "indefinite"
    if _FIXED_TERM.search(folded):
        return "fixed_term"
    if _EMPLOYEE.search(folded):
        return "employee"
    return None


def _salary(line: str, folded: str) -> Condition | None:
    label = _SALARY_LABEL.match(folded)
    if label is not None:
        value = line.split(":", 1)[1].strip(" .;") if ":" in line else ""
        if _SALARY_HIDDEN.search(folded) or not re.search(r"\d", value):
            return Condition(kind=ConditionKind.SALARY, source=line, shown=False, detail=value)
        return Condition(kind=ConditionKind.SALARY, source=line, shown=True, detail=value)
    if _SALARY_ASK.search(folded):
        return Condition(
            kind=ConditionKind.SALARY,
            source=line,
            detail="the posting asks for your salary expectation",
        )
    if _AMOUNT.search(folded):
        return Condition(kind=ConditionKind.SALARY, source=line, shown=True, detail=line)
    return None


def _modality_value(folded: str) -> str | None:
    if _HYBRID.search(folded):
        return "hybrid"
    onsite = bool(_ONSITE.search(folded))
    remote = bool(_REMOTE.search(folded))
    if onsite and remote:
        return "hybrid"
    if onsite:
        return "onsite"
    if remote:
        return "remote"
    return None


def _line_conditions(line: str, folded: str, flag: bool | None) -> list[Condition]:
    """Every non-requirement condition one line states."""
    out: list[Condition] = []
    deadline = _DEADLINE.search(_fold(line, keep_hyphens=True))
    if deadline is not None:
        when = _parse_date(deadline.group("date"))
        if when is not None:
            out.append(Condition(kind=ConditionKind.DEADLINE, source=line, when=when))
    residency = _residency(line, folded)
    if residency is not None:
        out.append(residency)
    if _WORK_AUTH.search(folded):
        out.append(
            Condition(
                kind=ConditionKind.WORK_AUTHORIZATION,
                source=line,
                mandatory=True,
                countries=countries_in_text(line),
            )
        )
    out.extend(_languages(line, folded, flag))
    contract = _contract_value(folded)
    if contract is not None:
        out.append(Condition(kind=ConditionKind.CONTRACT, source=line, contract=contract))
    salary = _salary(line, folded)
    if salary is not None:
        out.append(salary)
    modality = _modality_value(folded)
    if modality is not None and residency is None:
        out.append(Condition(kind=ConditionKind.MODALITY, source=line, modality=modality))
    if _AVAILABILITY.search(folded):
        out.append(Condition(kind=ConditionKind.AVAILABILITY, source=line))
    years = _YEARS.search(folded)
    if years is not None and _EXPERIENCE.search(folded):
        out.append(
            Condition(
                kind=ConditionKind.SENIORITY, source=line, detail=f"{years.group(1)}+ years"
            )
        )
    if _INSTRUCTION.search(folded):
        mentions = language_mentions(line)
        out.append(
            Condition(
                kind=ConditionKind.INSTRUCTION,
                source=line,
                language=mentions[0][0] if mentions else None,
            )
        )
    return out


def _heading_flag(line: str, folded: str) -> tuple[bool, bool | None]:
    """(is a heading, mandatory flag of the section it opens)."""
    words = line.rstrip(":").split()
    if not words:
        return False, None
    if line.endswith(":") and len(words) <= 6:
        return True, _flag(folded)
    if len(words) > 4 or line[-1] in ".!?" or ":" in line or not line[0].isupper():
        return False, None
    flag = _flag(folded)
    if flag is None:
        return True, None
    capitalized = all(w.strip("()")[:1].isupper() for w in words if w.strip("()"))
    bare = _fold(_strip_markers(line))
    if capitalized or _GENERIC_HEADING.match(bare) or not bare:
        return True, flag
    return False, None


def _requirement(line: str, folded: str, flag: bool | None) -> Condition | None:
    own = _flag(folded)
    mandatory = own if own is not None else flag
    if mandatory is None:
        return None
    detail = _strip_markers(line)
    if not detail:
        return None
    return Condition(
        kind=ConditionKind.REQUIREMENT, source=line, mandatory=mandatory, detail=detail
    )


def _text_conditions(text: str) -> list[Condition]:
    out: list[Condition] = []
    section: bool | None = None
    items_in_section = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if items_in_section:
                section, items_in_section = None, 0
            continue
        is_bullet = bool(_BULLET.match(stripped))
        line = _BULLET.sub("", stripped).strip()
        folded = _fold(line)
        if not line or _HEADER_LINE.match(folded):
            continue
        in_section = section is not None and (
            is_bullet or len(line.split()) <= _MAX_ITEM_WORDS
        )
        flag = section if in_section else None
        found = _line_conditions(line, folded, flag)
        if found:
            out.extend(found)
            items_in_section += 1 if in_section else 0
            continue
        is_heading, heading_flag = _heading_flag(line, folded)
        if is_heading and not is_bullet:
            section, items_in_section = heading_flag, 0
            continue
        if section is not None and not in_section:
            section, items_in_section = None, 0
        requirement = _requirement(line, folded, flag)
        if requirement is not None:
            out.append(requirement)
            items_in_section += 1 if in_section else 0
    return out


def _field_conditions(job: JobPosting, found: list[Condition]) -> list[Condition]:
    kinds = {c.kind for c in found}
    out: list[Condition] = []
    remote = _fold(job.remote_type or "").replace("_", " ")
    if remote and ConditionKind.MODALITY not in kinds:
        modality = (
            "hybrid"
            if "hybrid" in remote or "hibrid" in remote
            else "onsite"
            if re.search(r"on ?site|physical|presencial|office", remote)
            else "remote"
            if re.search(r"remot|anywhere", remote)
            else None
        )
        if modality is not None:
            out.append(
                Condition(
                    kind=ConditionKind.MODALITY,
                    source=f"remote_type: {job.remote_type}",
                    modality=modality,
                )
            )
    employment = _fold(job.employment_type or "")
    if employment and ConditionKind.CONTRACT not in kinds:
        contract = (
            "contractor"
            if re.search(r"freelance|contract", employment)
            else "indefinite"
            if "permanent" in employment
            else "fixed_term"
            if re.search(r"temporary|fixed", employment)
            else None
        )
        if contract is not None:
            out.append(
                Condition(
                    kind=ConditionKind.CONTRACT,
                    source=f"employment_type: {job.employment_type}",
                    contract=contract,
                )
            )
    if job.seniority and ConditionKind.SENIORITY not in kinds:
        out.append(
            Condition(
                kind=ConditionKind.SENIORITY,
                source=f"seniority: {job.seniority}",
                detail=job.seniority,
            )
        )
    return out


def _with_place(condition: Condition, job: JobPosting) -> Condition:
    """An onsite or hybrid seat sits in the posting's location: name its country."""
    if condition.kind is not ConditionKind.MODALITY or condition.modality == "remote":
        return condition
    countries = countries_in_text(job.location or "")
    if not countries:
        return condition
    return Condition(
        kind=condition.kind,
        source=condition.source,
        modality=condition.modality,
        countries=countries,
        detail=job.location,
    )


def _dedupe(conditions: list[Condition]) -> list[Condition]:
    seen: set[tuple[object, ...]] = set()
    out: list[Condition] = []
    for c in conditions:
        key = (c.kind, c.countries, c.anywhere, c.language, c.contract, c.modality, c.when,
               c.detail if c.kind in {ConditionKind.REQUIREMENT, ConditionKind.SENIORITY} else None,
               c.source if c.kind is ConditionKind.INSTRUCTION else None)  # fmt: skip
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def posting_conditions(job: JobPosting) -> list[Condition]:
    """The special conditions a stored posting states, in the order it states them."""
    text = job.description or job.raw_description or ""
    found: list[Condition] = []
    closed = closure_evidence(job.description) or closure_evidence(job.raw_description)
    if closed:
        found.append(Condition(kind=ConditionKind.CLOSED, source=closed))
    found.extend(_text_conditions(text))
    found.extend(_field_conditions(job, found))
    return _dedupe([_with_place(c, job) for c in found])
