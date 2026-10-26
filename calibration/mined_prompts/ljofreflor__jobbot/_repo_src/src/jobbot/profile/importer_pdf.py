"""Import a CV exported to PDF (Word-style layout) into a Candidate profile dict."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from email_validator import EmailNotValidError, validate_email
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from jobbot.jobs.geo import country_name, detect_country, normalize_country
from jobbot.jobs.normalization import fold_text
from jobbot.profile.importer_common import (
    ImportResult,
    empty_skills,
    is_present,
    map_skill_group,
    normalize_key,
    parse_one_date,
    section_match,
    slugify,
    unique_id,
)
from jobbot.profile.pdf_glyphs import repair_missing_ligatures

logger = logging.getLogger("jobbot.profile.importer_pdf")

NO_TEXT_LAYER_HINT = """\
This PDF carries no text layer, so it is a scan or an image export.

JobBot does not run OCR (no downloadable models, offline only). Options:
  - export the CV to PDF again from the original document
  - save it as .docx/.txt and rebuild it, or
  - use `jobbot profile import-latex` with the LaTeX source
"""

_BULLETS = "•●○▪▫◦‣∙·*"
_DASH = r"[-–—−]"
_PRESENT_WORDS = r"actualidad|presente|actual|hoy|present|current|now|today"
_MONTH_NAME = r"(?:ene|jan|feb|mar|abr|apr|may|jun|jul|ago|aug|sep|sept|oct|nov|dic|dec)[a-z.]*"
_MONTH_YEAR = r"\d{1,2}\s*/\s*\d{2,4}"
_YEAR = r"(?:19|20)\d{2}"
# Month/year forms come first: the year alternative must not win inside '10/2024'.
_DATE = rf"(?:{_MONTH_YEAR}|{_MONTH_NAME}\s+{_YEAR}|{_YEAR})"
_RANGE_SEP = rf"(?:\s*{_DASH}\s*|\s+(?:to|a|al|hasta|until)\s+)"

# A company name says "company"; a role name says "role". Used to decide which
# neighbouring line is which, because every CV design orders them differently.
_ORG_RE = re.compile(
    r"\b(spa|s a|sa|ltda|limitada|eirl|inc|llc|ltd|gmbh|corp|corporation|company|"
    r"group|holding|consultores|consultora|consulting|banco|bank|caja|seguros|"
    r"insurance|universidad|university|instituto|institute|fundacion|foundation|"
    r"ministerio|municipalidad|hospital|clinica|colegio|school|agencia|agency|"
    r"studio|labs|technologies|solutions|servicios|services|partners)\b"
)
_ROLE_RE = re.compile(
    r"\b(analista|analyst|ingenier\w*|engineer|coordinador\w*|coordinadora|coordinator|"
    r"gerente|manager|jefe|jefa|head|especialista|specialist|asistente|assistant|"
    r"consultor|consultora|consultant|cientific\w*|scientist|desarrollador\w*|developer|"
    r"director\w*|directora|lead|practicante|intern|tecnic\w*|technician|supervisor\w*|"
    r"operador\w*|operator|profesor\w*|teacher|docente|abogad\w*|lawyer|contador\w*|"
    r"accountant|auditor\w*|medic\w*|enfermer\w*|vendedor\w*|sales|ejecutiv\w*|executive|"
    r"arquitect\w*|architect|disenador\w*|designer|investigador\w*|researcher|"
    r"administrativ\w*|administrativa|secretari\w*|conductor|driver|guardia|prevencionista|"
    r"geolog\w*|geofisic\w*|topograf\w*|planificador\w*|planner|controller|tesorer\w*|"
    r"officer|chief|presidente|president|owner|founder)\b"
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*[\w]")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?:linkedin\.com|github\.com)/\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s().]{6,}\d")
_PHONE_HINT_RE = re.compile(r"m[óo]vil|celular|tel[eé]fono|tel\.|phone|fono", re.IGNORECASE)
_DOI_RE = re.compile(
    r"(?:doi[:\s]*|https?://doi\.org/)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)

_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE}){_RANGE_SEP}(?P<end>{_DATE}|{_PRESENT_WORDS})\b",
    re.IGNORECASE,
)
_PARTIAL_RANGE_RE = re.compile(
    rf"^{_DATE}\s*(?:{_DASH}|to|a|al|hasta|until)$",
    re.IGNORECASE,
)
_LONE_DATE_RE = re.compile(rf"(?<!\w){_DATE}(?!\w)", re.IGNORECASE)
_SHARED_YEAR_RE = re.compile(
    rf"\b(?P<start>{_MONTH_NAME})(?P<sep>{_RANGE_SEP})(?P<end>{_MONTH_NAME}\s+{_YEAR})\b",
    re.IGNORECASE,
)
# A role with a single date: the CV writes it right-aligned, so it closes the line.
_TRAILING_DATE_RE = re.compile(rf"(?<!\w)(?P<start>{_DATE})\s*$", re.IGNORECASE)

# Word ships 'fi' and 'ff' as one glyph; the profile wants letters.
_LIGATURE_CHARS = str.maketrans(
    {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
)
_INLINE_RANGE_RE = re.compile(
    rf"\(?\s*(?P<start>{_DATE})\s*{_DASH}\s*"
    rf"(?P<end>{_DATE}|[A-Za-zÁÉÍÓÚÑáéíóúñ]+)\s*\)?",
    re.IGNORECASE,
)
_INLINE_SINGLE_RE = re.compile(rf"\(\s*(?P<start>{_MONTH_YEAR}|{_YEAR})\s*\)")

_DUTIES_RE = re.compile(
    r"^(responsabilidades|responsabilidad|funciones|responsibilities|duties)\s*:?\s*",
    re.IGNORECASE,
)
_RESULTS_RE = re.compile(
    r"^(logros|logro|resultados|achievements|highlights)\s*:?\s*$",
    re.IGNORECASE,
)

# Column labels used as skill sub-headers; they are not skills themselves.
_SKILL_LABELS = frozenset(
    {
        "interpersonales",
        "tecnicas",
        "blandas",
        "duras",
        "soft skills",
        "hard skills",
        "otras",
        "otros",
    }
)


class PdfImportError(Exception):
    """Raised when the PDF cannot be read or carries no text."""


@dataclass
class _Block:
    """A heading and the lines under it; the CV header has kind None."""

    kind: str | None
    title: str
    lines: list[str] = field(default_factory=list)


@dataclass
class _Entry:
    """One role: the date line plus the lines around it, before deciding what is what."""

    range_text: str
    residual: str
    above: list[str] = field(default_factory=list)
    below: list[str] = field(default_factory=list)
    duties: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    prose: list[str] = field(default_factory=list)


def extract_pdf_text(path: Path) -> tuple[str, list[str]]:
    """Text of every page plus font warnings; raises when the PDF has no text layer."""
    if not path.is_file():
        msg = f"PDF CV not found: {path}"
        raise PdfImportError(msg)

    try:
        reader = PdfReader(str(path))
        warnings = repair_missing_ligatures(reader)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, OSError, ValueError) as exc:
        msg = f"Could not read PDF {path}: {exc}"
        raise PdfImportError(msg) from exc

    text = "\n".join(pages)
    if not text.strip():
        msg = f"{path} has no text layer.\n\n{NO_TEXT_LAYER_HINT}"
        raise PdfImportError(msg)
    return text, warnings


def import_pdf_cv(path: Path) -> ImportResult:
    """Parse a PDF CV into a profile mapping, warning about everything unclear."""
    text, font_warnings = extract_pdf_text(path)
    result = parse_cv_text(text, path)
    # A broken glyph explains every field that reads wrong, so it is reported first.
    result.warnings[:0] = font_warnings
    return result


def parse_cv_text(text: str, source: Path) -> ImportResult:
    """Parse already extracted CV text; the PDF layout rules live here."""
    lines = _normalize_lines(text)
    warnings: list[str] = []
    blocks = _split_blocks(lines)

    personal, summary = _parse_header(_header_lines(blocks), warnings)
    section_summary = _parse_summary_section(blocks)

    data: dict[str, Any] = {
        "personal": personal,
        "summary": section_summary or summary,
        "specialties": [],
        "experience": _parse_experience(blocks, warnings),
        "education": _parse_education(blocks, warnings),
        "skills": _parse_skills(blocks, warnings),
        "publications": _parse_publications(blocks),
    }

    glued = _glued_text_warning(lines)
    if glued:
        warnings.append(glued)
    stranded = _stranded_roles_warning(blocks)
    if stranded:
        warnings.append(stranded)
    if not data["experience"]:
        warnings.append("No experience entries detected; check the section headings")
    if not data["summary"]:
        warnings.append("No summary detected; write personal.summary by hand")

    logger.info(
        "Imported from %s: experiences=%s achievements=%s skills=%s",
        source,
        len(data["experience"]),
        sum(len(e["achievements"]) for e in data["experience"]),
        sum(len(v) for v in data["skills"].values()),
    )
    return ImportResult(data=data, source_path=source, warnings=warnings)


def _normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\xa0", " ").replace("\u2028", "\n").splitlines():
        line = _normalize_bullets(raw.rstrip())
        line = _collapse_letter_spacing(line)
        line = re.sub(r"[ \t]+", " ", line).strip().translate(_LIGATURE_CHARS)
        if line:
            lines.append(_expand_shared_year(line))
    return lines


def _expand_shared_year(line: str) -> str:
    """'julio – agosto 2026' is a range: only its end carries the year both dates share."""

    def repeat_year(match: re.Match[str]) -> str:
        year = match.group("end").split()[-1]
        return f"{match.group('start')} {year}{match.group('sep')}{match.group('end')}"

    return _SHARED_YEAR_RE.sub(repeat_year, line)


def _stranded_roles_warning(blocks: list[_Block]) -> str | None:
    """Two-column PDFs interleave the columns, so roles land under another heading."""
    stranded = 0
    for block in blocks:
        if block.kind in {"experience", "education", "publications"}:
            continue
        for line in block.lines:
            if _entry_range(line) is None:
                continue
            folded = fold_text(line)
            if _ROLE_RE.search(folded) or _ORG_RE.search(folded):
                stranded += 1
    if not stranded:
        return None
    return (
        f"{stranded} dated role-looking line(s) sit outside the experience section "
        "(two-column PDFs interleave columns); check the CV and add them by hand"
    )


def _header_lines(blocks: list[_Block]) -> list[str]:
    """When the page opens with a title such as 'Summary', the name falls inside it."""
    if not blocks:
        return []
    if blocks[0].lines or len(blocks) == 1:
        return blocks[0].lines
    return blocks[1].lines


def _glued_text_warning(lines: list[str]) -> str | None:
    """Some PDFs embed no space glyph: the words arrive stuck together."""
    long_lines = [line for line in lines if len(line) >= 40]
    if len(long_lines) < 2:
        return None
    glued = sum(1 for line in long_lines if _longest_unbroken_run(line) >= 25)
    if glued / len(long_lines) < 0.3:
        return None
    return (
        "This PDF stores its text without spaces between words, so every field "
        "arrives glued; re-export the CV or fix the YAML by hand"
    )


def _longest_unbroken_run(line: str) -> int:
    return max((len(run) for run in re.findall(r"[^\W\d_]+", line)), default=0)


def _normalize_bullets(line: str) -> str:
    """Word ships its bullet as a Symbol glyph in the private use area (U+F0B7)."""
    return "".join("•" if unicodedata.category(ch) == "Co" else ch for ch in line)


def _collapse_letter_spacing(line: str) -> str:
    """'E X P E R I E N C I A' → 'EXPERIENCIA' (Word spaces out headings)."""
    tokens = line.split()
    if len(tokens) < 4 or any(len(token) > 1 for token in tokens):
        return line
    words = [re.sub(r"\s+", "", word) for word in re.split(r"\s{2,}", line.strip())]
    return " ".join(word for word in words if word)


def _split_blocks(lines: list[str]) -> list[_Block]:
    blocks: list[_Block] = [_Block(kind=None, title="")]
    for line in lines:
        kind = _heading_kind(line)
        if kind is None:
            blocks[-1].lines.append(line)
        else:
            blocks.append(_Block(kind=kind, title=line))
    return blocks


def _heading_kind(line: str) -> str | None:
    """A heading is short, alias-dominated and not a sentence."""
    if len(line) > 60 or line[:1] in _BULLETS:
        return None
    if re.search(r"[.;:]", line) or _RANGE_RE.search(line):
        return None
    match = section_match(line)
    if match is None and _is_shouted(line) and " " in line:
        # Tracked headings lose their word gaps: 'R E S U  M E N' → 'RESU MEN'.
        match = section_match(line.replace(" ", ""))
    if match is None:
        return None
    kind, coverage = match
    if coverage < 0.4:
        return None
    if not _is_shouted(line) and coverage < 1.0:
        # A lowercase line only heads a section when it is exactly the alias.
        return None
    return kind


def _is_shouted(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.6


def _parse_header(lines: list[str], warnings: list[str]) -> tuple[dict[str, Any], str | None]:
    personal: dict[str, Any] = {
        "name": None,
        "headline": None,
        "city": None,
        "country": None,
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
    }

    leftovers: list[str] = []
    for line in lines:
        residual = _absorb_contacts(line, personal, warnings)
        if residual is None:
            leftovers.append(line)
            continue
        text = residual.strip(" |·-–—")
        if len(re.sub(r"[\W\d_]+", "", text)) >= 6:
            leftovers.append(text)

    name_index = next(
        (i for i, line in enumerate(leftovers) if _looks_like_name(line)),
        None,
    )
    if name_index is None:
        personal["name"] = "Unknown"
        warnings.append("Could not detect the candidate name; set personal.name by hand")
    else:
        personal["name"] = leftovers.pop(name_index)

    if leftovers:
        personal["headline"] = leftovers.pop(0)
    else:
        personal["headline"] = "Professional"
        warnings.append("Missing headline; used a placeholder")

    summary_parts: list[str] = []
    for line in leftovers:
        if personal["city"] is None and _looks_like_location(line):
            _absorb_location(line, personal)
            continue
        summary_parts.append(line)

    summary = " ".join(summary_parts).strip() or None
    return personal, summary


def _absorb_contacts(line: str, personal: dict[str, Any], warnings: list[str]) -> str | None:
    """Take email/phone/profile URLs out of a line; return what is left, or None."""
    found = False
    rest = line

    for raw in _EMAIL_RE.findall(line):
        rest = rest.replace(raw, " ")
        found = True
        if personal["email"] is not None:
            continue
        try:
            personal["email"] = validate_email(raw, check_deliverability=False).normalized
        except EmailNotValidError:
            warnings.append(f"Ignored an unreadable email: {raw}")

    for raw in _URL_RE.findall(rest) + _HANDLE_RE.findall(rest):
        rest = rest.replace(raw, " ")
        found = True
        _absorb_profile_url(raw.rstrip(".,;)"), personal)

    phone = _PHONE_RE.search(re.sub(r"\d{1,2}\s*/\s*\d{2,4}", " ", rest))
    looks_like_phone = phone is not None and (
        _PHONE_HINT_RE.search(line) is not None or "+" in phone.group(0)
    )
    if phone is not None and looks_like_phone and personal["phone"] is None:
        personal["phone"] = re.sub(r"\s+", " ", phone.group(0)).strip()
        rest = rest.replace(phone.group(0), " ")
        found = True

    if not found:
        return None
    return re.sub(r"\s+", " ", re.sub(_PHONE_HINT_RE, " ", rest)).strip()


def _absorb_profile_url(url: str, personal: dict[str, Any]) -> None:
    normalized = url if url.lower().startswith("http") else f"https://{url}"
    lowered = normalized.lower()
    if "linkedin.com" in lowered and personal["linkedin"] is None:
        personal["linkedin"] = normalized
    elif "github.com" in lowered and personal["github"] is None:
        personal["github"] = normalized


def _looks_like_name(line: str) -> bool:
    tokens = line.split()
    if not 2 <= len(tokens) <= 6 or len(line) > 60:
        return False
    if any(ch.isdigit() or ch in "@|/" for ch in line):
        return False
    return all(token[:1].isupper() for token in tokens if token[:1].isalpha())


def _looks_like_location(line: str) -> bool:
    if len(line) > 60 or line.endswith("."):
        return False
    return "," in line or detect_country(line) is not None


def _absorb_location(line: str, personal: dict[str, Any]) -> None:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    personal["city"] = parts[0] if parts else None
    tail = parts[-1] if len(parts) > 1 else ""
    code = normalize_country(tail) or detect_country(line)
    personal["country"] = country_name(code) or (tail or None)


def _parse_summary_section(blocks: list[_Block]) -> str | None:
    for block in blocks:
        if block.kind == "summary" and block.lines:
            return " ".join(block.lines).strip() or None
    return None


def _parse_experience(blocks: list[_Block], warnings: list[str]) -> list[dict[str, Any]]:
    experiences: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for block in blocks:
        if block.kind != "experience":
            continue
        previous_company: str | None = None
        lines = _join_wrapped_orgs(_join_wrapped_ranges(block.lines))
        for entry in _split_entries(lines):
            built = _build_experience(entry, previous_company, seen_ids, warnings)
            if built is not None:
                if built["company"] != "Unknown":
                    previous_company = built["company"]
                experiences.append(built)

    return experiences


def _join_wrapped_ranges(lines: list[str]) -> list[str]:
    """'2022 to' / 'present' on two lines is one date range."""
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if index + 1 < len(lines) and _PARTIAL_RANGE_RE.match(line):
            line = f"{line} {lines[index + 1]}"
            index += 1
        joined.append(line)
        index += 1
    return joined


def _join_wrapped_orgs(lines: list[str]) -> list[str]:
    """A long employer wraps: '…, Gabinete Ministra,' / 'Ministerio de Salud, Chile.'"""
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.rstrip().endswith(",") and _place_tail(following) and not _place_tail(line):
            line = f"{line} {following}"
            index += 1
        joined.append(line)
        index += 1
    return joined


def _place_tail(line: str) -> bool:
    """The line closes with a place of its own: '…, Ministerio de Salud, Chile.'

    Prose that merely names a country does not qualify: the place has to be the last
    comma field and short, the way a CV writes an employer's seat.
    """
    chunks = [chunk.strip(" .") for chunk in line.split(",") if chunk.strip(" .")]
    if not chunks:
        return False
    tail = chunks[-1]
    return len(tail.split()) <= 4 and _is_place(tail)


def _split_entries(lines: list[str]) -> list[_Entry]:
    """Group the section around its date lines; each date line starts a role."""
    entries: list[_Entry] = []
    pending: list[str] = []
    current: _Entry | None = None
    mode = "head"

    for index, line in enumerate(lines):
        found = _entry_range(line)
        if found is not None:
            current = _Entry(
                range_text=found.group(0),
                residual=_strip_dates(line),
                above=pending[-2:],
            )
            entries.append(current)
            pending = []
            mode = "head"
            continue

        is_bullet = line[:1] in _BULLETS
        next_starts_entry = index + 1 < len(lines) and _entry_range(lines[index + 1]) is not None

        if not is_bullet and next_starts_entry and _is_header_like(line):
            # A company or role line announcing the next role, not the tail of this one.
            pending.append(line)
            continue

        if current is None:
            if _is_header_like(line):
                pending.append(line)
            continue

        duties = _DUTIES_RE.match(line)
        if duties:
            mode = "duties"
            tail = line[duties.end() :].strip()
            if tail:
                current.duties.append(tail)
            continue

        if _RESULTS_RE.match(line):
            mode = "bullets"
            continue

        if is_bullet:
            bullet = line[1:].lstrip(" .-\u2013\u2014\u00b7").strip()
            if bullet:
                current.bullets.append(bullet)
            mode = "bullets"
            continue

        if mode == "head" and not current.below and "," in line and _place_tail(line):
            # The employer right under the dates may be long and end in a full stop,
            # which no header-like test accepts; its trailing place is the evidence.
            current.below.append(line)
            continue

        if mode == "head" and len(current.below) < 3 and _is_header_like(line):
            current.below.append(line)
            continue

        if mode == "duties":
            current.duties.append(line)
            continue

        if mode == "bullets" and current.bullets:
            current.bullets[-1] = f"{current.bullets[-1]} {line}".strip()
            continue

        mode = "prose"
        current.prose.append(line)

    return entries


def _entry_range(line: str) -> re.Match[str] | None:
    """A date on a header-like line starts a role; prose never does."""
    if not _is_header_like(line):
        return None
    found = _RANGE_RE.search(line)
    if found is not None:
        return found
    return _single_date_entry(line)


def _single_date_entry(line: str) -> re.Match[str] | None:
    """'Docente 2013': the role lasted one year, written right-aligned on its line."""
    found = _TRAILING_DATE_RE.search(line)
    if found is None:
        return None
    label = line[: found.start()].strip(" |\u00b7-\u2013\u2014,;/()")
    if len(re.sub(r"[\W\d_]+", "", label)) < 3:
        return None
    return found


def _is_header_like(line: str) -> bool:
    if line[:1] in _BULLETS or len(line) > 90:
        return False
    if line.rstrip().endswith((".", ":")):
        return False
    if _DUTIES_RE.match(line) or _RESULTS_RE.match(line):
        return False
    return len(line.split()) <= 12


def _strip_dates(line: str) -> str:
    """What the date line says besides the dates: a role, a company, or nothing."""
    text = _RANGE_RE.sub(" ", line)
    text = _LONE_DATE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |\u00b7-\u2013\u2014,;/()").strip()


def _build_experience(
    entry: _Entry,
    previous_company: str | None,
    seen_ids: set[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    dates = _parse_entry_range(entry.range_text)
    slots = [
        slot
        for slot in (entry.residual, *reversed(entry.above), *entry.below)
        if slot
    ]

    on_date_line = 0 if entry.residual else -1
    title_index = _best_slot(slots, _role_score, on_date_line, skip=None)
    company_index = _best_slot(slots, _company_score, on_date_line, skip=title_index)

    if company_index is None:
        company_index = next((i for i in range(len(slots)) if i != title_index), None)
    if title_index is None:
        title_index = next(
            (i for i in reversed(range(len(slots))) if i != company_index),
            None,
        )

    title = slots[title_index] if title_index is not None else ""
    company_line = slots[company_index] if company_index is not None else None
    leftovers = [
        slot
        for index, slot in enumerate(slots)
        if index not in {title_index, company_index}
    ]

    if company_line is not None:
        company, location = _split_company_and_location(company_line)
    else:
        company, location = (previous_company or "Unknown"), None
        if previous_company is None:
            warnings.append(f"{title or 'role'}: no company line near the dates")

    if not title:
        title = "Unknown"
        warnings.append(f"{company}: no role title near the dates")

    exp_id = unique_id(slugify(f"{company}-{title}") or "experience", seen_ids)
    if dates["start_date"] is None:
        warnings.append(f"{exp_id}: unreadable start date; entry dropped")
        return None

    if entry.bullets:
        achievement_texts = entry.bullets
        description_parts = [*leftovers, *entry.duties, *entry.prose]
    else:
        achievement_texts = _sentence_blocks(entry.prose)
        description_parts = [*leftovers, *entry.duties]

    built: dict[str, Any] = {
        "id": exp_id,
        "company": company,
        "title": title,
        "location": location,
        "start_date": dates["start_date"],
        "current": dates["current"],
        "description": " ".join(description_parts).strip() or None,
        "achievements": _achievements(exp_id, achievement_texts),
    }
    if dates["current"]:
        built["end_date"] = None
    elif dates["end_date"]:
        built["end_date"] = dates["end_date"]
    else:
        built["end_date"] = dates["start_date"]
        warnings.append(f"{exp_id}: unreadable end_date; mirrored start_date")

    return built


def _best_slot(
    slots: list[str],
    score: Callable[[str, bool], int],
    date_line: int,
    *,
    skip: int | None,
) -> int | None:
    best: tuple[int, int] | None = None
    for index, slot in enumerate(slots):
        if index == skip:
            continue
        value = score(slot, index == date_line)
        if value <= 0:
            continue
        if best is None or value > best[1]:
            best = (index, value)
    return best[0] if best else None


def _role_score(text: str, on_date_line: bool) -> int:
    """A role word scores; an employer word cancels it ('Auditora Consultores')."""
    folded = fold_text(text)
    score = 2 if _ROLE_RE.search(folded) else 0
    if _ORG_RE.search(folded):
        score -= 2
    if on_date_line:
        score += 1
    return score


def _company_score(text: str, on_date_line: bool) -> int:
    folded = fold_text(text)
    has_org = bool(_ORG_RE.search(folded))
    if _ROLE_RE.search(folded) and not has_org:
        return 0
    score = 2 if has_org else 0
    if _is_place(text):
        score += 1
    if text.count(",") >= 2:
        score += 1
    if _is_shouted(text):
        score += 1
    if on_date_line and score == 0:
        # The date line defaults to the role, so it needs its own evidence here.
        return 0
    return score


def _is_place(text: str) -> bool:
    return normalize_country(text) is not None or detect_country(text) is not None


def _split_company_and_location(line: str) -> tuple[str, str | None]:
    """Every design writes the place differently; only split on actual places."""
    text = line.strip(" |\u00b7,;.")
    dashed = re.split(rf"\s+{_DASH}\s+", text, maxsplit=1)
    if len(dashed) > 1 and _is_place(dashed[1]):
        return dashed[0].strip(" .,"), _drop_employment_type(dashed[1])

    chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if len(chunks) >= 2 and normalize_country(chunks[-1]) is not None:
        # The chunk before the country is the city unless it names an organisation:
        # 'Northbank, Toronto, Canada' has a city there, 'Ministerio de Salud' does not.
        if len(chunks) >= 3 and not _ORG_RE.search(fold_text(chunks[-2])):
            return ", ".join(chunks[:-2]), f"{chunks[-2]}, {chunks[-1]}"
        words = chunks[0].split()
        if len(chunks) == 2 and len(words) >= 2 and detect_country(words[-1]) is not None:
            return " ".join(words[:-1]), f"{words[-1]}, {chunks[-1]}"
        return ", ".join(chunks[:-1]), chunks[-1]

    return _drop_employment_type(text), None


def _drop_employment_type(text: str) -> str:
    return re.sub(
        r"\s*(part[- ]time|full[- ]time|jornada parcial|media jornada)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,")


def _sentence_blocks(lines: list[str]) -> list[str]:
    """CVs without bullets wrap one achievement over several lines."""
    blocks: list[str] = []
    buffer: list[str] = []
    for line in lines:
        buffer.append(line)
        if line.rstrip().endswith((".", ";")):
            blocks.append(" ".join(buffer))
            buffer = []
    if buffer:
        blocks.append(" ".join(buffer))
    return blocks


def _parse_entry_range(text: str) -> dict[str, Any]:
    found = _RANGE_RE.search(text)
    if found is None:
        # A lone date is not an unreadable range: the role covers that month or year.
        return {
            "start_date": parse_one_date(text, end=False),
            "end_date": parse_one_date(text, end=True),
            "current": False,
        }
    start = parse_one_date(found.group("start"), end=False)
    end_raw = found.group("end")
    if is_present(end_raw):
        return {"start_date": start, "end_date": None, "current": True}
    return {
        "start_date": start,
        "end_date": parse_one_date(end_raw, end=True),
        "current": False,
    }


def _achievements(exp_id: str, texts: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for idx, text in enumerate(texts, start=1):
        cleaned = text.strip()
        if not cleaned:
            continue
        ach_id = unique_id(slugify(f"{exp_id}-{cleaned[:40]}") or f"{exp_id}-a{idx}", seen)
        out.append({"id": ach_id, "text": cleaned, "tags": [], "metrics": {}})
    return out


def _parse_education(blocks: list[_Block], warnings: list[str]) -> list[dict[str, Any]]:
    education: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for block in blocks:
        if block.kind != "education":
            continue
        for record in _group_records(block.lines):
            item = _education_entry(record, seen_ids, warnings)
            if item:
                education.append(item)

    return education


def _group_records(lines: list[str]) -> list[str]:
    """Join wrapped lines: a record ends when a new bullet or a new date starts one."""
    records: list[str] = []
    buffer: list[str] = []

    for line in lines:
        starts_record = line[:1] in _BULLETS or (
            bool(buffer) and _has_date(" ".join(buffer)) and _has_date(line)
        )
        if starts_record and buffer:
            records.append(" ".join(buffer))
            buffer = []
        buffer.append(line.lstrip("".join(_BULLETS)).strip() if line[:1] in _BULLETS else line)

    if buffer:
        records.append(" ".join(buffer))
    return records


def _has_date(text: str) -> bool:
    return bool(_INLINE_RANGE_RE.search(text) or _INLINE_SINGLE_RE.search(text))


def _trim_date_lead_in(text: str) -> str:
    """'… y Auditor En 2015 - Dic 2020' keeps a dangling 'En' once the dates go."""
    trimmed = re.sub(
        r"\s+(en|desde|entre|de|from|since)\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return trimmed.strip(" .,;–—-")


def _split_institution_and_degree(text: str) -> tuple[str, str]:
    """Institution and degree share a line, split by a dash or by the last comma."""
    dashed = re.split(rf"\s+{_DASH}\s+", text, maxsplit=1)
    if len(dashed) > 1 and dashed[1].strip():
        return dashed[0].strip(" .,;"), dashed[1].strip(" .,;")

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 2:
        return text, ""
    return ", ".join(parts[:-1]), parts[-1]


def _education_entry(
    record: str,
    seen_ids: set[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    start: str | None = None
    end: str | None = None
    head, tail = record, ""

    ranged = _INLINE_RANGE_RE.search(record)
    single = _INLINE_SINGLE_RE.search(record)
    if ranged:
        start = parse_one_date(ranged.group("start"), end=False)
        end = None if is_present(ranged.group("end")) else parse_one_date(
            ranged.group("end"), end=True
        )
        head, tail = record[: ranged.start()], record[ranged.end() :]
    elif single:
        start = parse_one_date(single.group("start"), end=False)
        head, tail = record[: single.start()], record[single.end() :]

    institution = _trim_date_lead_in(head)
    degree = tail.strip(" .,;–—-")

    if not degree:
        institution, degree = _split_institution_and_degree(institution)
    if not degree:
        warnings.append(f"Skipped an unreadable education line: {record[:60]}")
        return None

    edu_id = unique_id(slugify(f"{institution}-{degree}") or "education", seen_ids)
    item: dict[str, Any] = {
        "id": edu_id,
        "institution": institution or "Unknown",
        "degree": degree,
    }
    if start:
        item["start_date"] = start
    if end:
        item["end_date"] = end
    if not start and not end:
        warnings.append(f"Education {edu_id}: no dates detected")
    return item


def _parse_skills(blocks: list[_Block], warnings: list[str]) -> dict[str, list[str]]:
    skills = empty_skills()

    for block in blocks:
        if block.kind != "skills":
            continue
        group = map_skill_group(block.title)
        for line in block.lines:
            stripped = line.lstrip("".join(_BULLETS)).strip()
            if normalize_key(stripped) in _SKILL_LABELS:
                group = map_skill_group(stripped)
                continue
            for item in _split_skill_line(stripped):
                skills.setdefault(group, [])
                if item not in skills[group]:
                    skills[group].append(item)

    if not any(skills.values()):
        warnings.append("No skills detected; fill skills by hand")
    return skills


def _split_skill_line(line: str) -> list[str]:
    """Skills sit in one line separated by pipes, commas, columns or bullets."""
    items: list[str] = []
    separators = rf"[|;,{re.escape(_BULLETS)}]|\s{{2,}}|\s/\s"
    for raw in re.split(separators, line):
        item = raw.strip(" .·-–—").strip()
        if not item or len(item) > 60:
            continue
        if normalize_key(item) in _SKILL_LABELS:
            continue
        items.append(item)
    return items


def _parse_publications(blocks: list[_Block]) -> list[dict[str, Any]]:
    publications: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for block in blocks:
        if block.kind != "publications":
            continue
        for record in _group_records(block.lines):
            year_match = re.search(r"(19|20)\d{2}", record)
            title = re.sub(r"\(?\b(19|20)\d{2}\b\)?", "", record).strip(" .,;")
            if not title:
                continue
            pub_id = unique_id(slugify(title) or "publication", seen_ids)
            item: dict[str, Any] = {"id": pub_id, "title": title}
            if year_match:
                item["year"] = int(year_match.group(0))
            doi = _DOI_RE.search(record)
            if doi:
                item["doi"] = doi.group(1).rstrip(".")
            publications.append(item)

    return publications
