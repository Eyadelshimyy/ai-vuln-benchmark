"""Import a legacy moderncv-style LaTeX CV into a Candidate profile dict."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobbot.profile.importer_common import (
    ImportResult,
    empty_skills,
    map_skill_group,
    parse_date_range,
    section_kind,
    slugify,
    unique_id,
    write_generated_profile,
)

__all__ = [
    "ImportResult",
    "LatexImportError",
    "import_latex_cv",
    "write_generated_profile",
]

logger = logging.getLogger("jobbot.profile.importer_latex")


class LatexImportError(Exception):
    """Raised when the legacy CV cannot be read or parsed."""


def import_latex_cv(path: Path) -> ImportResult:
    """Parse a moderncv-oriented .tex file into a profile mapping."""
    if not path.is_file():
        msg = f"LaTeX CV not found: {path}"
        raise LatexImportError(msg)

    raw = path.read_text(encoding="utf-8")
    source = _strip_comments(raw)
    warnings: list[str] = []

    personal = _parse_personal(source, warnings)
    sections = _split_sections(source)

    summary = _parse_summary(sections)
    experiences = _parse_experiences(sections, warnings)
    education = _parse_education(sections, warnings)
    skills = _parse_skills(sections, warnings)
    publications = _parse_publications(sections, warnings)

    if not personal.get("name"):
        warnings.append("Could not detect \\name{...}{...}; set personal.name manually")
    if not experiences:
        warnings.append("No \\cventry experience blocks detected")

    data: dict[str, Any] = {
        "personal": personal,
        "summary": summary,
        "specialties": [],
        "experience": experiences,
        "education": education,
        "skills": skills,
        "publications": publications,
    }
    logger.info(
        "Imported from %s: experiences=%s achievements=%s skills=%s publications=%s",
        path,
        len(experiences),
        sum(len(e["achievements"]) for e in experiences),
        sum(len(v) for v in skills.values()),
        len(publications),
    )
    return ImportResult(data=data, source_path=path, warnings=warnings)


def _strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        out: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            out.append(ch)
            i += 1
        lines.append("".join(out))
    return "\n".join(lines)


def _parse_personal(source: str, warnings: list[str]) -> dict[str, Any]:
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

    name_match = re.search(
        r"\\name\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        source,
    )
    if name_match:
        personal["name"] = _clean_text(f"{name_match.group(1)} {name_match.group(2)}")

    title_match = re.search(r"\\title\s*\{([^{}]*)\}", source)
    if title_match:
        personal["headline"] = _clean_text(title_match.group(1))

    address_match = re.search(
        r"\\address\s*\{([^{}]*)\}\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        source,
    )
    if address_match:
        city = _clean_text(address_match.group(1))
        region = _clean_text(address_match.group(2))
        # moderncv: {street/commune}{city, country}{optional}
        if "," in region:
            city_part, country_part = [p.strip() for p in region.split(",", 1)]
            personal["city"] = city_part or city
            personal["country"] = country_part
        else:
            personal["city"] = city or region
            if region and region != city:
                personal["country"] = region

    email_match = re.search(r"\\email\s*\{([^{}]*)\}", source)
    if email_match:
        personal["email"] = _clean_text(email_match.group(1))

    phone_match = re.search(r"\\phone(?:\[[^\]]*\])?\s*\{([^{}]*)\}", source)
    if phone_match:
        personal["phone"] = _clean_text(phone_match.group(1).replace(r"\,", " "))

    for kind, url_prefix in (
        ("linkedin", "https://www.linkedin.com/in/"),
        ("github", "https://github.com/"),
    ):
        social = re.search(
            rf"\\social\[{kind}\]\s*\{{([^{{}}]*)\}}",
            source,
            flags=re.IGNORECASE,
        )
        if social:
            handle = _clean_text(social.group(1)).lstrip("@")
            if handle.startswith("http"):
                personal[kind] = handle
            else:
                personal[kind] = f"{url_prefix}{handle}"

    # Drop empty optional keys later is fine; keep None for required review.
    if not personal["headline"]:
        personal["headline"] = "Professional"
        warnings.append("Missing \\title; used placeholder headline")
    if not personal["name"]:
        personal["name"] = "Unknown"
    return personal


def _split_sections(source: str) -> list[tuple[str, str]]:
    """Return (section_title, body) pairs after \\begin{document}."""
    doc_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", source, re.DOTALL)
    body = doc_match.group(1) if doc_match else source
    parts = re.split(r"\\section\*?\{([^{}]*)\}", body)
    # parts[0] = preamble before first section; then title, body, title, body...
    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        title = _clean_text(parts[i])
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((title, content))
    return sections


def _parse_summary(sections: list[tuple[str, str]]) -> str | None:
    for title, body in sections:
        if section_kind(title) == "summary":
            text = _clean_text(_strip_latex_commands(body))
            return text or None
    return None


def _parse_experiences(
    sections: list[tuple[str, str]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    experiences: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for title, body in sections:
        if section_kind(title) != "experience":
            continue
        for entry in _iter_cventries(body):
            exp = _cventry_to_experience(entry, seen_ids, warnings)
            if exp:
                experiences.append(exp)
    return experiences


def _parse_education(
    sections: list[tuple[str, str]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    education: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for title, body in sections:
        if section_kind(title) != "education":
            continue
        for entry in _iter_cventries(body):
            dates = _parse_date_range(entry.dates)
            degree = _clean_text(entry.position)
            institution = _clean_text(entry.company)
            details = _clean_text(entry.description) or None
            edu_id = unique_id(slugify(f"{institution}-{degree}") or "education", seen_ids)
            item: dict[str, Any] = {
                "id": edu_id,
                "institution": institution or "Unknown",
                "degree": degree or "Unknown",
            }
            if dates["start_date"]:
                item["start_date"] = dates["start_date"]
            if dates["end_date"]:
                item["end_date"] = dates["end_date"]
            elif dates["current"]:
                # ongoing degree: leave end_date unset
                pass
            if details:
                item["details"] = details
            if not dates["start_date"] and not dates["end_date"]:
                warnings.append(f"Education {edu_id}: could not parse dates '{entry.dates}'")
            education.append(item)
    return education


def _parse_skills(
    sections: list[tuple[str, str]],
    warnings: list[str],
) -> dict[str, list[str]]:
    skills: dict[str, list[str]] = empty_skills()
    for title, body in sections:
        if section_kind(title) != "skills":
            continue
        for label, value in _iter_cvitems(body):
            group = map_skill_group(label)
            items = [_clean_text(x) for x in re.split(r",|;", value)]
            items = [i for i in items if i]
            if group not in skills:
                skills[group] = []
            for item in items:
                if item not in skills[group]:
                    skills[group].append(item)
    if not any(skills.values()):
        warnings.append("No skills detected from \\cvitem entries")
    return skills


def _parse_publications(
    sections: list[tuple[str, str]],
    _warnings: list[str],
) -> list[dict[str, Any]]:
    pubs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for title, body in sections:
        if section_kind(title) != "publications":
            continue
        for year_raw, value in _iter_cvitems(body):
            year = None
            year_match = re.search(r"(19|20)\d{2}", year_raw)
            if year_match:
                year = int(year_match.group(0))
            cleaned = _clean_text(value)
            # Prefer emphasized title if present
            title_match = re.search(r"\\emph\{([^{}]*)\}", value)
            pub_title = _clean_text(title_match.group(1)) if title_match else cleaned
            journal = None
            journal_match = re.search(r"\\textbf\{([^{}]*)\}", value)
            if journal_match:
                journal = _clean_text(journal_match.group(1))
            pub_id = unique_id(slugify(pub_title) or f"pub-{year or 'x'}", seen_ids)
            item: dict[str, Any] = {"id": pub_id, "title": pub_title}
            if journal:
                item["journal"] = journal
            if year:
                item["year"] = year
            doi_match = re.search(
                r"(?:doi[:\s]*|https?://doi\.org/)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
                cleaned,
                re.IGNORECASE,
            )
            if doi_match:
                item["doi"] = doi_match.group(1).rstrip(".")
            pubs.append(item)
    return pubs


@dataclass
class _CVEntry:
    dates: str
    position: str
    company: str
    location: str
    description: str
    bullets: list[str]


def _iter_cventries(body: str) -> list[_CVEntry]:
    entries: list[_CVEntry] = []
    pattern = re.compile(r"\\cventry\s*")
    pos = 0
    while True:
        match = pattern.search(body, pos)
        if not match:
            break
        args, end = _read_brace_args(body, match.end(), expected=6)
        if len(args) < 6:
            pos = match.end()
            continue
        after = body[end:]
        bullets, consumed = _read_following_itemize(after)
        description = args[5]
        if not bullets and description.strip():
            # description-only cventry (common in academic roles)
            pass
        entries.append(
            _CVEntry(
                dates=args[0],
                position=args[1],
                company=args[2],
                location=args[3],
                description=description,
                bullets=bullets,
            )
        )
        pos = end + consumed
    return entries


def _iter_cvitems(body: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    pattern = re.compile(r"\\cvitem\s*")
    pos = 0
    while True:
        match = pattern.search(body, pos)
        if not match:
            break
        args, end = _read_brace_args(body, match.end(), expected=2)
        if len(args) >= 2:
            items.append((_clean_text(args[0]), args[1]))
        pos = end
    return items


def _read_brace_args(text: str, start: int, expected: int) -> tuple[list[str], int]:
    """Read consecutive {...} groups starting at start; skip whitespace between."""
    args: list[str] = []
    i = start
    while len(args) < expected and i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        content, i = _read_balanced_brace(text, i)
        args.append(content)
    return args, i


def _read_balanced_brace(text: str, start: int) -> tuple[str, int]:
    """start points at '{'; return inner content and index after closing '}'."""
    assert text[start] == "{"
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], len(text)


def _read_following_itemize(text: str) -> tuple[list[str], int]:
    """Parse an optional itemize block immediately after a cventry."""
    match = re.match(r"\s*\\begin\{itemize\}", text)
    if not match:
        return [], 0
    start = match.end()
    inner_end, consumed_end = _matching_itemize_end(text, start)
    if inner_end is None or consumed_end is None:
        return [], 0
    inner = text[start:inner_end]
    # Flatten nested itemize into parent bullets by stripping nested environments
    inner = re.sub(
        r"\\begin\{itemize\}.*?\\end\{itemize\}",
        "",
        inner,
        flags=re.DOTALL,
    )
    bullets: list[str] = []
    for item_match in re.finditer(r"\\item\b", inner):
        item_start = item_match.end()
        next_item = re.search(r"\\item\b", inner[item_start:])
        if next_item:
            chunk = inner[item_start : item_start + next_item.start()]
        else:
            chunk = inner[item_start:]
        cleaned = _clean_text(_strip_latex_commands(chunk))
        if cleaned:
            bullets.append(cleaned)
    return bullets, consumed_end


def _matching_itemize_end(text: str, start: int) -> tuple[int | None, int | None]:
    """Where the list opened at `start` really closes, counting nested lists.

    Taking the first `\\end{itemize}` closes an inner list instead, which cuts the
    block in half and leaves an orphan `\\begin` inside it.
    """
    depth = 1
    for token in re.finditer(r"\\(begin|end)\{itemize\}", text[start:]):
        depth += 1 if token.group(1) == "begin" else -1
        if depth == 0:
            return start + token.start(), start + token.end()
    return None, None


def _cventry_to_experience(
    entry: _CVEntry,
    seen_ids: set[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    company = _clean_text(entry.company) or "Unknown"
    title = _clean_text(entry.position) or "Unknown"
    location = _clean_text(entry.location) or None
    dates = _parse_date_range(entry.dates)
    exp_id = unique_id(slugify(f"{company}-{title}") or "experience", seen_ids)

    if not dates["start_date"]:
        warnings.append(f"{exp_id}: could not parse dates '{entry.dates}'")
        year_match = re.search(r"(19|20)\d{2}", entry.dates)
        if year_match:
            dates["start_date"] = f"{year_match.group(0)}-01"
        else:
            dates["start_date"] = "2000-01"
            dates["current"] = True
            warnings.append(f"{exp_id}: used placeholder start_date 2000-01")

    description = _clean_text(entry.description) or None
    achievements: list[dict[str, Any]] = []
    ach_seen: set[str] = set()
    for idx, bullet in enumerate(entry.bullets, start=1):
        ach_id = unique_id(slugify(f"{exp_id}-{bullet[:40]}") or f"{exp_id}-a{idx}", ach_seen)
        achievements.append({"id": ach_id, "text": bullet, "tags": [], "metrics": {}})

    if description and not achievements:
        # Keep narrative roles with description only
        pass

    result: dict[str, Any] = {
        "id": exp_id,
        "company": company,
        "title": title,
        "location": location,
        "start_date": dates["start_date"],
        "current": dates["current"],
        "description": description,
        "achievements": achievements,
    }
    if dates["end_date"]:
        result["end_date"] = dates["end_date"]
    elif not dates["current"]:
        result["end_date"] = dates["start_date"]
        warnings.append(f"{exp_id}: missing end_date; mirrored start_date")
    return result


def _parse_date_range(raw: str) -> dict[str, Any]:
    return parse_date_range(_clean_text(raw))


def _strip_latex_commands(text: str) -> str:
    out = text
    # Replace common commands with their args
    for _ in range(8):
        new = re.sub(r"\\(textbf|textit|emph|underline|href)\s*\{([^{}]*)\}", r"\2", out)
        new = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}", r"\1", new)
        if new == out:
            break
        out = new
    out = out.replace(r"\&", "&").replace(r"\%", "%").replace(r"\_", "_")
    out = out.replace(r"\,", " ").replace("~", " ").replace(r"\\", " ")
    out = re.sub(r"[{}]", "", out)
    # Orphan environment names must never reach profile.yaml / the PDF.
    out = re.sub(r"\b(itemize|enumerate|description)\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _clean_text(text: str) -> str:
    return _strip_latex_commands(text)
