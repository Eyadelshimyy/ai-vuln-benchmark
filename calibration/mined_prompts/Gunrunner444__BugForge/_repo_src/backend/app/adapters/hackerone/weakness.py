"""BugForge class → CWE candidates → program-specific numeric HackerOne weakness ids.

CWE identifiers such as ``cwe-89`` are never sent as ``weakness_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters.hackerone.models import HackerOneProgram, WeaknessRecord, parse_hackerone_id

# Candidate *CWE* identifiers used to look up program weaknesses.
_MAP: dict[str, tuple[str, ...]] = {
    "xss": ("cwe-79",),
    "cross_site_scripting": ("cwe-79",),
    "sql_injection": ("cwe-89",),
    "sqli": ("cwe-89",),
    "ssrf": ("cwe-918",),
    "csrf": ("cwe-352",),
    "open_redirect": ("cwe-601",),
    "path_traversal": ("cwe-22",),
    "command_injection": ("cwe-77", "cwe-78"),
    "insecure_direct_object_reference": ("cwe-639", "cwe-284"),
    "idor": ("cwe-639", "cwe-284"),
    "broken_access_control": ("cwe-284", "cwe-285", "cwe-639"),
    "authentication": ("cwe-287",),
    "sensitive_data_exposure": ("cwe-200",),
    "insecure_deserialization": ("cwe-502",),
    "xxe": ("cwe-611",),
    "misconfiguration": ("cwe-16",),
}


@dataclass(frozen=True)
class WeaknessMapping:
    """CWE-only mapping. Prefer :class:`ProgramWeaknessMapping` for report ids."""

    bugforge_class: str
    candidates: tuple[str, ...]
    requires_human_selection: bool

    @property
    def selected(self) -> str | None:
        if self.requires_human_selection or len(self.candidates) != 1:
            return None
        return self.candidates[0]


@dataclass(frozen=True)
class ProgramWeaknessMapping:
    bugforge_class: str
    cwe_candidates: tuple[str, ...]
    matched: tuple[WeaknessRecord, ...]
    selected_id: int | None
    requires_human_selection: bool
    reason: str = ""


def map_weakness(vulnerability_class: str | None) -> WeaknessMapping:
    key = _normalize_class(vulnerability_class)
    candidates = _MAP.get(key, ())
    if not candidates:
        return WeaknessMapping(key, (), requires_human_selection=True)
    if len(candidates) > 1:
        return WeaknessMapping(key, candidates, requires_human_selection=True)
    return WeaknessMapping(key, candidates, requires_human_selection=False)


def map_program_weakness(
    vulnerability_class: str | None,
    program: HackerOneProgram,
    *,
    requested_id: int | str | None = None,
) -> ProgramWeaknessMapping:
    """Resolve a BugForge class to a numeric HackerOne weakness id for this program."""
    cwe = map_weakness(vulnerability_class)
    matched = _match_program_weaknesses(cwe.candidates, program.weaknesses)
    requested = parse_hackerone_id(requested_id)
    if requested_id is not None and str(requested_id).strip() != "":
        if requested is None:
            return ProgramWeaknessMapping(
                bugforge_class=cwe.bugforge_class,
                cwe_candidates=cwe.candidates,
                matched=matched,
                selected_id=None,
                requires_human_selection=True,
                reason="weakness_id must be a numeric HackerOne id, not a CWE string",
            )
        record = next((item for item in program.weaknesses if item.id == requested), None)
        if record is None:
            return ProgramWeaknessMapping(
                bugforge_class=cwe.bugforge_class,
                cwe_candidates=cwe.candidates,
                matched=matched,
                selected_id=None,
                requires_human_selection=True,
                reason="Requested weakness_id is not in the synchronized program weakness list",
            )
        return ProgramWeaknessMapping(
            bugforge_class=cwe.bugforge_class,
            cwe_candidates=cwe.candidates,
            matched=matched or (record,),
            selected_id=record.id,
            requires_human_selection=False,
            reason="Human-selected program weakness",
        )
    if not program.weaknesses:
        return ProgramWeaknessMapping(
            bugforge_class=cwe.bugforge_class,
            cwe_candidates=cwe.candidates,
            matched=(),
            selected_id=None,
            requires_human_selection=True,
            reason="Program weaknesses have not been synchronized",
        )
    if not cwe.candidates:
        return ProgramWeaknessMapping(
            bugforge_class=cwe.bugforge_class,
            cwe_candidates=(),
            matched=(),
            selected_id=None,
            requires_human_selection=True,
            reason="No CWE mapping; a human must select a program weakness",
        )
    if not matched:
        return ProgramWeaknessMapping(
            bugforge_class=cwe.bugforge_class,
            cwe_candidates=cwe.candidates,
            matched=(),
            selected_id=None,
            requires_human_selection=True,
            reason="Program does not contain the expected weakness",
        )
    unique_ids = tuple(dict.fromkeys(item.id for item in matched))
    if len(unique_ids) != 1 or cwe.requires_human_selection:
        return ProgramWeaknessMapping(
            bugforge_class=cwe.bugforge_class,
            cwe_candidates=cwe.candidates,
            matched=matched,
            selected_id=None,
            requires_human_selection=True,
            reason="Multiple matching program weaknesses require human selection",
        )
    return ProgramWeaknessMapping(
        bugforge_class=cwe.bugforge_class,
        cwe_candidates=cwe.candidates,
        matched=matched,
        selected_id=unique_ids[0],
        requires_human_selection=False,
        reason="Unique CWE to program weakness mapping",
    )


def _match_program_weaknesses(
    cwes: tuple[str, ...],
    weaknesses: tuple[WeaknessRecord, ...],
) -> tuple[WeaknessRecord, ...]:
    wanted = {_normalize_cwe(item) for item in cwes}
    hits: list[WeaknessRecord] = []
    for record in weaknesses:
        if _normalize_cwe(record.external_id) in wanted:
            hits.append(record)
    return tuple(hits)


def _normalize_class(vulnerability_class: str | None) -> str:
    return (vulnerability_class or "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_cwe(value: str) -> str:
    text = (value or "").strip().lower().replace("_", "-")
    if text.startswith("cwe") and not text.startswith("cwe-"):
        text = text.replace("cwe", "cwe-", 1)
    return text
