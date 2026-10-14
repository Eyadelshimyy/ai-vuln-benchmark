"""Typed contract evaluated candidates must satisfy.

The field layout mirrors the client field-note template exactly. The checklist and
turn decision layer expresses the ported prototype heuristics: infer before asking,
never chase optional fields, allow a hard contradiction to raise a question, and ask
exactly one question at a time.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Finding(BaseModel):
    """One observed material/condition observation."""

    model_config = ConfigDict(extra="forbid")

    location_area: str | None = None
    material_condition: str | None = None
    suspect_status: str | None = None
    condition: str | None = None
    notes: str | None = None


class Sample(BaseModel):
    """One physical sample collected on site."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str | None = None
    location: str | None = None
    material_type: str | None = None
    sent_to_lab: str | None = None


class NoteFields(BaseModel):
    """Field-note state carried across turns of one conversation."""

    model_config = ConfigDict(extra="forbid")

    job_address: str | None = None
    job_number: str | None = None
    client: str | None = None
    date: str | None = None
    technician: str | None = None
    inspection_type: str | None = None

    building_type_age: str | None = None
    areas_accessed: str | None = None
    areas_not_accessed: str | None = None
    general_condition_notes: str | None = None

    findings: list[Finding] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)
    photos: list[str] = Field(default_factory=list)

    missing_info: str | None = None
    recommended_next_steps: str | None = None
    flags_for_office_review: str | None = None

    technician_summary: str | None = None


CRITICAL_FIELDS: tuple[str, ...] = (
    "job_address",
    "date",
    "technician",
    "inspection_type",
)

SCALAR_FIELDS: tuple[str, ...] = (
    "job_address",
    "job_number",
    "client",
    "date",
    "technician",
    "inspection_type",
    "building_type_age",
    "areas_accessed",
    "areas_not_accessed",
    "general_condition_notes",
    "missing_info",
    "recommended_next_steps",
    "flags_for_office_review",
    "technician_summary",
)

OPTIONAL_FIELDS: tuple[str, ...] = tuple(
    name for name in SCALAR_FIELDS if name not in CRITICAL_FIELDS
) + ("samples", "photos")


class ChecklistState(StrEnum):
    """Completeness state of a single checklist item."""

    SATISFIED = "satisfied"
    INFERRED = "inferred"
    MISSING = "missing"


class ChecklistEntry(BaseModel):
    """Per-item completeness verdict with the evidence that supports it."""

    model_config = ConfigDict(extra="forbid")

    item: str
    state: ChecklistState
    evidence: str | None = None


class TurnStatus(StrEnum):
    """Whether the note can be handed to review after this turn."""

    NEED_MORE_INFO = "need_more_info"
    READY_FOR_REVIEW = "ready_for_review"


class FollowUp(BaseModel):
    """The single question asked when something blocks review."""

    model_config = ConfigDict(extra="forbid")

    target: str
    question: str


class TurnResult(BaseModel):
    """Everything a candidate must return for one technician turn."""

    model_config = ConfigDict(extra="forbid")

    fields: NoteFields
    checklist: list[ChecklistEntry] = Field(default_factory=list)
    status: TurnStatus
    follow_up: FollowUp | None = None
    contradiction: str | None = None


def empty_fields() -> NoteFields:
    """Return a fully empty field state."""
    return NoteFields()


def missing_critical_fields(fields: NoteFields) -> list[str]:
    """Return critical field names that are still empty."""
    missing: list[str] = []
    for name in CRITICAL_FIELDS:
        value = getattr(fields, name)
        if value is None or not str(value).strip():
            missing.append(name)
    return missing
