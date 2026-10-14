"""Case file models and loader.

Cases are YAML so they stay reviewable by domain reviewers. Every turn declares the
full expectation set: gold fields, checklist evidence, permissible inferences,
forbidden facts, critical gaps, preserved prior state, contradiction topic, and
whether exactly one follow-up is warranted (and about what).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from field_notes.schema import (
    SCALAR_FIELDS,
    ChecklistState,
    Finding,
    Sample,
    TurnStatus,
)

CASES_ROOT = Path(__file__).parent / "cases"


class Split(StrEnum):
    """Prompt-development cases versus the held-out final-selection subset."""

    DEV = "dev"
    HOLDOUT = "holdout"


class Category(StrEnum):
    """Behavioral bucket a case exercises."""

    CLEAN_COMPLETE = "clean_complete"
    OPTIONAL_ABSENT = "optional_absent"
    MISSING_CRITICAL = "missing_critical"
    MULTIPLE_FINDINGS = "multiple_findings"
    CORRECTION = "correction"
    CONTRADICTION = "contradiction"
    TRANSCRIPTION_ERROR = "transcription_error"
    FOLLOW_UP_REPLY = "follow_up_reply"
    PREMATURE_FINALIZATION = "premature_finalization"


class PermissibleInference(BaseModel):
    """A value the candidate may fill in without being told it explicitly."""

    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(alias="field")
    allowed_values: list[str]
    rationale: str

    @model_validator(mode="after")
    def _validate_field_name(self) -> PermissibleInference:
        if self.field_name not in SCALAR_FIELDS:
            raise ValueError(f"inferences are only scored for scalar fields: {self.field_name}")
        if not self.allowed_values:
            raise ValueError(f"{self.field_name}: allowed_values must not be empty")
        return self


CONTAINER_FIELDS: tuple[str, ...] = ("findings", "samples", "photos")


class ForbiddenFact(BaseModel):
    """A value the candidate must never produce for this turn."""

    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(alias="field")
    reason: str
    forbidden_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_field_name(self) -> ForbiddenFact:
        if self.field_name not in SCALAR_FIELDS + CONTAINER_FIELDS:
            raise ValueError(f"unknown field name in forbidden fact: {self.field_name}")
        return self


class ChecklistExpectation(BaseModel):
    """Expected completeness verdict and supporting evidence for one item."""

    model_config = ConfigDict(extra="forbid")

    item: str
    state: ChecklistState
    evidence_contains: str | None = None


class FollowUpExpectation(BaseModel):
    """Whether one question is warranted, and its single target."""

    model_config = ConfigDict(extra="forbid")

    needed: bool
    target: str | None = None
    topic: str | None = None

    @model_validator(mode="after")
    def _target_required_when_needed(self) -> FollowUpExpectation:
        if self.needed and not self.target:
            raise ValueError("follow_up.target is required when follow_up.needed is true")
        if not self.needed and self.target:
            raise ValueError("follow_up.target must be omitted when follow_up.needed is false")
        return self


class TurnExpectation(BaseModel):
    """Full gold expectation for a single technician turn."""

    model_config = ConfigDict(extra="forbid")

    fields: dict[str, str] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)
    photos: list[str] = Field(default_factory=list)
    tolerated_fields: list[str] = Field(default_factory=list)
    checklist: list[ChecklistExpectation] = Field(default_factory=list)
    permissible_inferences: list[PermissibleInference] = Field(default_factory=list)
    forbidden_facts: list[ForbiddenFact] = Field(default_factory=list)
    critical_gaps: list[str] = Field(default_factory=list)
    preserved_fields: list[str] = Field(default_factory=list)
    contradiction: str | None = None
    follow_up: FollowUpExpectation
    status: TurnStatus
    premature_finalization: bool = False

    @model_validator(mode="after")
    def _validate_names(self) -> TurnExpectation:
        unknown = sorted(set(self.fields) - set(SCALAR_FIELDS))
        if unknown:
            raise ValueError(f"unknown scalar field names in expectation: {unknown}")
        unknown_tolerated = sorted(set(self.tolerated_fields) - set(SCALAR_FIELDS))
        if unknown_tolerated:
            raise ValueError(f"unknown tolerated field names: {unknown_tolerated}")
        overlap = sorted(set(self.tolerated_fields) & set(self.fields))
        if overlap:
            raise ValueError(f"tolerated fields must not also be asserted: {overlap}")
        filled = {name for name, value in self.fields.items() if value.strip()}
        declared_gaps = set(self.critical_gaps)
        if declared_gaps & filled:
            raise ValueError(
                f"critical gaps cannot also be filled: {sorted(declared_gaps & filled)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_decision(self) -> TurnExpectation:
        blocked = bool(self.critical_gaps) or self.contradiction is not None
        if blocked and self.status is not TurnStatus.NEED_MORE_INFO:
            raise ValueError("a critical gap or contradiction requires need_more_info")
        if blocked != self.follow_up.needed:
            raise ValueError("follow_up.needed must match the presence of a blocking issue")
        if self.critical_gaps and self.follow_up.target not in self.critical_gaps:
            raise ValueError("critical-field gaps take priority for the follow-up target")
        return self


class CaseTurn(BaseModel):
    """One technician utterance and the expectation it produces."""

    model_config = ConfigDict(extra="forbid")

    index: int
    speaker_note: str | None = None
    transcript: str
    expect: TurnExpectation


class EvalCase(BaseModel):
    """A multi-turn synthetic conversation with per-turn expectations."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: Split
    category: Category
    tags: list[str] = Field(default_factory=list)
    description: str
    turns: list[CaseTurn]

    @model_validator(mode="after")
    def _validate_turns(self) -> EvalCase:
        if len(self.turns) < 2:
            raise ValueError(f"{self.case_id}: cases must be multi-turn")
        expected = list(range(1, len(self.turns) + 1))
        if [turn.index for turn in self.turns] != expected:
            raise ValueError(f"{self.case_id}: turn indexes must be 1..n in order")
        return self


class CaseFile(BaseModel):
    """One YAML file holding a group of related cases."""

    model_config = ConfigDict(extra="forbid")

    cases: list[EvalCase]


def parse_case_file(payload: dict[str, Any]) -> CaseFile:
    """Validate a raw parsed YAML mapping as a case file."""
    return CaseFile.model_validate(payload)


def load_case_files(root: Path | None = None) -> dict[Path, CaseFile]:
    """Load and validate every case file under ``root``."""
    base = root or CASES_ROOT
    loaded: dict[Path, CaseFile] = {}
    for path in sorted(base.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: case file must be a mapping with a 'cases' key")
        case_file = parse_case_file(raw)
        expected_split = Split(path.parent.name)
        for case in case_file.cases:
            if case.split is not expected_split:
                raise ValueError(f"{path}: case {case.case_id} declares split {case.split}")
        loaded[path] = case_file
    return loaded


def load_cases(root: Path | None = None, splits: Iterable[Split] | None = None) -> list[EvalCase]:
    """Load all cases, optionally filtered to specific splits, with unique ids."""
    wanted = set(splits) if splits is not None else set(Split)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path, case_file in load_case_files(root).items():
        for case in case_file.cases:
            if case.case_id in seen:
                raise ValueError(f"{path}: duplicate case id {case.case_id}")
            seen.add(case.case_id)
            if case.split in wanted:
                cases.append(case)
    return sorted(cases, key=lambda case: case.case_id)
