"""Construction of the ideal turn result implied by a case expectation."""

from __future__ import annotations

from field_notes.cases import EvalCase, TurnExpectation
from field_notes.schema import (
    ChecklistEntry,
    FollowUp,
    NoteFields,
    TurnResult,
)


def gold_fields(expectation: TurnExpectation) -> NoteFields:
    """Return the field state a perfect candidate would hold after this turn.

    Fields that are only judged by permissible inference (because their exact wording
    is uncertain, e.g. a garbled proper noun) are not asserted in ``fields``, so the
    first allowed value stands in for them here.
    """
    payload: dict[str, object] = dict(expectation.fields)
    for inference in expectation.permissible_inferences:
        payload.setdefault(inference.field_name, inference.allowed_values[0])
    payload["findings"] = [finding.model_copy() for finding in expectation.findings]
    payload["samples"] = [sample.model_copy() for sample in expectation.samples]
    payload["photos"] = list(expectation.photos)
    return NoteFields.model_validate(payload)


def gold_question(expectation: TurnExpectation) -> str:
    """Return a single-question phrasing for the expected follow-up target."""
    topic = expectation.follow_up.topic or expectation.follow_up.target
    return f"Quick one before I write this up — what's the {topic}?"


def gold_result(expectation: TurnExpectation) -> TurnResult:
    """Return the reference result used by the oracle adapter and scoring tests."""
    follow_up: FollowUp | None = None
    if expectation.follow_up.needed and expectation.follow_up.target is not None:
        follow_up = FollowUp(
            target=expectation.follow_up.target,
            question=gold_question(expectation),
        )
    checklist = [
        ChecklistEntry(
            item=item.item,
            state=item.state,
            evidence=item.evidence_contains,
        )
        for item in expectation.checklist
    ]
    return TurnResult(
        fields=gold_fields(expectation),
        checklist=checklist,
        status=expectation.status,
        follow_up=follow_up,
        contradiction=expectation.contradiction,
    )


def gold_index(cases: list[EvalCase]) -> dict[tuple[str, int], TurnExpectation]:
    """Index every turn expectation by ``(case_id, turn_index)``."""
    return {(case.case_id, turn.index): turn.expect for case in cases for turn in case.turns}
