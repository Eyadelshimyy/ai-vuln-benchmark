"""Drives cases through an adapter and records what came back.

The runner is deliberately free of scoring logic: it produces records, and
``field_notes.scoring`` turns records into metrics. That split lets a future real run
persist its raw records and be re-scored without re-spending tokens.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from field_notes.adapters.base import (
    ExtractionAdapter,
    HistoryTurn,
    TurnOutcome,
    TurnRequest,
)
from field_notes.cases import Category, EvalCase, Split, TurnExpectation
from field_notes.gold import gold_fields
from field_notes.schema import NoteFields, empty_fields


class TurnRecord(BaseModel):
    """One evaluated turn: what was asked, what was expected, what came back."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: Split
    category: Category
    turn_index: int
    transcript: str
    prior_fields: NoteFields
    expectation: TurnExpectation
    outcome: TurnOutcome


class RunRecord(BaseModel):
    """All turn records produced by one candidate over one case selection."""

    model_config = ConfigDict(extra="forbid")

    candidate: str
    splits: list[Split]
    turns: list[TurnRecord] = Field(default_factory=list)


def run_case(adapter: ExtractionAdapter, case: EvalCase) -> list[TurnRecord]:
    """Run every turn of one case, threading state forward across turns.

    Prior state for turn *n* is the candidate's own turn *n-1* field state, so
    dropped facts propagate the way they would in production. When a turn produces
    no schema-valid result, state falls back to that turn's gold so the rest of the
    case stays scoreable instead of collapsing into cascade noise.
    """
    records: list[TurnRecord] = []
    prior: NoteFields = empty_fields()
    history: list[HistoryTurn] = []
    for turn in case.turns:
        request = TurnRequest(
            case_id=case.case_id,
            turn_index=turn.index,
            prior_fields=prior.model_copy(deep=True),
            history=list(history),
            transcript=turn.transcript,
        )
        outcome = adapter.run_turn(request)
        records.append(
            TurnRecord(
                case_id=case.case_id,
                split=case.split,
                category=case.category,
                turn_index=turn.index,
                transcript=turn.transcript,
                prior_fields=request.prior_fields,
                expectation=turn.expect,
                outcome=outcome,
            )
        )
        history.append(HistoryTurn(role="technician", text=turn.transcript))
        if outcome.result is None:
            prior = gold_fields(turn.expect)
            continue
        prior = outcome.result.fields.model_copy(deep=True)
        if outcome.result.follow_up is not None:
            history.append(HistoryTurn(role="assistant", text=outcome.result.follow_up.question))
    return records


def run_suite(
    adapter: ExtractionAdapter,
    cases: list[EvalCase],
    splits: list[Split] | None = None,
) -> RunRecord:
    """Run every case in ``cases`` through ``adapter``."""
    selected = splits or sorted({case.split for case in cases})
    turns: list[TurnRecord] = []
    for case in cases:
        turns.extend(run_case(adapter, case))
    return RunRecord(candidate=adapter.name, splits=list(selected), turns=turns)
