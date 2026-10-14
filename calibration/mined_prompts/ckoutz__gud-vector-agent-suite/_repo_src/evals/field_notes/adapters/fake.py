"""Deterministic local adapters.

``OracleAdapter`` replays the case gold, which pins every metric at its ceiling and
proves the harness wiring. ``DegradedAdapter`` injects named, deterministic faults so
each metric can be shown to actually penalize the behavior it claims to measure.
No network access, no provider SDKs, no credentials.
"""

from __future__ import annotations

from enum import StrEnum

from field_notes.adapters.base import TurnOutcome, TurnRequest
from field_notes.cases import EvalCase, TurnExpectation
from field_notes.gold import gold_index, gold_result
from field_notes.schema import ChecklistState, FollowUp, TurnResult, TurnStatus


class Fault(StrEnum):
    """A deterministic failure mode a degraded adapter can exhibit."""

    DROP_EVIDENCE = "drop_evidence"
    INVENT_JOB_NUMBER = "invent_job_number"
    FORGET_PRIOR_STATE = "forget_prior_state"
    ASK_ABOUT_OPTIONAL = "ask_about_optional"
    ASK_MULTIPLE_QUESTIONS = "ask_multiple_questions"
    MISS_CONTRADICTION = "miss_contradiction"
    MISS_CRITICAL_GAP = "miss_critical_gap"
    DROP_LAST_FINDING = "drop_last_finding"
    INVALID_SCHEMA = "invalid_schema"


class _GoldBackedAdapter:
    """Shared lookup of the expectation for a requested turn."""

    def __init__(self, cases: list[EvalCase], name: str) -> None:
        self._gold = gold_index(cases)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _expectation(self, request: TurnRequest) -> TurnExpectation:
        key = (request.case_id, request.turn_index)
        expectation = self._gold.get(key)
        if expectation is None:
            raise KeyError(f"no expectation loaded for {key}")
        return expectation


class OracleAdapter(_GoldBackedAdapter):
    """Returns exactly the expected result for every turn."""

    def __init__(self, cases: list[EvalCase], name: str = "fake-oracle") -> None:
        super().__init__(cases, name)

    def run_turn(self, request: TurnRequest) -> TurnOutcome:
        result = gold_result(self._expectation(request))
        return TurnOutcome(result=result, schema_valid=True, latency_ms=None, cost_usd=None)


class DegradedAdapter(_GoldBackedAdapter):
    """Returns the expected result with the configured faults applied."""

    def __init__(
        self,
        cases: list[EvalCase],
        faults: list[Fault],
        name: str = "fake-degraded",
    ) -> None:
        super().__init__(cases, name)
        self._faults = tuple(faults)

    @property
    def faults(self) -> tuple[Fault, ...]:
        return self._faults

    def run_turn(self, request: TurnRequest) -> TurnOutcome:
        expectation = self._expectation(request)
        if Fault.INVALID_SCHEMA in self._faults:
            return TurnOutcome(
                result=None,
                schema_valid=False,
                error="adapter returned a payload that failed schema validation",
                raw_response='{"fields": "not-an-object"}',
            )
        result = self._mutate(gold_result(expectation), expectation, request)
        return TurnOutcome(result=result, schema_valid=True)

    def _mutate(
        self,
        result: TurnResult,
        expectation: TurnExpectation,
        request: TurnRequest,
    ) -> TurnResult:
        mutated = result.model_copy(deep=True)
        if Fault.DROP_EVIDENCE in self._faults:
            for entry in mutated.checklist:
                entry.evidence = None
        if Fault.INVENT_JOB_NUMBER in self._faults and mutated.fields.job_number is None:
            mutated.fields.job_number = "J-00000"
        if Fault.FORGET_PRIOR_STATE in self._faults:
            for name in expectation.preserved_fields:
                setattr(mutated.fields, name, None)
        if Fault.DROP_LAST_FINDING in self._faults and mutated.fields.findings:
            mutated.fields.findings = mutated.fields.findings[:-1]
        if Fault.MISS_CRITICAL_GAP in self._faults and expectation.critical_gaps:
            mutated.status = TurnStatus.READY_FOR_REVIEW
            mutated.follow_up = None
            for entry in mutated.checklist:
                if entry.item in expectation.critical_gaps:
                    entry.state = ChecklistState.SATISFIED
        if Fault.MISS_CONTRADICTION in self._faults and expectation.contradiction is not None:
            mutated.contradiction = None
            mutated.status = TurnStatus.READY_FOR_REVIEW
            mutated.follow_up = None
        if Fault.ASK_ABOUT_OPTIONAL in self._faults and not expectation.follow_up.needed:
            mutated.status = TurnStatus.NEED_MORE_INFO
            mutated.follow_up = FollowUp(
                target="samples",
                question="Did you end up pulling any samples on this one?",
            )
        if Fault.ASK_MULTIPLE_QUESTIONS in self._faults and mutated.follow_up is not None:
            mutated.follow_up = FollowUp(
                target=mutated.follow_up.target,
                question=(
                    f"{mutated.follow_up.question} Also, who was the client contact on site? "
                    f"And did turn {request.turn_index} cover the whole basement?"
                ),
            )
        return mutated
