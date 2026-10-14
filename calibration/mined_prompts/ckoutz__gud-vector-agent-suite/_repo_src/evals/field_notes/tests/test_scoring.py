"""Scoring behaviour: every metric must actually punish the failure it names."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from field_notes.adapters.fake import DegradedAdapter, Fault, OracleAdapter
from field_notes.cases import Split, load_cases
from field_notes.runner import run_suite
from field_notes.schema import Finding, NoteFields
from field_notes.scoring import Scorecard, format_scorecard, normalize, score_run, values_match

CASES = load_cases()


def _score(*faults: Fault, name: str = "candidate") -> Scorecard:
    adapter = DegradedAdapter(CASES, faults=list(faults), name=name)
    return score_run(run_suite(adapter, CASES))


@pytest.fixture(scope="module")
def oracle() -> Scorecard:
    return score_run(run_suite(OracleAdapter(CASES), CASES))


def test_normalize_folds_case_punctuation_and_whitespace() -> None:
    assert normalize("  1420   Kenmore Ave.,  Building B ") == "1420 kenmore ave building b"
    assert normalize(None) == ""


def test_values_match_allows_a_more_specific_prediction() -> None:
    assert values_match("PT-4471", "pt 4471")
    assert values_match("Kitchen", "Kitchen ceiling")
    assert not values_match("Suspect", "Not Suspect")
    assert not values_match("March 4, 2025", None)


def test_oracle_scores_every_metric_at_ceiling(oracle: Scorecard) -> None:
    assert oracle.violations == []
    assert oracle.field_extraction.precision == 1.0
    assert oracle.field_extraction.recall == 1.0
    assert oracle.schema_valid_rate.rate == 1.0
    assert oracle.unsupported_fact_rate.rate == 0.0
    assert oracle.critical_gap_detection.recall == 1.0
    assert oracle.critical_gap_detection.precision == 1.0
    assert oracle.false_follow_up_rate.rate == 0.0
    assert oracle.one_question_target_accuracy.rate == 1.0
    assert oracle.prior_state_preservation.rate == 1.0
    assert oracle.contradiction_detection.recall == 1.0
    assert oracle.evidence_preservation.rate == 1.0
    assert oracle.inference_compliance.rate == 1.0
    assert oracle.premature_finalization_resistance.rate == 1.0


def test_oracle_covers_every_case_and_turn(oracle: Scorecard) -> None:
    assert oracle.case_count == len(CASES)
    assert oracle.turn_count == sum(len(case.turns) for case in CASES)


def test_operational_slots_are_empty_for_local_adapters(oracle: Scorecard) -> None:
    assert oracle.operational.latency_p50_ms is None
    assert oracle.operational.latency_p95_ms is None
    assert oracle.operational.cost_usd_per_case is None


def test_per_category_scores_cover_every_category_present(oracle: Scorecard) -> None:
    assert set(oracle.by_category) == {case.category for case in CASES}
    assert all(score.turns > 0 for score in oracle.by_category.values())


def test_dropping_prior_state_hurts_recall_and_preservation() -> None:
    card = _score(Fault.FORGET_PRIOR_STATE)
    assert card.prior_state_preservation.rate == 0.0
    assert card.field_extraction.recall is not None
    assert card.field_extraction.recall < 0.8
    assert any(v.metric == "prior_state_preservation" for v in card.violations)


def test_dropping_the_newest_finding_hurts_recall_but_not_precision() -> None:
    card = _score(Fault.DROP_LAST_FINDING)
    assert card.field_extraction.precision == 1.0
    assert card.field_extraction.recall is not None
    assert card.field_extraction.recall < 1.0


def test_inventing_a_job_number_is_an_unsupported_fact() -> None:
    card = _score(Fault.INVENT_JOB_NUMBER)
    assert card.unsupported_fact_rate.rate is not None
    assert card.unsupported_fact_rate.rate > 0.0
    assert any(v.metric == "unsupported_fact_rate" for v in card.violations)


def _score_oracle_with_mutation(
    case_id: str, turn_index: int, mutate: Callable[[NoteFields], None]
) -> Scorecard:
    """Score a perfect run whose one named turn had a fact grafted onto it."""
    record = run_suite(OracleAdapter(CASES), CASES)
    for turn in record.turns:
        if turn.case_id == case_id and turn.turn_index == turn_index:
            assert turn.outcome.result is not None
            mutate(turn.outcome.result.fields)
            return score_run(record)
    raise AssertionError(f"no turn {turn_index} for {case_id!r}")


def test_an_invented_optional_scalar_is_an_unsupported_fact() -> None:
    def invent_client(fields: NoteFields) -> None:
        fields.client = "Invented Customer"

    card = _score_oracle_with_mutation("fn-dev-contradiction-002", 1, invent_client)
    assert card.unsupported_fact_rate.hits == 1
    assert card.unsupported_fact_rate.rate is not None
    assert card.unsupported_fact_rate.rate > 0.0
    assert card.unsupported_fact_turn_rate.hits == 1
    assert [v.metric for v in card.violations].count("unsupported_fact_rate") == 1


def test_an_invented_container_fact_is_an_unsupported_fact() -> None:
    def invent_finding(fields: NoteFields) -> None:
        fields.findings.append(
            Finding(location_area="Boiler room", material_condition="Pipe lagging, suspect")
        )

    card = _score_oracle_with_mutation("fn-dev-contradiction-002", 1, invent_finding)
    assert card.unsupported_fact_rate.hits == 1
    assert card.unsupported_fact_turn_rate.hits == 1
    assert any(
        v.metric == "unsupported_fact_rate" and "finding" in v.detail for v in card.violations
    )


def test_a_fact_breaking_several_rules_is_counted_once() -> None:
    """A value that both fills a declared critical gap and is unsupported is one fact."""

    def invent_address(fields: NoteFields) -> None:
        fields.job_address = "17 Invented Row"

    card = _score_oracle_with_mutation("fn-dev-missing-002", 1, invent_address)
    assert card.unsupported_fact_rate.hits == 1


def test_asking_about_an_optional_field_is_a_false_follow_up() -> None:
    card = _score(Fault.ASK_ABOUT_OPTIONAL)
    assert card.false_follow_up_rate.rate == 1.0
    assert any(v.metric == "false_follow_up_rate" for v in card.violations)


def test_asking_several_questions_fails_one_question_accuracy() -> None:
    card = _score(Fault.ASK_MULTIPLE_QUESTIONS)
    assert card.one_question_target_accuracy.rate == 0.0
    assert any(v.metric == "one_question_target_accuracy" for v in card.violations)


def test_missing_a_critical_gap_fails_gap_recall_and_decisions() -> None:
    card = _score(Fault.MISS_CRITICAL_GAP)
    assert card.critical_gap_detection.recall == 0.0
    assert card.decision_accuracy.rate is not None
    assert card.decision_accuracy.rate < 1.0
    assert any(v.metric == "critical_gap_detection" for v in card.violations)


def test_missing_a_contradiction_fails_contradiction_recall() -> None:
    card = _score(Fault.MISS_CONTRADICTION)
    assert card.contradiction_detection.recall == 0.0
    assert any(v.metric == "contradiction_detection" for v in card.violations)


def test_dropping_checklist_evidence_fails_evidence_preservation() -> None:
    card = _score(Fault.DROP_EVIDENCE)
    assert card.evidence_preservation.rate == 0.0
    assert any(v.metric == "evidence_preservation" for v in card.violations)


def test_unparseable_output_zeroes_schema_validity_and_voids_the_rest() -> None:
    card = _score(Fault.INVALID_SCHEMA)
    assert card.schema_valid_rate.rate == 0.0
    assert card.field_extraction.precision is None
    assert card.critical_gap_detection.recall is None
    assert card.turn_count == sum(len(case.turns) for case in CASES)
    assert all(v.metric == "schema_valid_rate" for v in card.violations)


def test_violations_identify_the_case_turn_and_metric() -> None:
    card = _score(Fault.MISS_CRITICAL_GAP)
    violation = card.violations[0]
    assert violation.case_id
    assert violation.turn_index >= 1
    assert violation.metric
    assert violation.detail


def test_formatting_a_scorecard_reports_every_metric(oracle: Scorecard) -> None:
    text = format_scorecard(oracle)
    for label in (
        "schema_valid_rate",
        "field_precision",
        "field_recall",
        "unsupported_fact_rate",
        "unsupported_fact_turn_rate",
        "critical_gap_recall",
        "critical_gap_precision",
        "false_follow_up_rate",
        "one_question_target_accuracy",
        "prior_state_preservation",
        "contradiction_recall",
        "evidence_preservation",
        "latency_p50_ms",
        "cost_usd_per_case",
    ):
        assert label in text


def test_holdout_can_be_scored_independently() -> None:
    holdout = load_cases(splits=[Split.HOLDOUT])
    assert holdout
    card = score_run(run_suite(OracleAdapter(holdout), holdout, splits=[Split.HOLDOUT]))
    assert card.splits == [Split.HOLDOUT]
    assert card.violations == []
