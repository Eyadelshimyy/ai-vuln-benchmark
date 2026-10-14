"""Paired polytomous judge-control contracts."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest
from fast_mlsirm import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    ContextualOrchestratorJudge,
    JudgeCalibrationCase,
    JudgeCalibrationReport,
    JudgeCriterion,
    JudgeFormatError,
    LLMJudgeResult,
    build_multiple_choice_calibration_cases,
    evaluate_paired_calibration,
)

CRITERIA = (
    JudgeCriterion("evidence_quality", "The answer is supported by concrete evidence."),
    JudgeCriterion("risk_awareness", "The answer addresses material failure risks."),
)


def _cases() -> tuple[JudgeCalibrationCase, ...]:
    return build_multiple_choice_calibration_cases(
        case_id="release_case",
        question="Which rollout plan is safest?",
        options=("canary", "big_bang", "rollback_only"),
        answer="canary",
        correct_option_index=0,
        replacement_distractor="unreviewed_rollout",
        reference_answer="canary",
        contamination_status="held_out",
        shuffle_seed=7,
        gold_categories={"evidence_quality": 2, "risk_awareness": 1},
    )


def test_multiple_choice_controls_change_only_declared_presentation_factor() -> None:
    cases = _cases()
    by_variant = {case.variant: case for case in cases}

    assert set(by_variant) == {
        "baseline",
        "option_only",
        "shuffled_options",
        "replaced_distractor",
    }
    assert "Question:" in by_variant["baseline"].task
    assert "Question:" not in by_variant["option_only"].task
    assert by_variant["baseline"].metadata["question_present"] == "true"
    assert by_variant["option_only"].metadata["question_present"] == "false"
    assert by_variant["shuffled_options"].task != by_variant["baseline"].task
    assert "unreviewed_rollout" in by_variant["replaced_distractor"].task
    assert all(case.contamination_status == "held_out" for case in cases)
    assert all(case.option_count == 3 for case in cases)


class _ScriptedJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, **kwargs) -> LLMJudgeResult:
        task = kwargs["task"]
        self.calls.append(task)
        if "unreviewed_rollout" in task:
            raise JudgeFormatError(
                "provider parse failure",
                evidence={
                    "call_count": 1,
                    "answer": "must not be retained",
                    "records": [
                        {"meets_threshold": False, "output_preview": "secret"}
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    },
                },
            )
        option_only = "Question:" not in task
        categories = (
            {"evidence_quality": 1, "risk_awareness": 1}
            if option_only
            else {"evidence_quality": 2, "risk_awareness": 1}
        )
        scores = {key: value / 2 for key, value in categories.items()}
        return LLMJudgeResult(
            score=sum(scores.values()) / len(scores),
            accepted=True,
            rationale="scripted",
            criterion_scores=scores,
            raw_output="not retained in calibration report",
            orchestration_mode="route",
            trace_step_count=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            criterion_categories=categories,
            category_count=3,
            category_method="binary_threshold",
        )


class _RaisingJudge:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def judge(self, **kwargs) -> LLMJudgeResult:
        del kwargs
        raise self.error


class _ConcurrentOrchestrator:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self) -> None:
        self.client = SimpleNamespace(local_concurrency=2)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def complete(self, messages, mode="auto"):
        del messages, mode
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.02)
            return {
                "mode": "route",
                "answer": json.dumps(
                    {
                        "score": 0.75,
                        "accepted": True,
                        "rationale": "scripted",
                        "criterion_categories": {
                            "evidence_quality": 2,
                            "risk_awareness": 1,
                        },
                    }
                ),
                "trace": [
                    {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        }
                    }
                ],
            }
        finally:
            with self._lock:
                self.active -= 1


def test_paired_calibration_preserves_failures_and_reports_gold_and_deltas() -> None:
    judge = _ScriptedJudge()
    report = evaluate_paired_calibration(
        judge,
        _cases(),
        criteria=CRITERIA,
        category_count=3,
    )

    assert report.status_counts() == {"judge_failed": 1, "passed": 3}
    assert len(judge.calls) == 4
    assert all(len(outcome.irt_row or ()) == 2 for outcome in report.outcomes if outcome.status == "passed")
    serialized = report.to_dict()
    assert serialized["case_count"] == 1
    assert serialized["outcome_count"] == 4
    assert serialized["gold_scored_count"] == 3
    assert serialized["gold_exact_agreement"] == pytest.approx(2 / 3)
    assert serialized["category_occupancy"] == {
        "evidence_quality": {"0": 0, "1": 1, "2": 2},
        "risk_awareness": {"0": 0, "1": 3, "2": 0},
    }
    assert serialized["option_count_unstratified_count"] == 0
    assert [
        (group["option_count"], group["variant"], group["passed_count"])
        for group in serialized["option_count_summary"]
    ] == [
        (3, "baseline", 1),
        (3, "option_only", 1),
        (3, "replaced_distractor", 0),
        (3, "shuffled_options", 1),
    ]
    assert all("raw_output" not in outcome for outcome in serialized["outcomes"])
    failure = next(outcome for outcome in serialized["outcomes"] if outcome["status"] == "judge_failed")
    assert failure["evidence"] == {
        "call_count": 1,
        "records": [{"meets_threshold": False}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }

    effects = {effect["variant"]: effect for effect in report.paired_effects()}
    assert effects["option_only"]["score_delta"] == pytest.approx(-0.25)
    assert effects["shuffled_options"]["score_delta"] == pytest.approx(0.0)
    assert effects["replaced_distractor"]["control_status"] == "judge_failed"
    assert "score_delta" not in effects["replaced_distractor"]


@pytest.mark.parametrize(
    ("exception", "expected_type", "sentinels"),
    [
        (
            RuntimeError("provider-output-secret"),
            "RuntimeError",
            ("provider-output-secret",),
        ),
        (
            JudgeFormatError(
                "parser-output-secret",
                evidence={"error": "evidence-output-secret"},
            ),
            "JudgeFormatError",
            ("parser-output-secret", "evidence-output-secret"),
        ),
    ],
)
def test_calibration_failure_serialization_redacts_exception_text(
    exception: Exception, expected_type: str, sentinels: tuple[str, ...]
) -> None:
    report = evaluate_paired_calibration(
        _RaisingJudge(exception),
        _cases(),
        criteria=CRITERIA,
        category_count=3,
    )

    serialized = json.dumps(report.to_dict(), ensure_ascii=False)
    for sentinel in sentinels:
        assert sentinel not in serialized
    assert report.status_counts() == {"judge_failed": 4}
    for outcome in report.to_dict()["outcomes"]:
        assert outcome["error"] == "judge_call_failed"
        assert outcome["error_type"] == expected_type


def test_option_count_summary_preserves_unstratified_rows() -> None:
    cases = (
        JudgeCalibrationCase(
            case_id="unstratified",
            variant=variant,
            task="Question: q\n\nOptions:\n1. a\n2. b"
            if variant == "baseline"
            else "Options:\n1. a\n2. b",
            answer="a",
        )
        for variant in ("baseline", "option_only")
    )
    report = evaluate_paired_calibration(
        _ScriptedJudge(), cases, criteria=CRITERIA, category_count=3
    )
    serialized = report.to_dict()
    assert serialized["option_count_summary"] == []
    assert serialized["option_count_unstratified_count"] == 2


def test_option_count_summary_separates_multiple_k_strata() -> None:
    five_option_cases = build_multiple_choice_calibration_cases(
        case_id="release_case_k5",
        question="Which rollout plan is safest?",
        options=("canary", "big_bang", "rollback_only", "shadow", "manual"),
        answer="canary",
        correct_option_index=0,
        replacement_distractor="unreviewed_rollout",
        reference_answer="canary",
        contamination_status="held_out",
        shuffle_seed=7,
        gold_categories={"evidence_quality": 2, "risk_awareness": 1},
    )
    report = evaluate_paired_calibration(
        _ScriptedJudge(), _cases() + five_option_cases, criteria=CRITERIA, category_count=3
    )
    summary = report.to_dict()["option_count_summary"]
    assert {(group["option_count"], group["variant"]) for group in summary} == {
        (3, "baseline"),
        (3, "option_only"),
        (3, "shuffled_options"),
        (3, "replaced_distractor"),
        (5, "baseline"),
        (5, "option_only"),
        (5, "shuffled_options"),
        (5, "replaced_distractor"),
    }
    assert all(group["outcome_count"] == 1 for group in summary)


def test_report_rejects_malformed_direct_construction() -> None:
    report = evaluate_paired_calibration(
        _ScriptedJudge(), _cases(), criteria=CRITERIA, category_count=3
    )
    criterion_ids = tuple(criterion.criterion_id for criterion in CRITERIA)

    with pytest.raises(ValueError, match="criterion_ids"):
        JudgeCalibrationReport(3, (criterion_ids[0],), report.outcomes)
    with pytest.raises(ValueError, match="baseline"):
        JudgeCalibrationReport(3, criterion_ids, report.outcomes[:1])
    with pytest.raises(ValueError, match="category_count"):
        JudgeCalibrationReport(2, criterion_ids, report.outcomes)


@pytest.mark.parametrize("option_count", [True, 1, 65, 1.5])
def test_option_count_rejects_invalid_values(option_count: object) -> None:
    with pytest.raises(ValueError, match="option_count"):
        JudgeCalibrationCase(
            case_id="bad_k",
            variant="baseline",
            task="task",
            answer="answer",
            option_count=option_count,
        )


def test_option_count_rejects_conflicting_reserved_metadata() -> None:
    with pytest.raises(ValueError, match="agree with metadata"):
        JudgeCalibrationCase(
            case_id="conflict",
            variant="baseline",
            task="task",
            answer="answer",
            option_count=3,
            metadata={"option_count": "5"},
        )


def test_legacy_option_count_metadata_is_promoted_and_validated() -> None:
    case = JudgeCalibrationCase(
        case_id="legacy",
        variant="baseline",
        task="task",
        answer="answer",
        metadata={"option_count": "3"},
    )
    assert case.option_count == 3


def test_direct_paired_calibration_reuses_gateway_concurrency_and_order() -> None:
    orchestrator = _ConcurrentOrchestrator()
    judge = ContextualOrchestratorJudge(orchestrator)
    report = evaluate_paired_calibration(
        judge,
        _cases(),
        criteria=CRITERIA,
        category_count=3,
        category_method="direct",
    )

    assert [outcome.case.variant for outcome in report.outcomes] == [
        "baseline",
        "option_only",
        "shuffled_options",
        "replaced_distractor",
    ]
    assert report.status_counts() == {"passed": 4}
    assert [outcome.irt_row for outcome in report.outcomes] == [(2, 1)] * 4
    assert orchestrator.peak == 2


def test_calibration_requires_multiple_items_and_a_baseline_pair() -> None:
    case = JudgeCalibrationCase(
        case_id="single",
        variant="baseline",
        task="task",
        answer="answer",
    )
    with pytest.raises(ValueError, match="baseline and at least one control"):
        evaluate_paired_calibration(
            _ScriptedJudge(),
            [case],
            criteria=CRITERIA,
            category_count=3,
        )

    control = JudgeCalibrationCase(
        case_id="single",
        variant="option_only",
        task="options",
        answer="answer",
    )
    with pytest.raises(ValueError, match="multiple criterion items"):
        evaluate_paired_calibration(
            _ScriptedJudge(),
            [case, control],
            criteria=[CRITERIA[0]],
            category_count=3,
        )


def test_calibration_without_gold_does_not_claim_agreement() -> None:
    cases = tuple(
        JudgeCalibrationCase(
            case_id="no_gold",
            variant=variant,
            task="Question: q\n\nOptions:\n1. a\n2. b" if variant == "baseline" else "Options:\n1. a\n2. b",
            answer="a",
        )
        for variant in ("baseline", "option_only")
    )
    report = evaluate_paired_calibration(
        _ScriptedJudge(), cases, criteria=CRITERIA, category_count=3
    )
    assert report.to_dict()["gold_scored_count"] == 0
    assert report.to_dict()["gold_exact_agreement"] is None


def test_calibration_rejects_ambiguous_normalized_inputs() -> None:
    with pytest.raises(TypeError, match="iterable of distinct strings"):
        build_multiple_choice_calibration_cases(
            case_id="bad_options",
            question="Which option?",
            options="a",
            answer="a",
            correct_option_index=0,
            replacement_distractor="b",
        )

    with pytest.raises(ValueError, match="unique after trimming"):
        JudgeCalibrationCase(
            case_id="bad_metadata",
            variant="baseline",
            task="task",
            answer="answer",
            metadata={"key": "one", " key ": "two"},
        )

    with pytest.raises(ValueError, match="unique after trimming"):
        JudgeCalibrationCase(
            case_id="bad_gold",
            variant="baseline",
            task="task",
            answer="answer",
            gold_categories={"a": 0, " a ": 1},
        )
