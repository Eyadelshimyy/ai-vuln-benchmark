"""LLM judge parsing is strict and transport stays injected."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fast_mlsirm import CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1
from fast_mlsirm.irt_contract import validate_irt_response_matrix
from fast_mlsirm.llm_judge import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    MAX_BINARY_THRESHOLD_CALLS,
    ContextualOrchestratorJudge,
    JudgeCriterion,
    JudgeFormatError,
)


class _FakeOrchestrator:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = []

    def complete(self, messages, mode="auto"):
        self.calls.append((messages, mode))
        return {
            "mode": "route",
            "answer": self.answer,
            "trace": [{"usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}],
        }


class _CompletionOrchestrator:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self, completion):
        self.completion = completion

    def complete(self, messages, mode="auto"):
        return self.completion


class _RaisingOrchestrator:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self, message: str) -> None:
        self.message = message

    def complete(self, messages, mode="auto"):
        raise RuntimeError(self.message)


class _SequencedOrchestrator:
    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def complete(self, messages, mode="auto"):
        self.calls.append((messages, mode))
        return {
            "mode": "route",
            "answer": next(self.answers),
            "trace": [{"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}],
        }


class _ParallelOrchestrator(_SequencedOrchestrator):
    def __init__(self, answers):
        super().__init__(answers)
        self.client = SimpleNamespace(local_concurrency=2)
        self.barrier = threading.Barrier(2)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def complete(self, messages, mode="auto"):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.barrier.wait(timeout=2)
            return super().complete(messages, mode)
        finally:
            with self._lock:
                self.active -= 1


class _StructuredOrchestrator(_SequencedOrchestrator):
    def __init__(self, answers):
        super().__init__(answers)
        self.structured_formats = []

    def complete_structured(self, messages, mode="auto", *, response_format):
        self.structured_formats.append(response_format)
        return super().complete(messages, mode)


CRITERIA = [
    JudgeCriterion("task_alignment", "The answer directly addresses the task."),
    JudgeCriterion("factual_support", "The answer avoids unsupported claims."),
]

ANCHORED_CRITERIA = [
    JudgeCriterion(
        "task_alignment",
        "The answer directly addresses the task.",
        category_anchors=("no alignment", "some alignment", "clear alignment"),
    ),
    JudgeCriterion(
        "factual_support",
        "The answer avoids unsupported claims.",
        category_anchors=("no support", "some support", "fully supported"),
    ),
]


def test_contextual_orchestrator_contract_is_public_and_versioned() -> None:
    assert CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1 == "contextual-orchestrator-contract-v1"


def _payload(score=0.8, accepted=True):
    return json.dumps({
        "score": score,
        "accepted": accepted,
        "rationale": "The answer is concise and supported.",
        "criterion_scores": {"task_alignment": score, "factual_support": score},
    })


def _category_payload():
    return json.dumps({
        "score": 0.75,
        "accepted": False,
        "rationale": "The evidence supports the ordered criterion levels.",
        "criterion_categories": {"task_alignment": 4.0, "factual_support": 2.0},
    })


def _threshold_payload(thresholds=None):
    return json.dumps({
        "score": 0.0,
        "accepted": True,
        "rationale": "The ordered evidence supports separate cumulative thresholds.",
        "criterion_thresholds": thresholds or {
            "task_alignment": [True, True, True, True],
            "factual_support": [True, False, False, False],
        },
    })


def test_judge_rejects_an_unmarked_transport() -> None:
    class _Unmarked:
        def complete(self, messages, mode="auto"):
            del messages, mode
            return {}

    with pytest.raises(TypeError, match="contextual-orchestrator-contract-v1"):
        ContextualOrchestratorJudge(_Unmarked())


def test_judge_uses_contextual_orchestrator_route_and_reports_usage() -> None:
    orchestrator = _FakeOrchestrator(_payload())
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="Explain the release plan.",
        answer="Use a staged release with rollback.",
        criteria=CRITERIA,
    )
    assert result.accepted is True
    assert result.score == 0.8
    assert result.trace_step_count == 1
    assert dict(result.usage) == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    assert orchestrator.calls[0][1] == "auto"
    prompt = orchestrator.calls[0][0][1]["content"]
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["task"] == "Explain the release plan."
    assert payload["answer"] == "Use a staged release with rollback."


def test_direct_judge_derives_score_from_weighted_criteria_not_self_report() -> None:
    """A misleading self-reported aggregate score must not decide acceptance.

    The runtime must not trust the model's own top-level "score" field: it
    must derive the accepted score from the per-criterion scores and their
    configured weights, exactly like the category_count branch already does.
    Otherwise a criterion's weight has no effect on the outcome, and a model
    that reports a high aggregate score while giving a low score on the
    heavily-weighted criterion would be wrongly accepted.
    """
    weighted_criteria = [
        JudgeCriterion(
            "task_alignment", "The answer directly addresses the task.", weight=3.0
        ),
        JudgeCriterion(
            "factual_support", "The answer avoids unsupported claims.", weight=1.0
        ),
    ]
    misleading_payload = json.dumps({
        "score": 0.9,
        "accepted": True,
        "rationale": "The response looks strong overall.",
        "criterion_scores": {"task_alignment": 0.2, "factual_support": 0.9},
    })
    result = ContextualOrchestratorJudge(_FakeOrchestrator(misleading_payload)).judge(
        task="Explain the release plan.",
        answer="Use a staged release with rollback.",
        criteria=weighted_criteria,
    )
    # weighted average = (3 * 0.2 + 1 * 0.9) / 4 = 0.375, not the
    # self-reported 0.9.
    assert result.score == pytest.approx(0.375)
    assert result.accepted is False


def test_criteria_reject_overflowing_aggregate_weight_before_any_transport() -> None:
    """Individually valid weights must not silently overflow their sum.

    JudgeCriterion validates each weight as finite and positive, but two
    such weights (e.g. 1e308 each) can still overflow their aggregate to
    inf, which would otherwise collapse a weighted score to an incorrect
    finite value instead of failing closed. This must fail before any
    orchestrator call is made.
    """
    overflowing_criteria = [
        JudgeCriterion("task_alignment", "First criterion.", weight=1e308),
        JudgeCriterion("factual_support", "Second criterion.", weight=1e308),
    ]
    orchestrator = _FakeOrchestrator(_payload())
    with pytest.raises(ValueError, match="aggregate criterion weight must be finite"):
        ContextualOrchestratorJudge(orchestrator).judge(
            task="Explain the release plan.",
            answer="Use a staged release with rollback.",
            criteria=overflowing_criteria,
        )
    assert orchestrator.calls == []


def test_direct_judge_does_not_expose_provider_exception_text() -> None:
    sentinel = "provider-output-secret"
    with pytest.raises(JudgeFormatError, match="^judge call failed$") as exc_info:
        ContextualOrchestratorJudge(_RaisingOrchestrator(sentinel)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )

    assert sentinel not in str(exc_info.value)
    assert str(exc_info.value) == "judge call failed"
    assert exc_info.value.evidence["semantic_status"] == "transport_failure"
    assert exc_info.value.evidence["records"][0]["error_type"] == "RuntimeError"
    assert sentinel not in json.dumps(exc_info.value.evidence)


def test_direct_judge_failure_exposes_bounded_status_evidence() -> None:
    with pytest.raises(JudgeFormatError) as exc_info:
        ContextualOrchestratorJudge(_FakeOrchestrator("not json")).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )

    evidence = exc_info.value.evidence
    assert evidence["category_method"] == "direct"
    assert evidence["category_count"] is None
    assert evidence["semantic_status"] == "response_parse_failure"
    assert evidence["records"] == [
        {
            "call_status": "completed",
            "parse_status": "failed",
            "error_type": "JudgeFormatError",
            "failure_code": "judge_response_invalid",
        }
    ]
    assert "not json" not in json.dumps(evidence)


def test_cumulative_judge_failure_exposes_bounded_status_evidence() -> None:
    with pytest.raises(JudgeFormatError) as exc_info:
        ContextualOrchestratorJudge(_FakeOrchestrator("not json")).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=3,
            category_method="cumulative_threshold",
        )

    evidence = exc_info.value.evidence
    assert evidence["category_method"] == "cumulative_threshold"
    assert evidence["category_count"] == 3
    assert evidence["completed_call_count"] == 1
    assert evidence["failed_call_count"] == 0
    assert evidence["parse_status"] == "failed"


def test_cumulative_validation_failure_exposes_bounded_status_evidence() -> None:
    answer = _threshold_payload({
        "task_alignment": [False, True],
        "factual_support": [True, False],
    })
    with pytest.raises(JudgeFormatError, match="monotone") as exc_info:
        ContextualOrchestratorJudge(_FakeOrchestrator(answer)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=3,
            category_method="cumulative_threshold",
        )

    evidence = exc_info.value.evidence
    assert evidence["semantic_status"] == "response_validation_failure"
    assert evidence["parse_status"] == "passed"
    assert evidence["records"][0]["failure_code"] == "judge_response_invalid"


def test_direct_judgment_prompt_includes_complete_schema_example() -> None:
    orchestrator = _FakeOrchestrator(_payload())
    ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    prompt = orchestrator.calls[0][0][0]["content"]
    assert '"rationale": "brief evidence-based reason"' in prompt
    assert '"criterion_scores": {"task_alignment": 0.0, "factual_support": 0.0}' in prompt
    assert "keeping every key" in prompt
    assert "never an object" in prompt


def test_judge_rejects_malformed_decisions_and_derives_acceptance() -> None:
    with pytest.raises(JudgeFormatError):
        ContextualOrchestratorJudge(_FakeOrchestrator("not json")).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )
    result = ContextualOrchestratorJudge(
        _FakeOrchestrator(_payload(score=0.8, accepted=False))
    ).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.accepted is True


@pytest.mark.parametrize("accepted", [0, 1, "true", None])
def test_judge_rejects_non_boolean_advisory_acceptance(accepted) -> None:
    with pytest.raises(JudgeFormatError, match="accepted must be a boolean"):
        ContextualOrchestratorJudge(
            _FakeOrchestrator(_payload(accepted=accepted))
        ).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )


def test_judge_rejects_wrapped_or_fenced_json() -> None:
    for answer in (
        f"prefix {_payload()}",
        f"{_payload()} suffix",
        f"```json\n{_payload()}\n```",
    ):
        with pytest.raises(JudgeFormatError):
            ContextualOrchestratorJudge(_FakeOrchestrator(answer)).judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
            )



def test_judge_rejects_duplicate_and_unknown_top_level_fields() -> None:
    duplicate = (
        '{"score":0.8,"accepted":true,"rationale":"supported",'
        '"criterion_scores":{"task_alignment":0.8,"factual_support":0.8},'
        '"score":0.2}'
    )
    unknown = json.loads(_payload())
    unknown["unexpected"] = "ignored fields are unsafe"
    for answer in (duplicate, json.dumps(unknown)):
        with pytest.raises(JudgeFormatError, match="exactly|duplicate"):
            ContextualOrchestratorJudge(_FakeOrchestrator(answer)).judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
            )


def test_judge_rejects_duplicate_nested_criterion_fields() -> None:
    answer = (
        '{"score":0.8,"accepted":true,"rationale":"supported",'
        '"criterion_scores":{"task_alignment":0.8,"task_alignment":0.2,'
        '"factual_support":0.8}}'
    )
    with pytest.raises(JudgeFormatError, match="duplicate"):
        ContextualOrchestratorJudge(_FakeOrchestrator(answer)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )


def test_judge_result_projects_only_multiple_criteria_to_irt_items() -> None:
    result = ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.to_irt_row(item_type="dichotomous") == (1, 1)
    assert result.to_irt_row(item_type="polytomous", n_categories=5) == (4, 4)

    single_criterion = ContextualOrchestratorJudge(
        _FakeOrchestrator(
            json.dumps(
                {
                    "score": 0.8,
                    "accepted": True,
                    "rationale": "supported",
                    "criterion_scores": {"task_alignment": 0.8},
                }
            )
        )
    ).judge(
        task="task",
        answer="answer",
        criteria=[CRITERIA[0]],
    )
    with pytest.raises(JudgeFormatError):
        single_criterion.to_irt_row()


def test_irt_projection_rejects_malformed_result_mappings() -> None:
    result = ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    with pytest.raises(JudgeFormatError, match="keys must be strings"):
        replace(result, criterion_scores={1: 0.8, "factual_support": 0.8}).to_irt_row()
    with pytest.raises(JudgeFormatError, match="criterion_categories must be an object"):
        replace(
            result,
            criterion_categories=[0, 1],
            category_count=2,
        ).to_irt_row()
    with pytest.raises(JudgeFormatError, match="item_type"):
        result.to_irt_row(item_type=[])


def test_criteria_limit_is_enforced_during_iteration() -> None:
    yielded = 0

    def criteria():
        nonlocal yielded
        for index in range(33):
            yielded += 1
            yield JudgeCriterion(f"criterion_{index}", "observable evidence")

    with pytest.raises(ValueError, match="1..32"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=criteria(),
        )
    assert yielded == 33


def test_category_judgment_defaults_to_binary_threshold_for_polytomous_output() -> None:
    payload = json.dumps({"meets_threshold": True, "rationale": "supported"})
    orchestrator = _SequencedOrchestrator([payload] * 4)
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=3,
    )
    assert result.category_method == "binary_threshold"
    assert result.to_irt_row() == (2, 2)
    assert len(orchestrator.calls) == 4
    assert all("binary" in call[0][0]["content"] for call in orchestrator.calls)


def test_binary_threshold_uses_structured_contextual_transport_when_available() -> None:
    payload = json.dumps({"meets_threshold": True, "rationale": "supported"})
    orchestrator = _StructuredOrchestrator([payload] * 4)
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=3,
    )
    assert result.to_irt_row() == (2, 2)
    assert len(orchestrator.structured_formats) == 4
    schema = orchestrator.structured_formats[0]["json_schema"]["schema"]
    assert schema["required"] == ["meets_threshold", "rationale"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["rationale"]["maxLength"] == 256
    assert all(fmt == orchestrator.structured_formats[0] for fmt in orchestrator.structured_formats)


def test_category_anchors_are_bound_to_each_binary_threshold() -> None:
    payload = json.dumps({"meets_threshold": True, "rationale": "supported"})
    orchestrator = _SequencedOrchestrator([payload] * 4)
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=ANCHORED_CRITERIA,
        category_count=3,
    )
    assert result.category_anchors_provided is True
    assert result.to_dict()["category_anchors_provided"] is True
    first_messages = orchestrator.calls[0][0]
    first_payload = json.loads(first_messages[1]["content"].split("\n", 1)[1])
    assert first_payload["criterion"]["category_anchors"] == [
        "no alignment",
        "some alignment",
        "clear alignment",
    ]
    assert first_payload["category_anchor"] == "some alignment"
    assert "authoritative definition" in first_messages[0]["content"]
    assert "not an exact-category classification" in first_messages[0]["content"]
    assert "exceeds this boundary" in first_messages[0]["content"]
    assert "relevance is required" in first_messages[0]["content"]


def test_category_anchors_require_a_complete_polytomous_contract() -> None:
    with pytest.raises(ValueError, match="explicit category_count"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=ANCHORED_CRITERIA,
        )
    with pytest.raises(ValueError, match="match category_count"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[
                JudgeCriterion(
                    "task_alignment",
                    "The answer directly addresses the task.",
                    category_anchors=("none", "some"),
                ),
                JudgeCriterion(
                    "factual_support",
                    "The answer avoids unsupported claims.",
                    category_anchors=("none", "some"),
                ),
            ],
            category_count=3,
        )
    with pytest.raises(ValueError, match="for every criterion"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[ANCHORED_CRITERIA[0], CRITERIA[1]],
            category_count=3,
        )


def test_category_judgment_direct_method_remains_explicit_calibration_only() -> None:
    orchestrator = _FakeOrchestrator(_category_payload())
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=5,
        category_method="direct",
    )
    assert result.category_count == 5
    assert dict(result.criterion_categories) == {
        "factual_support": 2,
        "task_alignment": 4,
    }
    assert result.score == 0.75
    assert result.accepted is True
    assert result.to_irt_row() == (2, 4)
    prompt = orchestrator.calls[0][0][0]["content"]
    assert '"task_alignment"' in prompt
    assert "numeric keys" in prompt
    assert "whole-number values from [0, 1, 2, 3, 4]" in prompt
    assert "category 4 means fully satisfies" in prompt
    assert "no markdown fences" in prompt
    assert "category values are JSON integers" in prompt
    assert "category 4" in prompt


def test_cumulative_threshold_judgment_derives_monotone_polytomous_items() -> None:
    orchestrator = _FakeOrchestrator(_threshold_payload())
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=5,
        category_method="cumulative_threshold",
    )

    assert result.category_method == "cumulative_threshold"
    assert dict(result.criterion_categories) == {
        "factual_support": 1,
        "task_alignment": 4,
    }
    assert result.score == 0.625
    assert result.accepted is False
    assert result.to_dict()["category_method"] == "cumulative_threshold"
    row = result.to_irt_row()
    assert row == (1, 4)
    matrix = validate_irt_response_matrix([row], "polytomous", n_categories=5)
    assert matrix.shape == (1, 2)
    prompt = orchestrator.calls[0][0][0]["content"]
    assert "criterion_thresholds" in prompt
    assert "cumulative thresholds" in prompt
    assert "must be monotone" in prompt
    assert "K-way choice" in prompt


@pytest.mark.parametrize(
    ("thresholds", "match"),
    [
        (
            {"task_alignment": [True, False, True, False], "factual_support": [False] * 4},
            "monotone",
        ),
        (
            {"task_alignment": [True, 1, False, False], "factual_support": [False] * 4},
            "boolean",
        ),
        (
            {"task_alignment": [True, True], "factual_support": [False] * 4},
            "boolean array",
        ),
    ],
)
def test_cumulative_threshold_rejects_malformed_thresholds(thresholds, match) -> None:
    with pytest.raises(JudgeFormatError, match=match):
        ContextualOrchestratorJudge(
            _FakeOrchestrator(_threshold_payload(thresholds))
        ).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=5,
            category_method="cumulative_threshold",
        )


def test_cumulative_threshold_requires_explicit_category_count() -> None:
    with pytest.raises(ValueError, match="explicit category_count"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_method="cumulative_threshold",
        )


def test_binary_threshold_judgment_uses_independent_boolean_boundaries() -> None:
    orchestrator = _SequencedOrchestrator(
        [
            json.dumps({"meets_threshold": True, "rationale": "supported"}),
            json.dumps({"meets_threshold": True, "rationale": "supported"}),
            json.dumps({"meets_threshold": True, "rationale": "supported"}),
            json.dumps({"meets_threshold": False, "rationale": "not established"}),
        ]
    )
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=3,
        category_method="binary_threshold",
    )
    assert result.category_method == "binary_threshold"
    assert dict(result.criterion_categories) == {
        "factual_support": 1,
        "task_alignment": 2,
    }
    assert result.score == 0.75
    assert result.to_irt_row() == (1, 2)
    assert len(orchestrator.calls) == 4
    assert all("binary" in call[0][0]["content"] for call in orchestrator.calls)
    assert dict(result.usage) == {
        "prompt_tokens": 4,
        "completion_tokens": 4,
        "total_tokens": 8,
    }


def test_binary_threshold_reuses_bounded_gateway_concurrency() -> None:
    payload = json.dumps({"meets_threshold": True, "rationale": "supported"})
    orchestrator = _ParallelOrchestrator([payload, payload])
    result = ContextualOrchestratorJudge(orchestrator).judge(
        task="task",
        answer="answer",
        criteria=[CRITERIA[0]],
        category_count=3,
        category_method="binary_threshold",
    )
    assert result.score == 1.0
    assert orchestrator.peak == 2


def test_binary_threshold_judgment_rejects_non_monotone_boundaries() -> None:
    orchestrator = _SequencedOrchestrator(
        [
            json.dumps({"meets_threshold": False, "rationale": "not established"}),
            json.dumps({"meets_threshold": True, "rationale": "unsupported jump"}),
        ]
    )
    with pytest.raises(JudgeFormatError, match="monotone") as exc_info:
        ContextualOrchestratorJudge(orchestrator).judge(
            task="task",
            answer="answer",
            criteria=[CRITERIA[0]],
            category_count=3,
            category_method="binary_threshold",
        )
    evidence = exc_info.value.evidence
    assert evidence["parse_status"] == "passed"
    assert evidence["semantic_status"] == "non_monotone"
    assert evidence["call_count"] == 2
    assert evidence["completed_call_count"] == 2
    assert evidence["trace_step_count"] == 2
    assert evidence["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert [record["parse_status"] for record in evidence["records"]] == [
        "passed",
        "passed",
    ]
    assert [record["meets_threshold"] for record in evidence["records"]] == [
        False,
        True,
    ]


def test_binary_threshold_failure_exposes_bounded_failure_evidence() -> None:
    orchestrator = _SequencedOrchestrator(
        [
            json.dumps({"meets_threshold": True, "rationale": "supported"}),
            "not json",
        ]
    )
    with pytest.raises(JudgeFormatError, match="failed closed") as exc_info:
        ContextualOrchestratorJudge(orchestrator).judge(
            task="task",
            answer="answer",
            criteria=[CRITERIA[0]],
            category_count=3,
            category_method="binary_threshold",
        )
    evidence = exc_info.value.evidence
    assert evidence["category_method"] == "binary_threshold"
    assert evidence["category_count"] == 3
    assert evidence["call_count"] == 2
    assert evidence["completed_call_count"] == 2
    assert evidence["failed_call_count"] == 0
    assert evidence["parse_status"] == "failed"
    assert evidence["semantic_status"] == "boundary_failure"
    assert evidence["trace_step_count"] == 2
    assert evidence["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert evidence["records"][0]["parse_status"] == "passed"
    assert evidence["records"][1]["parse_status"] == "failed"
    assert evidence["records"][1]["error_type"] == "JudgeFormatError"
    assert evidence["records"][1]["failure_code"] == "binary_boundary_call_failed"
    assert "output_preview" not in evidence["records"][1]
    assert "error" not in evidence["records"][1]


def _assert_binary_failure_evidence_redacted(orchestrator, sentinel: str) -> None:
    with pytest.raises(JudgeFormatError) as exc_info:
        ContextualOrchestratorJudge(orchestrator).judge(
            task="task",
            answer="answer",
            criteria=[CRITERIA[0]],
            category_count=3,
            category_method="binary_threshold",
        )
    evidence = exc_info.value.evidence
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert sentinel not in serialized
    for record in evidence["records"]:
        assert "output_preview" not in record
        assert "error" not in record
        assert record["failure_code"] == "binary_boundary_call_failed"


def test_binary_threshold_failure_evidence_redacts_provider_output() -> None:
    sentinel = "provider-output-secret"
    _assert_binary_failure_evidence_redacted(
        _SequencedOrchestrator([sentinel, sentinel]),
        sentinel,
    )


def test_binary_threshold_failure_evidence_redacts_exception_text() -> None:
    sentinel = "exception-text-secret"
    _assert_binary_failure_evidence_redacted(_RaisingOrchestrator(sentinel), sentinel)


def test_binary_threshold_judgment_has_a_bounded_call_budget() -> None:
    with pytest.raises(ValueError, match="too many judge calls"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=MAX_BINARY_THRESHOLD_CALLS,
            category_method="binary_threshold",
        )


@pytest.mark.parametrize("category_method", ["unknown", [], {}])
def test_judge_rejects_unknown_category_method(category_method) -> None:
    with pytest.raises(ValueError, match="category_method"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_method=category_method,
        )


def test_judge_rejects_unhashable_mode_before_membership() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload()), mode=[])


def test_category_judgment_rejects_non_integral_categories() -> None:
    payload = json.dumps({
        "score": 0.5,
        "accepted": True,
        "rationale": "mixed evidence",
        "criterion_categories": {"task_alignment": 1.5, "factual_support": 1},
    })
    with pytest.raises(JudgeFormatError, match="integer"):
        ContextualOrchestratorJudge(_FakeOrchestrator(payload)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=3,
            category_method="direct",
        )


def test_category_count_and_category_values_reject_runtime_subclasses() -> None:
    class _ForgedInt(int):
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    judge = ContextualOrchestratorJudge(_FakeOrchestrator(_category_payload()))
    for value in (True, 1.0, 65, 10**1000, _ForgedInt(10**1000)):
        with pytest.raises(ValueError, match="category_count must be an integer"):
            judge.judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
                category_count=value,
            )

    result = judge.judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
        category_count=5,
        category_method="direct",
    )
    with pytest.raises(JudgeFormatError, match="criterion_categories"):
        replace(
            result,
            criterion_categories={
                "task_alignment": _ForgedInt(10**1000),
                "factual_support": 1,
            },
        ).to_irt_row()


def test_category_judgment_rejects_malformed_top_level_score() -> None:
    payload = json.dumps({
        "score": {"factual_support": 0.8},
        "accepted": True,
        "rationale": "mixed evidence",
        "criterion_categories": {"task_alignment": 1, "factual_support": 1},
    })
    with pytest.raises(JudgeFormatError, match="score must be a number"):
        ContextualOrchestratorJudge(_FakeOrchestrator(payload)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
            category_count=2,
            category_method="direct",
        )


def test_judge_rejects_overflowing_and_runtime_subclass_scores() -> None:
    class _HookedFloat(float):
        invoked = False

        def __float__(self):
            type(self).invoked = True
            return super().__float__()

    overflowing = json.dumps({
        "score": 10**1000,
        "accepted": True,
        "rationale": "unsupported numeric shape",
        "criterion_scores": {"task_alignment": 0.8, "factual_support": 0.8},
    })
    with pytest.raises(JudgeFormatError, match="score must be a number"):
        ContextualOrchestratorJudge(_FakeOrchestrator(overflowing)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )

    subclass_score = _HookedFloat(0.8)
    result = ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    with pytest.raises(JudgeFormatError, match="criterion_scores"):
        replace(
            result,
            criterion_scores={
                "task_alignment": subclass_score,
                "factual_support": 0.8,
            },
        ).to_irt_row(item_type="dichotomous")
    assert _HookedFloat.invoked is False


def test_judge_text_and_usage_boundaries_reject_runtime_subclasses() -> None:
    class _HookedString(str):
        invoked = False

        def strip(self, *args, **kwargs):
            type(self).invoked = True
            return super().strip(*args, **kwargs)

    class _ForgedInt(int):
        invoked = False

        def __ge__(self, other):
            type(self).invoked = True
            return True

    with pytest.raises(ValueError, match="task must be"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task=_HookedString("task"),
            answer="answer",
            criteria=CRITERIA,
        )
    assert _HookedString.invoked is False

    forged = _ForgedInt(7)
    result = _CompletionOrchestrator({
        "mode": "route",
        "answer": _payload(),
        "trace": [{
            "usage": {
                "prompt_tokens": forged,
                "completion_tokens": forged,
                "total_tokens": forged,
            }
        }],
    })
    judged = ContextualOrchestratorJudge(result).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert dict(judged.usage) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert _ForgedInt.invoked is False


def test_judge_rejects_missing_or_malformed_model_fields() -> None:
    cases = [
        {},
        {"answer": _payload().replace("rationale", "explanation")},
        "not a completion mapping",
    ]
    for completion in cases:
        with pytest.raises(JudgeFormatError):
            ContextualOrchestratorJudge(_CompletionOrchestrator(completion)).judge(
                task="task",
                answer="answer",
                criteria=CRITERIA,
            )


def test_judge_criteria_reject_invalid_runtime_types() -> None:
    class _HookedFloat(float):
        invoked = False

        def __float__(self):
            type(self).invoked = True
            return super().__float__()

    with pytest.raises(ValueError, match="criterion_id must be a string"):
        JudgeCriterion(1, "description")
    with pytest.raises(ValueError, match="criterion description must be a string"):
        JudgeCriterion("task_alignment", 1)
    with pytest.raises(ValueError, match="criterion weight must be a number"):
        JudgeCriterion("task_alignment", "description", "1")
    with pytest.raises(ValueError, match="category_anchors must be a tuple"):
        JudgeCriterion(
            "task_alignment",
            "description",
            category_anchors=["none", "some"],
        )
    with pytest.raises(ValueError, match="criterion weight must be a number"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[
                {
                    "criterion_id": "task_alignment",
                    "description": "ok",
                    "weight": "1",
                }
            ],
        )
    for criterion in (
        {"criterion_id": 1, "description": "ok"},
        {"criterion_id": "task_alignment", "description": 1},
        {
            "criterion_id": "task_alignment",
            "description": "ok",
            "weight": _HookedFloat(1.0),
        },
    ):
        with pytest.raises(ValueError):
            ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
                task="task",
                answer="answer",
                criteria=[criterion],
            )
    assert _HookedFloat.invoked is False


def test_judge_rejects_unhashable_criterion_id_before_category_template() -> None:
    class _UnhashableStr(str):
        __hash__ = None

    with pytest.raises(ValueError, match="criterion_id must be a string"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[
                {
                    "criterion_id": _UnhashableStr("task_alignment"),
                    "description": "ok",
                },
                CRITERIA[1],
            ],
            category_count=3,
        )


def test_judge_criteria_reject_non_contract_values_with_value_error() -> None:
    """Arbitrary criterion elements must fail through the stable benign error contract."""
    with pytest.raises(ValueError, match="JudgeCriterion or mapping"):
        ContextualOrchestratorJudge(_FakeOrchestrator(_payload())).judge(
            task="task",
            answer="answer",
            criteria=[object()],
        )


if __name__ == "__main__":
    test_judge_uses_contextual_orchestrator_route_and_reports_usage()
    test_judge_rejects_malformed_decisions_and_derives_acceptance()
    test_judge_result_projects_only_multiple_criteria_to_irt_items()
    test_irt_projection_rejects_malformed_result_mappings()
    test_criteria_limit_is_enforced_during_iteration()
    test_category_judgment_direct_method_remains_explicit_calibration_only()
    test_category_judgment_rejects_non_integral_categories()
    test_judge_rejects_missing_or_malformed_model_fields()
    test_judge_criteria_reject_invalid_runtime_types()
    test_judge_criteria_reject_non_contract_values_with_value_error()
    print("ok")

def test_judge_rejects_excessive_json_nesting() -> None:
    """Deeply nested JSON cannot expand into recursive parser DoS."""
    # Nesting depth 33 exceeds MAX_JUDGE_JSON_DEPTH (32).
    nested = "{" + '"k":{' * 32 + '"score": 0.8' + "}" * 32 + "}"
    assert nested.count("{") == 33
    with pytest.raises(JudgeFormatError, match="nesting exceeds maximum depth"):
        ContextualOrchestratorJudge(_FakeOrchestrator(nested)).judge(
            task="task",
            answer="answer",
            criteria=CRITERIA,
        )


def test_judge_accepts_bounded_json_nesting() -> None:
    """Nesting at the admitted depth still parses when the payload is valid."""
    # Build a valid judge payload with modest nesting under the limit.
    inner = {
        "score": 0.8,
        "accepted": True,
        "rationale": "ok",
        "criterion_scores": {"task_alignment": 0.8, "factual_support": 0.8},
    }
    raw = json.dumps(inner)
    result = ContextualOrchestratorJudge(_FakeOrchestrator(raw)).judge(
        task="task",
        answer="answer",
        criteria=CRITERIA,
    )
    assert result.score == 0.8
