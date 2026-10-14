"""Regressions for exact built-in LLM-judge trust-boundary values."""

import json

import pytest

from fast_mlsirm.llm_judge import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    ContextualOrchestratorJudge,
    JudgeCriterion,
)


def test_criterion_description_rejects_runtime_string_subclass_before_hooks() -> None:
    """Descriptions reject string subclasses before invoking caller hooks."""

    class _HookedString(str):
        invoked = False

        def strip(self, *args, **kwargs):
            type(self).invoked = True
            return super().strip(*args, **kwargs)

    with pytest.raises(ValueError, match="criterion description must be a string"):
        JudgeCriterion("task_alignment", _HookedString("observable evidence"))
    assert _HookedString.invoked is False


def test_trace_rejects_runtime_list_subclass_before_hooks() -> None:
    """Provider trace subclasses cannot execute iteration or length hooks."""

    class _HookedTrace(list):
        invoked = False

        def __iter__(self):
            type(self).invoked = True
            return super().__iter__()

        def __len__(self):
            type(self).invoked = True
            return super().__len__()

    trace = _HookedTrace(
        [{"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}]
    )
    answer = json.dumps(
        {
            "score": 0.8,
            "accepted": True,
            "rationale": "bounded evidence",
            "criterion_scores": {"task_alignment": 0.8, "factual_support": 0.8},
        }
    )

    class _Orchestrator:
        contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

        def complete(self, messages, mode="auto"):
            return {"mode": "route", "answer": answer, "trace": trace}

    result = ContextualOrchestratorJudge(_Orchestrator()).judge(
        task="task",
        answer="answer",
        criteria=(
            JudgeCriterion("task_alignment", "observable evidence"),
            JudgeCriterion("factual_support", "supported claims"),
        ),
    )

    assert result.trace_step_count == 0
    assert dict(result.usage) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert _HookedTrace.invoked is False
