"""Preserve contextual-orchestrator mode defaults and explicit overrides."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.llm_judge import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
    ContextualOrchestratorJudge,
    JudgeCriterion,
)


class _RecordingOrchestrator:
    """Return one valid decision while recording the requested mode."""

    contextual_orchestrator_contract = CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1

    def __init__(self) -> None:
        self.modes: list[str] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        mode: str,
    ) -> dict[str, object]:
        """Record ``mode`` and return a bounded contract-compatible fixture."""
        assert messages
        self.modes.append(mode)
        return {
            "mode": mode,
            "answer": json.dumps(
                {
                    "score": 1.0,
                    "accepted": True,
                    "rationale": "The answer satisfies the criterion.",
                    "criterion_scores": {"task_alignment": 1.0},
                }
            ),
            "trace": [],
        }


def _judge(mode: str | None = None) -> tuple[_RecordingOrchestrator, object]:
    """Execute one judge call with either the default or an explicit mode."""
    orchestrator = _RecordingOrchestrator()
    judge = (
        ContextualOrchestratorJudge(orchestrator)
        if mode is None
        else ContextualOrchestratorJudge(orchestrator, mode=mode)
    )
    result = judge.judge(
        task="Assess the release plan.",
        answer="Use a staged release with rollback.",
        criteria=[JudgeCriterion("task_alignment", "The answer addresses the task.")],
    )
    return orchestrator, result


def test_judge_defaults_to_adaptive_auto_mode() -> None:
    """Ordinary consumers delegate execution topology to contextual-orchestrator."""
    orchestrator, result = _judge()

    assert orchestrator.modes == ["auto"]
    assert result.orchestration_mode == "auto"


@pytest.mark.parametrize("mode", ["route", "conduct"])
def test_explicit_judge_mode_reaches_contextual_orchestrator(mode: str) -> None:
    """Explicit route/conduct overrides remain observable for controlled use."""
    orchestrator, result = _judge(mode)

    assert orchestrator.modes == [mode]
    assert result.orchestration_mode == mode
