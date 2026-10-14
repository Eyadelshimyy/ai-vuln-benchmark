"""Panel-level category-generation semantics for judge construct projection."""

from __future__ import annotations

from dataclasses import replace

import pytest

import fast_mlsirm.judge_construct as judge_construct
from fast_mlsirm.judge_construct import (
    JudgeFormatError,
    LLMJudgeResult,
    project_judge_results_to_matrix,
    validate_judge_construct,
)


CRITERION_IDS = tuple(f"criterion_{index}" for index in range(5))


def _explicit_result(category: int) -> LLMJudgeResult:
    """Build one explicit-category result for the five-item panel."""
    categories = dict.fromkeys(CRITERION_IDS, category)
    scores = {criterion_id: category / 3 for criterion_id in CRITERION_IDS}
    return LLMJudgeResult(
        score=sum(scores.values()) / len(scores),
        accepted=True,
        rationale="panel-mode fixture",
        criterion_scores=scores,
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=categories,
        category_count=4,
    )


def _score_only_result(category: int) -> LLMJudgeResult:
    """Build the corresponding score-derived result without explicit categories."""
    return replace(
        _explicit_result(category),
        criterion_categories=None,
        category_count=None,
    )


def test_mixed_panel_rejects_before_projecting_first_mode_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A panel must not mix explicit categories with score-derived bins."""
    spec = validate_judge_construct(CRITERION_IDS, n_categories=4)
    original = judge_construct.project_row_in_order
    calls = 0

    def _counting_projection(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(judge_construct, "project_row_in_order", _counting_projection)

    with pytest.raises(JudgeFormatError, match="one category-generation mode"):
        project_judge_results_to_matrix(
            [_explicit_result(1), _score_only_result(2)],
            spec,
        )

    assert calls == 1


def test_mixed_panel_is_rejected_in_reverse_order() -> None:
    """Mode consistency must not depend on which generation mode appears first."""
    spec = validate_judge_construct(CRITERION_IDS, n_categories=4)
    with pytest.raises(JudgeFormatError, match="one category-generation mode"):
        project_judge_results_to_matrix(
            [_score_only_result(1), _explicit_result(2)],
            spec,
        )
