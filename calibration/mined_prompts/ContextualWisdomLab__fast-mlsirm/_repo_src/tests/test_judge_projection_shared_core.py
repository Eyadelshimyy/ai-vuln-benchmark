"""Parity and delegation regressions for judge-result IRT projection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fast_mlsirm._judge_projection_order import project_row_in_order
from fast_mlsirm.llm_judge import JudgeFormatError, LLMJudgeResult


def _result() -> LLMJudgeResult:
    return LLMJudgeResult(
        score=0.5,
        accepted=True,
        rationale="shared projection fixture",
        criterion_scores={"criterion_alpha": 0.1, "criterion_beta": 0.9},
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
    )


def test_explicit_order_delegates_to_the_single_result_projection_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit-order adapter must not maintain a second binning implementation."""
    result = _result()
    original = LLMJudgeResult.to_irt_row
    calls = 0

    def _counting_core(self: LLMJudgeResult, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    monkeypatch.setattr(LLMJudgeResult, "to_irt_row", _counting_core)

    row = project_row_in_order(
        result,
        item_type="polytomous",
        n_categories=4,
        criterion_order=["criterion_beta", "criterion_alpha"],
    )

    assert calls == 1
    assert row == (3, 0)


def test_explicit_order_is_a_permutation_of_the_canonical_projection() -> None:
    """Changing item order changes only the output permutation, not category arithmetic."""
    result = _result()
    canonical = result.to_irt_row(item_type="polytomous", n_categories=4)
    explicit = project_row_in_order(
        result,
        item_type="polytomous",
        n_categories=4,
        criterion_order=("criterion_beta", "criterion_alpha"),
    )

    assert canonical == (0, 3)
    assert explicit == (canonical[1], canonical[0])


def test_explicit_category_path_uses_the_same_projection_core() -> None:
    """Direct judge categories retain the same semantics under explicit reordering."""
    result = replace(
        _result(),
        criterion_categories={"criterion_alpha": 1, "criterion_beta": 3},
        category_count=4,
    )

    assert result.to_irt_row(item_type="polytomous", n_categories=4) == (1, 3)
    assert project_row_in_order(
        result,
        item_type="polytomous",
        n_categories=4,
        criterion_order=["criterion_beta", "criterion_alpha"],
    ) == (3, 1)


def test_explicit_order_keeps_its_stricter_sealed_mapping_contract() -> None:
    """Delegation must not reopen protocol-bearing result mappings."""
    result = replace(_result(), criterion_scores={"criterion_alpha": 0.1})
    with pytest.raises(JudgeFormatError, match="exactly the rubric criterion ids"):
        project_row_in_order(
            result,
            item_type="polytomous",
            n_categories=4,
            criterion_order=["criterion_alpha", "criterion_beta"],
        )
