"""Trust-boundary regressions for judge projection mapping evidence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from fast_mlsirm.judge_construct import (
    JudgeFormatError,
    LLMJudgeResult,
    project_judge_results_to_matrix,
    validate_judge_construct,
)


CRITERIA = tuple(f"criterion_{index}" for index in range(5))


class _HostileMapping(Mapping[str, float]):
    """Mapping provider that fails if package validation executes callbacks."""

    def __init__(self) -> None:
        self.iter_calls = 0
        self.getitem_calls = 0
        self.len_calls = 0

    def __getitem__(self, key: str) -> float:
        self.getitem_calls += 1
        raise AssertionError(f"caller __getitem__ executed for {key!r}")

    def __iter__(self) -> Iterator[str]:
        self.iter_calls += 1
        raise AssertionError("caller __iter__ executed")

    def __len__(self) -> int:
        self.len_calls += 1
        raise AssertionError("caller __len__ executed")


def _result(
    *,
    criterion_scores: Mapping[str, float],
    criterion_categories: Mapping[str, int] | None = None,
) -> LLMJudgeResult:
    return LLMJudgeResult(
        score=0.5,
        accepted=True,
        rationale="mapping admission regression",
        criterion_scores=criterion_scores,
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=criterion_categories,
        category_count=4 if criterion_categories is not None else None,
    )


def test_projection_rejects_callback_bearing_score_mapping_before_iteration() -> None:
    hostile = _HostileMapping()
    spec = validate_judge_construct(CRITERIA, n_categories=4)

    with pytest.raises(JudgeFormatError, match="criterion_scores must be a plain dict keyed by criterion id"):
        project_judge_results_to_matrix(
            [_result(criterion_scores=hostile)],
            spec,
        )

    assert hostile.iter_calls == 0
    assert hostile.getitem_calls == 0
    assert hostile.len_calls == 0


def test_projection_rejects_callback_bearing_category_mapping_before_iteration() -> None:
    hostile = _HostileMapping()
    scores = {criterion_id: 0.5 for criterion_id in CRITERIA}
    spec = validate_judge_construct(CRITERIA, n_categories=4)

    with pytest.raises(JudgeFormatError, match="criterion_categories must be a plain dict keyed by criterion id"):
        project_judge_results_to_matrix(
            [
                _result(
                    criterion_scores=scores,
                    criterion_categories=hostile,
                )
            ],
            spec,
        )

    assert hostile.iter_calls == 0
    assert hostile.getitem_calls == 0
    assert hostile.len_calls == 0


def test_projection_preserves_exact_dict_mapping_compatibility() -> None:
    scores = {criterion_id: 0.5 for criterion_id in CRITERIA}
    categories = {criterion_id: index % 4 for index, criterion_id in enumerate(CRITERIA)}
    spec = validate_judge_construct(CRITERIA, n_categories=4)

    matrix = project_judge_results_to_matrix(
        [
            _result(
                criterion_scores=scores,
                criterion_categories=categories,
            )
        ],
        spec,
    )

    assert matrix.tolist() == [[0, 1, 2, 3, 0]]
