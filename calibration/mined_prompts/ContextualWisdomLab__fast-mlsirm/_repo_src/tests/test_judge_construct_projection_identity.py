"""Regression tests for judge-construct identity and policy boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.judge_construct import (
    ABSOLUTE_JUDGE_CONSTRUCT_FLOOR,
    JudgeConstructPolicy,
    JudgeConstructSpec,
    JudgeFormatError,
    MAX_JUDGE_CONSTRUCT_ITEMS,
    ZERO_BASED_CATEGORY_CODING,
    project_judge_results_to_matrix,
    validate_judge_construct,
)
from fast_mlsirm.llm_judge import LLMJudgeResult


def _result(categories: dict[str, int], *, category_count: int = 4) -> LLMJudgeResult:
    """Build one explicit-category judge result for projection tests."""
    scores = {
        criterion_id: category / (category_count - 1)
        for criterion_id, category in categories.items()
    }
    return LLMJudgeResult(
        score=sum(scores.values()) / len(scores),
        accepted=True,
        rationale="projection identity regression",
        criterion_scores=scores,
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=categories,
        category_count=category_count,
    )


def test_projection_rejects_same_count_different_criterion_set() -> None:
    """Equal row width must not substitute for exact item identity."""
    spec_ids = tuple(f"criterion_{index}" for index in range(5))
    spec = validate_judge_construct(spec_ids, n_categories=4)
    result = _result({f"other_{index}": index % 4 for index in range(5)})

    with pytest.raises(JudgeFormatError, match="criterion set does not match"):
        project_judge_results_to_matrix([result], spec)


def test_projection_columns_follow_spec_criterion_order() -> None:
    """Matrix column j must always represent spec.criterion_ids[j]."""
    spec_ids = (
        "criterion_c",
        "criterion_a",
        "criterion_e",
        "criterion_b",
        "criterion_d",
    )
    spec = validate_judge_construct(spec_ids, n_categories=4)
    result = _result(
        {
            "criterion_a": 0,
            "criterion_b": 1,
            "criterion_c": 3,
            "criterion_d": 1,
            "criterion_e": 2,
        }
    )

    matrix = project_judge_results_to_matrix([result], spec)

    assert matrix.dtype == np.int64
    assert matrix.tolist() == [[3, 0, 2, 1, 1]]


def test_custom_policy_cannot_raise_hard_facet_ceiling() -> None:
    """Custom quality policy may tighten, but never exceed, the package cap."""
    with pytest.raises(
        ValueError,
        match=rf"max_items cannot exceed {MAX_JUDGE_CONSTRUCT_ITEMS}",
    ):
        JudgeConstructPolicy(
            min_items=5,
            recommended_items=7,
            max_items=MAX_JUDGE_CONSTRUCT_ITEMS + 1,
        )


def test_direct_spec_replays_global_item_and_category_domains() -> None:
    """The exported frozen spec must itself enforce the global handoff domain."""
    too_short = tuple(
        f"criterion_{index}" for index in range(ABSOLUTE_JUDGE_CONSTRUCT_FLOOR - 1)
    )
    too_long = tuple(
        f"criterion_{index}" for index in range(MAX_JUDGE_CONSTRUCT_ITEMS + 1)
    )

    with pytest.raises(ValueError, match="criterion_ids must contain 3..11 items"):
        JudgeConstructSpec(
            too_short,
            "polytomous",
            4,
            ZERO_BASED_CATEGORY_CODING,
            True,
        )
    with pytest.raises(ValueError, match="criterion_ids must contain 3..11 items"):
        JudgeConstructSpec(
            too_long,
            "polytomous",
            4,
            ZERO_BASED_CATEGORY_CODING,
            True,
        )
    with pytest.raises(ValueError, match="dichotomous specs require n_categories == 2"):
        JudgeConstructSpec(
            ("a", "b", "c"),
            "dichotomous",
            4,
            ZERO_BASED_CATEGORY_CODING,
            True,
        )
    with pytest.raises(ValueError, match="polytomous specs require n_categories"):
        JudgeConstructSpec(
            ("a", "b", "c"),
            "polytomous",
            None,
            ZERO_BASED_CATEGORY_CODING,
            True,
        )


def test_direct_spec_rejects_blank_criterion_identity() -> None:
    """Direct construction must match validated non-empty identifier semantics."""
    with pytest.raises(TypeError, match="non-empty tuple of strings"):
        JudgeConstructSpec(
            ("criterion_a", " ", "criterion_c"),
            "polytomous",
            4,
            ZERO_BASED_CATEGORY_CODING,
            True,
        )


def test_projection_replays_spec_invariants_after_slot_rebinding() -> None:
    """Frozen-record rebinding must not bypass the item-count boundary."""
    spec_ids = tuple(f"criterion_{index}" for index in range(5))
    spec = validate_judge_construct(spec_ids, n_categories=4)
    object.__setattr__(spec, "criterion_ids", spec_ids[:2])
    result = _result(
        {criterion_id: index for index, criterion_id in enumerate(spec_ids[:2])}
    )

    with pytest.raises(ValueError, match="criterion_ids must contain 3..11 items"):
        project_judge_results_to_matrix([result], spec)


def test_projection_replays_originating_custom_policy_after_slot_rebinding() -> None:
    """A tightened custom policy must remain part of the projected handoff."""
    spec_ids = tuple(f"criterion_{index}" for index in range(6))
    policy = JudgeConstructPolicy(min_items=6, recommended_items=7, max_items=9)
    spec = validate_judge_construct(spec_ids, n_categories=4, policy=policy)
    object.__setattr__(spec, "criterion_ids", spec_ids[:5])
    result = _result(
        {criterion_id: index % 4 for index, criterion_id in enumerate(spec_ids[:5])}
    )

    with pytest.raises(ValueError, match="meets_policy does not match policy bounds"):
        project_judge_results_to_matrix([result], spec)


def test_validated_spec_serializes_originating_custom_policy() -> None:
    """A reconstructable handoff must carry the policy that defined compliance."""
    spec_ids = tuple(f"criterion_{index}" for index in range(6))
    policy = JudgeConstructPolicy(min_items=6, recommended_items=7, max_items=9)

    spec = validate_judge_construct(spec_ids, n_categories=4, policy=policy)

    assert spec.policy == policy
    assert spec.policy is not policy
    assert spec.to_dict()["policy"] == policy.to_dict()


def test_projection_replays_originating_policy_invariants() -> None:
    """Frozen-policy rebinding must fail before projected row marshalling."""
    spec_ids = tuple(f"criterion_{index}" for index in range(6))
    spec = validate_judge_construct(
        spec_ids,
        n_categories=4,
        policy=JudgeConstructPolicy(min_items=6, recommended_items=7, max_items=9),
    )
    assert spec.policy is not None
    object.__setattr__(spec.policy, "min_items", 2)

    with pytest.raises(ValueError, match="min_items must be between"):
        project_judge_results_to_matrix([], spec)
