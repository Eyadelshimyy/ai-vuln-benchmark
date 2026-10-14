"""Semantic-control admission regressions for judge construct validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.judge_construct import JudgeFormatError, validate_judge_construct


class _HostileCriteria:
    """Iterable whose execution proves control validation happened too late."""

    def __init__(self) -> None:
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        raise AssertionError("criterion evidence must not be iterated for invalid controls")


class _HostileBool:
    """Truth-value provider that must never execute during admission."""

    def __init__(self) -> None:
        self.bool_calls = 0

    def __bool__(self) -> bool:
        self.bool_calls += 1
        raise AssertionError("allow_short_form must not execute caller truthiness")


def test_invalid_item_type_fails_before_criterion_iteration() -> None:
    criteria = _HostileCriteria()

    with pytest.raises(JudgeFormatError, match="item_type must be"):
        validate_judge_construct(criteria, item_type="graded", n_categories=4)

    assert criteria.iter_calls == 0


def test_invalid_category_count_fails_before_criterion_iteration() -> None:
    criteria = _HostileCriteria()

    with pytest.raises(JudgeFormatError, match="n_categories in"):
        validate_judge_construct(criteria, item_type="polytomous", n_categories=1)

    assert criteria.iter_calls == 0


def test_allow_short_form_rejects_callback_bearing_truth_value_before_data() -> None:
    criteria = _HostileCriteria()
    hostile = _HostileBool()

    with pytest.raises(TypeError, match="allow_short_form must be a boolean"):
        validate_judge_construct(
            criteria,
            item_type="polytomous",
            n_categories=4,
            allow_short_form=hostile,  # type: ignore[arg-type]
        )

    assert hostile.bool_calls == 0
    assert criteria.iter_calls == 0


def test_numpy_boolean_short_form_control_is_normalized_without_data_callbacks() -> None:
    spec = validate_judge_construct(
        ("criterion_a", "criterion_b", "criterion_c", "criterion_d"),
        item_type="polytomous",
        n_categories=4,
        allow_short_form=np.bool_(True),  # type: ignore[arg-type]
    )

    assert spec.meets_policy is False
    assert any("short form admitted" in warning for warning in spec.warnings)
