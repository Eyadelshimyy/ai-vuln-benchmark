from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar, overload

import pytest

from fast_mlsirm.model_validation import (
    GeneralizationUnit,
    ModelValidationPlan,
    ValidationStrategy,
    validate_group_partition,
    validate_temporal_forward_window,
)


_T = TypeVar("_T")


class _SinglePassSequence(Sequence[_T]):
    """Expose stable data once and fail if validation re-enumerates caller state."""

    def __init__(self, values: tuple[_T, ...]) -> None:
        self._values = values
        self.iteration_count = 0

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_T]: ...

    def __getitem__(self, index: int | slice) -> _T | Sequence[_T]:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[_T]:
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise RuntimeError("caller sequence was re-enumerated")
        return iter(self._values)


class _HostileStripString(str):
    """String subclass whose normalization callback must never execute."""

    def strip(self, chars: str | None = None) -> str:
        raise RuntimeError("caller-controlled identity callback leaked")


def test_group_holdout_requires_a_declared_noncell_generalization_unit() -> None:
    plan = ModelValidationPlan(
        strategy=ValidationStrategy.GROUP_HOLDOUT,
        unit=GeneralizationUnit.QUERY_TESTLET,
    )

    assert plan.strategy is ValidationStrategy.GROUP_HOLDOUT
    assert plan.unit is GeneralizationUnit.QUERY_TESTLET
    assert "response_cell" not in {unit.value for unit in GeneralizationUnit}


def test_temporal_forward_requires_temporal_period_unit() -> None:
    ModelValidationPlan(
        strategy=ValidationStrategy.TEMPORAL_FORWARD,
        unit=GeneralizationUnit.TEMPORAL_PERIOD,
    )

    with pytest.raises(
        ValueError,
        match="temporal_forward validation requires the temporal_period unit",
    ):
        ModelValidationPlan(
            strategy=ValidationStrategy.TEMPORAL_FORWARD,
            unit=GeneralizationUnit.PERSON_SYSTEM,
        )


def test_plan_rejects_truthy_substitutes_for_closed_enums() -> None:
    with pytest.raises(TypeError, match="strategy must be a ValidationStrategy"):
        ModelValidationPlan(  # type: ignore[arg-type]
            strategy="group_holdout",
            unit=GeneralizationUnit.CLUSTER_CONTEXT,
        )

    with pytest.raises(TypeError, match="unit must be a GeneralizationUnit"):
        ModelValidationPlan(  # type: ignore[arg-type]
            strategy=ValidationStrategy.GROUP_HOLDOUT,
            unit="cluster_context",
        )


def test_group_partition_rejects_leakage_across_folds() -> None:
    with pytest.raises(
        ValueError,
        match="generalization group 'query-a' appears in multiple folds",
    ):
        validate_group_partition(
            group_ids=("query-a", "query-b", "query-a"),
            fold_ids=("fold-1", "fold-1", "fold-2"),
        )


def test_group_partition_accepts_repeated_groups_within_one_fold() -> None:
    validate_group_partition(
        group_ids=("query-a", "query-a", "query-b", "query-c", "query-c"),
        fold_ids=("fold-1", "fold-1", "fold-2", "fold-3", "fold-3"),
    )


def test_group_partition_snapshots_caller_sequences_once() -> None:
    """Leakage checks operate on one stable caller snapshot, not live callbacks."""
    group_ids = _SinglePassSequence(("query-a", "query-b"))
    fold_ids = _SinglePassSequence(("fold-1", "fold-2"))

    validate_group_partition(group_ids=group_ids, fold_ids=fold_ids)

    assert group_ids.iteration_count == 1
    assert fold_ids.iteration_count == 1


def test_group_partition_rejects_string_subclasses_before_callbacks() -> None:
    """Caller-defined identity callbacks cannot cross the validation boundary."""
    hostile_group = _HostileStripString("query-a")
    with pytest.raises(
        ValueError,
        match="group_ids entries must be non-empty strings",
    ) as group_error:
        validate_group_partition(
            group_ids=(hostile_group, "query-b"),
            fold_ids=("fold-1", "fold-2"),
        )
    assert "caller-controlled" not in str(group_error.value)

    hostile_fold = _HostileStripString("fold-1")
    with pytest.raises(
        ValueError,
        match="fold_ids entries must be non-empty strings",
    ) as fold_error:
        validate_group_partition(
            group_ids=("query-a", "query-b"),
            fold_ids=(hostile_fold, "fold-2"),
        )
    assert "caller-controlled" not in str(fold_error.value)


def test_group_partition_rejects_malformed_identity_vectors() -> None:
    with pytest.raises(ValueError, match="group_ids and fold_ids must have equal length"):
        validate_group_partition(
            group_ids=("query-a",),
            fold_ids=("fold-1", "fold-2"),
        )

    with pytest.raises(ValueError, match="group_ids must not be empty"):
        validate_group_partition(group_ids=(), fold_ids=())

    with pytest.raises(ValueError, match="group_ids entries must be non-empty strings"):
        validate_group_partition(group_ids=("",), fold_ids=("fold-1",))

    with pytest.raises(ValueError, match="group_ids entries must be non-empty strings"):
        validate_group_partition(
            group_ids=("query-a", " \t"),
            fold_ids=("fold-1", "fold-2"),
        )

    with pytest.raises(ValueError, match="fold_ids entries must be non-empty strings"):
        validate_group_partition(group_ids=("query-a",), fold_ids=("",))

    with pytest.raises(ValueError, match="fold_ids entries must be non-empty strings"):
        validate_group_partition(
            group_ids=("query-a", "query-b"),
            fold_ids=("fold-1", " \t"),
        )


def test_group_partition_rejects_surrounding_identity_whitespace() -> None:
    """Presentation whitespace cannot manufacture distinct groups or folds."""
    with pytest.raises(
        ValueError,
        match="group_ids entries must not contain surrounding whitespace",
    ):
        validate_group_partition(
            group_ids=("query-a", " query-a ", "query-b"),
            fold_ids=("fold-1", "fold-2", "fold-2"),
        )

    with pytest.raises(
        ValueError,
        match="fold_ids entries must not contain surrounding whitespace",
    ):
        validate_group_partition(
            group_ids=("query-a", "query-b"),
            fold_ids=("fold-1", " fold-1 "),
        )


def test_group_partition_rejects_degenerate_single_group_or_fold() -> None:
    with pytest.raises(
        ValueError,
        match="group partition requires at least two generalization groups",
    ):
        validate_group_partition(
            group_ids=("query-a", "query-a"),
            fold_ids=("fold-1", "fold-1"),
        )

    with pytest.raises(
        ValueError,
        match="group partition requires at least two folds",
    ):
        validate_group_partition(
            group_ids=("query-a", "query-b"),
            fold_ids=("fold-1", "fold-1"),
        )


def test_group_partition_rejects_scalar_strings_as_identity_vectors() -> None:
    with pytest.raises(
        TypeError,
        match="group_ids and fold_ids must be identity sequences, not strings",
    ):
        validate_group_partition(group_ids="ab", fold_ids="12")  # type: ignore[arg-type]


def test_group_partition_is_order_invariant() -> None:
    groups = ("query-a", "query-b", "query-c", "query-a")
    folds = ("fold-1", "fold-2", "fold-2", "fold-1")
    validate_group_partition(group_ids=groups, fold_ids=folds)

    order = (3, 2, 1, 0)
    validate_group_partition(
        group_ids=tuple(groups[index] for index in order),
        fold_ids=tuple(folds[index] for index in order),
    )


def test_temporal_forward_window_requires_strictly_future_validation_periods() -> None:
    validate_temporal_forward_window(
        training_periods=(202601, 202602, 202603),
        validation_periods=(202604, 202605),
    )

    with pytest.raises(
        ValueError,
        match="training periods must strictly precede validation periods",
    ):
        validate_temporal_forward_window(
            training_periods=(202601, 202604),
            validation_periods=(202604, 202605),
        )


def test_temporal_forward_window_snapshots_period_sequences_once() -> None:
    """Temporal ordering is evaluated against one stable caller snapshot."""
    training = _SinglePassSequence((202601, 202602))
    validation = _SinglePassSequence((202603, 202604))

    validate_temporal_forward_window(
        training_periods=training,
        validation_periods=validation,
    )

    assert training.iteration_count == 1
    assert validation.iteration_count == 1


def test_temporal_forward_window_rejects_malformed_period_vectors() -> None:
    with pytest.raises(ValueError, match="training_periods must not be empty"):
        validate_temporal_forward_window(
            training_periods=(),
            validation_periods=(202604,),
        )

    with pytest.raises(ValueError, match="validation_periods must not be empty"):
        validate_temporal_forward_window(
            training_periods=(202603,),
            validation_periods=(),
        )

    with pytest.raises(ValueError, match="training_periods entries must be integers"):
        validate_temporal_forward_window(
            training_periods=(True,),
            validation_periods=(202604,),
        )

    with pytest.raises(ValueError, match="validation_periods entries must be integers"):
        validate_temporal_forward_window(
            training_periods=(202603,),
            validation_periods=(202604.0,),  # type: ignore[arg-type]
        )

    with pytest.raises(
        TypeError,
        match="training_periods and validation_periods must be integer sequences",
    ):
        validate_temporal_forward_window(
            training_periods="202603",  # type: ignore[arg-type]
            validation_periods=(202604,),
        )
