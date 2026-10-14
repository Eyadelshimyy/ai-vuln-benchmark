"""Security regressions for caller-controlled rubric integer normalization."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rubric import BlueprintPlan, RubricLevel


class _ExecutableIndex:
    """Expose integer-like callbacks whose execution is observable to the test."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __index__(self) -> int:
        self.calls.append("__index__")
        return 1

    def __repr__(self) -> str:
        self.calls.append("__repr__")
        return "1"

    def __eq__(self, other: object) -> bool:
        self.calls.append("__eq__")
        return False

    def __hash__(self) -> int:
        self.calls.append("__hash__")
        return 1

    def __lt__(self, other: object) -> bool:
        self.calls.append("__lt__")
        return False

    def __le__(self, other: object) -> bool:
        self.calls.append("__le__")
        return False

    def __gt__(self, other: object) -> bool:
        self.calls.append("__gt__")
        return False

    def __ge__(self, other: object) -> bool:
        self.calls.append("__ge__")
        return False


class _ExecutableInt(int):
    """Caller-defined Python integer subclass with observable numeric hooks."""

    calls: list[str] = []

    def __int__(self) -> int:
        type(self).calls.append("__int__")
        return 1

    def __index__(self) -> int:
        type(self).calls.append("__index__")
        return 1

    def __repr__(self) -> str:
        type(self).calls.append("__repr__")
        return "1"

    def __eq__(self, other: object) -> bool:
        type(self).calls.append("__eq__")
        return False

    def __hash__(self) -> int:
        type(self).calls.append("__hash__")
        return 1

    def __lt__(self, other: object) -> bool:
        type(self).calls.append("__lt__")
        return False

    def __le__(self, other: object) -> bool:
        type(self).calls.append("__le__")
        return False

    def __gt__(self, other: object) -> bool:
        type(self).calls.append("__gt__")
        return False

    def __ge__(self, other: object) -> bool:
        type(self).calls.append("__ge__")
        return False


class _ExecutableNumpyInt(np.int64):
    """Caller-defined NumPy integer subclass with observable numeric hooks."""

    calls: list[str] = []

    def __int__(self) -> int:
        type(self).calls.append("__int__")
        return 1

    def __index__(self) -> int:
        type(self).calls.append("__index__")
        return 1

    def __repr__(self) -> str:
        type(self).calls.append("__repr__")
        return "1"

    def __eq__(self, other: object) -> bool:
        type(self).calls.append("__eq__")
        return False

    def __hash__(self) -> int:
        type(self).calls.append("__hash__")
        return 1

    def __lt__(self, other: object) -> bool:
        type(self).calls.append("__lt__")
        return False

    def __le__(self, other: object) -> bool:
        type(self).calls.append("__le__")
        return False

    def __gt__(self, other: object) -> bool:
        type(self).calls.append("__gt__")
        return False

    def __ge__(self, other: object) -> bool:
        type(self).calls.append("__ge__")
        return False


def _level(score: object) -> RubricLevel:
    """Construct one otherwise-valid public rubric level."""
    return RubricLevel(score, "supported", "Supported evidence.", ("supported",))


def test_rubric_level_rejects_index_provider_before_callback() -> None:
    """Arbitrary index providers cannot execute during public construction."""
    value = _ExecutableIndex()

    with pytest.raises(ValueError, match="score must be an integer"):
        _level(value)

    assert value.calls == []


def test_rubric_level_rejects_python_integer_subclasses_before_callback() -> None:
    """Python integer subclasses fail before any caller numeric hook executes."""
    _ExecutableInt.calls.clear()
    value = _ExecutableInt(1)

    with pytest.raises(ValueError, match="score must be an integer"):
        _level(value)

    assert _ExecutableInt.calls == []


def test_blueprint_plan_rejects_numpy_integer_subclass_before_callback() -> None:
    """Shared integer controls reject NumPy subclasses before numeric dispatch."""
    _ExecutableNumpyInt.calls.clear()
    value = _ExecutableNumpyInt(2)

    with pytest.raises(ValueError, match="items_per_cell must be an integer"):
        BlueprintPlan(items_per_cell=value)

    assert _ExecutableNumpyInt.calls == []


@pytest.mark.parametrize("scalar_type", [np.int8, np.int64, np.uint16, np.uint64])
def test_public_rubric_controls_preserve_exact_numpy_integer_scalars(
    scalar_type: type[np.integer],
) -> None:
    """Genuine NumPy integer scalars normalize to inert built-in integers."""
    level = _level(scalar_type(1))
    plan = BlueprintPlan(items_per_cell=scalar_type(2), seed=scalar_type(3))

    assert level.score == 1
    assert type(level.score) is int
    assert plan.items_per_cell == 2
    assert type(plan.items_per_cell) is int
    assert plan.seed == 3
    assert type(plan.seed) is int
