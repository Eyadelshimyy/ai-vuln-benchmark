"""Callback-safety regressions for IRT experiment-readiness controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from fast_mlsirm.irt_contract import validate_irt_experiment_readiness


_READY_BINARY = [
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
]


def _hostile_python_int(value: int, callbacks: list[str]) -> int:
    """Return a Python integer subclass whose callbacks must never execute."""

    class HostileInt(int):
        """Integer subclass exposing conversion and comparison callbacks."""

        def __int__(self) -> int:
            """Record forbidden integer conversion."""
            callbacks.append("__int__")
            raise AssertionError("caller integer conversion must not run")

        def __index__(self) -> int:
            """Record forbidden integer indexing."""
            callbacks.append("__index__")
            raise AssertionError("caller integer indexing must not run")

        def __lt__(self, other: object) -> bool:
            """Record forbidden lower-bound comparison."""
            callbacks.append("__lt__")
            raise AssertionError("caller integer comparison must not run")

        def __ge__(self, other: object) -> bool:
            """Record forbidden upper-bound comparison."""
            callbacks.append("__ge__")
            raise AssertionError("caller integer comparison must not run")

    return HostileInt(value)


def _hostile_numpy_int(value: int, callbacks: list[str]) -> np.integer:
    """Return a NumPy integer subclass whose callbacks must never execute."""

    class HostileNumpyInt(np.int64):
        """NumPy integer subclass exposing conversion and comparison callbacks."""

        def __int__(self) -> int:
            """Record forbidden NumPy-integer conversion."""
            callbacks.append("__int__")
            raise AssertionError("caller NumPy-integer conversion must not run")

        def __index__(self) -> int:
            """Record forbidden NumPy-integer indexing."""
            callbacks.append("__index__")
            raise AssertionError("caller NumPy-integer indexing must not run")

        def __lt__(self, other: object) -> bool:
            """Record forbidden lower-bound comparison."""
            callbacks.append("__lt__")
            raise AssertionError("caller NumPy-integer comparison must not run")

        def __ge__(self, other: object) -> bool:
            """Record forbidden upper-bound comparison."""
            callbacks.append("__ge__")
            raise AssertionError("caller NumPy-integer comparison must not run")

    return HostileNumpyInt(value)


@pytest.mark.parametrize(
    ("control_name", "valid_value", "extra_kwargs"),
    [
        ("min_persons", 5, {}),
        ("min_observed_per_item", 3, {}),
        ("min_item_distinct_values", 2, {}),
        ("min_items_per_factor", 1, {"factor_ids": ("g", "f")}),
    ],
)
@pytest.mark.parametrize("hostile_factory", [_hostile_python_int, _hostile_numpy_int])
def test_readiness_integer_controls_reject_subclasses_without_callbacks(
    control_name: str,
    valid_value: int,
    extra_kwargs: dict[str, object],
    hostile_factory: Callable[[int, list[str]], object],
) -> None:
    """Reject hostile integer subclasses before conversion or comparison callbacks."""
    callbacks: list[str] = []
    hostile_value = hostile_factory(valid_value, callbacks)
    kwargs = dict(extra_kwargs)
    kwargs[control_name] = hostile_value

    with pytest.raises(TypeError, match=control_name):
        validate_irt_experiment_readiness(
            _READY_BINARY,
            "dichotomous",
            **kwargs,
        )

    assert callbacks == []


def test_readiness_integer_controls_preserve_concrete_numpy_scalars() -> None:
    """Keep concrete NumPy integer scalars compatible with readiness validation."""
    matrix = validate_irt_experiment_readiness(
        _READY_BINARY,
        "dichotomous",
        min_persons=np.int64(5),
        min_observed_per_item=np.uint16(3),
        min_item_distinct_values=np.int32(2),
        factor_ids=("g", "f"),
        min_items_per_factor=np.uint8(1),
    )

    assert matrix.shape == (6, 2)


def test_readiness_integer_controls_preserve_minimum_domain_errors() -> None:
    """Preserve the package-owned minimum-value error for exact integers."""
    with pytest.raises(ValueError, match="min_persons must be at least 1"):
        validate_irt_experiment_readiness(
            _READY_BINARY,
            "dichotomous",
            min_persons=0,
        )


def test_readiness_integer_controls_do_not_dispatch_integer_protocols() -> None:
    """Reject arbitrary integer protocol providers without invoking callbacks."""
    callbacks: list[str] = []

    class IntegerProtocol:
        """Arbitrary integer-like object that must not cross the trust boundary."""

        def __int__(self) -> int:
            """Record forbidden arbitrary integer conversion."""
            callbacks.append("__int__")
            raise AssertionError("caller conversion must not run")

        def __index__(self) -> int:
            """Record forbidden arbitrary integer indexing."""
            callbacks.append("__index__")
            raise AssertionError("caller indexing must not run")

    with pytest.raises(TypeError, match="min_persons"):
        validate_irt_experiment_readiness(
            _READY_BINARY,
            "dichotomous",
            min_persons=IntegerProtocol(),
        )

    assert callbacks == []
