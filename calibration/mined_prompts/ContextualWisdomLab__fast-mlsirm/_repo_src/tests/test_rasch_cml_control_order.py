"""Fail-before-materialization regressions for Rasch CML semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rasch_cml import andersen_lr_test, fit_rasch_cml


def _binary() -> np.ndarray:
    """Return a small valid complete response matrix for Andersen ordering tests."""

    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


class _HostileResponses:
    """Array provider that must remain untouched for rejected semantic controls."""

    calls = 0

    def __array__(self, dtype=None, copy=None):
        """Fail if rejected controls trigger response conversion."""
        type(self).calls += 1
        raise AssertionError("caller-owned response materialization executed")


class _HostileGroup:
    """Group provider that must remain untouched for rejected semantic controls."""

    calls = 0

    def __array__(self, dtype=None, copy=None):
        """Fail if rejected controls trigger group conversion."""
        type(self).calls += 1
        raise AssertionError("caller-owned group materialization executed")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_fit_rasch_cml_rejects_controls_before_response_materialization(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid controls fail before caller response conversion can execute."""

    _HostileResponses.calls = 0

    with pytest.raises(ValueError, match=message):
        fit_rasch_cml(_HostileResponses(), **{"max_iter": 100, "tol": 1e-8, **kwargs})

    assert _HostileResponses.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_andersen_rejects_controls_before_response_materialization(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Andersen controls fail before caller response conversion can execute."""

    _HostileResponses.calls = 0

    with pytest.raises(ValueError, match=message):
        andersen_lr_test(_HostileResponses(), np.array([0, 1]), **{"max_iter": 100, "tol": 1e-8, **kwargs})

    assert _HostileResponses.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_andersen_rejects_controls_before_group_materialization(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Andersen controls also precede caller-owned group conversion."""

    _HostileGroup.calls = 0

    with pytest.raises(ValueError, match=message):
        andersen_lr_test(_binary(), _HostileGroup(), **{"max_iter": 100, "tol": 1e-8, **kwargs})

    assert _HostileGroup.calls == 0
