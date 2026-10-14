"""Fail-closed tests for public continuous-response-model control marshalling."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import crm


_RESPONSES = np.array([[0.25], [0.75]], dtype=np.float64)
_SUPPORTED_Q = (7, 11, 15, 21, 31, 41)


def _unexpected_core() -> object:
    """Fail if native-core discovery happens before CRM control validation."""

    raise AssertionError("compiled core discovered before CRM control validation")


def _result(tol: float = 1e-6) -> dict[str, object]:
    """Return the smallest shape-consistent trusted-core result fixture."""

    return {
        "slope": [1.0],
        "intercept": [0.0],
        "resid_sd": [1.0],
        "discrimination": [1.0],
        "difficulty": [0.0],
        "theta": [0.0, 0.0],
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "n_parameters": 3,
        "termination_reason": "tolerance",
        "final_delta": 0.0,
        "stopping_tolerance": tol,
    }


def test_hostile_scalar_controls_fail_before_callbacks_and_core(monkeypatch):
    """Rejected scalar subclasses execute neither callbacks nor native discovery."""

    class HostileMeta(type):
        def __eq__(cls, other):  # pragma: no cover - must never execute
            raise AssertionError("hostile type equality executed")

        def __hash__(cls):  # pragma: no cover - must never execute
            raise AssertionError("hostile type hash executed")

    class HostileInt(int, metaclass=HostileMeta):
        def __int__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile integer conversion executed")

        def __le__(self, other):  # pragma: no cover - must never execute
            raise AssertionError("hostile integer comparison executed")

        def __ge__(self, other):  # pragma: no cover - must never execute
            raise AssertionError("hostile integer comparison executed")

        def __repr__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile integer repr executed")

    class HostileNumpyInt(np.int64):
        def __int__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile NumPy integer conversion executed")

        def __repr__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile NumPy integer repr executed")

    class HostileFloat(float):
        def __float__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile float conversion executed")

        def __le__(self, other):  # pragma: no cover - must never execute
            raise AssertionError("hostile float comparison executed")

        def __ge__(self, other):  # pragma: no cover - must never execute
            raise AssertionError("hostile float comparison executed")

        def __array_ufunc__(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("hostile NumPy ufunc executed")

        def __repr__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile float repr executed")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        ({"q_theta": HostileInt(41)}, "q_theta"),
        ({"q_theta": HostileNumpyInt(41)}, "q_theta"),
        ({"max_iter": HostileInt(5)}, "max_iter"),
        ({"tol": HostileFloat(1e-6)}, "tol"),
    )
    for kwargs, field in cases:
        with pytest.raises(ValueError, match=field):
            crm.fit_crm(_RESPONSES, **kwargs)


def test_invalid_exact_controls_fail_before_core(monkeypatch):
    """Malformed trusted controls fail at the Python marshalling boundary."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        # #1929: no node-count cap; q_theta=9 is now accepted, only < 1 is not.
        ({"q_theta": 0}, "q_theta"),
        ({"q_theta": True}, "q_theta"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"tol": 0.0}, "tol"),
        ({"tol": -1.0}, "tol"),
        ({"tol": float("nan")}, "tol"),
        ({"tol": float("inf")}, "tol"),
        ({"tol": 10**400}, "tol"),
        ({"tol": True}, "tol"),
    )
    for kwargs, field in cases:
        with pytest.raises(ValueError, match=field):
            crm.fit_crm(_RESPONSES, **kwargs)


def test_genuine_numpy_controls_are_normalized_before_native_dispatch(monkeypatch):
    """Supported genuine NumPy scalars remain trusted after exact-type admission."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_crm(self, *args):
            captured["args"] = args
            return _result(float(args[6]))

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = crm.fit_crm(
        _RESPONSES,
        q_theta=np.uint16(21),
        max_iter=np.int32(3),
        tol=np.float32(1e-5),
    )

    args = captured["args"]
    assert type(args[4]) is int and args[4] == 21
    assert type(args[5]) is int and args[5] == 3
    assert type(args[6]) is float and args[6] > 0.0
    assert fitted.n_parameters == 3


def test_supported_quadrature_orders_reach_native_boundary(monkeypatch):
    """Python validation preserves every Gauss-Hermite order embedded by Rust."""

    captured: list[int] = []

    class CapturingCore:
        def fit_crm(self, *args):
            captured.append(args[4])
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    for q_theta in _SUPPORTED_Q:
        crm.fit_crm(_RESPONSES, q_theta=q_theta, max_iter=1)

    assert tuple(captured) == _SUPPORTED_Q
