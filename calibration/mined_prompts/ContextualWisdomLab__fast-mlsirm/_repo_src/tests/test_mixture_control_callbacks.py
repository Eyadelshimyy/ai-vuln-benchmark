"""Fail-closed tests for public mixture-IRT control marshalling."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mixture


_RESPONSES = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _unexpected_core() -> object:
    """Fail if native-core discovery happens before control validation."""

    raise AssertionError("compiled core discovered before mixture control validation")


def _result(n_persons: int, n_items: int, n_classes: int) -> dict[str, object]:
    """Return the smallest shape-consistent trusted-core result fixture."""

    class_items = n_classes * n_items
    return {
        "model": "rasch",
        "n_classes": n_classes,
        "a": [1.0] * class_items,
        "b": [0.0] * class_items,
        "pi": [1.0 / n_classes] * n_classes,
        "class_posterior": [1.0 / n_classes] * (n_persons * n_classes),
        "map_class": [0] * n_persons,
        "theta": [0.0] * n_persons,
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "n_parameters": class_items + n_classes - 1,
    }


def test_hostile_scalar_controls_fail_before_callbacks_and_core(monkeypatch):
    """Rejected scalar subclasses must execute neither callbacks nor native discovery."""

    class HostileMeta(type):
        def __eq__(cls, other):  # pragma: no cover - must never execute
            raise AssertionError("hostile type equality executed")

        def __hash__(cls):  # pragma: no cover - must never execute
            raise AssertionError("hostile type hash executed")

    class HostileInt(int, metaclass=HostileMeta):
        def __int__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile integer conversion executed")

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

        def __repr__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile float repr executed")

    class HostileStr(str):
        def __str__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile string conversion executed")

        def lower(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile string normalization executed")

        def __repr__(self):  # pragma: no cover - must never execute
            raise AssertionError("hostile string repr executed")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        ({"n_classes": HostileInt(2)}, "n_classes"),
        ({"n_starts": HostileNumpyInt(1)}, "n_starts"),
        ({"max_iter": HostileInt(3)}, "max_iter"),
        ({"tol": HostileFloat(1e-6)}, "tol"),
        ({"model": HostileStr("rasch")}, "model"),
        ({"seed": HostileInt(7)}, "seed"),
    )
    for kwargs, field in cases:
        with pytest.raises(ValueError, match=field):
            mixture.fit_mixture(_RESPONSES, **{"model": "rasch", "n_starts": 1, "max_iter": 500, "tol": 1e-6, "seed": 0x2545F491, **kwargs})


def test_invalid_exact_controls_fail_before_core(monkeypatch):
    """Malformed trusted built-ins must fail at the Python marshalling boundary."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    cases = (
        ({"model": "not-a-mixture-model"}, "model"),
        ({"tol": -1.0}, "tol"),
        ({"tol": float("nan")}, "tol"),
        ({"tol": float("inf")}, "tol"),
        ({"tol": True}, "tol"),
        ({"seed": -1}, "seed"),
        ({"seed": 1 << 64}, "seed"),
        ({"seed": True}, "seed"),
    )
    for kwargs, field in cases:
        with pytest.raises(ValueError, match=field):
            mixture.fit_mixture(_RESPONSES, **{"model": "rasch", "n_starts": 1, "max_iter": 500, "tol": 1e-6, "seed": 0x2545F491, **kwargs})


def test_genuine_numpy_controls_are_normalized_before_native_dispatch(monkeypatch):
    """Supported genuine NumPy scalars remain accepted as trusted controls."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_mixture(self, *args):
            captured["args"] = args
            return _result(n_persons=2, n_items=2, n_classes=2)

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = mixture.fit_mixture(
        _RESPONSES,
        n_classes=np.int64(2),
        model="2PL",
        n_starts=np.uint16(1),
        max_iter=np.int32(3),
        tol=np.float32(0.0),
        seed=np.uint64(7),
    )

    args = captured["args"]
    assert type(args[4]) is int and args[4] == 2
    assert type(args[5]) is str and args[5] == "2PL"
    assert type(args[6]) is int and args[6] == 1
    assert type(args[7]) is int and args[7] == 3
    assert type(args[8]) is float and args[8] == 0.0
    assert type(args[9]) is int and args[9] == 7
    assert fitted.n_classes == 2


def test_existing_model_aliases_remain_supported(monkeypatch):
    """Python validation must preserve every alias accepted by the Rust binding."""

    captured: list[str] = []

    class CapturingCore:
        def fit_mixture(self, *args):
            captured.append(args[5])
            return _result(n_persons=2, n_items=2, n_classes=2)

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    aliases = ("rasch", "Rasch", "RASCH", "2pl", "2PL", "twopl", "TwoPl")
    for alias in aliases:
        mixture.fit_mixture(_RESPONSES, model=alias, n_starts=1, max_iter=500, tol=1e-6, seed=0x2545F491)

    assert tuple(captured) == aliases
