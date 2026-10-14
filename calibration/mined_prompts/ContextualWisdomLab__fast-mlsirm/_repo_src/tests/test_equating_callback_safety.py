"""Regression tests for inert observed-score equating controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.equating as E
import fast_mlsirm.fitstats as fitstats

_TOTAL = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_ANCHOR = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_COUNTS = np.array([1.0, 2.0, 1.0], dtype=np.float64)


class _ExecutableProvider:
    """Record any caller-controlled protocol callback that validation executes."""

    def __init__(self) -> None:
        self.callbacks: list[str] = []

    def _fail(self, callback: str):
        """Record one forbidden callback and stop the call immediately."""
        self.callbacks.append(callback)
        raise AssertionError(f"CALLBACK_MUST_NOT_RUN:{callback}")

    def __str__(self) -> str:
        return self._fail("__str__")

    def __repr__(self) -> str:
        return self._fail("__repr__")

    def __int__(self) -> int:
        return self._fail("__int__")

    def __index__(self) -> int:
        return self._fail("__index__")

    def __float__(self) -> float:
        return self._fail("__float__")

    def __eq__(self, other: object) -> bool:
        del other
        return self._fail("__eq__")

    def __hash__(self) -> int:
        return self._fail("__hash__")

    def __lt__(self, other: object) -> bool:
        del other
        return self._fail("__lt__")

    def __le__(self, other: object) -> bool:
        del other
        return self._fail("__le__")

    def __gt__(self, other: object) -> bool:
        del other
        return self._fail("__gt__")

    def __ge__(self, other: object) -> bool:
        del other
        return self._fail("__ge__")


class _StringSubclass(str):
    """String subclass whose normalization callback must remain inert."""

    callbacks: list[str] = []

    def __str__(self) -> str:
        type(self).callbacks.append("__str__")
        raise AssertionError("STRING_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _IntegerSubclass(int):
    """Python integer subclass whose conversion callbacks must remain inert."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("INTEGER_SUBCLASS_CALLBACK_MUST_NOT_RUN")

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        raise AssertionError("INTEGER_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _FloatSubclass(float):
    """Python float subclass whose conversion callback must remain inert."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        raise AssertionError("FLOAT_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _NumpyIntegerSubclass(np.int64):
    """NumPy integer subclass whose conversion callbacks must remain inert."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("NUMPY_INTEGER_SUBCLASS_CALLBACK_MUST_NOT_RUN")

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        raise AssertionError("NUMPY_INTEGER_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _NumpyFloatSubclass(np.float64):
    """NumPy float subclass whose conversion callback must remain inert."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        raise AssertionError("NUMPY_FLOAT_SUBCLASS_CALLBACK_MUST_NOT_RUN")


_STRING_CASES = (
    "equate_neat.method",
    "equate_neat_linear.method",
    "equate_neat_linear.anchor_kind",
    "kernel.continuization",
    "see.method",
    "see.route",
)
_INTEGER_CASES = (
    "equate_neat.k_x",
    "equate_neat.k_y",
    "equate_neat.k_v",
    "equate_neat_linear.k_x",
    "equate_neat_linear.k_y",
    "loglinear.degree",
    "kernel.k_x",
    "kernel.k_y",
    "kernel.smooth_x",
    "kernel.smooth_y",
    "see.k_x",
    "see.k_y",
    "see.n_boot",
    "see.seed",
)
_REAL_CASES = (
    "equate_neat.w1",
    "equate_neat_linear.w1",
    "kernel.bandwidth_x",
    "kernel.bandwidth_y",
    "see.ci_level",
)
_ALL_CASES = _STRING_CASES + _INTEGER_CASES + _REAL_CASES


def _invoke(case: str, value: object) -> object:
    """Invoke one public equating adapter with exactly one selected control value."""
    if case.startswith("equate_neat."):
        kwargs: dict[str, object] = {
            "method": "chained",
            "k_x": 2,
            "k_y": 2,
            "k_v": 2,
            "w1": 0.5,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equate_neat(_TOTAL, _ANCHOR, _TOTAL, _ANCHOR, **kwargs)
    if case.startswith("equate_neat_linear."):
        kwargs = {
            "method": "tucker",
            "anchor_kind": "internal",
            "k_x": 2,
            "k_y": 2,
            "w1": 0.5,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equate_neat_linear(_TOTAL, _ANCHOR, _TOTAL, _ANCHOR, **kwargs)
    if case == "loglinear.degree":
        return E.loglinear_smooth(_COUNTS, degree=value)
    if case.startswith("kernel."):
        kwargs = {
            "continuization": "gaussian",
            "k_x": 2,
            "k_y": 2,
            "smooth_x": 1,
            "smooth_y": 1,
            "bandwidth_x": 0.5,
            "bandwidth_y": 0.5,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equate_observed_scores_kernel(_TOTAL, _TOTAL, **kwargs)
    if case.startswith("see."):
        kwargs = {
            "method": "mean",
            "route": "bootstrap",
            "k_x": 2,
            "k_y": 2,
            "n_boot": 2,
            "ci_level": 0.95,
            "seed": 0,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equating_standard_errors(_TOTAL, _TOTAL, **kwargs)
    raise AssertionError(f"unhandled test case: {case}")


def _forbid_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace Rust discovery with a recorder that proves validation ordering."""
    calls: list[str] = []

    def forbidden_core():
        calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)
    return calls


@pytest.mark.parametrize("case", _ALL_CASES)
def test_arbitrary_protocol_providers_are_inert_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Arbitrary providers fail without callbacks or Rust-core discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()

    with pytest.raises(ValueError, match=case.rsplit(".", 1)[1]):
        _invoke(case, provider)

    assert provider.callbacks == []
    assert core_calls == []


@pytest.mark.parametrize("case", _STRING_CASES)
def test_string_subclasses_are_inert_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Python string subclasses are not normalized through caller callbacks."""
    core_calls = _forbid_core_discovery(monkeypatch)
    _StringSubclass.callbacks.clear()

    with pytest.raises(ValueError, match=case.rsplit(".", 1)[1]):
        _invoke(case, _StringSubclass("chained"))

    assert _StringSubclass.callbacks == []
    assert core_calls == []


@pytest.mark.parametrize("case", _INTEGER_CASES)
@pytest.mark.parametrize(
    "factory, callback_log",
    [
        (_IntegerSubclass, _IntegerSubclass.callbacks),
        (_NumpyIntegerSubclass, _NumpyIntegerSubclass.callbacks),
    ],
    ids=["python-int-subclass", "numpy-int-subclass"],
)
def test_integer_subclasses_are_inert_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    factory: Callable[[int], object],
    callback_log: list[str],
) -> None:
    """Python and NumPy integer subclasses are rejected without conversion."""
    core_calls = _forbid_core_discovery(monkeypatch)
    callback_log.clear()

    with pytest.raises(ValueError, match=case.rsplit(".", 1)[1]):
        _invoke(case, factory(2))

    assert callback_log == []
    assert core_calls == []


@pytest.mark.parametrize("case", _REAL_CASES)
@pytest.mark.parametrize(
    "factory, callback_log",
    [
        (_FloatSubclass, _FloatSubclass.callbacks),
        (_NumpyFloatSubclass, _NumpyFloatSubclass.callbacks),
    ],
    ids=["python-float-subclass", "numpy-float-subclass"],
)
def test_float_subclasses_are_inert_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    factory: Callable[[float], object],
    callback_log: list[str],
) -> None:
    """Python and NumPy float subclasses are rejected without conversion."""
    core_calls = _forbid_core_discovery(monkeypatch)
    callback_log.clear()

    with pytest.raises(ValueError, match=case.rsplit(".", 1)[1]):
        _invoke(case, factory(0.5))

    assert callback_log == []
    assert core_calls == []


def _equate_payload() -> dict[str, object]:
    """Return the minimum successful Rust-shaped equating payload."""
    return {
        "x_scores": [0.0, 1.0, 2.0],
        "y_equivalents": [0.0, 1.0, 2.0],
        "mu_x": 1.0,
        "sigma_x": 1.0,
        "mu_y": 1.0,
        "sigma_y": 1.0,
        "mu_eq": 1.0,
        "sigma_eq": 1.0,
        "slope": 1.0,
        "intercept": 0.0,
        "n_x": 3,
        "n_y": 3,
        "h_x": 0.5,
        "h_y": 0.5,
    }


def test_genuine_numpy_scalars_preserve_all_control_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact NumPy numeric scalars normalize to inert Python primitives."""
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class Core:
        """Capture normalized arguments at each Rust-shaped boundary."""

        def equate_neat(self, *args, **kwargs):
            calls.append(("neat", args, kwargs))
            return _equate_payload()

        def equate_neat_linear(self, *args, **kwargs):
            calls.append(("linear", args, kwargs))
            return _equate_payload()

        def loglinear_smooth(self, *args, **kwargs):
            calls.append(("loglinear", args, kwargs))
            return {
                "probs": [0.25, 0.5, 0.25],
                "log_lik": -1.0,
                "aic": 4.0,
                "bic": 4.0,
                "moments": [0.5],
                "converged": True,
                "iters": 1,
                "termination_reason": "gradient_tolerance",
                "final_gradient_max": 0.0,
                "gradient_tolerance": 1e-8,
            }

        def equate_observed_scores_ext(self, *args, **kwargs):
            calls.append(("kernel", args, kwargs))
            return _equate_payload()

        def bootstrap_see(self, *args, **kwargs):
            calls.append(("see", args, kwargs))
            return {
                "x_scores": [0.0, 1.0, 2.0],
                "y_equivalents": [0.0, 1.0, 2.0],
                "se": [0.1, 0.1, 0.1],
                "ci_lo": [0.0, 0.9, 1.9],
                "ci_hi": [0.1, 1.1, 2.1],
                "n_boot": 2,
                "ci_level": 0.95,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    E.equate_neat(
        _TOTAL,
        _ANCHOR,
        _TOTAL,
        _ANCHOR,
        method="frequency_estimation",
        k_x=np.int64(2),
        k_y=np.int32(2),
        k_v=np.uint16(2),
        w1=np.float32(0.5),
    )
    E.equate_neat_linear(
        _TOTAL,
        _ANCHOR,
        _TOTAL,
        _ANCHOR,
        method="tucker",
        anchor_kind="internal",
        k_x=np.int64(2),
        k_y=np.int32(2),
        w1=np.float64(0.5),
    )
    E.loglinear_smooth(_COUNTS, degree=np.int16(1))
    E.equate_observed_scores_kernel(
        _TOTAL,
        _TOTAL,
        continuization="gaussian",
        k_x=np.int64(2),
        k_y=np.int32(2),
        smooth_x=np.int16(1),
        smooth_y=np.uint16(1),
        bandwidth_x=np.float32(0.5),
        bandwidth_y=np.float64(0.5),
    )
    E.equating_standard_errors(
        _TOTAL,
        _TOTAL,
        method="mean",
        route="bootstrap",
        k_x=np.int64(2),
        k_y=np.int32(2),
        n_boot=np.int16(2),
        ci_level=np.float64(0.95),
        seed=np.uint16(0),
    )

    assert [name for name, _, _ in calls] == [
        "neat",
        "linear",
        "loglinear",
        "kernel",
        "see",
    ]
    for _, args, kwargs in calls:
        for value in (*args[4:], *kwargs.values()):
            if isinstance(value, (str, type(None), np.ndarray)):
                continue
            assert type(value) in (int, float)
