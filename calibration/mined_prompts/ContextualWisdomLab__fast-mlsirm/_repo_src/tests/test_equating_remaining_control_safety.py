"""Regression tests for remaining observed-score equating control boundaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.equating as E
import fast_mlsirm.fitstats as fitstats

_TOTAL = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_POINT_LOW = (0.0, 0.0)
_POINT_MID = (1.0, 1.0)
_POINT_HIGH = (2.0, 2.0)


class _ExecutableProvider:
    """Record every caller-controlled conversion/comparison callback."""

    def __init__(self) -> None:
        self.callbacks: list[str] = []

    def _fail(self, callback: str):
        """Record one forbidden callback and abort if validation executes it."""
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
    """String subclass whose callback must remain inert."""

    callbacks: list[str] = []

    def __str__(self) -> str:
        type(self).callbacks.append("__str__")
        raise AssertionError("STRING_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _IntegerSubclass(int):
    """Integer subclass whose conversion callbacks must remain inert."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("INTEGER_SUBCLASS_CALLBACK_MUST_NOT_RUN")

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        raise AssertionError("INTEGER_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _FloatSubclass(float):
    """Float subclass whose conversion callback must remain inert."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        raise AssertionError("FLOAT_SUBCLASS_CALLBACK_MUST_NOT_RUN")


class _TupleSubclass(tuple):
    """Tuple subclass whose iteration callback must remain inert."""

    callbacks: list[str] = []

    def __iter__(self):
        type(self).callbacks.append("__iter__")
        raise AssertionError("TUPLE_SUBCLASS_CALLBACK_MUST_NOT_RUN")


def _forbid_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace native discovery with a recorder for validation-order assertions."""
    calls: list[str] = []

    def forbidden_core():
        calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)
    return calls


def test_circle_arc_method_provider_is_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject executable method providers before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()

    with pytest.raises(ValueError, match="method"):
        E.circle_arc_equate(
            _TOTAL,
            _POINT_LOW,
            _POINT_MID,
            _POINT_HIGH,
            method=provider,
        )

    assert provider.callbacks == []
    assert core_calls == []


def test_circle_arc_string_subclass_is_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject string subclasses without invoking their conversion callback."""
    core_calls = _forbid_core_discovery(monkeypatch)
    _StringSubclass.callbacks.clear()

    with pytest.raises(ValueError, match="method"):
        E.circle_arc_equate(
            _TOTAL,
            _POINT_LOW,
            _POINT_MID,
            _POINT_HIGH,
            method=_StringSubclass("arc2"),
        )

    assert _StringSubclass.callbacks == []
    assert core_calls == []


def test_circle_arc_endpoint_provider_is_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject executable endpoint scalars before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()

    with pytest.raises(ValueError, match="low"):
        E.circle_arc_equate(
            _TOTAL,
            (provider, 0.0),
            _POINT_MID,
            _POINT_HIGH,
        )

    assert provider.callbacks == []
    assert core_calls == []


def test_circle_arc_endpoint_container_subclass_is_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject executable endpoint containers before iterating them."""
    core_calls = _forbid_core_discovery(monkeypatch)
    _TupleSubclass.callbacks.clear()

    with pytest.raises(ValueError, match="low"):
        E.circle_arc_equate(
            _TOTAL,
            _TupleSubclass((0.0, 0.0)),
            _POINT_MID,
            _POINT_HIGH,
        )

    assert _TupleSubclass.callbacks == []
    assert core_calls == []


@pytest.mark.parametrize("name", ["m_xa", "m_va", "m_yb", "s_yb", "m_vb", "s_vb"])
def test_circle_arc_middle_scalar_providers_are_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Reject each executable middle-anchor scalar before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()
    kwargs: dict[str, object] = {
        "m_xa": 1.0,
        "m_va": 0.5,
        "m_yb": 1.0,
        "s_yb": 0.5,
        "m_vb": 0.5,
        "s_vb": 0.5,
    }
    kwargs[name] = provider

    with pytest.raises(ValueError, match=name):
        E.circle_arc_middle_anchor(**kwargs)

    assert provider.callbacks == []
    assert core_calls == []


@pytest.mark.parametrize("name", ["k_x", "k_y", "k_v"])
def test_nominal_weight_integer_providers_are_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Reject executable score ceilings before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()
    kwargs: dict[str, object] = {"k_x": 2, "k_y": 2, "k_v": 2, "w1": 0.5}
    kwargs[name] = provider

    with pytest.raises(ValueError, match=name):
        E.nominal_weights_mean_equate(
            _TOTAL,
            _TOTAL,
            _TOTAL,
            _TOTAL,
            **kwargs,
        )

    assert provider.callbacks == []
    assert core_calls == []


def test_nominal_weight_float_provider_is_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject executable synthetic-population weights before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()

    with pytest.raises(ValueError, match="w1"):
        E.nominal_weights_mean_equate(
            _TOTAL,
            _TOTAL,
            _TOTAL,
            _TOTAL,
            2,
            2,
            2,
            w1=provider,
        )

    assert provider.callbacks == []
    assert core_calls == []


def test_composite_exponent_provider_is_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject executable composite exponents before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)
    provider = _ExecutableProvider()

    with pytest.raises(ValueError, match="p"):
        E.composite_linking([_TOTAL], [1.0], slopes=[1.0], p=provider)

    assert provider.callbacks == []
    assert core_calls == []


@pytest.mark.parametrize(
    ("invoke", "callback_log"),
    [
        (
            lambda value: E.nominal_weights_mean_equate(
                _TOTAL, _TOTAL, _TOTAL, _TOTAL, value, 2, 2
            ),
            _IntegerSubclass.callbacks,
        ),
        (
            lambda value: E.nominal_weights_mean_equate(
                _TOTAL, _TOTAL, _TOTAL, _TOTAL, 2, 2, 2, w1=value
            ),
            _FloatSubclass.callbacks,
        ),
        (
            lambda value: E.composite_linking(
                [_TOTAL], [1.0], slopes=[1.0], p=value
            ),
            _FloatSubclass.callbacks,
        ),
    ],
    ids=["score-ceiling", "population-weight", "composite-exponent"],
)
def test_python_numeric_subclasses_are_inert_before_core(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[object], object],
    callback_log: list[str],
) -> None:
    """Reject Python numeric subclasses without executing conversion methods."""
    core_calls = _forbid_core_discovery(monkeypatch)
    callback_log.clear()
    value: object
    if callback_log is _IntegerSubclass.callbacks:
        value = _IntegerSubclass(2)
    else:
        value = _FloatSubclass(1.0)

    with pytest.raises(ValueError):
        invoke(value)

    assert callback_log == []
    assert core_calls == []


@pytest.mark.parametrize(
    ("invoke", "match"),
    [
        (
            lambda: E.circle_arc_equate(
                _TOTAL, _POINT_LOW, _POINT_MID, _POINT_HIGH, method="arc3"
            ),
            "method",
        ),
        (
            lambda: E.circle_arc_middle_anchor(1.0, 0.5, 1.0, np.nan, 0.5, 0.5),
            "s_yb",
        ),
        (
            lambda: E.nominal_weights_mean_equate(
                _TOTAL, _TOTAL, _TOTAL, _TOTAL, 0, 2, 2
            ),
            "k_x",
        ),
        (
            lambda: E.nominal_weights_mean_equate(
                _TOTAL, _TOTAL, _TOTAL, _TOTAL, 2, 2, 2, w1=np.inf
            ),
            "w1",
        ),
        (
            lambda: E.composite_linking(
                [_TOTAL], [1.0], slopes=[1.0], p=0.5
            ),
            "p",
        ),
    ],
    ids=["arc-method", "arc-finite", "positive-k", "finite-weight", "p-domain"],
)
def test_control_domains_fail_before_core(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[], object],
    match: str,
) -> None:
    """Enforce established semantic control domains before native discovery."""
    core_calls = _forbid_core_discovery(monkeypatch)

    with pytest.raises(ValueError, match=match):
        invoke()

    assert core_calls == []


def _equate_payload() -> dict[str, object]:
    """Return a minimal Rust-shaped equating payload."""
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
    }


def test_genuine_numpy_scalars_normalize_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve exact NumPy scalar compatibility while dispatching built-ins."""
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class Core:
        """Capture normalized arguments at each Rust-shaped boundary."""

        def circle_arc_equate(self, *args, **kwargs):
            calls.append(("arc", args, kwargs))
            return {
                "equated": [0.0, 1.0, 2.0],
                "xc": 1.0,
                "yc": 1.0,
                "r2": 1.0,
                "collinear": False,
                "middle": [1.0, 1.0],
            }

        def circle_arc_middle_anchor(self, *args, **kwargs):
            calls.append(("middle", args, kwargs))
            return (1.0, 1.0)

        def nominal_weights_mean_equate(self, *args, **kwargs):
            calls.append(("nominal", args, kwargs))
            return _equate_payload()

        def composite_linking(self, *args, **kwargs):
            calls.append(("composite", args, kwargs))
            return {
                "composite": [0.0, 1.0, 2.0],
                "adjusted_weights": [1.0],
                "symmetric": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    E.circle_arc_equate(
        _TOTAL,
        (np.float32(0.0), np.float64(0.0)),
        (np.float32(1.0), np.float64(1.0)),
        (np.float32(2.0), np.float64(2.0)),
        method="ARC2",
    )
    E.circle_arc_middle_anchor(
        np.float32(1.0),
        np.float64(0.5),
        np.float32(1.0),
        np.float64(0.5),
        np.float32(0.5),
        np.float64(0.5),
    )
    E.nominal_weights_mean_equate(
        _TOTAL,
        _TOTAL,
        _TOTAL,
        _TOTAL,
        np.int64(2),
        np.int32(2),
        np.uint16(2),
        w1=np.float32(0.5),
    )
    E.composite_linking(
        [_TOTAL],
        np.array([1.0]),
        slopes=np.array([1.0]),
        p=np.float64(1.0),
    )

    assert [name for name, _, _ in calls] == ["arc", "middle", "nominal", "composite"]
    arc_args = calls[0][1]
    assert arc_args[-1] == "ARC2"
    for point in arc_args[1:4]:
        assert all(type(value) is float for value in point)
    assert all(type(value) is float for value in calls[1][1])
    nominal_args = calls[2][1]
    assert all(type(value) is int for value in nominal_args[-3:])
    assert type(calls[2][2]["w1"]) is float
    assert type(calls[3][2]["p"]) is float
