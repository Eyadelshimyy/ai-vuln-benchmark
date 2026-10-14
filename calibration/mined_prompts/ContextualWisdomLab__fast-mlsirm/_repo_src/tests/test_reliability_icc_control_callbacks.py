"""Regression coverage for ICC control values at the Python/Rust trust boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_mlsirm.reliability import icc


class _RatingsSentinel:
    """Fail if invalid controls allow ratings materialization to begin."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise when NumPy is asked to materialize this sentinel."""
        raise AssertionError("ratings materialization must not run")


class _ExplosiveControl:
    """Fail on every caller-owned conversion/comparison callback in scope."""

    def __str__(self) -> str:
        """Reject string conversion."""
        raise AssertionError("__str__ must not run")

    def __repr__(self) -> str:
        """Reject representation conversion."""
        raise AssertionError("__repr__ must not run")

    def __int__(self) -> int:
        """Reject integer conversion."""
        raise AssertionError("__int__ must not run")

    def __index__(self) -> int:
        """Reject index conversion."""
        raise AssertionError("__index__ must not run")

    def __float__(self) -> float:
        """Reject floating-point conversion."""
        raise AssertionError("__float__ must not run")

    def __eq__(self, other: object) -> bool:
        """Reject equality comparison."""
        raise AssertionError("__eq__ must not run")

    def __hash__(self) -> int:
        """Reject hashing."""
        raise AssertionError("__hash__ must not run")

    def __lt__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__lt__ must not run")

    def __le__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__le__ must not run")

    def __gt__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__gt__ must not run")

    def __ge__(self, other: object) -> bool:
        """Reject ordering comparison."""
        raise AssertionError("__ge__ must not run")


class _ExplosiveString(str):
    """A built-in string subclass whose conversion must never execute."""

    def __str__(self) -> str:
        """Reject normalization through ``str``."""
        raise AssertionError("string subclass __str__ must not run")

    def __eq__(self, other: object) -> bool:
        """Reject equality comparison on the subclass instance."""
        raise AssertionError("string subclass __eq__ must not run")

    __hash__ = str.__hash__


class _ExplosiveFloat(float):
    """A built-in float subclass whose conversion must never execute."""

    def __float__(self) -> float:
        """Reject normalization through ``float``."""
        raise AssertionError("float subclass __float__ must not run")


class _ExplosiveInt(int):
    """A built-in integer subclass whose conversion must never execute."""

    def __float__(self) -> float:
        """Reject normalization through ``float``."""
        raise AssertionError("int subclass __float__ must not run")


class _ExplosiveNumpyFloat(np.float64):
    """A NumPy floating subclass whose conversion must never execute."""

    def __float__(self) -> float:
        """Reject normalization through ``float``."""
        raise AssertionError("NumPy subclass __float__ must not run")


class _FakeCore:
    """Capture trusted native-dispatch arguments without running Rust arithmetic."""

    def __init__(self) -> None:
        """Initialize an empty call ledger."""
        self.calls: list[tuple[Any, ...]] = []

    def icc(self, *args: Any) -> dict[str, float | int]:
        """Record one dispatch and return a structurally valid ICC result."""
        self.calls.append(args)
        return {
            "value": 0.5,
            "subjects": 2,
            "raters": 2,
            "fvalue": 1.0,
            "df1": 1.0,
            "df2": 1.0,
            "p_value": 0.5,
            "lbound": 0.1,
            "ubound": 0.9,
        }


def _install_fake_core(monkeypatch: pytest.MonkeyPatch) -> _FakeCore:
    """Install a deterministic compiled-core substitute and return it."""
    from fast_mlsirm import fitstats

    core = _FakeCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    return core


def _valid_ratings() -> np.ndarray:
    """Return the smallest ordinary numeric matrix accepted by the adapter."""
    return np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("model", {"model": "bogus"}),
        ("type", {"type": "bogus"}),
        ("unit", {"unit": "bogus"}),
        ("r0", {"r0": -0.1}),
        ("conf_level", {"conf_level": 1.0}),
    ],
)
def test_invalid_controls_fail_before_core_or_ratings(
    monkeypatch: pytest.MonkeyPatch, name: str, kwargs: dict[str, object]
) -> None:
    """Reject each semantic control before native discovery or ratings access."""
    from fast_mlsirm import fitstats

    def explode_core() -> None:
        raise AssertionError("_core_module must not run")

    monkeypatch.setattr(fitstats, "_core_module", explode_core)
    with pytest.raises(ValueError, match=name):
        icc(_RatingsSentinel(), **kwargs)


@pytest.mark.parametrize("field", ["model", "type", "unit"])
def test_executable_string_controls_are_rejected_without_callbacks(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Reject arbitrary string providers without invoking caller protocols."""
    core = _install_fake_core(monkeypatch)
    with pytest.raises(ValueError, match=field):
        icc(_valid_ratings(), **{field: _ExplosiveControl()})
    assert core.calls == []


@pytest.mark.parametrize("field", ["model", "type", "unit"])
def test_string_subclasses_are_rejected_without_callbacks(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Require exact built-in string identity for every Rust vocabulary field."""
    core = _install_fake_core(monkeypatch)
    with pytest.raises(ValueError, match=field):
        icc(_valid_ratings(), **{field: _ExplosiveString("oneway")})
    assert core.calls == []


@pytest.mark.parametrize("field", ["r0", "conf_level"])
@pytest.mark.parametrize(
    "value",
    [_ExplosiveControl(), _ExplosiveFloat(0.5), _ExplosiveInt(0), _ExplosiveNumpyFloat(0.5)],
)
def test_executable_numeric_controls_are_rejected_without_callbacks(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Reject protocol providers and scalar subclasses before numeric coercion."""
    core = _install_fake_core(monkeypatch)
    with pytest.raises(ValueError, match=field):
        icc(_valid_ratings(), **{field: value})
    assert core.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("r0", True),
        ("r0", 1 + 0j),
        ("r0", float("nan")),
        ("r0", float("inf")),
        ("r0", -0.01),
        ("r0", 1.0),
        ("conf_level", False),
        ("conf_level", 0 + 0j),
        ("conf_level", float("nan")),
        ("conf_level", float("-inf")),
        ("conf_level", 0.0),
        ("conf_level", 1.0),
    ],
)
def test_invalid_numeric_identities_and_ranges_fail_locally(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Enforce exact real-scalar identities, finiteness, and Rust parameter ranges."""
    core = _install_fake_core(monkeypatch)
    with pytest.raises(ValueError, match=field):
        icc(_valid_ratings(), **{field: value})
    assert core.calls == []


@pytest.mark.parametrize(
    ("r0", "conf_level"),
    [
        (0, 1.0 - 1e-6),
        (0.25, 0.95),
        (np.int8(0), np.float16(0.95)),
        (np.uint64(0), np.float32(0.95)),
        (np.float64(0.25), np.longdouble("0.95")),
    ],
)
def test_trusted_numeric_scalars_dispatch_as_exact_python_floats(
    monkeypatch: pytest.MonkeyPatch, r0: object, conf_level: object
) -> None:
    """Normalize trusted built-in and genuine NumPy real scalars before dispatch."""
    core = _install_fake_core(monkeypatch)
    result = icc(_valid_ratings(), r0=r0, conf_level=conf_level)

    assert result.value == 0.5
    assert len(core.calls) == 1
    native_args = core.calls[0]
    assert native_args[3:6] == ("oneway", "consistency", "single")
    assert type(native_args[6]) is float
    assert type(native_args[7]) is float
