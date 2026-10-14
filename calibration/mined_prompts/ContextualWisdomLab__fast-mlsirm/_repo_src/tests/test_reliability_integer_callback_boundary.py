"""Trust-boundary regressions for reliability integer controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.reliability import feldt_alpha_ci, finn_coefficient, guttman_lambdas


class _HostileInt(int):
    """Python integer subclass whose callbacks must stay unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the shared callback counter."""
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return int.__int__(self)

    def __index__(self):
        type(self).calls += 1
        return int.__index__(self)

    def __lt__(self, other):
        type(self).calls += 1
        return int.__lt__(self, other)

    def __le__(self, other):
        type(self).calls += 1
        return int.__le__(self, other)


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass whose callbacks must stay unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the shared callback counter."""
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return super().__int__()

    def __index__(self):
        type(self).calls += 1
        return super().__index__()

    def __lt__(self, other):
        type(self).calls += 1
        return super().__lt__(other)

    def __le__(self, other):
        type(self).calls += 1
        return super().__le__(other)


class _IndexProvider:
    """Arbitrary integer-protocol provider that must never be coerced."""

    calls = 0

    def __init__(self, value: int):
        self.value = value

    @classmethod
    def reset(cls) -> None:
        """Reset the shared callback counter."""
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return self.value

    def __index__(self):
        type(self).calls += 1
        return self.value

    def __lt__(self, other):
        type(self).calls += 1
        return self.value < other

    def __le__(self, other):
        type(self).calls += 1
        return self.value <= other


class _RatingsSentinel:
    """Fail if invalid controls allow ratings or response materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise when NumPy is asked to materialize this sentinel."""
        raise AssertionError("array materialization must not run")


_HOSTILE_INTEGER_FACTORIES = (_HostileInt, _HostileNumpyInt, _IndexProvider)
_TRUSTED_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)


def _unexpected_core_discovery():
    """Fail if an invalid scalar reaches compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


def _ratings() -> np.ndarray:
    """Return a complete subjects-by-raters matrix for Finn boundary tests."""
    return np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float64)


def _items() -> np.ndarray:
    """Return a complete persons-by-items matrix for Guttman boundary tests."""
    return np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("factory", _HOSTILE_INTEGER_FACTORIES)
def test_finn_rejects_untrusted_s_levels_without_callbacks(monkeypatch, factory):
    """Finn scale-level validation rejects subclasses/protocol providers first."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()

    with pytest.raises(ValueError, match="s_levels must be an integer"):
        finn_coefficient(_RatingsSentinel(), factory(5))

    assert factory.calls == 0


@pytest.mark.parametrize("factory", _HOSTILE_INTEGER_FACTORIES)
@pytest.mark.parametrize("field", ("n_sample_splits", "seed"))
def test_guttman_rejects_untrusted_integer_controls_without_callbacks(
    monkeypatch, factory, field
):
    """Guttman split/seed validation rejects subclasses/protocol providers first."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        guttman_lambdas(_RatingsSentinel(), **{field: factory(2)})

    assert factory.calls == 0


@pytest.mark.parametrize("factory", _HOSTILE_INTEGER_FACTORIES)
@pytest.mark.parametrize("field", ("n_persons", "n_items"))
def test_feldt_rejects_untrusted_integer_controls_without_callbacks(
    monkeypatch, factory, field
):
    """Feldt count validation rejects subclasses/protocol providers first."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        feldt_alpha_ci(0.8, **{field: factory(10), "n_persons" if field != "n_persons" else "n_items": 5})

    assert factory.calls == 0


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: finn_coefficient(_RatingsSentinel(), True), "s_levels must be an integer"),
        (
            lambda: guttman_lambdas(_RatingsSentinel(), n_sample_splits=True),
            "n_sample_splits must be an integer",
        ),
        (lambda: guttman_lambdas(_RatingsSentinel(), seed=False), "seed must be an integer"),
        (lambda: feldt_alpha_ci(0.8, True, 5), "n_persons must be an integer"),
        (lambda: feldt_alpha_ci(0.8, 10, False), "n_items must be an integer"),
    ),
)
def test_boolean_integer_controls_stay_rejected_before_core_discovery(
    monkeypatch, call, message
):
    """Boolean controls remain invalid despite ``bool`` being an ``int`` subclass."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        call()


class _FakeReliabilityCore:
    """Capture trusted native-dispatch arguments without running Rust arithmetic."""

    def __init__(self) -> None:
        """Initialize an empty call ledger."""
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def finn_coefficient(self, *args: Any) -> dict[str, float | int]:
        """Record one Finn dispatch and return a structurally valid result."""
        self.calls.append(("finn", args))
        return {
            "value": 0.75,
            "statistic": 2.0,
            "df2": 3.0,
            "p_value": 0.2,
            "subjects": 4,
            "raters": 2,
        }

    def guttman_lambdas(self, *args: Any) -> dict[str, float | int | bool]:
        """Record one Guttman dispatch and return a structurally valid result."""
        self.calls.append(("guttman", args))
        return {
            "lambda1": 0.4,
            "lambda2": 0.5,
            "lambda3": 0.6,
            "lambda4": 0.7,
            "lambda5": 0.55,
            "lambda6": 0.65,
            "beta": 0.3,
            "mean_split": 0.5,
            "n_splits": 3,
            "exhaustive": True,
        }

    def feldt_alpha_ci(self, *args: Any) -> dict[str, float]:
        """Record one Feldt dispatch and return a structurally valid result."""
        self.calls.append(("feldt", args))
        return {
            "alpha": 0.8,
            "lower": 0.5,
            "upper": 0.9,
            "r_bar": 0.4,
            "df1": 9.0,
            "df2": 36.0,
        }


def _install_fake_core(monkeypatch: pytest.MonkeyPatch) -> _FakeReliabilityCore:
    """Install a deterministic compiled-core substitute and return it."""
    core = _FakeReliabilityCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    return core


@pytest.mark.parametrize("numpy_type", _TRUSTED_NUMPY_INTEGER_TYPES)
def test_genuine_numpy_integer_controls_dispatch_as_python_ints(
    monkeypatch, numpy_type
):
    """Exact supported NumPy integer scalars normalize before native dispatch."""
    core = _install_fake_core(monkeypatch)

    finn = finn_coefficient(_ratings(), numpy_type(5))
    guttman = guttman_lambdas(_items(), n_sample_splits=numpy_type(8), seed=numpy_type(2))
    feldt = feldt_alpha_ci(0.8, numpy_type(10), numpy_type(5))

    assert finn.value == 0.75
    assert guttman.lambda3 == 0.6
    assert feldt.alpha == 0.8
    assert [name for name, _args in core.calls] == ["finn", "guttman", "feldt"]
    assert type(core.calls[0][1][3]) is int
    assert core.calls[0][1][3] == 5
    assert type(core.calls[1][1][3]) is int
    assert type(core.calls[1][1][4]) is int
    assert core.calls[1][1][3] == 8
    assert core.calls[1][1][4] == 2
    assert type(core.calls[2][1][1]) is int
    assert type(core.calls[2][1][2]) is int
    assert core.calls[2][1][1] == 10
    assert core.calls[2][1][2] == 5


def test_trusted_python_and_numpy_finn_controls_are_identical(monkeypatch):
    """Finn scale-level identity matches the ordinary Python-int happy path."""
    core = _install_fake_core(monkeypatch)
    python_result = finn_coefficient(_ratings(), 5)
    numpy_result = finn_coefficient(_ratings(), np.int64(5))

    assert python_result.value == numpy_result.value
    assert core.calls[0][1][3] == core.calls[1][1][3] == 5
