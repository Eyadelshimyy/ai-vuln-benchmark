"""Trust-boundary regressions for answer-copying integer controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.security import k_index, k_variants, wollack_omega


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


def _responses() -> np.ndarray:
    """Return complete binary response data for row-index boundary tests."""
    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _identity_responses() -> np.ndarray:
    """Return deterministic non-degenerate data for native parity checks."""
    rng = np.random.default_rng(0)
    responses = (rng.random((60, 20)) < 0.6).astype(np.float64)
    responses[1, :] = 0.0
    return responses


def _options() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid nominal-option inputs for Wollack omega."""
    copier = np.array([0, 1, 2], dtype=np.int64)
    source = np.array([0, 1, 0], dtype=np.int64)
    probs = np.full((3, 3), 1.0 / 3.0)
    return copier, source, probs


@pytest.mark.parametrize("factory", _HOSTILE_INTEGER_FACTORIES)
def test_wollack_rejects_untrusted_integer_controls_without_callbacks(
    monkeypatch, factory
):
    """Wollack option-count validation rejects subclasses/protocol providers first."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()
    copier, source, probs = _options()

    with pytest.raises(ValueError, match="n_options must be an integer"):
        wollack_omega(copier, source, probs, factory(3))

    assert factory.calls == 0


@pytest.mark.parametrize("factory", _HOSTILE_INTEGER_FACTORIES)
@pytest.mark.parametrize("argument_name", ("copier", "source"))
def test_k_index_rejects_untrusted_integer_controls_without_callbacks(
    monkeypatch, factory, argument_name
):
    """K-index validates each row control before callbacks or core discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()
    copier = factory(0) if argument_name == "copier" else 0
    source = factory(1) if argument_name == "source" else 1

    with pytest.raises(
        ValueError, match=f"{argument_name} must be an integer row index"
    ):
        k_index(_responses(), copier, source)

    assert factory.calls == 0


@pytest.mark.parametrize("factory", _HOSTILE_INTEGER_FACTORIES)
@pytest.mark.parametrize("argument_name", ("copier", "source"))
def test_k_variants_rejects_untrusted_integer_controls_without_callbacks(
    monkeypatch, factory, argument_name
):
    """K-variant validation covers both row controls before native discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()
    copier = factory(0) if argument_name == "copier" else 0
    source = factory(1) if argument_name == "source" else 1

    with pytest.raises(
        ValueError, match=f"{argument_name} must be an integer row index"
    ):
        k_variants(_responses(), copier, source)

    assert factory.calls == 0


@pytest.mark.parametrize("numpy_type", _TRUSTED_NUMPY_INTEGER_TYPES)
def test_genuine_numpy_integer_controls_reach_dispatch_boundary(
    monkeypatch, numpy_type
):
    """Exact supported NumPy integer scalars survive trusted normalization."""
    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    copier, source, probs = _options()

    with pytest.raises(
        RuntimeError, match="wollack_omega requires the compiled Rust core"
    ):
        wollack_omega(copier, source, probs, numpy_type(3))
    with pytest.raises(RuntimeError, match="k_index requires the compiled Rust core"):
        k_index(_responses(), numpy_type(0), numpy_type(1))
    with pytest.raises(
        RuntimeError, match="k_variants requires the compiled Rust core"
    ):
        k_variants(_responses(), numpy_type(0), numpy_type(1))

    assert calls == 3


def test_numpy_integer_controls_preserve_native_statistic_identity() -> None:
    """Representative NumPy controls produce the same Rust statistics as ints."""
    if fitstats._core_module() is None:
        pytest.skip("compiled Rust core required for native statistic parity")

    copier, source, probs = _options()
    py_omega = wollack_omega(copier, source, probs, 3)
    np_omega = wollack_omega(copier, source, probs, np.int64(3))
    assert np_omega.omega == py_omega.omega

    responses = _identity_responses()
    py_k = k_index(responses, 0, 1)
    np_k = k_index(responses, np.int64(0), np.int64(1))
    assert np_k.k_index == py_k.k_index

    py_variants = k_variants(responses, 0, 1)
    np_variants = k_variants(responses, np.int64(0), np.int64(1))
    assert (
        np_variants.k1,
        np_variants.k2,
        np_variants.s1_index,
        np_variants.s2_index,
    ) == (
        py_variants.k1,
        py_variants.k2,
        py_variants.s1_index,
        py_variants.s2_index,
    )


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (
            lambda: wollack_omega(*_options(), True),
            "n_options must be an integer",
        ),
        (
            lambda: k_index(_responses(), True, 1),
            "copier must be an integer row index",
        ),
        (
            lambda: k_variants(_responses(), 0, False),
            "source must be an integer row index",
        ),
    ),
)
def test_boolean_integer_controls_stay_rejected_before_core_discovery(
    monkeypatch, call, message
):
    """Boolean controls remain invalid despite ``bool`` being an ``int`` subclass."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        call()
