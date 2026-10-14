"""Trust-boundary regressions for Lord flexilevel Python marshalling."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from fast_mlsirm import exposure


class _ArrayBomb:
    """Caller-owned array provider that must not run for invalid controls."""

    callbacks: list[str] = []

    def __array__(self, *args, **kwargs):
        type(self).callbacks.append("__array__")
        raise AssertionError("caller array materialization executed")


class _FloatBomb:
    """Object-dtype cell that must never reach numeric conversion."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        raise AssertionError("caller numeric conversion executed")


def _guard_native_discovery(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Fail if the wrapper discovers the compiled core before admission completes."""

    discoveries: list[tuple[str, tuple[str, ...]]] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 1 and "_core" in fromlist:
            discoveries.append((name, tuple(fromlist)))
            raise AssertionError("native core discovery preceded flexilevel admission")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return discoveries


def test_flexilevel_rejects_even_item_count_before_data_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rust odd-item scientific contract is a pre-materialization boundary."""

    discoveries = _guard_native_discovery(monkeypatch)
    _ArrayBomb.callbacks.clear()

    with pytest.raises(ValueError, match="n_items must be odd"):
        exposure.flexilevel_administer(_ArrayBomb(), n_persons=1, n_items=4)

    assert _ArrayBomb.callbacks == []
    assert discoveries == []


def test_flexilevel_rejects_platform_product_overflow_before_data_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python must mirror the Rust usize product boundary before caller work."""

    discoveries = _guard_native_discovery(monkeypatch)
    _ArrayBomb.callbacks.clear()
    usize_max = int(np.iinfo(np.uintp).max)

    with pytest.raises(ValueError, match=r"n_persons \* n_items exceeds platform size"):
        exposure.flexilevel_administer(
            _ArrayBomb(), n_persons=usize_max, n_items=3
        )

    assert _ArrayBomb.callbacks == []
    assert discoveries == []


def test_flexilevel_rejects_flat_size_mismatch_before_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact response product is established before native dispatch."""

    discoveries = _guard_native_discovery(monkeypatch)

    with pytest.raises(
        ValueError, match=r"responses size must equal n_persons \* n_items"
    ):
        exposure.flexilevel_administer(
            np.array([0, 1, 0, 1], dtype=np.int8), n_persons=1, n_items=3
        )

    assert discoveries == []


@pytest.mark.parametrize(
    "responses",
    [
        np.array([0 + 0j, 1 + 1j, 0 + 0j]),
        np.array(["0", "1", "0"]),
        np.array([_FloatBomb(), 1, 0], dtype=object),
    ],
)
def test_flexilevel_rejects_lossy_response_storage_before_core(
    monkeypatch: pytest.MonkeyPatch,
    responses: np.ndarray,
) -> None:
    """Complex/text/object cells cannot be reinterpreted as binary evidence."""

    discoveries = _guard_native_discovery(monkeypatch)
    _FloatBomb.callbacks.clear()

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        exposure.flexilevel_administer(responses, n_persons=1, n_items=3)

    assert _FloatBomb.callbacks == []
    assert discoveries == []


@pytest.mark.parametrize(
    "p",
    [
        np.array([0.1 + 0j, 0.5 + 1j, 0.9 + 0j]),
        np.array(["0.1", "0.5", "0.9"]),
        np.array([_FloatBomb(), 0.5, 0.9], dtype=object),
    ],
)
def test_flexilevel_distribution_rejects_lossy_probability_storage_before_core(
    monkeypatch: pytest.MonkeyPatch,
    p: np.ndarray,
) -> None:
    """Probability evidence is admitted without caller numeric conversion."""

    discoveries = _guard_native_discovery(monkeypatch)
    _FloatBomb.callbacks.clear()

    with pytest.raises(ValueError, match="p must be a real numeric array"):
        exposure.flexilevel_score_distribution(p)

    assert _FloatBomb.callbacks == []
    assert discoveries == []


@pytest.mark.parametrize(
    ("p", "message"),
    [
        (np.array([0.2, 0.4, 0.6, 0.8]), "p length must be odd and at least 3"),
        (np.array([0.2, np.nan, 0.8]), "p must contain finite values in \\[0, 1\\]"),
        (np.array([0.2, np.inf, 0.8]), "p must contain finite values in \\[0, 1\\]"),
        (np.array([0.2, -0.1, 0.8]), "p must contain finite values in \\[0, 1\\]"),
        (np.array([0.2, 1.1, 0.8]), "p must contain finite values in \\[0, 1\\]"),
    ],
)
def test_flexilevel_distribution_rejects_invalid_domain_before_core(
    monkeypatch: pytest.MonkeyPatch,
    p: np.ndarray,
    message: str,
) -> None:
    """Rust length/probability domains are mirrored before native discovery."""

    discoveries = _guard_native_discovery(monkeypatch)

    with pytest.raises(ValueError, match=message):
        exposure.flexilevel_score_distribution(p)

    assert discoveries == []


def test_flexilevel_preserves_supported_binary_response_storage() -> None:
    """Boolean, integer, and real binary evidence retain the public contract."""

    for responses in (
        np.array([[True, False, True]]),
        np.array([1, 0, 1], dtype=np.int16),
        np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    ):
        result = exposure.flexilevel_administer(responses, n_persons=1, n_items=3)
        assert int(result["n_administered"]) == 2


def test_flexilevel_preserves_plain_sequence_binary_response_storage() -> None:
    """Plain list/tuple response array-likes remain callback-safe and supported."""

    for responses in (
        [1, 0, 1],
        [[True, False, True]],
        ((np.int16(1), np.float32(0.0), np.uint8(1)),),
    ):
        result = exposure.flexilevel_administer(responses, n_persons=1, n_items=3)
        assert int(result["n_administered"]) == 2


def test_flexilevel_distribution_preserves_real_numeric_probabilities() -> None:
    """Valid probabilities continue to reach the Rust recursion unchanged."""

    result = exposure.flexilevel_score_distribution(
        np.array([0.2, 0.5, 0.8], dtype=np.float32)
    )
    assert np.isclose(float(np.sum(result["probs"])), 1.0)


def test_flexilevel_distribution_preserves_numpy_scalars_in_plain_sequences() -> None:
    """Trusted NumPy scalar probability sequences retain array-like compatibility."""

    for probabilities in (
        [np.float64(0.2), np.float64(0.5), np.float64(0.8)],
        (np.float32(0.2), np.int16(0), np.uint8(1)),
    ):
        result = exposure.flexilevel_score_distribution(probabilities)
        assert np.isclose(float(np.sum(result["probs"])), 1.0)
