"""Callback-safety regressions for IRTree evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.preprocessing import irtree_expand


class _HostileArrayProvider:
    """Array-like caller object whose NumPy protocol must never run."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls += 1
        raise AssertionError("caller __array__ callback executed")


class _HostileFloat:
    """Object-dtype cell whose numeric conversion must never run."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ callback executed")


@pytest.mark.parametrize("position", ["responses", "mapping", "node_dims"])
def test_irtree_rejects_array_protocols_without_callbacks(position: str) -> None:
    provider = _HostileArrayProvider()
    kwargs: dict[str, object] = {
        "responses": np.array([[0.0], [1.0]]),
        "mapping": np.array([[0.0, 1.0]]),
        "node_dims": np.array([0]),
    }
    kwargs[position] = provider

    with pytest.raises(ValueError):
        irtree_expand(**kwargs)  # type: ignore[arg-type]

    assert provider.calls == 0


@pytest.mark.parametrize("position", ["responses", "mapping", "node_dims"])
def test_irtree_rejects_object_storage_without_numeric_callbacks(position: str) -> None:
    hostile = _HostileFloat()
    kwargs: dict[str, object] = {
        "responses": np.array([[0.0], [1.0]]),
        "mapping": np.array([[0.0, 1.0]]),
        "node_dims": np.array([0]),
    }
    if position == "responses":
        kwargs[position] = np.array([[hostile], [1.0]], dtype=object)
    elif position == "mapping":
        kwargs[position] = np.array([[hostile, 1.0]], dtype=object)
    else:
        kwargs[position] = np.array([hostile], dtype=object)

    with pytest.raises(ValueError):
        irtree_expand(**kwargs)  # type: ignore[arg-type]

    assert hostile.calls == 0


def test_irtree_preserves_builtin_sequence_and_numpy_scalar_compatibility() -> None:
    expanded, factor_id = irtree_expand(
        [[np.int8(0)], [np.float32(1)], [np.nan]],
        [[np.uint8(0), np.float64(1)]],
        node_dims=[np.uint8(0)],
    )

    assert np.array_equal(expanded[:2, 0], np.array([0.0, 1.0]))
    assert np.isnan(expanded[2, 0])
    assert np.array_equal(factor_id, np.array([0]))
