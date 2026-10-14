"""Trust-boundary regressions for Delta-plot population evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.deltaplot import delta_plot


class _HostileGroupNumber:
    """Object-dtype group cell whose numeric callback must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the conversion callback counter."""
        cls.calls = 0

    def __float__(self) -> float:
        """Fail if package admission attempts caller numeric conversion."""
        type(self).calls += 1
        raise AssertionError("group element conversion must not execute")


class _ReachedCore(RuntimeError):
    """Sentinel proving trusted group evidence passed package admission."""


def _unexpected_core_discovery():
    """Fail if invalid group evidence reaches compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid group evidence")


def _expected_core_discovery():
    """Stop after trusted group evidence reaches the compiled-core boundary."""
    raise _ReachedCore("trusted group evidence reached compiled-core discovery")


def _responses() -> np.ndarray:
    """Return a small valid person-by-item response matrix."""
    return np.array([[0, 1], [1, 0]], dtype=np.int64)


def test_object_group_storage_fails_before_element_conversion_or_core(monkeypatch):
    """Object storage is rejected without executing per-cell numeric callbacks."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    hostile = _HostileGroupNumber()
    hostile.reset()
    group = np.array([0, hostile], dtype=object)

    with pytest.raises(ValueError, match="group must be a numeric array"):
        delta_plot(_responses(), group, alpha=0.05, max_iter=10)

    assert hostile.calls == 0


def test_text_group_storage_is_not_reinterpreted_as_population_identity(monkeypatch):
    """Textual 0/1 labels fail before conversion can make them numeric evidence."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    group = np.array(["0", "1"], dtype=np.str_)

    with pytest.raises(ValueError, match="group must be a numeric array"):
        delta_plot(_responses(), group, alpha=0.05, max_iter=10)


@pytest.mark.parametrize(
    "group",
    [
        np.array([False, True], dtype=np.bool_),
        np.array([0, 1], dtype=np.int64),
        np.array([0.0, 1.0], dtype=np.float32),
    ],
)
def test_trusted_zero_one_group_storage_reaches_rust_boundary(monkeypatch, group):
    """Boolean and ordinary numeric 0/1 populations keep their public path."""
    monkeypatch.setattr(fitstats, "_core_module", _expected_core_discovery)

    with pytest.raises(_ReachedCore, match="trusted group evidence"):
        delta_plot(_responses(), group, alpha=0.05, max_iter=10)
