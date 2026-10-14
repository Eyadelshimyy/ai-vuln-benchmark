"""Resource-boundary regressions for persisted conformance JSON replay."""

from __future__ import annotations

import pytest

from fast_mlsirm.cross_engine_conformance import ConformanceInventory


def test_json_replay_rejects_unpaired_surrogate_with_stable_error() -> None:
    """Non-UTF-8-encodable text fails through the package validation contract."""
    with pytest.raises(ValueError, match="manifest JSON must be UTF-8 encodable"):
        ConformanceInventory.from_json("\ud800")


def test_json_replay_rejects_excessive_nesting_with_stable_error() -> None:
    """A bounded but deeply nested payload cannot escape as RecursionError."""
    payload = "[" * 2_000 + "]" * 2_000

    with pytest.raises(ValueError, match="manifest JSON nesting is too deep"):
        ConformanceInventory.from_json(payload)
