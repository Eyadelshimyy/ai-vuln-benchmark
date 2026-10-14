"""Semantic-control trust-boundary tests for weighted contextual effects."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import fast_mlsirm.multilevel as multilevel
import fast_mlsirm.multilevel.estimation as estimation
from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
)


def _revision(tag: str) -> str:
    """Return one deterministic content fingerprint for the test fixture."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _design():
    """Build a minimal two-person contextual-membership design."""
    return build_context_membership_design(
        [
            build_context_membership(
                observation_id="person_alpha",
                context_dimension_id="school_membership",
                context_id="school_east",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("alpha-east"),
            ),
            build_context_membership(
                observation_id="person_beta",
                context_dimension_id="school_membership",
                context_id="school_west",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("beta-west"),
            ),
        ]
    )


def _effects() -> dict[tuple[str, str], float]:
    """Return one complete inert effect mapping for the fixture design."""
    return {
        ("school_membership", "school_east"): 0.25,
        ("school_membership", "school_west"): -0.25,
    }


class _HostileWorkerInt(int):
    """Integer subclass whose comparison/conversion callbacks must not run."""

    callback_count = 0

    def __lt__(self, other: object) -> bool:
        type(self).callback_count += 1
        raise AssertionError("caller __lt__ must not execute")

    def __int__(self) -> int:
        type(self).callback_count += 1
        raise AssertionError("caller __int__ must not execute")

    def __index__(self) -> int:
        type(self).callback_count += 1
        raise AssertionError("caller __index__ must not execute")


def test_weighted_effect_worker_subclass_fails_before_callbacks_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public worker admission rejects subclasses before comparison callbacks."""
    _HostileWorkerInt.callback_count = 0
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not be discovered for invalid controls")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="worker_count"):
        multilevel.weighted_contextual_effect(
            _design(),
            _effects(),
            worker_count=_HostileWorkerInt(1),
        )

    assert _HostileWorkerInt.callback_count == 0
    assert core_discoveries == 0


def test_weighted_effect_numpy_worker_normalizes_to_builtin_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported concrete NumPy integer controls reach Rust as built-in ints."""
    captured_worker: object | None = None

    class _Core:
        @staticmethod
        def weighted_contextual_effect(
            row_offsets: np.ndarray,
            context_indices: np.ndarray,
            weights: np.ndarray,
            effects: np.ndarray,
            worker_count: object,
        ) -> np.ndarray:
            nonlocal captured_worker
            captured_worker = worker_count
            return np.zeros(row_offsets.shape[0] - 1, dtype=np.float64)

    monkeypatch.setattr(estimation, "multilevel_core", lambda: _Core())

    result = multilevel.weighted_contextual_effect(
        _design(),
        _effects(),
        worker_count=np.int16(2),
    )

    assert result.shape == (2,)
    assert type(captured_worker) is int
    assert captured_worker == 2
