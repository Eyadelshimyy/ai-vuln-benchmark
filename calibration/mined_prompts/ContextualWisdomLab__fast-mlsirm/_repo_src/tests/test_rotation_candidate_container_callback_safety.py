"""Callback-free candidate-container admission for rotation selection."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rotation_selection as selection


class _HostileCandidates(list[str]):
    """List subclass whose iteration callback must never run during admission."""

    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("caller candidate-container callback must not execute")


def test_selection_rejects_candidate_container_subclass_before_iteration_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate-set identity is sealed before caller iteration or Rust discovery."""

    candidates = _HostileCandidates(["varimax", "geomin"])
    _HostileCandidates.calls = 0
    core_calls: list[int] = []

    def _deny_core() -> object:
        core_calls.append(1)
        raise AssertionError("rotation core must not be discovered")

    monkeypatch.setattr(selection, "rotation_core", _deny_core)

    with pytest.raises(
        ValueError,
        match="candidates must be an exact list or tuple of criterion names",
    ):
        selection.select_rotation_criterion(
            np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64),
            candidates,
        )

    assert _HostileCandidates.calls == 0
    assert core_calls == []


def test_candidate_container_keeps_exact_list_and_tuple_compatibility() -> None:
    """Ordinary package-inert candidate containers retain normalization semantics."""

    assert selection._candidate_names(["varimax", "geomin"]) == (
        "varimax",
        "geomin",
    )
    assert selection._candidate_names(("quartimax", "geomin")) == (
        "quartimax",
        "geomin",
    )
