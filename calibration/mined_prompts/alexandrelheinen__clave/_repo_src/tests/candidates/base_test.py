"""Tests for the candidate interface.

None of these import a deep learning library, which is the point: the registry
must be readable on a machine with none installed.
"""

import numpy as np

from clave.candidates.base import (
    MATERIAL_CLASS_COUNT,
    Candidate,
    CandidateSpec,
    Stage,
)
from clave.candidates.fixture import frame, state

SPEC = CandidateSpec(
    name="demo",
    stage=Stage.POLICY,
    license="MIT",
    source="x",
    requires="nonexistent_lib",
)


def test_a_spec_is_constructible_without_any_heavy_dependency() -> None:
    """Describing does not import."""

    assert SPEC.name == "demo"
    assert SPEC.license == "MIT"


def test_a_spec_reports_its_stage() -> None:
    """A spec reports its stage."""
    assert SPEC.stage is Stage.POLICY
    assert SPEC.stage.value == "policy"


def test_a_missing_dependency_is_reported_not_raised() -> None:
    """An absent library names itself and does not raise."""

    def build() -> object:
        # This module does not exist, which is the condition under test.
        import nonexistent_lib  # type: ignore[import-not-found]  # noqa: F401

        return object()

    result = Candidate(SPEC, build).load()
    assert not result.loaded
    assert result.unavailable_reason is not None
    assert "nonexistent_lib" in result.unavailable_reason


def test_any_other_failure_is_reported_without_substitution() -> None:
    """A broken build records its reason, not another candidate."""

    def build() -> object:
        raise ValueError("head size mismatch")

    result = Candidate(SPEC, build).load()
    assert not result.loaded
    assert result.unavailable_reason is not None
    assert "head size mismatch" in result.unavailable_reason
    assert result.spec is SPEC


def test_a_successful_build_is_loaded() -> None:
    """The happy path returns the model it built."""
    sentinel = object()
    result = Candidate(SPEC, lambda: sentinel).load()
    assert result.loaded
    assert result.model is sentinel


def test_class_count_matches_the_taxonomy() -> None:
    """The head is sized to M-01 through M-11."""
    assert MATERIAL_CLASS_COUNT == 11


def test_the_fixture_is_identical_across_two_seeded_runs() -> None:
    """Two benchmark runs see the same input."""
    assert np.array_equal(frame(7), frame(7))
    assert np.array_equal(state(7), state(7))


def test_the_fixture_differs_across_two_seeds() -> None:
    """The guard above would pass for a constant without this."""
    assert not np.array_equal(frame(1), frame(2))


def test_the_fixture_has_the_declared_shape() -> None:
    """The adapters rely on the channel-first convention."""
    assert frame().shape == (3, 96, 96)
    assert frame().dtype == np.float32
