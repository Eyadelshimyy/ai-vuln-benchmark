"""Tests for the sparse cross-classified multiple-membership Rust predictor.

Exercises the same one-hot-nesting-parity and permutation-invariance
contracts as the Rust unit tests (``crates/mlsirm-core/src/multilevel.rs``),
plus the Python-side marshalling from a validated ``ContextMembershipDesign``
through to the returned ``numpy.ndarray``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.multilevel import (
    ContextMembershipDesign,
    build_context_membership,
    build_context_membership_design,
    weighted_contextual_effect,
)
from fast_mlsirm._multilevel_core_loader import multilevel_core


def _membership(
    *,
    observation_id: str,
    context_dimension_id: str,
    context_id: str,
    membership_weight: float,
    revision: str,
):
    """Build one deterministic membership edge for focused contracts.

    ``revision`` is a short hex tag, zero-padded to the required 64-character
    fingerprint shape (never repeated -- repeating a multi-character tag
    would silently overflow past 64 characters and fail validation).
    """
    return build_context_membership(
        observation_id=observation_id,
        context_dimension_id=context_dimension_id,
        context_id=context_id,
        membership_weight=membership_weight,
        membership_revision_fingerprint=revision.rjust(64, "0"),
    )


def test_one_hot_nesting_reproduces_direct_lookup() -> None:
    """Ordinary single-membership nesting returns each context's own effect."""
    design = build_context_membership_design(
        [
            _membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                revision="a",
            ),
            _membership(
                observation_id="person_two",
                context_dimension_id="team_membership",
                context_id="team_beta",
                membership_weight=1.0,
                revision="b",
            ),
            _membership(
                observation_id="person_three",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                revision="c",
            ),
        ]
    )
    effects = {
        ("team_membership", "team_alpha"): 5.0,
        ("team_membership", "team_beta"): -3.0,
    }

    result = weighted_contextual_effect(design, effects)

    # observation_ids sorts lexicographically: person_one, person_three,
    # person_two ("three" < "two"), so results follow that order:
    # alpha, alpha, beta.
    assert list(design.observation_ids) == ["person_one", "person_three", "person_two"]
    assert list(result) == [5.0, 5.0, -3.0]


def test_weighted_membership_computes_the_convex_combination() -> None:
    """A weighted multiple-membership design mixes its context effects."""
    design = build_context_membership_design(
        [
            _membership(
                observation_id="person_one",
                context_dimension_id="school_membership",
                context_id="school_a",
                membership_weight=0.25,
                revision="d",
            ),
            _membership(
                observation_id="person_one",
                context_dimension_id="school_membership",
                context_id="school_b",
                membership_weight=0.75,
                revision="e",
            ),
        ]
    )
    effects = {
        ("school_membership", "school_a"): 8.0,
        ("school_membership", "school_b"): 4.0,
    }

    result = weighted_contextual_effect(design, effects)

    assert result == pytest.approx([0.25 * 8.0 + 0.75 * 4.0])


def test_cross_classified_dimensions_sum_their_separate_contributions() -> None:
    """Simultaneous membership in two dimensions sums both contributions."""
    design = build_context_membership_design(
        [
            _membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                revision="f",
            ),
            _membership(
                observation_id="person_one",
                context_dimension_id="site_membership",
                context_id="site_alpha",
                membership_weight=1.0,
                revision="0",
            ),
        ]
    )
    effects = {
        ("team_membership", "team_alpha"): 10.0,
        ("site_membership", "site_alpha"): 100.0,
    }

    result = weighted_contextual_effect(design, effects)

    assert list(result) == [110.0]


def test_permuting_the_input_membership_list_does_not_change_the_result() -> None:
    """Edge order fed into build_context_membership_design does not matter.

    build_context_membership_design already canonically sorts memberships,
    so this also proves the Rust layer's own permutation-invariance claim is
    exercised end to end, not just at the Rust unit-test level.
    """
    edges = [
        _membership(
            observation_id="person_one",
            context_dimension_id="school_membership",
            context_id="school_a",
            membership_weight=0.4,
            revision="1",
        ),
        _membership(
            observation_id="person_one",
            context_dimension_id="school_membership",
            context_id="school_b",
            membership_weight=0.6,
            revision="2",
        ),
    ]
    effects = {
        ("school_membership", "school_a"): 3.0,
        ("school_membership", "school_b"): 7.0,
    }

    forward = weighted_contextual_effect(
        build_context_membership_design(edges), effects
    )
    reversed_result = weighted_contextual_effect(
        build_context_membership_design(list(reversed(edges))), effects
    )

    assert list(forward) == list(reversed_result)


def test_result_is_identical_across_worker_counts() -> None:
    """The Python-facing worker_count parameter never changes the result."""
    edges = [
        _membership(
            observation_id=f"person_{index:03d}",
            context_dimension_id="team_membership",
            context_id=f"team_{index % 5}",
            membership_weight=1.0,
            revision=f"{index:02x}",
        )
        for index in range(40)
    ]
    design = build_context_membership_design(edges)
    effects = {("team_membership", f"team_{i}"): float(i) * 1.5 for i in range(5)}

    baseline = weighted_contextual_effect(design, effects, worker_count=1)
    for worker_count in (2, 4, 8, 64):
        candidate = weighted_contextual_effect(
            design, effects, worker_count=worker_count
        )
        assert list(baseline) == list(candidate), worker_count


def test_rejects_worker_count_below_one() -> None:
    design = build_context_membership_design(
        [
            _membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                revision="3",
            ),
        ]
    )
    with pytest.raises(ValueError, match="worker_count"):
        weighted_contextual_effect(
            design, {("team_membership", "team_alpha"): 1.0}, worker_count=0
        )


def test_rejects_missing_context_effect_keys() -> None:
    design = build_context_membership_design(
        [
            _membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                revision="4",
            ),
        ]
    )
    with pytest.raises(KeyError):
        weighted_contextual_effect(design, {})


def test_rejects_a_non_exact_context_membership_design() -> None:
    """A hand-crafted (non-factory) design must not reach the Rust marshaller."""

    class FakeDesign:
        pass

    with pytest.raises(ValueError, match="ContextMembershipDesign"):
        weighted_contextual_effect(FakeDesign(), {})  # type: ignore[arg-type]


def test_rejects_a_tampered_design() -> None:
    """Post-factory attribute tampering is caught before Rust ever runs."""
    design = build_context_membership_design(
        [
            _membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                revision="5",
            ),
        ]
    )
    object.__setattr__(design, "schema_version", "9.9")
    assert type(design) is ContextMembershipDesign
    with pytest.raises(Exception):  # noqa: B017 - MultilevelContractError, re-raised as-is
        weighted_contextual_effect(design, {("team_membership", "team_alpha"): 1.0})


def test_rust_core_rejects_malformed_arrays_directly() -> None:
    """Direct core access still fails closed on malformed CSR arrays.

    Exercises the compiled extension module's own error path independent of
    the Python-side marshaller above.
    """
    core = multilevel_core()
    with pytest.raises(ValueError):
        core.weighted_contextual_effect(
            np.array([0, 5], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1,
        )
