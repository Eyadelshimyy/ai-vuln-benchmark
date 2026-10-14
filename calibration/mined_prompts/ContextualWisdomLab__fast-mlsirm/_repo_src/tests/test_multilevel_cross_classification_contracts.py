"""Cross-classified contextual-membership contracts for multilevel designs."""

from __future__ import annotations

import inspect

import pytest

from fast_mlsirm.multilevel import (
    MultilevelContractError,
    build_context_membership,
    build_context_membership_design,
)


def _edge(
    *,
    observation_id: str,
    context_dimension_id: str,
    context_id: str,
    weight: float,
    revision: str,
):
    """Build one deterministic observation-context edge."""
    return build_context_membership(
        observation_id=observation_id,
        context_dimension_id=context_dimension_id,
        context_id=context_id,
        membership_weight=weight,
        membership_revision_fingerprint=revision,
    )


def _assert_error(code: str, callback) -> MultilevelContractError:
    """Return one structured contract error after checking its machine code."""
    with pytest.raises(MultilevelContractError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def test_context_dimension_is_an_explicit_required_factory_field() -> None:
    """Schema 1.0 never invents a random-effect family for the caller."""
    parameter = inspect.signature(build_context_membership).parameters[
        "context_dimension_id"
    ]
    assert parameter.default is inspect.Parameter.empty


def test_cross_classified_dimensions_each_normalize_independently() -> None:
    """School and site classifications each carry a complete unit-weight design."""
    design = build_context_membership_design(
        (
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="school_north",
                weight=1.0,
                revision="1" * 64,
            ),
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="site_context",
                context_id="site_east",
                weight=1.0,
                revision="2" * 64,
            ),
            _edge(
                observation_id="observation_beta",
                context_dimension_id="school_context",
                context_id="school_south",
                weight=1.0,
                revision="3" * 64,
            ),
            _edge(
                observation_id="observation_beta",
                context_dimension_id="site_context",
                context_id="site_east",
                weight=1.0,
                revision="4" * 64,
            ),
        )
    )

    assert design.context_dimension_ids == ("school_context", "site_context")
    assert design.observation_ids == ("observation_alpha", "observation_beta")
    assert design.membership_counts_by_dimension == ((1, 1), (1, 1))
    assert design.membership_weights_by_dimension == (
        ((1.0,), (1.0,)),
        ((1.0,), (1.0,)),
    )
    serialized = design.to_dict()
    assert serialized["context_dimension_ids"] == ["school_context", "site_context"]
    assert serialized["context_keys"] == [
        ["school_context", "school_north"],
        ["school_context", "school_south"],
        ["site_context", "site_east"],
    ]
    assert serialized["membership_counts_by_dimension"] == [[1, 1], [1, 1]]
    assert serialized["membership_weights_by_dimension"] == [
        [[1.0], [1.0]],
        [[1.0], [1.0]],
    ]
    assert all("context_dimension_id" in row for row in serialized["memberships"])


def test_weighted_membership_normalizes_within_each_context_dimension() -> None:
    """Multiple school membership can coexist with a one-hot site classification."""
    design = build_context_membership_design(
        (
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="school_north",
                weight=0.25,
                revision="1" * 64,
            ),
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="school_south",
                weight=0.75,
                revision="2" * 64,
            ),
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="site_context",
                context_id="site_east",
                weight=1.0,
                revision="3" * 64,
            ),
        )
    )

    assert design.membership_counts_by_dimension == ((2, 1),)
    assert design.membership_weights_by_dimension == (((0.25, 0.75), (1.0,)),)
    assert design.to_dict()["membership_weights_by_dimension"] == [
        [[0.25, 0.75], [1.0]]
    ]


def test_context_identity_is_scoped_by_dimension() -> None:
    """The same context label in two classifications remains two random-effect levels."""
    design = build_context_membership_design(
        (
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="context_shared",
                weight=1.0,
                revision="1" * 64,
            ),
            _edge(
                observation_id="observation_alpha",
                context_dimension_id="site_context",
                context_id="context_shared",
                weight=1.0,
                revision="2" * 64,
            ),
        )
    )

    assert design.context_keys == (
        ("school_context", "context_shared"),
        ("site_context", "context_shared"),
    )


def test_duplicate_membership_is_dimension_scoped() -> None:
    """Only an exact observation-dimension-context cell is a duplicate."""
    first = _edge(
        observation_id="observation_alpha",
        context_dimension_id="school_context",
        context_id="context_shared",
        weight=1.0,
        revision="1" * 64,
    )
    duplicate = _edge(
        observation_id="observation_alpha",
        context_dimension_id="school_context",
        context_id="context_shared",
        weight=1.0,
        revision="2" * 64,
    )

    caught = _assert_error(
        "duplicate_context_membership",
        lambda: build_context_membership_design((first, duplicate)),
    )
    assert caught.path.startswith("$.memberships[")


def test_every_observation_has_every_declared_context_dimension() -> None:
    """A silently missing classification cannot be interpreted as a zero random effect."""
    memberships = (
        _edge(
            observation_id="observation_alpha",
            context_dimension_id="school_context",
            context_id="school_north",
            weight=1.0,
            revision="1" * 64,
        ),
        _edge(
            observation_id="observation_alpha",
            context_dimension_id="site_context",
            context_id="site_east",
            weight=1.0,
            revision="2" * 64,
        ),
        _edge(
            observation_id="observation_beta",
            context_dimension_id="school_context",
            context_id="school_south",
            weight=1.0,
            revision="3" * 64,
        ),
    )

    caught = _assert_error(
        "missing_context_dimension_membership",
        lambda: build_context_membership_design(memberships),
    )
    assert caught.path == "$.memberships"


def test_revision_fingerprint_binds_dimension_context_and_weight() -> None:
    """One revision digest cannot be replayed for a different weighted assignment."""
    revision = "d" * 64
    memberships = (
        _edge(
            observation_id="observation_alpha",
            context_dimension_id="school_context",
            context_id="school_north",
            weight=0.25,
            revision=revision,
        ),
        _edge(
            observation_id="observation_alpha",
            context_dimension_id="school_context",
            context_id="school_south",
            weight=0.75,
            revision=revision,
        ),
    )

    caught = _assert_error(
        "membership_revision_conflict",
        lambda: build_context_membership_design(memberships),
    )
    assert caught.path.endswith(".membership_revision_fingerprint")
    assert revision not in str(caught)
