"""Dimension-qualified identity contracts for contextual measurement levels."""

from __future__ import annotations

from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
)


def test_same_context_label_in_two_dimensions_remains_two_levels() -> None:
    """Bare labels cannot collapse random-effect levels across classifications."""

    design = build_context_membership_design(
        (
            build_context_membership(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="shared_context",
                membership_weight=1.0,
                membership_revision_fingerprint="a" * 64,
            ),
            build_context_membership(
                observation_id="observation_alpha",
                context_dimension_id="site_context",
                context_id="shared_context",
                membership_weight=1.0,
                membership_revision_fingerprint="b" * 64,
            ),
        )
    )

    expected_keys = (
        ("school_context", "shared_context"),
        ("site_context", "shared_context"),
    )
    assert design.context_keys == expected_keys
    assert not hasattr(design, "context_ids")

    payload = design.to_dict()
    assert payload["context_keys"] == [list(value) for value in expected_keys]
    assert "context_ids" not in payload
    assert len(payload["memberships"]) == 2
    assert {
        (value["context_dimension_id"], value["context_id"])
        for value in payload["memberships"]
    } == set(expected_keys)


def test_dimension_qualified_identity_changes_the_design_fingerprint() -> None:
    """Changing only the classification changes the governed design identity."""

    school_design = build_context_membership_design(
        (
            build_context_membership(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="shared_context",
                membership_weight=1.0,
                membership_revision_fingerprint="c" * 64,
            ),
        )
    )
    site_design = build_context_membership_design(
        (
            build_context_membership(
                observation_id="observation_alpha",
                context_dimension_id="site_context",
                context_id="shared_context",
                membership_weight=1.0,
                membership_revision_fingerprint="c" * 64,
            ),
        )
    )

    assert school_design.context_keys != site_design.context_keys
    assert school_design.design_fingerprint != site_design.design_fingerprint
