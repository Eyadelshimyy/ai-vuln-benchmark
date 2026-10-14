"""Aggregate seal regressions for multilevel and longitudinal artifacts."""

from __future__ import annotations

import pytest

from fast_mlsirm.multilevel import (
    MultilevelContractError,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)


def _assert_error(code: str, callback) -> MultilevelContractError:
    """Return one structured error after checking its stable code."""
    with pytest.raises(MultilevelContractError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def _membership_design():
    """Return one valid two-dimension membership design."""
    return build_context_membership_design(
        (
            build_context_membership(
                observation_id="observation_alpha",
                context_dimension_id="school_context",
                context_id="school_north",
                membership_weight=1.0,
                membership_revision_fingerprint="a" * 64,
            ),
            build_context_membership(
                observation_id="observation_alpha",
                context_dimension_id="site_context",
                context_id="site_east",
                membership_weight=1.0,
                membership_revision_fingerprint="b" * 64,
            ),
        )
    )


def _longitudinal_design():
    """Return one valid two-occasion longitudinal design."""
    return build_longitudinal_design(
        occasions=(
            build_temporal_occasion(
                respondent_id="respondent_alpha",
                occasion_id="occasion_baseline",
                sequence_index=0,
                time_offset_milliseconds=0,
                occasion_revision_fingerprint="c" * 64,
            ),
            build_temporal_occasion(
                respondent_id="respondent_alpha",
                occasion_id="occasion_followup",
                sequence_index=1,
                time_offset_milliseconds=86_400_000,
                occasion_revision_fingerprint="d" * 64,
            ),
        ),
        state_spec=build_longitudinal_state_spec(
            state_kind="random_intercept_slope"
        ),
    )


def test_membership_design_rejects_post_factory_child_replacement() -> None:
    """A design cannot silently re-sign a different membership collection."""
    design = _membership_design()
    original_fingerprint = design.design_fingerprint
    object.__setattr__(design, "memberships", design.memberships[:1])

    caught = _assert_error(
        "context_membership_design_integrity_mismatch",
        design.to_dict,
    )

    assert caught.path == "$"
    assert original_fingerprint not in str(caught)


def test_longitudinal_design_rejects_post_factory_occasion_replacement() -> None:
    """A design cannot silently re-sign a different occasion collection."""
    design = _longitudinal_design()
    original_fingerprint = design.design_fingerprint
    object.__setattr__(design, "occasions", design.occasions[:1])

    caught = _assert_error(
        "longitudinal_design_integrity_mismatch",
        design.to_dict,
    )

    assert caught.path == "$"
    assert original_fingerprint not in str(caught)


def test_longitudinal_design_rejects_post_factory_state_replacement() -> None:
    """A design cannot silently bind a different latent-state specification."""
    design = _longitudinal_design()
    object.__setattr__(
        design,
        "state_spec",
        build_longitudinal_state_spec(
            state_kind="stationary_autoregressive",
            autoregressive_coefficient=0.25,
        ),
    )

    caught = _assert_error(
        "longitudinal_design_integrity_mismatch",
        design.to_dict,
    )

    assert caught.path == "$"
