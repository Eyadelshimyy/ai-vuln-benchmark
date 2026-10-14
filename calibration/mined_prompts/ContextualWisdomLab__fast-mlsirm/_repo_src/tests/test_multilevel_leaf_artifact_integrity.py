"""Post-factory integrity contracts for multilevel leaf artifacts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from fast_mlsirm.multilevel import (
    MultilevelContractError,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)


def _assert_error(code: str, callback: Callable[[], object]) -> None:
    """Require one stable package-owned integrity error."""

    with pytest.raises(MultilevelContractError) as caught:
        callback()
    assert caught.value.code == code
    assert caught.value.path == "$"


def _membership():
    """Return one valid package-owned contextual membership edge."""

    return build_context_membership(
        observation_id="observation_alpha",
        context_dimension_id="school_context",
        context_id="school_north",
        membership_weight=1.0,
        membership_revision_fingerprint="a" * 64,
    )


def _occasion():
    """Return one valid package-owned temporal occasion."""

    return build_temporal_occasion(
        respondent_id="respondent_alpha",
        occasion_id="occasion_alpha",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="b" * 64,
    )


def _state_spec():
    """Return one valid package-owned longitudinal state specification."""

    return build_longitudinal_state_spec(state_kind="random_intercept_slope")


@pytest.mark.parametrize("public_view", ["membership_fingerprint", "membership_handle"])
def test_membership_public_views_reject_post_factory_content_mutation(
    public_view: str,
) -> None:
    """A leaf cannot re-sign a changed observation or contextual assignment."""

    value = _membership()
    object.__setattr__(value, "observation_id", "observation_changed")

    _assert_error(
        "context_membership_integrity_mismatch",
        lambda: getattr(value, public_view),
    )


def test_membership_serialization_rejects_private_seal_mutation() -> None:
    """Changing only the private membership seal remains detectable."""

    value = _membership()
    object.__setattr__(value, "_sealed_fingerprint", "0" * 64)

    _assert_error("context_membership_integrity_mismatch", value.to_dict)


@pytest.mark.parametrize("public_view", ["occasion_fingerprint", "occasion_handle"])
def test_occasion_public_views_reject_post_factory_content_mutation(
    public_view: str,
) -> None:
    """A temporal leaf cannot re-sign a changed ordering field."""

    value = _occasion()
    object.__setattr__(value, "time_offset_milliseconds", 1)

    _assert_error(
        "temporal_occasion_integrity_mismatch",
        lambda: getattr(value, public_view),
    )


def test_occasion_serialization_rejects_private_seal_mutation() -> None:
    """Changing only the private occasion seal remains detectable."""

    value = _occasion()
    object.__setattr__(value, "_sealed_fingerprint", "0" * 64)

    _assert_error("temporal_occasion_integrity_mismatch", value.to_dict)


@pytest.mark.parametrize("public_view", ["state_spec_fingerprint", "state_spec_handle"])
def test_state_public_views_reject_post_factory_content_mutation(
    public_view: str,
) -> None:
    """A state leaf cannot re-sign a changed latent-state contract."""

    value = _state_spec()
    object.__setattr__(value, "include_lagged_response_dependence", True)

    _assert_error(
        "longitudinal_state_spec_integrity_mismatch",
        lambda: getattr(value, public_view),
    )


def test_state_serialization_rejects_private_seal_mutation() -> None:
    """Changing only the private state seal remains detectable."""

    value = _state_spec()
    object.__setattr__(value, "_sealed_fingerprint", "0" * 64)

    _assert_error("longitudinal_state_spec_integrity_mismatch", value.to_dict)


def test_membership_aggregate_rejects_child_seal_only_mutation() -> None:
    """Aggregate verification replays each exact child seal before reporting."""

    design = build_context_membership_design((_membership(),))
    object.__setattr__(design.memberships[0], "_sealed_fingerprint", "0" * 64)

    _assert_error(
        "context_membership_design_integrity_mismatch",
        design.to_dict,
    )


def test_longitudinal_aggregate_rejects_child_seal_only_mutation() -> None:
    """Occasion and state seals remain authoritative inside the aggregate."""

    occasion_design = build_longitudinal_design(
        occasions=(_occasion(),),
        state_spec=_state_spec(),
    )
    object.__setattr__(
        occasion_design.occasions[0],
        "_sealed_fingerprint",
        "0" * 64,
    )
    _assert_error(
        "longitudinal_design_integrity_mismatch",
        occasion_design.to_dict,
    )

    state_design = build_longitudinal_design(
        occasions=(_occasion(),),
        state_spec=_state_spec(),
    )
    object.__setattr__(
        state_design.state_spec,
        "_sealed_fingerprint",
        "0" * 64,
    )
    _assert_error(
        "longitudinal_design_integrity_mismatch",
        state_design.to_dict,
    )
