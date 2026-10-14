"""Exact package-type boundaries for multilevel design aggregation."""

from __future__ import annotations

import pytest

import fast_mlsirm.multilevel.contracts as contracts
from fast_mlsirm.multilevel import (
    ContextMembership,
    LongitudinalStateSpec,
    MultilevelContractError,
    TemporalOccasion,
    build_context_membership_design,
    build_longitudinal_design,
)


class DerivedMembership(ContextMembership):
    """Untrusted subclass of the package-owned membership record."""


class DerivedOccasion(TemporalOccasion):
    """Untrusted subclass of the package-owned temporal record."""


class DerivedStateSpec(LongitudinalStateSpec):
    """Untrusted subclass of the package-owned state specification."""


def _assert_error(code: str, callback) -> MultilevelContractError:
    """Return one structured error after checking its stable code."""
    with pytest.raises(MultilevelContractError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def test_membership_design_rejects_package_type_subclasses() -> None:
    """Aggregate replay never dispatches through an untrusted membership subclass."""
    value = DerivedMembership(
        observation_id="observation_alpha",
        context_dimension_id="school_context",
        context_id="school_north",
        membership_weight=1.0,
        membership_revision_fingerprint="a" * 64,
        _membership_token=contracts._MEMBERSHIP_TOKEN,
    )

    caught = _assert_error(
        "invalid_context_membership",
        lambda: build_context_membership_design((value,)),
    )

    assert caught.path == "$.memberships[0]"


def test_longitudinal_design_rejects_occasion_subclasses() -> None:
    """Aggregate replay never reads fields through an untrusted occasion subclass."""
    value = DerivedOccasion(
        respondent_id="respondent_alpha",
        occasion_id="occasion_alpha",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="b" * 64,
        _occasion_token=contracts._OCCASION_TOKEN,
    )
    state = contracts.build_longitudinal_state_spec(
        state_kind="random_intercept_slope"
    )

    caught = _assert_error(
        "invalid_temporal_occasion",
        lambda: build_longitudinal_design(occasions=(value,), state_spec=state),
    )

    assert caught.path == "$.occasions[0]"


def test_longitudinal_design_rejects_state_spec_subclasses() -> None:
    """A subclass cannot supply a state model to an aggregate design."""
    state = DerivedStateSpec(
        state_kind=contracts.LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        autoregressive_coefficient=None,
        include_lagged_response_dependence=False,
        _state_spec_token=contracts._STATE_SPEC_TOKEN,
    )
    occasion = contracts.build_temporal_occasion(
        respondent_id="respondent_alpha",
        occasion_id="occasion_alpha",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="c" * 64,
    )

    caught = _assert_error(
        "invalid_longitudinal_state_spec",
        lambda: build_longitudinal_design(
            occasions=(occasion,),
            state_spec=state,
        ),
    )

    assert caught.path == "$.state_spec"
