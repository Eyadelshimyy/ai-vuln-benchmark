"""Integrity and hostile-callback contracts for contextual design artifacts."""

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
    """Return one structured error after checking its stable machine code."""
    with pytest.raises(MultilevelContractError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def test_membership_mutation_is_rejected_before_aggregate_identity() -> None:
    """A frozen child changed through ``object.__setattr__`` cannot be re-signed."""
    edge = build_context_membership(
        observation_id="observation_alpha",
        context_dimension_id="school_context",
        context_id="school_north",
        membership_weight=1.0,
        membership_revision_fingerprint="a" * 64,
    )
    object.__setattr__(edge, "membership_weight", 0.5)

    caught = _assert_error(
        "context_membership_integrity_mismatch",
        lambda: build_context_membership_design((edge,)),
    )
    assert caught.path.startswith("$.memberships[")
    assert "0.5" not in str(caught)


def test_temporal_occasion_mutation_is_rejected_before_ordering() -> None:
    """A changed sequence or revision never reaches sorting or design hashing."""
    occasion = build_temporal_occasion(
        respondent_id="respondent_alpha",
        occasion_id="occasion_alpha",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="b" * 64,
    )
    object.__setattr__(occasion, "sequence_index", True)

    caught = _assert_error(
        "temporal_occasion_integrity_mismatch",
        lambda: build_longitudinal_design(
            occasions=(occasion,),
            state_spec=build_longitudinal_state_spec(
                state_kind="random_intercept_slope"
            ),
        ),
    )
    assert caught.path.startswith("$.occasions[")
    assert "True" not in str(caught)


def test_state_spec_mutation_is_rejected_before_design_identity() -> None:
    """A post-construction state-kind change cannot authorize another model."""
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    object.__setattr__(state, "state_kind", "stationary_autoregressive")
    occasion = build_temporal_occasion(
        respondent_id="respondent_alpha",
        occasion_id="occasion_alpha",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="c" * 64,
    )

    caught = _assert_error(
        "longitudinal_state_spec_integrity_mismatch",
        lambda: build_longitudinal_design(
            occasions=(occasion,),
            state_spec=state,
        ),
    )
    assert caught.path == "$.state_spec"


def test_hostile_membership_iterator_is_redacted() -> None:
    """Ordinary iterator exceptions become non-reflective domain errors."""

    class HostileIterator:
        """Raise one caller-controlled ordinary exception during iteration."""

        def __iter__(self):
            """Return this object as its iterator."""
            return self

        def __next__(self):
            """Raise an exception containing a private caller value."""
            raise RuntimeError("private_membership_payload")

    caught = _assert_error(
        "invalid_memberships",
        lambda: build_context_membership_design(HostileIterator()),
    )
    assert caught.path == "$.memberships"
    assert "private_membership_payload" not in str(caught)


def test_hostile_state_value_is_redacted() -> None:
    """Enum normalization cannot leak caller-controlled comparison failures."""

    class HostileStateValue:
        """Raise from equality used by enum value lookup."""

        def __eq__(self, _other):
            """Raise an ordinary exception with a private marker."""
            raise RuntimeError("private_state_payload")

    caught = _assert_error(
        "invalid_state_kind",
        lambda: build_longitudinal_state_spec(
            state_kind=HostileStateValue(),  # type: ignore[arg-type]
        ),
    )
    assert caught.path == "$.state_kind"
    assert "private_state_payload" not in str(caught)


def test_negative_zero_ar_coefficient_has_canonical_identity() -> None:
    """Positive and negative zero cannot create distinct state fingerprints."""
    positive = build_longitudinal_state_spec(
        state_kind="stationary_autoregressive",
        autoregressive_coefficient=0.0,
    )
    negative = build_longitudinal_state_spec(
        state_kind="stationary_autoregressive",
        autoregressive_coefficient=-0.0,
    )

    assert positive.to_dict() == negative.to_dict()
    assert positive.state_spec_fingerprint == negative.state_spec_fingerprint
