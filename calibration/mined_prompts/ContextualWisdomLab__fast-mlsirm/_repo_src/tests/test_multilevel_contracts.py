"""Intentional RED contracts for multilevel and longitudinal measurement designs."""

from __future__ import annotations

from itertools import repeat

import pytest

import fast_mlsirm.multilevel.contracts as contracts
from fast_mlsirm.multilevel import (
    ContextMembershipDesign,
    LongitudinalDesign,
    LongitudinalStateKind,
    MultilevelContractError,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)


def _assert_error(code: str, callback) -> MultilevelContractError:
    """Return one stable structured error after asserting its machine code."""
    with pytest.raises(MultilevelContractError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def _membership(
    *,
    observation_id: str = "observation_alpha",
    context_dimension_id: str = "school_context",
    context_id: str = "context_alpha",
    membership_weight: float = 1.0,
    revision: str = "a" * 64,
):
    """Build one deterministic membership edge for focused contracts."""
    return build_context_membership(
        observation_id=observation_id,
        context_dimension_id=context_dimension_id,
        context_id=context_id,
        membership_weight=membership_weight,
        membership_revision_fingerprint=revision,
    )


def _occasion(
    *,
    respondent_id: str = "respondent_alpha",
    occasion_id: str = "occasion_alpha",
    sequence_index: int = 0,
    time_offset_milliseconds: int = 0,
    revision: str = "b" * 64,
):
    """Build one deterministic temporal occasion for focused contracts."""
    return build_temporal_occasion(
        respondent_id=respondent_id,
        occasion_id=occasion_id,
        sequence_index=sequence_index,
        time_offset_milliseconds=time_offset_milliseconds,
        occasion_revision_fingerprint=revision,
    )


def test_one_hot_nesting_is_a_canonical_multiple_membership_design() -> None:
    """Ordinary nesting is retained as the one-edge weight-one special case."""
    design = build_context_membership_design(
        (
            _membership(),
            _membership(
                observation_id="observation_beta",
                context_id="context_beta",
                revision="c" * 64,
            ),
        )
    )

    assert design.observation_ids == (
        "observation_alpha",
        "observation_beta",
    )
    assert design.context_keys == (
        ("school_context", "context_alpha"),
        ("school_context", "context_beta"),
    )
    assert not hasattr(design, "context_ids")
    assert design.membership_counts == (1, 1)
    assert design.membership_weights == ((1.0,), (1.0,))
    assert design.to_dict()["schema_version"] == "1.0"
    assert design.design_handle == f"context_membership_design_{design.design_fingerprint[:32]}"
    assert len(design.design_fingerprint) == 64


def test_weighted_multiple_membership_is_permutation_invariant() -> None:
    """Two- and three-context assignments preserve exact weights and identity."""
    memberships = (
        _membership(
            observation_id="observation_alpha",
            context_id="context_alpha",
            membership_weight=0.25,
            revision="1" * 64,
        ),
        _membership(
            observation_id="observation_alpha",
            context_id="context_beta",
            membership_weight=0.75,
            revision="2" * 64,
        ),
        _membership(
            observation_id="observation_beta",
            context_id="context_alpha",
            membership_weight=0.2,
            revision="3" * 64,
        ),
        _membership(
            observation_id="observation_beta",
            context_id="context_beta",
            membership_weight=0.3,
            revision="4" * 64,
        ),
        _membership(
            observation_id="observation_beta",
            context_id="context_gamma",
            membership_weight=0.5,
            revision="5" * 64,
        ),
    )

    first = build_context_membership_design(memberships)
    second = build_context_membership_design(tuple(reversed(memberships)))

    assert first.to_dict() == second.to_dict()
    assert first.design_fingerprint == second.design_fingerprint
    assert first.membership_weights == ((0.25, 0.75), (0.2, 0.3, 0.5))
    assert [
        value["context_id"] for value in first.to_dict()["memberships"][:2]
    ] == ["context_alpha", "context_beta"]


@pytest.mark.parametrize(
    "invalid_weight",
    [True, False, 0, -0.0, -0.1, 1.1, float("nan"), float("inf"), -float("inf")],
)
def test_membership_weights_reject_invalid_numeric_values(invalid_weight: object) -> None:
    """Weights are finite real values in the open-zero, closed-one interval."""
    caught = _assert_error(
        "invalid_membership_weight",
        lambda: _membership(membership_weight=invalid_weight),  # type: ignore[arg-type]
    )
    assert caught.path == "$.membership_weight"
    assert repr(invalid_weight) not in str(caught)


def test_membership_weight_totals_fail_closed_without_silent_renormalization() -> None:
    """Materially invalid observation totals are rejected rather than rescaled."""
    memberships = (
        _membership(membership_weight=0.4, revision="1" * 64),
        _membership(
            context_id="context_beta",
            membership_weight=0.5,
            revision="2" * 64,
        ),
    )

    caught = _assert_error(
        "membership_weight_total_mismatch",
        lambda: build_context_membership_design(memberships),
    )
    assert caught.path == "$.memberships"
    assert "0.9" not in str(caught)


def test_duplicate_membership_cells_and_revision_rebinding_are_distinct() -> None:
    """Duplicate cells and conflicting revision provenance have stable errors."""
    duplicate = _membership()
    caught = _assert_error(
        "duplicate_context_membership",
        lambda: build_context_membership_design((duplicate, duplicate)),
    )
    assert caught.path.startswith("$.memberships[")

    rebound = (
        _membership(
            observation_id="observation_alpha",
            context_id="context_alpha",
            membership_weight=0.5,
            revision="d" * 64,
        ),
        _membership(
            observation_id="observation_beta",
            context_id="context_beta",
            membership_weight=0.5,
            revision="d" * 64,
        ),
        _membership(
            observation_id="observation_alpha",
            context_id="context_beta",
            membership_weight=0.5,
            revision="e" * 64,
        ),
        _membership(
            observation_id="observation_beta",
            context_id="context_alpha",
            membership_weight=0.5,
            revision="f" * 64,
        ),
    )
    caught = _assert_error(
        "membership_revision_conflict",
        lambda: build_context_membership_design(rebound),
    )
    assert caught.path.endswith(".membership_revision_fingerprint")
    assert "d" * 64 not in str(caught)


def test_membership_design_is_factory_sealed_and_resource_bounded(monkeypatch) -> None:
    """Aggregate provenance cannot be forged or materialized without a bound."""
    with pytest.raises(MultilevelContractError) as caught:
        ContextMembershipDesign(  # type: ignore[call-arg]
            memberships=(_membership(),),
        )
    assert caught.value.code == "unverified_context_membership_design"

    monkeypatch.setattr(contracts, "MAX_CONTEXT_MEMBERSHIPS", 3)
    caught = _assert_error(
        "invalid_memberships",
        lambda: build_context_membership_design(repeat(_membership())),
    )
    assert caught.path == "$.memberships"


def test_temporal_design_preserves_irregular_order_and_permutation_identity() -> None:
    """Respondent sequences retain exact irregular intervals in canonical order."""
    occasions = (
        _occasion(),
        _occasion(
            occasion_id="occasion_beta",
            sequence_index=1,
            time_offset_milliseconds=86_400_000,
            revision="c" * 64,
        ),
        _occasion(
            occasion_id="occasion_gamma",
            sequence_index=2,
            time_offset_milliseconds=302_400_000,
            revision="d" * 64,
        ),
        _occasion(
            respondent_id="respondent_beta",
            occasion_id="occasion_delta",
            sequence_index=0,
            time_offset_milliseconds=1_000,
            revision="e" * 64,
        ),
    )
    state = build_longitudinal_state_spec(
        state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        include_lagged_response_dependence=False,
    )

    first = build_longitudinal_design(occasions=occasions, state_spec=state)
    second = build_longitudinal_design(
        occasions=tuple(reversed(occasions)),
        state_spec=state,
    )

    assert first.to_dict() == second.to_dict()
    assert first.respondent_ids == ("respondent_alpha", "respondent_beta")
    assert first.occasion_counts == (3, 1)
    assert first.time_offsets_milliseconds == (
        (0, 86_400_000, 302_400_000),
        (1_000,),
    )
    assert first.design_handle == f"longitudinal_design_{first.design_fingerprint[:32]}"


@pytest.mark.parametrize("invalid_integer", [True, False, 1.5, "1"])
def test_temporal_indices_are_exact_integers(invalid_integer: object) -> None:
    """Sequence and time fields reject Boolean, fractional, and text coercion."""
    sequence_error = _assert_error(
        "invalid_sequence_index",
        lambda: _occasion(sequence_index=invalid_integer),  # type: ignore[arg-type]
    )
    assert sequence_error.path == "$.sequence_index"

    offset_error = _assert_error(
        "invalid_time_offset_milliseconds",
        lambda: _occasion(time_offset_milliseconds=invalid_integer),  # type: ignore[arg-type]
    )
    assert offset_error.path == "$.time_offset_milliseconds"


def test_temporal_identity_and_order_conflicts_fail_before_design_creation() -> None:
    """Occasion, sequence, and time collisions remain distinguishable."""
    first = _occasion()
    duplicate_id = _occasion(
        occasion_id="occasion_alpha",
        sequence_index=1,
        time_offset_milliseconds=1,
        revision="c" * 64,
    )
    caught = _assert_error(
        "duplicate_temporal_occasion",
        lambda: build_longitudinal_design(
            occasions=(first, duplicate_id),
            state_spec=build_longitudinal_state_spec(
                state_kind="random_intercept_slope",
            ),
        ),
    )
    assert caught.path.endswith(".occasion_id")

    duplicate_sequence = _occasion(
        occasion_id="occasion_beta",
        sequence_index=0,
        time_offset_milliseconds=1,
        revision="d" * 64,
    )
    caught = _assert_error(
        "duplicate_temporal_sequence",
        lambda: build_longitudinal_design(
            occasions=(first, duplicate_sequence),
            state_spec=build_longitudinal_state_spec(
                state_kind="random_intercept_slope",
            ),
        ),
    )
    assert caught.path.endswith(".sequence_index")

    duplicate_time = _occasion(
        occasion_id="occasion_beta",
        sequence_index=1,
        time_offset_milliseconds=0,
        revision="e" * 64,
    )
    caught = _assert_error(
        "duplicate_temporal_offset",
        lambda: build_longitudinal_design(
            occasions=(first, duplicate_time),
            state_spec=build_longitudinal_state_spec(
                state_kind="random_intercept_slope",
            ),
        ),
    )
    assert caught.path.endswith(".time_offset_milliseconds")

    reversed_time = _occasion(
        occasion_id="occasion_beta",
        sequence_index=1,
        time_offset_milliseconds=-1,
        revision="f" * 64,
    )
    caught = _assert_error(
        "nonincreasing_temporal_order",
        lambda: build_longitudinal_design(
            occasions=(first, reversed_time),
            state_spec=build_longitudinal_state_spec(
                state_kind="random_intercept_slope",
            ),
        ),
    )
    assert caught.path == "$.occasions"


def test_longitudinal_state_models_validate_ar_and_lag_contracts() -> None:
    """Growth, AR state, and lagged response remain independent contracts."""
    growth = build_longitudinal_state_spec(
        state_kind="random_intercept_slope",
        include_lagged_response_dependence=True,
    )
    assert growth.state_kind is LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE
    assert growth.autoregressive_coefficient is None
    assert growth.include_lagged_response_dependence is True

    autoregressive = build_longitudinal_state_spec(
        state_kind="stationary_autoregressive",
        autoregressive_coefficient=0.75,
        include_lagged_response_dependence=False,
    )
    assert autoregressive.state_kind is LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE
    assert autoregressive.autoregressive_coefficient == 0.75
    assert autoregressive.include_lagged_response_dependence is False

    for invalid_coefficient in (None, True, -1.0, 1.0, float("nan"), float("inf")):
        caught = _assert_error(
            "invalid_autoregressive_coefficient",
            lambda value=invalid_coefficient: build_longitudinal_state_spec(
                state_kind="stationary_autoregressive",
                autoregressive_coefficient=value,
            ),
        )
        assert caught.path == "$.autoregressive_coefficient"
        assert repr(invalid_coefficient) not in str(caught)

    caught = _assert_error(
        "unexpected_autoregressive_coefficient",
        lambda: build_longitudinal_state_spec(
            state_kind="random_intercept_slope",
            autoregressive_coefficient=0.25,
        ),
    )
    assert caught.path == "$.autoregressive_coefficient"

    caught = _assert_error(
        "invalid_include_lagged_response_dependence",
        lambda: build_longitudinal_state_spec(
            state_kind="random_intercept_slope",
            include_lagged_response_dependence=1,  # type: ignore[arg-type]
        ),
    )
    assert caught.path == "$.include_lagged_response_dependence"


def test_longitudinal_design_is_factory_sealed_and_source_text_free() -> None:
    """Aggregate design identity is package-owned and contains no raw text."""
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    with pytest.raises(MultilevelContractError) as caught:
        LongitudinalDesign(  # type: ignore[call-arg]
            occasions=(_occasion(),),
            state_spec=state,
        )
    assert caught.value.code == "unverified_longitudinal_design"

    design = build_longitudinal_design(
        occasions=(_occasion(),),
        state_spec=state,
    )
    serialized = repr(design.to_dict()).lower()
    for forbidden in (
        "raw_response",
        "response_text",
        "source_text",
        "prompt_text",
        "provider_output",
    ):
        assert forbidden not in serialized
