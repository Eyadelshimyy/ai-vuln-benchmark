"""Hostile numeric callback boundaries for multilevel contracts."""

from __future__ import annotations

import pytest

from fast_mlsirm.multilevel import (
    LongitudinalStateKind,
    MultilevelContractError,
    build_context_membership,
    build_longitudinal_state_spec,
)


class ExplosiveFloat(float):
    """Float subclass that exposes whether validation dispatches callbacks."""

    def __float__(self) -> float:
        """Raise an attacker-controlled exception instead of coercing."""
        raise RuntimeError("sensitive_numeric_callback")


def test_membership_weight_rejects_numeric_subclass_without_callback() -> None:
    """Membership validation fails closed before untrusted numeric coercion."""
    value = ExplosiveFloat(1.0)

    with pytest.raises(MultilevelContractError) as caught:
        build_context_membership(
            observation_id="observation_alpha",
            context_dimension_id="school_context",
            context_id="school_north",
            membership_weight=value,
            membership_revision_fingerprint="a" * 64,
        )

    assert caught.value.code == "invalid_membership_weight"
    assert caught.value.path == "$.membership_weight"
    assert "sensitive_numeric_callback" not in str(caught.value)


def test_autoregressive_coefficient_rejects_numeric_subclass_without_callback() -> None:
    """AR-state validation fails closed before untrusted numeric coercion."""
    value = ExplosiveFloat(0.5)

    with pytest.raises(MultilevelContractError) as caught:
        build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE,
            autoregressive_coefficient=value,
        )

    assert caught.value.code == "invalid_autoregressive_coefficient"
    assert caught.value.path == "$.autoregressive_coefficient"
    assert "sensitive_numeric_callback" not in str(caught.value)
