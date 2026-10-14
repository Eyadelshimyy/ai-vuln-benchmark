"""Callback-safety regressions for scoring-policy integer controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EnginePolicy
from fast_mlsirm.scoring import _contract_safety, _validation


class _HostileIndex:
    """Index provider whose callback records any attempted coercion."""

    calls = 0

    def __index__(self) -> int:
        """Record execution and return an otherwise valid policy value."""
        type(self).calls += 1
        return 2


class _HostileInt(int):
    """Integer subclass that must not cross the trusted control boundary."""


def _assert_rejected_without_index_callback(callable_) -> None:
    """Require one hostile index provider to fail before callback dispatch."""
    _HostileIndex.calls = 0
    with pytest.raises(AssessmentSpecError):
        callable_(_HostileIndex())
    assert _HostileIndex.calls == 0


def test_engine_policy_rejects_index_provider_without_callback() -> None:
    """The public policy boundary rejects arbitrary index providers inertly."""
    _assert_rejected_without_index_callback(
        lambda value: EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
            minimum_raters_per_response=value,
        )
    )


def test_integer_validators_reject_index_provider_without_callback() -> None:
    """Both validator layers reject before caller-controlled coercion."""
    for validator in (
        _validation.bounded_positive_integer,
        _contract_safety.bounded_positive_integer,
    ):
        _assert_rejected_without_index_callback(
            lambda value, validator=validator: validator(
                value,
                "minimum_raters_per_response",
                _validation.MAX_RATERS_PER_RESPONSE,
            )
        )


def test_engine_policy_rejects_integer_subclass() -> None:
    """A caller-defined integer subclass is not a trusted policy control."""
    with pytest.raises(AssessmentSpecError) as captured:
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
            minimum_raters_per_response=_HostileInt(2),
        )

    assert captured.value.code == "invalid_minimum_raters_per_response"
    assert captured.value.path == "$.minimum_raters_per_response"


@pytest.mark.parametrize("value", [1, np.int32(2), np.int64(3), np.uint64(4)])
def test_engine_policy_preserves_trusted_integer_scalars(value: object) -> None:
    """Built-in and genuine NumPy integer scalars retain compatibility."""
    policy = EnginePolicy(
        policy_id="engine_policy",
        engine_ids=(),
        allow_human_raters=True,
        allow_automated_raters=False,
        minimum_raters_per_response=value,  # type: ignore[arg-type]
    )

    assert type(policy.minimum_raters_per_response) is int
    assert policy.minimum_raters_per_response == int(value)
