"""Continuous-evidence trust-boundary tests for crossed person-effect estimation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.multilevel.estimation as estimation
from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
)


def _revision(tag: str) -> str:
    """Return one deterministic content fingerprint for the test fixture."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _design():
    """Build the smallest identified one-classification membership design."""
    return build_context_membership_design(
        [
            build_context_membership(
                observation_id="person_alpha",
                context_dimension_id="school_membership",
                context_id="school_east",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("alpha-east"),
            ),
            build_context_membership(
                observation_id="person_beta",
                context_dimension_id="school_membership",
                context_id="school_west",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("beta-west"),
            ),
        ]
    )


def _python_bool(size: int) -> object:
    """Return Boolean continuous evidence in an ordinary built-in container."""
    return [True] * size


def _numpy_bool_scalar(size: int) -> object:
    """Return NumPy Boolean scalar evidence in an ordinary built-in container."""
    return [np.bool_(True)] * size


def _numpy_bool_array(size: int) -> object:
    """Return one exact NumPy Boolean evidence vector."""
    return np.ones(size, dtype=np.bool_)


@pytest.mark.parametrize(
    ("field", "size"),
    [
        ("item_intercepts", 1),
        ("item_slopes", 1),
        ("person_offsets", 2),
    ],
)
@pytest.mark.parametrize(
    "factory",
    [_python_bool, _numpy_bool_scalar, _numpy_bool_array],
)
def test_boolean_continuous_evidence_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    size: int,
    factory: Callable[[int], object],
) -> None:
    """Boolean item/person parameters must not be reinterpreted as 0.0/1.0."""
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not see Boolean continuous evidence")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)
    kwargs: dict[str, object] = {
        "responses": [[True], [False]],
        "design": _design(),
        "item_intercepts": [0.0],
        "item_slopes": [1.0],
        "person_offsets": [0.0, 0.0],
        "device": "cpu",
    }
    kwargs[field] = factory(size)

    with pytest.raises(ValueError, match=rf"{field} must be real-valued numeric evidence"):
        estimation.estimate_crossed_person_effects(**kwargs)

    assert core_discoveries == 0


def test_boolean_responses_remain_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Boolean exclusion is field-specific and must not narrow response input."""
    captured: dict[str, object] = {}

    class _Core:
        @staticmethod
        def estimate_crossed_person_effects(*args: object, **kwargs: object):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "effects": np.zeros(2, dtype=np.float64),
                "loglik": 0.0,
                "n_iter": 1,
                "converged": True,
                "used_gpu": False,
                "termination_reason": "converged",
            }

    monkeypatch.setattr(estimation, "multilevel_core", lambda: _Core())

    result = estimation.estimate_crossed_person_effects(
        np.array([[True], [False]], dtype=np.bool_),
        _design(),
        item_intercepts=[0.0],
        item_slopes=[1.0],
        person_offsets=[0.0, 0.0],
        device="cpu",
    )

    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[0].dtype == np.float64
    assert np.array_equal(args[0], np.array([1.0, 0.0], dtype=np.float64))
    assert result.converged is True
