"""Binary-response trust-boundary tests for crossed person-effect estimation."""

from __future__ import annotations

import hashlib

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


class _HostileArrayProvider:
    """Array provider that must never run during package evidence admission."""

    callback_count = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callback_count += 1
        raise AssertionError("caller __array__ must not execute")


class _HostileNumericProvider:
    """Numeric provider that must never run during nested evidence admission."""

    callback_count = 0

    def __float__(self) -> float:
        type(self).callback_count += 1
        raise AssertionError("caller __float__ must not execute")


@pytest.mark.parametrize("invalid_response", [0.5, 1.5, 2.0])
def test_nonbinary_observed_response_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: float,
) -> None:
    """Finite observed cells outside {0, 1} must never reach the Rust core."""
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not be discovered for invalid responses")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)
    responses = np.array([[invalid_response], [0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="binary responses"):
        estimation.estimate_crossed_person_effects(
            responses,
            _design(),
            item_intercepts=np.array([0.0], dtype=np.float64),
            device="cpu",
        )

    assert core_discoveries == 0


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("responses", "responses must be a numeric array"),
        ("item_intercepts", "item_intercepts must be a numeric array"),
        ("item_slopes", "item_slopes must be a numeric array"),
        ("person_offsets", "person_offsets must be a numeric array"),
    ],
)
def test_array_providers_fail_before_callbacks_or_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected_error: str,
) -> None:
    """Scientific evidence providers must be rejected before NumPy protocols run."""
    _HostileArrayProvider.callback_count = 0
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not be discovered for hostile evidence")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)
    kwargs: dict[str, object] = {
        "responses": np.array([[1.0], [0.0]], dtype=np.float64),
        "design": _design(),
        "item_intercepts": np.array([0.0], dtype=np.float64),
        "item_slopes": np.array([1.0], dtype=np.float64),
        "person_offsets": np.array([0.0, 0.0], dtype=np.float64),
        "device": "cpu",
    }
    kwargs[field] = _HostileArrayProvider()

    with pytest.raises(ValueError, match=expected_error):
        estimation.estimate_crossed_person_effects(**kwargs)

    assert _HostileArrayProvider.callback_count == 0
    assert core_discoveries == 0


def test_nested_numeric_provider_fails_before_conversion_or_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested vector cells cannot run numeric conversion during admission."""
    _HostileNumericProvider.callback_count = 0
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not be discovered for hostile evidence")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="item_intercepts must be a numeric array"):
        estimation.estimate_crossed_person_effects(
            [[1.0], [0.0]],
            _design(),
            item_intercepts=[_HostileNumericProvider()],
            device="cpu",
        )

    assert _HostileNumericProvider.callback_count == 0
    assert core_discoveries == 0


def test_invalid_controls_fail_before_hostile_evidence_is_inspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safety adapter must preserve semantic-control-before-data ordering."""
    _HostileArrayProvider.callback_count = 0
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not be discovered for invalid controls")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="max_iter"):
        estimation.estimate_crossed_person_effects(
            _HostileArrayProvider(),
            _design(),
            item_intercepts=[0.0],
            max_iter=0,
            device="cpu",
        )

    assert _HostileArrayProvider.callback_count == 0
    assert core_discoveries == 0


def test_builtin_and_numpy_scalar_evidence_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain containers with concrete NumPy scalars normalize to Rust payloads."""
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
        [[np.int8(1)], [np.float32(0.0)]],
        _design(),
        item_intercepts=[np.float32(0.25)],
        item_slopes=(np.uint8(1),),
        person_offsets=[np.float64(0.0), np.int16(0)],
        device="cpu",
    )

    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[0].dtype == np.float64
    assert args[4].dtype == np.float64
    assert args[5].dtype == np.float64
    assert args[6].dtype == np.float64
    assert result.converged is True


@pytest.mark.parametrize(
    ("field", "lossy_value"),
    [
        ("item_intercepts", 2**53 + 1),
        ("item_slopes", np.uint64(2**53 + 1)),
        ("person_offsets", np.int64(2**53 + 1)),
    ],
)
def test_lossy_integer_vector_evidence_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    lossy_value: object,
) -> None:
    """Exact integers must not change identity during float64 marshalling."""
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not see lossy vector evidence")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)
    kwargs: dict[str, object] = {
        "responses": [[1.0], [0.0]],
        "design": _design(),
        "item_intercepts": [0.0],
        "item_slopes": [1.0],
        "person_offsets": [0.0, 0.0],
        "device": "cpu",
    }
    kwargs[field] = [lossy_value] if field != "person_offsets" else [lossy_value, 0]

    with pytest.raises(ValueError, match=rf"{field} could not be converted losslessly"):
        estimation.estimate_crossed_person_effects(**kwargs)

    assert core_discoveries == 0


def test_exact_integer_float64_boundary_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The largest consecutive binary64 integer remains valid evidence."""
    captured: dict[str, object] = {}

    class _Core:
        @staticmethod
        def estimate_crossed_person_effects(*args: object, **kwargs: object):
            captured["args"] = args
            return {
                "effects": np.zeros(2, dtype=np.float64),
                "loglik": 0.0,
                "n_iter": 1,
                "converged": True,
                "used_gpu": False,
                "termination_reason": "converged",
            }

    monkeypatch.setattr(estimation, "multilevel_core", lambda: _Core())

    estimation.estimate_crossed_person_effects(
        [[1.0], [0.0]],
        _design(),
        item_intercepts=[2**53],
        item_slopes=[1],
        person_offsets=[np.uint64(2**53), 0],
        device="cpu",
    )

    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[5][0] == float(2**53)
    assert args[6][0] == float(2**53)


def test_extended_precision_real_evidence_must_not_round_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wider NumPy real values must retain identity or fail before Rust."""
    wide = np.longdouble(1.0) + np.finfo(np.longdouble).eps
    if np.longdouble(np.float64(wide)) == wide:
        pytest.skip("np.longdouble has no precision beyond float64 on this platform")

    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not see rounded longdouble evidence")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="item_intercepts could not be converted losslessly"):
        estimation.estimate_crossed_person_effects(
            [[1.0], [0.0]],
            _design(),
            item_intercepts=[wide],
            device="cpu",
        )

    assert core_discoveries == 0
