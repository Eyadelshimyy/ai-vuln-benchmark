"""Callback-safety regressions for model-comparison control metadata."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    compare_nonnested_models,
)


_CASEWISE_A = (0.0, 0.1)
_CASEWISE_B = (0.0, 0.1)


class _HostileIndex:
    """Arbitrary integer protocol provider whose callback must stay inert."""

    calls = 0

    def __index__(self) -> int:
        """Record forbidden integer coercion."""
        type(self).calls += 1
        return 1


class _HostileInt(int):
    """Caller-defined integer subclass outside the trusted control boundary."""

    calls = 0

    def __index__(self) -> int:
        """Record forbidden integer coercion if dispatched."""
        type(self).calls += 1
        return int.__index__(self)


class _HostileStr(str):
    """Caller-defined string subclass whose normalization must not execute."""

    calls = 0

    def strip(self, *args: object, **kwargs: object) -> str:
        """Record forbidden label normalization."""
        type(self).calls += 1
        return str.strip(self, *args, **kwargs)


class _HostileNumpyFloat(np.float64):
    """NumPy-scalar subclass spoofing trusted-looking module metadata."""

    __module__ = "numpy.user_controlled"
    calls = 0

    def __float__(self) -> float:
        """Record forbidden real-scalar normalization."""
        type(self).calls += 1
        return np.float64.__float__(self)


def _compare(**overrides: object):
    """Call the public comparison boundary without requiring native dispatch."""
    kwargs: dict[str, object] = {
        "k_a": 1,
        "k_b": 1,
        "model_a": "A",
        "model_b": "B",
        "relation": ModelRelation.UNKNOWN,
        "alpha": 0.05,
        "omega_tol": 1e-12,
    }
    kwargs.update(overrides)
    return compare_nonnested_models(
        _CASEWISE_A,
        _CASEWISE_B,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("field", "value", "owner"),
    [
        ("k_a", _HostileIndex(), _HostileIndex),
        ("k_b", _HostileInt(1), _HostileInt),
    ],
)
def test_parameter_controls_reject_untrusted_integer_types_without_callbacks(
    field: str,
    value: object,
    owner: type,
) -> None:
    """Parameter counts fail closed before integer-protocol dispatch."""
    owner.calls = 0
    with pytest.raises(ValueError, match=rf"{field} must be a non-negative integer"):
        _compare(**{field: value})
    assert owner.calls == 0


def test_model_label_rejects_string_subclass_without_normalization_callback() -> None:
    """Audit labels require exact strings before any caller method can run."""
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="model_a must be a non-empty string"):
        _compare(model_a=_HostileStr("A"))
    assert _HostileStr.calls == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alpha", _HostileNumpyFloat(0.05), "alpha must be finite and in \\(0, 1\\)"),
        (
            "omega_tol",
            _HostileNumpyFloat(1e-12),
            "omega_tol must be finite and non-negative",
        ),
    ],
)
def test_real_controls_reject_numpy_subclasses_without_float_callback(
    field: str,
    value: object,
    message: str,
) -> None:
    """Real-valued controls reject caller NumPy subclasses before coercion."""
    _HostileNumpyFloat.calls = 0
    with pytest.raises(ValueError, match=message):
        _compare(**{field: value})
    assert _HostileNumpyFloat.calls == 0


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("alpha", "alpha must be finite and in \\(0, 1\\)"),
        ("omega_tol", "omega_tol must be finite and non-negative"),
    ],
)
def test_real_controls_normalize_builtin_integer_overflow_to_value_error(
    field: str,
    message: str,
) -> None:
    """Huge trusted integers retain the public field-specific error contract."""
    with pytest.raises(ValueError, match=message):
        _compare(**{field: 10**10000})


def test_genuine_numpy_scalars_remain_supported() -> None:
    """Trusted NumPy scalar compatibility remains part of the public contract."""
    result = _compare(
        k_a=np.int64(1),
        k_b=np.uint32(1),
        alpha=np.float64(0.05),
        omega_tol=np.float32(1e-6),
    )
    assert result.status is ComparisonStatus.UNKNOWN_RELATION
    assert result.k_a == 1
    assert result.k_b == 1
