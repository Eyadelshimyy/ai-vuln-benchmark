"""Trust-boundary regressions for model-comparison casewise values."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.model_comparison as model_comparison
from fast_mlsirm.model_comparison import ModelRelation, compare_nonnested_models


class _FloatProvider:
    """Hostile arbitrary float protocol provider used to detect callbacks."""

    def __init__(self, calls: list[str]) -> None:
        """Record the callback sink without invoking caller conversion."""
        self._calls = calls

    def __float__(self) -> float:
        """Record an attempted conversion and return a hostile NaN."""
        self._calls.append("__float__")
        return float("nan")


class _FloatSubclass(float):
    """Hostile float subclass whose conversion hook must never execute."""

    calls: list[str] = []

    def __float__(self) -> float:
        """Record an attempted subclass conversion and return a hostile NaN."""
        type(self).calls.append("__float__")
        return float("nan")


@pytest.mark.parametrize("side", ["loglik_a", "loglik_b"])
@pytest.mark.parametrize("kind", ["provider", "subclass"])
def test_casewise_values_reject_untrusted_float_callbacks_without_execution(
    side: str,
    kind: str,
) -> None:
    """Rejected casewise scalars must fail before caller conversion callbacks."""
    calls: list[str] = []
    if kind == "provider":
        hostile: object = _FloatProvider(calls)
    else:
        _FloatSubclass.calls = calls
        hostile = _FloatSubclass(0.0)

    values_a: list[object] = [0.0, 0.0]
    values_b: list[object] = [0.0, 0.0]
    target = values_a if side == "loglik_a" else values_b
    target[0] = hostile

    with pytest.raises(ValueError, match=rf"{side}\[0\] must be a finite number"):
        compare_nonnested_models(
            values_a,
            values_b,
            1,
            1,
            relation=ModelRelation.STRICTLY_NON_NESTED,
        )

    assert calls == []


def test_casewise_values_preserve_genuine_numpy_scalar_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete NumPy real scalars remain admitted at the public boundary."""
    observed: dict[str, tuple[float, ...]] = {}

    def fake_vuong(
        values_a: tuple[float, ...],
        values_b: tuple[float, ...],
        k_a: int,
        k_b: int,
        *,
        bic_correction: bool,
    ) -> dict[str, float]:
        """Capture normalized NumPy scalars instead of invoking Rust."""
        observed["a"] = values_a
        observed["b"] = values_b
        assert (k_a, k_b, bic_correction) == (1, 1, False)
        return {
            "mean_diff": 0.25,
            "omega": 0.5,
            "z": 1.0,
            "p_two_sided": 0.31731050786291415,
        }

    monkeypatch.setattr(model_comparison, "vuong_nonnested", fake_vuong)

    result = compare_nonnested_models(
        [np.float32(-1.5), np.int16(-1)],
        [np.float64(-1.75), np.int32(-1)],
        np.int16(1),
        np.int32(1),
        relation=ModelRelation.STRICTLY_NON_NESTED,
        bic_correction=False,
    )

    assert observed == {"a": (-1.5, -1.0), "b": (-1.75, -1.0)}
    assert result.raw_mean_loglik_difference == pytest.approx(0.25)
