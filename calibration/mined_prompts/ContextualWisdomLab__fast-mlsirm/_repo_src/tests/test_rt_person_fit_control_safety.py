"""Trust-boundary regressions for response-time person-fit controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.rt import rt_person_fit


class _HostileFloat(float):
    """Caller-defined real scalar whose coercion callback must remain dormant."""

    calls = 0

    def __float__(self) -> float:
        """Record forbidden conversion and return an otherwise plausible value."""
        type(self).calls += 1
        return float.__float__(self)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the smallest ordinary response-time person-fit evidence."""
    return (
        np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64),
        np.array([1.1, 1.3], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    )


def _bomb_core() -> object:
    """Fail if an invalid semantic control reaches native discovery."""
    raise AssertionError("native core discovered before person-fit control rejection")


@pytest.mark.parametrize(
    ("control", "value", "message"),
    [
        ("alpha_level", _HostileFloat(0.05), "alpha_level must be in \\(0, 1\\)"),
        ("z_fast", _HostileFloat(1.645), "z_fast must be finite and non-negative"),
    ],
)
def test_rt_person_fit_rejects_float_subclasses_before_callbacks_and_native(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    value: object,
    message: str,
) -> None:
    """Caller real subclasses cannot participate in semantic-control admission."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    _HostileFloat.calls = 0
    times, alpha, beta = _inputs()

    with pytest.raises(ValueError, match=message):
        rt_person_fit(times, alpha, beta, **{control: value})  # type: ignore[arg-type]

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_rt_person_fit_rejects_invalid_alpha_level_before_native(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    """Aggregate flag probability must satisfy the Rust open-unit interval."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    times, alpha, beta = _inputs()

    with pytest.raises(ValueError, match="alpha_level must be in \\(0, 1\\)"):
        rt_person_fit(times, alpha, beta, alpha_level=value)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_rt_person_fit_rejects_invalid_z_fast_before_native(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    """Per-item fast-response threshold must remain finite and non-negative."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    times, alpha, beta = _inputs()

    with pytest.raises(ValueError, match="z_fast must be finite and non-negative"):
        rt_person_fit(times, alpha, beta, z_fast=value)


def test_rt_person_fit_normalizes_supported_numpy_controls_before_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete NumPy real controls remain supported and reach Rust as floats."""

    class _Core:
        def rt_person_fit(self, *args):
            assert type(args[-2]) is float
            assert type(args[-1]) is float
            assert args[-2] == pytest.approx(0.05)
            assert args[-1] == pytest.approx(0.0)
            raise RuntimeError("reached Rust boundary")

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    times, alpha, beta = _inputs()

    with pytest.raises(RuntimeError, match="reached Rust boundary"):
        rt_person_fit(
            times,
            alpha,
            beta,
            alpha_level=np.float32(0.05),
            z_fast=np.float32(0.0),
        )
