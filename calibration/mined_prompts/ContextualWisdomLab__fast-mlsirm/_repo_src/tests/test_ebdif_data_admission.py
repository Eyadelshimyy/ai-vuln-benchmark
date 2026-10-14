"""Regression coverage for Empirical Bayes DIF evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.ebdif import eb_mh_dif


class _HostileArrayProvider:
    """Array-protocol provider that must never execute during admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, dtype=None, copy=None):  # noqa: ANN001, ANN201, ARG002
        """Fail if package validation invokes the caller-owned protocol."""
        self.calls += 1
        raise AssertionError("caller __array__ must not execute")


@pytest.mark.parametrize("position", ["mh", "se"])
def test_eb_mh_dif_rejects_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
    position: str,
) -> None:
    """Reject arbitrary array providers before protocol or native discovery."""
    provider = _HostileArrayProvider()

    def _unexpected_core():
        raise AssertionError("compiled core must not be discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    mh = provider if position == "mh" else [0.1, -0.2]
    se = provider if position == "se" else [0.3, 0.4]

    with pytest.raises(ValueError, match=rf"{position} must be a numeric 1-D array"):
        eb_mh_dif(mh, se)

    assert provider.calls == 0


def test_eb_mh_dif_preserves_plain_sequence_numpy_scalar_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted plain sequences retain their existing float64 Rust payload."""
    captured: dict[str, np.ndarray] = {}

    class _Core:
        """Minimal Rust-boundary stand-in for marshalling assertions."""

        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> dict[str, object]:
            captured["mh"] = mh.copy()
            captured["se"] = se.copy()
            return {
                "mu": 0.0,
                "tau2": 0.0,
                "tau2_raw": 0.0,
                "weight": np.zeros(2),
                "post_mean": np.zeros(2),
                "post_var": np.zeros(2),
                "cat_probs": np.array([[0.0, 0.0, 1.0, 0.0, 0.0]] * 2),
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = eb_mh_dif(
        [np.float32(0.1), np.int16(-1)],
        (np.float32(0.3), np.float64(0.4)),
    )

    assert captured["mh"].dtype == np.float64
    assert captured["se"].dtype == np.float64
    assert captured["mh"].flags.c_contiguous
    assert captured["se"].flags.c_contiguous
    assert np.allclose(captured["mh"], [0.1, -1.0])
    assert np.allclose(captured["se"], [0.3, 0.4])
    assert result.cat_probs.shape == (2, 5)
