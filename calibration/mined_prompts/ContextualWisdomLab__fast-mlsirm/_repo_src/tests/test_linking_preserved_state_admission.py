"""Regressions for preserved fixed-link source state at the trust boundary."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.linking import link_fixed_item_parameters
from fast_mlsirm.types import MLSIRMParams


class _ArrayBomb:
    calls = 0

    def __array__(self, *_args, **_kwargs):
        type(self).calls += 1
        raise AssertionError("caller array callback executed")


class _FloatBomb:
    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("caller float callback executed")


def _params(seed: int) -> MLSIRMParams:
    rng = np.random.default_rng(seed)
    return MLSIRMParams(
        theta=rng.standard_normal((8, 1)),
        alpha=np.zeros(4),
        b=np.linspace(-0.5, 0.5, 4),
        xi=rng.standard_normal((8, 2)),
        zeta=rng.standard_normal((4, 2)),
        tau=0.0,
    )


def _rust_must_not_run(*_args, **_kwargs):
    raise AssertionError("Rust linking dispatch reached untrusted preserved state")


@pytest.mark.parametrize("field", ["xi", "zeta"])
def test_fixed_link_rejects_callback_bearing_preserved_array_before_rust(
    monkeypatch, field: str
) -> None:
    source = _params(1)
    target = _params(2)
    setattr(source, field, _ArrayBomb())
    _ArrayBomb.calls = 0

    import fast_mlsirm._core as core

    monkeypatch.setattr(core, "link_fixed_item_parameters", _rust_must_not_run)

    with pytest.raises(ValueError, match=rf"source\.{field} must be a numeric array"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))

    assert _ArrayBomb.calls == 0


def test_fixed_link_rejects_callback_bearing_tau_before_rust(monkeypatch) -> None:
    source = _params(3)
    target = _params(4)
    source.tau = _FloatBomb()
    _FloatBomb.calls = 0

    import fast_mlsirm._core as core

    monkeypatch.setattr(core, "link_fixed_item_parameters", _rust_must_not_run)

    with pytest.raises(ValueError, match="source.tau must be a real number"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))

    assert _FloatBomb.calls == 0
