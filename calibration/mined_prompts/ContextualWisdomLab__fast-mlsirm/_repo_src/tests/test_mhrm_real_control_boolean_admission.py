"""Regressions for Boolean exclusion from MH-RM real-valued controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mhrm


class _ResponseShouldNotBeTouched:
    """Fail if invalid semantic controls reach caller response work."""

    callbacks = 0

    def __array__(self, dtype=None):
        del dtype
        type(self).callbacks += 1
        raise AssertionError("response materialized before MH-RM real controls were admitted")


def _unexpected_core() -> object:
    """Fail if native discovery precedes semantic-control admission."""

    raise AssertionError("compiled core discovered before MH-RM real controls were admitted")


@pytest.mark.parametrize("control_name", ("proposal_sd", "target_accept", "tol"))
@pytest.mark.parametrize("boolean_value", (True, np.bool_(True)))
def test_boolean_real_controls_fail_before_response_or_native_work(
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
    boolean_value: object,
) -> None:
    """Boolean identity cannot stand in for a continuous MH-RM tuning control."""

    _ResponseShouldNotBeTouched.callbacks = 0
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    controls = {
        "model": 1,
        "max_cycles": 2,
        "burn_in": 1,
        "mh_steps": 1,
        "target_accept": 0.3,
        "tol": 1e-3,
        "seed": 0x9E37_79B9_7F4A_7C15,
        control_name: boolean_value,
    }
    with pytest.raises(ValueError, match=rf"{control_name} must be a finite real scalar"):
        mhrm.fit_mhrm(_ResponseShouldNotBeTouched(), **controls)

    assert _ResponseShouldNotBeTouched.callbacks == 0
