"""Trust-boundary regressions for MH-RM semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mhrm


class _ResponseShouldNotBeTouched(list):
    """A response placeholder whose subclass identity must not win over bad controls."""


class _HostileArrayControl:
    """Expose a caller array protocol that package control admission must never execute."""

    callbacks = 0

    def __array__(self, dtype=None):
        del dtype
        type(self).callbacks += 1
        raise AssertionError("caller __array__ executed during MH-RM control admission")


class _HostileFloat(float):
    """Fail if package validation invokes caller-defined floating conversion."""

    callbacks = 0

    def __float__(self):
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during MH-RM control admission")


class _HostileInt(int):
    """Fail if package validation invokes caller-defined integer conversion."""

    callbacks = 0

    def __int__(self):
        type(self).callbacks += 1
        raise AssertionError("caller __int__ executed during MH-RM control admission")


class _HostileBool:
    """Fail if package validation invokes caller-defined truth conversion."""

    callbacks = 0

    def __bool__(self):
        type(self).callbacks += 1
        raise AssertionError("caller __bool__ executed during MH-RM control admission")


class _HostileStr(str):
    """Fail if package validation invokes caller-defined text conversion."""

    callbacks = 0

    def __str__(self):
        type(self).callbacks += 1
        raise AssertionError("caller __str__ executed during MH-RM control admission")


def _unexpected_core() -> object:
    """Fail if compiled-core discovery precedes semantic-control admission."""

    raise AssertionError("compiled core discovered before MH-RM controls were admitted")


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core 2PL MH-RM result."""

    return {
        "loading": [1.0, 1.0],
        "intercept": [0.0, 0.0],
        "step": [],
        "n_cat": 2,
        "theta": [0.0, 0.0],
        "corr": [1.0],
        "se_loading": [],
        "se_intercept": [],
        "se_step": [],
        "acceptance_rate": 0.25,
        "n_cycles": 2,
        "converged": False,
        "termination_reason": "max_cycles_reached",
        "final_param_change": 0.1,
        "n_parameters": 4,
    }


def test_invalid_control_precedes_response_admission(monkeypatch):
    """An invalid response container is irrelevant when an independent control is invalid first."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="max_cycles must be a finite integer"):
        mhrm.fit_mhrm(_ResponseShouldNotBeTouched(), max_cycles=object(), model=1, burn_in=200, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)


@pytest.mark.parametrize(
    ("controls", "message"),
    [
        ({"max_cycles": -1}, "require 0 <= burn_in < max_cycles"),
        ({"max_cycles": 2, "burn_in": -1}, "require 0 <= burn_in < max_cycles"),
        ({"max_cycles": 2, "burn_in": 1, "mh_steps": -1}, "mh_steps must be positive"),
    ],
)
def test_native_unsigned_control_domains_fail_before_response_admission(monkeypatch, controls, message):
    """Values that cannot enter Rust usize controls fail before caller response work."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match=message):
        mhrm.fit_mhrm(_ResponseShouldNotBeTouched(), **{"model": 1, "max_cycles": 2000, "burn_in": 200, "mh_steps": 5, "target_accept": 0.3, "tol": 1e-3, "seed": 0x9E3779B97F4A7C15, **controls})


@pytest.mark.parametrize(
    "controls",
    [
        {"max_cycles": 2**64},
        {"max_cycles": 2, "burn_in": 2**64},
        {"max_cycles": 2, "burn_in": 1, "mh_steps": 2**64},
    ],
)
def test_native_usize_upper_bound_fails_before_response_admission(monkeypatch, controls):
    """Controls outside the supported 64-bit usize boundary fail before caller response work."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match=r"must be in \[0, 2\*\*64\)"):
        mhrm.fit_mhrm(_ResponseShouldNotBeTouched(), **{"model": 1, "max_cycles": 2000, "burn_in": 200, "mh_steps": 5, "target_accept": 0.3, "tol": 1e-3, "seed": 0x9E3779B97F4A7C15, **controls})


def test_callback_bearing_numeric_and_text_controls_fail_without_callbacks(monkeypatch):
    """Caller conversion protocols cannot define MH-RM estimator controls."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)

    _HostileArrayControl.callbacks = 0
    with pytest.raises(ValueError, match="max_cycles must be a finite integer"):
        mhrm.fit_mhrm(responses, max_cycles=_HostileArrayControl(), model=1, burn_in=200, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)
    assert _HostileArrayControl.callbacks == 0

    _HostileFloat.callbacks = 0
    with pytest.raises(ValueError, match="proposal_sd must be a finite real scalar"):
        mhrm.fit_mhrm(responses, max_cycles=2, burn_in=1, proposal_sd=_HostileFloat(1.0), model=1, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)
    assert _HostileFloat.callbacks == 0

    _HostileInt.callbacks = 0
    with pytest.raises(TypeError, match="seed must be a non-negative integer"):
        mhrm.fit_mhrm(responses, max_cycles=2, burn_in=1, seed=_HostileInt(7), model=1, mh_steps=5, target_accept=0.3, tol=1e-3)
    assert _HostileInt.callbacks == 0

    _HostileStr.callbacks = 0
    with pytest.raises(ValueError, match="family must be '2pl' or 'gpcm'"):
        mhrm.fit_mhrm(responses, family=_HostileStr("2pl"), max_cycles=2, burn_in=1, model=1, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)
    assert _HostileStr.callbacks == 0


def test_boolean_controls_fail_before_native_discovery_without_truth_callbacks(monkeypatch):
    """Boolean estimator flags are admitted before capability discovery and without __bool__."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)

    _HostileBool.callbacks = 0
    with pytest.raises(TypeError, match="estimate_se must be a boolean"):
        mhrm.fit_mhrm(
            responses,
            max_cycles=2,
            burn_in=1,
            estimate_se=_HostileBool(), model=1, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
        )
    assert _HostileBool.callbacks == 0

    with pytest.raises(TypeError, match="estimate_corr must be a boolean"):
        mhrm.fit_mhrm(
            responses,
            max_cycles=2,
            burn_in=1,
            estimate_corr=_HostileBool(), model=1, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
        )
    assert _HostileBool.callbacks == 0


def test_trusted_numpy_controls_are_normalized_to_builtin_primitives(monkeypatch):
    """Concrete NumPy scalars keep compatibility while Rust receives inert built-in controls."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_mhrm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)

    mhrm.fit_mhrm(
        responses,
        family="2pl",
        n_cat=np.int16(2),
        max_cycles=np.float32(2.0),
        burn_in=np.int16(1),
        mh_steps=np.uint8(1),
        proposal_sd=np.float32(1.0),
        target_accept=np.float64(0.30),
        tol=np.float32(1e-3),
        seed=np.uint64(7),
        estimate_se=np.bool_(False),
        estimate_corr=np.bool_(False), model=1
    )

    args = captured["args"]
    for index in (6, 7, 8, 12, 16):
        assert type(args[index]) is int
    for index in (9, 10, 11):
        assert type(args[index]) is float
    assert type(args[13]) is bool
    assert type(args[14]) is bool
    assert type(args[15]) is str


def test_full_uint64_iteration_control_domain_remains_compatible(monkeypatch):
    """The 64-bit usize boundary remains unsigned rather than being narrowed to signed 63 bits."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_mhrm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)
    maximum = np.uint64(2**64 - 1)

    mhrm.fit_mhrm(
        responses,
        max_cycles=2,
        burn_in=1,
        mh_steps=maximum,
        estimate_se=False, model=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
    )

    assert type(captured["args"][8]) is int
    assert captured["args"][8] == 2**64 - 1
