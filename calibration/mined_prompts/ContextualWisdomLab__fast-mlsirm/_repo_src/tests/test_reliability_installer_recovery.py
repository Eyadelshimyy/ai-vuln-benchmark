"""Regression tests for interrupted reliability-adapter installation recovery."""

from __future__ import annotations

from types import ModuleType

from fast_mlsirm import _icc_control_safety as icc_control_safety
from fast_mlsirm import reliability

_PRIMARY_APIS = (
    "guttman_lambdas",
    "tenberge_mu",
    "cronbach_alpha",
    "separation_reliability",
    "mean_pairwise_cor",
    "mean_pairwise_rho",
)

_RATER_APIS = (
    "kripp_alpha",
    "finn_coefficient",
    "maxwell_re",
    "robinson_a",
)


def test_icc_installer_repairs_missing_rater_adapters_after_partial_install():
    """A hardened ICC surface must not prevent repair of sibling rater adapters."""
    module = ModuleType("fast_mlsirm_test_reliability")
    module.__package__ = "fast_mlsirm_test"
    module.icc = reliability.icc
    for name in _PRIMARY_APIS:
        setattr(module, name, getattr(reliability, name))

    assert getattr(module.icc, "__fast_mlsirm_icc_control_hardened__", False)

    originals = {}
    for name in _RATER_APIS:
        wrapped = getattr(reliability, name)
        original = wrapped.__wrapped__
        originals[name] = original
        setattr(module, name, original)

    assert not getattr(
        module.kripp_alpha,
        "__fast_mlsirm_rater_evidence_hardened__",
        False,
    )

    icc_control_safety.install(module)

    assert getattr(
        module.kripp_alpha,
        "__fast_mlsirm_rater_evidence_hardened__",
        False,
    )
    for name, original in originals.items():
        assert getattr(module, name) is not original


def test_icc_installer_repairs_missing_primary_adapter_after_partial_install():
    """A hardened ICC flag must not strand a missing primary reliability wrapper."""
    module = ModuleType("fast_mlsirm_test_primary_reliability")
    module.__package__ = "fast_mlsirm_test"

    module.icc = reliability.icc
    originals = {}
    for name in _PRIMARY_APIS:
        wrapped = getattr(reliability, name)
        original = wrapped.__wrapped__
        originals[name] = original
        setattr(module, name, wrapped)
    for name in _RATER_APIS:
        setattr(module, name, getattr(reliability, name))

    # Model an interrupted install after ICC was hardened but before one sibling
    # binding became durable. A retry must repair the whole primary surface.
    module.guttman_lambdas = originals["guttman_lambdas"]
    assert getattr(module.icc, "__fast_mlsirm_icc_control_hardened__", False)
    assert module.guttman_lambdas is originals["guttman_lambdas"]

    icc_control_safety.install(module)

    assert module.guttman_lambdas is not originals["guttman_lambdas"]
    assert module.guttman_lambdas.__wrapped__ is originals["guttman_lambdas"]


def test_icc_installer_repairs_partially_hardened_rater_surface():
    """A hardened Krippendorff wrapper must not strand an unwrapped rater sibling."""
    module = ModuleType("fast_mlsirm_test_partial_rater_reliability")
    module.__package__ = "fast_mlsirm_test"
    module.icc = reliability.icc
    for name in _PRIMARY_APIS:
        setattr(module, name, getattr(reliability, name))

    originals = {}
    for name in _RATER_APIS:
        wrapped = getattr(reliability, name)
        originals[name] = wrapped.__wrapped__
        setattr(module, name, wrapped)

    # Model interruption after kripp_alpha was rebound but before finn_coefficient.
    module.finn_coefficient = originals["finn_coefficient"]
    assert getattr(
        module.kripp_alpha,
        "__fast_mlsirm_rater_evidence_hardened__",
        False,
    )
    assert module.finn_coefficient is originals["finn_coefficient"]

    icc_control_safety.install(module)

    assert module.finn_coefficient is not originals["finn_coefficient"]
    assert module.finn_coefficient.__wrapped__ is originals["finn_coefficient"]


def test_icc_installer_keeps_complete_rater_surface_idempotent():
    """A complete hardened rater surface must not gain another wrapper layer."""
    module = ModuleType("fast_mlsirm_test_complete_rater_reliability")
    module.__package__ = "fast_mlsirm_test"
    module.icc = reliability.icc
    for name in _PRIMARY_APIS:
        setattr(module, name, getattr(reliability, name))
    for name in _RATER_APIS:
        setattr(module, name, getattr(reliability, name))

    before = tuple(getattr(module, name) for name in _RATER_APIS)
    icc_control_safety.install(module)
    after_first = tuple(getattr(module, name) for name in _RATER_APIS)
    icc_control_safety.install(module)
    after_second = tuple(getattr(module, name) for name in _RATER_APIS)

    assert after_first == before
    assert after_second == before
    assert all(
        getattr(function, "__fast_mlsirm_rater_evidence_hardened__", False)
        for function in after_second
    )
