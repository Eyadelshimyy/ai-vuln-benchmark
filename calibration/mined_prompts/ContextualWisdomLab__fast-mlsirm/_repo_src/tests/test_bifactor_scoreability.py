"""Public Python contracts for Rust-native bifactor scoreability diagnostics."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm._bifactor_core_loader import bifactor_core
from fast_mlsirm.bifactor_scoreability import (
    BifactorScoreabilityResult,
    assess_bifactor_scoreability,
    assess_bifactor_scoreability_from_logit_slopes,
    bifactor_scoreability,
    bifactor_scoreability_from_logit_slopes,
)


def _loadings() -> np.ndarray:
    """Return one standardized strict-bifactor loading matrix."""
    return np.asarray(
        [
            [0.70, 0.40, 0.00],
            [0.70, 0.30, 0.00],
            [0.70, 0.00, 0.50],
            [0.70, 0.00, 0.60],
        ],
        dtype=np.float64,
    )


def _uniquenesses() -> np.ndarray:
    """Return residual variances satisfying the standardized identity."""
    return np.asarray([0.35, 0.42, 0.26, 0.15], dtype=np.float64)


def _logit_slopes() -> np.ndarray:
    """Invert the documented logistic latent-response standardization."""
    loadings = _loadings()
    uniquenesses = _uniquenesses()
    logistic_sd = np.pi / np.sqrt(3.0)
    return loadings * (logistic_sd / np.sqrt(uniquenesses))[:, None]


def _assert_result_matches_raw(result: BifactorScoreabilityResult, raw: dict) -> None:
    """Assert exact field-by-field wrapper parity with the compiled module."""
    assert result.factor_item_counts == tuple(raw["factor_item_counts"])
    assert result.is_strict_bifactor is raw["is_strict_bifactor"]
    assert result.puc == pytest.approx(raw["puc"])
    np.testing.assert_allclose(result.ecv_ss, raw["ecv_ss"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.ecv_sg, raw["ecv_sg"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.ecv_gs, raw["ecv_gs"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.item_ecv, raw["item_ecv"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.omega_total, raw["omega_total"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.omega_hierarchical,
        raw["omega_hierarchical"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.construct_replicability,
        raw["construct_replicability"],
        rtol=0.0,
        atol=0.0,
    )


def test_standardized_loading_wrapper_has_exact_rust_parity():
    """The Python layer only marshals the compiled standardized-loading result."""
    loadings = _loadings()
    uniquenesses = _uniquenesses()
    result = assess_bifactor_scoreability(
        loadings, uniquenesses, general_factor=0, zero_tolerance=0.0
    )
    raw = bifactor_core().bifactor_indices(
        loadings,
        uniquenesses,
        0,
        0.0,
    )

    assert isinstance(result, BifactorScoreabilityResult)
    assert result.puc == pytest.approx(2.0 / 3.0)
    _assert_result_matches_raw(result, raw)


def test_logit_slope_wrapper_has_exact_rust_parity():
    """Fitted logistic slopes are standardized and scored only in Rust."""
    slopes = _logit_slopes()
    result = assess_bifactor_scoreability_from_logit_slopes(
        slopes, general_factor=0, zero_tolerance=0.0
    )
    raw = bifactor_core().bifactor_indices_from_logit_slopes(slopes, 0, 0.0)

    _assert_result_matches_raw(result, raw)


def test_zero_tolerance_is_a_required_argument():
    """ADR-0028 (#1963): zero_tolerance has no default and must be supplied."""
    with pytest.raises(TypeError):
        assess_bifactor_scoreability(_loadings(), _uniquenesses(), general_factor=0)
    with pytest.raises(TypeError):
        assess_bifactor_scoreability_from_logit_slopes(
            _logit_slopes(), general_factor=0
        )


def test_deprecated_bifactor_scoreability_alias_still_works_and_warns():
    """ADR-0028 (#1963): the pre-rename name is kept as a deprecated alias."""
    loadings = _loadings()
    uniquenesses = _uniquenesses()
    with pytest.warns(DeprecationWarning):
        old_result = bifactor_scoreability(
            loadings, uniquenesses, general_factor=0, zero_tolerance=0.0
        )
    new_result = assess_bifactor_scoreability(
        loadings, uniquenesses, general_factor=0, zero_tolerance=0.0
    )
    _assert_result_matches_raw(
        old_result,
        bifactor_core().bifactor_indices(loadings, uniquenesses, 0, 0.0),
    )
    np.testing.assert_allclose(old_result.omega_total, new_result.omega_total)


def test_deprecated_bifactor_scoreability_from_logit_slopes_alias_still_works_and_warns():
    """ADR-0028 (#1963): the pre-rename name is kept as a deprecated alias."""
    slopes = _logit_slopes()
    with pytest.warns(DeprecationWarning):
        old_result = bifactor_scoreability_from_logit_slopes(
            slopes, general_factor=0, zero_tolerance=0.0
        )
    new_result = assess_bifactor_scoreability_from_logit_slopes(
        slopes, general_factor=0, zero_tolerance=0.0
    )
    np.testing.assert_allclose(old_result.omega_total, new_result.omega_total)


def test_package_root_exports_the_public_bifactor_api():
    """Buyers can discover the typed diagnostics from the package root."""
    assert fast_mlsirm.BifactorScoreabilityResult is BifactorScoreabilityResult
    assert fast_mlsirm.bifactor_scoreability is bifactor_scoreability
    assert (
        fast_mlsirm.bifactor_scoreability_from_logit_slopes
        is bifactor_scoreability_from_logit_slopes
    )


def test_every_declared_package_export_exists():
    """The modular root split cannot leave stale names in ``__all__``."""
    missing = [name for name in fast_mlsirm.__all__ if not hasattr(fast_mlsirm, name)]
    assert missing == []


def test_result_vectors_are_immutable():
    """Typed diagnostics cannot be silently altered after Rust validation."""
    result = assess_bifactor_scoreability(
        _loadings(), _uniquenesses(), zero_tolerance=0.0
    )
    for vector in (
        result.ecv_ss,
        result.ecv_sg,
        result.ecv_gs,
        result.item_ecv,
        result.omega_total,
        result.omega_hierarchical,
        result.construct_replicability,
    ):
        assert vector.flags.writeable is False


@pytest.mark.parametrize(
    ("loadings", "uniquenesses", "message"),
    [
        (np.ones(4), _uniquenesses(), "loadings must be a 2-D"),
        (_loadings(), np.ones((2, 2)), "uniquenesses must be a 1-D"),
        (_loadings(), np.ones(3), "uniquenesses length"),
    ],
)
def test_public_wrapper_validates_array_shapes(loadings, uniquenesses, message):
    """Shape mismatches fail before crossing the compiled boundary."""
    with pytest.raises(ValueError, match=message):
        assess_bifactor_scoreability(loadings, uniquenesses, zero_tolerance=0.0)


def test_missing_general_factor_fails_through_rust():
    """The Python wrapper preserves the Rust fail-closed bifactor definition."""
    loadings = _loadings()
    loadings[0, 0] = 0.0
    uniquenesses = 1.0 - np.square(loadings).sum(axis=1)
    with pytest.raises(ValueError, match="general factor"):
        assess_bifactor_scoreability(
            loadings, uniquenesses, general_factor=0, zero_tolerance=0.0
        )


def test_nonfirst_general_factor_and_single_item_domain_reach_rust():
    """Python preserves labelled factor columns and strict-pattern PUC."""
    loadings = np.asarray(
        [
            [0.40, 0.70, 0.00],
            [0.30, 0.70, 0.00],
            [0.00, 0.70, 0.50],
        ],
        dtype=np.float64,
    )
    uniquenesses = 1.0 - np.square(loadings).sum(axis=1)
    result = assess_bifactor_scoreability(
        loadings, uniquenesses, general_factor=1, zero_tolerance=0.0
    )

    assert result.factor_item_counts == (2, 3, 1)
    assert result.is_strict_bifactor is True
    assert result.puc == pytest.approx(2.0 / 3.0)


def test_cross_loaded_specific_pattern_is_descriptive_but_has_no_puc():
    """Specific cross-loadings remain valid while strict-bifactor PUC fails closed."""
    loadings = _loadings()
    loadings[0, 2] = 0.10
    uniquenesses = 1.0 - np.square(loadings).sum(axis=1)
    result = assess_bifactor_scoreability(loadings, uniquenesses, zero_tolerance=0.0)

    assert result.is_strict_bifactor is False
    assert result.puc is None


def test_standardized_identity_accepts_roundoff_but_rejects_material_error():
    """The public path preserves the documented Rust identity tolerance."""
    loadings = np.asarray([[0.70, 0.20], [0.80, 0.30]], dtype=np.float64)
    roundoff = np.asarray([0.47 + 5e-9, 0.27 - 5e-9], dtype=np.float64)
    assert assess_bifactor_scoreability(
        loadings, roundoff, zero_tolerance=0.0
    ).factor_item_counts == (2, 2)

    material = roundoff.copy()
    material[0] += 1e-5
    with pytest.raises(ValueError, match="sum to one"):
        assess_bifactor_scoreability(loadings, material, zero_tolerance=0.0)


def test_sign_cancelled_composite_variance_is_rejected_by_rust():
    """Perfect loading cancellation cannot produce a misleading zero omega."""
    root_half = np.sqrt(0.5)
    loadings = np.asarray(
        [[root_half, root_half], [-root_half, -root_half]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="omega denominator must be positive"):
        assess_bifactor_scoreability(
            loadings, np.zeros(2, dtype=np.float64), zero_tolerance=0.0
        )


def test_python_rejects_oversized_work_before_loading_the_rust_module(monkeypatch):
    """The public boundary rejects expensive shapes before extension dispatch."""
    class UnexpectedCore:
        """Fail if the compiled extension is consulted after a shape rejection."""

        def bifactor_indices(self, *_args, **_kwargs):
            """Signal that Python failed to enforce the pre-dispatch budget."""
            raise AssertionError("compiled bifactor function must not be called")

    module = importlib.import_module("fast_mlsirm.bifactor_scoreability")
    monkeypatch.setattr(module, "bifactor_core", lambda: UnexpectedCore())
    loadings = np.zeros((12_208, 64), dtype=np.float64)
    uniquenesses = np.ones(12_208, dtype=np.float64)
    with pytest.raises(ValueError, match="work budget"):
        assess_bifactor_scoreability(loadings, uniquenesses, zero_tolerance=0.0)


def test_loader_caches_the_secondary_extension_module():
    """Repeated calls reuse one initialized `_bifactor_core` module."""
    assert bifactor_core() is bifactor_core()


def test_plain_nested_sequences_keep_the_public_array_like_contract():
    """Ordinary nested Python lists remain accepted after bounded preflight."""
    result = assess_bifactor_scoreability(
        _loadings().tolist(), _uniquenesses().tolist(), zero_tolerance=0.0
    )
    assert result.factor_item_counts == (4, 2, 2)


def test_numpy_row_sequences_keep_the_public_array_like_contract():
    """Lists of one-dimensional NumPy rows remain valid two-dimensional inputs."""
    loadings = [row.copy() for row in _loadings()]
    result = assess_bifactor_scoreability(loadings, _uniquenesses(), zero_tolerance=0.0)
    assert result.factor_item_counts == (4, 2, 2)


@pytest.mark.parametrize("loadings", [[], [0.7, 0.4], [[[0.7, 0.4]]]])
def test_plain_sequences_reject_non_matrix_shapes_before_core(loadings):
    """Empty, one-dimensional, and three-dimensional lists fail closed."""
    with pytest.raises(ValueError, match="2-D item-by-factor matrix"):
        assess_bifactor_scoreability(loadings, _uniquenesses(), zero_tolerance=0.0)
