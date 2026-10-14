"""Regression evidence for the NumPy MMLE EAP matrix-vector projection."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

fit_module = importlib.import_module("fast_mlsirm.fit")
from fast_mlsirm.config import FitConfig
from fast_mlsirm.estimators import mmle as mmle_module


ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
CHANGELOG_SECTION_HEADING = "#### MMLE theta calculation memory optimization"
DOCTORING = ROOT / "docs" / "doctoring" / "mmle-eap-matvec-projection.md"


def _changelog_section(heading: str) -> str:
    """Return one released or unreleased CHANGELOG.md subsection's exact text."""
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    start = text.index(heading)
    next_heading = text.index("\n#### ", start + len(heading))
    return text[start:next_heading]


def _initial_posterior(
    responses: np.ndarray,
    observed: np.ndarray,
    *,
    n_nodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the first MMLE E-step without using production posterior helpers."""
    y_filled = np.where(observed, responses, 0.0)
    obs_f = observed.astype(np.float64)
    nodes, raw_weights = hermegauss(n_nodes)
    weights = raw_weights / raw_weights.sum()

    rng = np.random.default_rng(seed)
    p_item = (y_filled * obs_f).sum(0) / np.clip(obs_f.sum(0), 1.0, None)
    p_item = np.clip(p_item, 0.02, 0.98)
    discrimination = np.ones(responses.shape[1]) + 0.01 * rng.standard_normal(
        responses.shape[1]
    )
    intercept = np.log(p_item / (1.0 - p_item))

    logit = nodes[:, None] * discrimination[None, :] + intercept[None, :]
    log_p1 = -np.logaddexp(0.0, -logit)
    log_p0 = -np.logaddexp(0.0, logit)
    positive = (y_filled * obs_f) @ log_p1.T
    negative = ((1.0 - y_filled) * obs_f) @ log_p0.T
    log_joint = positive + negative + np.log(weights)[None, :]
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    stabilized = np.exp(log_joint - max_log_joint)
    posterior = stabilized / stabilized.sum(axis=1, keepdims=True)
    return posterior, nodes


def test_eap_matvec_matches_previous_weighted_sum_under_partial_missingness() -> None:
    """The optimized EAP projection must preserve the previous weighted sum."""
    responses = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [True, True, False, True],
            [True, False, True, True],
            [True, True, True, False],
            [False, True, True, True],
        ],
        dtype=bool,
    )
    n_nodes = 9
    seed = 17

    result = mmle_module.fit_mmle_2pl(
        responses,
        observed,
        n_nodes=n_nodes,
        max_iter=1,
        tol=0.0,
        seed=seed,
    )
    posterior, nodes = _initial_posterior(
        responses,
        observed,
        n_nodes=n_nodes,
        seed=seed,
    )
    previous_projection = (posterior * nodes[None, :]).sum(axis=1)

    np.testing.assert_allclose(
        result["theta"],
        previous_projection,
        rtol=1e-13,
        atol=1e-13,
    )


def test_eap_projection_source_uses_matrix_multiplication_without_broadcast_product() -> None:
    """Pin the allocation-bounded EAP source shape so the N-by-Q product stays absent."""
    source = textwrap.dedent(inspect.getsource(mmle_module.fit_mmle_2pl))
    tree = ast.parse(source)
    theta_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "theta" for target in node.targets)
    ]

    assert len(theta_assignments) == 1
    theta_expression = theta_assignments[0].value
    assert isinstance(theta_expression, ast.BinOp)
    assert isinstance(theta_expression.op, ast.MatMult)
    assert isinstance(theta_expression.left, ast.Name)
    assert theta_expression.left.id == "posterior"
    assert isinstance(theta_expression.right, ast.Name)
    assert theta_expression.right.id == "nodes"


@pytest.mark.parametrize("n_nodes", [0, 101, True, 3.5])
def test_quadrature_node_count_fails_closed_outside_supported_range(n_nodes: object) -> None:
    """Reject invalid or untested quadrature counts before constructing nodes."""
    with pytest.raises(ValueError, match=r"n_nodes must be"):
        mmle_module.gauss_hermite_nodes(n_nodes)  # type: ignore[arg-type]


def test_mmle_rejects_oversized_virtual_problem_before_owned_allocations() -> None:
    """A huge zero-stride input view must fail before the fallback materializes grids."""
    responses = np.broadcast_to(np.array(0.0), (100_000, 100_000))
    observed = np.broadcast_to(np.array(True), responses.shape)

    with pytest.raises(
        ValueError,
        match=r"workspace estimate .* exceeds .* safe limit; use the Rust backend",
    ):
        mmle_module.fit_mmle_2pl(
            responses,
            observed,
            n_nodes=41,
            max_iter=1, tol=1e-6, seed=1
        )


def test_workspace_cap_precedes_dtype_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback must reject its budget before allocating dtype-conversion copies."""
    monkeypatch.setattr(mmle_module, "MAX_MMLE_FALLBACK_WORKSPACE_BYTES", 1)
    original_asarray = np.asarray

    def reject_typed_asarray(
        value: object,
        *args: object,
        dtype: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        if dtype is not None:
            pytest.fail("workspace validation must run before dtype conversion")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(mmle_module.np, "asarray", reject_typed_asarray)

    responses = np.array([[1.0, 0.0]], dtype=np.float32)
    observed = np.array([[1, 1]], dtype=np.int8)
    with pytest.raises(ValueError, match=r"workspace estimate .* exceeds"):
        mmle_module.fit_mmle_2pl(
            responses,
            observed,
            n_nodes=9,
            max_iter=1, tol=1e-6, seed=1
        )


def test_workspace_cap_precedes_response_grid_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback must reject its budget before calling ``numpy.where``."""
    monkeypatch.setattr(mmle_module, "MAX_MMLE_FALLBACK_WORKSPACE_BYTES", 1)

    def unexpected_where(*_args: object, **_kwargs: object) -> np.ndarray:
        pytest.fail("workspace validation must run before numpy.where")

    monkeypatch.setattr(mmle_module.np, "where", unexpected_where)

    responses = np.array([[1.0, 0.0]], dtype=np.float64)
    observed = np.array([[True, True]], dtype=bool)
    with pytest.raises(ValueError, match=r"workspace estimate .* exceeds"):
        mmle_module.fit_mmle_2pl(
            responses,
            observed,
            n_nodes=9,
            max_iter=1, tol=1e-6, seed=1
        )


def test_public_mmle_fails_closed_without_allocating_fallback_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the Rust MMLE kernel, the public route fails closed before NumPy fallback."""
    responses = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    try:
        from fast_mlsirm import _core
    except ImportError:
        pass
    else:
        monkeypatch.setattr(_core, "fit_mmle_2pl", None)

    from fast_mlsirm.config import FitConfig
    from fast_mlsirm.fit import _fit_mmle

    observed = np.ones_like(responses, dtype=bool)
    with pytest.raises(RuntimeError, match="compiled Rust core is required for MMLE"):
        _fit_mmle(
            responses,
            observed,
            model="ULS2PLM",
            config=FitConfig(estimator="mmle", model="ULS2PLM", max_iter=1),
            backend="rust",
        )


def test_changelog_avoids_universal_eap_speedup_claims() -> None:
    """Release notes must describe the optimization without a fixed speedup promise."""
    fragment = _changelog_section(CHANGELOG_SECTION_HEADING)
    lowered = fragment.casefold()

    assert "30x" not in lowered
    assert "runtime" in lowered
    assert "hardware" in lowered
    assert "blas" in lowered


def test_eap_projection_doctoring_records_semantics_and_architecture_boundary() -> None:
    """Doctoring must record semantics, resource caps, and the Rust boundary."""
    doctoring = DOCTORING.read_text(encoding="utf-8")
    lowered = doctoring.casefold()

    assert "numpy.matmul" in lowered
    assert "posterior @ nodes" in doctoring
    assert "rust" in lowered
    assert "reference/fallback" in lowered
    assert "https://numpy.org/doc/stable/reference/generated/numpy.matmul.html" in doctoring
    assert "runtime" in lowered
    assert "hardware" in lowered
    assert "512 mib" in lowered
    assert "1 through 100" in lowered
    assert (
        "https://numpy.org/doc/stable/reference/generated/"
        "numpy.polynomial.hermite_e.hermegauss.html"
    ) in doctoring
