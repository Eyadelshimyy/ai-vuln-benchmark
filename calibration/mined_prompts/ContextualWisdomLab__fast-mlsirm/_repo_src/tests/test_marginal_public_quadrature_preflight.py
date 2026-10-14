"""Public-entry regressions for bounded marginal quadrature preflight."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


class _HostileInteger:
    """Caller-controlled object whose integer conversion leaks hostile text."""

    def __int__(self) -> int:
        """Raise if production code attempts caller-controlled conversion."""
        raise RuntimeError("caller-secret-q-xi")


@pytest.mark.parametrize(
    "q_xi",
    (True, 7.5, np.int64(7), _HostileInteger()),
)
def test_spatial_q_xi_fails_before_array_coercion(
    monkeypatch: pytest.MonkeyPatch,
    q_xi: object,
) -> None:
    """Malformed tensor quadrature cannot reach arrays or caller callbacks."""

    def forbidden_asarray(*_args: object, **_kwargs: object) -> np.ndarray:
        pytest.fail("q_xi validation must precede NumPy array coercion")

    monkeypatch.setattr(marginal.np, "asarray", forbidden_asarray)

    with pytest.raises(
        ValueError,
        match="q_xi must be an exact built-in integer",
    ) as caught:
        marginal.fit_marginal_numpy(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            model="MLSRM",
            latent_dim=1,
            q_xi=q_xi,  # type: ignore[arg-type]
            q_theta=21, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0,
        )

    assert "caller-secret-q-xi" not in str(caught.value)


def test_unsupported_exact_q_xi_fails_before_array_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An astronomical exact integer is rejected without constructing its power."""

    def forbidden_asarray(*_args: object, **_kwargs: object) -> np.ndarray:
        pytest.fail("quadrature validation must precede NumPy array coercion")

    monkeypatch.setattr(marginal.np, "asarray", forbidden_asarray)

    # #1929: no node-count cap; an astronomical q_xi is now caught by the
    # overflow-safe tensor-grid-size bound instead of a fixed-table lookup.
    with pytest.raises(ValueError, match="tensor-grid limit"):
        marginal.fit_marginal_numpy(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            model="MLSRM",
            latent_dim=3,
            q_xi=10**200, q_theta=21, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_tensor_node_count_caps_before_multiplication() -> None:
    """The tensor-node count uses division checks and contains no power operator."""
    assert marginal._bounded_tensor_node_count(  # type: ignore[attr-defined]
        10**200,
        3,
        limit=1_000_000,
    ) == 1_000_001

    tree = ast.parse(inspect.getsource(marginal._bounded_tensor_node_count))  # type: ignore[attr-defined]
    assert not any(isinstance(node, ast.Pow) for node in ast.walk(tree))


def test_nonspatial_preflight_does_not_inspect_unused_q_xi() -> None:
    """MIRT owns one latent-space placeholder node and ignores tensor settings."""
    assert marginal._preflight_xi_node_count(  # type: ignore[attr-defined]
        xi_rule="gh",
        q_xi=_HostileInteger(),
        xi_points=256,
        latent_dim=3,
        uses_space=False,
    ) == 1


def test_qmc_preflight_uses_exact_point_count_without_inspecting_q_xi() -> None:
    """QMC preflight is governed by exact xi_points rather than tensor q_xi."""
    assert marginal._preflight_xi_node_count(  # type: ignore[attr-defined]
        xi_rule="qmc",
        q_xi=_HostileInteger(),
        xi_points=17,
        latent_dim=3,
        uses_space=True,
    ) == 17
