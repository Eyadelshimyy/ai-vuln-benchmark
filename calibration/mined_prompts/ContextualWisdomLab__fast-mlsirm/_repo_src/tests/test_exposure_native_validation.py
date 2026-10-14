"""Native validation coverage retained when Python mirrors Rust domains."""

from __future__ import annotations

import numpy as np
import pytest


def _binary_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return package-owned native-compatible binary item evidence."""

    a = np.array([1.0, 1.0], dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    c = np.zeros(2, dtype=np.float64)
    responses = np.array([1, 0], dtype=np.uint8)
    return a, b, c, responses


@pytest.mark.parametrize(
    ("theta_cut", "delta", "alpha", "beta"),
    [
        (np.nan, 0.5, 0.05, 0.05),
        (0.0, 0.0, 0.05, 0.05),
        (0.0, 0.5, 0.0, 0.05),
        (0.0, 0.5, 0.05, 0.0),
        (0.0, 0.5, 0.6, 0.5),
    ],
)
def test_sprt_native_retains_public_domain_guards(
    theta_cut: float,
    delta: float,
    alpha: float,
    beta: float,
) -> None:
    """Exercise Rust SPRT domains without the public Python prevalidation layer."""

    from fast_mlsirm import _core

    a, b, c, responses = _binary_inputs()
    with pytest.raises(ValueError):
        _core.py_sprt_classify(
            a,
            b,
            c,
            responses,
            theta_cut,
            delta,
            alpha,
            beta,
        )


@pytest.mark.parametrize(("theta_cut", "z_crit"), [(np.inf, 1.96), (0.0, 0.0)])
def test_ci_native_retains_public_domain_guards(
    theta_cut: float,
    z_crit: float,
) -> None:
    """Exercise Rust CI domains without the public Python prevalidation layer."""

    from fast_mlsirm import _core

    a, b, c, responses = _binary_inputs()
    with pytest.raises(ValueError):
        _core.py_ci_classify(a, b, c, responses, theta_cut, z_crit)


@pytest.mark.parametrize(("r_max", "tol"), [(0.0, 0.02), (0.25, -0.01)])
def test_sympson_native_retains_shared_domain_guards(
    r_max: float,
    tol: float,
) -> None:
    """Exercise Rust Sympson-Hetter backstops shared with Python admission."""

    from fast_mlsirm import _core

    a = np.ones(4, dtype=np.float64)
    b = np.zeros(4, dtype=np.float64)
    c = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError):
        _core.py_sympson_hetter(
            a,
            b,
            c,
            r_max,
            1,
            8,
            2,
            tol,
            20250724,
            11,
        )
