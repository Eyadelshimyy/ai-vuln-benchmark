"""Fail-first contracts for bounded item-model integration-rule validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.gpcm import fit_gpcm
from fast_mlsirm.grm import fit_grm
from fast_mlsirm.nominal import fit_nominal
from fast_mlsirm.twopl import fit_2pl


class _HostileNodeRule:
    """Invalid rule whose representation hooks must never run during validation."""

    def __str__(self) -> str:
        raise AssertionError("NODE_RULE_STR_SENTINEL")

    def __repr__(self) -> str:
        raise AssertionError("NODE_RULE_REPR_SENTINEL")


def _polytomous_responses() -> np.ndarray:
    """Return a tiny valid three-category response matrix for public-fit preflight."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )


def _binary_responses() -> np.ndarray:
    """Return a tiny valid binary response matrix for the public 2PL preflight."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("fitter", [fit_grm, fit_gpcm, fit_nominal])
def test_public_polytomous_fit_rejects_hostile_node_rule_without_stringification(
    fitter,
) -> None:
    """Invalid integration controls fail with a package error before caller hooks run."""
    with pytest.raises(ValueError, match="node_rule must be one of the supported integration rules"):
        fitter(
            _polytomous_responses(),
            3,
            model=1,
            q=7,
            max_iter=1,
            tol=1e-6,
            xi_points=4000,
            xi_seed=0x9E37_79B9_7F4A_7C15,
            node_rule=_HostileNodeRule(),
        )


def test_public_2pl_fit_rejects_hostile_node_rule_without_stringification() -> None:
    """The binary 2PL boundary shares the same bounded integration-rule contract."""
    with pytest.raises(ValueError, match="node_rule must be one of the supported integration rules"):
        fit_2pl(
            _binary_responses(),
            model=1,
            q=7,
            max_iter=1,
            node_rule=_HostileNodeRule(), tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15
        )
