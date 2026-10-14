"""Genuine Rust-backed coverage for every public circle-arc method alias."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.equating as E


@pytest.mark.parametrize(
    "method",
    ["1", "arc1", "circlearc1", "2", "arc2", "circlearc2"],
)
def test_circle_arc_method_aliases_reach_rust(method: str) -> None:
    """Keep the Python admission set executable against the Rust parser."""
    scores = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    result = E.circle_arc_equate(
        scores,
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        method=method,
    )

    assert isinstance(result, E.CircleArcResult)
    assert result.method == method
    np.testing.assert_allclose(result.equated, scores)
