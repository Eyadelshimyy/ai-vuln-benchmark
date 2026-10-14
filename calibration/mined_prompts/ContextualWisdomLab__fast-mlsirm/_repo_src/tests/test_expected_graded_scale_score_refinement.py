"""Fixed-fixture refinement characterization, not scientific acceptance.

References are retained independent continuous integrals. Twice each retained
error is engineering regression headroom, not a universal accuracy guarantee.
No production fitting or runtime reference integrator is used.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from fast_mlsirm.graded_item import expected_graded_scale_score

_DATA = json.loads((Path(__file__).parent / "fixtures" /
                    "expected_graded_scale_refinement.json").read_text())


@pytest.mark.parametrize("name", sorted(_DATA["fixtures"]))
def test_fixed_fixture_refines_toward_independent_continuous_reference(name):
    fixture = _DATA["fixtures"][name]
    inputs = fixture["inputs"]
    arrays = {key: np.asarray(value, dtype=float) for key, value in inputs.items()
              if key != "integrate_columns"}
    errors, values = [], []
    for n in (121, 241, 481, 961):
        budget = sum(n ** len(cols) for cols in inputs["integrate_columns"])
        value = expected_graded_scale_score(
            **arrays, integrate_columns=inputs["integrate_columns"],
            n_nodes=n, max_grid_points=budget,
        )
        assert math.isfinite(value)
        error = abs(value - fixture["continuous_reference"])
        assert error <= 2 * fixture["retained_errors"][str(n)], (name, n, error)
        values.append(value)
        errors.append(error)
    # These deliberately nonsymmetric fixtures have nonzero approximation error.
    assert all(a > b > 0 for a, b in zip(errors, errors[1:])), (name, errors)
    differences = [abs(b - a) for a, b in zip(values, values[1:])]
    assert all(a > b > 0 for a, b in zip(differences, differences[1:]))
