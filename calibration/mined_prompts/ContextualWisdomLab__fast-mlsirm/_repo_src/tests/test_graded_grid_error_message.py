"""Work-budget diagnostics must survive Python's decimal integer safety limit."""
import sys

import pytest

from fast_mlsirm.graded_item import _admit_grid_work


@pytest.mark.parametrize(
    "work,budget", [(10**5000, 1), (10**5001, 10**5000)],
    ids=["huge-work", "huge-work-and-budget"],
)
def test_large_integer_rejection_retains_domain_diagnostic(work, budget):
    original = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        with pytest.raises(ValueError, match="Cartesian quadrature grid.*max_grid_points"):
            _admit_grid_work(work, budget, "synthetic exact integer workload")
        assert sys.get_int_max_str_digits() == 4300
    finally:
        sys.set_int_max_str_digits(original)


def test_small_integer_rejection_keeps_exact_counts():
    with pytest.raises(ValueError, match="grid has 9 points.*max_grid_points=8"):
        _admit_grid_work(9, 8, "fixture")


def test_huge_budget_still_admits_equal_work():
    _admit_grid_work(10**5000, 10**5000, "fixture")


def test_public_scale_huge_count_rejects_before_node_construction(monkeypatch):
    import numpy as np
    import fast_mlsirm.graded_item as graded

    def forbid_nodes(count):
        raise AssertionError("nodes must not be constructed")

    monkeypatch.setattr(graded, "equal_probability_normal_nodes", forbid_nodes)
    with pytest.raises(ValueError, match="Cartesian quadrature grid.*max_grid_points"):
        graded.expected_graded_scale_score(
            np.array([[1.0]]), np.array([[0.0]]), np.array([0.0]),
            [[0]], np.array([1.0]), n_nodes=10**5000, max_grid_points=1,
        )
