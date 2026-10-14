"""Bounded-admission regressions for shared IRT response and mask evidence."""

from __future__ import annotations

import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from fast_mlsirm import irt_contract


def _run_probe(source: str) -> subprocess.CompletedProcess[str]:
    """Run a potentially cyclic public-boundary probe under a hard deadline."""
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )


def test_cyclic_response_tree_fails_closed_within_deadline() -> None:
    """A self-referential response list must not keep the preflight stack alive."""
    completed = _run_probe(
        "\n".join(
            [
                "from fast_mlsirm.irt_contract import validate_irt_response_matrix",
                "responses = []",
                "responses.append(responses)",
                "validate_irt_response_matrix(responses, 'dichotomous')",
            ]
        )
    )

    assert completed.returncode != 0
    assert "cyclic" in completed.stderr.lower()


def test_cyclic_mask_tree_fails_closed_within_deadline() -> None:
    """A self-referential mask list must not keep the mask preflight stack alive."""
    completed = _run_probe(
        "\n".join(
            [
                "import numpy as np",
                "from fast_mlsirm.irt_contract import fit_irt_experiment",
                "mask = []",
                "mask.append(mask)",
                "responses = np.array([[0,1],[1,0],[0,1],[1,0],[0,1]], dtype=float)",
                "fit_irt_experiment(lambda matrix, **kwargs: matrix, responses, 'dichotomous', factor_ids=(0,0), mask=mask)",
            ]
        )
    )

    assert completed.returncode != 0
    assert "cyclic" in completed.stderr.lower()


def test_non_2d_response_rejected_before_contiguous_materialization(monkeypatch) -> None:
    """Rank-invalid exact arrays must fail before dtype/contiguous copying."""
    responses = np.broadcast_to(np.array([0], dtype=np.int8), (2, 3, 4))

    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("contiguous materialization must not run")

    monkeypatch.setattr(irt_contract.np, "ascontiguousarray", fail_materialization)

    with pytest.raises(ValueError, match="2-D persons x items"):
        irt_contract.validate_irt_response_matrix(responses, "dichotomous")


def test_oversized_response_rejected_before_contiguous_materialization(monkeypatch) -> None:
    """Zero-stride logical size must be bounded before a dense float64 copy."""
    responses = np.broadcast_to(np.array([[0, 1]], dtype=np.int8), (10_000_001, 2))

    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("contiguous materialization must not run")

    monkeypatch.setattr(irt_contract.np, "ascontiguousarray", fail_materialization)

    with pytest.raises(ValueError, match="20,000,000"):
        irt_contract.validate_irt_response_matrix(responses, "dichotomous")


def test_empty_container_fanout_has_bounded_preflight_work(monkeypatch) -> None:
    """Zero-cell container fan-out must remain bounded before NumPy materialization."""
    monkeypatch.setattr(irt_contract, "MAX_IRT_RESPONSE_CELLS", 2)
    responses = [[], [], [], [], [], []]

    with pytest.raises(ValueError, match="structural-work"):
        irt_contract.validate_irt_response_matrix(responses, "dichotomous")


def test_wide_fanout_does_not_allocate_a_sibling_sized_traversal_stack(monkeypatch) -> None:
    """Structural rejection must keep transient traversal memory independent of fan-out."""
    monkeypatch.setattr(irt_contract, "MAX_IRT_RESPONSE_CELLS", 2)
    responses = [[] for _ in range(100_000)]

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="structural-work"):
            irt_contract.validate_irt_response_matrix(responses, "dichotomous")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 2_000_000


def test_oversized_nested_numpy_row_rejected_before_numpy_materialization(monkeypatch) -> None:
    """Logical cells hidden in an inert NumPy row must count before sequence stacking."""
    row = np.broadcast_to(np.array([0], dtype=np.int8), (20_000_001,))
    responses = [row]

    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("NumPy sequence materialization must not run")

    monkeypatch.setattr(irt_contract.np, "asarray", fail_materialization)

    with pytest.raises(ValueError, match="20,000,000"):
        irt_contract.validate_irt_response_matrix(responses, "dichotomous")
