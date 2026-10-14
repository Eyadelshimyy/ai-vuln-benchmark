"""Contracts for the supported Python versions exercised by pull-request CI."""

from pathlib import Path


_CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _python_matrix_job_source() -> str:
    """Return the matrix Python suite job without unrelated CI jobs."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  python-matrix:\n")
    end = workflow.index("\n  python:\n", start)
    return workflow[start:end]


def _python_gate_job_source() -> str:
    """Return the required-status aggregate job named exactly ``python``."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  python:\n")
    end = workflow.index("\n  rust:\n", start)
    return workflow[start:end]


def test_python_job_runs_full_suite_on_supported_cpythons() -> None:
    """Pull-request CI must execute the full fail-slow suite on supported CPythons."""
    python_job = _python_matrix_job_source()

    assert 'python-version: ["3.12", "3.14"]' in python_job
    assert "python-version: ${{ matrix.python-version }}" in python_job
    assert "fail-fast: false" in python_job
    assert "- run: pytest" in python_job


def test_advertised_python_floor_matches_ci_and_dependency_lock() -> None:
    """pyproject must not advertise a floor below the CI matrix / installable lock."""
    pyproject = _PYPROJECT.read_text(encoding="utf-8")
    python_job = _python_matrix_job_source()

    assert 'requires-python = ">=3.12"' in pyproject
    assert 'python-version: ["3.12", "3.14"]' in python_job
    # Hashed CI deps currently pin NumPy 2.5.x which needs CPython >=3.12.
    assert "3.10" not in python_job


def test_required_python_check_context_aggregates_matrix() -> None:
    """Branch protection requires the exact check name ``python``.

    Matrix legs alone report as versioned Python checks and do not satisfy that
    context. A non-matrix aggregate job must re-export matrix success under the
    protected name.
    """
    gate = _python_gate_job_source()
    assert "needs: python-matrix" in gate
    assert "name: python" in gate
    assert 'test "${{ needs.python-matrix.result }}" = "success"' in gate
