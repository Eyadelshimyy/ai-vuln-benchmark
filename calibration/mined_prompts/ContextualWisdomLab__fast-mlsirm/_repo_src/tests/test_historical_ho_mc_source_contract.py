"""Contracts for the retired historical higher-order DINA Monte Carlo gate.

The historical ``cdm::tests::mc_ho_recovery_500`` unit study asserted the
deterministic exact convergence proportion ``conv_rate >= 0.95``, which a
finite 500-replication Monte Carlo experiment cannot be required to attain.
The reviewed integration study
``higher_order_dina_recovery_respects_monte_carlo_tolerance`` reproduces the
same generating design, fixed seeds, and RMSE/bias/agreement thresholds while
gating convergence on the two-standard-error binomial floor. The historical
test has therefore been removed at the source level, and no workflow may
quarantine test names instead of fixing their source.
"""

from __future__ import annotations

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CDM_TESTS = _REPOSITORY_ROOT / "tests" / "unit" / "cdm_tests.rs"
_REPLACEMENT_STUDY = (
    _REPOSITORY_ROOT
    / "crates"
    / "mlsirm-core"
    / "tests"
    / "higher_order_mc_recovery.rs"
)
_STATISTICAL_WORKFLOW = (
    _REPOSITORY_ROOT / ".github" / "workflows" / "statistical-studies.yml"
)


def test_historical_exact_threshold_test_stays_removed() -> None:
    """The superseded exact-proportion study must not return, nor be skipped."""
    source = _CDM_TESTS.read_text(encoding="utf-8")
    workflow = _STATISTICAL_WORKFLOW.read_text(encoding="utf-8")

    assert "mc_ho_recovery_500" not in source
    assert "conv_rate >= 0.95" not in source
    assert "mc_ho_recovery_500" not in workflow


def test_replacement_gate_owns_finite_monte_carlo_acceptance() -> None:
    """The dedicated reviewed study must implement and execute the MCSE floor."""
    source = _REPLACEMENT_STUDY.read_text(encoding="utf-8")
    workflow = _STATISTICAL_WORKFLOW.read_text(encoding="utf-8")

    assert "fn higher_order_dina_recovery_respects_monte_carlo_tolerance()" in source
    assert "nominal_convergence" in source
    assert "monte_carlo_se" in source
    assert "convergence_floor" in source
    test_name = "higher_order_dina_recovery_respects_monte_carlo_tolerance"
    target_qualified_name = (
        "mlsirm-core/test/higher_order_mc_recovery::"
        "higher_order_dina_recovery_respects_monte_carlo_tolerance"
    )
    assert workflow.count(test_name) == 2
    assert f"--skip {target_qualified_name}" in workflow
    assert f"--skip {test_name}" not in workflow


def test_no_source_mutating_statistical_patch_workflow_is_tracked() -> None:
    """Reviewed source changes cannot be delegated to a write-capable CI patcher."""
    forbidden = (
        _REPOSITORY_ROOT
        / ".github"
        / "workflows"
        / "apply-reviewed-ho-mc-source-fix.yml"
    )
    assert not forbidden.exists()
