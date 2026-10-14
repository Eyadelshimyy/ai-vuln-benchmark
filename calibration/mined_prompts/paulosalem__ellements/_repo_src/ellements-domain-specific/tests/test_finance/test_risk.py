"""Tests for risk analytics and portfolio optimization."""

import numpy as np
import pytest
from ellements.domain_specific.finance.risk import (
    RiskMetrics,
    conditional_value_at_risk,
    optimize_portfolio_min_variance,
    risk_metrics,
    simulated_portfolio_scenarios,
    value_at_risk,
)


class TestRiskTailMetrics:
    def test_cvar_is_not_less_than_var(self):
        np.random.seed(42)
        returns = np.random.normal(0.0004, 0.02, 1000).tolist()
        var95 = value_at_risk(returns, confidence_level=0.95, method="historical")
        cvar95 = conditional_value_at_risk(
            returns, confidence_level=0.95, method="historical"
        )
        assert cvar95 >= var95

    def test_gaussian_var_supported(self):
        np.random.seed(42)
        returns = np.random.normal(0.0004, 0.02, 1000).tolist()
        var95 = value_at_risk(returns, confidence_level=0.95, method="gaussian")
        assert var95 > 0

    def test_risk_metrics_contains_var_and_cvar(self):
        returns = [0.01, -0.02, 0.015, -0.01, 0.004, -0.005, 0.009, 0.006]
        result = risk_metrics(returns, periods_per_year=252)
        assert isinstance(result, RiskMetrics)
        assert result.value_at_risk_95 is not None
        assert result.conditional_value_at_risk_95 is not None
        assert result.conditional_value_at_risk_95 >= result.value_at_risk_95
        assert result.calmar_ratio is not None


class TestPortfolioOptimization:
    def test_min_variance_weights_are_valid(self):
        expected_returns = [0.08, 0.06, 0.10]
        cov_matrix = [
            [0.04, 0.01, 0.015],
            [0.01, 0.03, 0.01],
            [0.015, 0.01, 0.05],
        ]
        result = optimize_portfolio_min_variance(
            expected_returns, cov_matrix, allow_short=False
        )
        assert len(result.weights) == 3
        assert sum(result.weights) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= 0 for w in result.weights)
        assert result.volatility > 0

    def test_target_return_optimization_runs(self):
        expected_returns = [0.05, 0.08, 0.11]
        cov_matrix = [
            [0.02, 0.005, 0.003],
            [0.005, 0.03, 0.008],
            [0.003, 0.008, 0.05],
        ]
        result = optimize_portfolio_min_variance(
            expected_returns,
            cov_matrix,
            target_return=0.085,
            allow_short=True,
        )
        assert result.method in {
            "minimum_variance_target_return",
            "minimum_variance_fallback",
        }
        assert sum(result.weights) == pytest.approx(1.0, abs=1e-6)


class TestSimulatedScenarios:
    def test_scenarios_are_deterministic_and_complete(self):
        scenarios = simulated_portfolio_scenarios(periods=120, seed=7)
        assert {
            "balanced",
            "concentrated_growth",
            "defensive_income",
            "high_volatility",
            "regime_shift",
        } == set(scenarios.keys())
        assert all(len(series) == 120 for series in scenarios.values())

        scenarios2 = simulated_portfolio_scenarios(periods=120, seed=7)
        assert scenarios["balanced"] == scenarios2["balanced"]

    def test_high_volatility_scenario_is_riskier_than_defensive(self):
        scenarios = simulated_portfolio_scenarios(periods=252, seed=13)
        high_vol = np.std(np.array(scenarios["high_volatility"]), ddof=1)
        defensive = np.std(np.array(scenarios["defensive_income"]), ddof=1)
        assert high_vol > defensive
