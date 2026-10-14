"""Tests for the valuation module — DCF, WACC, CAPM, risk metrics, portfolio."""

import math

import numpy as np
import pytest
from ellements.domain_specific.finance.risk import (
    PortfolioResult,
    RiskMetrics,
    calculate_beta,
    max_drawdown,
    portfolio_analysis,
    portfolio_return,
    portfolio_volatility,
    risk_metrics,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from ellements.domain_specific.finance.valuation import (
    DCFResult,
    EquityValueResult,
    FCFResult,
    TerminalValueResult,
    WACCResult,
    capm,
    dcf,
    ev_to_equity,
    fcfe,
    fcff,
    gordon_growth_model,
    terminal_value_exit_multiple,
    terminal_value_gordon,
    wacc,
)

# ── WACC ─────────────────────────────────────────────────────────

class TestWACC:

    def test_basic_wacc(self):
        """Classic example: 60% equity at 10%, 40% debt at 5%, 21% tax."""
        result = wacc(
            equity_value=600, debt_value=400,
            cost_of_equity=0.10, cost_of_debt=0.05, tax_rate=0.21,
        )
        assert isinstance(result, WACCResult)
        # WACC = 0.6×0.10 + 0.4×0.05×0.79 = 0.06 + 0.0158 = 0.0758
        assert result.wacc == pytest.approx(0.0758, abs=1e-4)
        assert result.equity_weight == pytest.approx(0.6)
        assert result.debt_weight == pytest.approx(0.4)

    def test_all_equity(self):
        result = wacc(equity_value=1000, debt_value=0,
                      cost_of_equity=0.12, cost_of_debt=0.0, tax_rate=0.25)
        assert result.wacc == pytest.approx(0.12)

    def test_all_debt(self):
        result = wacc(equity_value=0, debt_value=1000,
                      cost_of_equity=0.0, cost_of_debt=0.06, tax_rate=0.30)
        assert result.wacc == pytest.approx(0.06 * 0.70, abs=1e-6)

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError):
            wacc(0, 0, 0.10, 0.05, 0.21)


# ── CAPM ─────────────────────────────────────────────────────────

class TestCAPM:

    def test_basic_capm(self):
        # Re = 0.04 + 1.2 × (0.10 − 0.04) = 0.04 + 0.072 = 0.112
        result = capm(risk_free_rate=0.04, beta=1.2, market_return=0.10)
        assert result == pytest.approx(0.112)

    def test_zero_beta(self):
        result = capm(risk_free_rate=0.03, beta=0.0, market_return=0.10)
        assert result == pytest.approx(0.03)

    def test_beta_one(self):
        result = capm(risk_free_rate=0.04, beta=1.0, market_return=0.10)
        assert result == pytest.approx(0.10)

    def test_negative_beta(self):
        result = capm(risk_free_rate=0.04, beta=-0.5, market_return=0.10)
        assert result == pytest.approx(0.04 + (-0.5) * 0.06)


# ── Terminal Value ───────────────────────────────────────────────

class TestTerminalValue:

    def test_gordon_growth(self):
        # TV = 100 × 1.02 / (0.10 − 0.02) = 102 / 0.08 = 1275
        result = terminal_value_gordon(100, growth_rate=0.02, discount_rate=0.10)
        assert isinstance(result, TerminalValueResult)
        assert result.terminal_value == pytest.approx(1275.0)
        assert result.method == "gordon_growth"

    def test_gordon_growth_rate_exceeds_discount(self):
        with pytest.raises(ValueError, match="must exceed"):
            terminal_value_gordon(100, growth_rate=0.10, discount_rate=0.05)

    def test_exit_multiple(self):
        result = terminal_value_exit_multiple(final_metric=200, multiple=10)
        assert result.terminal_value == pytest.approx(2000.0)
        assert result.method == "exit_multiple"


# ── Gordon Growth Model ──────────────────────────────────────────

class TestGordonGrowth:

    def test_basic_gordon(self):
        # P = 5 × 1.03 / (0.10 − 0.03) = 5.15 / 0.07 ≈ 73.57
        result = gordon_growth_model(5.0, growth_rate=0.03, discount_rate=0.10)
        assert result == pytest.approx(73.5714, abs=0.01)

    def test_rate_exceeds_discount_raises(self):
        with pytest.raises(ValueError):
            gordon_growth_model(5.0, growth_rate=0.10, discount_rate=0.05)


# ── DCF ──────────────────────────────────────────────────────────

class TestDCF:

    def test_basic_dcf_no_terminal(self):
        """Simple 3-year DCF at 10%."""
        # PV = 100/1.1 + 110/1.21 + 121/1.331
        result = dcf([100, 110, 121], discount_rate=0.10)
        assert isinstance(result, DCFResult)
        expected = 100/1.1 + 110/1.21 + 121/1.331
        assert result.pv_cash_flows == pytest.approx(expected, abs=0.01)
        assert result.pv_terminal_value == pytest.approx(0.0)
        assert result.enterprise_value == pytest.approx(expected, abs=0.01)
        assert result.num_periods == 3

    def test_dcf_with_terminal_value(self):
        """5-year DCF with terminal value."""
        cfs = [100, 105, 110, 116, 122]
        tv = 2000
        result = dcf(cfs, discount_rate=0.10, terminal_value=tv)
        pv_tv = tv / (1.10 ** 5)
        assert result.pv_terminal_value == pytest.approx(pv_tv, abs=0.01)
        assert result.enterprise_value == pytest.approx(
            result.pv_cash_flows + pv_tv, abs=0.01
        )
        assert result.terminal_value_undiscounted == tv

    def test_dcf_single_period(self):
        result = dcf([500], discount_rate=0.08)
        assert result.pv_cash_flows == pytest.approx(500 / 1.08, abs=0.01)

    def test_dcf_empty_raises(self):
        with pytest.raises(ValueError):
            dcf([], discount_rate=0.10)


# ── Free Cash Flow ───────────────────────────────────────────────

class TestFCF:

    def test_fcff(self):
        # FCFF = 1000×(1−0.25) + 200 − 300 − 50 = 750 + 200 − 300 − 50 = 600
        result = fcff(ebit=1000, tax_rate=0.25, depreciation=200,
                      capex=300, change_in_working_capital=50)
        assert isinstance(result, FCFResult)
        assert result.fcf == pytest.approx(600.0)
        assert result.method == "fcff"

    def test_fcfe(self):
        # FCFE = 500 + 100 − 200 − 30 + 50 = 420
        result = fcfe(net_income=500, depreciation=100, capex=200,
                      change_in_working_capital=30, net_borrowing=50)
        assert result.fcf == pytest.approx(420.0)
        assert result.method == "fcfe"


# ── EV to Equity ─────────────────────────────────────────────────

class TestEVToEquity:

    def test_basic_bridge(self):
        result = ev_to_equity(enterprise_value=10000, total_debt=3000,
                              cash_and_equivalents=500)
        assert isinstance(result, EquityValueResult)
        assert result.equity_value == pytest.approx(7500.0)
        assert result.equity_value_per_share is None

    def test_with_shares(self):
        result = ev_to_equity(enterprise_value=10000, total_debt=2000,
                              cash_and_equivalents=1000, shares_outstanding=100)
        assert result.equity_value == pytest.approx(9000.0)
        assert result.equity_value_per_share == pytest.approx(90.0)


# ── Sharpe Ratio ─────────────────────────────────────────────────

class TestSharpeRatio:

    def test_positive_sharpe(self):
        returns = [0.01, 0.02, 0.015, 0.03, 0.005]
        result = sharpe_ratio(returns, risk_free_rate=0.001)
        assert result > 0

    def test_zero_volatility(self):
        returns = [0.01, 0.01, 0.01]
        assert sharpe_ratio(returns) == 0.0

    def test_negative_returns(self):
        returns = [-0.02, -0.01, -0.03, -0.015]
        result = sharpe_ratio(returns, risk_free_rate=0.0)
        assert result < 0


# ── Sortino Ratio ────────────────────────────────────────────────

class TestSortinoRatio:

    def test_positive_sortino(self):
        returns = [0.01, 0.02, -0.005, 0.03, 0.015]
        result = sortino_ratio(returns, risk_free_rate=0.0)
        assert result > 0

    def test_no_downside(self):
        returns = [0.01, 0.02, 0.03]
        result = sortino_ratio(returns, risk_free_rate=0.0)
        assert result == float("inf")


# ── Beta ─────────────────────────────────────────────────────────

class TestBeta:

    def test_beta_of_market(self):
        """Market vs itself should have beta ≈ 1.0."""
        market = [0.01, -0.02, 0.015, 0.03, -0.01, 0.02]
        result = calculate_beta(market, market)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError):
            calculate_beta([0.01, 0.02], [0.01])

    def test_too_few_periods_raises(self):
        with pytest.raises(ValueError):
            calculate_beta([0.01], [0.01])

    def test_higher_beta(self):
        """An asset that moves 2x the market should have beta ≈ 2."""
        np.random.seed(42)
        market = np.random.normal(0, 0.01, 100).tolist()
        asset = [2 * r + np.random.normal(0, 0.001) for r in market]
        result = calculate_beta(asset, market)
        assert result == pytest.approx(2.0, abs=0.15)


# ── Value at Risk ────────────────────────────────────────────────

class TestVaR:

    def test_var_positive(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000).tolist()
        var95 = value_at_risk(returns, 0.95)
        assert var95 > 0  # Should be a positive loss number

    def test_var_99_larger_than_95(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000).tolist()
        var95 = value_at_risk(returns, 0.95)
        var99 = value_at_risk(returns, 0.99)
        assert var99 > var95


# ── Max Drawdown ─────────────────────────────────────────────────

class TestMaxDrawdown:

    def test_no_drawdown(self):
        returns = [0.01, 0.01, 0.01, 0.01]
        assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # Up 10%, then down 20% → peak at 1.1, trough at 0.88
        returns = [0.10, -0.20]
        dd = max_drawdown(returns)
        assert dd == pytest.approx(0.20, abs=0.01)


# ── Risk Metrics (aggregate) ────────────────────────────────────

class TestRiskMetrics:

    def test_risk_metrics_with_market(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252).tolist()
        market = np.random.normal(0.0008, 0.015, 252).tolist()
        result = risk_metrics(returns, market_returns=market, periods_per_year=252)
        assert isinstance(result, RiskMetrics)
        assert result.beta is not None
        assert result.sharpe_ratio is not None
        assert result.annualized_return is not None
        assert result.max_drawdown is not None

    def test_risk_metrics_without_market(self):
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        result = risk_metrics(returns, periods_per_year=12)
        assert result.beta is None
        assert result.sharpe_ratio is not None


# ── Portfolio Analytics ──────────────────────────────────────────

class TestPortfolioReturn:

    def test_equal_weight(self):
        result = portfolio_return([0.5, 0.5], [0.10, 0.06])
        assert result == pytest.approx(0.08)

    def test_single_asset(self):
        result = portfolio_return([1.0], [0.12])
        assert result == pytest.approx(0.12)


class TestPortfolioVolatility:

    def test_single_asset(self):
        # Single asset with variance 0.04 → vol = 0.2
        result = portfolio_volatility([1.0], [[0.04]])
        assert result == pytest.approx(0.2)

    def test_two_asset_uncorrelated(self):
        # 50/50 split, equal variance 0.04, zero correlation
        cov = [[0.04, 0.0], [0.0, 0.04]]
        result = portfolio_volatility([0.5, 0.5], cov)
        # σp = √(0.25×0.04 + 0.25×0.04) = √0.02 ≈ 0.1414
        assert result == pytest.approx(math.sqrt(0.02), abs=0.001)


class TestPortfolioAnalysis:

    def test_full_analysis(self):
        result = portfolio_analysis(
            weights=[0.6, 0.4],
            expected_returns=[0.10, 0.06],
            cov_matrix=[[0.04, 0.01], [0.01, 0.02]],
            risk_free_rate=0.03,
        )
        assert isinstance(result, PortfolioResult)
        assert result.expected_return == pytest.approx(0.084)
        assert result.volatility > 0
        assert result.sharpe_ratio > 0
        assert result.weights == [0.6, 0.4]


# ── Pydantic serialization ──────────────────────────────────────

class TestSerialization:

    def test_dcf_result_json(self):
        result = dcf([100, 200], discount_rate=0.10)
        data = result.model_dump()
        assert "enterprise_value" in data
        assert "pv_cash_flows" in data

    def test_wacc_result_json(self):
        result = wacc(600, 400, 0.10, 0.05, 0.21)
        data = result.model_dump()
        assert "wacc" in data
        assert "equity_weight" in data

    def test_risk_metrics_json(self):
        result = risk_metrics([0.01, -0.02, 0.015], periods_per_year=12)
        data = result.model_dump()
        assert "sharpe_ratio" in data
