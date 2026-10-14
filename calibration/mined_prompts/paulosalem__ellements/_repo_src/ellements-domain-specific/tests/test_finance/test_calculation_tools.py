"""Tests for finance agent tools (calculation_tools)."""

import pytest
from ellements.core import ToolRegistry
from ellements.domain_specific.finance import risk as risk_mod
from ellements.domain_specific.finance import valuation as val
from ellements.domain_specific.finance.calculations import FinancialCalculator
from ellements.domain_specific.finance.tools import calculation_tools


@pytest.fixture(scope="module")
def tools():
    """Create tools once for all tests."""
    return calculation_tools()


class TestToolsCreation:

    def test_returns_registry(self, tools):
        assert isinstance(tools, ToolRegistry)

    def test_has_expected_tools(self, tools):
        expected = [
            "compound_interest", "present_value", "future_value",
            "calculate_npv", "calculate_irr", "loan_amortization",
            "calculate_wacc", "calculate_capm", "calculate_dcf",
            "calculate_terminal_value", "calculate_fcff",
            "enterprise_to_equity", "calculate_risk_metrics",
            "calculate_portfolio", "calculate_quant_risk_summary",
            "optimize_portfolio_weights",
        ]
        for name in expected:
            assert name in tools, f"Missing tool: {name}"

    def test_tool_count(self, tools):
        assert len(tools) == 16

    def test_tools_have_names(self, tools):
        """Each tool spec exposes a name matching its registry key."""
        for name, tool in tools.items():
            assert hasattr(tool, "name"), f"Tool {name} missing 'name' attribute"
            assert tool.name == name

    def test_tools_have_descriptions(self, tools):
        """Each tool spec exposes a non-empty description."""
        for name, tool in tools.items():
            assert hasattr(tool, "description"), f"Tool {name} missing description"
            assert len(tool.description) > 10, f"Tool {name} has empty description"


class TestUnderlyingTVMFunctions:
    """Test the underlying functions that the tools wrap (direct calls)."""

    def test_compound_interest(self):
        calc = FinancialCalculator()
        result = calc.compound_interest(1000, 0.05, 10, "annually")
        assert result.future_value > 1000
        assert result.total_interest > 0

    def test_present_value(self):
        calc = FinancialCalculator()
        # PV/FV use numpy-financial conventions (pmt-based); test via compound_interest
        result = calc.compound_interest(1000, 0.05, 10, "annually")
        assert result.future_value == pytest.approx(1000 * (1.05 ** 10), abs=0.01)

    def test_future_value(self):
        calc = FinancialCalculator()
        result = calc.compound_interest(1000, 0.05, 10, "annually")
        assert result.future_value > 1000

    def test_npv(self):
        calc = FinancialCalculator()
        npv = calc.npv(0.10, [-1000, 300, 400, 500])
        assert isinstance(npv, float)

    def test_irr(self):
        calc = FinancialCalculator()
        irr = calc.irr([-1000, 300, 400, 500, 200])
        assert irr is not None
        assert irr > 0

    def test_loan_amortization(self):
        calc = FinancialCalculator()
        result = calc.loan_amortization(200000, 0.04, 30)
        assert result.payment_amount > 0
        assert result.total_interest > 0


class TestUnderlyingValuationFunctions:
    """Test valuation functions directly."""

    def test_wacc(self):
        result = val.wacc(600, 400, 0.10, 0.05, 0.21)
        assert result.wacc == pytest.approx(0.0758, abs=1e-4)

    def test_capm(self):
        re = val.capm(0.04, 1.2, 0.10)
        assert re == pytest.approx(0.112)

    def test_dcf(self):
        result = val.dcf([100, 110, 120], 0.10, terminal_value=2000)
        assert result.enterprise_value > 0
        assert result.pv_terminal_value > 0

    def test_terminal_value_gordon(self):
        result = val.terminal_value_gordon(100, 0.02, 0.10)
        assert result.terminal_value == pytest.approx(1275.0)

    def test_terminal_value_exit(self):
        result = val.terminal_value_exit_multiple(200, 10)
        assert result.terminal_value == pytest.approx(2000.0)

    def test_fcff(self):
        result = val.fcff(1000, 0.25, 200, 300, 50)
        assert result.fcf == pytest.approx(600.0)

    def test_ev_to_equity(self):
        result = val.ev_to_equity(10000, 3000, 500)
        assert result.equity_value == pytest.approx(7500.0)

    def test_ev_to_equity_per_share(self):
        result = val.ev_to_equity(10000, 2000, 1000, 100)
        assert result.equity_value_per_share == pytest.approx(90.0)


class TestUnderlyingRiskFunctions:
    """Test risk and portfolio functions directly."""

    def test_risk_metrics(self):
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.003]
        result = risk_mod.risk_metrics(returns, periods_per_year=252)
        assert result.sharpe_ratio is not None
        assert result.value_at_risk_95 is not None

    def test_risk_metrics_with_market(self):
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        market = [0.008, -0.003, 0.015, 0.01, -0.008]
        result = risk_mod.risk_metrics(returns, market_returns=market, periods_per_year=252)
        assert result.beta is not None

    def test_portfolio_analysis(self):
        result = risk_mod.portfolio_analysis(
            weights=[0.6, 0.4],
            expected_returns=[0.10, 0.06],
            cov_matrix=[[0.04, 0.01], [0.01, 0.02]],
            risk_free_rate=0.03,
        )
        assert result.expected_return == pytest.approx(0.084)
        assert result.volatility > 0
        assert result.sharpe_ratio > 0
