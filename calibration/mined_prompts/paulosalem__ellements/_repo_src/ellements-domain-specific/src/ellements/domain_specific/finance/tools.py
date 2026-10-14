"""Agent tools for financial calculations and valuation.

Provides plain tool functions that agents can call for:
- Corporate finance: DCF, WACC, CAPM, terminal value, free cash flow
- Risk metrics: Sharpe, Sortino, Beta, VaR
- Portfolio analytics
- Time-value-of-money: compound interest, PV, FV, NPV, IRR
- Loan amortization, bond pricing

These tools stay framework-agnostic; agent backends adapt them as needed.
"""

from __future__ import annotations

from ellements.core import ToolRegistry


def calculation_tools() -> ToolRegistry:
    """Create financial calculation tools for agents.

    Returns tools wrapping FinancialCalculator (TVM, loans, bonds,
    depreciation) and the valuation module (DCF, WACC, CAPM, risk).

    Returns:
        Dict of plain tool callables ready for adaptation by agent frameworks.

    Example:
        >>> tools = calculation_tools()
        >>> agent = AgentBuilder("FinanceBot").with_tools(tools).build()
    """

    from ellements.domain_specific.finance import quant_analysis as qrisk
    from ellements.domain_specific.finance import risk as risk_mod
    from ellements.domain_specific.finance import valuation as val
    from ellements.domain_specific.finance.calculations import (
        CompoundingFrequency,
        FinancialCalculator,
    )

    calc = FinancialCalculator()

    # ── TVM & Basic ──────────────────────────────────────────────
    def compound_interest(
        principal: float,
        annual_rate: float,
        years: float,
        compounding: str = "annually",
    ) -> str:
        """Calculate compound interest on a principal amount.

        Args:
            principal: Initial investment amount
            annual_rate: Annual interest rate as decimal (e.g. 0.05 for 5%)
            years: Number of years
            compounding: Frequency — annually, semi_annually, quarterly, monthly, daily, continuous
        """
        result = calc.compound_interest(
            principal, annual_rate, years, CompoundingFrequency(compounding)
        )
        return (
            f"Principal: ${principal:,.2f}\n"
            f"Rate: {annual_rate:.2%} ({compounding})\n"
            f"Years: {years}\n"
            f"Final Amount: ${result.future_value:,.2f}\n"
            f"Interest Earned: ${result.total_interest:,.2f}\n"
            f"Effective Annual Rate: {result.effective_annual_rate:.4%}"
        )
    def present_value(
        future_value: float,
        rate: float,
        periods: int,
    ) -> str:
        """Calculate present value of a future amount.

        Args:
            future_value: Future amount
            rate: Discount rate per period (e.g. 0.05 for 5%)
            periods: Number of periods
        """
        pv = calc.present_value(rate, periods, 0, future_value)
        return f"Present Value: ${pv:,.2f} (FV=${future_value:,.2f}, rate={rate:.2%}, periods={periods})"
    def future_value(
        present_value_amount: float,
        rate: float,
        periods: int,
    ) -> str:
        """Calculate future value of a present amount.

        Args:
            present_value_amount: Current amount
            rate: Growth rate per period (e.g. 0.05 for 5%)
            periods: Number of periods
        """
        fv = calc.future_value(rate, periods, 0, present_value_amount)
        return f"Future Value: ${fv:,.2f} (PV=${present_value_amount:,.2f}, rate={rate:.2%}, periods={periods})"
    def calculate_npv(
        rate: float,
        cash_flows: list[float],
    ) -> str:
        """Calculate Net Present Value of a series of cash flows.

        Args:
            rate: Discount rate per period
            cash_flows: Cash flows starting from period 0 (initial investment usually negative)
        """
        npv = calc.npv(rate, cash_flows)
        return f"NPV: ${npv:,.2f} at discount rate {rate:.2%}\nCash flows: {cash_flows}"
    def calculate_irr(
        cash_flows: list[float],
    ) -> str:
        """Calculate Internal Rate of Return for a series of cash flows.

        Args:
            cash_flows: Cash flows starting from period 0 (initial investment usually negative)
        """
        irr = calc.irr(cash_flows)
        if irr is None:
            return "No IRR found for the given cash flows."
        return f"IRR: {irr:.4%}\nCash flows: {cash_flows}"
    def loan_amortization(
        principal: float,
        annual_rate: float,
        years: int,
        payments_per_year: int = 12,
    ) -> str:
        """Calculate loan amortization schedule summary.

        Args:
            principal: Loan amount
            annual_rate: Annual interest rate as decimal
            years: Loan term in years
            payments_per_year: Payments per year (12 for monthly)
        """
        result = calc.loan_amortization(principal, annual_rate, years, payments_per_year)
        return (
            f"Loan: ${principal:,.2f} at {annual_rate:.2%} for {years} years\n"
            f"Payment: ${result.payment_amount:,.2f}/period\n"
            f"Total Payments: ${result.total_paid:,.2f}\n"
            f"Total Interest: ${result.total_interest:,.2f}\n"
            f"Number of Payments: {result.periods}"
        )

    # ── Valuation & Corporate Finance ────────────────────────────
    def calculate_wacc(
        equity_value: float,
        debt_value: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float,
    ) -> str:
        """Calculate Weighted Average Cost of Capital (WACC).

        WACC = (E/V × Re) + (D/V × Rd × (1 − Tc))

        Args:
            equity_value: Market value of equity
            debt_value: Market value of debt
            cost_of_equity: Required return on equity (decimal, e.g. 0.10)
            cost_of_debt: Pre-tax cost of debt (decimal)
            tax_rate: Corporate tax rate (decimal, e.g. 0.21)
        """
        result = val.wacc(equity_value, debt_value, cost_of_equity, cost_of_debt, tax_rate)
        return (
            f"WACC: {result.wacc:.4%}\n"
            f"Equity Weight: {result.equity_weight:.2%} × Cost of Equity {cost_of_equity:.2%}\n"
            f"Debt Weight: {result.debt_weight:.2%} × After-tax Cost of Debt {result.after_tax_cost_of_debt:.4%}\n"
            f"Tax Rate: {tax_rate:.2%}"
        )
    def calculate_capm(
        risk_free_rate: float,
        beta: float,
        market_return: float,
    ) -> str:
        """Calculate cost of equity using the Capital Asset Pricing Model (CAPM).

        Re = Rf + β × (Rm − Rf)

        Args:
            risk_free_rate: Risk-free rate (e.g. 0.04 for 4%)
            beta: Asset's beta (systematic risk)
            market_return: Expected market return (e.g. 0.10 for 10%)
        """
        re = val.capm(risk_free_rate, beta, market_return)
        premium = beta * (market_return - risk_free_rate)
        return (
            f"Cost of Equity (CAPM): {re:.4%}\n"
            f"Risk-Free Rate: {risk_free_rate:.2%}\n"
            f"Beta: {beta:.2f}\n"
            f"Market Premium: {market_return - risk_free_rate:.2%}\n"
            f"Risk Premium: {premium:.4%}"
        )
    def calculate_dcf(
        projected_cash_flows: list[float],
        discount_rate: float,
        terminal_value: float | None = None,
    ) -> str:
        """Perform a full Discounted Cash Flow (DCF) valuation.

        Args:
            projected_cash_flows: Projected free cash flows for periods 1 to n
            discount_rate: Discount rate (e.g. WACC), as decimal
            terminal_value: Undiscounted terminal value at end of projection (optional)
        """
        result = val.dcf(projected_cash_flows, discount_rate, terminal_value)
        lines = [
            "═══ DCF Valuation ═══",
            f"Discount Rate: {discount_rate:.2%}",
            f"Projection Periods: {result.num_periods}",
            f"PV of Cash Flows: ${result.pv_cash_flows:,.2f}",
        ]
        if terminal_value:
            lines.append(f"Terminal Value (undiscounted): ${terminal_value:,.2f}")
            lines.append(f"PV of Terminal Value: ${result.pv_terminal_value:,.2f}")
        lines.append(f"Enterprise Value: ${result.enterprise_value:,.2f}")
        return "\n".join(lines)
    def calculate_terminal_value(
        final_cash_flow: float,
        method: str = "gordon",
        growth_rate: float = 0.02,
        discount_rate: float = 0.10,
        exit_multiple: float = 10.0,
    ) -> str:
        """Calculate terminal value using Gordon Growth or exit multiple method.

        Args:
            final_cash_flow: Last projected cash flow or EBITDA
            method: 'gordon' for perpetuity growth or 'exit_multiple'
            growth_rate: Perpetual growth rate (for gordon method)
            discount_rate: Discount rate (for gordon method)
            exit_multiple: Valuation multiple (for exit_multiple method)
        """
        if method == "gordon":
            result = val.terminal_value_gordon(final_cash_flow, growth_rate, discount_rate)
            return (
                f"Terminal Value (Gordon Growth): ${result.terminal_value:,.2f}\n"
                f"Final CF: ${final_cash_flow:,.2f} × (1+{growth_rate:.2%}) / ({discount_rate:.2%} − {growth_rate:.2%})"
            )
        else:
            result = val.terminal_value_exit_multiple(final_cash_flow, exit_multiple)
            return (
                f"Terminal Value (Exit Multiple): ${result.terminal_value:,.2f}\n"
                f"Final Metric: ${final_cash_flow:,.2f} × {exit_multiple:.1f}x"
            )
    def calculate_fcff(
        ebit: float,
        tax_rate: float,
        depreciation: float,
        capex: float,
        change_in_working_capital: float,
    ) -> str:
        """Calculate Free Cash Flow to the Firm (FCFF).

        FCFF = EBIT × (1 − T) + D&A − CapEx − ΔWC

        Args:
            ebit: Earnings before interest and taxes
            tax_rate: Corporate tax rate
            depreciation: Depreciation & amortization
            capex: Capital expenditures
            change_in_working_capital: Change in net working capital
        """
        result = val.fcff(ebit, tax_rate, depreciation, capex, change_in_working_capital)
        return (
            f"FCFF: ${result.fcf:,.2f}\n"
            f"EBIT×(1−T): ${ebit * (1 - tax_rate):,.2f} + D&A: ${depreciation:,.2f}"
            f" − CapEx: ${capex:,.2f} − ΔWC: ${change_in_working_capital:,.2f}"
        )
    def enterprise_to_equity(
        enterprise_value: float,
        total_debt: float,
        cash: float,
        shares_outstanding: float | None = None,
    ) -> str:
        """Convert enterprise value to equity value (and per-share if shares provided).

        Equity Value = EV − Debt + Cash

        Args:
            enterprise_value: Enterprise value from DCF or comparable analysis
            total_debt: Total interest-bearing debt
            cash: Cash and cash equivalents
            shares_outstanding: Number of shares (optional, for per-share price)
        """
        result = val.ev_to_equity(enterprise_value, total_debt, cash, shares_outstanding)
        lines = [
            f"Enterprise Value: ${enterprise_value:,.2f}",
            f"− Debt: ${total_debt:,.2f}",
            f"+ Cash: ${cash:,.2f}",
            f"= Equity Value: ${result.equity_value:,.2f}",
        ]
        if result.equity_value_per_share is not None:
            lines.append(f"Per Share ({shares_outstanding:,.0f} shares): ${result.equity_value_per_share:,.2f}")
        return "\n".join(lines)

    # ── Risk Metrics ─────────────────────────────────────────────
    def calculate_risk_metrics(
        returns: list[float],
        market_returns: list[float] | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> str:
        """Calculate comprehensive risk metrics for a return series.

        Args:
            returns: Periodic returns (e.g. daily returns as decimals)
            market_returns: Optional market returns for beta calculation
            risk_free_rate: Risk-free rate per period
            periods_per_year: 252 for daily, 12 for monthly, 4 for quarterly
        """
        result = risk_mod.risk_metrics(
            returns,
            market_returns,
            risk_free_rate,
            periods_per_year,
        )
        lines = ["═══ Risk Metrics ═══"]
        if result.annualized_return is not None:
            lines.append(f"Annualized Return: {result.annualized_return:.4%}")
        if result.annualized_volatility is not None:
            lines.append(f"Annualized Volatility: {result.annualized_volatility:.4%}")
        if result.sharpe_ratio is not None:
            lines.append(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        if result.sortino_ratio is not None:
            lines.append(f"Sortino Ratio: {result.sortino_ratio:.4f}")
        if result.beta is not None:
            lines.append(f"Beta: {result.beta:.4f}")
        if result.value_at_risk_95 is not None:
            lines.append(f"VaR (95%): {result.value_at_risk_95:.4%}")
        if result.value_at_risk_99 is not None:
            lines.append(f"VaR (99%): {result.value_at_risk_99:.4%}")
        if result.conditional_value_at_risk_95 is not None:
            lines.append(f"CVaR (95%): {result.conditional_value_at_risk_95:.4%}")
        if result.conditional_value_at_risk_99 is not None:
            lines.append(f"CVaR (99%): {result.conditional_value_at_risk_99:.4%}")
        if result.max_drawdown is not None:
            lines.append(f"Max Drawdown: {result.max_drawdown:.4%}")
        if result.calmar_ratio is not None:
            lines.append(f"Calmar Ratio: {result.calmar_ratio:.4f}")
        if result.win_rate is not None:
            lines.append(f"Win Rate: {result.win_rate:.2%}")
        return "\n".join(lines)
    def calculate_portfolio(
        weights: list[float],
        expected_returns: list[float],
        cov_matrix: list[list[float]],
        risk_free_rate: float = 0.0,
    ) -> str:
        """Analyze a portfolio — return, volatility, and Sharpe ratio.

        Args:
            weights: Portfolio weights (should sum to 1.0)
            expected_returns: Expected return per asset
            cov_matrix: Covariance matrix of asset returns (n×n)
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        result = risk_mod.portfolio_analysis(
            weights,
            expected_returns,
            cov_matrix,
            risk_free_rate,
        )
        return (
            f"═══ Portfolio Analysis ═══\n"
            f"Expected Return: {result.expected_return:.4%}\n"
            f"Volatility: {result.volatility:.4%}\n"
            f"Sharpe Ratio: {result.sharpe_ratio:.4f}\n"
            f"Weights: {[f'{w:.2%}' for w in result.weights]}"
        )
    def calculate_quant_risk_summary(
        returns: list[float],
        benchmark_returns: list[float] | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        benchmark_symbol: str | None = None,
        var_method: str = "historical",
    ) -> str:
        """Generate a quantitative portfolio risk summary.

        Uses QuantStats when available and falls back to internal risk
        analytics otherwise.

        Args:
            returns: Portfolio periodic returns.
            benchmark_returns: Optional benchmark periodic returns.
            risk_free_rate: Risk-free rate per period.
            periods_per_year: 252 for daily, 12 for monthly.
            benchmark_symbol: Optional benchmark label (e.g. SPY).
            var_method: VaR/CVaR method ('historical' or 'gaussian').
        """

        result = qrisk.summarize_quant_risk(
            returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            benchmark_symbol=benchmark_symbol,
            var_method=var_method,
        )
        lines = ["═══ Quantitative Risk Summary ═══"]
        lines.append(
            f"Engine: {'QuantStats' if result.quantstats_used else 'Internal fallback'}"
        )
        lines.append(f"Samples: {result.sample_size}")
        if result.benchmark_symbol:
            lines.append(f"Benchmark: {result.benchmark_symbol}")
        if result.annualized_return is not None:
            lines.append(f"Annualized Return: {result.annualized_return:.4%}")
        if result.annualized_volatility is not None:
            lines.append(f"Annualized Volatility: {result.annualized_volatility:.4%}")
        if result.sharpe_ratio is not None:
            lines.append(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        if result.sortino_ratio is not None:
            lines.append(f"Sortino Ratio: {result.sortino_ratio:.4f}")
        if result.calmar_ratio is not None:
            lines.append(f"Calmar Ratio: {result.calmar_ratio:.4f}")
        if result.max_drawdown is not None:
            lines.append(f"Max Drawdown: {result.max_drawdown:.4%}")
        if result.value_at_risk_95 is not None:
            lines.append(f"VaR (95%): {result.value_at_risk_95:.4%}")
        if result.value_at_risk_99 is not None:
            lines.append(f"VaR (99%): {result.value_at_risk_99:.4%}")
        if result.conditional_value_at_risk_95 is not None:
            lines.append(f"CVaR (95%): {result.conditional_value_at_risk_95:.4%}")
        if result.conditional_value_at_risk_99 is not None:
            lines.append(f"CVaR (99%): {result.conditional_value_at_risk_99:.4%}")
        if result.beta is not None:
            lines.append(f"Beta: {result.beta:.4f}")
        if result.win_rate is not None:
            lines.append(f"Win Rate: {result.win_rate:.2%}")
        return "\n".join(lines)
    def optimize_portfolio_weights(
        expected_returns: list[float],
        cov_matrix: list[list[float]],
        target_return: float | None = None,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
    ) -> str:
        """Optimize portfolio weights with a mean-variance model.

        Args:
            expected_returns: Expected return per asset.
            cov_matrix: Return covariance matrix.
            target_return: Optional target expected return.
            risk_free_rate: Risk-free rate used for Sharpe output.
            allow_short: Whether short positions are allowed.
        """

        result = risk_mod.optimize_portfolio_min_variance(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
        )
        return (
            f"═══ Portfolio Optimization ({result.method}) ═══\n"
            f"Expected Return: {result.expected_return:.4%}\n"
            f"Volatility: {result.volatility:.4%}\n"
            f"Sharpe Ratio: {result.sharpe_ratio:.4f}\n"
            f"Weights: {[f'{w:.2%}' for w in result.weights]}"
        )

    return ToolRegistry.from_mapping({
        # TVM & Basic
        "compound_interest": compound_interest,
        "present_value": present_value,
        "future_value": future_value,
        "calculate_npv": calculate_npv,
        "calculate_irr": calculate_irr,
        "loan_amortization": loan_amortization,
        # Valuation
        "calculate_wacc": calculate_wacc,
        "calculate_capm": calculate_capm,
        "calculate_dcf": calculate_dcf,
        "calculate_terminal_value": calculate_terminal_value,
        "calculate_fcff": calculate_fcff,
        "enterprise_to_equity": enterprise_to_equity,
        # Risk & Portfolio
        "calculate_risk_metrics": calculate_risk_metrics,
        "calculate_portfolio": calculate_portfolio,
        "calculate_quant_risk_summary": calculate_quant_risk_summary,
        "optimize_portfolio_weights": optimize_portfolio_weights,
    })
