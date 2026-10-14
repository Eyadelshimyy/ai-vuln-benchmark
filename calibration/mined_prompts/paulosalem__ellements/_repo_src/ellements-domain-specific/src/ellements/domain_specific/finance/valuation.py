"""Valuation and corporate finance module.

Provides:
- DCF valuation (projected cash flows + terminal value → enterprise value)
- WACC (weighted average cost of capital)
- CAPM (cost of equity)
- Gordon Growth Model (terminal value / intrinsic value)
- Free Cash Flow (FCFF, FCFE)
- Enterprise → equity value bridge

All functions are pure, stateless, and return Pydantic models where
the result is non-trivial.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

# ── Result models ────────────────────────────────────────────────

class WACCResult(BaseModel):
    """Weighted average cost of capital breakdown."""
    wacc: float = Field(description="WACC as a decimal (e.g. 0.08 = 8%)")
    equity_weight: float = Field(description="E / (E + D)")
    debt_weight: float = Field(description="D / (E + D)")
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    after_tax_cost_of_debt: float = Field(description="Rd × (1 − Tc)")


class DCFResult(BaseModel):
    """Discounted cash flow valuation result."""
    pv_cash_flows: float = Field(description="Present value of projected cash flows")
    pv_terminal_value: float = Field(description="Present value of terminal value (0 if none)")
    enterprise_value: float = Field(description="Total DCF enterprise value")
    discount_rate: float
    num_periods: int
    terminal_value_undiscounted: float | None = None
    cash_flows: list[float]


class TerminalValueResult(BaseModel):
    """Terminal value calculation result."""
    terminal_value: float
    method: str  # "gordon_growth" or "exit_multiple"


class FCFResult(BaseModel):
    """Free cash flow calculation result."""
    fcf: float
    method: str  # "fcff" or "fcfe"


class EquityValueResult(BaseModel):
    """Enterprise value to equity value bridge."""
    enterprise_value: float
    total_debt: float
    cash_and_equivalents: float
    equity_value: float
    shares_outstanding: float | None = None
    equity_value_per_share: float | None = None


# ── WACC & Cost of Capital ───────────────────────────────────────

def wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> WACCResult:
    """Calculate Weighted Average Cost of Capital.

    WACC = (E/V × Re) + (D/V × Rd × (1 − Tc))

    Args:
        equity_value: Market value of equity (E)
        debt_value: Market value of debt (D)
        cost_of_equity: Required return on equity (Re), e.g. 0.10 for 10%
        cost_of_debt: Cost of debt before tax (Rd), e.g. 0.05 for 5%
        tax_rate: Corporate tax rate (Tc), e.g. 0.21 for 21%
    """
    total = equity_value + debt_value
    if total <= 0:
        raise ValueError("Total capital (E + D) must be positive.")

    e_weight = equity_value / total
    d_weight = debt_value / total
    after_tax_rd = cost_of_debt * (1 - tax_rate)
    result = (e_weight * cost_of_equity) + (d_weight * after_tax_rd)

    return WACCResult(
        wacc=result,
        equity_weight=e_weight,
        debt_weight=d_weight,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate,
        after_tax_cost_of_debt=after_tax_rd,
    )


def capm(
    risk_free_rate: float,
    beta: float,
    market_return: float,
) -> float:
    """Calculate cost of equity via the Capital Asset Pricing Model.

    Re = Rf + β × (Rm − Rf)

    Args:
        risk_free_rate: Risk-free rate (Rf), e.g. 0.04 for 4%
        beta: Asset's systematic risk (β)
        market_return: Expected market return (Rm), e.g. 0.10 for 10%

    Returns:
        Cost of equity as a decimal.
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)


# ── Terminal Value ───────────────────────────────────────────────

def terminal_value_gordon(
    final_cash_flow: float,
    growth_rate: float,
    discount_rate: float,
) -> TerminalValueResult:
    """Terminal value via Gordon Growth Model (perpetuity growth).

    TV = FCF × (1 + g) / (r − g)

    Args:
        final_cash_flow: Last projected cash flow
        growth_rate: Perpetual growth rate (g), e.g. 0.02 for 2%
        discount_rate: Discount rate (r), must be > growth_rate
    """
    if discount_rate <= growth_rate:
        raise ValueError(
            f"Discount rate ({discount_rate}) must exceed growth rate ({growth_rate})."
        )
    tv = final_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
    return TerminalValueResult(terminal_value=tv, method="gordon_growth")


def terminal_value_exit_multiple(
    final_metric: float,
    multiple: float,
) -> TerminalValueResult:
    """Terminal value via exit multiple method.

    TV = Metric × Multiple (e.g., EBITDA × EV/EBITDA)

    Args:
        final_metric: Final-year metric (e.g., EBITDA)
        multiple: Valuation multiple (e.g., 10x EV/EBITDA)
    """
    return TerminalValueResult(
        terminal_value=final_metric * multiple,
        method="exit_multiple",
    )


# ── Gordon Growth Model (intrinsic value) ────────────────────────

def gordon_growth_model(
    dividend_or_fcf: float,
    growth_rate: float,
    discount_rate: float,
) -> float:
    """Intrinsic value via Gordon Growth / Dividend Discount Model.

    P = D₁ / (r − g) = D₀ × (1 + g) / (r − g)

    Args:
        dividend_or_fcf: Current period dividend or cash flow (D₀)
        growth_rate: Constant growth rate (g)
        discount_rate: Required rate of return (r), must be > g

    Returns:
        Intrinsic value per share/unit.
    """
    if discount_rate <= growth_rate:
        raise ValueError(
            f"Discount rate ({discount_rate}) must exceed growth rate ({growth_rate})."
        )
    return dividend_or_fcf * (1 + growth_rate) / (discount_rate - growth_rate)


# ── DCF Valuation ────────────────────────────────────────────────

def dcf(
    projected_cash_flows: Sequence[float],
    discount_rate: float,
    terminal_value: float | None = None,
) -> DCFResult:
    """Full Discounted Cash Flow valuation.

    PV = Σ (CFₜ / (1+r)^t) + TV / (1+r)^n

    Args:
        projected_cash_flows: Cash flows for periods 1…n (NOT period 0)
        discount_rate: WACC or required return, e.g. 0.10 for 10%
        terminal_value: Undiscounted terminal value at end of last period (optional)

    Returns:
        DCFResult with enterprise value breakdown.
    """
    cfs = list(projected_cash_flows)
    n = len(cfs)
    if n == 0:
        raise ValueError("At least one projected cash flow is required.")

    # PV of each cash flow
    pv_cfs = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cfs, start=1))

    # PV of terminal value
    pv_tv = 0.0
    if terminal_value is not None and terminal_value != 0:
        pv_tv = terminal_value / (1 + discount_rate) ** n

    return DCFResult(
        pv_cash_flows=pv_cfs,
        pv_terminal_value=pv_tv,
        enterprise_value=pv_cfs + pv_tv,
        discount_rate=discount_rate,
        num_periods=n,
        terminal_value_undiscounted=terminal_value,
        cash_flows=cfs,
    )


# ── Free Cash Flow ───────────────────────────────────────────────

def fcff(
    ebit: float,
    tax_rate: float,
    depreciation: float,
    capex: float,
    change_in_working_capital: float,
) -> FCFResult:
    """Free Cash Flow to the Firm.

    FCFF = EBIT × (1 − T) + D&A − CapEx − ΔWC

    Args:
        ebit: Earnings before interest and taxes
        tax_rate: Corporate tax rate
        depreciation: Depreciation & amortization
        capex: Capital expenditures (positive number)
        change_in_working_capital: Increase in net working capital (positive = cash outflow)
    """
    result = ebit * (1 - tax_rate) + depreciation - capex - change_in_working_capital
    return FCFResult(fcf=result, method="fcff")


def fcfe(
    net_income: float,
    depreciation: float,
    capex: float,
    change_in_working_capital: float,
    net_borrowing: float,
) -> FCFResult:
    """Free Cash Flow to Equity.

    FCFE = Net Income + D&A − CapEx − ΔWC + Net Borrowing

    Args:
        net_income: Net income after tax
        depreciation: Depreciation & amortization
        capex: Capital expenditures (positive number)
        change_in_working_capital: Increase in net working capital (positive = outflow)
        net_borrowing: Net new debt issued (positive = inflow)
    """
    result = net_income + depreciation - capex - change_in_working_capital + net_borrowing
    return FCFResult(fcf=result, method="fcfe")


# ── Enterprise → Equity Bridge ───────────────────────────────────

def ev_to_equity(
    enterprise_value: float,
    total_debt: float,
    cash_and_equivalents: float,
    shares_outstanding: float | None = None,
) -> EquityValueResult:
    """Convert enterprise value to equity value.

    Equity Value = EV − Debt + Cash

    Args:
        enterprise_value: Enterprise value (e.g., from DCF)
        total_debt: Total interest-bearing debt
        cash_and_equivalents: Cash and cash equivalents
        shares_outstanding: Optional — if provided, calculates per-share value
    """
    equity = enterprise_value - total_debt + cash_and_equivalents
    per_share = equity / shares_outstanding if shares_outstanding and shares_outstanding > 0 else None

    return EquityValueResult(
        enterprise_value=enterprise_value,
        total_debt=total_debt,
        cash_and_equivalents=cash_and_equivalents,
        equity_value=equity,
        shares_outstanding=shares_outstanding,
        equity_value_per_share=per_share,
    )
