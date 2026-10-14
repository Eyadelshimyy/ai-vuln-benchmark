"""Financial calculations module.

This module provides comprehensive financial calculation capabilities including:
- Time value of money (TVM) calculations
- Loan and mortgage calculations
- Investment analysis (NPV, IRR, MIRR, Payback Period)
- Bond pricing and yield calculations
- Discounted cash flow (DCF) analysis
- Compounding and present/future value calculations
- Depreciation methods
- Financial ratios and metrics

Uses numpy-financial for standard calculations and implements additional
advanced financial formulas with Pydantic models for type safety.
"""

from enum import StrEnum

import numpy_financial as npf
from pydantic import BaseModel, Field


class CompoundingFrequency(StrEnum):
    """Compounding frequency options."""

    ANNUALLY = "annually"
    SEMI_ANNUALLY = "semi_annually"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    CONTINUOUS = "continuous"


class DepreciationMethod(StrEnum):
    """Depreciation calculation methods."""

    STRAIGHT_LINE = "straight_line"
    DECLINING_BALANCE = "declining_balance"
    DOUBLE_DECLINING = "double_declining"
    SUM_OF_YEARS_DIGITS = "sum_of_years_digits"


class PaymentTiming(StrEnum):
    """Payment timing options."""

    END = "end"  # Payments at end of period (ordinary annuity)
    BEGIN = "begin"  # Payments at beginning of period (annuity due)


# ============================================================================
# Core TVM (Time Value of Money) Models
# ============================================================================


class CompoundInterestResult(BaseModel):
    """Result of compound interest calculation."""

    principal: float = Field(description="Initial principal amount")
    rate: float = Field(description="Annual interest rate (as decimal)")
    time_years: float = Field(description="Time period in years")
    frequency: CompoundingFrequency = Field(description="Compounding frequency")
    future_value: float = Field(description="Future value after compounding")
    total_interest: float = Field(description="Total interest earned")
    effective_annual_rate: float = Field(
        description="Effective annual rate (APY)"
    )


class LoanSchedulePayment(BaseModel):
    """Single payment in an amortization schedule."""

    period: int = Field(description="Payment period number")
    payment: float = Field(description="Total payment amount")
    principal: float = Field(description="Principal portion of payment")
    interest: float = Field(description="Interest portion of payment")
    balance: float = Field(description="Remaining loan balance")


class LoanAmortization(BaseModel):
    """Complete loan amortization schedule."""

    principal: float = Field(description="Original loan amount")
    rate: float = Field(description="Annual interest rate (as decimal)")
    periods: int = Field(description="Number of payment periods")
    payment_amount: float = Field(description="Fixed payment per period")
    total_paid: float = Field(description="Total amount paid over loan life")
    total_interest: float = Field(description="Total interest paid")
    schedule: list[LoanSchedulePayment] = Field(
        description="Detailed payment schedule"
    )


# ============================================================================
# Investment Analysis Models
# ============================================================================


class CashFlowAnalysis(BaseModel):
    """Analysis of a series of cash flows."""

    cash_flows: list[float] = Field(description="Series of cash flows")
    discount_rate: float = Field(description="Discount rate (as decimal)")
    npv: float = Field(description="Net Present Value")
    irr: float | None = Field(
        None, description="Internal Rate of Return (if calculable)"
    )
    mirr: float | None = Field(
        None, description="Modified Internal Rate of Return"
    )
    payback_period: float | None = Field(
        None, description="Payback period in periods"
    )
    discounted_payback_period: float | None = Field(
        None, description="Discounted payback period"
    )
    profitability_index: float | None = Field(
        None, description="Profitability index (PI)"
    )


class InvestmentMetrics(BaseModel):
    """Comprehensive investment performance metrics."""

    initial_investment: float = Field(description="Initial investment amount")
    ending_value: float = Field(description="Ending investment value")
    cash_flows: list[float] = Field(
        default_factory=list, description="Intermediate cash flows"
    )
    time_years: float = Field(description="Investment period in years")
    total_return: float = Field(description="Total return (as decimal)")
    annualized_return: float = Field(description="Annualized return (CAGR)")
    roi: float = Field(description="Return on Investment (as decimal)")


# ============================================================================
# Bond Calculations Models
# ============================================================================


class BondPrice(BaseModel):
    """Bond pricing calculation result."""

    face_value: float = Field(description="Bond face/par value")
    coupon_rate: float = Field(description="Annual coupon rate (as decimal)")
    yield_to_maturity: float = Field(description="YTM (as decimal)")
    years_to_maturity: float = Field(description="Years until maturity")
    frequency: int = Field(description="Coupon payments per year")
    price: float = Field(description="Current bond price")
    premium_discount: float = Field(
        description="Premium (+) or discount (-) from par"
    )
    current_yield: float = Field(description="Current yield (as decimal)")


class BondYield(BaseModel):
    """Bond yield calculation result."""

    price: float = Field(description="Current bond price")
    face_value: float = Field(description="Bond face/par value")
    coupon_rate: float = Field(description="Annual coupon rate (as decimal)")
    years_to_maturity: float = Field(description="Years until maturity")
    frequency: int = Field(description="Coupon payments per year")
    current_yield: float = Field(description="Current yield (as decimal)")
    yield_to_maturity: float | None = Field(
        None, description="Yield to maturity (as decimal)"
    )


# ============================================================================
# Depreciation Models
# ============================================================================


class DepreciationSchedule(BaseModel):
    """Depreciation schedule for an asset."""

    asset_cost: float = Field(description="Original cost of asset")
    salvage_value: float = Field(description="Salvage value at end of life")
    useful_life: int = Field(description="Useful life in years")
    method: DepreciationMethod = Field(description="Depreciation method")
    schedule: list[dict[str, float]] = Field(
        description="Year-by-year depreciation"
    )
    total_depreciation: float = Field(description="Total depreciation")


# ============================================================================
# Financial Calculator Class
# ============================================================================


class FinancialCalculator:
    """Comprehensive financial calculations toolkit.

    Provides methods for time value of money, investment analysis,
    loan calculations, bond pricing, and more.
    """

    @staticmethod
    def _get_compounding_periods(frequency: CompoundingFrequency) -> int:
        """Get number of compounding periods per year."""
        periods = {
            CompoundingFrequency.ANNUALLY: 1,
            CompoundingFrequency.SEMI_ANNUALLY: 2,
            CompoundingFrequency.QUARTERLY: 4,
            CompoundingFrequency.MONTHLY: 12,
            CompoundingFrequency.WEEKLY: 52,
            CompoundingFrequency.DAILY: 365,
        }
        return periods.get(frequency, 1)

    @staticmethod
    def _payment_timing_value(timing: PaymentTiming) -> int:
        """Convert payment timing to numpy-financial format."""
        return 1 if timing == PaymentTiming.BEGIN else 0

    # ========================================================================
    # Compound Interest & Time Value
    # ========================================================================

    def compound_interest(
        self,
        principal: float,
        rate: float,
        time_years: float,
        frequency: CompoundingFrequency = CompoundingFrequency.ANNUALLY,
    ) -> CompoundInterestResult:
        """Calculate compound interest.

        Args:
            principal: Initial principal amount
            rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
            time_years: Time period in years
            frequency: Compounding frequency

        Returns:
            CompoundInterestResult with future value and details
        """
        import math

        if frequency == CompoundingFrequency.CONTINUOUS:
            # Continuous compounding: A = Pe^(rt)
            future_value = principal * math.exp(rate * time_years)
            effective_rate = math.exp(rate) - 1
        else:
            # Periodic compounding: A = P(1 + r/n)^(nt)
            n = self._get_compounding_periods(frequency)
            future_value = principal * (1 + rate / n) ** (n * time_years)
            effective_rate = (1 + rate / n) ** n - 1

        total_interest = future_value - principal

        return CompoundInterestResult(
            principal=principal,
            rate=rate,
            time_years=time_years,
            frequency=frequency,
            future_value=future_value,
            total_interest=total_interest,
            effective_annual_rate=effective_rate,
        )

    def future_value(
        self,
        rate: float,
        nper: int,
        pmt: float,
        pv: float = 0,
        when: PaymentTiming = PaymentTiming.END,
    ) -> float:
        """Calculate future value of an investment.

        Args:
            rate: Interest rate per period (as decimal)
            nper: Total number of payment periods
            pmt: Payment per period
            pv: Present value (default 0)
            when: Payment timing (end or begin)

        Returns:
            Future value
        """
        when_val = self._payment_timing_value(when)
        return float(npf.fv(rate, nper, pmt, pv, when=when_val))

    def present_value(
        self,
        rate: float,
        nper: int,
        pmt: float,
        fv: float = 0,
        when: PaymentTiming = PaymentTiming.END,
    ) -> float:
        """Calculate present value of an investment.

        Args:
            rate: Interest rate per period (as decimal)
            nper: Total number of payment periods
            pmt: Payment per period
            fv: Future value (default 0)
            when: Payment timing (end or begin)

        Returns:
            Present value
        """
        when_val = self._payment_timing_value(when)
        return float(npf.pv(rate, nper, pmt, fv, when=when_val))

    def payment(
        self,
        rate: float,
        nper: int,
        pv: float,
        fv: float = 0,
        when: PaymentTiming = PaymentTiming.END,
    ) -> float:
        """Calculate fixed periodic payment.

        Args:
            rate: Interest rate per period (as decimal)
            nper: Total number of payment periods
            pv: Present value (loan amount)
            fv: Future value (default 0)
            when: Payment timing (end or begin)

        Returns:
            Payment amount per period
        """
        when_val = self._payment_timing_value(when)
        return float(npf.pmt(rate, nper, pv, fv, when=when_val))

    def number_of_periods(
        self,
        rate: float,
        pmt: float,
        pv: float,
        fv: float = 0,
        when: PaymentTiming = PaymentTiming.END,
    ) -> float:
        """Calculate number of payment periods.

        Args:
            rate: Interest rate per period (as decimal)
            pmt: Payment per period
            pv: Present value
            fv: Future value (default 0)
            when: Payment timing (end or begin)

        Returns:
            Number of periods
        """
        when_val = self._payment_timing_value(when)
        return float(npf.nper(rate, pmt, pv, fv, when=when_val))

    def interest_rate(
        self,
        nper: int,
        pmt: float,
        pv: float,
        fv: float = 0,
        when: PaymentTiming = PaymentTiming.END,
        guess: float = 0.1,
    ) -> float:
        """Calculate interest rate per period.

        Args:
            nper: Number of payment periods
            pmt: Payment per period
            pv: Present value
            fv: Future value (default 0)
            when: Payment timing (end or begin)
            guess: Starting guess for rate (default 0.1)

        Returns:
            Interest rate per period
        """
        when_val = self._payment_timing_value(when)
        return float(npf.rate(nper, pmt, pv, fv, when=when_val, guess=guess))

    # ========================================================================
    # Loan Amortization
    # ========================================================================

    def loan_amortization(
        self,
        principal: float,
        annual_rate: float,
        periods: int,
        payments_per_year: int = 12,
    ) -> LoanAmortization:
        """Calculate complete loan amortization schedule.

        Args:
            principal: Loan amount
            annual_rate: Annual interest rate (as decimal)
            periods: Total number of payment periods
            payments_per_year: Number of payments per year (default 12)

        Returns:
            LoanAmortization with complete schedule
        """
        period_rate = annual_rate / payments_per_year
        payment = -self.payment(period_rate, periods, principal)

        schedule = []
        balance = principal

        for period in range(1, periods + 1):
            interest_payment = balance * period_rate
            principal_payment = payment - interest_payment
            balance -= principal_payment

            schedule.append(
                LoanSchedulePayment(
                    period=period,
                    payment=payment,
                    principal=principal_payment,
                    interest=interest_payment,
                    balance=max(0, balance),  # Avoid negative due to rounding
                )
            )

        total_paid = payment * periods
        total_interest = total_paid - principal

        return LoanAmortization(
            principal=principal,
            rate=annual_rate,
            periods=periods,
            payment_amount=payment,
            total_paid=total_paid,
            total_interest=total_interest,
            schedule=schedule,
        )

    # ========================================================================
    # Investment Analysis
    # ========================================================================

    def npv(self, rate: float, cash_flows: list[float]) -> float:
        """Calculate Net Present Value.

        Args:
            rate: Discount rate per period (as decimal)
            cash_flows: List of cash flows (first is typically negative)

        Returns:
            Net Present Value
        """
        return float(npf.npv(rate, cash_flows))

    def irr(self, cash_flows: list[float]) -> float | None:
        """Calculate Internal Rate of Return.

        Args:
            cash_flows: List of cash flows (first is typically negative)

        Returns:
            IRR as decimal, or None if not calculable
        """
        import math

        try:
            result = float(npf.irr(cash_flows))
            # Check for invalid results
            if math.isnan(result) or math.isinf(result) or abs(result) > 1e6:
                return None
            return result
        except Exception:
            return None

    def mirr(
        self,
        cash_flows: list[float],
        finance_rate: float,
        reinvest_rate: float,
    ) -> float | None:
        """Calculate Modified Internal Rate of Return.

        Args:
            cash_flows: List of cash flows
            finance_rate: Interest rate paid on cash flow investments
            reinvest_rate: Interest rate received on reinvestment

        Returns:
            MIRR as decimal, or None if not calculable
        """
        try:
            return float(npf.mirr(cash_flows, finance_rate, reinvest_rate))
        except Exception:
            return None

    def payback_period(
        self, cash_flows: list[float], discounted: bool = False, rate: float = 0
    ) -> float | None:
        """Calculate payback period.

        Args:
            cash_flows: List of cash flows (first is typically negative)
            discounted: Whether to use discounted cash flows
            rate: Discount rate if using discounted payback

        Returns:
            Payback period in number of periods, or None if never breaks even
        """
        cumulative = 0.0
        for i, cf in enumerate(cash_flows):
            if discounted and rate > 0:
                cf = cf / ((1 + rate) ** i)
            cumulative += cf
            if cumulative >= 0:
                # Interpolate for fractional period
                if i > 0:
                    prev_cumulative = cumulative - cf
                    fraction = -prev_cumulative / cf
                    return i - 1 + fraction
                return float(i)
        return None

    def profitability_index(
        self, rate: float, cash_flows: list[float]
    ) -> float | None:
        """Calculate Profitability Index (PI).

        PI = PV of future cash flows / Initial investment

        Args:
            rate: Discount rate (as decimal)
            cash_flows: List of cash flows (first is typically negative)

        Returns:
            Profitability index, or None if initial investment is zero
        """
        if not cash_flows or cash_flows[0] >= 0:
            return None

        initial_investment = abs(cash_flows[0])
        future_cash_flows = cash_flows[1:]

        pv_future = sum(
            cf / ((1 + rate) ** (i + 1))
            for i, cf in enumerate(future_cash_flows)
        )

        return pv_future / initial_investment if initial_investment != 0 else None

    def analyze_cash_flows(
        self,
        cash_flows: list[float],
        discount_rate: float,
        finance_rate: float | None = None,
        reinvest_rate: float | None = None,
    ) -> CashFlowAnalysis:
        """Comprehensive analysis of cash flows.

        Args:
            cash_flows: List of cash flows
            discount_rate: Discount rate for NPV (as decimal)
            finance_rate: Finance rate for MIRR (defaults to discount_rate)
            reinvest_rate: Reinvestment rate for MIRR (defaults to discount_rate)

        Returns:
            CashFlowAnalysis with all metrics
        """
        finance_rate = finance_rate or discount_rate
        reinvest_rate = reinvest_rate or discount_rate

        npv_result = self.npv(discount_rate, cash_flows)
        irr_result = self.irr(cash_flows)
        mirr_result = self.mirr(cash_flows, finance_rate, reinvest_rate)
        payback = self.payback_period(cash_flows, discounted=False)
        discounted_payback = self.payback_period(
            cash_flows, discounted=True, rate=discount_rate
        )
        pi = self.profitability_index(discount_rate, cash_flows)

        return CashFlowAnalysis(
            cash_flows=cash_flows,
            discount_rate=discount_rate,
            npv=npv_result,
            irr=irr_result,
            mirr=mirr_result,
            payback_period=payback,
            discounted_payback_period=discounted_payback,
            profitability_index=pi,
        )

    def investment_metrics(
        self,
        initial_investment: float,
        ending_value: float,
        time_years: float,
        cash_flows: list[float] | None = None,
    ) -> InvestmentMetrics:
        """Calculate comprehensive investment metrics.

        Args:
            initial_investment: Initial investment amount
            ending_value: Ending value of investment
            time_years: Investment period in years
            cash_flows: Optional intermediate cash flows

        Returns:
            InvestmentMetrics with performance measures
        """
        total_return = (ending_value - initial_investment) / initial_investment
        annualized_return = (
            (ending_value / initial_investment) ** (1 / time_years) - 1
        )
        roi = total_return

        return InvestmentMetrics(
            initial_investment=initial_investment,
            ending_value=ending_value,
            cash_flows=cash_flows or [],
            time_years=time_years,
            total_return=total_return,
            annualized_return=annualized_return,
            roi=roi,
        )

    # ========================================================================
    # Bond Calculations
    # ========================================================================

    def bond_price(
        self,
        face_value: float,
        coupon_rate: float,
        yield_to_maturity: float,
        years_to_maturity: float,
        frequency: int = 2,
    ) -> BondPrice:
        """Calculate bond price.

        Args:
            face_value: Bond face/par value
            coupon_rate: Annual coupon rate (as decimal)
            yield_to_maturity: YTM (as decimal)
            years_to_maturity: Years until maturity
            frequency: Coupon payments per year (default 2 for semi-annual)

        Returns:
            BondPrice with calculated price and metrics
        """
        periods = int(years_to_maturity * frequency)
        coupon_payment = (face_value * coupon_rate) / frequency
        ytm_per_period = yield_to_maturity / frequency

        # PV of coupon payments
        pv_coupons = coupon_payment * (
            (1 - (1 + ytm_per_period) ** -periods) / ytm_per_period
        )

        # PV of face value
        pv_face = face_value / ((1 + ytm_per_period) ** periods)

        price = pv_coupons + pv_face
        premium_discount = price - face_value
        current_yield = (face_value * coupon_rate) / price

        return BondPrice(
            face_value=face_value,
            coupon_rate=coupon_rate,
            yield_to_maturity=yield_to_maturity,
            years_to_maturity=years_to_maturity,
            frequency=frequency,
            price=price,
            premium_discount=premium_discount,
            current_yield=current_yield,
        )

    def bond_yield(
        self,
        price: float,
        face_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        frequency: int = 2,
    ) -> BondYield:
        """Calculate bond yield metrics.

        Args:
            price: Current bond price
            face_value: Bond face/par value
            coupon_rate: Annual coupon rate (as decimal)
            years_to_maturity: Years until maturity
            frequency: Coupon payments per year (default 2)

        Returns:
            BondYield with calculated yields
        """
        current_yield = (face_value * coupon_rate) / price

        # Use Newton-Raphson to approximate YTM
        ytm = None
        try:
            # Initial guess
            guess = coupon_rate
            for _ in range(100):
                bond_price_result = self.bond_price(
                    face_value, coupon_rate, guess, years_to_maturity, frequency
                )
                price_diff = bond_price_result.price - price

                if abs(price_diff) < 0.01:  # Close enough
                    ytm = guess
                    break

                # Adjust guess
                if price_diff > 0:
                    guess += 0.001
                else:
                    guess -= 0.001

                if guess <= 0:
                    break
        except Exception:
            pass

        return BondYield(
            price=price,
            face_value=face_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            frequency=frequency,
            current_yield=current_yield,
            yield_to_maturity=ytm,
        )

    # ========================================================================
    # Depreciation
    # ========================================================================

    def depreciation_schedule(
        self,
        asset_cost: float,
        salvage_value: float,
        useful_life: int,
        method: DepreciationMethod = DepreciationMethod.STRAIGHT_LINE,
    ) -> DepreciationSchedule:
        """Calculate depreciation schedule.

        Args:
            asset_cost: Original cost of asset
            salvage_value: Estimated salvage value
            useful_life: Useful life in years
            method: Depreciation method

        Returns:
            DepreciationSchedule with year-by-year breakdown
        """
        depreciable_base = asset_cost - salvage_value
        schedule = []

        if method == DepreciationMethod.STRAIGHT_LINE:
            annual_depreciation = depreciable_base / useful_life
            for year in range(1, useful_life + 1):
                schedule.append(
                    {
                        "year": year,
                        "depreciation": annual_depreciation,
                        "accumulated": annual_depreciation * year,
                        "book_value": asset_cost
                        - (annual_depreciation * year),
                    }
                )

        elif method == DepreciationMethod.DECLINING_BALANCE:
            rate = 1.0 / useful_life
            book_value = asset_cost
            accumulated: float = 0.0
            for year in range(1, useful_life + 1):
                depreciation = book_value * rate
                depreciation = min(
                    depreciation, book_value - salvage_value
                )  # Don't go below salvage
                accumulated += depreciation
                book_value -= depreciation
                schedule.append(
                    {
                        "year": year,
                        "depreciation": depreciation,
                        "accumulated": accumulated,
                        "book_value": book_value,
                    }
                )

        elif method == DepreciationMethod.DOUBLE_DECLINING:
            rate = 2.0 / useful_life
            book_value = asset_cost
            accumulated = 0.0
            for year in range(1, useful_life + 1):
                depreciation = book_value * rate
                depreciation = min(depreciation, book_value - salvage_value)
                accumulated += depreciation
                book_value -= depreciation
                schedule.append(
                    {
                        "year": year,
                        "depreciation": depreciation,
                        "accumulated": accumulated,
                        "book_value": book_value,
                    }
                )

        elif method == DepreciationMethod.SUM_OF_YEARS_DIGITS:
            sum_of_years = (useful_life * (useful_life + 1)) // 2
            accumulated = 0.0
            for year in range(1, useful_life + 1):
                remaining_life = useful_life - year + 1
                depreciation = (
                    remaining_life / sum_of_years
                ) * depreciable_base
                accumulated += depreciation
                schedule.append(
                    {
                        "year": year,
                        "depreciation": depreciation,
                        "accumulated": accumulated,
                        "book_value": asset_cost - accumulated,
                    }
                )

        return DepreciationSchedule(
            asset_cost=asset_cost,
            salvage_value=salvage_value,
            useful_life=useful_life,
            method=method,
            schedule=schedule,
            total_depreciation=depreciable_base,
        )


# Convenience instance
calculator = FinancialCalculator()
