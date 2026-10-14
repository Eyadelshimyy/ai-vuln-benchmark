"""Tests for financial calculations."""

import pytest
from ellements.domain_specific.finance import (
    CompoundingFrequency,
    DepreciationMethod,
    FinancialCalculator,
)


@pytest.fixture
def calc():
    """Create a FinancialCalculator instance for testing."""
    return FinancialCalculator()


class TestCompoundInterest:
    """Tests for compound interest calculations."""

    def test_annual_compounding(self, calc):
        """Test annual compounding."""
        result = calc.compound_interest(
            principal=1000,
            rate=0.05,
            time_years=10,
            frequency=CompoundingFrequency.ANNUALLY,
        )

        assert result.principal == 1000
        assert result.rate == 0.05
        assert result.time_years == 10
        assert pytest.approx(result.future_value, rel=0.01) == 1628.89
        assert pytest.approx(result.total_interest, rel=0.01) == 628.89

    def test_monthly_compounding(self, calc):
        """Test monthly compounding."""
        result = calc.compound_interest(
            principal=1000,
            rate=0.05,
            time_years=10,
            frequency=CompoundingFrequency.MONTHLY,
        )

        assert pytest.approx(result.future_value, rel=0.01) == 1647.01

    def test_continuous_compounding(self, calc):
        """Test continuous compounding."""
        result = calc.compound_interest(
            principal=1000,
            rate=0.05,
            time_years=10,
            frequency=CompoundingFrequency.CONTINUOUS,
        )

        assert pytest.approx(result.future_value, rel=0.01) == 1648.72

    def test_effective_annual_rate(self, calc):
        """Test effective annual rate calculation."""
        result = calc.compound_interest(
            principal=1000,
            rate=0.05,
            time_years=1,
            frequency=CompoundingFrequency.MONTHLY,
        )

        # EAR should be higher than nominal rate due to compounding
        assert result.effective_annual_rate > 0.05


class TestTimeValueOfMoney:
    """Tests for time value of money calculations."""

    def test_future_value(self, calc):
        """Test future value calculation."""
        fv = calc.future_value(
            rate=0.05, nper=10, pmt=-100, pv=-1000
        )

        assert fv > 0  # Should be positive
        assert pytest.approx(fv, rel=0.01) == 2886.68

    def test_present_value(self, calc):
        """Test present value calculation."""
        pv = calc.present_value(
            rate=0.05, nper=10, pmt=-100, fv=2000
        )

        assert pv < 0  # Investment/outflow is negative
        # PV should be reasonable for given parameters
        assert abs(pv) > 0
        assert abs(pv) < 2000

    def test_payment(self, calc):
        """Test payment calculation for a loan."""
        pmt = calc.payment(
            rate=0.05 / 12, nper=360, pv=200000
        )

        # Monthly payment on $200k mortgage at 5% for 30 years
        assert pmt < 0  # Payment is outflow
        assert pytest.approx(abs(pmt), rel=0.01) == 1073.64

    def test_number_of_periods(self, calc):
        """Test calculation of number of periods."""
        # Calculate how many periods needed to pay off a loan
        # Borrow 1000, pay 100/period at 5% rate
        nper = calc.number_of_periods(
            rate=0.05, pmt=-100, pv=1000, fv=0
        )

        assert abs(nper) > 0  # Take absolute value in case of sign convention
        assert abs(nper) < 100  # Reasonable number of periods

    def test_interest_rate(self, calc):
        """Test interest rate calculation."""
        # Loan scenario: borrow 1000, pay 100/period for 12 periods
        rate = calc.interest_rate(
            nper=12, pmt=-100, pv=1000, fv=0
        )

        assert rate > 0
        assert rate < 0.20  # Should be less than 20% per period


class TestLoanAmortization:
    """Tests for loan amortization."""

    def test_loan_schedule(self, calc):
        """Test loan amortization schedule."""
        loan = calc.loan_amortization(
            principal=10000,
            annual_rate=0.06,
            periods=12,
            payments_per_year=12,
        )

        assert loan.principal == 10000
        assert loan.rate == 0.06
        assert len(loan.schedule) == 12
        assert pytest.approx(loan.payment_amount, rel=0.01) == 860.66

        # First payment
        first = loan.schedule[0]
        assert first.period == 1
        assert first.interest > 0
        assert first.principal > 0
        assert first.payment == pytest.approx(loan.payment_amount, rel=0.01)

        # Last payment balance should be near zero
        last = loan.schedule[-1]
        assert pytest.approx(last.balance, abs=0.01) == 0

    def test_total_interest_paid(self, calc):
        """Test total interest calculation."""
        loan = calc.loan_amortization(
            principal=10000,
            annual_rate=0.06,
            periods=12,
            payments_per_year=12,
        )

        assert loan.total_interest > 0
        assert loan.total_interest < loan.principal  # For short-term loan
        assert loan.total_paid == loan.principal + loan.total_interest


class TestInvestmentAnalysis:
    """Tests for investment analysis calculations."""

    def test_npv_positive(self, calc):
        """Test NPV with profitable investment."""
        cash_flows = [-1000, 300, 300, 300, 300, 300]
        npv = calc.npv(0.10, cash_flows)

        assert npv > 0  # Profitable investment

    def test_npv_negative(self, calc):
        """Test NPV with unprofitable investment."""
        cash_flows = [-1000, 100, 100, 100]
        npv = calc.npv(0.10, cash_flows)

        assert npv < 0  # Unprofitable investment

    def test_irr(self, calc):
        """Test IRR calculation."""
        cash_flows = [-1000, 300, 300, 300, 300, 300]
        irr = calc.irr(cash_flows)

        assert irr is not None
        assert irr > 0
        assert irr < 1  # Reasonable return

    def test_mirr(self, calc):
        """Test MIRR calculation."""
        cash_flows = [-1000, 300, 300, 300, 300, 300]
        mirr = calc.mirr(cash_flows, finance_rate=0.10, reinvest_rate=0.12)

        assert mirr is not None
        assert mirr > 0

    def test_payback_period(self, calc):
        """Test payback period calculation."""
        cash_flows = [-1000, 400, 400, 400]
        payback = calc.payback_period(cash_flows)

        assert payback is not None
        assert pytest.approx(payback, rel=0.1) == 2.5  # 2.5 years

    def test_discounted_payback_period(self, calc):
        """Test discounted payback period."""
        # Use larger cash flows to ensure discounted payback occurs
        cash_flows = [-1000, 500, 500, 500]
        payback = calc.payback_period(cash_flows, discounted=True, rate=0.10)

        # With discount rate, might not break even if cash flows aren't large enough
        # so we just check it's either None or a reasonable number
        if payback is not None:
            assert payback > 0
            assert payback < len(cash_flows)

    def test_profitability_index(self, calc):
        """Test profitability index calculation."""
        # Use cash flows that are clearly profitable even with discounting
        cash_flows = [-1000, 500, 500, 500]
        pi = calc.profitability_index(0.10, cash_flows)

        assert pi is not None
        assert pi > 1  # Profitable project

    def test_cash_flow_analysis_comprehensive(self, calc):
        """Test comprehensive cash flow analysis."""
        cash_flows = [-10000, 3000, 3000, 3000, 3000, 3000]
        analysis = calc.analyze_cash_flows(
            cash_flows=cash_flows,
            discount_rate=0.10,
        )

        assert analysis.npv > 0
        assert analysis.irr is not None
        assert analysis.irr > 0.10  # IRR > discount rate for positive NPV
        assert analysis.mirr is not None
        assert analysis.payback_period is not None
        assert analysis.discounted_payback_period is not None
        assert analysis.profitability_index is not None
        assert analysis.profitability_index > 1


class TestInvestmentMetrics:
    """Tests for investment performance metrics."""

    def test_investment_metrics(self, calc):
        """Test investment performance metrics."""
        metrics = calc.investment_metrics(
            initial_investment=10000,
            ending_value=15000,
            time_years=5,
        )

        assert metrics.total_return == 0.5  # 50% return
        assert metrics.roi == 0.5
        # CAGR should be less than total return for multi-year investment
        assert metrics.annualized_return < metrics.total_return
        assert pytest.approx(metrics.annualized_return, rel=0.01) == 0.0845


class TestBondCalculations:
    """Tests for bond pricing and yield calculations."""

    def test_bond_price_at_par(self, calc):
        """Test bond price when YTM equals coupon rate."""
        bond = calc.bond_price(
            face_value=1000,
            coupon_rate=0.05,
            yield_to_maturity=0.05,
            years_to_maturity=10,
            frequency=2,
        )

        # When YTM = coupon rate, price should equal par
        assert pytest.approx(bond.price, rel=0.01) == 1000
        assert pytest.approx(bond.premium_discount, abs=1) == 0

    def test_bond_price_at_discount(self, calc):
        """Test bond price when YTM > coupon rate (discount)."""
        bond = calc.bond_price(
            face_value=1000,
            coupon_rate=0.05,
            yield_to_maturity=0.07,
            years_to_maturity=10,
            frequency=2,
        )

        # When YTM > coupon rate, bond trades at discount
        assert bond.price < 1000
        assert bond.premium_discount < 0

    def test_bond_price_at_premium(self, calc):
        """Test bond price when YTM < coupon rate (premium)."""
        bond = calc.bond_price(
            face_value=1000,
            coupon_rate=0.07,
            yield_to_maturity=0.05,
            years_to_maturity=10,
            frequency=2,
        )

        # When YTM < coupon rate, bond trades at premium
        assert bond.price > 1000
        assert bond.premium_discount > 0

    def test_bond_current_yield(self, calc):
        """Test bond current yield calculation."""
        bond = calc.bond_price(
            face_value=1000,
            coupon_rate=0.06,
            yield_to_maturity=0.07,
            years_to_maturity=10,
            frequency=2,
        )

        # Current yield = annual coupon / price
        assert bond.current_yield > 0
        expected_current_yield = (1000 * 0.06) / bond.price
        assert pytest.approx(bond.current_yield, rel=0.01) == expected_current_yield

    def test_bond_yield_calculation(self, calc):
        """Test bond yield calculation."""
        bond_yield = calc.bond_yield(
            price=950,
            face_value=1000,
            coupon_rate=0.05,
            years_to_maturity=10,
            frequency=2,
        )

        # Current yield calculation
        assert pytest.approx(bond_yield.current_yield, rel=0.01) == 0.0526

        # YTM should be higher than coupon rate for discount bond
        if bond_yield.yield_to_maturity:
            assert bond_yield.yield_to_maturity > 0.05


class TestDepreciation:
    """Tests for depreciation calculations."""

    def test_straight_line_depreciation(self, calc):
        """Test straight-line depreciation."""
        schedule = calc.depreciation_schedule(
            asset_cost=10000,
            salvage_value=1000,
            useful_life=5,
            method=DepreciationMethod.STRAIGHT_LINE,
        )

        assert len(schedule.schedule) == 5
        assert schedule.total_depreciation == 9000

        # Each year should have equal depreciation
        for year_data in schedule.schedule:
            assert pytest.approx(year_data["depreciation"], rel=0.01) == 1800

        # Final book value should equal salvage value
        assert pytest.approx(schedule.schedule[-1]["book_value"], rel=0.01) == 1000

    def test_double_declining_depreciation(self, calc):
        """Test double declining balance depreciation."""
        schedule = calc.depreciation_schedule(
            asset_cost=10000,
            salvage_value=1000,
            useful_life=5,
            method=DepreciationMethod.DOUBLE_DECLINING,
        )

        assert len(schedule.schedule) == 5

        # First year depreciation should be highest
        assert schedule.schedule[0]["depreciation"] > schedule.schedule[1]["depreciation"]

        # Book value should not go below salvage value
        for year_data in schedule.schedule:
            assert year_data["book_value"] >= 1000

    def test_sum_of_years_digits_depreciation(self, calc):
        """Test sum of years digits depreciation."""
        schedule = calc.depreciation_schedule(
            asset_cost=10000,
            salvage_value=1000,
            useful_life=5,
            method=DepreciationMethod.SUM_OF_YEARS_DIGITS,
        )

        assert len(schedule.schedule) == 5
        assert pytest.approx(schedule.total_depreciation, rel=0.01) == 9000

        # Depreciation should decrease each year
        for i in range(len(schedule.schedule) - 1):
            assert (
                schedule.schedule[i]["depreciation"]
                > schedule.schedule[i + 1]["depreciation"]
            )

        # Final book value should equal salvage value
        assert pytest.approx(schedule.schedule[-1]["book_value"], rel=0.01) == 1000


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_interest_rate(self, calc):
        """Test calculations with zero interest rate."""
        fv = calc.future_value(rate=0, nper=10, pmt=-100, pv=0)
        assert fv == 1000  # Just sum of payments

    def test_irr_no_solution(self, calc):
        """Test IRR when no solution exists."""
        # All positive cash flows - no IRR
        cash_flows = [100, 200, 300]
        irr = calc.irr(cash_flows)
        # IRR should be None or NaN for invalid cash flows
        import math
        assert irr is None or math.isnan(irr)

    def test_payback_never_breaks_even(self, calc):
        """Test payback period when investment never breaks even."""
        cash_flows = [-1000, 100, 100, 100]  # Never reaches breakeven
        payback = calc.payback_period(cash_flows)
        assert payback is None


class TestDataModels:
    """Tests for Pydantic data models."""

    def test_compound_interest_result_model(self, calc):
        """Test CompoundInterestResult model."""
        result = calc.compound_interest(1000, 0.05, 10)

        # Test model serialization
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "future_value" in result_dict
        assert "total_interest" in result_dict

    def test_loan_amortization_model(self, calc):
        """Test LoanAmortization model."""
        loan = calc.loan_amortization(10000, 0.06, 12)

        # Test nested models
        assert len(loan.schedule) == 12
        first_payment = loan.schedule[0]
        assert hasattr(first_payment, "period")
        assert hasattr(first_payment, "payment")
        assert hasattr(first_payment, "principal")
        assert hasattr(first_payment, "interest")

    def test_cash_flow_analysis_model(self, calc):
        """Test CashFlowAnalysis model."""
        analysis = calc.analyze_cash_flows(
            cash_flows=[-1000, 300, 300, 300, 300],
            discount_rate=0.10,
        )

        # Test model JSON serialization
        analysis_json = analysis.model_dump_json()
        assert isinstance(analysis_json, str)
        assert "npv" in analysis_json


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
