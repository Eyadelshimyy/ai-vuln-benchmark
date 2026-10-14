"""Pydantic data models and serialization helpers for Yahoo Finance.

Data shapes only; live data retrieval lives in :mod:`yahoo_finance`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN check
        return None
    return parsed


def _serialize_indicator_rows(indicators_df: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in indicators_df.iterrows():
        payload: dict[str, Any] = {"date": idx.isoformat()}
        for col in indicators_df.columns:
            payload[col] = _float_or_none(row[col])
        rows.append(payload)
    return rows


class AssetInfo(BaseModel):
    """Basic information about a financial asset."""
    symbol: str = Field(description="Ticker symbol")
    name: str = Field(description="Asset name")
    asset_type: str = Field(description="Type of asset (stock, etf, etc.)")
    exchange: str = Field(default="", description="Exchange where traded")
    currency: str = Field(default="USD", description="Trading currency")


class AssetQuote(BaseModel):
    """Current quote data for a financial asset."""
    symbol: str = Field(description="Ticker symbol")
    name: str = Field(description="Asset name")
    current_price: float | None = Field(default=None, description="Current price")
    previous_close: float | None = Field(default=None, description="Previous closing price")
    open_price: float | None = Field(default=None, description="Opening price")
    day_high: float | None = Field(default=None, description="Day's high price")
    day_low: float | None = Field(default=None, description="Day's low price")
    volume: int | None = Field(default=None, description="Trading volume")
    market_cap: float | None = Field(default=None, description="Market capitalization")
    pe_ratio: float | None = Field(default=None, description="Price-to-earnings ratio")
    dividend_yield: float | None = Field(default=None, description="Dividend yield")
    fifty_two_week_high: float | None = Field(default=None, description="52-week high")
    fifty_two_week_low: float | None = Field(default=None, description="52-week low")
    currency: str = Field(default="USD", description="Currency")
    exchange: str = Field(default="", description="Exchange")

    def price_change(self) -> float | None:
        """Calculate price change from previous close."""
        if self.current_price is not None and self.previous_close is not None:
            return self.current_price - self.previous_close
        return None

    def price_change_percent(self) -> float | None:
        """Calculate percentage price change from previous close."""
        if self.current_price is not None and self.previous_close is not None and self.previous_close != 0:
            return ((self.current_price - self.previous_close) / self.previous_close) * 100
        return None


class AssetProfile(BaseModel):
    """Detailed profile information for a financial asset."""
    symbol: str = Field(description="Ticker symbol")
    name: str = Field(description="Asset name")
    sector: str = Field(default="", description="Business sector")
    industry: str = Field(default="", description="Industry")
    description: str = Field(default="", description="Business description")
    website: str = Field(default="", description="Company website")
    country: str = Field(default="", description="Country")
    employees: int | None = Field(default=None, description="Number of employees")
    address: str = Field(default="", description="Business address")


class HistoricalPrice(BaseModel):
    """Single historical price data point."""
    date: datetime = Field(description="Date of the price")
    open: float | None = Field(default=None, description="Opening price")
    high: float | None = Field(default=None, description="High price")
    low: float | None = Field(default=None, description="Low price")
    close: float | None = Field(default=None, description="Closing price")
    volume: int | None = Field(default=None, description="Trading volume")
    adj_close: float | None = Field(default=None, description="Adjusted closing price")


class HistoricalData(BaseModel):
    """Historical price data for a financial asset."""
    symbol: str = Field(description="Ticker symbol")
    prices: list[HistoricalPrice] = Field(description="Historical price data")
    period: str = Field(description="Time period (e.g., '1mo', '1y')")
    interval: str = Field(description="Data interval (e.g., '1d', '1wk')")

    def get_latest(self) -> HistoricalPrice | None:
        """Get the most recent price data."""
        return self.prices[-1] if self.prices else None

    def get_oldest(self) -> HistoricalPrice | None:
        """Get the oldest price data."""
        return self.prices[0] if self.prices else None

    def get_price_range(self) -> tuple[float | None, float | None]:
        """Get the min and max closing prices in the period."""
        closes = [p.close for p in self.prices if p.close is not None]
        if closes:
            return min(closes), max(closes)
        return None, None

    def calculate_return(self) -> float | None:
        """Calculate percentage return over the period."""
        oldest = self.get_oldest()
        latest = self.get_latest()
        if oldest and latest and oldest.close and latest.close and oldest.close != 0:
            return ((latest.close - oldest.close) / oldest.close) * 100
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to a pandas DataFrame with datetime index."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Pandas is required to convert historical data to DataFrame. "
                "Install with: pip install pandas"
            ) from exc

        rows: list[dict[str, object]] = []
        for point in self.prices:
            rows.append(
                {
                    "date": point.date,
                    "open": point.open,
                    "high": point.high,
                    "low": point.low,
                    "close": point.close,
                    "adj_close": point.adj_close,
                    "volume": point.volume,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").set_index("date")
        return frame


class FinancialMetrics(BaseModel):
    """Key financial metrics and ratios."""
    symbol: str = Field(description="Ticker symbol")

    # Valuation Ratios
    pe_ratio: float | None = Field(default=None, description="Price-to-Earnings ratio")
    forward_pe: float | None = Field(default=None, description="Forward P/E ratio")
    peg_ratio: float | None = Field(default=None, description="PEG ratio")
    price_to_book: float | None = Field(default=None, description="Price-to-Book ratio")
    price_to_sales: float | None = Field(default=None, description="Price-to-Sales ratio")
    enterprise_value: float | None = Field(default=None, description="Enterprise Value")
    ev_to_revenue: float | None = Field(default=None, description="EV/Revenue ratio")
    ev_to_ebitda: float | None = Field(default=None, description="EV/EBITDA ratio")

    # Profitability Metrics
    profit_margin: float | None = Field(default=None, description="Profit margin")
    operating_margin: float | None = Field(default=None, description="Operating margin")
    gross_margin: float | None = Field(default=None, description="Gross margin")
    return_on_assets: float | None = Field(default=None, description="Return on Assets (ROA)")
    return_on_equity: float | None = Field(default=None, description="Return on Equity (ROE)")

    # Growth Metrics
    revenue_growth: float | None = Field(default=None, description="Revenue growth rate")
    earnings_growth: float | None = Field(default=None, description="Earnings growth rate")

    # Financial Health
    current_ratio: float | None = Field(default=None, description="Current ratio")
    quick_ratio: float | None = Field(default=None, description="Quick ratio")
    debt_to_equity: float | None = Field(default=None, description="Debt-to-Equity ratio")
    total_debt: float | None = Field(default=None, description="Total debt")
    total_cash: float | None = Field(default=None, description="Total cash")

    # Per Share Data
    book_value_per_share: float | None = Field(default=None, description="Book value per share")
    revenue_per_share: float | None = Field(default=None, description="Revenue per share")
    earnings_per_share: float | None = Field(default=None, description="EPS (TTM)")

    # Dividends
    dividend_rate: float | None = Field(default=None, description="Annual dividend rate")
    dividend_yield: float | None = Field(default=None, description="Dividend yield")
    payout_ratio: float | None = Field(default=None, description="Payout ratio")

    # Trading Metrics
    beta: float | None = Field(default=None, description="Beta (volatility)")
    shares_outstanding: float | None = Field(default=None, description="Shares outstanding")
    float_shares: float | None = Field(default=None, description="Float shares")
    shares_short: float | None = Field(default=None, description="Shares short")
    short_ratio: float | None = Field(default=None, description="Short ratio")


class IncomeStatement(BaseModel):
    """Income statement data for a period."""
    date: datetime = Field(description="Period end date")
    total_revenue: float | None = Field(default=None, description="Total revenue")
    cost_of_revenue: float | None = Field(default=None, description="Cost of revenue")
    gross_profit: float | None = Field(default=None, description="Gross profit")
    operating_expense: float | None = Field(default=None, description="Operating expenses")
    operating_income: float | None = Field(default=None, description="Operating income")
    net_income: float | None = Field(default=None, description="Net income")
    ebitda: float | None = Field(default=None, description="EBITDA")
    basic_eps: float | None = Field(default=None, description="Basic EPS")
    diluted_eps: float | None = Field(default=None, description="Diluted EPS")


class BalanceSheet(BaseModel):
    """Balance sheet data for a period."""
    date: datetime = Field(description="Period end date")
    total_assets: float | None = Field(default=None, description="Total assets")
    current_assets: float | None = Field(default=None, description="Current assets")
    cash_and_equivalents: float | None = Field(default=None, description="Cash and cash equivalents")
    total_liabilities: float | None = Field(default=None, description="Total liabilities")
    current_liabilities: float | None = Field(default=None, description="Current liabilities")
    long_term_debt: float | None = Field(default=None, description="Long-term debt")
    stockholders_equity: float | None = Field(default=None, description="Stockholders' equity")
    retained_earnings: float | None = Field(default=None, description="Retained earnings")


class CashFlowStatement(BaseModel):
    """Cash flow statement data for a period."""
    date: datetime = Field(description="Period end date")
    operating_cash_flow: float | None = Field(default=None, description="Operating cash flow")
    investing_cash_flow: float | None = Field(default=None, description="Investing cash flow")
    financing_cash_flow: float | None = Field(default=None, description="Financing cash flow")
    free_cash_flow: float | None = Field(default=None, description="Free cash flow")
    capital_expenditures: float | None = Field(default=None, description="Capital expenditures")


class EarningsData(BaseModel):
    """Earnings and estimates data."""
    symbol: str = Field(description="Ticker symbol")
    earnings_dates: list[datetime] = Field(default_factory=list, description="Upcoming/past earnings dates")
    earnings_history: list[dict[str, Any]] = Field(default_factory=list, description="Historical earnings")
    earnings_estimates: dict[str, Any] = Field(default_factory=dict, description="Analyst estimates")


class Fundamentals(BaseModel):
    """Comprehensive fundamental data for an asset."""
    symbol: str = Field(description="Ticker symbol")
    metrics: FinancialMetrics | None = Field(default=None, description="Key financial metrics")
    income_statements: list[IncomeStatement] = Field(default_factory=list, description="Income statements")
    balance_sheets: list[BalanceSheet] = Field(default_factory=list, description="Balance sheets")
    cash_flows: list[CashFlowStatement] = Field(default_factory=list, description="Cash flow statements")
    earnings: EarningsData | None = Field(default=None, description="Earnings data")

    def get_latest_income_statement(self) -> IncomeStatement | None:
        """Get the most recent income statement."""
        return self.income_statements[-1] if self.income_statements else None

    def get_latest_balance_sheet(self) -> BalanceSheet | None:
        """Get the most recent balance sheet."""
        return self.balance_sheets[-1] if self.balance_sheets else None

    def get_latest_cash_flow(self) -> CashFlowStatement | None:
        """Get the most recent cash flow statement."""
        return self.cash_flows[-1] if self.cash_flows else None


class SearchResult(BaseModel):
    """Single asset search result."""
    symbol: str = Field(description="Ticker symbol")
    name: str = Field(description="Asset name")
    type: str = Field(description="Asset type")
    exchange: str = Field(default="", description="Exchange")


class SearchResults(BaseModel):
    """Container for multiple search results."""
    query: str = Field(description="Search query")
    results: list[SearchResult] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results")

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)

    def __iter__(self) -> Iterator[SearchResult]:  # type: ignore[override]
        """Allow iteration over results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> SearchResult:
        """Allow indexing of results."""
        return self.results[index]


__all__ = [
    "AssetInfo",
    "AssetProfile",
    "AssetQuote",
    "BalanceSheet",
    "CashFlowStatement",
    "EarningsData",
    "FinancialMetrics",
    "Fundamentals",
    "HistoricalData",
    "HistoricalPrice",
    "IncomeStatement",
    "SearchResult",
    "SearchResults",
]
