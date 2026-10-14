"""
Yahoo Finance market data provider.
"""
import pandas as pd
import yfinance as yf

from qsf.common.logging import get_current_trace


class YFinanceMarketData:
    def get_history(self, ticker: str, period: str) -> pd.DataFrame:
        trace = get_current_trace()
        span = trace.span(name="yfinance.get_history", input={"ticker": ticker, "period": period}) if trace else None
        df = yf.Ticker(ticker).history(period=period)
        if span: span.end(output={"rows": len(df)})
        return df

    def get_company_name(self, ticker: str) -> str:
        trace = get_current_trace()
        span = trace.span(name="yfinance.get_company_name", input={"ticker": ticker}) if trace else None
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
        if span: span.end(output={"name": name})
        return name
