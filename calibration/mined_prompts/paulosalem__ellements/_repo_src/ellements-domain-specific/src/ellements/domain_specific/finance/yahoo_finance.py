"""Yahoo Finance asset search and data retrieval functionality."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any

import yfinance as yf
from ellements.core import ToolRegistry
from ellements.core.exceptions import LLMError

from .charts import technical_chart_assets
from .technical_indicators import (
    compute_technical_indicators,
    talib_available,
)
from .yahoo_finance_models import (
    AssetProfile,
    AssetQuote,
    BalanceSheet,
    CashFlowStatement,
    EarningsData,
    FinancialMetrics,
    Fundamentals,
    HistoricalData,
    HistoricalPrice,
    IncomeStatement,
    SearchResult,
    SearchResults,
    _serialize_indicator_rows,
)

if TYPE_CHECKING:
    pass


class YahooFinanceSearcher:
    """Yahoo Finance asset search and data retrieval."""

    # Common exchange suffixes to try when a bare ticker is not found.
    # Ordered by likelihood for international portfolios.
    EXCHANGE_SUFFIXES = [
        ".SA",   # B3 (Brazil)
        ".L",    # London
        ".TO",   # Toronto
        ".AX",   # ASX (Australia)
        ".HK",   # Hong Kong
        ".DE",   # XETRA (Germany)
        ".PA",   # Euronext Paris
        ".MI",   # Milan
        ".AS",   # Amsterdam
        ".MC",   # Madrid
        ".SI",   # SGX (Singapore)
        ".KS",   # KRX (Korea)
        ".TW",   # TWSE (Taiwan)
        ".MX",   # BMV (Mexico)
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Yahoo Finance searcher.

        Args:
            **kwargs: Additional configuration
        """
        self.config = kwargs
        # Cache of resolved symbols: bare_ticker -> resolved_symbol
        self._symbol_cache: dict[str, str] = {}
        # Set of bare symbols that were verified to have data
        self._verified_bare: set[str] = set()

    def resolve_symbol(self, symbol: str) -> str:
        """Resolve a ticker symbol, trying exchange suffixes if needed.

        If the bare symbol already contains a dot (e.g., "BBAS3.SA"),
        it is returned as-is. Purely alphabetic symbols (e.g., "GOOG",
        "MSFT") are assumed to be valid international tickers and are
        returned without probing, since Yahoo Finance API probing is
        unreliable for well-known symbols. Symbols containing digits
        (e.g., "BBAS3") are probed with exchange suffixes.

        Args:
            symbol: Ticker symbol (e.g., "BBAS3" or "BBAS3.SA")

        Returns:
            The resolved symbol that works with Yahoo Finance.
        """
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        # If symbol already has a suffix, use it directly
        if "." in symbol:
            self._symbol_cache[symbol] = symbol
            return symbol

        # Purely alphabetic symbols with typical US/intl length (1–5 letters:
        # GOOG, MSFT, META, NU etc.) are almost certainly valid tickers.
        # Longer all-alpha strings (KNCALL, BOVAL etc.) may be broker
        # codes that need resolution, so we still probe for them.
        if symbol.isalpha() and len(symbol) <= 5:
            self._symbol_cache[symbol] = symbol
            self._verified_bare.add(symbol)
            return symbol

        # Try the bare symbol first
        if self._symbol_has_data(symbol):
            self._symbol_cache[symbol] = symbol
            self._verified_bare.add(symbol)
            return symbol

        # Try common exchange suffixes
        for suffix in self.EXCHANGE_SUFFIXES:
            candidate = f"{symbol}{suffix}"
            if self._symbol_has_data(candidate):
                self._symbol_cache[symbol] = candidate
                return candidate

        # Nothing worked — return original symbol so the caller gets
        # the normal error for the bare ticker.
        self._symbol_cache[symbol] = symbol
        return symbol

    def resolve_symbol_by_name(
        self, symbol: str, name: str, asset_type: str = ""
    ) -> str:
        """Resolve a ticker via Yahoo Finance name search as fallback.

        First tries ``resolve_symbol`` (exchange-suffix probing).
        If that returns the bare symbol unchanged *and* a descriptive
        ``name`` is provided, performs a Yahoo Finance text search
        using the name and picks the best match.

        Args:
            symbol: Bare ticker (e.g. ``"BOVAL1"``).
            name: Human-readable name (e.g. ``"ETF BOVAL1"``).
            asset_type: Optional hint like ``"etf"``, ``"bdr"`` etc.

        Returns:
            Resolved symbol that (hopefully) works with Yahoo Finance.
        """
        # Fast path: already in cache
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        resolved = self.resolve_symbol(symbol)
        # If resolve_symbol found a different symbol or verified the bare one, done
        if resolved != symbol or symbol in self._verified_bare:
            return resolved

        # Try Yahoo search using the descriptive name
        if name:
            try:
                search_results = self.search(name, max_results=5)
                for sr in search_results.results:
                    candidate = sr.symbol
                    if self._symbol_has_data(candidate):
                        self._symbol_cache[symbol] = candidate
                        return candidate
            except Exception:
                pass

            # Try a more specific search if asset_type is a BDR
            if asset_type and asset_type.lower() == "bdr":
                # BDRs track an underlying foreign asset; search for
                # the name without "BDR" to find the original ticker.
                clean_name = (
                    name.replace("BDR", "")
                    .replace("bdr", "")
                    .replace("(Nubank)", "")
                    .replace("(nubank)", "")
                    .strip()
                )
                if clean_name:
                    try:
                        search_results = self.search(clean_name, max_results=3)
                        for sr in search_results.results:
                            candidate = sr.symbol
                            if self._symbol_has_data(candidate):
                                self._symbol_cache[symbol] = candidate
                                return candidate
                    except Exception:
                        pass

        return resolved

    def resolve_symbol_with_web_search(
        self, symbol: str, name: str, asset_type: str = ""
    ) -> str:
        """Resolve ticker using name search and web search as final fallback.

        Extends :meth:`resolve_symbol_by_name` with a DuckDuckGo web
        search to discover the correct ticker when Yahoo Finance search
        alone doesn't find it (e.g., mistyped Brazilian B3 tickers).

        Args:
            symbol: Bare ticker
            name: Human-readable asset name
            asset_type: Optional hint (``"etf"``, ``"fii"``, ``"bdr"`` …)

        Returns:
            Best resolved ticker found.
        """
        resolved = self.resolve_symbol_by_name(symbol, name, asset_type)
        if resolved != symbol or symbol in self._verified_bare:
            return resolved

        # Try common B3 ticker variations before web search.
        # Some broker codes append an extra letter to standard tickers
        # (e.g. BOVAL1 → BOVA11, VGIAL1 → VGIA11, KNCALL → KNCA11).
        b3_variants = self._generate_b3_variants(symbol)
        for variant in b3_variants:
            if self._symbol_has_data(variant):
                self._symbol_cache[symbol] = variant
                return variant
            sa = f"{variant}.SA"
            if self._symbol_has_data(sa):
                self._symbol_cache[symbol] = sa
                return sa

        # Web search fallback — look for the correct ticker
        try:
            from ellements.standard_tools.web.search import WebSearcher
        except ImportError:
            return resolved

        ws = WebSearcher(max_results=5)
        queries = [
            f"{symbol} ticker B3 bolsa de valores",
            f"{name} ticker stock exchange",
        ]
        for query in queries:
            try:
                results = ws.search(query, max_results=5)
                for r in results.results:
                    # Look for ticker-like patterns in snippets
                    candidate = self._extract_ticker_from_text(
                        r.snippet + " " + r.title, symbol
                    )
                    if candidate and self._symbol_has_data(candidate):
                        self._symbol_cache[symbol] = candidate
                        return candidate
                    # Try with .SA suffix
                    if candidate:
                        sa = f"{candidate}.SA"
                        if self._symbol_has_data(sa):
                            self._symbol_cache[symbol] = sa
                            return sa
            except Exception:
                continue

        return resolved

    @staticmethod
    def _generate_b3_variants(symbol: str) -> list[str]:
        """Generate plausible B3 ticker variations for a broker code.

        Common pattern: brokers append an extra letter to standard tickers
        (e.g. BOVAL1 → BOVA11, VGIAL1 → VGIA11, KNCALL → KNCA11).
        """
        import re

        sym = symbol.upper()
        variants: list[str] = []

        # Pattern: 4+ letters followed by 1-2 digits
        m = re.match(r"^([A-Z]+?)([A-Z])(\d{1,2})$", sym)
        if m:
            prefix, extra, num = m.groups()
            # Try dropping the extra letter and adjusting number
            # BOVAL1 → BOV + A + L + 1 → BOVA + 11
            if len(prefix) >= 3:
                # Try prefix + extra letter + "11" (most common B3 suffix)
                for suffix in ["11", "3", "4", "5", "6"]:
                    variants.append(f"{prefix}{extra}{suffix}")
                    variants.append(f"{prefix}{suffix}")

        # Also try 4-char prefix + common suffixes
        if len(sym) >= 5:
            base4 = sym[:4]
            for suffix in ["11", "3", "4", "5", "6"]:
                candidate = f"{base4}{suffix}"
                if candidate != sym:
                    variants.append(candidate)

        return variants

    @staticmethod
    def _extract_ticker_from_text(text: str, original: str) -> str | None:
        """Try to extract a plausible ticker from search result text.

        Looks for B3-style tickers (4 letters + 2 digits) that resemble
        the original symbol.
        """
        import re

        # Common pattern: 4 uppercase letters + 1-2 digits (B3 style)
        candidates = re.findall(r"\b([A-Z]{4}\d{1,2})\b", text.upper())
        # Prefer candidates that share prefix with original
        prefix = original[:3].upper()
        for c in candidates:
            if c.startswith(prefix) and c != original.upper():
                return str(c)
        # Any B3-style ticker as fallback
        for c in candidates:
            if c != original.upper():
                return str(c)
        return None

    async def resolve_symbol_with_web_search_async(
        self, symbol: str, name: str, asset_type: str = ""
    ) -> str:
        """Async version of :meth:`resolve_symbol_with_web_search`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.resolve_symbol_with_web_search(symbol, name, asset_type),
        )

    async def resolve_symbol_by_name_async(
        self, symbol: str, name: str, asset_type: str = ""
    ) -> str:
        """Async version of :meth:`resolve_symbol_by_name`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.resolve_symbol_by_name(symbol, name, asset_type),
        )

    def _symbol_has_data(self, symbol: str) -> bool:
        """Quick check whether Yahoo Finance recognises a symbol."""
        import logging

        # Suppress noisy HTTP 404 warnings during probing
        loggers_to_quiet = [
            logging.getLogger(name)
            for name in ("yfinance", "urllib3", "peewee", "requests")
        ]
        prev_levels = [(lg, lg.level) for lg in loggers_to_quiet]
        for lg in loggers_to_quiet:
            lg.setLevel(logging.CRITICAL)
        try:
            ticker = yf.Ticker(symbol)
            # Try fast_info first (less API-intensive)
            try:
                fi = ticker.fast_info
                if hasattr(fi, "last_price") and fi.last_price is not None and fi.last_price > 0:
                    return True
            except Exception:
                pass

            info = ticker.info or {}
            if info.get("currentPrice") or info.get("regularMarketPrice"):
                return True
            if info.get("quoteType") and info["quoteType"] != "NONE":
                return True
            # Fallback: try recent history
            hist = ticker.history(period="5d")
            return not hist.empty
        except Exception:
            return False
        finally:
            for lg, level in prev_levels:
                lg.setLevel(level)

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResults:
        """Search for financial assets by name or symbol.

        Args:
            query: Search query (company name or ticker symbol)
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            SearchResults object containing the results
        """
        try:
            # Use yfinance Lookup to search
            lookup = yf.Lookup(query, **kwargs)

            # Get all results (returns a DataFrame)
            all_results_df = lookup.all

            # Convert DataFrame to SearchResult objects
            results = []
            for idx, row in all_results_df.head(max_results).iterrows():
                result = SearchResult(
                    symbol=str(idx) if isinstance(idx, str) else row.get('symbol', str(idx)),
                    name=row.get('longName', row.get('shortName', row.get('name', ''))),
                    type=row.get('typeDisp', row.get('quoteType', 'Unknown')),
                    exchange=row.get('exchange', row.get('exchDisp', ''))
                )
                results.append(result)

            return SearchResults(
                query=query,
                results=results,
                total_results=len(results)
            )

        except Exception as e:
            raise LLMError(f"Asset search failed: {e}") from e

    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResults:
        """Async version of search.

        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional parameters

        Returns:
            SearchResults object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query=query, max_results=max_results, **kwargs)
        )

    def get_quote(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> AssetQuote:
        """Get current quote data for an asset.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            AssetQuote with current market data
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)
            info = ticker.info

            # Try to get current price from info first
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            # Fallback: If info doesn't have price, use history (more reliable)
            if current_price is None:
                hist = ticker.history(period='1d')
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    # Also get previous close from history if available
                    if len(hist) > 1:
                        previous_close = float(hist['Close'].iloc[-2])
                    else:
                        previous_close = None
                else:
                    current_price = None
                    previous_close = None
            else:
                previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')

            return AssetQuote(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                current_price=current_price,
                previous_close=previous_close,
                open_price=info.get('open') or info.get('regularMarketOpen'),
                day_high=info.get('dayHigh') or info.get('regularMarketDayHigh'),
                day_low=info.get('dayLow') or info.get('regularMarketDayLow'),
                volume=info.get('volume') or info.get('regularMarketVolume'),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE') or info.get('forwardPE'),
                dividend_yield=info.get('dividendYield'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow'),
                currency=info.get('currency', 'USD'),
                exchange=info.get('exchange', '')
            )

        except Exception as e:
            raise LLMError(f"Failed to get quote for {symbol}: {e}") from e

    async def get_quote_async(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> AssetQuote:
        """Async version of get_quote.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            AssetQuote object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_quote(symbol, **kwargs)
        )

    def get_profile(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> AssetProfile:
        """Get detailed profile information for an asset.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            AssetProfile with company details
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)
            info = ticker.info

            return AssetProfile(
                symbol=symbol,
                name=info.get('longName', info.get('shortName', symbol)),
                sector=info.get('sector', ''),
                industry=info.get('industry', ''),
                description=info.get('longBusinessSummary', ''),
                website=info.get('website', ''),
                country=info.get('country', ''),
                employees=info.get('fullTimeEmployees'),
                address=f"{info.get('address1', '')} {info.get('city', '')} {info.get('state', '')} {info.get('zip', '')}".strip()
            )

        except Exception as e:
            raise LLMError(f"Failed to get profile for {symbol}: {e}") from e

    async def get_profile_async(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> AssetProfile:
        """Async version of get_profile.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            AssetProfile object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_profile(symbol, **kwargs)
        )

    def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        **kwargs: Any,
    ) -> HistoricalData:
        """Get historical price data for an asset.

        Args:
            symbol: Ticker symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Start date (alternative to period)
            end: End date (alternative to period)
            **kwargs: Additional parameters

        Returns:
            HistoricalData with price history
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)

            # Fetch history
            if start and end:
                hist = ticker.history(start=start, end=end, interval=interval, **kwargs)
            else:
                hist = ticker.history(period=period, interval=interval, **kwargs)

            # Convert to HistoricalPrice objects
            prices = []
            for date, row in hist.iterrows():
                price = HistoricalPrice(
                    date=date.to_pydatetime(),
                    open=row.get('Open'),
                    high=row.get('High'),
                    low=row.get('Low'),
                    close=row.get('Close'),
                    volume=int(row.get('Volume')) if row.get('Volume') is not None else None,
                    adj_close=row.get('Adj Close') if 'Adj Close' in row else row.get('Close')
                )
                prices.append(price)

            return HistoricalData(
                symbol=symbol,
                prices=prices,
                period=period if not start else f"{start.date()} to {end.date() if end else 'now'}",
                interval=interval
            )

        except Exception as e:
            raise LLMError(f"Failed to get historical data for {symbol}: {e}") from e

    async def get_historical_data_async(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        **kwargs: Any,
    ) -> HistoricalData:
        """Async version of get_historical_data.

        Args:
            symbol: Ticker symbol
            period: Time period
            interval: Data interval
            start: Start date
            end: End date
            **kwargs: Additional parameters

        Returns:
            HistoricalData object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_historical_data(
                symbol=symbol,
                period=period,
                interval=interval,
                start=start,
                end=end,
                **kwargs
            )
        )

    def get_asset_info(
        self,
        symbol: str,
        include_profile: bool = True,
        include_quote: bool = True,
        include_history: bool = False,
        history_period: str = "1mo",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get comprehensive information about an asset.

        Args:
            symbol: Ticker symbol
            include_profile: Include profile information
            include_quote: Include current quote
            include_history: Include historical data
            history_period: Period for historical data
            **kwargs: Additional parameters

        Returns:
            Dictionary with requested information
        """
        result: dict[str, Any] = {"symbol": symbol}

        try:
            if include_quote:
                result["quote"] = self.get_quote(symbol, **kwargs)

            if include_profile:
                result["profile"] = self.get_profile(symbol, **kwargs)

            if include_history:
                result["history"] = self.get_historical_data(
                    symbol,
                    period=history_period,
                    **kwargs
                )

            return result

        except Exception as e:
            raise LLMError(f"Failed to get asset info for {symbol}: {e}") from e

    async def get_asset_info_async(
        self,
        symbol: str,
        include_profile: bool = True,
        include_quote: bool = True,
        include_history: bool = False,
        history_period: str = "1mo",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of get_asset_info.

        Args:
            symbol: Ticker symbol
            include_profile: Include profile information
            include_quote: Include current quote
            include_history: Include historical data
            history_period: Period for historical data
            **kwargs: Additional parameters

        Returns:
            Dictionary with requested information
        """
        result: dict[str, Any] = {"symbol": symbol}

        # Collect tasks for concurrent execution
        tasks: list[Coroutine[Any, Any, Any]] = []
        task_names: list[str] = []

        if include_quote:
            tasks.append(self.get_quote_async(symbol, **kwargs))
            task_names.append("quote")

        if include_profile:
            tasks.append(self.get_profile_async(symbol, **kwargs))
            task_names.append("profile")

        if include_history:
            tasks.append(self.get_historical_data_async(
                symbol,
                period=history_period,
                **kwargs
            ))
            task_names.append("history")

        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks)
            for name, data in zip(task_names, results, strict=False):
                result[name] = data

        return result

    def compare_assets(
        self,
        symbols: list[str],
        period: str = "1mo",
        **kwargs: Any,
    ) -> dict[str, HistoricalData]:
        """Get historical data for multiple assets for comparison.

        Args:
            symbols: List of ticker symbols
            period: Time period
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping symbols to historical data
        """
        result = {}
        for symbol in symbols:
            try:
                data = self.get_historical_data(symbol, period=period, **kwargs)
            except Exception:
                continue
            if data.prices:
                result[symbol] = data
        return result

    async def compare_assets_async(
        self,
        symbols: list[str],
        period: str = "1mo",
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> dict[str, HistoricalData]:
        """Async version of compare_assets with concurrency control.

        Args:
            symbols: List of ticker symbols
            period: Time period
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping symbols to historical data
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_history(symbol: str) -> tuple[str, HistoricalData | None]:
            async with semaphore:
                try:
                    data = await self.get_historical_data_async(symbol, period=period, **kwargs)
                    return symbol, data
                except Exception:
                    return symbol, None

        tasks = [fetch_history(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        return {
            symbol: data
            for symbol, data in results
            if data is not None and data.prices
        }

    def get_financial_metrics(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> FinancialMetrics:
        """Get comprehensive financial metrics and ratios.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            FinancialMetrics with all available metrics
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)
            info = ticker.info

            return FinancialMetrics(
                symbol=symbol,
                # Valuation Ratios
                pe_ratio=info.get('trailingPE') or info.get('forwardPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                enterprise_value=info.get('enterpriseValue'),
                ev_to_revenue=info.get('enterpriseToRevenue'),
                ev_to_ebitda=info.get('enterpriseToEbitda'),
                # Profitability
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins'),
                gross_margin=info.get('grossMargins'),
                return_on_assets=info.get('returnOnAssets'),
                return_on_equity=info.get('returnOnEquity'),
                # Growth
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                # Financial Health
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio'),
                debt_to_equity=info.get('debtToEquity'),
                total_debt=info.get('totalDebt'),
                total_cash=info.get('totalCash'),
                # Per Share
                book_value_per_share=info.get('bookValue'),
                revenue_per_share=info.get('revenuePerShare'),
                earnings_per_share=info.get('trailingEps'),
                # Dividends
                dividend_rate=info.get('dividendRate'),
                dividend_yield=info.get('dividendYield'),
                payout_ratio=info.get('payoutRatio'),
                # Trading
                beta=info.get('beta'),
                shares_outstanding=info.get('sharesOutstanding'),
                float_shares=info.get('floatShares'),
                shares_short=info.get('sharesShort'),
                short_ratio=info.get('shortRatio'),
            )

        except Exception as e:
            raise LLMError(f"Failed to get financial metrics for {symbol}: {e}") from e

    async def get_financial_metrics_async(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> FinancialMetrics:
        """Async version of get_financial_metrics."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_financial_metrics(symbol, **kwargs)
        )

    def get_financials(
        self,
        symbol: str,
        quarterly: bool = False,
        **kwargs: Any,
    ) -> Fundamentals:
        """Get comprehensive fundamental data including financial statements.

        Args:
            symbol: Ticker symbol
            quarterly: Get quarterly data (default: annual)
            **kwargs: Additional parameters

        Returns:
            Fundamentals object with all financial data
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)

            # Get financial metrics
            metrics = self.get_financial_metrics(symbol, **kwargs)

            # Get income statements
            income_statements = []
            income_df = ticker.quarterly_financials if quarterly else ticker.financials
            if income_df is not None and not income_df.empty:
                for col in income_df.columns:
                    row_data = income_df[col]
                    stmt = IncomeStatement(
                        date=col.to_pydatetime(),
                        total_revenue=row_data.get('Total Revenue'),
                        cost_of_revenue=row_data.get('Cost Of Revenue'),
                        gross_profit=row_data.get('Gross Profit'),
                        operating_expense=row_data.get('Operating Expense'),
                        operating_income=row_data.get('Operating Income'),
                        net_income=row_data.get('Net Income'),
                        ebitda=row_data.get('EBITDA'),
                        basic_eps=row_data.get('Basic EPS'),
                        diluted_eps=row_data.get('Diluted EPS'),
                    )
                    income_statements.append(stmt)

            # Get balance sheets
            balance_sheets = []
            balance_df = ticker.quarterly_balance_sheet if quarterly else ticker.balance_sheet
            if balance_df is not None and not balance_df.empty:
                for col in balance_df.columns:
                    row_data = balance_df[col]
                    sheet = BalanceSheet(
                        date=col.to_pydatetime(),
                        total_assets=row_data.get('Total Assets'),
                        current_assets=row_data.get('Current Assets'),
                        cash_and_equivalents=row_data.get('Cash And Cash Equivalents'),
                        total_liabilities=row_data.get('Total Liabilities Net Minority Interest'),
                        current_liabilities=row_data.get('Current Liabilities'),
                        long_term_debt=row_data.get('Long Term Debt'),
                        stockholders_equity=row_data.get('Stockholders Equity'),
                        retained_earnings=row_data.get('Retained Earnings'),
                    )
                    balance_sheets.append(sheet)

            # Get cash flow statements
            cash_flows = []
            cashflow_df = ticker.quarterly_cashflow if quarterly else ticker.cashflow
            if cashflow_df is not None and not cashflow_df.empty:
                for col in cashflow_df.columns:
                    row_data = cashflow_df[col]
                    cf = CashFlowStatement(
                        date=col.to_pydatetime(),
                        operating_cash_flow=row_data.get('Operating Cash Flow'),
                        investing_cash_flow=row_data.get('Investing Cash Flow'),
                        financing_cash_flow=row_data.get('Financing Cash Flow'),
                        free_cash_flow=row_data.get('Free Cash Flow'),
                        capital_expenditures=row_data.get('Capital Expenditure'),
                    )
                    cash_flows.append(cf)

            # Get earnings data
            earnings_data = None
            try:
                earnings_dates = []
                if hasattr(ticker, 'earnings_dates') and ticker.earnings_dates is not None:
                    for idx in ticker.earnings_dates.index:
                        if isinstance(idx, datetime):
                            earnings_dates.append(idx)

                earnings_history = []
                if hasattr(ticker, 'earnings_history') and ticker.earnings_history is not None:
                    earnings_history = ticker.earnings_history.to_dict('records')

                earnings_estimates = {}
                if hasattr(ticker, 'earnings_estimate') and ticker.earnings_estimate is not None:
                    earnings_estimates = ticker.earnings_estimate.to_dict()

                earnings_data = EarningsData(
                    symbol=symbol,
                    earnings_dates=earnings_dates,
                    earnings_history=earnings_history,
                    earnings_estimates=earnings_estimates,
                )
            except Exception:
                # Earnings data is optional
                pass

            return Fundamentals(
                symbol=symbol,
                metrics=metrics,
                income_statements=income_statements,
                balance_sheets=balance_sheets,
                cash_flows=cash_flows,
                earnings=earnings_data,
            )

        except Exception as e:
            raise LLMError(f"Failed to get financials for {symbol}: {e}") from e

    async def get_financials_async(
        self,
        symbol: str,
        quarterly: bool = False,
        **kwargs: Any,
    ) -> Fundamentals:
        """Async version of get_financials."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_financials(symbol, quarterly=quarterly, **kwargs)
        )

    def get_analyst_recommendations(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get analyst recommendations and price targets.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            Dictionary with recommendations data
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)
            info = ticker.info

            recommendations = {
                "symbol": symbol,
                "target_high_price": info.get('targetHighPrice'),
                "target_low_price": info.get('targetLowPrice'),
                "target_mean_price": info.get('targetMeanPrice'),
                "target_median_price": info.get('targetMedianPrice'),
                "recommendation_key": info.get('recommendationKey'),
                "recommendation_mean": info.get('recommendationMean'),
                "number_of_analyst_opinions": info.get('numberOfAnalystOpinions'),
            }

            # Get historical recommendations if available
            if hasattr(ticker, 'recommendations') and ticker.recommendations is not None:
                recommendations["history"] = ticker.recommendations.to_dict('records')

            return recommendations

        except Exception as e:
            raise LLMError(f"Failed to get analyst recommendations for {symbol}: {e}") from e

    async def get_analyst_recommendations_async(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of get_analyst_recommendations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_analyst_recommendations(symbol, **kwargs)
        )

    def get_institutional_holders(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get institutional holders information.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            Dictionary with institutional holders data
        """
        try:
            resolved = self.resolve_symbol(symbol)
            ticker = yf.Ticker(resolved)

            result = {"symbol": symbol}

            if hasattr(ticker, 'institutional_holders') and ticker.institutional_holders is not None:
                result["institutional_holders"] = ticker.institutional_holders.to_dict('records')

            if hasattr(ticker, 'major_holders') and ticker.major_holders is not None:
                result["major_holders"] = ticker.major_holders.to_dict()

            if hasattr(ticker, 'mutualfund_holders') and ticker.mutualfund_holders is not None:
                result["mutualfund_holders"] = ticker.mutualfund_holders.to_dict('records')

            return result

        except Exception as e:
            raise LLMError(f"Failed to get institutional holders for {symbol}: {e}") from e

    async def get_institutional_holders_async(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of get_institutional_holders."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_institutional_holders(symbol, **kwargs)
        )

    def get_all_data(
        self,
        symbol: str,
        include_financials: bool = True,
        include_recommendations: bool = True,
        include_holders: bool = True,
        quarterly: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get all available data for comprehensive fundamental analysis.

        Args:
            symbol: Ticker symbol
            include_financials: Include financial statements
            include_recommendations: Include analyst recommendations
            include_holders: Include institutional holders
            quarterly: Use quarterly data for financials
            **kwargs: Additional parameters

        Returns:
            Dictionary with all available data
        """
        result = {
            "symbol": symbol,
            "quote": self.get_quote(symbol, **kwargs),
            "profile": self.get_profile(symbol, **kwargs),
        }

        if include_financials:
            result["fundamentals"] = self.get_financials(symbol, quarterly=quarterly, **kwargs)

        if include_recommendations:
            result["analyst_recommendations"] = self.get_analyst_recommendations(symbol, **kwargs)

        if include_holders:
            result["institutional_holders"] = self.get_institutional_holders(symbol, **kwargs)

        return result

    async def get_all_data_async(
        self,
        symbol: str,
        include_financials: bool = True,
        include_recommendations: bool = True,
        include_holders: bool = True,
        quarterly: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of get_all_data with concurrent fetching.

        Args:
            symbol: Ticker symbol
            include_financials: Include financial statements
            include_recommendations: Include analyst recommendations
            include_holders: Include institutional holders
            quarterly: Use quarterly data for financials
            **kwargs: Additional parameters

        Returns:
            Dictionary with all available data
        """
        # Fetch core data concurrently
        tasks: list[Coroutine[Any, Any, Any]] = []
        task_names: list[str] = []

        tasks.append(self.get_quote_async(symbol, **kwargs))
        task_names.append("quote")

        tasks.append(self.get_profile_async(symbol, **kwargs))
        task_names.append("profile")

        if include_financials:
            tasks.append(self.get_financials_async(symbol, quarterly=quarterly, **kwargs))
            task_names.append("fundamentals")

        if include_recommendations:
            tasks.append(self.get_analyst_recommendations_async(symbol, **kwargs))
            task_names.append("analyst_recommendations")

        if include_holders:
            tasks.append(self.get_institutional_holders_async(symbol, **kwargs))
            task_names.append("institutional_holders")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Build result dictionary
        result: dict[str, Any] = {"symbol": symbol}
        for name, data in zip(task_names, results, strict=False):
            result[name] = data

        return result


# Agent Tools
# ---------------------------------------------------------------------------

def finance_tools() -> ToolRegistry:
    """Create Yahoo Finance tools for agents.

    These tools are returned as plain callables. Framework-specific backends
    adapt them when needed.

    Returns:
        Dict of plain tool callables ready for adaptation by agent frameworks

    Example:
        >>> tools = finance_tools()
        >>> agent = AgentBuilder("FinanceAnalyst").with_tools(tools).build()
    """

    # Create searcher instance
    searcher = YahooFinanceSearcher()
    def search_asset(query: str, max_results: int = 10) -> SearchResults:
        """Search for financial assets by name or ticker symbol.

        Args:
            query: Company name or ticker symbol to search for
            max_results: Maximum number of results to return

        Returns:
            SearchResults object containing matching assets
        """
        return searcher.search(query, max_results=max_results)
    def get_asset_quote(symbol: str) -> AssetQuote:
        """Get current quote and market data for a financial asset.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')

        Returns:
            AssetQuote with current price, volume, market cap, and other market data
        """
        return searcher.get_quote(symbol)
    def get_asset_profile(symbol: str) -> AssetProfile:
        """Get detailed company profile information.

        Args:
            symbol: Ticker symbol

        Returns:
            AssetProfile with company description, sector, industry, website, and business details
        """
        return searcher.get_profile(symbol)
    def get_financial_metrics(symbol: str) -> FinancialMetrics:
        """Get comprehensive financial metrics and ratios.

        Includes valuation ratios (P/E, P/B, PEG), profitability metrics (ROE, ROA, margins),
        growth metrics, financial health indicators, and trading metrics.

        Args:
            symbol: Ticker symbol

        Returns:
            FinancialMetrics with all available financial ratios and metrics
        """
        return searcher.get_financial_metrics(symbol)
    def get_income_statement(symbol: str, quarterly: bool = False) -> str:
        """Get income statement data (revenue, expenses, profit).

        Args:
            symbol: Ticker symbol
            quarterly: If True, get quarterly data; if False, get annual data

        Returns:
            Formatted string with income statement data from recent periods
        """
        fundamentals = searcher.get_financials(symbol, quarterly=quarterly)
        if not fundamentals.income_statements:
            return f"No income statement data available for {symbol}"

        lines = [f"Income Statements for {symbol} ({'Quarterly' if quarterly else 'Annual'}):"]
        for stmt in fundamentals.income_statements[:4]:  # Show last 4 periods
            lines.append(f"\nPeriod ending {stmt.date.date()}:")
            if stmt.total_revenue:
                lines.append(f"  Total Revenue: ${stmt.total_revenue:,.0f}")
            if stmt.gross_profit:
                lines.append(f"  Gross Profit: ${stmt.gross_profit:,.0f}")
            if stmt.operating_income:
                lines.append(f"  Operating Income: ${stmt.operating_income:,.0f}")
            if stmt.net_income:
                lines.append(f"  Net Income: ${stmt.net_income:,.0f}")
            if stmt.diluted_eps:
                lines.append(f"  Diluted EPS: ${stmt.diluted_eps:.2f}")

        return "\n".join(lines)
    def get_balance_sheet(symbol: str, quarterly: bool = False) -> str:
        """Get balance sheet data (assets, liabilities, equity).

        Args:
            symbol: Ticker symbol
            quarterly: If True, get quarterly data; if False, get annual data

        Returns:
            Formatted string with balance sheet data from recent periods
        """
        fundamentals = searcher.get_financials(symbol, quarterly=quarterly)
        if not fundamentals.balance_sheets:
            return f"No balance sheet data available for {symbol}"

        lines = [f"Balance Sheets for {symbol} ({'Quarterly' if quarterly else 'Annual'}):"]
        for sheet in fundamentals.balance_sheets[:4]:  # Show last 4 periods
            lines.append(f"\nPeriod ending {sheet.date.date()}:")
            if sheet.total_assets:
                lines.append(f"  Total Assets: ${sheet.total_assets:,.0f}")
            if sheet.current_assets:
                lines.append(f"  Current Assets: ${sheet.current_assets:,.0f}")
            if sheet.cash_and_equivalents:
                lines.append(f"  Cash & Equivalents: ${sheet.cash_and_equivalents:,.0f}")
            if sheet.total_liabilities:
                lines.append(f"  Total Liabilities: ${sheet.total_liabilities:,.0f}")
            if sheet.long_term_debt:
                lines.append(f"  Long-term Debt: ${sheet.long_term_debt:,.0f}")
            if sheet.stockholders_equity:
                lines.append(f"  Stockholders' Equity: ${sheet.stockholders_equity:,.0f}")

        return "\n".join(lines)
    def get_cash_flow(symbol: str, quarterly: bool = False) -> str:
        """Get cash flow statement data (operating, investing, financing cash flows).

        Args:
            symbol: Ticker symbol
            quarterly: If True, get quarterly data; if False, get annual data

        Returns:
            Formatted string with cash flow statement data from recent periods
        """
        fundamentals = searcher.get_financials(symbol, quarterly=quarterly)
        if not fundamentals.cash_flows:
            return f"No cash flow data available for {symbol}"

        lines = [f"Cash Flow Statements for {symbol} ({'Quarterly' if quarterly else 'Annual'}):"]
        for cf in fundamentals.cash_flows[:4]:  # Show last 4 periods
            lines.append(f"\nPeriod ending {cf.date.date()}:")
            if cf.operating_cash_flow:
                lines.append(f"  Operating Cash Flow: ${cf.operating_cash_flow:,.0f}")
            if cf.investing_cash_flow:
                lines.append(f"  Investing Cash Flow: ${cf.investing_cash_flow:,.0f}")
            if cf.financing_cash_flow:
                lines.append(f"  Financing Cash Flow: ${cf.financing_cash_flow:,.0f}")
            if cf.free_cash_flow:
                lines.append(f"  Free Cash Flow: ${cf.free_cash_flow:,.0f}")
            if cf.capital_expenditures:
                lines.append(f"  Capital Expenditures: ${cf.capital_expenditures:,.0f}")

        return "\n".join(lines)
    def get_analyst_recommendations(symbol: str) -> str:
        """Get analyst recommendations and price targets.

        Args:
            symbol: Ticker symbol

        Returns:
            Formatted string with analyst recommendations and price targets
        """
        recs = searcher.get_analyst_recommendations(symbol)

        lines = [f"Analyst Recommendations for {symbol}:"]
        if recs.get('recommendation_key'):
            lines.append(f"  Recommendation: {recs['recommendation_key'].upper()}")
        if recs.get('recommendation_mean'):
            lines.append(f"  Recommendation Mean: {recs['recommendation_mean']:.2f} (1=Strong Buy, 5=Sell)")
        if recs.get('number_of_analyst_opinions'):
            lines.append(f"  Number of Analysts: {recs['number_of_analyst_opinions']}")

        lines.append("\nPrice Targets:")
        if recs.get('target_mean_price'):
            lines.append(f"  Mean: ${recs['target_mean_price']:.2f}")
        if recs.get('target_median_price'):
            lines.append(f"  Median: ${recs['target_median_price']:.2f}")
        if recs.get('target_high_price'):
            lines.append(f"  High: ${recs['target_high_price']:.2f}")
        if recs.get('target_low_price'):
            lines.append(f"  Low: ${recs['target_low_price']:.2f}")

        if recs.get('history'):
            lines.append(f"\nRecent Recommendations History: {len(recs['history'])} entries available")

        return "\n".join(lines)
    def get_technical_indicators(
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_period: int = 20,
        ema_period: int = 20,
        bollinger_window: int = 20,
        bollinger_dev: float = 2.0,
        stochastic_window: int = 14,
        stochastic_smooth_window: int = 3,
        atr_window: int = 14,
        adx_window: int = 14,
        include_chart: bool = True,
        chart_image_format: str = "png",
    ) -> dict[str, Any]:
        """Compute technical indicators and optional chart asset for a ticker.

        Args:
            symbol: Ticker symbol (e.g., AAPL).
            period: Yahoo period (e.g., 1mo, 6mo, 1y).
            interval: Yahoo interval (e.g., 1d, 1wk).
            include_chart: Include chart assets compatible with canvas-chart UI flow.

        Returns:
            Dictionary with indicator snapshot, aligned series data, and optional chart assets.
        """
        history = searcher.get_historical_data(symbol, period=period, interval=interval)
        frame = history.to_dataframe()
        indicators_df, snapshot = compute_technical_indicators(
            frame,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            sma_period=sma_period,
            ema_period=ema_period,
            bollinger_window=bollinger_window,
            bollinger_dev=bollinger_dev,
            stochastic_window=stochastic_window,
            stochastic_smooth_window=stochastic_smooth_window,
            atr_window=atr_window,
            adx_window=adx_window,
        )
        series_rows = _serialize_indicator_rows(indicators_df)

        result: dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "talib_available": talib_available(),
            "snapshot": snapshot.model_dump(),
            "indicators": series_rows,
        }
        if include_chart:
            chart_assets = technical_chart_assets(
                history,
                title=f"{symbol} Technical Indicators ({period}, {interval})",
                image_format=chart_image_format,
                rsi_period=rsi_period,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                sma_period=sma_period,
                ema_period=ema_period,
                bollinger_window=bollinger_window,
                bollinger_dev=bollinger_dev,
                stochastic_window=stochastic_window,
                stochastic_smooth_window=stochastic_smooth_window,
                atr_window=atr_window,
                adx_window=adx_window,
            )
            result["chart"] = chart_assets
        return result
    def get_technical_chart(
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        image_format: str = "png",
        include_ohlc_fallback: bool = False,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_period: int = 20,
        ema_period: int = 20,
        bollinger_window: int = 20,
        bollinger_dev: float = 2.0,
        stochastic_window: int = 14,
        stochastic_smooth_window: int = 3,
        atr_window: int = 14,
        adx_window: int = 14,
    ) -> dict[str, Any]:
        """Render a technical analysis chart payload compatible with canvas-chart blocks."""
        history = searcher.get_historical_data(symbol, period=period, interval=interval)
        return technical_chart_assets(
            history,
            title=f"{symbol} Technical Indicators ({period}, {interval})",
            image_format=image_format,
            include_ohlc_fallback=include_ohlc_fallback,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            sma_period=sma_period,
            ema_period=ema_period,
            bollinger_window=bollinger_window,
            bollinger_dev=bollinger_dev,
            stochastic_window=stochastic_window,
            stochastic_smooth_window=stochastic_smooth_window,
            atr_window=atr_window,
            adx_window=adx_window,
        )

    return ToolRegistry.from_mapping({
        "search_asset": search_asset,
        "get_asset_quote": get_asset_quote,
        "get_asset_profile": get_asset_profile,
        "get_financial_metrics": get_financial_metrics,
        "get_income_statement": get_income_statement,
        "get_balance_sheet": get_balance_sheet,
        "get_cash_flow": get_cash_flow,
        "get_analyst_recommendations": get_analyst_recommendations,
        "get_technical_indicators": get_technical_indicators,
        "get_technical_chart": get_technical_chart,
    })
